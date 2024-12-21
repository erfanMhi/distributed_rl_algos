"""
Asynchronous Advantage Actor-Critic (A3C) implementation based on the paper 
"Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016).

This implementation features:
- Parallel training across multiple workers
- Shared model architecture with target network
- Epsilon-greedy exploration with annealing
- Learning rate annealing
- Wandb logging integration

The algorithm uses multiple processes to gather experience and update a shared model
asynchronously, which can significantly speed up training compared to standard DQN.
"""

import random
from typing import List, Callable, Any, Dict, Tuple, Optional
from omegaconf import DictConfig

import wandb
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.utils import seeding
import torch.multiprocessing as mp

from distributed_rl_algos.common.utils import SharedAdam
from distributed_rl_algos.common.network_factory import NetworkFactory
from distributed_rl_algos.algorithms.base import BaseRLAlgorithm

class A3CLearner(BaseRLAlgorithm):
    """Main A3C learner class that manages the training across multiple workers."""
    
    def __init__(self, env_func: Callable[[], gym.Env], cfg: Dict):
        """
        Initialize the A3C learner.

        Args:
            env_func: A function that creates and returns a Gymnasium environment
            cfg: Configuration dictionary containing algorithm parameters
        """
        assert cfg['algorithm']['num_workers'] > 0, "Number of workers must be greater than 0"
        
        self._env_func = env_func
        self._num_workers = cfg['algorithm']['num_workers']
        self._device = torch.device(cfg['algorithm']['device'])
        self._cfg = cfg
        
    def learn(self) -> None:
        """
        Start the learning process across multiple workers.
        
        This method:
        1. Creates the shared model and target network
        2. Initializes the shared optimizer
        3. Spawns worker processes for parallel training
        """
        env = self._env_func()

        # Validate environment spaces
        assert isinstance(env.observation_space, gym.spaces.Box), "Observation space must be a Box"
        assert isinstance(env.action_space, gym.spaces.Discrete), "Action space must be a Discrete"
        assert self._device.type == 'cpu', "Device must be CPU for multiprocessing"

        algo_config = self._cfg['algorithm']

        # Initialize shared networks
        shared_model = self._create_shared_network(env, algo_config)

        # Initialize optimizer
        shared_optimizer = self._create_shared_optimizer(shared_model, algo_config['optimizer'])
        global_step = mp.Value('i', 0)

        if self._num_workers == 1:
            self._run_single_process(shared_model, shared_optimizer, global_step)
        else:
            self._run_multi_process(shared_model, shared_optimizer, global_step)

    def _create_shared_network(self, env: gym.Env, algo_config: Dict) -> nn.Module:
        """Create and initialize a shared network."""
        network = NetworkFactory.create_network(
            network_config=algo_config['network'],
            input_dim=env.observation_space.shape[0],
            output_dim=env.action_space.n+1
        )
        network.share_memory()
        return network

    def _create_shared_optimizer(self, model: nn.Module, optimizer_config: Dict) -> SharedAdam:
        """Create and initialize the shared optimizer."""
        return SharedAdam(
            model.parameters(), 
            lr=optimizer_config['learning_rate'],
            betas=(optimizer_config['adam_beta1'], optimizer_config['adam_beta2']),
            weight_decay=optimizer_config['adam_weight_decay'],
            amsgrad=optimizer_config['adam_amsgrad'],
            log_params_norm=optimizer_config['log_params_norm']
        )

    def _run_single_process(self, shared_model: nn.Module, shared_optimizer: SharedAdam, global_step: mp.Value) -> None:
        """Run training in single process mode."""
        process = A3CLearnerProcess(
            env=self._env_func(), 
            shared_model=shared_model, 
            shared_optimizer=shared_optimizer,
            global_step=global_step,
            cfg=self._cfg,
            rank=0
        )
        process.run()

    def _run_multi_process(self, shared_model: nn.Module, shared_optimizer: SharedAdam, global_step: mp.Value) -> None:
        """Run training in multi-process mode."""
        processes = [
            A3CLearnerProcess(
                env=self._env_func(), 
                shared_model=shared_model, 
                shared_optimizer=shared_optimizer,
                global_step=global_step,
                cfg=self._cfg,
                rank=i
            ) for i in range(self._num_workers)
        ]

        for p in processes:
            p.start()
        for p in processes:
            p.join()


class A3CLearnerProcess(mp.Process):
    """Worker process that runs the A3C algorithm and updates the shared model."""

    def __init__(self, 
                 env: gym.Env,
                 shared_model: nn.Module, 
                 shared_optimizer: torch.optim.Optimizer,
                 global_step: 'mp.Value',
                 cfg: Dict,
                 rank: int):
        """
        Initialize a worker process.

        Args:
            env: The environment to train on
            shared_model: The shared policy network
            shared_optimizer: The shared optimizer
            global_step: Shared counter for total steps taken
            cfg: Configuration dictionary
            rank: Worker rank/ID
        """
        super().__init__()
        
        self._env = env
        self._shared_model = shared_model
        self._shared_optimizer = shared_optimizer
        self._global_step = global_step
        self._rank = rank

        # Extract configuration
        algo_config = cfg['algorithm']
        optimizer_config = algo_config['optimizer']
        self._network_config = algo_config['network']
        
        # Setup training parameters
        self._device = torch.device(algo_config['device'])
        self._seed = algo_config['seed']
        self._setup_learning_params(algo_config)
        self._setup_optimizer_params(optimizer_config)
        self._setup_wandb_config(cfg['wandb'])

        # Initialize episode state
        self._last_obs: Optional[torch.Tensor] = None
        self._done: bool = False
        self._rollout_step: int = 1

    def _setup_learning_params(self, algo_config: Dict) -> None:
        """Setup learning-related parameters."""
        self._total_timesteps = algo_config['total_timesteps']
        self._max_rollout_length = algo_config['max_rollout_length']
        self._gamma = algo_config['gamma']
        self._value_loss_coef = algo_config.get('value_loss_coef', 0.5)
        self._entropy_coef = algo_config.get('entropy_coef', 0.01)

    def _setup_optimizer_params(self, optimizer_config: Dict) -> None:
        """Setup optimizer-related parameters."""
        self._initial_lr = optimizer_config['learning_rate']
        self._final_lr = optimizer_config['final_learning_rate']
        self._max_grad_norm = optimizer_config['max_grad_norm']

    def _setup_wandb_config(self, wandb_config: Dict) -> None:
        """Setup WandB logging configuration."""
        self._wandb_project = wandb_config['project']
        self._wandb_run_name = wandb_config['run_name']

    def _set_model_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self._seed + self._rank)
        torch.cuda.manual_seed_all(self._seed + self._rank)

    def _create_model(self) -> nn.Module:
        """
        Create a local copy of the model for the worker.
        
        Returns:
            A new model instance with weights copied from the shared model
        """
        model = NetworkFactory.create_network(
            network_config=self._network_config,
            input_dim=self._env.observation_space.shape[0],
            output_dim=self._env.action_space.n+1
        )
        model.to(self._device)
        return model
    
    def _predict(self, model: nn.Module, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the value and action logits for the given observation.
        
        Args:
            model: The local model to use for prediction
            obs: The current observation
            
        Returns:
            A tuple containing:
            - The predicted action logits
            - The predicted value
        """
        predictions = model(torch.tensor(obs))
        return predictions[:self._env.action_space.n], predictions[self._env.action_space.n:]
    
    def _rollout(self, model: nn.Module) -> Tuple[List[float], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[int], List[torch.Tensor]]:
        """
        Perform a rollout in the environment using the current policy.
        
        Args:
            model: The local model to use for action selection
            
        Returns:
            A tuple containing:
            - List of rewards received
            - List of observations
            - List of values
            - List of logits
            - List of actions
            - List of entropy
        """
        reward_buffer = []
        value_buffer = []
        entropy_buffer = []
        logits_buffer = []
        actions_buffer = []
        
        rollout_start_step = self._rollout_step

        while not self._done and self._rollout_step - rollout_start_step != self._max_rollout_length:
            # Select an action
            logits, value = self._predict(model, self._last_obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

            # Execute action
            self._last_obs, reward, terminated, truncated, info = self._env.step(action)
            self._done = terminated or truncated
    
            # Store transition
            reward_buffer.append(reward)
            value_buffer.append(value)
            logits_buffer.append(logits)
            actions_buffer.append(action)
            entropy_buffer.append(dist.entropy())

            # Update step counters
            self._rollout_step += 1
            with self._global_step.get_lock():
                self._global_step.value += 1

        return reward_buffer, value_buffer, logits_buffer, actions_buffer, entropy_buffer
    
    def _train_model(self, model: nn.Module, 
                     reward_buffer: List[float], 
                     value_buffer: List[torch.Tensor],
                     logits_buffer: List[torch.Tensor], 
                     actions_buffer: List[int],
                     entropy_buffer: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update the model using collected experience.
        
        Args:
            model: The local model to update
            reward_buffer: List of rewards from the rollout
            obs_buffer: List of observations from the rollout
            value_buffer: List of values from the rollout
            logits_buffer: List of logits from the rollout
            actions_buffer: List of actions from the rollout
            entropy_buffer: List of entropy from the rollout
        """
        # Calculate bootstrap value
        R = torch.tensor(0.0, device=self._device)
        if not self._done:
            with torch.no_grad():
                R = self._predict(model, self._last_obs)[1].detach().item()

        # Calculate losses
        value_loss = torch.zeros(1, device=self._device)
        policy_loss = torch.zeros(1, device=self._device)
        entropy_loss = torch.zeros(1, device=self._device)
        for i in reversed(range(len(reward_buffer))):
            R = reward_buffer[i] + self._gamma * R # computing v(s_{t-1})
            advantage = R - value_buffer[i]
            value_loss += 0.5 * advantage.pow(2) 
            
            policy_loss -= torch.log_softmax(logits_buffer[i], dim=0)[actions_buffer[i]] * advantage.detach()
            entropy_loss -= entropy_buffer[i]
        
        self._shared_optimizer.zero_grad()

        # Computing gradients
        (
            policy_loss + 
            self._value_loss_coef * value_loss + 
            self._entropy_coef * entropy_loss
        ).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), self._max_grad_norm)

        # Updating gradients in shared model
        for param, shared_param in zip(model.parameters(), self._shared_model.parameters()):
            shared_param._grad = param.grad.clone()

        # Update learning rate and apply gradients
        current_lr = self._get_current_lr()
        for param_group in self._shared_optimizer.param_groups:
            param_group['lr'] = current_lr

        self._shared_optimizer.step()

        return value_loss.detach(), policy_loss.detach(), entropy_loss.detach()

    def _setup_logging(self) -> None:
        """Initialize WandB logging for this worker."""
        self._wandb_logger = wandb.init(
            project=self._wandb_project,
            name=f'{self._wandb_run_name}-{self._rank}',
            group=self._wandb_run_name,
            job_type=f"process_{self._rank}")

    def _compute_stats(self, values: List[torch.Tensor], logits: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compute statistics for values and logits.
        
        Args:
            values: List of value predictions
            logits: List of action logits
        
        Returns:
            Dictionary of computed statistics
        """
        # Convert to tensors for easier computation
        values_tensor = torch.cat([v.detach() for v in values])
        logits_tensor = torch.stack([l.detach() for l in logits])
        
        # Compute probabilities from logits
        probs = torch.softmax(logits_tensor, dim=1)
        
        stats = {
            # Value statistics
            'value/mean': values_tensor.mean().item(),
            'value/std': values_tensor.std().item(),
            'value/min': values_tensor.min().item(),
            'value/max': values_tensor.max().item(),
            
            # Action probability statistics
            'action_prob/mean': probs.mean().item(),
            'action_prob/std': probs.std().item(),
            'action_prob/entropy': -(probs * torch.log(probs + 1e-10)).sum(1).mean().item(),
            
            # Per-action probabilities
            **{f'action_prob/{i}': probs[:, i].mean().item() 
               for i in range(probs.shape[1])}
        }
        
        return stats

    def _log_metrics(self, episode_return: float, episode_length: int, 
                     value_loss: torch.Tensor, policy_loss: torch.Tensor, entropy_loss: torch.Tensor,
                     values: List[torch.Tensor], logits: List[torch.Tensor]) -> None:
        """
        Log episode metrics to WandB.
        
        Args:
            episode_return: Total reward for the episode
            episode_length: Number of steps in the episode
            values: List of value predictions from the episode
            logits: List of action logits from the episode
        """
        # Basic episode metrics
        metrics = {
            'episode/return': episode_return,
            'episode/length': episode_length,
            'train/learning_rate': self._get_current_lr(),
            'train/value_loss': value_loss.item(),
            'train/policy_loss': policy_loss.item(),
            'train/entropy_loss': entropy_loss.item(),
        }
        
        # Add value and action statistics if episode completed
        metrics.update(self._compute_stats(values, logits))
        
        wandb.log(metrics, step=self._global_step.value)

    def run(self) -> None:
        """
        Main training loop for the worker process.
        
        This method:
        1. Initializes the environment and models
        2. Runs the training loop until total timesteps is reached
        3. Performs rollouts and updates the shared model
        4. Logs metrics and manages the training state
        """
        self._set_model_seeds()  
        self._setup_logging()

        # Initialize environment
        self._last_obs, _ = self._env.reset(seed=self._seed + self._rank)
        episode_return = 0
        episode_length = 0

        # Create local model
        model = self._create_model()
        model.load_state_dict(self._shared_model.state_dict())

        while self._global_step.value < self._total_timesteps:
            # Sync with shared model
            model.load_state_dict(self._shared_model.state_dict())

            # Collect experience
            reward_buffer, values_buffer, logits_buffer, actions_buffer, entropy_buffer = self._rollout(model)

            # Update episode statistics
            episode_return += sum(reward_buffer)
            episode_length += len(reward_buffer)

            # Update model
            value_loss, policy_loss, entropy_loss = self._train_model(model, reward_buffer, values_buffer, logits_buffer, actions_buffer, entropy_buffer)

            # Handle episode completion
            if self._done:
                self._log_metrics(
                    episode_return, 
                    episode_length,
                    value_loss,
                    policy_loss,
                    entropy_loss,
                    values_buffer,
                    logits_buffer
                )
                self._last_obs, _ = self._env.reset()
                self._done = False
                episode_return = 0
                episode_length = 0

        self._env.close()
        self._wandb_logger.finish()

    def _get_current_lr(self) -> float:
        """
        Calculate current learning rate based on linear annealing schedule.
        
        Returns:
            The current learning rate
        """
        progress = min(self._global_step.value, self._total_timesteps) / self._total_timesteps
        return self._initial_lr + progress * (self._final_lr - self._initial_lr)

