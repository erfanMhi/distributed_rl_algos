"""Training script for distributed reinforcement learning algorithms.

This script provides a configurable training pipeline for distributed RL algorithms,
currently supporting Asynchronous DQN (ADQN). It uses Hydra for configuration management
and Gymnasium for the RL environments.
"""

import os
import time
from typing import Dict, Any, Callable, Type

import torch
import wandb
import hydra
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

from distributed_rl_algos.common.env_wrappers import (
    FrameStack,
    EpisodeMonitor,
    NormalizeObservation
)
from distributed_rl_algos.common.utils import set_global_seed
from distributed_rl_algos.algorithms import BaseRLAlgorithm
from distributed_rl_algos.algorithms import AsynchronousDQNLearner
from distributed_rl_algos.algorithms import A3CLearner

def make_env(
    env_id: str,
    seed: int,
    idx: int,
    capture_video: bool = False
) -> Callable[[], gym.Env]:
    """Create a wrapped, monitored gymnasium environment.
    
    Args:
        env_id: The ID of the Gymnasium environment to create.
        seed: Random seed for the environment.
        idx: Index of the environment (used for parallel environments).
        capture_video: Whether to capture videos of the environment.
    
    Returns:
        A function that creates and returns a wrapped environment.
    """
    def _init() -> gym.Env:
        # Initialize the base environment
        env = gym.make(env_id)
        
        # # Add wrappers based on env type
        # if isinstance(env.observation_space, gym.spaces.Box):
        #     env = NormalizeObservation(env)
        #     if len(env.observation_space.shape) == 3:  # Image observation
        #         env = FrameStack(env, n_frames=4)
        
        # Set seeds
        env.action_space.seed(seed + idx)
        env.reset(seed=seed + idx)
        return env
    
    return _init


def get_algorithm_class(algo_name: str) -> Type[BaseRLAlgorithm]:
    """Factory function to get the appropriate algorithm class.
    
    Args:
        algo_name: Name of the algorithm to use.
        
    Returns:
        The algorithm class.
        
    Raises:
        ValueError: If the algorithm name is not recognized.
    """
    algorithms = {
        "adqn": AsynchronousDQNLearner,
        "a3c": A3CLearner,
    }
    
    if algo_name not in algorithms:
        raise ValueError(f"Unknown algorithm: {algo_name}. "
                        f"Available algorithms: {list(algorithms.keys())}")
    
    return algorithms[algo_name]


@hydra.main(version_base=None, config_path="../config", config_name="a3c")
def main(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration object containing all training parameters.
    """
    # Set random seeds for reproducibility
    set_global_seed(cfg.algorithm.seed, fully_deterministic=True)
    
    # Initialize the specified algorithm
    algo_cls = get_algorithm_class(cfg.algorithm.name)
    
    # Create environment factory
    env_func = make_env(
        env_id=cfg.env.id,
        seed=cfg.algorithm.seed,
        idx=0
    )
    
    # Convert config to dictionary
    cfg_dict: dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize and train the algorithm
    algo = algo_cls(env_func=env_func, cfg=cfg_dict)
    
    print("\nStarting training...")
    start_time = time.time()
    algo.learn()
    
    # Log training duration
    duration = time.time() - start_time
    minutes, seconds = divmod(duration, 60)
    print(f"\nTraining completed in {duration:.2f} seconds "
          f"({int(minutes)} minutes and {seconds:.2f} seconds)")


if __name__ == "__main__":
    main() 