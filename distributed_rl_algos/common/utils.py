"""Common utility functions for distributed RL implementations."""

from collections.abc import Callable
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional
import wandb  # Import wandb

from torch.optim.optimizer import ParamsT

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def create_optimizer(
    parameters: Any,
    optimizer_type: str = "Adam",
    lr: float = 1e-4,
    **kwargs
) -> torch.optim.Optimizer:
    """Create an optimizer instance."""
    optimizer_class = getattr(torch.optim, optimizer_type)
    return optimizer_class(parameters, lr=lr, **kwargs)

def compute_gradient_norm(parameters: Any) -> float:
    """Compute the gradient norm for clipping."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute explained variance for value function analysis."""
    var_y = torch.var(y_true)
    return 1 - torch.var(y_true - y_pred) / (var_y + 1e-8) 

class SharedAdam(torch.optim.Adam):
    """Adam optimizer where states are shared across processes.
    
    This is particularly useful for distributed training scenarios where
    optimizer states need to be synchronized across different processes.
    """
    
    def __init__(self, params: ParamsT, 
                 lr: float = 1e-3, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8,
                 weight_decay: float = 0, 
                 amsgrad: bool = False, 
                 log_params_norm: bool = False):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        
        self._log_params_norm = log_params_norm
        print('log params norm is initialized to:', self._log_params_norm)

        # Share states across all processes
        for param_group in self.param_groups:
            for p in param_group['params']:
                state = self.state[p]
                # Initialize state if not already done
                if len(state) == 0:
                    state['step'] = torch.zeros(1)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                # Share memory for all state tensors
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'].share_memory_()

    def log_optimizer_states(self):
        """Log optimizer states to wandb."""
        for param_group in self.param_groups:
            for i, p in enumerate(param_group['params']):
                state = self.state[p]
                wandb.log({
                    f'step_{i}': state['step'].item(),
                    f'exp_avg_norm_{i}': state['exp_avg'].norm().item(),
                    f'exp_avg_sq_norm_{i}': state['exp_avg_sq'].norm().item(),
                    f'max_exp_avg_sq_norm_{i}': state.get('max_exp_avg_sq', torch.tensor(0.0)).norm().item()
                })

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step."""

        super().step(closure)

        # if self._log_params_norm:
        # self.log_optimizer_states()  # Log states after each step