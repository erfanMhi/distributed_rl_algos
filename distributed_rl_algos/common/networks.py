"""Common neural network architectures for RL algorithms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class NatureCNN(nn.Module):
    """CNN architecture from DQN Nature paper."""
    
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        features_dim: int = 512
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.conv(torch.zeros(1, input_channels, 84, 84)).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.conv(x / 255.0))

class MLPNetwork(nn.Module):
    """Simple MLP network."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation_fn: nn.Module = nn.ReLU
    ):
        super().__init__()
        
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, hidden_dim),
                activation_fn(),
            ])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class RecurrentNetwork(nn.Module):
    """Recurrent network for handling sequential data."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        rnn_layers: int = 1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=rnn_layers,
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        
        output, hidden = self.rnn(x, hidden)
        output = self.output(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            self.rnn_layers,
            batch_size,
            self.hidden_dim,
            device=next(self.parameters()).device
        ) 