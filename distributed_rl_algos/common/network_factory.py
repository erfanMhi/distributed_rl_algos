from typing import Dict
import torch.nn as nn
from .networks import MLPNetwork, NatureCNN, RecurrentNetwork

class NetworkFactory:
    """Factory class for creating neural network architectures."""
    
    @staticmethod
    def create_network(
        network_config: Dict,
        input_dim: int,
        output_dim: int,
        **kwargs
    ) -> nn.Module:
        """
        Create a neural network based on the specified config.
        
        Args:
            network_config: Dictionary containing network configuration
                - type: Type of network ('mlp', 'nature_cnn', or 'recurrent')
                - hidden_sizes: List of hidden layer sizes
                - Additional network-specific parameters
            input_dim: Input dimension
            output_dim: Output dimension
            **kwargs: Additional arguments specific to network types
        
        Returns:
            nn.Module: The created neural network
        """
        network_type = network_config.get('type', 'mlp')
        hidden_sizes = network_config.get('hidden_sizes', [64])
        
        if network_type == "mlp":
            return MLPNetwork(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_sizes
            )
        elif network_type == "nature_cnn":
            return NatureCNN(
                input_channels=input_dim,
                output_dim=output_dim,
                features_dim=network_config.get('features_dim', 512)
            )
        elif network_type == "recurrent":
            return RecurrentNetwork(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_sizes[0] if hidden_sizes else 256,
                rnn_layers=network_config.get('rnn_layers', 1)
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}") 