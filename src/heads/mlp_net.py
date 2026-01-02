"""MLP network module for final prediction."""

from typing import List

import torch
import torch.nn as nn


class MLPNet(nn.Module):
    """Multi-layer perceptron with skip connections."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float
    ):
        """Initialize MLP network.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension (num_classes)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Build layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.norms = nn.ModuleList()

        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, self.output_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Predictions [batch_size, output_dim]
        """
        for layer, dropout, norm in zip(self.layers, self.dropouts, self.norms):
            identity = x
            x = layer(x)

            if x.shape == identity.shape:
                x = x + identity

            x = self.activation(x)
            x = norm(x)
            x = dropout(x)

        x = self.output_layer(x)
        return x
