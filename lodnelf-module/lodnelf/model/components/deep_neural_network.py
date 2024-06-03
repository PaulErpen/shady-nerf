from typing import List
from torch import nn
from lodnelf.model.components.fc_layer import FCLayer
import torch


class DeepNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        init_weights: bool = False,
        skips: List[int] = [],
    ):
        """
        Initialize the Deep Neural Network.

        Args:
            input_dim (int): The input dimension.
            hidden_dims (List[int]): The list of hidden dimensions.
            output_dim (int): The output dimension.
            init_weights (bool): Whether to initialize the weights.
            skips (List[int]): The list of layers to skip.
        """
        super(DeepNeuralNetwork, self).__init__()
        self.skips = skips

        # Ensure hidden_dims is a list of dimensions
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Create the list of layers
        layers = []

        # Add the input layer
        layers.append(FCLayer(input_dim, hidden_dims[0], init_weights=init_weights))

        # Add the hidden layers
        for i in range(1, len(hidden_dims)):
            in_dim = hidden_dims[i - 1]
            if i in skips:
                in_dim += input_dim
            layers.append(FCLayer(in_dim, hidden_dims[i], init_weights=init_weights))

        # Add the output layer
        layers.append(FCLayer(hidden_dims[-1], output_dim, init_weights=init_weights))

        # Create the model
        self.linears = nn.ModuleList(layers)

    def forward(self, x):
        identity = x
        for idx, layer in enumerate(self.linears):
            if idx in self.skips:
                x = torch.cat([x, identity], dim=-1)
            x = layer(x)
        return x
