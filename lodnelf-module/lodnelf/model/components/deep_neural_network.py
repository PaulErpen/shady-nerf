from typing import List
from torch import nn
from lodnelf.model.components.fc_layer import FCLayer


class DeepNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        init_weights: bool = False,
    ):
        super(DeepNeuralNetwork, self).__init__()

        # Ensure hidden_dims is a list of dimensions
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Create the list of layers
        layers = []

        # Add the input layer
        layers.append(FCLayer(input_dim, hidden_dims[0], init_weights=init_weights))

        # Add the hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(
                FCLayer(hidden_dims[i - 1], hidden_dims[i], init_weights=init_weights)
            )

        # Add the output layer
        layers.append(FCLayer(hidden_dims[-1], output_dim, init_weights=init_weights))

        # Create the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
