from typing import List, Literal, Tuple
from torch import nn
from lodnelf.model.components.sine_layer import SineLayer
from lodnelf.geometry.plucker_coordinates import plucker_coordinates
import torch


class SirenPlucker(nn.Module):
    def __init__(self, hidden_dims: List[int], mode: Literal["rgb", "rgba"] = "rgb"):
        super(SirenPlucker, self).__init__()

        self.mode = mode

        # Ensure hidden_dims is a list of dimensions
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Create the list of layers
        layers = []

        # Add the input layer
        layers.append(SineLayer(6, hidden_dims[0], is_first=True))

        # Add the hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(SineLayer(hidden_dims[i - 1], hidden_dims[i]))

        # Add the output layer
        layers.append(SineLayer(hidden_dims[-1], 3 if mode == "rgb" else 4))

        # Create the model
        self.siren = nn.Sequential(*layers)

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        ray_origin, ray_dir_world, _ = input
        plucker_embeddings = plucker_coordinates(ray_origin, ray_dir_world)

        # network
        x = self.siren(plucker_embeddings)

        return x
