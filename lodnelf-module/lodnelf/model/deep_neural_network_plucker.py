from typing import List, Literal, Tuple
import torch
import torch.nn as nn
from lodnelf.geometry.plucker_coordinates import plucker_coordinates
from lodnelf.model.components.deep_neural_network import DeepNeuralNetwork


class DeepNeuralNetworkPlucker(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        mode: Literal["rgb", "rgba"] = "rgb",
        init_weights: bool = False,
    ):
        super(DeepNeuralNetworkPlucker, self).__init__()
        self.deep_neural_network = DeepNeuralNetwork(
            6, hidden_dims, 3 if mode == "rgb" else 4, init_weights=init_weights
        )
        self.mode = mode

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        # embedding
        ray_origin, ray_dir_world, _ = input
        plucker = plucker_coordinates(ray_origin, ray_dir_world)

        # network
        x = self.deep_neural_network(plucker)

        return x
