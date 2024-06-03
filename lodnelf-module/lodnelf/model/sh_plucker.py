from typing import Literal
from lodnelf.geometry.plucker_coordinates import plucker_coordinates
from lodnelf.geometry.embed_spherical_harmonics import embed_spherical_harmonics
from lodnelf.model.components.deep_neural_network import DeepNeuralNetwork
from torch import nn
import torch


class ShPlucker(nn.Module):
    def __init__(self, mode: Literal["rbg", "rgba"] = "rbg"):
        super(ShPlucker, self).__init__()

        self.deep_neural_network = DeepNeuralNetwork(
            input_dim=64 + 6,
            hidden_dims=[256, 256, 256],
            output_dim=3 if mode == "rbg" else 4,
        )

    def forward(self, x):
        ray_origin, ray_direction, _ = x
        embedding = embed_spherical_harmonics(ray_direction)
        plucker = plucker_coordinates(
            ray_origins=ray_origin, ray_directions=ray_direction
        )
        x = torch.cat(
            [embedding, plucker],
            dim=-1,
        )
        return self.deep_neural_network(x)
