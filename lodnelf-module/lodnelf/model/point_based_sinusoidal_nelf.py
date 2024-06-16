from typing import List, Tuple
import torch
from torch import nn
import numpy as np
from lodnelf.model.components.deep_neural_network import DeepNeuralNetwork


class PointBasedSinusoidalNelf(nn.Module):
    def __init__(
        self,
        initial_point: np.ndarray,
        hidden_dims: List[int],
        point_embedding_size: int,
        sinusoidal_embedding_size: int,
    ):
        super(PointBasedSinusoidalNelf, self).__init__()

        self.points = nn.Parameter(torch.from_numpy(initial_point), requires_grad=True)

        self.point_embedding_size = point_embedding_size
        self.sinusoidal_embedding_size = sinusoidal_embedding_size

        self.dnn = DeepNeuralNetwork(
            point_embedding_size + sinusoidal_embedding_size, hidden_dims, 4
        )

        self.point_embedding_parameters = nn.Parameter(
            torch.randn(self.points.shape[0], point_embedding_size)
        )

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        ray_origin, ray_dir_world, _ = input

        # compute the distance from the ray origin
        v = self.points.unsqueeze(0) - ray_origin.unsqueeze(1)
        len_perpendicular = torch.norm(
            torch.cross(v, ray_dir_world.unsqueeze(1)), dim=-1
        )
        ray_alignment = torch.sum(v * ray_dir_world.unsqueeze(1), dim=-1) / torch.norm(
            v, dim=-1
        )

        return len_perpendicular, ray_alignment
