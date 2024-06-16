from typing import List, Tuple
import torch
from torch import nn
import numpy as np
from lodnelf.model.components.deep_neural_network import DeepNeuralNetwork


class PointBasedNelf(nn.Module):
    def __init__(
        self,
        initial_point: np.ndarray,
        hidden_dims: List[int],
        point_embedding_size: int,
    ):
        super(PointBasedNelf, self).__init__()

        self.points = nn.Parameter(torch.from_numpy(initial_point), requires_grad=True)

        self.point_embedding_size = point_embedding_size

        self.dnn = DeepNeuralNetwork(point_embedding_size * 2, hidden_dims, 4)

        self.point_embedding_parameters = nn.Parameter(
            torch.randn(self.points.shape[0], point_embedding_size)
        )

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        ray_origin, ray_dir_world, _ = input

        # compute the distance from the ray origin
        v = self.points.unsqueeze(0) - ray_origin.unsqueeze(1)
        len_perpendicular = torch.norm(
            torch.cross(v, ray_dir_world.unsqueeze(1), dim=-1), dim=-1
        )
        ray_alignment = torch.norm(v * ray_dir_world.unsqueeze(1), dim=-1)

        # compute the weights for the points
        w_perpendicular = torch.exp(-(len_perpendicular**2))
        w_alignment = torch.exp(-(ray_alignment**2))

        # weight the point embeddings
        weighted_perpendicular = torch.einsum(
            "bn,ne -> ben", w_perpendicular, self.point_embedding_parameters
        )
        weighted_alignment = torch.einsum(
            "bn,ne -> ben", w_alignment, self.point_embedding_parameters
        )

        # sum the weighted embeddings
        s_perpendicular = torch.sum(weighted_perpendicular, dim=-1)
        s_alignment = torch.sum(weighted_alignment, dim=-1)

        s = torch.cat([s_perpendicular, s_alignment], dim=-1)

        return self.dnn(s)
