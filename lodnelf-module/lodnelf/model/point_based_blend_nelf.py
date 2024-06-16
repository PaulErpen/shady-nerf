from typing import List
from torch import nn
import torch
from lodnelf.model.components.deep_neural_network import DeepNeuralNetwork


class PointBasedBlendNelf(nn.Module):
    def __init__(self, initial_points, embed_size, hidden_dims: List[int], alpha=0.5):
        super(PointBasedBlendNelf, self).__init__()

        self.alpha = alpha

        self.points = nn.Parameter(torch.from_numpy(initial_points), requires_grad=True)
        self.embed_size = embed_size

        self.embedded_parameters = nn.Parameter(
            torch.randn(self.points.shape[0], embed_size)
        )

        self.dnn = DeepNeuralNetwork(embed_size, hidden_dims, 4)

    def forward(self, input):
        # input is a tuple of ray_origin, ray_dir_world
        ray_origin, ray_dir_world, _ = input

        # compute the distance to the points
        v = self.points.unsqueeze(0) - ray_origin.unsqueeze(1)
        len_perpendicular = torch.norm(
            torch.cross(v, ray_dir_world.unsqueeze(1)), dim=-1
        )
        ray_alignment = torch.sum(v * ray_dir_world.unsqueeze(1), dim=-1) / torch.norm(
            v, dim=-1
        )
        weights = self.alpha * len_perpendicular + (1 - self.alpha) * ray_alignment
        weights = torch.softmax(weights, dim=-1)

        # compute the weighted sum of the points
        embedded_points = torch.matmul(weights, self.embedded_parameters)

        # compute the output
        return self.dnn(embedded_points)
