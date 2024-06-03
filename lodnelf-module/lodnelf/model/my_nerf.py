from typing import Literal
from torch import nn
import torch
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(
        self,
        near: float,
        far: float,
        n_samples_along_ray: int,
    ):
        self.near = near
        self.far = far
        self.n_samples_along_ray = n_samples_along_ray

    def forward(self, x):
        pass

    def compute_sample_points(
        self, ray_origins: torch.Tensor, ray_directions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the sample points along the rays defined by the ray origins and directions.
        Args:
            ray_origins (torch.Tensor): The origins of the rays [batch_size, 3].
            ray_directions (torch.Tensor): The directions of the rays [batch_size, 3].
        Returns:
            torch.Tensor: The sample points along the rays [batch_size, n_samples_along_ray, 3].
        """
        batch_size, dim = ray_origins.shape

        t_vals = torch.linspace(self.near, self.far, self.n_samples_along_ray).to(
            ray_origins
        )
        t_vals = self.near + t_vals * (self.far - self.near)
        t_vals = t_vals.repeat(batch_size, dim, 1).transpose(1, 2)

        ray_origins = ray_origins.repeat(1, 1, self.n_samples_along_ray).view(
            batch_size, self.n_samples_along_ray, dim
        )
        ray_directions = ray_directions.repeat(1, 1, self.n_samples_along_ray).view(
            batch_size, self.n_samples_along_ray, dim
        )

        sample_points = ray_origins + t_vals * ray_directions

        return sample_points
