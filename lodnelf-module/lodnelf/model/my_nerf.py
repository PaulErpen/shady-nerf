from typing import Tuple
from lodnelf.geometry.embed_spherical_harmonics import embed_spherical_harmonics
from lodnelf.model.components.deep_neural_network import DeepNeuralNetwork
from lodnelf.model.components.fourier_features import FourierFeatures
from torch import nn
import torch


class NeRF(nn.Module):
    def __init__(
        self,
        near: float,
        far: float,
        n_samples_along_ray: int,
        embed_pos: int = 6,
    ):
        super(NeRF, self).__init__()
        self.near = near
        self.far = far
        self.n_samples_along_ray = n_samples_along_ray

        self.fourier_features = FourierFeatures(embed_pos)

        self.dnn_features = DeepNeuralNetwork(
            input_dim=embed_pos * 3 * 2 + 3,
            hidden_dims=[256, 256, 256, 256],
            output_dim=256,
            skips=[3],
        )

        self.alpha = nn.Linear(256, 1)

        self.rgb = nn.Linear(256 + 64, 3)

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        rgb_map, depth_map, acc_map, point_alpha = self.extended_forward(x)
        return rgb_map

    def extended_forward(
        self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x[0].shape[0]
        ray_origin, ray_dir, col = x
        sample_points = self.compute_sample_points(ray_origin, ray_dir)
        fourier_points = self.fourier_features.forward_batched(sample_points)

        features = self.dnn_features(fourier_points)

        point_alpha = torch.sigmoid(self.alpha(features))

        spherical = embed_spherical_harmonics(ray_dir).expand(
            batch_size, self.n_samples_along_ray, 64
        )
        point_rgb = torch.sigmoid(self.rgb(torch.cat([features, spherical], dim=-1)))

        rgb_map, depth_map, acc_map = self.render_rays(
            point_alpha.squeeze(-1), point_rgb
        )

        return (rgb_map, depth_map, acc_map, point_alpha)

    def render_rays(
        self, point_alpha: torch.Tensor, point_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_vals = torch.linspace(self.near, self.far, self.n_samples_along_ray)
        dist = self.far - self.near
        alpha = 1.0 - torch.exp(-point_alpha * dist)
        weights = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=-1)

        rgb_map = torch.sum(weights[..., None] * point_rgb, -2)
        depth_map = torch.sum(weights * t_vals, -1)
        acc_map = torch.sum(weights, -1)

        return rgb_map, depth_map, acc_map

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

        t_vals = torch.linspace(self.near, self.far, self.n_samples_along_ray)
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
