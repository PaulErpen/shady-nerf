import torch
import torch.nn as nn
import numpy as np
from lodnelf.geometry import geometry


class FourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=10):
        super(FourierFeatures, self).__init__()
        self.B = nn.Parameter(
            scale * torch.randn((input_dim, mapping_size)), requires_grad=False
        )

    def forward(self, x):
        # normalise x as vectors in their last dimension
        x = x / torch.norm(x, dim=-1, keepdim=True)
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PlanarFourier(nn.Module):
    def __init__(
        self, hidden_dim: int, output_dim: int, fourier_mapping_size: int, scale=10
    ):
        super(PlanarFourier, self).__init__()
        self.fourier_features = FourierFeatures(3, fourier_mapping_size, scale=scale)
        self.fc1 = nn.Linear(6 + fourier_mapping_size * 2, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        out_dict = {}

        # embedding
        b, n_qry = input["uv"].shape[0:2]
        plucker_embeddings = geometry.plucker_embedding(
            input["cam2world"], input["uv"], input["intrinsics"]
        )
        plucker_embeddings.requires_grad_(True)
        plucker_embeddings = plucker_embeddings.view(b, n_qry, 6)
        fourier_features = self.fourier_features(plucker_embeddings[:, :, 3:])
        x = torch.cat([plucker_embeddings, fourier_features], dim=-1)

        # network
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)

        out_dict["rgb"] = x

        return out_dict
