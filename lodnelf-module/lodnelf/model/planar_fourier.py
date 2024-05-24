from typing import List
from lodnelf.model.deep_neural_network import DeepNeuralNetwork
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
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PlanarFourier(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        output_dim: int,
        fourier_mapping_size: int,
        scale=10,
    ):
        super(PlanarFourier, self).__init__()
        self.fourier_features = FourierFeatures(3, fourier_mapping_size, scale=scale)

        self.deep_neural_network = DeepNeuralNetwork(
            input_dim=6 + fourier_mapping_size * 2,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )

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
        x = self.deep_neural_network(x)

        out_dict["rgb"] = x

        return out_dict