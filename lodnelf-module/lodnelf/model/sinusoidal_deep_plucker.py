from torch import nn
from lodnelf.model.deep_neural_network import DeepNeuralNetwork
from lodnelf.geometry import geometry
import torch
import numpy as np

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=10):
        super(FourierFeatures, self).__init__()
        self.B = nn.Parameter(
            scale * torch.randn((input_dim, mapping_size)), requires_grad=False
        )

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SinusoidalDeepPlucker(nn.Module):
    def __init__(self):
        super(SinusoidalDeepPlucker, self).__init__()

        self.n_freqs = 12

        self.fourier_features = FourierFeatures(6, self.n_freqs)

        self.deep_neural_network = DeepNeuralNetwork(
            input_dim=self.n_freqs * 2,
            hidden_dims=[256, 256, 256],
            output_dim=3,
        )
    
    def normalize_plucker_embeddings(self, plucker_embeddings):
        plucker_embeddings = plucker_embeddings / plucker_embeddings.norm(dim=-1, keepdim=True)
        return plucker_embeddings
    
    def forward(self, input):
        out_dict = {}

        # embedding
        b, n_qry = input["uv"].shape[0:2]
        plucker_embeddings = geometry.plucker_embedding(
            input["cam2world"], input["uv"], input["intrinsics"]
        )
        plucker_embeddings.requires_grad_(True)
        plucker_embeddings = plucker_embeddings.view(b, n_qry, 6)

        plucker_embeddings = self.normalize_plucker_embeddings(plucker_embeddings)
        x = self.fourier_features(plucker_embeddings)

        x = self.deep_neural_network(x)

        out_dict["rgb"] = x

        return out_dict
