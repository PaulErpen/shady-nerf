from typing import List
from lodnelf.model.deep_neural_network import DeepNeuralNetwork
import torch
import torch.nn as nn
from lodnelf.geometry import geometry
from lodnelf.model.components.fourier_features import FourierFeatures


class PlanarFourier(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        output_dim: int,
        fourier_mapping_size: int,
    ):
        super(PlanarFourier, self).__init__()
        self.fourier_features = FourierFeatures(fourier_mapping_size)

        self.deep_neural_network = DeepNeuralNetwork(
            input_dim=3 + fourier_mapping_size * 2 * 3,
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
        plucker_embeddings = plucker_embeddings.view(b, n_qry, 6)
        fourier_features = self.fourier_features(plucker_embeddings[:, :, :3]) * 30.0
        x = torch.cat([fourier_features, plucker_embeddings[:, :, 3:]], dim=-1)

        x.requires_grad_(True)


        # network
        x = self.deep_neural_network(x)

        out_dict["rgb"] = x

        return out_dict
