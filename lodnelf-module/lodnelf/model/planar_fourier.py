from typing import List, Literal
from lodnelf.model.components.deep_neural_network import DeepNeuralNetwork
import torch
import torch.nn as nn
from lodnelf.geometry import geometry
from lodnelf.model.components.fourier_features import FourierFeatures


class PlanarFourier(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        fourier_mapping_size: int,
        mode: Literal["rgb", "rgba"] = "rgb",
        init_weights: bool = False,
    ):
        super(PlanarFourier, self).__init__()
        self.fourier_features = FourierFeatures(fourier_mapping_size)

        self.deep_neural_network = DeepNeuralNetwork(
            input_dim=3 + fourier_mapping_size * 2 * 3,
            hidden_dims=hidden_dims,
            output_dim=3 if mode == "rgb" else 4,
            init_weights=init_weights,
        )

    def forward(self, input):
        # embedding
        b, n_qry = input["uv"].shape[0:2]
        plucker_embeddings = geometry.plucker_embedding(
            input["cam2world"], input["uv"], input["intrinsics"]
        )
        plucker_embeddings = plucker_embeddings.view(b, n_qry, 6)
        fourier_features = self.fourier_features(plucker_embeddings[:, :, 3:])
        x = torch.cat([fourier_features, plucker_embeddings[:, :, :3]], dim=-1)

        # network
        x = self.deep_neural_network(x)

        return x
