from torch import nn
from lodnelf.model.custom_layers import Siren
from lodnelf.model.deep_neural_network import DeepNeuralNetwork
from lodnelf.geometry import geometry
import torch


class MySiren(nn.Module):
    def __init__(self):
        super(MySiren, self).__init__()

        self.siren = Siren(
            in_features=6,
            hidden_features=256,
            hidden_layers=3,
            out_features=256,
            outermost_linear=True,
            hidden_omega_0=30,
            first_omega_0=int(30),
        )

        self.deep_neural_network = DeepNeuralNetwork(
            input_dim=256,
            hidden_dims=[256, 256, 256],
            output_dim=256,
        )

        self.combined_layers = nn.Sequential(
            nn.Linear(256 * 2, 256), nn.ReLU(), nn.Linear(256, 3)
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

        # network
        siren_out = self.siren(plucker_embeddings)
        dnn_out = self.deep_neural_network(siren_out)
        x = torch.cat([siren_out, dnn_out], dim=-1)
        x = self.combined_layers(x)

        out_dict["rgb"] = x
        return out_dict
