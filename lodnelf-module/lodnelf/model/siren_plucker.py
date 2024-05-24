from typing import List
from torch import nn
from lodnelf.model.components.sine_layer import SineLayer
from lodnelf.geometry import geometry


class SirenPlucker(nn.Module):
    def __init__(self, hidden_dims: List[int]):
        super(SirenPlucker, self).__init__()

        # Ensure hidden_dims is a list of dimensions
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Create the list of layers
        layers = []

        # Add the input layer
        layers.append(SineLayer(6, hidden_dims[0], is_first=True))

        # Add the hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(SineLayer(hidden_dims[i - 1], hidden_dims[i]))

        # Add the output layer
        layers.append(SineLayer(hidden_dims[-1], 3))

        # Create the model
        self.siren = nn.Sequential(*layers)

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
        x = self.siren(plucker_embeddings)

        out_dict["rgb"] = x
        return out_dict
