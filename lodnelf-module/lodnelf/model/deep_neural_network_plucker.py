import torch.nn as nn
from lodnelf.geometry import geometry
from lodnelf.model.components.deep_neural_network import DeepNeuralNetwork


class DeepNeuralNetworkPlucker(nn.Module):
    def __init__(self, hidden_dims, output_dim):
        super(DeepNeuralNetworkPlucker, self).__init__()
        self.deep_neural_network = DeepNeuralNetwork(6, hidden_dims, output_dim)

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
        x = self.deep_neural_network(plucker_embeddings)

        out_dict["rgb"] = x

        return out_dict
