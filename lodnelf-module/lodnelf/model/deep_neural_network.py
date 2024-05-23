import torch.nn as nn
from lodnelf.geometry import geometry
from lodnelf.model.components.fc_layer import FCLayer


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DeepNeuralNetwork, self).__init__()

        # Ensure hidden_dims is a list of dimensions
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Create the list of layers
        layers = []

        # Add the input layer
        layers.append(FCLayer(input_dim, hidden_dims[0]))

        # Add the hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(FCLayer(hidden_dims[i - 1], hidden_dims[i]))

        # Add the output layer
        layers.append(FCLayer(hidden_dims[-1], output_dim))

        # Create the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


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
