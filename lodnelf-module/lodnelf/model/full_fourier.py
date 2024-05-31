from typing import List, Literal, Tuple
from lodnelf.geometry.plucker_coordinates import plucker_coordinates
from lodnelf.model.components.deep_neural_network import DeepNeuralNetwork
from lodnelf.model.components.fourier_features import FourierFeatures
from torch import nn
import torch

class FullFourier(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        fourier_mapping_size: int,
        mode: Literal["rgb", "rgba", "custom"] = "rgb",
        init_weights: bool = False,
        custom_out_dim: int | None = None,
    ):
        super(FullFourier, self).__init__()
        self.fourier_features = FourierFeatures(fourier_mapping_size)

        out = 3
        if mode == "rgba":
            out = 4
        elif mode == "custom":
            out = custom_out_dim

        if out is None:
            raise ValueError("out_dim must be provided when mode is 'custom'")

        self.deep_neural_network = DeepNeuralNetwork(
            input_dim=6 + fourier_mapping_size * 2 * 6,
            hidden_dims=hidden_dims,
            output_dim=out,
            init_weights=init_weights,
        )

    def prepare_fourier_features(self, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        ray_origin, ray_dir_world, _ = input
        plucker_embeddings = plucker_coordinates(ray_origin, ray_dir_world)
        return self.fourier_features(plucker_embeddings)

    def forward(self, input):
        # embedding
        x = self.prepare_fourier_features(input)

        # network
        x = self.deep_neural_network(x)

        return x