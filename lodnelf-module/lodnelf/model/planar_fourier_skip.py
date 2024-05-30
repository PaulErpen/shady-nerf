from torch import nn
from lodnelf.model.components.deep_neural_network import DeepNeuralNetwork
from lodnelf.model.planar_fourier import PlanarFourier
from typing import List, Literal, Tuple
import torch


class PlanarFourierSkip(nn.Module):
    def __init__(
        self,
        hd_before_skip: List[int],
        hd_after_skip: List[int],
        fourier_mapping_size: int,
        mode: Literal["rgb", "rgba"] = "rgb",
        init_weights: bool = False,
    ):
        super(PlanarFourierSkip, self).__init__()

        self.planar_fourier = PlanarFourier(
            hidden_dims=hd_before_skip,
            fourier_mapping_size=fourier_mapping_size,
            init_weights=init_weights,
            mode="custom",
            custom_out_dim=hd_before_skip[-1],
        )

        self.deep_neural_network_2 = DeepNeuralNetwork(
            input_dim=hd_before_skip[-1] + 6 + fourier_mapping_size * 2 * 3,
            hidden_dims=hd_after_skip,
            output_dim=3 if mode == "rgb" else 4,
            init_weights=init_weights,
        )

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        # embedding
        identity = self.planar_fourier.prepare_fourier_features(input)
        x = self.planar_fourier(input)
        x = torch.cat([x, identity], dim=-1)
        x = self.deep_neural_network_2(x)

        return x
