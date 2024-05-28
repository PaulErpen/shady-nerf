from typing import List, Literal
import torch.nn as nn
from lodnelf.geometry import geometry
from lodnelf.model.components.deep_neural_network import DeepNeuralNetwork
import torch


class DeepMLP(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        mode: Literal["rgb", "rgba"] = "rgb",
        init_weights: bool = False,
    ):
        super(DeepMLP, self).__init__()
        self.deep_neural_network = DeepNeuralNetwork(
            6, hidden_dims, 3 if mode == "rgb" else 4, init_weights=init_weights
        )
        self.mode = mode

    def forward(self, input):
        # embedding
        uv = input["uv"]
        cam2world = input["cam2world"]
        intrinsics = input["intrinsics"]
        ray_dirs = geometry.get_ray_directions(
            uv, cam2world=cam2world, intrinsics=intrinsics
        )
        cam_pos = geometry.get_ray_origin(cam2world)
        model_input = torch.cat([ray_dirs, cam_pos], dim=-1)
        model_input.requires_grad_(True)

        # network
        x = self.deep_neural_network(model_input)

        return x
