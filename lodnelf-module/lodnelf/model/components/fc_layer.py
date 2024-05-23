import torch.nn as nn
from typing import List, Literal
from lodnelf.model.components.init_weights_normal import init_weights_normal


class FCLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        nonlinearity: Literal["relu"] | Literal["leaky_relu"] = "relu",
        norm: Literal["layernorm"] | Literal["layernorm_na"] | None = None,
    ):
        super().__init__()
        self.nets: List[nn.Module] = [nn.Linear(in_features, out_features)]

        if norm == "layernorm":
            self.nets.append(
                nn.LayerNorm([out_features], elementwise_affine=True),
            )
        elif norm == "layernorm_na":
            self.nets.append(
                nn.LayerNorm([out_features], elementwise_affine=False),
            )

        if nonlinearity == "relu":
            self.nets.append(nn.ReLU(inplace=True))
        elif nonlinearity == "leaky_relu":
            self.nets.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*self.nets)
        self.net.apply(init_weights_normal)

    def forward(self, input):
        return self.net(input)
