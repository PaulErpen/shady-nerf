from typing import List
from collections import OrderedDict
import torch
from torch import nn


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class BatchLinear(nn.Linear):
    """A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork."""

    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get("bias", None)
        weight = params["weight"]

        output = input.matmul(
            weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2)
        )
        if bias is not None:
            output += bias.unsqueeze(-2)
        return output


class FCLayer(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity="relu", norm=None):
        super().__init__()
        self.nets: List[nn.Module] = [BatchLinear(in_features, out_features)]

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

    def forward(self, input, params=None):
        return self.net(input)


class FCBlock(nn.Module):
    def __init__(
        self,
        hidden_ch,
        num_hidden_layers,
        in_features,
        out_features,
        outermost_linear=False,
        norm=None,
        activation="relu",
        nonlinearity="relu",
    ):
        super().__init__()

        self.nets = []
        self.nets.append(
            FCLayer(
                in_features=in_features,
                out_features=hidden_ch,
                nonlinearity=nonlinearity,
                norm=norm,
            )
        )

        for i in range(num_hidden_layers):
            self.nets.append(
                FCLayer(
                    in_features=hidden_ch,
                    out_features=hidden_ch,
                    nonlinearity=nonlinearity,
                    norm=norm,
                )
            )

        if outermost_linear:
            self.nets.append(
                BatchLinear(in_features=hidden_ch, out_features=out_features)
            )
        else:
            self.nets.append(
                FCLayer(
                    in_features=hidden_ch,
                    out_features=out_features,
                    nonlinearity=nonlinearity,
                    norm=norm,
                )
            )

        self.net = nn.Sequential(*self.nets)
        self.net.apply(init_weights_normal)

    def forward(self, input, params=None):
        return self.net(input)
