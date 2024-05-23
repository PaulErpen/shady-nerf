from typing import List
from collections import OrderedDict
import torch
from torch import nn
import numpy as np
from lodnelf.model.components.init_weights_normal import init_weights_normal
from lodnelf.model.components.fc_layer import FCLayer


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


class SineLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = float(omega_0)

        self.is_first = is_first

        self.in_features = in_features
        self.linear = BatchLinear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward_with_film(self, input, gamma, beta):
        intermed = self.linear(input)
        return torch.sin(gamma * self.omega_0 * intermed + beta)

    def forward(self, input, params=None):
        intermed = self.linear(input)
        return torch.sin(self.omega_0 * intermed)


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
        special_first=True,
    ):
        super().__init__()
        self.hidden_omega_0 = hidden_omega_0

        layer = SineLayer

        self.nets = []
        self.nets.append(
            layer(
                in_features,
                hidden_features,
                is_first=special_first,
                omega_0=first_omega_0,
            )
        )

        for i in range(hidden_layers):
            self.nets.append(
                layer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=int(hidden_omega_0),
                )
            )

        if outermost_linear:
            final_linear = BatchLinear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / 30.0,
                    np.sqrt(6 / hidden_features) / 30.0,
                )
            self.nets.append(final_linear)
        else:
            self.nets.append(
                layer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=int(hidden_omega_0),
                )
            )

        self.nets = nn.ModuleList(self.nets)
        self.net = nn.Sequential(*self.nets)
        self.net.apply(init_weights_normal)

    def forward(self, coords, params=None):
        x = coords

        return self.net(x)

    def forward_with_film(self, coords, film):
        x = coords

        for i, (layer, layer_film) in enumerate(zip(self.nets, film)):
            if i < len(self.nets) - 1:
                x = layer.forward_with_film(x, layer_film["gamma"], layer_film["beta"])
            else:
                x = layer.forward(x)

        return x
