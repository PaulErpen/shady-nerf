from torch import nn
import torch
import numpy as np


class FourierFeatures(nn.Module):
    def __init__(self, mapping_size: int):
        super(FourierFeatures, self).__init__()
        self.exp = 2 ** torch.arange(0, mapping_size).float() * np.pi
        self.exp = nn.Parameter(self.exp, requires_grad=False)

    def forward(self, x: torch.Tensor):
        b, d = x.shape
        x_proj = torch.einsum("bd, p -> bdp", x, self.exp).view(b, d * len(self.exp))
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward_batched(self, x: torch.Tensor):
        b, n_samples, d = x.shape
        x_proj = torch.einsum("bnd, p -> bndp", x, self.exp).view(
            b, n_samples, d * len(self.exp)
        )
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
