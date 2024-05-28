from torch import nn
import torch
import numpy as np


class FourierFeatures(nn.Module):
    def __init__(self, mapping_size: int):
        super(FourierFeatures, self).__init__()
        self.exp = 2 ** torch.arange(0, mapping_size).float() * np.pi
        self.exp = nn.Parameter(self.exp, requires_grad=False)

    def forward(self, x):
        b, q, d = x.shape
        x_proj = torch.einsum("bqd, p -> bqdp", x, self.exp).view(
            b, q, d * len(self.exp)
        )
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
