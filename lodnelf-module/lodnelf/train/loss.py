from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class _LossFn(ABC):
    @abstractmethod
    def __call__(
        self,
        model_out: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        model=None,
        val=False,
    ) -> _Loss:
        pass


class LFLoss(_LossFn):
    def __init__(self, l2_weight=1, reg_weight=1e2):
        self.l2_weight = l2_weight
        self.reg_weight = reg_weight
        self.loss = nn.MSELoss()

    def __call__(
        self,
        model_out: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> _Loss:
        ray_origin, ray_dir_world, color = batch
        if model_out.shape[-1] == 3 and color.shape[-1] == 4:
            color = color[:, :3]
        return self.loss(color, model_out) * 200
