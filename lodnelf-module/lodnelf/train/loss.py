from abc import ABC, abstractmethod
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class _LossFn(ABC):
    @abstractmethod
    def __call__(self, model_out, batch, model=None, val=False) -> _Loss:
        pass


def image_loss(model_out, batch, mask=None):
    batch_rgb = batch["rgb"]
    return nn.MSELoss()(batch_rgb, model_out["rgb"]) * 200


class LFLoss(_LossFn):
    def __init__(self, l2_weight=1, reg_weight=1e2):
        self.l2_weight = l2_weight
        self.reg_weight = reg_weight

    def __call__(self, model_out, batch, model=None, val=False) -> _Loss:
        return image_loss(model_out, batch)
