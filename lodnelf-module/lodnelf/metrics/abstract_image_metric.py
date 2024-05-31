from abc import ABC, abstractmethod

import torch


class AbstractImageMetric(ABC):
    @abstractmethod
    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> "AbstractMetricResult":
        pass


class AbstractMetricResult(ABC):
    @abstractmethod
    def is_better_than(self, other: "AbstractMetricResult") -> bool:
        pass
