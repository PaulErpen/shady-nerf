from lodnelf.metrics.abstract_image_metric import (
    AbstractImageMetric,
    AbstractMetricResult,
)
import torch


class PsnrMetricResult(AbstractMetricResult):
    def __init__(self, psnr: float, n: int = 1):
        self.psnr = psnr
        self.n = n

    def is_better_than(self, other: "PsnrMetricResult") -> bool:
        if self.n != other.n:
            raise ValueError(
                f"Expected the number of images to be the same, got {self.n} and {other.n}"
            )
        return self.psnr > other.psnr

    def merge(self, other: "PsnrMetricResult") -> "PsnrMetricResult":
        return PsnrMetricResult(
            (self.psnr * self.n + other.psnr * other.n) / (self.n + other.n),
            self.n + other.n,
        )

    def get_value(self) -> float:
        return self.psnr


class PsnrMetric(AbstractImageMetric):
    def __call__(
        self, image1: torch.Tensor, image2: torch.Tensor
    ) -> "AbstractMetricResult":
        if image1.shape[-1] != 3 and image1.shape[-1] != 4:
            raise ValueError(
                f"Expected image1 to have 3 or 4 channels, got {image1.shape[-1]}"
            )
        if image2.shape[-1] != 3 and image2.shape[-1] != 4:
            raise ValueError(
                f"Expected image2 to have 3 or 4 channels, got {image2.shape[-1]}"
            )
        if image1.shape != image2.shape:
            raise ValueError(
                f"Expected image1 and image2 to have the same shape, got {image1.shape} and {image2.shape}"
            )
        psnr = self.calculate_psnr(image1, image2)
        return PsnrMetricResult(psnr)

    def calculate_psnr(self, image1: torch.Tensor, image2: torch.Tensor) -> float:
        max_image_val = max(torch.max(image1).item(), torch.max(image2).item())
        mse = torch.mean((image1 - image2) ** 2)
        return 20 * torch.log10(max_image_val / torch.sqrt(mse)).item()
