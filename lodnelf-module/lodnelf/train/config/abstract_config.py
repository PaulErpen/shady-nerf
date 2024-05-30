from abc import ABC, abstractmethod
from typing import Dict, Tuple
from pathlib import Path
from torch import nn
import torch.utils.data


class AbstractConfig(ABC):
    def __init__(self, config: Dict[str, str]):
        self.config: Dict[str, str] = config

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    @abstractmethod
    def get_train_data_set(self, data_directory: str) -> torch.utils.data.Dataset:
        pass

    @abstractmethod
    def get_output_image_size(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_camera_focal_length(self) -> float:
        pass

    @abstractmethod
    def get_initial_cam2world_matrix(self) -> torch.Tensor:
        pass

    @abstractmethod
    def run(
        self, run_name: str, model_save_path: Path, data_directory: str, device: str
    ):
        pass
