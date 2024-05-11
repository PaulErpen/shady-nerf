from abc import ABC, abstractmethod
from typing import Dict
from lodnelf.train.train_handler import TrainHandler
from pathlib import Path


class AbstractConfig(ABC):
    def __init__(self, config: Dict[str, str]):
        self.config: Dict[str, str] = config

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def run(self, run_name: str, model_save_path: Path, data_directory: str):
        pass
