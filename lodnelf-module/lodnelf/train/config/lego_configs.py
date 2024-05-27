from typing import Dict
from lodnelf.data.lego_dataset import LegoDataset
from lodnelf.train.config.abstract_config import AbstractConfig
from lodnelf.model.deep_neural_network_plucker import DeepNeuralNetworkPlucker
from lodnelf.train.loss import LFLoss
from lodnelf.train.train_executor import TrainExecutor
from lodnelf.train.train_handler import TrainHandler
from lodnelf.train.validation_executor import ValidationExecutor
import torch.utils.data
from pathlib import Path


class DeepPluckerLegoThreeConfig(AbstractConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "DeepPlucker with hidden_dims=[256] * 3",
            "dataset": "cars_train.hdf5",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "DeepPluckerLegoThree"

    def get_model(self):
        return DeepNeuralNetworkPlucker(
            hidden_dims=[256] * 3,
            mode="rgba",
        )

    def get_train_data_set(self, data_directory: str) -> torch.utils.data.Dataset:
        return LegoDataset(data_root=data_directory, split="train")

    def get_val_data_set(self, data_directory: str) -> torch.utils.data.Dataset:
        return LegoDataset(data_root=data_directory, split="val")

    def run(
        self, run_name: str, model_save_path: Path, data_directory: str, device: str
    ):
        self.config["run_name"] = run_name
        self.config["model_save_path"] = str(model_save_path)
        self.config["data_directory"] = data_directory

        train_dataset = self.get_train_data_set(data_directory)
        val_dataset = self.get_val_data_set(data_directory)

        simple_model = self.get_model()
        executor = TrainExecutor(
            model=simple_model,
            optimizer=torch.optim.AdamW(simple_model.parameters(), lr=1e-4),
            loss=LFLoss(mode="rgba"),
            batch_size=1,
            device=device,
            train_data=train_dataset,
        )
        val_executor = ValidationExecutor(
            model=simple_model,
            loss=LFLoss(mode="rgba"),
            batch_size=1,
            device=device,
            val_data=val_dataset,
        )
        model_save_path.mkdir(exist_ok=True)
        train_handler = TrainHandler(
            max_epochs=150,
            train_executor=executor,
            validation_executor=val_executor,
            stop_after_no_improvement=150,
            model_save_path=model_save_path,
            train_config=self.config,
            group_name="experiment",
        )
        train_handler.run(run_name)
