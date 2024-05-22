from typing import Dict
from lodnelf.data.hdf5dataset import get_instance_datasets_hdf5
from lodnelf.model.simple_light_field_model import SimpleLightFieldModel
from lodnelf.train.train_executor import TrainExecutor
import torch.optim
from lodnelf.train.loss import LFLoss
from lodnelf.train.train_handler import TrainHandler
from lodnelf.train.validation_executor import ValidationExecutor
from lodnelf.train.config.abstract_config import AbstractConfig
from pathlib import Path
from lodnelf.model.planar_fourier import PlanarFourier


def get_red_car_train_dataset(data_directory: str):
    return get_instance_datasets_hdf5(
        root=f"{data_directory}/hdf5/cars_train.hdf5",
        max_num_instances=1,
        specific_observation_idcs=list(range(42)),
        sidelen=128,
        max_observations_per_instance=None,
    )[0]


def get_red_car_val_dataset(data_directory: str):
    return get_instance_datasets_hdf5(
        root=f"{data_directory}/hdf5/cars_train.hdf5",
        max_num_instances=1,
        specific_observation_idcs=list(range(42, 50)),
        sidelen=128,
        max_observations_per_instance=None,
    )[0]


class AbstractSimpleRedCarModelConfig(AbstractConfig):
    def __init__(self, config: Dict[str, str]):
        super().__init__(config)

    def get_train_data_set(self, data_directory: str):
        return get_red_car_train_dataset(data_directory)

    def get_val_data_set(self, data_directory: str):
        return get_red_car_val_dataset(data_directory)

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
            loss=LFLoss(),
            batch_size=1,
            device=device,
            train_data=train_dataset,
        )
        val_executor = ValidationExecutor(
            model=simple_model,
            loss=LFLoss(),
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


class SimpleRedCarModelConfigDepthAlpha(AbstractSimpleRedCarModelConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "SimpleLightFieldModel with latent_dim=256, depth=True, alpha=True",
            "dataset": "cars_train.hdf5",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "SimpleRedCarModelDepthAlpha"

    def get_model(self):
        return SimpleLightFieldModel(latent_dim=256, depth=True, alpha=True)


class SimpleRedCarModelConfig(AbstractSimpleRedCarModelConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "SimpleLightFieldModel with latent_dim=256, depth=False, alpha=False",
            "dataset": "cars_train.hdf5",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "SimpleRedCarModel"

    def get_model(self):
        return SimpleLightFieldModel(latent_dim=256, depth=False, alpha=False)


class SimpleRedCarModelConfigSiren(AbstractSimpleRedCarModelConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "SimpleLightFieldModel (Siren) with latent_dim=256, depth=False, alpha=False",
            "dataset": "cars_train.hdf5",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "SimpleRedCarModelSiren"

    def get_model(self):
        return SimpleLightFieldModel(
            latent_dim=256, depth=False, alpha=False, model_type="siren"
        )


class SimpleRedCarModelConfigPlanarFourier(AbstractSimpleRedCarModelConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "PlanarFourier with hidden_dim=256, output_dim=3, fourier_mapping_size=128",
            "dataset": "cars_train.hdf5",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "SimpleRedCarModelPlanarFourier"

    def get_model(self):
        return PlanarFourier(hidden_dim=256, output_dim=3, fourier_mapping_size=128)
