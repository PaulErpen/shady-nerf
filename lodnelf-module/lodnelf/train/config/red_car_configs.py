from typing import Dict, List
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
from lodnelf.model.deep_neural_network import DeepNeuralNetworkPlucker
import random
from lodnelf.model.my_siren import MySiren
from lodnelf.model.sinusoidal_deep_plucker import SinusoidalDeepPlucker

def get_red_car_dataset(data_directory: str, idx: List[int] | None = None):
    return get_instance_datasets_hdf5(
        root=f"{data_directory}/hdf5/cars_train.hdf5",
        max_num_instances=1,
        specific_observation_idcs=idx,
        sidelen=128,
        max_observations_per_instance=None,
    )[0]


class AbstractSimpleRedCarModelConfig(AbstractConfig):
    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        # random sample of indices from 0-50 without replacement
        self.train_idx = random.sample(range(50), 44)
        self.val_idx = [i for i in range(50) if i not in self.train_idx]

    def get_train_data_set(self, data_directory: str):
        return get_red_car_dataset(data_directory, self.train_idx)

    def get_val_data_set(self, data_directory: str):
        return get_red_car_dataset(data_directory, self.val_idx)

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
            "model_description": "PlanarFourier with hidden_dims=[256, 256, 256, 256, 256, 256], output_dim=3, fourier_mapping_size=256",
            "dataset": "cars_train.hdf5",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "SimpleRedCarModelPlanarFourier"

    def get_model(self):
        return PlanarFourier(
            hidden_dims=[256, 256, 256, 256, 256, 256],
            output_dim=3,
            fourier_mapping_size=256,
        )


class SimpleRedCarModelConfigDeepPlucker(AbstractSimpleRedCarModelConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "DeepNeuralNetworkPlucker with latent_dim=[128] * 3",
            "dataset": "cars_train.hdf5",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "SimpleRedCarModelDeepPlucker"

    def get_model(self):
        return DeepNeuralNetworkPlucker(hidden_dims=[128] * 3, output_dim=3)


class SimpleRedCarModelConfigDeepPlucker6(AbstractSimpleRedCarModelConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "DeepNeuralNetworkPlucker with latent_dim=[256] * 6",
            "dataset": "cars_train.hdf5",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "SimpleRedCarModelDeepPlucker6"

    def get_model(self):
        return DeepNeuralNetworkPlucker(hidden_dims=[256] * 6, output_dim=3)
    
class SimpleRedCarModelMySiren(AbstractSimpleRedCarModelConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "MySiren",
            "dataset": "cars_train.hdf5",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "SimpleRedCarModelMySiren"

    def get_model(self):
        return MySiren()

class SimpleRedCarModelConfigSinusoidalDeepPlucker(AbstractSimpleRedCarModelConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "SinusoidalDeepPlucker",
            "dataset": "cars_train.hdf5",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "SimpleRedCarModelSinusoidalDeepPlucker"

    def get_model(self):
        return SinusoidalDeepPlucker()