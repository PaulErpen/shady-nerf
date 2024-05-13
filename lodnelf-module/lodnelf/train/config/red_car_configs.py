from typing import Dict
from lodnelf.data.hdf5dataset import get_instance_datasets_hdf5
from lodnelf.model.simple_light_field_model import SimpleLightFieldModel
from lodnelf.train.train_executor import TrainExecutor
import torch.optim
from lodnelf.train.loss import LFLoss
from lodnelf.train.train_handler import TrainHandler
from lodnelf.train.config.abstract_config import AbstractConfig
from pathlib import Path
from lodnelf.util import util


def get_red_car_dataset(data_directory: str):
    return get_instance_datasets_hdf5(
        root=f"{data_directory}/hdf5/cars_train.hdf5",
        max_num_instances=1,
        specific_observation_idcs=None,
        sidelen=128,
        max_observations_per_instance=None,
    )[0]


class SimpleRedCarModelConfig(AbstractConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "SimpleLightFieldModel with latent_dim=256, depth=False, alpha=True",
            "dataset": "cars_train.hdf5",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "SimpleRedCarModel"

    def get_model(self):
        return SimpleLightFieldModel(latent_dim=256, depth=False, alpha=True)

    def get_data_set(self, data_directory: str):
        return get_red_car_dataset(data_directory)

    def run(
        self, run_name: str, model_save_path: Path, data_directory: str, device: str
    ):
        self.config["run_name"] = run_name
        self.config["model_save_path"] = str(model_save_path)
        self.config["data_directory"] = data_directory

        dataset = self.get_data_set(data_directory)

        simple_model = self.get_model()
        executor = TrainExecutor(
            model=simple_model,
            optimizer=torch.optim.AdamW(simple_model.parameters(), lr=1e-4),
            loss=LFLoss(),
            batch_size=1,
            device=device,
        )
        model_save_path.mkdir(exist_ok=True)
        train_handler = TrainHandler(
            max_epochs=150,
            dataset=dataset,
            train_executor=executor,
            prepare_input_fn=lambda x: util.assemble_model_input(x, x),
            stop_after_no_improvement=150,
            model_save_path=model_save_path,
            train_config=self.config,
            group_name="experiment",
        )
        train_handler.run(run_name)
