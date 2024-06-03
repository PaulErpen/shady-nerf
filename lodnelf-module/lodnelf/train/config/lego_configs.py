from typing import Dict, Tuple
from lodnelf.data.lego_dataset import LegoDataset
from lodnelf.model.full_fourier import FullFourier
from lodnelf.model.sh_plucker import ShPlucker
from lodnelf.train.config.abstract_config import AbstractConfig
from lodnelf.model.deep_neural_network_plucker import DeepNeuralNetworkPlucker
from lodnelf.model.planar_fourier import PlanarFourier
from lodnelf.model.planar_fourier_skip import PlanarFourierSkip
from lodnelf.train.loss import LFLoss
from lodnelf.train.train_executor import TrainExecutor
from lodnelf.train.train_handler import TrainHandler
from lodnelf.train.validation_executor import ValidationExecutor
import torch.utils.data
from pathlib import Path


class AbstractLegoConfig(AbstractConfig):
    def get_train_data_set(self, data_directory: str) -> torch.utils.data.Dataset:
        return LegoDataset(
            data_root=data_directory,
            split="train",
            image_size=self.get_output_image_size(),
        )

    def get_val_data_set(self, data_directory: str) -> torch.utils.data.Dataset:
        return LegoDataset(
            data_root=data_directory,
            split="val",
            image_size=self.get_output_image_size(),
        )

    def get_output_image_size(self) -> Tuple[int, int]:
        return 128, 128

    def get_camera_focal_length(self) -> float:
        return 1111.111 * (self.get_output_image_size()[0] / 800)

    def get_initial_cam2world_matrix(self):
        return torch.tensor(
            [
                [
                    -0.9999021887779236,
                    0.004192245192825794,
                    -0.013345719315111637,
                    -0.05379832163453102,
                ],
                [
                    -0.013988681137561798,
                    -0.2996590733528137,
                    0.95394366979599,
                    3.845470428466797,
                ],
                [
                    -4.656612873077393e-10,
                    0.9540371894836426,
                    0.29968830943107605,
                    1.2080823183059692,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).float()

    def get_subset_size(self) -> float | None:
        return None

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
            batch_size=64,
            device=device,
            train_data=train_dataset,
            subset_size=self.get_subset_size(),
        )
        val_executor = ValidationExecutor(
            model=simple_model,
            loss=LFLoss(),
            batch_size=64,
            device=device,
            val_data=val_dataset,
            subset_size=self.get_subset_size(),
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


class DeepPluckerLegoThreeConfig(AbstractLegoConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "DeepPlucker with hidden_dims=[256] * 3",
            "dataset": "lego rescaled to 128x128",
            "train_batch_size": "64",
            "val_batch_size": "64",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "DeepPluckerLegoThree"

    def get_model(self):
        return DeepNeuralNetworkPlucker(
            hidden_dims=[256] * 3,
            mode="rgba",
            init_weights=True,
        )


class DeepPluckerLegoSixConfig(AbstractLegoConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "DeepPlucker with hidden_dims=[256] * 6",
            "dataset": "lego rescaled to 128x128",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "DeepPluckerLegoSix"

    def get_model(self):
        return DeepNeuralNetworkPlucker(
            hidden_dims=[256] * 6,
            mode="rgba",
            init_weights=True,
        )


class PlanarFourierLegoThreeConfig(AbstractLegoConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "PlanarFourier with hidden_dims=[256] * 3, mapping size 6",
            "dataset": "lego rescaled to 128x128",
            "fourier_mapping_size": "6",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "PlanarFourierLegoThree"

    def get_model(self):
        return PlanarFourier(
            hidden_dims=[256] * 3,
            mode="rgba",
            fourier_mapping_size=6,
            init_weights=True,
        )


class PlanarFourierLegoThreeToThreeConfig(AbstractLegoConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "PlanarFourierSkip with hd_before_skip=[256] * 3, hd_after_skip=[256] * 3",
            "dataset": "lego rescaled to 128x128",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "PlanarFourierLegoThreeToThree"

    def get_model(self):
        return PlanarFourierSkip(
            hd_before_skip=[256] * 3,
            hd_after_skip=[256] * 3,
            mode="rgba",
            fourier_mapping_size=64,
            init_weights=True,
        )


class FullFourierLegoThreeConfig(AbstractLegoConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "PlanarFourierSkip with hidden_dims=[256] * 3, fourier_mapping_size=6",
            "dataset": "lego rescaled to 128x128",
            "fourier_mapping_size": "6",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "FullFourierThree"

    def get_model(self):
        return FullFourier(
            hidden_dims=[256] * 3,
            mode="rgba",
            fourier_mapping_size=6,
            init_weights=True,
        )


class LegoShPlucker(AbstractLegoConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "ShPlucker with hidden_dims=[256] * 3",
            "dataset": "lego rescaled to 128x128",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "LegoShPlucker"

    def get_model(self):
        return ShPlucker(mode="rgba")


class LargeDeepPluckerLego(AbstractLegoConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(1),
            "max_epochs": str(150),
            "model_description": "ShPlucker with hidden_dims=[256] * 3",
            "dataset": "lego in 800x800",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "DeepNeuralNetworkPluckerLegoLarge"

    def get_model(self):
        return DeepNeuralNetworkPlucker(
            hidden_dims=[256] * 3,
            mode="rgba",
            init_weights=True,
        )

    def get_output_image_size(self) -> Tuple[int, int]:
        return 800, 800

    def get_subset_size(self) -> float | None:
        return 0.1
