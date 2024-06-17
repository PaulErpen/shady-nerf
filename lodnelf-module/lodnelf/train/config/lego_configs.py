from typing import Callable, Dict, Tuple
from lodnelf.data.lego_dataset import LegoDataset
from lodnelf.model.full_fourier import FullFourier
from lodnelf.model.my_nerf import NeRF
from lodnelf.model.point_based_nelf import PointBasedNelf
from lodnelf.model.sh_plucker import ShPlucker
from lodnelf.train.config.abstract_config import AbstractConfig
from lodnelf.model.deep_neural_network_plucker import DeepNeuralNetworkPlucker
from lodnelf.model.planar_fourier import PlanarFourier
from lodnelf.model.planar_fourier_skip import PlanarFourierSkip
from lodnelf.train.loss import LFLoss
from lodnelf.train.train_executor import TrainExecutor
from lodnelf.train.train_handler import TrainHandler
from lodnelf.train.validation_executor import ValidationExecutor
from lodnelf.viz.holdout_view import HoldoutViewHandler
import torch.utils.data
from pathlib import Path
import pickle


class AbstractLegoConfig(AbstractConfig):
    def get_train_data_set(self, data_directory: str) -> torch.utils.data.Dataset:
        return LegoDataset(
            data_root=data_directory,
            split="train",
            image_size=self.get_output_image_size(),
            transform=self.get_train_transform(),
        )

    def get_val_data_set(self, data_directory: str) -> torch.utils.data.Dataset:
        return LegoDataset(
            data_root=data_directory,
            split="val",
            image_size=self.get_output_image_size(),
            transform=self.get_val_transform(),
        )

    def get_train_transform(
        self,
    ) -> (
        None
        | Callable[
            [Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ):
        return None

    def get_val_transform(
        self,
    ) -> (
        None
        | Callable[
            [Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ):
        return None

    def get_output_image_size(self) -> Tuple[int, int]:
        return 128, 128

    def get_camera_focal_length(self) -> float:
        return 1111.111

    def get_initial_cam2world_matrix(self):
        return torch.tensor(
            [
                [
                    0.842908501625061,
                    -0.09502744674682617,
                    0.5295989513397217,
                    2.1348819732666016,
                ],
                [
                    0.5380570292472839,
                    0.14886793494224548,
                    -0.8296582698822021,
                    -3.3444597721099854,
                ],
                [
                    7.450582373280668e-09,
                    0.9842804074287415,
                    0.17661221325397491,
                    0.7119466662406921,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).float()

    def get_subset_size(self) -> float | None:
        return None

    def get_holdout_handler(self, holdout_path: Path):
        H, W = self.get_output_image_size()
        return HoldoutViewHandler(
            H=H,
            W=W,
            focal_length=self.get_camera_focal_length(),
            cam2world_matrix=self.get_initial_cam2world_matrix(),
            holdout_path_directory=holdout_path,
        )

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
            holdout_handler=self.get_holdout_handler(model_save_path / "holdouts"),
        )
        train_handler.run(run_name)


class DeepPluckerLegoThreeConfig(AbstractLegoConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(64),
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
            "batch_size": str(64),
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
            "batch_size": str(64),
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
            "batch_size": str(64),
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
            "batch_size": str(64),
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

    def get_train_transform(
        self,
    ) -> (
        None
        | Callable[
            [Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ):
        return lambda x: (x[0] / 4.04, x[1] / 1.01, x[2])

    def get_val_transform(
        self,
    ) -> (
        None
        | Callable[
            [Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ):
        return lambda x: (x[0] / 4.04, x[1] / 1.01, x[2])


class FullSkipFourierLegoThreeConfig(AbstractLegoConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(64),
            "max_epochs": str(150),
            "model_description": "PlanarFourierSkip with hidden_dims=[256] * 5, fourier_mapping_size=6",
            "dataset": "lego rescaled to 128x128",
            "fourier_mapping_size": "6",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "FullSkipFourier5"

    def get_model(self):
        return FullFourier(
            hidden_dims=[256] * 5,
            mode="rgba",
            fourier_mapping_size=6,
            skips=[3],
            init_weights=True,
        )

    def get_train_transform(
        self,
    ) -> (
        None
        | Callable[
            [Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ):
        return lambda x: (x[0] / 4.0311, x[1], x[2])

    def get_val_transform(
        self,
    ) -> (
        None
        | Callable[
            [Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ):
        return lambda x: (x[0] / 4.0311, x[1], x[2])


class LegoShPlucker(AbstractLegoConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(64),
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
            "batch_size": str(64),
            "max_epochs": str(150),
            "model_description": "DeepNeuralNetworkPlucker with hidden_dims=[256] * 3",
            "dataset": "lego in 400x400",
            "subsample": "0.5",
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
        return 400, 400

    def get_subset_size(self) -> float | None:
        return 0.5


class LargeNeRFLego(AbstractLegoConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(64),
            "max_epochs": str(150),
            "model_description": "NeRF with hidden dims [256, 256, 256, 256]",
            "dataset": "lego in 800x800",
            "subsample": "0.01",
            "near": "2.0",
            "far": "6.0",
            "n_samples_along_ray": "64",
            "embed_pos": "6",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "LargeNeRFLego"

    def get_model(self):
        return NeRF(
            near=2.0,
            far=6.0,
            n_coarse_samples=64,
            embed_pos=6,
        )

    def get_output_image_size(self) -> Tuple[int, int]:
        return 800, 800

    def get_subset_size(self) -> float | None:
        return 0.01


class PointBasedLegoNelf(AbstractLegoConfig):
    def __init__(self):
        config: Dict[str, str] = {
            "optimizer": "AdamW (lr 1e-4)",
            "loss": "LFLoss",
            "batch_size": str(64),
            "max_epochs": str(150),
            "model_description": "PointBasedNelf",
            "hidden dims": "[64, 64]",
            "embedding size": "64",
            "dataset": "lego in 400x400",
            "subsample": "0.5",
            "initial_points": "./data/lego-points/lego_256.pkl",
        }
        super().__init__(config)

    def get_name(self) -> str:
        return "PointBasedLegoNelf"

    def get_model(self):
        initial_points = pickle.load(open(self.config["initial_points"], "rb"))
        return PointBasedNelf(
            initial_point=initial_points,
            hidden_dims=[64, 64],
            point_embedding_size=64,
        )

    def get_output_image_size(self) -> Tuple[int, int]:
        return 400, 400

    def get_subset_size(self) -> float | None:
        return 0.5
