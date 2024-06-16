from pathlib import Path
import unittest
from lodnelf.train.config.abstract_config import AbstractConfig
from lodnelf.viz.holdout_view import save_holdout_view
from torch import nn
import torch


class HoldoutViewTest(unittest.TestCase):
    class MockedModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            ray_origins, ray_directions, col = x
            batch_size = ray_origins.shape[0]
            return torch.rand((batch_size, 4))

    class MockedConfig(AbstractConfig):
        def __init__(self):
            super(AbstractConfig).__init__()

        def get_name(self):
            return "mocked_config"

        def get_output_image_size(self):
            return 800, 800

        def get_camera_focal_length(self):
            return 1.0

        def get_initial_cam2world_matrix(self):
            return torch.eye(4)

        def get_model(self):
            return None

        def get_train_data_set(self, data_directory: str):
            return None

        def run(
            self, run_name: str, model_save_path: Path, data_directory: str, device: str
        ):
            pass

    def test_given_a_mocked_model__when_printing_holdout_view__then_save_the_holdout_view(
        self,
    ):
        holdout_path = Path("temp_test.png")

        # given
        model = self.MockedModel()
        config = self.MockedConfig()
        # when
        save_holdout_view(model, config, holdout_path)

        # then
        self.assertTrue(holdout_path.exists())

        holdout_path.unlink()
