from pathlib import Path
import unittest
from lodnelf.train.config.abstract_config import AbstractConfig
from lodnelf.viz.holdout_view import HoldoutViewHandler
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

    def test_given_a_mocked_model__when_printing_holdout_view__then_save_the_holdout_view(
        self,
    ):
        holdout_path = Path("./holdouts/")

        self.assertFalse(holdout_path.exists())

        # given
        model = self.MockedModel()
        holdout_handler = HoldoutViewHandler(
            H=256,
            W=256,
            focal_length=1,
            cam2world_matrix=torch.eye(4),
            holdout_path_directory=holdout_path,
        )
        # when
        image_name = "temp_test.png"
        holdout_handler.save_holdout_view(model, image_name)

        # then
        self.assertTrue(holdout_path.exists())
        self.assertTrue((holdout_path / image_name).exists())

        (holdout_path / image_name).unlink()
        holdout_path.rmdir()
