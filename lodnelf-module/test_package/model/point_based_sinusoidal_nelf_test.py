import unittest
import torch
from lodnelf.model.point_based_sinusoidal_nelf import PointBasedSinusoidalNelf


class PointBasedSinusoidalNelfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 10
        cls.mocked_lego_batch = (
            torch.rand(cls.batch_size, 3),
            torch.rand(cls.batch_size, 3),
            torch.rand(cls.batch_size, 4),
        )

    def test_forward(self):
        model = PointBasedSinusoidalNelf(
            initial_point=torch.rand(128, 3).numpy(),
            hidden_dims=[32, 32, 32],
            point_embedding_size=8,
            sinusoidal_embedding_size=16,
        )

        self.assertTrue(model(self.mocked_lego_batch).shape, (128,))
