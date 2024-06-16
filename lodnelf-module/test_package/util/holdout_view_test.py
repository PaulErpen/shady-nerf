import unittest
from lodnelf.viz.holdout_view import print_holdout_view
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

    def test_given_a_mocked_model__when_printing_holdout_view__then_display_the_holdout_view(
        self,
    ):
        # given
        model = self.MockedModel()
        # when
        print_holdout_view(model)
        # then
        self.assertTrue(True)
