import unittest

from lodnelf.data.lego_dataset import LegoDataset
from lodnelf.model.planar_fourier import PlanarFourier
from lodnelf.util import util

class PlanarFourierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lego = LegoDataset(
            data_root="data/lego", split="train", image_size=(128, 128)
        )

    def test_given_valid_parameters__when_instantiating_the_model__then_no_error_is_raised(
        self,
    ):
        model = PlanarFourier(hidden_dims=[256], fourier_mapping_size=64, mode="rgb")

        self.assertIsNotNone(model)

    def test_given_a_valid_observation__when_forwarding_through_the_model__then_output_is_correct(
        self,
    ):
        model = PlanarFourier(hidden_dims=[256], fourier_mapping_size=64, mode="rgba")

        model_input = util.add_batch_dim_to_dict(util.add_batch_dim_to_dict(self.lego[0]))
        output = model(model_input)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (1, 1, 4))
