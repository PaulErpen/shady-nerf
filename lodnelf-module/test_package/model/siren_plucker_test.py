import unittest
from lodnelf.model.siren_plucker import SirenPlucker
from lodnelf.data.hdf5dataset import get_instance_datasets_hdf5
from lodnelf.util import util


class SirenPluckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=[0],
            sidelen=128,
            max_observations_per_instance=1,
        )

    def test_given_valid_parameters__when_instantiating_the_model__then_no_error_is_raised(
        self,
    ):
        model = SirenPlucker(hidden_dims=[256])

        self.assertIsNotNone(model)

    def test_given_a_valid_observation__when_forwarding_through_the_model__then_output_is_correct(
        self,
    ):
        model = SirenPlucker(hidden_dims=[256])

        query = self.dataset[0][0]
        model_input = util.add_batch_dim_to_dict(query)
        output = model(model_input)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (1, 128 * 128, 3))
