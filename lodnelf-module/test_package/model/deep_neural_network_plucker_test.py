import unittest
from lodnelf.data.lego_dataset import LegoDataset
from lodnelf.model.deep_neural_network_plucker import (
    DeepNeuralNetworkPlucker,
)
from lodnelf.data.hdf5dataset import get_instance_datasets_hdf5
from lodnelf.util import util


class DeepNeuralNetworkPluckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=[0],
            sidelen=128,
            max_observations_per_instance=1,
        )
        cls.lego = LegoDataset(data_root="data/lego", split="train")

    def test_given_a_valid_batch_of_queries_and_a_plucker_deep_network__when_forwarding__then_the_output_has_the_correct_shape(
        self,
    ):
        model = DeepNeuralNetworkPlucker(
            hidden_dims=[20, 30],
            mode="rgb",
        )

        model_input = util.add_batch_dim_to_dict(self.dataset[0][0])

        output = model(model_input)

        self.assertEqual(output.shape, (1, 128 * 128, 3))

    def test_given_a_valid_batch_of_lego_queries_and_a_plucker_deep_network__when_forwarding__then_the_output_has_the_correct_shape(
        self,
    ):
        model = DeepNeuralNetworkPlucker(
            hidden_dims=[20, 30],
            mode="rgba",
        )

        model_input = util.add_batch_dim_to_dict(self.lego[0])

        output = model(model_input)

        self.assertEqual(output.shape, (1, 800 * 800, 4))
