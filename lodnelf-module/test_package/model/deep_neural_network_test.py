import unittest
import torch
from lodnelf.model.deep_neural_network import (
    DeepNeuralNetwork,
    DeepNeuralNetworkPlucker,
)
from lodnelf.data.hdf5dataset import get_instance_datasets_hdf5
from lodnelf.util import util


class DeepNeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        self.dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=[0],
            sidelen=128,
            max_observations_per_instance=1,
        )

    def test_given_valid_parameters__when_instantiating_the_model__then_no_error_is_raised(
        self,
    ):
        model = DeepNeuralNetwork(
            input_dim=10,
            hidden_dims=[20, 30],
            output_dim=1,
        )

    def test_given_a_valid_batch_of_queries__when_forwarding__then_the_output_has_the_correct_shape(
        self,
    ):
        model = DeepNeuralNetwork(
            input_dim=128,
            hidden_dims=[20, 30],
            output_dim=6,
        )

        x = torch.randn(4, 128 * 128, 128)

        output = model(x)

        self.assertEqual(output.shape, (4, 128 * 128, 6))

    def test_given_a_valid_batch_of_queries_and_a_plucker_deep_network__when_forwarding__then_the_output_has_the_correct_shape(
        self,
    ):
        model = DeepNeuralNetworkPlucker(
            hidden_dims=[20, 30],
            output_dim=6,
        )

        model_input = util.add_batch_dim_to_dict(self.dataset[0][0])

        output = model(model_input)

        self.assertEqual(output["rgb"].shape, (1, 128 * 128, 6))
