from lodnelf.model.components.deep_neural_network import DeepNeuralNetwork
import unittest
import torch


class DeeptNeuralNetworkTest(unittest.TestCase):
    def test_given_valid_parameters__when_instantiating_the_model__then_no_error_is_raised(
        self,
    ):
        DeepNeuralNetwork(
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

    def test_given_a_skip_connection__when_forwarding__then_the_output_has_the_correct_shape(
        self,
    ):
        model = DeepNeuralNetwork(
            input_dim=128,
            hidden_dims=[20, 30],
            output_dim=6,
            skips=[1],
        )

        x = torch.randn(4, 128 * 128, 128)

        output = model(x)

        self.assertEqual(output.shape, (4, 128 * 128, 6))
