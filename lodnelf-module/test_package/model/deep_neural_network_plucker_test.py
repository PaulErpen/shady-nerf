import unittest
from lodnelf.data.lego_dataset import LegoDataset
from lodnelf.model.deep_neural_network_plucker import (
    DeepNeuralNetworkPlucker,
)


class DeepNeuralNetworkPluckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lego = LegoDataset(data_root="data/lego", split="train")

    def test_given_a_valid_batch_of_lego_queries_and_a_plucker_deep_network__when_forwarding__then_the_output_has_the_correct_shape(
        self,
    ):
        model = DeepNeuralNetworkPlucker(
            hidden_dims=[20, 30],
            mode="rgba",
        )

        model_input = self.lego[0]

        output = model(model_input)

        self.assertEqual(output.shape, (4,))
