import unittest

from lodnelf.data.lego_dataset import LegoDataset
from lodnelf.train.loss import LFLoss
from lodnelf.util import util
from lodnelf.model.deep_neural_network_plucker import DeepNeuralNetworkPlucker


class LossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lego = LegoDataset(data_root="data/lego", split="train")

    def test_given_an_actual_lego_model_output_and_a_target__when_calculating_the_loss__then_raise_no_errors(
        self,
    ):
        batch = util.add_batch_dim_to_dict(self.lego[0])
        model_output = DeepNeuralNetworkPlucker([123], mode="rgba")(batch)
        LFLoss(mode="rgba")(
            model_out=model_output,
            batch=batch,
        )
