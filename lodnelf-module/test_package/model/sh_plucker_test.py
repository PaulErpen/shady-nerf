import unittest

from lodnelf.data.lego_dataset import LegoDataset
from lodnelf.model.sh_plucker import ShPlucker
from lodnelf.util import util


class ShPluckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = LegoDataset(data_root="data/lego", split="train", limit=10)

    def test_given_a_sample_from_the_lego_dataset__when_forwarding__then_do_not_raise_an_error(
        self,
    ):
        sample = self.dataset[0]

        model = ShPlucker()

        model(util.add_batch_dim_to_dict(sample))
