import unittest
import torch.nn as nn
import torch
from lodnelf.train.train_executor import TrainExecutor
import torch.utils
import torch.utils.data
from lodnelf.data.hdf5dataset import get_instance_datasets_hdf5
from lodnelf.model.simple_light_field_model import SimpleLightFieldModel
from lodnelf.util import util
from lodnelf.train.loss import LFLoss, _LossFn


class TrainExecutorTest(unittest.TestCase):
    def setUp(self):
        self.dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=None,
            sidelen=128,
            max_observations_per_instance=9,
        )[0]

    def test_given_a_valid_mocks__when_executing_a_training_run__then_the_output_is_a_float(
        self,
    ):
        executor = TrainExecutor(
            model=self.MockedModel(),
            optimizer=torch.optim.Adam(self.MockedModel().parameters()),
            loss=self.MockedLoss(),
            batch_size=3,
            device="cpu",
        )
        dataset = self.MockedDataset()

        loss = executor.train(dataset, prepare_input_fn=lambda x: x[0])
        self.assertIsInstance(loss, float)

    def test_given_valid_props_for_model_training__when_executing_a_training_run__no_errors_are_raised(
        self,
    ):
        simple_model = SimpleLightFieldModel(latent_dim=256, depth=False, alpha=False)
        executor = TrainExecutor(
            model=simple_model,
            optimizer=torch.optim.Adam(self.MockedModel().parameters()),
            loss=LFLoss(),
            batch_size=3,
            device="cpu",
        )

        executor.train(self.dataset)

        self.assertTrue(True)

    class MockedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    class MockedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 9

        def __getitem__(self, idx):
            return (
                torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                torch.tensor([1.0]),
            )

    class MockedLoss(_LossFn):
        def __init__(self):
            super().__init__()

        def __call__(self, output, ground_truth):
            return nn.MSELoss()(output, ground_truth[1])
