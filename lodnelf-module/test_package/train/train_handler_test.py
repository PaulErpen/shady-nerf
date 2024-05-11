import unittest
from lodnelf.train.train_handler import TrainHandler
from lodnelf.train.train_executor import TrainExecutor
import torch.nn as nn
from torch.optim import AdamW
import torch
import torch.utils.data
from lodnelf.train.loss import _LossFn
from pathlib import Path


class TrainHandlerTest(unittest.TestCase):
    def test_given_a_valid_mocks__when_initializing__then_do_not_throw_errors(
        self,
    ):
        train_handler = TrainHandler(
            max_epochs=10,
            dataset=self.MockedDataset(),
            train_executor=self.MockedTrainExecutor(
                self.MockedModel(), self.MockedLoss()
            ),
            train_config={"config": "config"},
            group_name="unittest",
        )

    def test_given_a_model_with_improving_loss__when_running__then_save_the_model_to_the_specified_path(
        self,
    ):
        model_save_path = Path("model_save_path")
        model_save_path.mkdir(exist_ok=True)
        train_handler = TrainHandler(
            max_epochs=5,
            dataset=self.MockedDataset(),
            train_executor=self.MockedTrainExecutor(
                self.MockedModel(), self.MockedLoss(), is_improving=True
            ),
            model_save_path=model_save_path,
            train_config={"config": "config"},
            group_name="unittest",
        )
        train_handler.run("unittest_run")
        self.assertTrue((model_save_path / "model_epoch_0.pt").exists())
        self.assertTrue((model_save_path / "model_epoch_1.pt").exists())
        self.assertTrue((model_save_path / "model_epoch_2.pt").exists())
        self.assertTrue((model_save_path / "model_epoch_3.pt").exists())
        self.assertTrue((model_save_path / "model_epoch_4.pt").exists())

        # cleanup
        for i in range(5):
            (model_save_path / f"model_epoch_{i}.pt").unlink()
        model_save_path.rmdir()

    def test_given_a_model_with_no_improving_loss__when_running__then_only_save_the_first_model(
        self,
    ):
        model_save_path = Path("model_save_path")
        model_save_path.mkdir(exist_ok=True)
        train_handler = TrainHandler(
            max_epochs=5,
            dataset=self.MockedDataset(),
            train_executor=self.MockedTrainExecutor(
                self.MockedModel(), self.MockedLoss(), is_improving=False
            ),
            model_save_path=model_save_path,
            train_config={"config": "config"},
            group_name="unittest",
        )
        train_handler.run("unittest_run")
        self.assertTrue((model_save_path / "model_epoch_0.pt").exists())
        self.assertFalse((model_save_path / "model_epoch_1.pt").exists())
        self.assertFalse((model_save_path / "model_epoch_2.pt").exists())
        self.assertFalse((model_save_path / "model_epoch_3.pt").exists())
        self.assertFalse((model_save_path / "model_epoch_4.pt").exists())

        # cleanup
        (model_save_path / "model_epoch_0.pt").unlink()
        model_save_path.rmdir()

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

    class MockedTrainExecutor(TrainExecutor):
        def __init__(self, model: nn.Module, loss: _LossFn, is_improving=False):
            super().__init__(
                model=model,
                optimizer=AdamW(model.parameters(), lr=0.001),
                loss=loss,
                batch_size=3,
            )
            self.loss = 10.0
            self.is_improving = is_improving

        def train(self, dataset, prepare_input_fn):
            if self.is_improving:
                self.loss -= 1
            return self.loss
