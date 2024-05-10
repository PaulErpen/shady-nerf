from typing import Any, Callable
import torch.nn as nn
from torch import optim
import torch.nn.modules.loss as loss
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from lodnelf.util import util


class TrainExecutor:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss: loss._Loss,
        batch_size: int,
        device="cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.batch_size = batch_size

    def train(
        self,
        train_data: Dataset,
        prepare_input_fn: Callable[[Any], Any] | None = None,
    ):
        self.model.train()

        train_data_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True
        )

        total_loss = 0

        for batch in tqdm(train_data_loader):
            model_input = batch
            if prepare_input_fn is not None:
                model_input = prepare_input_fn(batch)
            ground_truth = batch["rgb"]
            images = util.to_device(model_input, self.device)
            ground_truth = util.to_device(ground_truth, self.device)

            self.optimizer.zero_grad()

            output = self.model(images)
            loss = self.loss(output, ground_truth)
            loss.backward()
            total_loss += loss.item()

            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / len(train_data_loader)
