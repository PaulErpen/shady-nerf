from typing import Any, Callable
import torch.nn as nn
from torch import optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from lodnelf.util import util
from lodnelf.train.loss import _LossFn


class TrainExecutor:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss: _LossFn,
        batch_size: int,
        device: str,
        train_data: Dataset,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.batch_size = batch_size
        self.train_data = train_data

    def train(self):
        self.model.train()

        train_data_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

        total_loss = 0

        for batch in tqdm(train_data_loader):
            model_input = batch
            model_input = util.to_device(model_input, self.device)
            batch = util.to_device(batch, self.device)

            self.optimizer.zero_grad()

            output = self.model(model_input)
            loss = self.loss(output, batch)
            loss.backward()
            total_loss += loss.item()

            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / len(train_data_loader)
