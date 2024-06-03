import torch.nn as nn
from torch import optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
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
        subset_size: float | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.batch_size = batch_size
        self.train_data = train_data
        self.subset_size = subset_size

    def train(self):
        self.model.train()

        sampler = None
        if self.subset_size is not None:
            if self.subset_size > 1 or self.subset_size <= 0:
                raise ValueError("subset_size should be between 0 and 1")
            sampler = RandomSampler(
                self.train_data,
                replacement=False,
                num_samples=int(self.subset_size * len(self.train_data)),
            )
        train_data_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, sampler=sampler
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
