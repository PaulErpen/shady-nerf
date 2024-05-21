from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from lodnelf.train.loss import _LossFn
import torch
from tqdm import tqdm
from lodnelf.util import util


class ValidationExecutor:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        val_data: Dataset,
        loss: _LossFn,
        batch_size: int,
    ) -> None:
        self.model = model
        self.device = device
        self.device = device
        self.val_data = val_data
        self.loss = loss
        self.batch_size = batch_size

    def validate(self):
        self.model.eval()

        val_data_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )

        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_data_loader):
                model_input = batch
                model_input = util.to_device(model_input, self.device)
                batch = util.to_device(batch, self.device)

                output = self.model(model_input)
                loss = self.loss(output, batch)
                total_loss += loss.item()

        return total_loss / len(val_data_loader)
