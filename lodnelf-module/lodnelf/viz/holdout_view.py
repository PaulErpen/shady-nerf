from pathlib import Path
from lodnelf.train.config.abstract_config import AbstractConfig
from lodnelf.util.generate_model_input import generate_model_input
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt


class HoldoutViewHandler:
    def __init__(
        self,
        H: int,
        W: int,
        focal_length: float,
        cam2world_matrix: torch.Tensor,
        holdout_path_directory: Path,
    ):
        self.H = H
        self.W = W
        self.focal_length = focal_length
        self.cam2world_matrix = cam2world_matrix
        self.holdout_path_directory = holdout_path_directory

        if not self.holdout_path_directory.exists():
            self.holdout_path_directory.mkdir(parents=True)

    def save_holdout_view(self, model: nn.Module, filename: str):
        # create a holdout view
        with torch.no_grad():
            model.eval()

            model_input = generate_model_input(
                self.H,
                self.W,
                self.focal_length,
                cam2world_matrix=self.cam2world_matrix,
                output_size=128,
            )

            # batch model input into smaller chunks
            model_input = [x.chunk(1000, dim=0) for x in model_input]

            model_output = []

            for model_input_chunk in zip(*model_input):
                model_output.extend(model(model_input_chunk))

            rgba = torch.stack(model_output)
            rgba = rgba.numpy()
            rgba = np.clip(rgba, 0, 1)
            plt.imshow(rgba.reshape(128, 128, 4))

            plt.savefig(self.holdout_path_directory / Path(filename))
