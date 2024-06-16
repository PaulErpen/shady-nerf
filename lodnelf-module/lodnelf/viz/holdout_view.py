from pathlib import Path
from lodnelf.train.config.abstract_config import AbstractConfig
from lodnelf.util.generate_model_input import generate_model_input
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_holdout_view(model: nn.Module, config: AbstractConfig, holdout_path: Path):
    # create a holdout view
    with torch.no_grad():
        model.eval()

        H, W = config.get_output_image_size()
        focal_length = config.get_camera_focal_length()
        cam2world = config.get_initial_cam2world_matrix()

        model_input = generate_model_input(
            H,
            W,
            focal_length,
            cam2world_matrix=cam2world,
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

        plt.savefig(holdout_path)
