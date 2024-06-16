from lodnelf.util.generate_model_input import generate_model_input
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt


def print_holdout_view(model: nn.Module):
    # create a holdout view
    with torch.no_grad():
        model.eval()

        model_input = generate_model_input(
            800,
            800,
            1111.111,
            torch.tensor(
                [
                    [
                        0.842908501625061,
                        -0.09502744674682617,
                        0.5295989513397217,
                        2.1348819732666016,
                    ],
                    [
                        0.5380570292472839,
                        0.14886793494224548,
                        -0.8296582698822021,
                        -3.3444597721099854,
                    ],
                    [
                        7.450582373280668e-09,
                        0.9842804074287415,
                        0.17661221325397491,
                        0.7119466662406921,
                    ],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ).float(),
            output_size=800,
        )

        # batch model input into smaller chunks
        model_input = [x.chunk(1000, dim=0) for x in model_input]

        model_output = []

        for model_input_chunk in zip(*model_input):
            model_output.extend(model(model_input_chunk))

        rgba = torch.stack(model_output)
        rgba = rgba.numpy()
        rgba = np.clip(rgba, 0, 1)
        plt.imshow(rgba.reshape(800, 800, 4))
        plt.show()
