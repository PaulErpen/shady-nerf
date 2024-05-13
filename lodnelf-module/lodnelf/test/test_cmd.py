import torch
from lodnelf.util import util
from matplotlib import pyplot as plt
from lodnelf.train.config.config_factory import ConfigFactory
import argparse


def run_testing(config_name, model_path, data_dir, plot_alpha=False, plot_depth=False):
    config_factory = ConfigFactory()
    config = config_factory.get_by_name(config_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(model_path, map_location=device)

    model = config.get_model()
    model.load_state_dict(state_dict)

    dataset = config.get_data_set(data_dir)

    n_rows = 2

    if plot_alpha:
        n_rows = n_rows + 1

    if plot_depth:
        n_rows = n_rows + 1

    fig, axs = plt.subplots(n_rows, 10, figsize=(18, n_rows * 1.7))
    fig.tight_layout()
    axs[0][0].set_ylabel("Actual")
    axs[1][0].set_ylabel("Predicted RGB")

    if plot_alpha:
        axs[2][0].set_ylabel("Predicted Alpha")

    if plot_depth:
        axs[n_rows - 1][0].set_ylabel("Predicted Depth")

    for i in range(10):
        query = dataset[i]
        model_input = util.add_batch_dim_to_dict(
            util.assemble_model_input(query, query)
        )
        output = model(model_input)

        axs[0][i].imshow(query["rgb"].reshape(128, 128, 3).detach().cpu().numpy())
        axs[1][i].imshow(output["rgb"].reshape(128, 128, 3).detach().cpu().numpy())

        if "alpha" in output and plot_alpha:
            axs[2][i].imshow(
                output["alpha"].reshape(128, 128, 1).detach().cpu().numpy(),
                cmap="gray",
            )

        if "depth" in output and plot_depth:
            max_depth = output["depth"].max()
            axs[n_rows - 1][i].imshow(
                1 - (output["depth"] / max_depth).reshape(128, 128, 1).detach().cpu().numpy(),
                cmap="gray",
            )

    plt.show()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Training")

    args.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the config to use for training.",
    )

    args.add_argument(
        "--model_save_path",
        type=str,
        required=True,
        help="Path where the trained model is stored.",
    )

    args.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory.",
    )

    args.add_argument(
        "--plot_alpha",
        required=False,
        default=False,
        action="store_true",
        help="Whether or not to plot alpha channel",
    )

    args.add_argument(
        "--plot_depth",
        required=False,
        default=False,
        action="store_true",
        help="Whether or not to plot depth",
    )

    args = args.parse_args()

    run_testing(
        args.config,
        args.model_save_path,
        args.data_dir,
        args.plot_alpha,
        args.plot_depth,
    )
