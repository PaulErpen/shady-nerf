import argparse
from lodnelf.train.config.config_factory import ConfigFactory
from pathlib import Path
import torch


def run_training(config_name, run_name, model_save_dir, data_dir):
    config_factory = ConfigFactory()
    config = config_factory.get_by_name(config_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Using CUDA.")
    else:
        print("Using CPU.")

    config.run(run_name, Path(model_save_dir), data_dir, device=device)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Training")

    args.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the config to use for training.",
    )

    args.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run.",
    )

    args.add_argument(
        "--model_save_dir",
        type=str,
        required=True,
        help="Path to save the model.",
    )

    args.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory.",
    )

    args = args.parse_args()

    run_training(args.config, args.run_name, args.model_save_dir, args.data_dir)
