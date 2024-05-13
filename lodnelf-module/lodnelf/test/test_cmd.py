from lodnelf.model.simple_light_field_model import SimpleLightFieldModel
import torch
from lodnelf.data.hdf5dataset import get_instance_datasets_hdf5
from lodnelf.util import util
from matplotlib import pyplot as plt
from lodnelf.train.config.config_factory import ConfigFactory
import argparse


def run_testing(config_name, model_path, data_dir):
    config_factory = ConfigFactory()
    config = config_factory.get_by_name(config_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(model_path, map_location=device)

    model = config.get_model()
    model.load_state_dict(state_dict)

    dataset = config.get_data_set(data_dir)

    fig, axs = plt.subplots(2, 10, figsize=(18, 3))
    for i in range(10):
        query = dataset[i]
        model_input = util.add_batch_dim_to_dict(
            util.assemble_model_input(query, query)
        )
        output = model(model_input)

        axs[0][i].imshow(output["rgb"].reshape(128, 128, 3).detach().cpu().numpy())
        axs[1][i].imshow(query["rgb"].reshape(128, 128, 3).detach().cpu().numpy())
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

    args = args.parse_args()

    run_testing(args.config, args.model_save_path, args.data_dir)
