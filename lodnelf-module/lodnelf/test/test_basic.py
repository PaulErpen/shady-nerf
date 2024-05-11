from lodnelf.model.simple_light_field_model import SimpleLightFieldModel
import torch
from lodnelf.data.hdf5dataset import get_instance_datasets_hdf5
from lodnelf.util import util
from matplotlib import pyplot as plt

dataset = get_instance_datasets_hdf5(
    root="data/hdf5/cars_train.hdf5",
    max_num_instances=1,
    specific_observation_idcs=None,
    sidelen=128,
    max_observations_per_instance=None,
)[0]

simple_model = SimpleLightFieldModel(latent_dim=256, depth=False, alpha=False)
state_dict = torch.load("models/experiment_2/model_epoch_41.pt", map_location="cpu")
simple_model.load_state_dict(state_dict)

fig, axs = plt.subplots(1, 10, figsize=(18, 3))
for i in range(10):
    query = dataset[i]
    model_input = util.add_batch_dim_to_dict(util.assemble_model_input(query, query))
    output = simple_model(model_input)

    rgb = output["rgb"]

    axs[i].imshow(rgb.reshape(128, 128, 3).detach().cpu().numpy())
plt.show()