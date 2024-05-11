from lodnelf.data.hdf5dataset import get_instance_datasets_hdf5
from lodnelf.model.simple_light_field_model import SimpleLightFieldModel
from lodnelf.train.train_executor import TrainExecutor
import torch.optim
from lodnelf.train.loss import LFLoss
from lodnelf.train.train_handler import TrainHandler
from pathlib import Path
from lodnelf.util import util

dataset = get_instance_datasets_hdf5(
    root="data/hdf5/cars_train.hdf5",
    max_num_instances=1,
    specific_observation_idcs=None,
    sidelen=128,
    max_observations_per_instance=None,
)[0]

simple_model = SimpleLightFieldModel(latent_dim=256, depth=False, alpha=False)
executor = TrainExecutor(
    model=simple_model,
    optimizer=torch.optim.Adam(simple_model.parameters()),
    loss=LFLoss(),
    batch_size=5,
)
model_save_path = Path("models/experiment_2")
model_save_path.mkdir(exist_ok=True)
train_handler = TrainHandler(
    max_epochs=150,
    dataset=dataset,
    train_executor=executor,
    prepare_input_fn=lambda x: util.assemble_model_input(x, x),
    stop_after_no_improvement=150,
    model_save_path=model_save_path,
)
train_handler.run()
