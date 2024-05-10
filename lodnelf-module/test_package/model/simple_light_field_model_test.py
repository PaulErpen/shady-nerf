import unittest
from lodnelf.model.simple_light_field_model import SimpleLightFieldModel
from lodnelf.data.hdf5dataset import get_instance_datasets_hdf5
from lodnelf.util import util
from matplotlib import pyplot as plt

class SimpleLightFieldModelTest(unittest.TestCase):
    def setUp(self):
        self.dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=[0],
            sidelen=128,
            max_observations_per_instance=1,
        )

    def test_given_valid_parameters__when_instantiating_the_model__then_model_is_created(self):
        model = SimpleLightFieldModel(latent_dim=256, depth=False, alpha=False)

        self.assertIsNotNone(model)

    def test_given_a_valid_observation__when_forwarding_through_the_model__then_output_is_correct(self):
        model = SimpleLightFieldModel(latent_dim=256, depth=False, alpha=False)

        query = self.dataset[0][0]
        model_input = util.assemble_model_input(query, query)
        output = model(model_input)

        self.assertIsNotNone(output)
    
    def test_givan_a_valid_observation__when_forwarding_through_the_model__then_outputs_rgb_can_be_displayed(self):
        model = SimpleLightFieldModel(latent_dim=256, depth=False, alpha=False)

        query = self.dataset[0][0]
        model_input = util.assemble_model_input(query, query)
        output = model(model_input)

        rgb = output['rgb']

        plt.imshow(rgb.reshape(128, 128, 3).detach().cpu().numpy())
        plt.show()