import unittest
from lodnelf.test.interactive_display import InteractiveDisplay
import numpy as np
from lodnelf.geometry.rotation_matrix import rotation_matrix
import torch


class InteractiveDisplayTest(unittest.TestCase):
    def test_given_a_valid_config_and_model_path__when_initializing_the_interactive_display__then_nothing_is_raised(
        self,
    ):
        InteractiveDisplay("SimpleRedCarModel", "models/experiment_5/model_epoch_53.pt")

    def test_given_a_valid_config_and_model_path__when_generating_a_new_image__then_nothing_is_raised(
        self,
    ):
        interactive_display = InteractiveDisplay(
            "SimpleRedCarModel", "models/experiment_5/model_epoch_53.pt"
        )
        cam2world_matrix = np.eye(4)  # Initial camera to world matrix (identity matrix)

        interactive_display.update_image(cam2world_matrix)

    def test_given_a_valid_config_and_model_path__when_generating_a_new_image__then_display_the_image(
        self,
    ):
        interactive_display = InteractiveDisplay(
            "SimpleRedCarModel", "models/experiment_5/model_epoch_53.pt"
        )
        cam2world_matrix = np.array(
            [
                [9.8176e-01, 1.2364e-01, -1.4440e-01, 1.8772e-01],
                [1.9010e-01, -6.3854e-01, 7.4574e-01, -9.6946e-01],
                [-2.2352e-08, -7.5959e-01, -6.5040e-01, 8.4552e-01],
                [-0.0000e00, 0.0000e00, -0.0000e00, 1.0000e00],
            ]
        )

        interactive_display.update_image(cam2world_matrix).show()

    def test_given_a_valid_config_and_model_path__when_generating_a_new_image_based_on_a_rotation__then_nothing_is_raised(
        self,
    ):
        interactive_display = InteractiveDisplay(
            "SimpleRedCarModel", "models/experiment_5/model_epoch_53.pt"
        )
        cam2world_matrix = np.eye(4)  # Initial camera to world matrix (identity matrix)
        rotated = np.matmul(
            rotation_matrix([0, 1, 0], np.radians(10)), cam2world_matrix[:3, :3]
        )
        cam2world_matrix[:3, :3] = rotated

        interactive_display.update_image(cam2world_matrix)

    def test_given_a_valid_config_and_model_path__when_generating_a_new_image_based_on_a_rotation__then_display_the_image(
        self,
    ):
        interactive_display = InteractiveDisplay(
            "SimpleRedCarModel", "models/experiment_5/model_epoch_53.pt"
        )
        cam2world_matrix = torch.tensor(
            [
                [9.8176e-01, 1.2364e-01, -1.4440e-01, 1.8772e-01],
                [1.9010e-01, -6.3854e-01, 7.4574e-01, -9.6946e-01],
                [-2.2352e-08, -7.5959e-01, -6.5040e-01, 8.4552e-01],
                [-0.0000e00, 0.0000e00, -0.0000e00, 1.0000e00],
            ]
        )
        rotated = np.matmul(
            rotation_matrix([0, 1, 0], np.radians(10)), cam2world_matrix[:3, :3]
        )
        cam2world_matrix[:3, :3] = rotated

        interactive_display.update_image(cam2world_matrix).show()

    def test_given_a_valid_config_and_model_path__when_running__then_run_the_application(
        self,
    ):
        interactive_display = InteractiveDisplay(
            "SimpleRedCarModel", "models/experiment_5/model_epoch_53.pt"
        )

        interactive_display.run()

    def test_given_a_valid_siren_config_and_model_path__when_running__then_run_the_application(
        self,
    ):
        interactive_display = InteractiveDisplay(
            "SimpleRedCarModelSiren", "models/experiment_siren/model_epoch_110.pt"
        )

        interactive_display.run()
