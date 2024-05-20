import unittest
from lodnelf.test.interactive_display import InteractiveDisplay
import numpy as np
from lodnelf.geometry.rotation_matrix import rotation_matrix


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
        cam2world_matrix = np.eye(4)  # Initial camera to world matrix (identity matrix)

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
        cam2world_matrix = np.eye(4)  # Initial camera to world matrix (identity matrix)
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
