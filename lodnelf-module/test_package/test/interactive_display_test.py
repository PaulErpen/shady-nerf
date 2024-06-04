import unittest
from lodnelf.test.interactive_display import InteractiveDisplay


class InteractiveDisplayTest(unittest.TestCase):

    def test_given_a_valid_config_and_model_path__when_initializing_the_interactive_display__then_nothing_is_raised(
        self,
    ):
        InteractiveDisplay(
            "DeepPluckerLegoThree",
            "models/lego_deep_plucker_ray_based_1/model_epoch_5.pt",
        )

    def test_given_a_valid_deep_plucker_config_with_train_val_split_and_model_path__when_running__then_run_the_application(
        self,
    ):
        interactive_display = InteractiveDisplay(
            "DeepPluckerLegoThree",
            "models/lego_deep_plucker_ray_based_1/model_epoch_5.pt",
            mode="rgba",
        )

        interactive_display.run()

    def test_given_a_valid_planar_fourier_3_config_with_train_val_split_and_model_path__when_running__then_run_the_application(
        self,
    ):
        interactive_display = InteractiveDisplay(
            "PlanarFourierLegoThree",
            "models/lego_planar_fourier_three_2/model_epoch_9.pt",
            mode="rgba",
        )

        interactive_display.run()

    def test_given_a_valid_full_fourier_config_with_train_val_split_and_model_path__when_running__then_run_the_application(
        self,
    ):
        interactive_display = InteractiveDisplay(
            "FullFourierThree",
            "models/lego_full_fourier_1/model_epoch_5.pt",
            mode="rgba",
        )

        interactive_display.run()

    def test_given_a_valid_subsampled_plucker_config_with_train_val_split_and_model_path__when_running__then_run_the_application(
        self,
    ):
        interactive_display = InteractiveDisplay(
            "DeepNeuralNetworkPluckerLegoLarge",
            "models/lego_deep_plucker_3_subsampled_2/model_epoch_33.pt",
            mode="rgba",
        )

        interactive_display.run()
