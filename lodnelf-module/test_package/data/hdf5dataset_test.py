import unittest
from lodnelf.data.hdf5dataset import Hdf5Dataset, get_instance_datasets_hdf5
from matplotlib import pyplot as plt


class Hdf5DatasetTest(unittest.TestCase):
    def test_given_an_hdf5_dataset__when_loading_a_single_observation__then_the_return_value_must_be_of_length_1(
        self,
    ):
        single_dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=[0],
            sidelen=128,
            max_observations_per_instance=1,
        )
        self.assertEqual(1, len(single_dataset))

    def test_given_an_hdf5_dataset__when_loading_a_single_observation__then_the_individual_object_dataset_must_have_the_correct_length(
        self,
    ):
        single_dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=[0, 1, 2, 3],
            sidelen=128,
            max_observations_per_instance=1,
        )
        self.assertEqual(4, len(single_dataset[0]))
    
    def test_given_an_hdf5_dataset__when_loading_all_observations__then_the_individual_object_dataset_must_have_the_correct_length(
        self,
    ):
        single_dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=None,
            sidelen=128,
            max_observations_per_instance=None,
        )
        self.assertEqual(50, len(single_dataset[0]))

    def test_given_an_hdf5_dataset__when_loading_a_single_observation__then_the_observation_must_have_the_correct_entries(
        self,
    ):
        single_dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=[0],
            sidelen=128,
            max_observations_per_instance=1,
        )
        sample = single_dataset[0][0]
        self.assertCountEqual(
            ["instance_idx", "rgb", "cam2world", "uv", "intrinsics", "instance_name"],
            sample.keys(),
        )

    def test_given_an_hdf5_dataset__when_loading_a_single_observations_rgb__then_it_should_conform_to_the_expected_shape(self):
        single_dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=[0],
            sidelen=128,
            max_observations_per_instance=1,
        )
        sample = single_dataset[0][0]
        self.assertEqual((128 * 128, 3), sample["rgb"].shape)
    
    def test_given_an_hdf5_dataset__when_loading_8_observations_rgb__then_display_all_8_rgb_images(
        self,
    ):
        single_dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=None,
            sidelen=128,
            max_observations_per_instance=8,
        )
        fig, axs = plt.subplots(2, 4, figsize=(8, 4))
        for i in range(2):
            for j in range(4):
                    sample = single_dataset[0][i * 4 + j]
                    axs[i][j].imshow(sample["rgb"].reshape(128, 128, 3))
                    # hide axis ticks and other elements
                    axs[i][j].axis("off")
        plt.show()

    def test_given_an_hdf5_dataset__when_loading_a_single_observations_uv__then_it_should_conform_to_the_expected_shape(self):
        single_dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=[0],
            sidelen=128,
            max_observations_per_instance=1,
        )
        sample = single_dataset[0][0]
        self.assertEqual((128 * 128, 2), sample["uv"].shape)
    
    def test_given_an_hdf5_dataset__when_loading_a_single_observations_uv__then_the_first_dimension_should_be_displayable_as_an_image(self):
        single_dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=[0],
            sidelen=128,
            max_observations_per_instance=1,
        )
        sample = single_dataset[0][0]
        plt.imshow(sample["uv"][:, 0].reshape(128, 128))
        plt.show()

    def test_given_an_hdf5_dataset__when_loading_a_single_observations_intrinsics__then_it_should_conform_to_the_expected_shape(self):
        single_dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=[0],
            sidelen=128,
            max_observations_per_instance=1,
        )
        sample = single_dataset[0][0]
        self.assertEqual((4, 4), sample["intrinsics"].shape)

    def test_given_an_hdf5_dataset__when_loading_a_single_observations_cam2world__then_it_should_conform_to_the_expected_shape(self):
        single_dataset = get_instance_datasets_hdf5(
            root="data/hdf5/cars_train.hdf5",
            max_num_instances=1,
            specific_observation_idcs=[0],
            sidelen=128,
            max_observations_per_instance=1,
        )
        sample = single_dataset[0][0]
        self.assertEqual((4, 4), sample["cam2world"].shape)


if __name__ == "__main__":
    unittest.main()
