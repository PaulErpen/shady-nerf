import unittest
from lodnelf.data.lego_dataset import LegoDataset


class LegoDatasetTest(unittest.TestCase):
    def test_given_a_valid_data_root__when_instantiating__then_do_not_throw_any_errors(
        self,
    ):
        LegoDataset(data_root="data/lego", split="train")

    def test_given_a_valid_data_root__when_instantiating__then_the_meta_data_should_be_loaded(
        self,
    ):
        dataset = LegoDataset(data_root="data/lego", split="train")
        self.assertEqual(len(dataset), 100)

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_it_must_return_a_dictionary_with_the_correct_keys(
        self,
    ):
        dataset = LegoDataset(data_root="data/lego", split="train")
        sample = dataset[0]
        self.assertCountEqual(
            ["rgba", "cam2world", "uv", "intrinsics"],
            sample.keys(),
        )

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_the_rgba_image_should_have_the_correct_shape(
        self,
    ):
        dataset = LegoDataset(data_root="data/lego", split="train")
        sample = dataset[0]
        self.assertEqual((800 * 800, 4), sample["rgba"].shape)

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_the_cam2world_matrix_should_have_the_correct_shape(
        self,
    ):
        dataset = LegoDataset(data_root="data/lego", split="train")
        sample = dataset[0]
        self.assertEqual((4, 4), sample["cam2world"].shape)

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_the_uv_coordinates_should_have_the_correct_shape(
        self,
    ):
        dataset = LegoDataset(data_root="data/lego", split="train")
        sample = dataset[0]
        self.assertEqual((800 * 800, 2), sample["uv"].shape)

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_the_intrinsics_matrix_should_have_the_correct_shape(
        self,
    ):
        dataset = LegoDataset(data_root="data/lego", split="train")
        sample = dataset[0]
        self.assertEqual((4, 4), sample["intrinsics"].shape)
