import unittest
from lodnelf.data.lego_dataset import LegoDataset
from matplotlib import pyplot as plt
import torch


class LegoDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_dataset = LegoDataset(data_root="data/lego", split="train", limit=10)
        cls.test_dataset = LegoDataset(data_root="data/lego", split="test", limit=10)
        cls.val_dataset = LegoDataset(data_root="data/lego", split="val", limit=10)

    def test_given_a_valid_data_root__when_instantiating__then_the_meta_data_should_be_loaded(
        self,
    ):
        self.assertEqual(len(self.train_dataset), 10 * 800 * 800)

    def test_given_a_valid_setup__when_instantiating__the_ray_origins_must_have_the_right_dimensions(
        self,
    ):
        ray_origins = self.train_dataset.ray_origins
        self.assertEqual(ray_origins.shape, (10, 3))

    def test_given_a_valid_setup__when_instantiating__the_images_must_have_the_right_dimensions(
        self,
    ):
        images = self.train_dataset.images
        self.assertEqual(images.shape, (10, 800, 800, 4))

    def test_given_a_valid_setup__when_instantiating__the_ray_directions_must_have_the_right_dimensions(
        self,
    ):
        ray_directions = self.train_dataset.ray_directions
        self.assertEqual(ray_directions.shape, (10, 800, 800, 3))

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_it_return_a_tuple(
        self,
    ):
        sample = self.train_dataset[0]
        self.assertIsInstance(sample, tuple)

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_the_ray_origin_must_have_the_correct_shape(
        self,
    ):
        ray_origin, ray_direction, color = self.train_dataset[0]
        self.assertEqual(ray_origin.shape, (3,))

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_the_ray_direction_must_have_the_correct_shape(
        self,
    ):
        ray_origin, ray_direction, color = self.train_dataset[0]
        self.assertEqual(ray_direction.shape, (3,))

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_the_ray_color_must_have_the_correct_shape(
        self,
    ):
        ray_origin, ray_direction, color = self.train_dataset[0]
        self.assertEqual(color.shape, (4,))

    def test_given_a_full_image_batch__when_loading__then_the_batch_should_result_in_a_valid_image(
        self,
    ):
        batch = [self.train_dataset[i] for i in range(800 * 800)]
        batch = torch.stack(
            [color for ray_origin, ray_direction, color in batch], dim=0
        )

        plt.imshow(batch.view(800, 800, 4))
        plt.show()

    def test_given_all_images_in_the_dataset__when_loading__then_all_ray_directions_should_be_smaller_or_equal_1_2(
        self,
    ):
        for i in range(0, len(self.train_dataset), 800 * 800):
            ray_origin, ray_direction, color = self.train_dataset[i]
            self.assertTrue(torch.norm(ray_direction).item() <= (1.0 + 1e-5))

    def test_given_all_images_in_the_dataset_from_the_test_split__when_loading__then_all_ray_directions_should_be_smaller_or_equal_1(
        self,
    ):
        for i in range(len(self.test_dataset), 800 * 800):
            ray_origin, ray_direction, color = self.train_dataset[i]
            self.assertTrue(torch.norm(ray_direction).item() <= (1.0 + 1e-5))

    def test_given_all_images_in_the_dataset_from_the_val_split__when_loading__then_all_ray_directions_should_be_smaller_or_equal_1(
        self,
    ):
        for i in range(len(self.val_dataset), 800 * 800):
            ray_origin, ray_direction, color = self.val_dataset[i]
            self.assertTrue(torch.norm(ray_direction).item() <= (1.0 + 1e-5))

    def test_given_all_images_in_the_dataset__when_loading__then_all_ray_origins_should_be_length_4_0311(
        self,
    ):
        for i in range(len(self.val_dataset), 800 * 800):
            ray_origin, ray_direction, color = self.val_dataset[i]
            self.assertAlmostEquals(torch.norm(ray_direction).item(), 4.0311, delta=1e-4)

    def test_given_all_images_in_the_dataset_from_the_test_split__when_loading__then_all_ray_origins_should_be_length_4_0311(
        self,
    ):
        for i in range(len(self.test_dataset), 800 * 800):
            ray_origin, ray_direction, color = self.test_dataset[i]
            self.assertAlmostEquals(torch.norm(ray_direction).item(), 4.0311, delta=1e-4)

    def test_given_all_images_in_the_dataset_from_the_val_split__when_loading__then_all_ray_origins_should_be_length_4_0311(
        self,
    ):
        for i in range(len(self.val_dataset), 800 * 800):
            ray_origin, ray_direction, color = self.val_dataset[i]
            self.assertAlmostEquals(torch.norm(ray_direction).item(), 4.0311, delta=1e-4)

    def test_given_a_data_set_that_employs_a_normalizing_transform__when_loading_all_images__all_ray_origins_should_be_length_1(
        self,
    ):
        transform_dataset = LegoDataset(
            data_root="data/lego",
            split="train",
            limit=10,
            transform=lambda x: (x[0] / 4.0311, x[1], x[2]),
        )
        for i in range(len(transform_dataset), 800 * 800):
            ray_origin, ray_direction, color = transform_dataset[i]
            self.assertAlmostEquals(torch.norm(ray_origin).item(), 1.0, delta=1e-4)

    def test_given_a_full_image_batch__when_loading__then_all_ray_directions_should_result_in_a_sensible_picture(
        self,
    ):
        batch = [self.train_dataset[i] for i in range(800 * 800)]
        batch = torch.stack(
            [ray_direction for ray_origin, ray_direction, color in batch], dim=0
        )

        plt.imshow(batch.view(800, 800, 3) + 1 / 2)
        plt.show()

    def test_given_a_full_image_batch__when_loading__then_all_ray_directions_should_result_in_a_sensible_picture_by_cosine_with_cam_forward(
        self,
    ):
        batch = [self.train_dataset[i] for i in range(800 * 800)]
        batch = torch.stack(
            [ray_direction for ray_origin, ray_direction, color in batch], dim=0
        )

        plt.imshow(
            torch.einsum(
                "hwi,i->hw",
                batch.view(800, 800, 3),
                torch.tensor(
                    [
                        -4.656612873077393e-10,
                        0.9540371894836426,
                        0.29968830943107605,
                    ]
                ).float(),
            ).view(800, 800)
            + 1 / 2
        )
        plt.show()
