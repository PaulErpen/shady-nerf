import unittest
from lodnelf.data.lego_dataset import LegoDataset
from matplotlib import pyplot as plt
import torch


class LegoDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = LegoDataset(data_root="data/lego", split="train", limit=10)

    def test_given_a_valid_data_root__when_instantiating__then_the_meta_data_should_be_loaded(
        self,
    ):
        self.assertEqual(len(self.dataset), 10 * 800 * 800)

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_it_return_a_tuple(
        self,
    ):
        sample = self.dataset[0]
        self.assertIsInstance(sample, tuple)

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_the_ray_origin_must_have_the_correct_shape(
        self,
    ):
        ray_origin, ray_direction, color = self.dataset[0]
        self.assertEqual(ray_origin.shape, (3,))

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_the_ray_direction_must_have_the_correct_shape(
        self,
    ):
        ray_origin, ray_direction, color = self.dataset[0]
        self.assertEqual(ray_direction.shape, (3,))

    def test_given_a_valid_data_root__when_loading_a_single_observation__then_the_ray_color_must_have_the_correct_shape(
        self,
    ):
        ray_origin, ray_direction, color = self.dataset[0]
        self.assertEqual(color.shape, (4,))

    def test_given_a_full_image_batch__when_loading__then_the_batch_should_result_in_a_valid_image(
        self,
    ):
        batch = [self.dataset[i] for i in range(800 * 800)]
        batch = torch.stack(
            [color for ray_origin, ray_direction, color in batch], dim=0
        )

        plt.imshow(batch.view(800, 800, 4))
        plt.show()

    def test_given_a_full_image_batch__when_loading__then_all_ray_directions_should_be_length_one(
        self,
    ):
        batch = [self.dataset[i] for i in range(800 * 800)]
        batch = torch.stack(
            [ray_direction for ray_origin, ray_direction, color in batch], dim=0
        )

        self.assertTrue(
            torch.allclose(torch.norm(batch, dim=-1), torch.ones_like(batch[:, 0]))
        )

    def test_given_a_full_image_batch__when_loading__then_all_ray_directions_should_result_in_a_sensible_picture(
        self,
    ):
        batch = [self.dataset[i] for i in range(800 * 800)]
        batch = torch.stack(
            [ray_direction for ray_origin, ray_direction, color in batch], dim=0
        )

        plt.imshow(batch.view(800, 800, 3) + 1 / 2)
        plt.show()

    def test_given_a_full_image_batch__when_loading__then_all_ray_directions_should_result_in_a_sensible_picture_by_cosine_with_cam_forward(
        self,
    ):
        batch = [self.dataset[i] for i in range(800 * 800)]
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
