import unittest

from lodnelf.data.lego_dataset import LegoDataset
from lodnelf.model.my_nerf import NeRF
import torch
from lodnelf.util import util


class MyNerfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.near = 0.0
        cls.far = 1.0
        cls.n_samples_along_ray = 10
        cls.nerf = NeRF(cls.near, cls.far, cls.n_samples_along_ray)
        cls.lego = LegoDataset(data_root="data/lego", split="train", limit=10)

    def test_given_a_single_ray_origin_and_direction__when_computing_the_sample_points__then_return_the_correct_tensor(
        self,
    ):
        ray_origin = torch.tensor([[0.0, 0.0, 0.0]])
        ray_direction = torch.tensor([[0.0, 0.0, 1.0]])

        sample_points = self.nerf.compute_sample_points(ray_origin, ray_direction)

        expected_sample_points = torch.tensor(
            [
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.1111],
                    [0.0000, 0.0000, 0.2222],
                    [0.0000, 0.0000, 0.3333],
                    [0.0000, 0.0000, 0.4444],
                    [0.0000, 0.0000, 0.5556],
                    [0.0000, 0.0000, 0.6667],
                    [0.0000, 0.0000, 0.7778],
                    [0.0000, 0.0000, 0.8889],
                    [0.0000, 0.0000, 1.0000],
                ]
            ]
        )
        self.assertTrue(
            torch.allclose(sample_points, expected_sample_points, atol=1e-4)
        )

    def test_given_two_batches_of_ray_origins_and_directions__when_computing_the_sample_points__then_return_the_correct_tensor(
        self,
    ):
        ray_origins = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        ray_directions = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        sample_points = self.nerf.compute_sample_points(ray_origins, ray_directions)

        expected_sample_points = torch.tensor(
            [
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.1111],
                    [0.0000, 0.0000, 0.2222],
                    [0.0000, 0.0000, 0.3333],
                    [0.0000, 0.0000, 0.4444],
                    [0.0000, 0.0000, 0.5556],
                    [0.0000, 0.0000, 0.6667],
                    [0.0000, 0.0000, 0.7778],
                    [0.0000, 0.0000, 0.8889],
                    [0.0000, 0.0000, 1.0000],
                ],
                [
                    [1.0000, 1.0000, 1.0000],
                    [1.1111, 1.1111, 1.1111],
                    [1.2222, 1.2222, 1.2222],
                    [1.3333, 1.3333, 1.3333],
                    [1.4444, 1.4444, 1.4444],
                    [1.5556, 1.5556, 1.5556],
                    [1.6667, 1.6667, 1.6667],
                    [1.7778, 1.7778, 1.7778],
                    [1.8889, 1.8889, 1.8889],
                    [2.0000, 2.0000, 2.0000],
                ],
            ]
        )

        self.assertTrue(
            torch.allclose(sample_points, expected_sample_points, atol=1e-4)
        )

    def test_given_some_random_data__when_computing_the_directions__then_return_the_correct_shape(
        self,
    ):
        batch_size = 10
        ray_origins = torch.rand((batch_size, 3))
        ray_directions = torch.rand((batch_size, 3))

        sample_points = self.nerf.compute_sample_points(ray_origins, ray_directions)

        self.assertEqual(sample_points.shape, (batch_size, self.n_samples_along_ray, 3))

    def test_given_some_random_alpha_and_rgb_values__when_rendering_the_rays__then_return_the_correct_rgb_map(
        self,
    ):
        batch_size = 10
        point_alpha = torch.rand((batch_size, self.n_samples_along_ray))
        point_rgb = torch.rand((batch_size, self.n_samples_along_ray, 3))

        rgb_map, _, _ = self.nerf.render_rays(point_alpha, point_rgb)

        self.assertEqual(rgb_map.shape, (batch_size, 3))

    def test_given_some_random_alpha_and_rgb_values__when_rendering_the_rays__then_return_the_correct_depth_map(
        self,
    ):
        batch_size = 10
        point_alpha = torch.rand((batch_size, self.n_samples_along_ray))
        point_rgb = torch.rand((batch_size, self.n_samples_along_ray, 3))

        _, depth_map, _ = self.nerf.render_rays(point_alpha, point_rgb)

        self.assertEqual(depth_map.shape, (batch_size,))

    def test_given_some_random_alpha_and_rgb_values__when_rendering_the_rays__then_return_the_correct_acc_map(
        self,
    ):
        batch_size = 10
        point_alpha = torch.rand((batch_size, self.n_samples_along_ray))
        point_rgb = torch.rand((batch_size, self.n_samples_along_ray, 3))

        _, _, acc_map = self.nerf.render_rays(point_alpha, point_rgb)

        self.assertEqual(acc_map.shape, (batch_size,))

    def test_given_a_single_sample__when_forwarding__then_return_the_correct_rgb_map(
        self,
    ):
        sample = self.lego[0]

        rgb_map = self.nerf(util.add_batch_dim_to_dict(sample))

        self.assertEqual(rgb_map.shape, (1, 3))

    def test_given_single_observation__when_using_extended_forward__then_return_a_point_alpha_with_correct_shape(
        self,
    ):
        sample = self.lego[0]

        rgb_map, depth_map, acc_map, point_alpha = self.nerf.extended_forward(
            util.add_batch_dim_to_dict(sample)
        )

        self.assertEqual(point_alpha.shape, (1, self.n_samples_along_ray, 1))
