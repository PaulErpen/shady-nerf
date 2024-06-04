import unittest
from lodnelf.geometry.compute_ray_space_ray_directions import (
    compute_cam_space_ray_directions,
)
from matplotlib import pyplot as plt
import torch


class ComputeRaySpaceRayDirectionsTest(unittest.TestCase):
    def test_given_a_valid_input_of_800_by_400__when_computing_directions__it_should_return_a_tensor_of_the_correct_shape(
        self,
    ):
        ray_dirs = compute_cam_space_ray_directions(800, 400, 1111.0)

        self.assertEqual(ray_dirs.shape, (400, 800, 3))

    def test_given_a_valid_input_of_800_by_400_and_fraction_of_50_percent__when_computing_the_direction__it_should_return_a_tensor_of_the_correct_shape(
        self,
    ):
        ray_dirs = compute_cam_space_ray_directions(800, 400, 1111.0, 0.5)

        self.assertEqual(ray_dirs.shape, (200, 400, 3))

    def test_given_valid_input__when_computing_ray_directions__then_it_should_result_in_a_valid_image(
        self,
    ):
        ray_dirs = compute_cam_space_ray_directions(800, 800, 1111.0)

        plt.imshow(ray_dirs.view(800, 800, 3) + 1 / 2)
        plt.show()

    def test_given_valid_input__when_computing_ray_directions__then_all_ray_directions_should_result_in_a_sensible_picture_by_cosine_with_cam_forward(
        self,
    ):
        ray_dirs = compute_cam_space_ray_directions(800, 800, 1111.0)

        plt.imshow(
            torch.einsum(
                "hwi,i->hw",
                ray_dirs.view(800, 800, 3),
                torch.tensor([0, 0, -1]).float(),
            ).view(800, 800)
            + 1 / 2
        )
        plt.show()
