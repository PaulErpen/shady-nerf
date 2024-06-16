import unittest
import numpy as np
from lodnelf.viz.viz3d import viz_points, viz_data_point_dir


class Viz3dTest(unittest.TestCase):
    def test_given_a_numpy_array_of_points__when_visualizing__then_display_the_points(
        self,
    ):
        points = np.linspace(0, 1, 300).reshape(-1, 3) * np.array([-1, 1, 1])

        viz_points(points)

    def test_given_a_numpy_array_of_points_and_colors__when_visualizing__then_display_the_points(
        self,
    ):
        points = np.linspace(0, 1, 300).reshape(-1, 3) * np.array([-1, 1, 1])
        other_color = np.zeros_like(points[:, 0])

        viz_points(points, other_color=other_color)

    def test_given_a_numpy_array_of_points__when_visualizing_based_on_a_camera_ray_depth__then_display_the_points(
        self,
    ):
        # sample 16000 points in a 3D space
        points = np.random.rand(16000, 3) * 10
        ray_origin = np.array([10, 10, 10])
        ray_dir_world = np.array([-1, -1, -1])

        viz_data_point_dir(
            points,
            ray_origin,
            ray_dir_world,
            std_dev_perpendicular=np.Inf,
            std_dev_distance=5.0,
        )

    def test_given_a_numpy_array_of_points__when_visualizing_based_on_a_camera_ray_residual__then_display_the_points(
        self,
    ):
        # sample 16000 points in a 3D space
        points = np.random.rand(16000, 3) * 10
        ray_origin = np.array([10, 10, 10])
        ray_dir_world = np.array([-1, -1, -1])

        viz_data_point_dir(
            points,
            ray_origin,
            ray_dir_world,
            std_dev_perpendicular=2.0,
            std_dev_distance=np.Inf,
        )

    def test_given_a_numpy_array_of_points__when_visualizing_based_on_both__then_display_the_points(
        self,
    ):
        # sample 16000 points in a 3D space
        points = np.random.rand(16000, 3) * 10
        ray_origin = np.array([10, 10, 10])
        ray_dir_world = np.array([-1, -1, -1])

        viz_data_point_dir(
            points,
            ray_origin,
            ray_dir_world,
            std_dev_perpendicular=2.0,
            std_dev_distance=2.0,
        )
