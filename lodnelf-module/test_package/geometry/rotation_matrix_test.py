import unittest
from lodnelf.geometry.rotation_matrix import rotation_matrix
import numpy as np


class RotationMatrixTest(unittest.TestCase):
    def test_given_a_valid_up_axis_and_a_theta_of_0__when_generating_a_rotation_matrix__then_matrix_is_identity(
        self,
    ):
        axis = [0, 1, 0]
        theta = 0

        matrix = rotation_matrix(axis, theta)

        self.assertTrue(np.allclose(matrix, np.eye(3)))

    def test_given_a_valid_up_axis_and_a_theta_of_2_pi__when_generating_a_rotation_matrix__then_matrix_is_correct(
        self,
    ):
        axis = [0, 1, 0]
        theta = 2 * np.pi

        matrix = rotation_matrix(axis, theta)

        self.assertTrue(np.allclose(matrix, np.eye(3)))

    def test_given_a_valid_up_axis_and_a_theta_of_pi__when_generating_a_rotation_matrix__then_matrix_is_correct(
        self,
    ):
        axis = [0, 1, 0]
        theta = np.pi

        matrix = rotation_matrix(axis, theta)
        expected_matrix = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )

        self.assertTrue(np.allclose(matrix, expected_matrix))
