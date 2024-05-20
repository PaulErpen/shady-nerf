import unittest
from lodnelf.geometry.generate_uv_coordinates import generate_uv_coordinates


class GenerateUvCoordinatesTest(unittest.TestCase):
    def test_given_a_valid_image_size__when_generating_the_uv_coordinates__then_return_an_array_of_correct_shape(
        self,
    ):
        image_size = (128, 128)

        uv_coordinates = generate_uv_coordinates(image_size)

        self.assertEqual(uv_coordinates.shape, (128 * 128, 2))

    def test_given_a_valid_image_size__when_generating_the_uv_coordinates__then_all_array_elements_are_in_range(
        self,
    ):
        image_size = (128, 128)

        uv_coordinates = generate_uv_coordinates(image_size)

        self.assertTrue((uv_coordinates >= 0).all())
        self.assertTrue((uv_coordinates <= 1).all())
