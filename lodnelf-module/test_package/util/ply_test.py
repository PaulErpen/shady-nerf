import unittest

from lodnelf.util.ply import load_ply


class PlyTest(unittest.TestCase):
    def test_given_a_valid_path_to_a_ply_file__when_loading__then_return_a_numpy_array(
        self,
    ):
        pass
        # when
        points = load_ply("data/lego-points/lego.ply")
        # then
        self.assertEqual(points.shape, (992533, 3))
