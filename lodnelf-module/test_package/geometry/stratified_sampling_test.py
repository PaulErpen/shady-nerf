import unittest
from lodnelf.geometry.stratified_sampling import stratified_sampling
import numpy as np


class StratifiedSamplingTest(unittest.TestCase):
    def test_given_a_number_of_points__when_sampling__then_return_k_points(self):
        pass
        # given
        points = np.random.rand(1000, 3)
        k = 100
        # when
        sampled_points = stratified_sampling(points, k)
        # then
        self.assertEqual(sampled_points.shape, (k, 3))
