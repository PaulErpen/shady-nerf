import unittest

from lodnelf.geometry.embed_spherical_harmonics import embed_spherical_harmonics
import torch


class EmbedSphericalHarmonicsTest(unittest.TestCase):
    def test_given_a_tensor_of_batch_size_10__when_embedding__then_return_a_tensor_with_the_correct_shape(
        self,
    ):
        dirs = torch.rand(10, 3)

        embedding = embed_spherical_harmonics(dirs)

        self.assertEqual(embedding.shape, (10, 64))
