import unittest
from lodnelf.model.components.fourier_features import FourierFeatures
import torch
import numpy as np


class FourierFeaturesTest(unittest.TestCase):
    def test_given_valid_parameters__when_initializing_fourier_features__then_nothing_is_raised(
        self,
    ):
        FourierFeatures(10)

    def test_given_valid_parameters__when_initializing_fourier_features__then_the_correct_exponents_are_computed(
        self,
    ):
        f = FourierFeatures(10)

        self.assertEqual(f.exp.shape, (10,))

    def test_given_valid_parameters__when_forwarding_input__then_the_output_has_the_correct_shape(
        self,
    ):
        fourier_features = FourierFeatures(10)
        input = torch.randn(2, 3, 6)
        output = fourier_features(input)
        self.assertEqual(output.shape, (2, 3, 120))

    def test_given_a_simple_input__when_forwarding_input__then_the_output_is_correct(
        self,
    ):
        fourier_features = FourierFeatures(2)
        input = torch.tensor([[[1, 2]]])
        output = fourier_features(input)
        self.assertTrue(
            torch.allclose(
                output.float(),
                torch.tensor(
                    [
                        [
                            [
                                np.sin(np.pi),
                                np.sin(np.pi * 2),
                                np.sin(np.pi * 2),
                                np.sin(np.pi * 4),
                                np.cos(np.pi),
                                np.cos(np.pi * 2),
                                np.cos(np.pi * 2),
                                np.cos(np.pi * 4),
                            ]
                        ]
                    ]
                ).float(),
                atol=0.01,
            )
        )
