import unittest

from lodnelf.metrics.psnr_metric import calculate_psnr
import torch


class PsnrMetricTest(unittest.TestCase):
    def test_given_a_loss_of_0__when_calculating_psnr__then_return_infinity(self):
        loss = 0
        psnr = calculate_psnr(loss)
        self.assertEqual(psnr, float("inf"))

    def test_given_a_loss_of_1__when_calculating_psnr__then_return_0(self):
        loss = 1
        psnr = calculate_psnr(loss)
        self.assertEqual(psnr, 0)
