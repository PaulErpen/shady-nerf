import unittest

from lodnelf.metrics.psnr_metric import PsnrMetric, PsnrMetricResult
import torch


class PsnrMetricTest(unittest.TestCase):
    def test_given_valid_params__when_instantiating_the_metric__then_do_not_raise_exception(
        self,
    ):
        PsnrMetric()

    def test_given_images_with_different_shapes__when_calculating_psnr__then_raise_exception(
        self,
    ):
        metric = PsnrMetric()
        image1 = torch.rand(1, 256, 256, 3)
        image2 = torch.rand(1, 128, 128, 3)
        with self.assertRaises(ValueError):
            metric(image1, image2)

    def test_given_the_first_image_with_5_channels__when_calculating_psnr__then_raise_exception(
        self,
    ):
        metric = PsnrMetric()
        image1 = torch.rand(1, 256, 256, 5)
        image2 = torch.rand(1, 256, 256, 3)
        with self.assertRaises(ValueError):
            metric(image1, image2)

    def test_given_the_second_image_with_5_channels__when_calculating_psnr__then_raise_exception(
        self,
    ):
        metric = PsnrMetric()
        image1 = torch.rand(1, 256, 256, 3)
        image2 = torch.rand(1, 256, 256, 5)
        with self.assertRaises(ValueError):
            metric(image1, image2)

    def test_given_two_completely_similar_images__when_calculating_psnr__then_return_infinity(
        self,
    ):
        metric = PsnrMetric()
        image1 = torch.rand(1, 256, 256, 3)
        image2 = image1.clone()
        result = metric(image1, image2)
        self.assertEqual(result.get_value(), float("inf"))

    def test_given_two_different_images__when_calculating_psnr__then_return_a_float_value(
        self,
    ):
        metric = PsnrMetric()
        image1 = torch.rand(1, 256, 256, 3)
        image2 = torch.rand(1, 256, 256, 3)
        result = metric(image1, image2)
        self.assertIsInstance(result.get_value(), float)

    def test_given_two_psnr_results__when_comparing_them__then_return_true_if_the_first_result_is_better(
        self,
    ):
        result1 = PsnrMetricResult(10)
        result2 = PsnrMetricResult(5)
        self.assertTrue(result1.is_better_than(result2))

    def test_given_two_psnr_results__when_comparing_them__then_return_false_if_the_second_result_is_better(
        self,
    ):
        result1 = PsnrMetricResult(10)
        result2 = PsnrMetricResult(20)
        self.assertFalse(result1.is_better_than(result2))

    def test_given_two_psnr_results__when_merging_them__then_return_a_new_result_with_the_average_psnr(
        self,
    ):
        result1 = PsnrMetricResult(6, 1)
        result2 = PsnrMetricResult(3, 2)
        merged_result = result1.merge(result2)
        self.assertEqual(merged_result.psnr, 4)

    def test_given_two_results_with_different_ns__when_comparing_them__then_raise_exception(
        self,
    ):
        result1 = PsnrMetricResult(6, 1)
        result2 = PsnrMetricResult(3, 2)
        with self.assertRaises(ValueError):
            result1.is_better_than(result2)
