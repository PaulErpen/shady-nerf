import unittest
from lodnelf.train.config.config_factory import ConfigFactory
from lodnelf.train.config.lego_configs import DeepPluckerLegoSixConfig


class ConfigFactoryTest(unittest.TestCase):
    def test_given_a_valid_config_name__when_getting_by_name__then_return_the_config(
        self,
    ):
        factory = ConfigFactory()
        config = factory.get_by_name("DeepPluckerLegoSix")
        self.assertIsInstance(config, DeepPluckerLegoSixConfig)

    def test_given_an_invalid_config_name__when_getting_by_name__then_throw_error(
        self,
    ):
        factory = ConfigFactory()
        with self.assertRaises(ValueError):
            factory.get_by_name("InvalidConfigName")
