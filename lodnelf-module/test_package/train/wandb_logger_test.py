import unittest
from lodnelf.train.wandb_logger import WandBLogger


class WandBLoggerTest(unittest.TestCase):
    def test_given_a_valid_api_key__when_instantiating_the_logger__then_to_not_throw_any_errors(
        self,
    ):
        WandBLogger.from_env(
            run_config={"config": "config"},
            run_name="unittest_run",
            group_name="unittest",
        )
