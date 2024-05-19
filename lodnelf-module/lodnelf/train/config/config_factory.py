from typing import Dict
from lodnelf.train.config.abstract_config import AbstractConfig
from lodnelf.train.config.red_car_configs import (
    SimpleRedCarModelConfigDepthAlpha,
    SimpleRedCarModelConfig,
    SimpleRedCarModelConfigSiren,
)
from lodnelf.train.config.abstract_config import AbstractConfig


class ConfigFactory:
    def __init__(self):
        self.configs: Dict[str, AbstractConfig] = {}

        simple_red_car_model_depth_alpha = SimpleRedCarModelConfigDepthAlpha()
        self.configs[simple_red_car_model_depth_alpha.get_name()] = (
            simple_red_car_model_depth_alpha
        )

        simple_red_car_model = SimpleRedCarModelConfig()
        self.configs[simple_red_car_model.get_name()] = simple_red_car_model

        simple_red_car_model_siren = SimpleRedCarModelConfigSiren()
        self.configs[simple_red_car_model_siren.get_name()] = simple_red_car_model_siren

    def get_by_name(self, name: str) -> AbstractConfig:
        if name not in self.configs:
            raise ValueError(f"Config with name {name} not found.")
        return self.configs[name]
