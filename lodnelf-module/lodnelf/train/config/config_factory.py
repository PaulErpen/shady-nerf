from typing import Dict
from lodnelf.train.config.abstract_config import AbstractConfig
from lodnelf.train.config.red_car_configs import (
    SimpleRedCarModelConfigDepthAlpha,
    SimpleRedCarModelConfig,
    SimpleRedCarModelConfigSiren,
    SimpleRedCarModelConfigPlanarFourier,
    SimpleRedCarModelConfigDeepPlucker,
    SimpleRedCarModelConfigDeepPlucker6,
    SimpleRedCarModelMySiren,
    SimpleRedCarModelConfigSinusoidalDeepPlucker,
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

        red_car_planar_fourier = SimpleRedCarModelConfigPlanarFourier()
        self.configs[red_car_planar_fourier.get_name()] = red_car_planar_fourier

        red_car_deep_plucker = SimpleRedCarModelConfigDeepPlucker()
        self.configs[red_car_deep_plucker.get_name()] = red_car_deep_plucker

        red_car_deep_plucker6 = SimpleRedCarModelConfigDeepPlucker6()
        self.configs[red_car_deep_plucker6.get_name()] = red_car_deep_plucker6

        red_car_my_siren = SimpleRedCarModelMySiren()
        self.configs[red_car_my_siren.get_name()] = red_car_my_siren

        red_car_sinusoidal_deep_plucker = SimpleRedCarModelConfigSinusoidalDeepPlucker()
        self.configs[red_car_sinusoidal_deep_plucker.get_name()] = (
            red_car_sinusoidal_deep_plucker
        )

    def get_by_name(self, name: str) -> AbstractConfig:
        if name not in self.configs:
            raise ValueError(f"Config with name {name} not found.")
        return self.configs[name]
