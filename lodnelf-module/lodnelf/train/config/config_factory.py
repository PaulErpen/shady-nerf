from typing import Dict
from lodnelf.train.config.abstract_config import AbstractConfig
from lodnelf.train.config.red_car_configs import (
    SimpleRedCarModelConfigDeepPlucker,
    SimpleRedCarModelConfigDeepPlucker6,
    SimpleRedCarModelMySiren,
)
from lodnelf.train.config.lego_configs import (
    DeepPluckerLegoThreeConfig,
    DeepPluckerLegoSixConfig,
    PlanarFourierLegoThreeConfig,
)
from lodnelf.train.config.abstract_config import AbstractConfig


class ConfigFactory:
    def __init__(self):
        self.configs: Dict[str, AbstractConfig] = {}

        red_car_deep_plucker = SimpleRedCarModelConfigDeepPlucker()
        self.configs[red_car_deep_plucker.get_name()] = red_car_deep_plucker

        red_car_deep_plucker6 = SimpleRedCarModelConfigDeepPlucker6()
        self.configs[red_car_deep_plucker6.get_name()] = red_car_deep_plucker6

        red_car_my_siren = SimpleRedCarModelMySiren()
        self.configs[red_car_my_siren.get_name()] = red_car_my_siren

        lego_three = DeepPluckerLegoThreeConfig()
        self.configs[lego_three.get_name()] = lego_three

        lego_six = DeepPluckerLegoSixConfig()
        self.configs[lego_six.get_name()] = lego_six

        lego_planar_fourier = PlanarFourierLegoThreeConfig()
        self.configs[lego_planar_fourier.get_name()] = lego_planar_fourier

    def get_by_name(self, name: str) -> AbstractConfig:
        if name not in self.configs:
            raise ValueError(f"Config with name {name} not found.")
        return self.configs[name]
