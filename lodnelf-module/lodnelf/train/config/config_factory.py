from typing import Dict
from lodnelf.train.config.abstract_config import AbstractConfig
from lodnelf.train.config.lego_configs import (
    DeepPluckerLegoThreeConfig,
    DeepPluckerLegoSixConfig,
    PlanarFourierLegoThreeConfig,
    PlanarFourierLegoThreeToThreeConfig,
    FullFourierLegoThreeConfig,
)
from lodnelf.train.config.abstract_config import AbstractConfig


class ConfigFactory:
    def __init__(self):
        self.configs: Dict[str, AbstractConfig] = {}

        lego_three = DeepPluckerLegoThreeConfig()
        self.configs[lego_three.get_name()] = lego_three

        lego_six = DeepPluckerLegoSixConfig()
        self.configs[lego_six.get_name()] = lego_six

        lego_planar_fourier = PlanarFourierLegoThreeConfig()
        self.configs[lego_planar_fourier.get_name()] = lego_planar_fourier

        lego_planar_fourier_3_to_3 = PlanarFourierLegoThreeToThreeConfig()
        self.configs[lego_planar_fourier_3_to_3.get_name()] = lego_planar_fourier_3_to_3

        lego_full_fourier = FullFourierLegoThreeConfig()
        self.configs[lego_full_fourier.get_name()] = lego_full_fourier

    def get_by_name(self, name: str) -> AbstractConfig:
        if name not in self.configs:
            raise ValueError(f"Config with name {name} not found.")
        return self.configs[name]
