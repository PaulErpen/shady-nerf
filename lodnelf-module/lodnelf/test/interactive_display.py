import numpy as np
import pygame
from pygame.locals import *
from PIL import Image
from lodnelf.util import util
import torch
from lodnelf.train.config.config_factory import ConfigFactory
from lodnelf.geometry.generate_uv_coordinates import generate_uv_coordinates
from lodnelf.geometry.rotation_matrix import rotation_matrix


class InteractiveDisplay:
    def __init__(self, config_name: str, model_save_path: str):
        config_factory = ConfigFactory()
        loaded_config = config_factory.get_by_name(config_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(model_save_path, map_location=device)

        self.model = loaded_config.get_model()
        self.model.load_state_dict(state_dict)

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((128, 128))
        pygame.display.set_caption("Interactive Showcase")

        # Initial camera to world matrix (identity matrix), 4x4 matrix
        cam2world_matrix = np.eye(4)

        while True:
            rotation = np.eye(3)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                elif event.type == KEYDOWN:
                    if event.key == K_LEFT:
                        rotation = np.dot(
                            rotation_matrix([0, 1, 0], np.radians(10)),
                            cam2world_matrix[:3, :3],
                        )
                    elif event.key == K_RIGHT:
                        rotation = np.dot(
                            rotation_matrix([0, 1, 0], np.radians(-10)),
                            cam2world_matrix[:3, :3],
                        )
                    elif event.key == K_UP:
                        rotation = np.dot(
                            rotation_matrix([1, 0, 0], np.radians(10)),
                            cam2world_matrix[:3, :3],
                        )
                    elif event.key == K_DOWN:
                        rotation = np.dot(
                            rotation_matrix([1, 0, 0], np.radians(-10)),
                            cam2world_matrix[:3, :3],
                        )
            cam2world_matrix[:3, :3] = rotation

            image = self.update_image(cam2world_matrix)
            mode = image.mode
            size = image.size
            data = image.tobytes()

            pygame_image = pygame.image.fromstring(data, size, mode)
            screen.blit(pygame_image, (0, 0))
            pygame.display.flip()

    def update_image(self, cam2world_matrix):
        uv = generate_uv_coordinates((128, 128))
        # repeat the cam2world matrix for each pixel
        basic_intrinsic = torch.tensor(
            [
                [131.2500, 0.0000, 64.0000, 0.0000],
                [0.0000, 131.2500, 64.0000, 0.0000],
                [0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ]
        )
        model_input = util.add_batch_dim_to_dict(
            {
                "cam2world": torch.tensor(cam2world_matrix).to(torch.float32),
                "uv": torch.tensor(uv).to(torch.float32),
                "intrinsics": torch.tensor(basic_intrinsic).to(torch.float32),
            }
        )
        model_output = self.model(model_input)
        return Image.fromarray(
            model_output["rgb"].reshape(128, 128, 3).detach().cpu().numpy(),
            "RGB",
        )
