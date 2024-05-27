import numpy as np
import pygame
from pygame.locals import *
from PIL import Image
from lodnelf.util import util
import torch
from lodnelf.train.config.config_factory import ConfigFactory
from lodnelf.geometry.generate_uv_coordinates import generate_uv_coordinates
from lodnelf.geometry.rotation_matrix import rotation_matrix
from lodnelf.geometry.look_at import look_at


class InteractiveDisplay:
    def __init__(self, config_name: str, model_save_path: str, image_size = 440):
        config_factory = ConfigFactory()
        loaded_config = config_factory.get_by_name(config_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(model_save_path, map_location=device)

        self.model = loaded_config.get_model()
        self.model.load_state_dict(state_dict)

        self.image_size = image_size

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.image_size, self.image_size))
        pygame.display.set_caption("Interactive Showcase")

        # Initial camera to world matrix 4x4 matrix
        cam2world_matrix = np.array(
            [
                [9.8176e-01, 1.2364e-01, -1.4440e-01, 1.8772e-01],
                [1.9010e-01, -6.3854e-01, 7.4574e-01, -9.6946e-01],
                [-2.2352e-08, -7.5959e-01, -6.5040e-01, 8.4552e-01],
                [-0.0000e00, 0.0000e00, -0.0000e00, 1.0000e00],
            ]
        )
        print(cam2world_matrix)

        while True:
            rotation: int | None = None
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                elif event.type == KEYDOWN:
                    if event.key == K_LEFT:
                        rotation = 10
                    if event.key == K_RIGHT:
                        rotation = -10
                    # if event.key == K_LEFT:
                    #     rotation = rotation_matrix([0, 1, 0], np.radians(10))
                    # elif event.key == K_RIGHT:
                    #     rotation = rotation_matrix([0, 1, 0], np.radians(-10))
                    # elif event.key == K_UP:
                    #     rotation = rotation_matrix([1, 0, 0], np.radians(10))
                    # elif event.key == K_DOWN:
                    #     rotation = rotation_matrix([1, 0, 0], np.radians(-10))

            if rotation is not None:
                current_rotation_matrix = rotation_matrix(
                    [0, 0, 1], np.radians(rotation)
                )
                initial_translation = cam2world_matrix[:3, 3]
                new_translation = np.matmul(
                    current_rotation_matrix, initial_translation
                )
                # look_at_matrix = look_at(
                #    initial_translation,
                #    np.array([0, 0, 0]),
                #    up_vector=np.array([0, 0, 1]),
                # )
                cam2world_matrix[:3, :3] = np.matmul(
                    current_rotation_matrix,
                    cam2world_matrix[:3, :3], 
                )
                cam2world_matrix[:3, 3] = new_translation

            image = self.update_image(cam2world_matrix)
            image = image.resize((self.image_size, self.image_size))
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
        rgb = model_output
        rgb = rgb.view(128, 128, 3).detach().cpu().numpy()
        rgb = np.clip(rgb, 0, 1)
        return Image.fromarray((rgb * 255).astype(np.uint8))
