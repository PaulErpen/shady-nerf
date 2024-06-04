from typing import Literal
from lodnelf.geometry.compute_ray_space_ray_directions import (
    compute_cam_space_ray_directions,
)
import numpy as np
import pygame
from pygame.locals import *
from PIL import Image
from lodnelf.util import util
import torch
from lodnelf.train.config.config_factory import ConfigFactory
from lodnelf.geometry.rotation_matrix import rotation_matrix


class InteractiveDisplay:
    def __init__(
        self,
        config_name: str,
        model_save_path: str,
        display_size=440,
        mode: Literal["rgb", "rgba"] = "rgb",
    ):
        self.mode = mode
        config_factory = ConfigFactory()
        self.config = config_factory.get_by_name(config_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(model_save_path, map_location=device)

        self.model = self.config.get_model()
        self.model.load_state_dict(state_dict)

        self.display_size = display_size

        self.start_cam2world_matrix = self.config.get_initial_cam2world_matrix()
        self.focal_length = self.config.get_camera_focal_length()

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.display_size, self.display_size))
        pygame.display.set_caption("Interactive Showcase")

        # Initial camera to world matrix 4x4 matrix
        cam2world_matrix = self.start_cam2world_matrix

        while True:
            rotation_x: int | None = None
            rotation_y: int | None = None
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                elif event.type == KEYDOWN:
                    if event.key == K_LEFT:
                        rotation_x = 10
                    if event.key == K_RIGHT:
                        rotation_x = -10
                    if event.key == K_UP:
                        rotation_y = 10
                    if event.key == K_DOWN:
                        rotation_y = -10
            current_rotation_matrix = np.eye(3)
            if rotation_x is not None:
                current_rotation_matrix = rotation_matrix(
                    [0, 0, 1], np.radians(rotation_x)
                )
            if rotation_y is not None:
                current_rotation_matrix = (
                    rotation_matrix([1, 0, 0], np.radians(rotation_y))
                    @ current_rotation_matrix
                )
            if rotation_x is not None or rotation_y is not None:
                initial_translation = cam2world_matrix[:3, 3]
                new_translation = np.matmul(
                    current_rotation_matrix, initial_translation
                )
                cam2world_matrix[:3, :3] = np.matmul(
                    current_rotation_matrix,
                    cam2world_matrix[:3, :3],
                )
                cam2world_matrix[:3, 3] = new_translation

            image = self.update_image(cam2world_matrix)
            image = image.resize((self.display_size, self.display_size))
            mode = image.mode
            size = image.size
            data = image.tobytes()

            pygame_image = pygame.image.fromstring(data, size, mode)
            screen.fill((255, 255, 255))
            screen.blit(pygame_image, (0, 0))
            pygame.display.flip()

    def update_image(self, cam2world_matrix):
        H, W = self.config.get_output_image_size()
        directions = compute_cam_space_ray_directions(
            H, W, self.focal_length, fraction=float(128 / H)
        )
        world_space_ray_directions = directions.view(-1, 3) @ cam2world_matrix[:3, :3].T
        # repeat the cam2world matrix for each pixel
        model_input = (
            cam2world_matrix[:3, 3].expand(world_space_ray_directions.shape[0], 3),
            world_space_ray_directions,
            torch.zeros((world_space_ray_directions.shape[0], 3)),
        )
        model_output = self.model(model_input)
        model_output = (
            model_output.view(128, 128, 3 if self.mode == "rgb" else 4)
            .detach()
            .cpu()
            .numpy()
        )
        model_output = np.clip(model_output, 0, 1)
        return Image.fromarray((model_output * 255).astype(np.uint8))
