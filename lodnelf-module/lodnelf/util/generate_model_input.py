from lodnelf.geometry.compute_ray_space_ray_directions import (
    compute_cam_space_ray_directions,
)
import torch


def generate_model_input(H, W, focal_length, cam2world_matrix, output_size=128):
    directions = compute_cam_space_ray_directions(
        H, W, focal_length, fraction=float(output_size / H)
    )
    world_space_ray_directions = directions.view(-1, 3) @ cam2world_matrix[:3, :3].T
    # repeat the cam2world matrix for each pixel
    return (
        cam2world_matrix[:3, 3].expand(world_space_ray_directions.shape[0], 3),
        world_space_ray_directions,
        torch.zeros((world_space_ray_directions.shape[0], 3)),
    )
