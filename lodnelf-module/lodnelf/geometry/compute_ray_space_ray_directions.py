import torch


def compute_cam_space_ray_directions(H, W, focal_length):
    x, y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy")
    directions = torch.stack(
        [
            (x - W * 0.5) / focal_length,
            -(y - H * 0.5) / focal_length,
            -torch.ones_like(x),
        ],
        -1,
    )
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions
