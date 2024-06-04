import torch


def compute_cam_space_ray_directions(
    H: int, W: int, focal_length: float, fraction: None | float = None
) -> torch.Tensor:
    if fraction is not None and (fraction > 1.0 or fraction < 0.0):
        raise ValueError("fraction must be between 0 and 1")
    fraction = fraction or 1.0

    x, y = torch.meshgrid(
        torch.linspace(0, H, steps=int(H * fraction)),
        torch.linspace(0, W, steps=int(W * fraction)),
        indexing="xy",
    )
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
