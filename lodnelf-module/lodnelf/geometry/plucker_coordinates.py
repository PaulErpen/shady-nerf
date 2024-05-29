import torch


def plucker_coordinates(
    ray_origins: torch.Tensor, ray_directions: torch.Tensor
) -> torch.Tensor:
    """Compute Plücker coordinates from ray origins and directions.

    Args:
        ray_origins (torch.Tensor): Ray origins with shape (batch_size, num_rays, 3).
        ray_directions (torch.Tensor): Ray directions with shape (batch_size, num_rays, 3).

    Returns:
        torch.Tensor: Plücker coordinates with shape (batch_size, num_rays, 6).
    """
    cross = torch.cross(ray_origins, ray_directions, dim=-1)
    return torch.cat((cross, ray_directions), dim=-1)
