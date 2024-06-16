import random
import numpy as np


def stratified_sampling(points: np.ndarray, k: int) -> np.ndarray:
    """Select k points using stratified sampling to ensure even distribution."""
    # Determine the bounding box of the points
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    bbox_size = max_coords - min_coords

    # Define the number of divisions along each axis
    divisions = int(
        np.cbrt(k)
    )  # Cube root to divide into approximately equal grid cells
    cell_size = bbox_size / divisions

    # Create a dictionary to hold points in each grid cell
    grid = {}
    for point in points:
        cell_index = tuple(((point - min_coords) // cell_size).astype(int))
        if cell_index not in grid:
            grid[cell_index] = []
        grid[cell_index].append(point)

    # Select one point from each non-empty grid cell until we have k points
    selected_points = []
    for cell_points in grid.values():
        selected_points.append(random.choice(cell_points))
        if len(selected_points) >= k:
            break

    # If we don't have enough points yet, randomly select remaining points
    if len(selected_points) < k:
        remaining_points = np.array([p for sublist in grid.values() for p in sublist])
        remaining_points = remaining_points[
            np.random.choice(
                remaining_points.shape[0], k - len(selected_points), replace=False
            )
        ]
        selected_points.extend(remaining_points)

    return np.array(selected_points)
