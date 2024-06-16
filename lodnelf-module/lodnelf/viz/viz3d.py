# visualize the points
import matplotlib.pyplot as plt
import numpy as np


def viz_points(points: np.ndarray, other_color=None, ax=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # compute color based on distance to camera
    if other_color is not None:
        col = other_color
    else:
        col = np.linalg.norm(points - np.array([1, -1, 1]), axis=1)
    plot = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1, c=col)
    # change color map
    plot.set_cmap("cool")
    # show the legend for the colors
    fig.colorbar(plot)
    plt.show()


def viz_data_point_dir(
    points: np.ndarray,
    ray_origin: np.ndarray,
    ray_dir_world: np.ndarray,
    std_dev_perpendicular: float = 1.0,
    std_dev_distance: float = 1.0,
):
    v = points - ray_origin
    len_perpendicular = np.linalg.norm(np.cross(v, ray_dir_world), axis=1)
    ray_alignment = np.dot(v, ray_dir_world)
    w_perp = (1 / (np.sqrt(2 * np.pi * std_dev_perpendicular**2))) * np.exp(
        -0.5 * (len_perpendicular / std_dev_perpendicular) ** 2
    )
    w_dist = (1 / (np.sqrt(2 * np.pi * std_dev_distance**2))) * np.exp(
        -0.5 * (ray_alignment / std_dev_distance) ** 2
    )
    other_color = w_dist + w_perp

    viz_points(points, other_color)
