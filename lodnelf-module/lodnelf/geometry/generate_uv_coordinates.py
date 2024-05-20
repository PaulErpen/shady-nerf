import numpy as np


def generate_uv_coordinates(image_size):
    width, height = image_size
    # Generate a grid of (x, y) coordinates
    y, x = np.mgrid[0:height, 0:width]

    # Normalize to range [0, 1]
    u = x / float(width - 1)
    v = y / float(height - 1)

    # Stack to get (u, v) pairs
    uv_coordinates = np.array([u, v]).transpose(1, 2, 0).reshape(-1, 2)
    return uv_coordinates
