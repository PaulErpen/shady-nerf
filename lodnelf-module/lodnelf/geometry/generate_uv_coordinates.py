import numpy as np


def generate_uv_coordinates(image_size):
    width, height = image_size
    # Generate a grid of (x, y) coordinates
    u, v = np.mgrid[0:height, 0:width]

    # Stack to get (u, v) pairs
    uv_coordinates = np.array([u, v]).transpose(2, 1, 0).reshape(-1, 2)
    return uv_coordinates
