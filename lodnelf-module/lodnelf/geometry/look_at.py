import numpy as np


def look_at(camera_position, target_position, up_vector=np.array([0, 1, 0])):
    """
    Create a look-at rotation matrix to make the camera look at the target position.
    """
    forward = target_position - camera_position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up_vector, forward)
    right = right / np.linalg.norm(right)

    up = np.cross(forward, right)

    look_at_matrix = np.array(
        [
            [right[0], right[1], right[2]],
            [up[0], up[1], up[2]],
            [-forward[0], -forward[1], -forward[2]],
        ]
    )

    return look_at_matrix
