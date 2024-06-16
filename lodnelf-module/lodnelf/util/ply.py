# load all the points in a ply file
import plyfile
import numpy as np


def load_ply(file_name: str) -> np.ndarray:
    ply_data = plyfile.PlyData.read(file_name)
    vertices = ply_data["vertex"]

    points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    return points
