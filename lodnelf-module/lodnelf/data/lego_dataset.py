from typing import Literal
import imageio
from lodnelf.geometry.generate_uv_coordinates import generate_uv_coordinates
import torch.utils.data
import json
from pathlib import Path
import numpy as np


class LegoDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str, split: Literal["train", "val", "test"]):
        self.meta = {}
        self.split = split
        self.data_root = Path(data_root)

        meta_path = Path(data_root) / Path("transforms_{}.json".format(self.split))

        if not self.data_root.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "The specified split file does not exist: {}".format(self.data_root)
            )

        with open(meta_path, "r") as fp:
            self.meta = json.load(fp)

    def __len__(self):
        len(self.meta["frames"])

    def __getitem__(self, idx):
        img_path = self.data_root / Path(
            self.meta["frames"][idx]["file_path"] + ".png"
        )
        # Load the image
        img = (imageio.imread(img_path) / 255.0).astype(np.float32)
        # Load the camera to world matrix
        cam2world = torch.Tensor(self.meta["frames"][idx]["transform_matrix"])
        H, W = img.shape[:2]
        # Load the uv coordinates
        uv = generate_uv_coordinates((H, W))
        camera_angle_x = float(self.meta["camera_angle_x"])
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        intrinsics = np.eye(4)
        intrinsics[0, 0] = intrinsics[1, 1] = focal
        return {"rgba": img, "cam2world": cam2world, "uv": uv, "intrinsics": intrinsics}
