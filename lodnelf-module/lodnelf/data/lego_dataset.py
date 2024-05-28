from typing import Literal, Tuple
import imageio
from lodnelf.geometry.generate_uv_coordinates import generate_uv_coordinates
import torch.utils.data
import json
from pathlib import Path
import numpy as np
import PIL.Image


class LegoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        split: Literal["train", "val", "test"],
        image_size: Tuple[int, int] = (800, 800),
    ):
        self.meta = {}
        self.split = split
        self.data_root = Path(data_root)
        self.image_size = image_size

        meta_path = Path(data_root) / Path("transforms_{}.json".format(self.split))

        if not self.data_root.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "The specified split file does not exist: {}".format(self.data_root)
            )

        with open(meta_path, "r") as fp:
            self.meta = json.load(fp)

        tmp_img = []
        for frame in self.meta["frames"]:
            img_path = self.data_root / Path(frame["file_path"] + ".png")
            # Load the image
            img = imageio.imread(img_path)
            img = PIL.Image.fromarray(img)
            img = img.resize(self.image_size)
            img = (
                torch.tensor(np.array(img) / 255.0)
                .view(self.image_size[0] * self.image_size[1], 4)
                .float()
            )
            tmp_img.append(img)
        self.images = torch.stack(tmp_img, dim=0)

    def __len__(self):
        return len(self.meta["frames"])

    def __getitem__(self, idx):

        # Load the camera to world matrix
        cam2world = torch.Tensor(self.meta["frames"][idx]["transform_matrix"])
        H, W = self.image_size
        # Load the uv coordinates
        uv = generate_uv_coordinates((H, W))
        camera_angle_x = float(self.meta["camera_angle_x"])
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        intrinsics = np.eye(4)
        intrinsics[0, 0] = intrinsics[1, 1] = focal
        return {
            "rgba": self.images[idx],
            "cam2world": cam2world.float(),
            "uv": torch.from_numpy(uv).float(),
            "intrinsics": torch.from_numpy(intrinsics).float(),
        }
