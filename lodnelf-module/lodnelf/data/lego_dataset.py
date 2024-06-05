from typing import Callable, Literal, Tuple
import imageio
from lodnelf.geometry.compute_ray_space_ray_directions import (
    compute_cam_space_ray_directions,
)
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
        limit: int | None = None,
        image_size: Tuple[int, int] = (800, 800),
        transform: (
            None
            | Callable[
                [Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ]
        ) = None,
    ):
        self.meta = {}
        self.split = split
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.transform = transform

        if self.image_size[0] != self.image_size[1]:
            raise ValueError("Only square images are supported")
        if self.image_size[0] > 800:
            raise ValueError("Images larger than 800x800 are not supported")

        meta_path = Path(data_root) / Path("transforms_{}.json".format(self.split))

        if not self.data_root.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "The specified split file does not exist: {}".format(self.data_root)
            )

        with open(meta_path, "r") as fp:
            self.meta = json.load(fp)

        tmp_img = []
        poses = []
        for idx, frame in enumerate(self.meta["frames"]):
            if limit is not None and idx >= limit:
                break

            img_path = self.data_root / Path(frame["file_path"] + ".png")
            # Load the image
            img = imageio.imread(img_path)
            img = PIL.Image.fromarray(img)
            img = img.resize(self.image_size)
            img = torch.tensor(np.array(img) / 255.0).float()
            tmp_img.append(img)
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)
        self.poses = torch.tensor(poses)
        self.images = torch.stack(tmp_img, dim=0)
        self.camera_angle_x = float(self.meta["camera_angle_x"])

        focal_length = (
            0.5 * 800 / np.tan(0.5 * self.camera_angle_x) * (self.image_size[0] / 800)
        )

        self.ray_origins = self.poses[:, :3, 3]

        self.ray_directions = self.compute_ray_directions(
            self.image_size[0], self.image_size[1], focal_length
        )

    def compute_ray_directions(self, H, W, focal_length):
        directions = compute_cam_space_ray_directions(H, W, focal_length)
        ray_directions = []
        for pose in self.poses:
            ray_direction = directions @ pose[:3, :3].T
            ray_directions.append(ray_direction)

        return torch.stack(ray_directions, dim=0)

    def __len__(self):
        return self.images.shape[0] * self.image_size[0] * self.image_size[1]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        col = self.images.view(-1, 4)[idx]
        img_idx = idx // (self.image_size[0] * self.image_size[1])
        ray_origin = self.ray_origins[img_idx]
        ray_dir_world = self.ray_directions.view(-1, 3)[idx]
        item = (ray_origin, ray_dir_world, col)

        return item if self.transform is None else self.transform(item)
