import torch.utils.data
import numpy as np
import cv2
import torch
from lodnelf.data import data_util
from lodnelf.util import util
import h5py


class Hdf5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        instance_idx,
        instance_ds,
        img_sidelength,
        instance_name,
        specific_observation_idcs=None,
        num_images=None,
        cache=None,
    ):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.cache = cache
        self.instance_ds = instance_ds
        self.has_depth = False

        self.color_keys = sorted(list(instance_ds["rgb"].keys()))
        self.pose_keys = sorted(list(instance_ds["pose"].keys()))
        self.instance_name = instance_name

        if specific_observation_idcs is not None:
            self.color_keys = util.pick(self.color_keys, specific_observation_idcs)
            self.pose_keys = util.pick(self.pose_keys, specific_observation_idcs)
        elif num_images is not None:
            idcs = np.linspace(
                0, stop=len(self.color_keys), num=num_images, endpoint=False, dtype=int
            )
            self.color_keys = util.pick(self.color_keys, idcs)
            self.pose_keys = util.pick(self.pose_keys, idcs)

        dummy_img = data_util.load_rgb_hdf5(self.instance_ds, self.color_keys[0])
        self.org_sidelength = dummy_img.shape[1]

        if self.org_sidelength < self.img_sidelength:
            uv = (
                np.mgrid[0 : self.img_sidelength, 0 : self.img_sidelength]
                .astype(np.int32)
                .transpose(1, 2, 0)
            )
            self.intrinsics, _, _ = util.parse_intrinsics_hdf5(
                instance_ds["intrinsics.txt"], trgt_sidelength=self.img_sidelength
            )
        else:
            uv = (
                np.mgrid[0 : self.org_sidelength, 0 : self.org_sidelength]
                .astype(np.int32)
                .transpose(1, 2, 0)
            )
            uv = cv2.resize(
                uv,
                (self.img_sidelength, self.img_sidelength),
                interpolation=cv2.INTER_NEAREST,
            )
            self.intrinsics, _, _ = util.parse_intrinsics_hdf5(
                instance_ds["intrinsics.txt"], trgt_sidelength=self.org_sidelength
            )

        uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
        self.uv = uv.reshape(-1, 2).float()
        self.intrinsics = torch.from_numpy(self.intrinsics).float()

    def __len__(self):
        return min(len(self.pose_keys), len(self.color_keys))

    def __getitem__(self, idx):
        key = f"{self.instance_idx}_{idx}"
        if (self.cache is not None) and (key in self.cache):
            rgb, pose = self.cache[key]
        else:
            rgb = data_util.load_rgb_hdf5(self.instance_ds, self.color_keys[idx])
            pose = data_util.load_pose_hdf5(self.instance_ds, self.pose_keys[idx])

            if (self.cache is not None) and (key not in self.cache):
                self.cache[key] = rgb, pose

        rgb = cv2.resize(
            rgb,
            (self.img_sidelength, self.img_sidelength),
            interpolation=cv2.INTER_NEAREST,
        )
        rgb = rgb.reshape(-1, 3)

        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze().long(),
            "rgb": torch.from_numpy(rgb).float(),
            "cam2world": torch.from_numpy(pose).float(),
            "uv": self.uv,
            "intrinsics": self.intrinsics,
            "instance_name": self.instance_name,
        }
        return sample


def get_instance_datasets_hdf5(
    root,
    max_num_instances=None,
    specific_observation_idcs=None,
    cache=None,
    sidelen=None,
    max_observations_per_instance=None,
    start_idx=0,
):
    file = h5py.File(root, "r")
    instances = sorted(list(file.keys()))
    print(f"File {root}, {len(instances)} instances")

    if max_num_instances is not None:
        instances = instances[:max_num_instances]

    all_instances = [
        Hdf5Dataset(
            instance_idx=idx + start_idx,
            instance_ds=file[instance_name],
            specific_observation_idcs=specific_observation_idcs,
            img_sidelength=sidelen,
            num_images=max_observations_per_instance,
            cache=cache,
            instance_name=instance_name,
        )
        for idx, instance_name in enumerate(instances)
    ]
    return all_instances
