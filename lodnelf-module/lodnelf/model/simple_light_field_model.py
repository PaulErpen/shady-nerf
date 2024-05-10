import torch.nn as nn
import torch
from lodnelf.model import custom_layers
from lodnelf.util import util
from lodnelf.geometry import geometry
import time


class SimpleLightFieldModel(nn.Module):
    def __init__(self, latent_dim, depth=False, alpha=False, device="cpu"):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_hidden_units_phi = 256
        self.depth = depth
        self.alpha = alpha

        out_channels = 3

        if self.depth:
            out_channels += 1
        if self.alpha:
            out_channels += 1
            self.background = torch.ones((1, 1, 1, 3)).to(device)

        self.phi = custom_layers.FCBlock(
            hidden_ch=self.num_hidden_units_phi,
            num_hidden_layers=6,
            in_features=6,
            out_features=out_channels,
            outermost_linear=True,
            norm="layernorm_na",
        )

    def get_light_field_function(self):
        return self.phi

    def get_query_cam(self, input):
        query_dict = input["query"]
        pose = util.flatten_first_two(query_dict["cam2world"])
        intrinsics = util.flatten_first_two(query_dict["intrinsics"])
        uv = util.flatten_first_two(query_dict["uv"].float())
        return pose, intrinsics, uv

    def forward(self, input, val=False, compute_depth=False, timing=False):
        out_dict = {}
        query = input["query"]
        b, n_ctxt = query["uv"].shape[:2]
        n_qry, n_pix = query["uv"].shape[1:3]

        query_pose, query_intrinsics, query_uv = self.get_query_cam(input)

        light_field_coords = geometry.plucker_embedding(
            query["cam2world"], query["uv"], query["intrinsics"]
        )

        light_field_coords.requires_grad_(True)
        out_dict["coords"] = light_field_coords.view(b, n_qry, 6)

        lf_function = self.get_light_field_function()
        out_dict["lf_function"] = lf_function

        if timing:
            t0 = time.time()
        lf_out = lf_function(out_dict["coords"])
        if timing:
            t1 = time.time()
            total_n = t1 - t0
            print(f"{total_n}")

        rgb = lf_out[..., :3]

        if self.depth:
            depth = lf_out[..., 3:4]
            out_dict["depth"] = depth.view(b, n_qry, 1)

        rgb = rgb.view(b, n_qry, 3)

        if self.alpha:
            alpha = lf_out[..., -1:].view(b, n_qry, 1)
            weight = 1 - torch.exp(-torch.abs(alpha))
            rgb = weight * rgb + (1 - weight) * self.background
            out_dict["alpha"] = weight

        if compute_depth:
            with torch.enable_grad():
                lf_function = self.get_light_field_function()
                depth = util.light_field_depth_map(
                    light_field_coords, query_pose, lf_function
                )["depth"]
                depth = depth.view(b, n_qry, n_pix, 1)
                out_dict["depth"] = depth

        out_dict["rgb"] = rgb
        return out_dict
