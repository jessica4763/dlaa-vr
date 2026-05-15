import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import imageio.v3
import numpy as np
from pathlib import Path
from scipy import stats
import torch
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.utils import save_image

from utils import gamma_to_linear
from network.vr_network import VRConfig


def warp_frames(left_frame_path: Path, right_frame_path: Path, depth_path: Path) -> None:
    def right_to_left_warp(
        right_frame: torch.Tensor,
        left_depth: torch.Tensor,
        camera_baseline: float,
        focal_length: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = right_frame.shape

        ys, xs = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing="ij"
        )
        ys = ys + 0.5 
        xs = xs + 0.5

        disparity = (camera_baseline * focal_length) / ((left_depth * 99990.0) + 10.0) 
        warp_xs = xs - disparity  # Note xs is broadcast here
        warp_xs = torch.permute(warp_xs, dims=(0, 2, 3, 1))  # (B, C, H, W) --> (B, H, W, C)
        warp_xs = 2.0 * (warp_xs / W) - 1.0  # Normalise to range [-1, 1]. Divide by W rather than W - 1 because we set align_corners=False
        ys = ys.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        ys = torch.permute(ys, dims=(0, 2, 3, 1))  # (B, C, H, W) --> (B, H, W, C)
        ys = 2.0 * (ys / H) - 1.0  # Normalise to range [-1, 1]. Divide by H rather than H - 1 because we set align_corners=False
        warp_grid = torch.cat((warp_xs, ys), dim=-1)

        # Warping the right frame on to the left frame
        warped_left_frame = F.grid_sample(
            right_frame,
            warp_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        return warped_left_frame
    
    vr_config = VRConfig(
        camera_baseline=6.4, 
        horizontal_fov=100.0,
        vertical_fov=105.8809,
        horizontal_resolution=1440,
        vertical_resolution=1600
    )

    right_frame = gamma_to_linear(decode_image(right_frame_path.resolve())[0:3].float())
    depth = imageio.v3.imread(depth_path.resolve())
    depth = torch.permute(torch.from_numpy(depth), (2, 0, 1))
    depth = depth[0:1]

    warped_left_frame = right_to_left_warp(
        right_frame.unsqueeze(0),
        depth.unsqueeze(0),
        vr_config.camera_baseline,
        vr_config.focal_length
    )

    left_frame = gamma_to_linear(decode_image(left_frame_path.resolve())[0:3].float())

    diff = torch.abs(left_frame - warped_left_frame)
    save_image(diff, "warped_left.png")
