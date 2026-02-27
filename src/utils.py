from dataclasses import dataclass, field
import imageio.v3
import math
from natsort import natsorted
import os
from pathlib import Path
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import save_image

from sanity_checks import save_input


class Scene:
    def __init__(
        self,
        scene_input_imgs_path: Path,
        scene_output_imgs_path: Path,
        colour_path_suffix: str,
    ) -> None:
        self.scene_input_imgs_path = scene_input_imgs_path
        self.scene_output_imgs_path = scene_output_imgs_path

        instances = os.listdir(scene_input_imgs_path / colour_path_suffix)
        frames = os.listdir(scene_input_imgs_path / colour_path_suffix / instances[0])

        self.num_instances = len(instances)
        self.num_frames_per_instance = len(frames)
        self.num_frames = self.num_instances * self.num_frames_per_instance


@dataclass
class VRConfig:
    camera_baseline: float = 0.065
    diagonal_fov: float = 110.0
    horizontal_resolution: int = 1440
    vertical_resolution: int = 1600
    focal_length: float = field(init=False)

    @staticmethod
    def get_focal_length(
        diagonal_fov: float, 
        horizontal_resolution: int, 
        vertical_resolution: int
    ) -> float:
        diagonal_fov_rad = math.radians(diagonal_fov)
        diagonal = math.sqrt(horizontal_resolution ** 2 + vertical_resolution ** 2)
        focal_length = diagonal / (2 * math.tan(diagonal_fov_rad / 2))
        return focal_length

    def __post_init__(self):
        self.focal_length = VRConfig.get_focal_length(self.diagonal_fov, self.horizontal_resolution, self.vertical_resolution)


def checkpoint(
    checkpoints_path: Path,
    sanity_checks_output_path: Path,
    device: str,
    model: nn.Module,
    data: Dataset,
    iterations: int,
    input_frame_height: int,
    input_frame_width: int,
    scale_factor: int,
    use_jitter: bool,
    mode: str = "training",
) -> None:
    model.eval()
    with torch.no_grad():
        # frames = [n for n in range(0, 7258) if (n + 1) % 30 != 0]
        # n = random.choice(frames)
        # print(f"{n=}")

        n = 0

        if mode == "training":
            inputs, motion_vectors, jitter, output, curr_frame_num = data[(n, 0, 0, input_frame_width, input_frame_height)]
            inputs_next, motion_vectors_next, jitter_next, output_next, curr_frame_num_next = data[(n + 1, 0, 0, input_frame_width, input_frame_height)]
        else:
            inputs, motion_vectors, jitter, output, curr_frame_num = data[n]
            inputs_next, motion_vectors_next, jitter_next, output_next, curr_frame_num_next = data[n + 1]

        # Verify input to the network
        save_input(sanity_checks_output_path, model, inputs, motion_vectors, scale_factor)

        # Verify warping 
        warped_prev = model.warp(
            output.unsqueeze(0),
            motion_vectors_next.unsqueeze(0)
        ).squeeze(0)
        save_image(linear_to_gamma(warped_prev), sanity_checks_output_path / "warped_prev.png")

        diff = torch.abs(output_next - warped_prev)
        save_image(linear_to_gamma(diff), sanity_checks_output_path / "diff.png")

        # Verify the goal of the network
        save_image(linear_to_gamma(output), sanity_checks_output_path / "ground_truth.png")

        # Strictly a training diagnostic, so it's OK if training data is used here

        # Output of the network when history is invalid
        inputs = inputs.to(device).unsqueeze(0).unsqueeze(0)
        motion_vectors = motion_vectors.to(device).unsqueeze(0).unsqueeze(0)
        jitter = jitter.to(device).unsqueeze(0).unsqueeze(0) if use_jitter else None
        anti_aliased_img, prev_pred_features, out_blending_mask = model(inputs, motion_vectors, curr_frame_num, jitter, "evaluation")
        anti_aliased_img = anti_aliased_img.squeeze(0).squeeze(0)
        anti_aliased_img = linear_to_gamma(anti_aliased_img)
        save_image(anti_aliased_img, checkpoints_path / f"colour_invalid_{iterations}.png")
        
        # Blending mask when history is invalid
        out_blending_mask = F.pixel_shuffle(out_blending_mask, upscale_factor=scale_factor)
        save_image(out_blending_mask, checkpoints_path / f"blending_mask_invalid_{iterations}.png")

        # Output of the network when history is valid
        inputs_next = inputs_next.to(device).unsqueeze(0).unsqueeze(0)
        motion_vectors_next = motion_vectors_next.to(device).unsqueeze(0).unsqueeze(0)
        jitter_next = jitter_next.to(device).unsqueeze(0).unsqueeze(0) if use_jitter else None

        c0 = model.num_curr_colour + model.num_curr_depth + model.num_curr_jitter
        c1 = c0 + model.num_prev_colour
        inputs_next[:, :, c0:c1] = F.pixel_unshuffle(output.squeeze(0).squeeze(0), downscale_factor=scale_factor)

        c0 = c1
        c1 = c0 + model.num_prev_feature
        inputs_next[:, :, c0:c1] = prev_pred_features.squeeze(0)

        anti_aliased_img, _, out_blending_mask = model(inputs_next, motion_vectors_next, curr_frame_num_next, jitter_next, "evaluation")

        anti_aliased_img = anti_aliased_img.squeeze(0).squeeze(0)
        anti_aliased_img = linear_to_gamma(anti_aliased_img)
        save_image(anti_aliased_img, checkpoints_path / f"colour_valid_{iterations}.png")

        # Blending mask when history is valid
        out_blending_mask = F.pixel_shuffle(out_blending_mask.squeeze(0), upscale_factor=scale_factor)
        save_image(out_blending_mask, checkpoints_path / f"blending_mask_valid_{iterations}.png")


def gamma_to_linear(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32) / 255.0
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4
    )


def linear_to_gamma(image: torch.Tensor) -> torch.Tensor:
    image = torch.clamp(image, 0.0, 1.0)

    # Adding this doesn't change the result of the computation at all, 
    # but sidesteps the fact that PyTorch computes the gradients of 
    # both branches of torch.where, which could result in NaN 
    # if the gradient of one of the branches is NaN even if that 
    # branch wasn't going to be taken anyway
    max_image = torch.maximum(image, torch.tensor(0.0031308, device=image.device))

    return torch.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * (max_image ** (1.0 / 2.4)) - 0.055
    )


def rgb_to_y(frame: torch.Tensor) -> torch.Tensor:
        r, g, b = frame[:, 0:1, ...], frame[:, 1:2, ...], frame[:, 2:3, ...]
        y = 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
        return y.repeat(1, 3, 1, 1)


def cumsum(xs):
    cumsum_xs = [0]
    for x in xs:
        cumsum_xs.append(cumsum_xs[-1] + x)

    return cumsum_xs


def write_frames(
    evaluation_output_path: Path,
    frames: torch.Tensor,
    batch: int
) -> None:
    for idx, frame in enumerate(frames):
        save_image(frame, evaluation_output_path / f"{batch + idx}.png")


def write_video(
    imgs_path: Path,
    filename: str,
    fps: int = 24
) -> None:
    writer = imageio.get_writer(
        imgs_path / filename,
        fps=fps,
        codec="libx264",
        quality=10,
        pixelformat='yuv420p',
        macro_block_size=8
    )

    imgs_path = imgs_path / "pred"
    for img_name in natsorted(os.listdir(imgs_path)):
        img_path = imgs_path / img_name
        img = imageio.v3.imread(img_path)
        writer.append_data(img)

    writer.close()
