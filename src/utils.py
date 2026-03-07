from dataclasses import dataclass, field, InitVar
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
from typing import Any


# -------------------------------------------------------------------------
# ------------------------------ Dataclasses ------------------------------
# -------------------------------------------------------------------------


@dataclass
class Scene:
    scene_input_imgs_path: Path
    scene_output_imgs_path: Path

    path_suffix: InitVar[str]

    num_instances: int = field(init=False)
    num_frames_per_instance: int = field(init=False)
    num_frames: int = field(init=False)

    def __post_init__(self, path_suffix):
        instances = os.listdir(self.scene_input_imgs_path / path_suffix)
        frames = os.listdir(self.scene_input_imgs_path / path_suffix / instances[0])

        self.num_instances = len(instances)
        self.num_frames_per_instance = len(frames)
        self.num_frames = self.num_instances * self.num_frames_per_instance


@dataclass
class VRConfig:
    camera_baseline: float
    horizontal_fov: float
    vertical_fov: float
    horizontal_resolution: int 
    vertical_resolution: int

    focal_length: float = field(init=False)

    def __post_init__(self):
        # Whether we use horizontal or vertical does not matter
        self.focal_length = VRConfig.get_focal_length(self.horizontal_fov, self.horizontal_resolution)

    @staticmethod
    def get_focal_length(
        fov: float, 
        resolution: int, 
    ) -> float:
        fov_rad = math.radians(fov)
        focal_length = resolution / (2 * math.tan(fov_rad / 2))
        return focal_length
    
    @staticmethod
    def right_to_left_warp(
        right_frame: torch.Tensor,
        right_depth: torch.Tensor,
        camera_baseline: float,
        focal_length: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @staticmethod
    def left_to_right_warp(
        left_frame: torch.Tensor,
        left_depth: torch.Tensor,
        camera_baseline: float,
        focal_length: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_frame = torch.permute(left_frame, dims=(2, 0, 1))
        H, W, _ = left_frame.shape

        ys, xs = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing='ij'
        )
        base_grid = torch.stack((xs, ys), dim=-1)

        # Disparity represents how much we need to shift the pixel in the 
        # left frame to obtain the corresponding pixel in the right frame.
        # In the case of occlusions, two pixels in the left frame map to a 
        # single corresponding pixel in the right frame. 
        # focal_length here is measured in pixels, and therefore disparity 
        # is measured in pixels. 
        disparity = (camera_baseline * focal_length) / left_depth

        # warped_grid contains the new x coordinate to which each pixel is shifted.
        # There may be duplicates in warped_grid, when two pixels in the left frame
        # map to a single corresponding pixel in the right frame.
        left_to_right_warped_grid = torch.round(xs - disparity)

        # We need to deal with the duplicates in warped_grid and also compute a 
        # grid that can be used in F.grid_sample. 
        # right_to_left_warped_grid contains the locations of the pixels in the 
        # left frame from which a pixel the right frame should be obtained, and 
        # the pixel in the left frame needs to be the one at the lower depth. 
        # This is necessarily the right-most pixel in the left frame.
        closest = torch.zeros_like(xs)
        flattened_closest = closest.view(-1)

        flattened_xs = xs.view(-1)
        flattened_ys = ys.view(-1)
        flattened_indices = flattened_ys * W + flattened_xs

        flattened_closest.scatter_reduce_(
            dim=0,
            index=left_to_right_warped_grid,
            src=flattened_indices,
            reduce="amax",
            include_self=True
        )
        closest = flattened_closest.view(H, W)
        closest = torch.stack((closest, ys), dim=-1)

        valid_mask = torch.zeros_like(base_grid)
        xs = closest[..., 0]
        ys = closest[..., 1]
        valid_mask[ys, xs] = 1

        right_to_left_warped_grid = torch.copy(base_grid)
        right_to_left_warped_grid[closest] = base_grid

        # Warping the left frame on to the right frame
        warped_left_frame = F.grid_sample(
            left_frame, 
            right_to_left_warped_grid, 
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        return warped_left_frame, valid_mask


# -------------------------------------------------------------------------
# ---------------------------- Helper functions ---------------------------
# -------------------------------------------------------------------------


def cumsum(xs):
    cumsum_xs = [0]
    for x in xs:
        cumsum_xs.append(cumsum_xs[-1] + x)

    return cumsum_xs


# -------------------------------------------------------------------------
# ---------------------------- Image transforms ---------------------------
# -------------------------------------------------------------------------


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
    # Computes luma, Y', from R', G', B'
    r, g, b = frame[:, 0:1, ...], frame[:, 1:2, ...], frame[:, 2:3, ...]
    y = 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y.repeat(1, 3, 1, 1)


# -------------------------------------------------------------------------
# --------------------------------- Output --------------------------------
# -------------------------------------------------------------------------


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
        imgs_path.parent / filename,
        fps=fps,
        codec="libx264",
        quality=10,
        pixelformat='yuv420p',
        macro_block_size=8
    )

    for img_name in natsorted(os.listdir(imgs_path)):
        img_path = imgs_path / img_name
        img = imageio.v3.imread(img_path)
        writer.append_data(img)

    writer.close()


# -------------------------------------------------------------------------
# ----------------------------- Sanity checks -----------------------------
# -------------------------------------------------------------------------


def checkpoint_vr(
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
    mode: str = "training"
):
    model.eval()
    with torch.no_grad():
        # frames = [n for n in range(0, 7258) if (n + 1) % 30 != 0]
        # n = random.choice(frames)
        # print(f"{n=}")

        index = 0
        if mode == "training":
            (
                left_inputs,
                right_inputs, 
                left_motion_vectors, 
                right_motion_vectors, 
                left_jitter,
                right_jitter, 
                left_output, 
                right_output, 
                curr_frame_num
            ) = data[(index, 0, 0, input_frame_width, input_frame_height)]

            (
                left_inputs_next,
                right_inputs_next, 
                left_motion_vectors_next, 
                right_motion_vectors_next, 
                left_jitter_next,
                right_jitter_next, 
                left_output_next, 
                right_output_next, 
                curr_frame_num_next
            ) = data[(index + 1, 0, 0, input_frame_width, input_frame_height)]
        else:
            (
                left_inputs,
                right_inputs, 
                left_motion_vectors, 
                right_motion_vectors, 
                left_jitter,
                right_jitter, 
                left_output, 
                right_output, 
                curr_frame_num
            ) = data[index]

            (
                left_inputs_next,
                right_inputs_next, 
                left_motion_vectors_next, 
                right_motion_vectors_next, 
                left_jitter_next,
                right_jitter_next, 
                left_output_next, 
                right_output_next, 
                curr_frame_num_next
            ) = data[index + 1]

        # Verify input to the network
        save_input(sanity_checks_output_path / "left", model, left_inputs, left_motion_vectors, scale_factor)
        save_input(sanity_checks_output_path / "right", model, right_inputs, right_motion_vectors, scale_factor)

        # Verify warping 
        left_warped_prev = model.warp(
            left_output.unsqueeze(0),
            left_motion_vectors_next.unsqueeze(0)
        ).squeeze(0)
        save_image(linear_to_gamma(left_warped_prev), sanity_checks_output_path / "left_warped_prev.png")

        left_diff = torch.abs(left_output_next - left_warped_prev)
        save_image(linear_to_gamma(left_diff), sanity_checks_output_path / "left_diff.png")

        right_warped_prev = model.warp(
            right_output.unsqueeze(0),
            right_motion_vectors_next.unsqueeze(0)
        ).squeeze(0)
        save_image(linear_to_gamma(right_warped_prev), sanity_checks_output_path / "right_warped_prev.png")

        right_diff = torch.abs(right_output_next - right_warped_prev)
        save_image(linear_to_gamma(right_diff), sanity_checks_output_path / "right_diff.png")

        # Verify the goal of the network
        save_image(linear_to_gamma(left_output), sanity_checks_output_path / "left_ground_truth.png")
        save_image(linear_to_gamma(right_output), sanity_checks_output_path / "right_ground_truth.png")

        # Strictly a training diagnostic, so it's OK if training data is used here

        # Output of the network when history is invalid
        left_inputs = left_inputs.to(device).unsqueeze(0).unsqueeze(0)
        left_motion_vectors = left_motion_vectors.to(device).unsqueeze(0).unsqueeze(0)
        left_jitter = left_jitter.to(device).unsqueeze(0).unsqueeze(0) if use_jitter else None

        right_inputs = right_inputs.to(device).unsqueeze(0).unsqueeze(0)
        right_motion_vectors = right_motion_vectors.to(device).unsqueeze(0).unsqueeze(0)
        right_jitter = right_jitter.to(device).unsqueeze(0).unsqueeze(0) if use_jitter else None

        (
            pred_left_frame, 
            pred_right_frame, 
            left_features, 
            right_features, 
            left_out_blending_mask, 
            right_out_blending_mask
        ) = model(
            left_inputs,
            right_inputs,
            left_motion_vectors, 
            right_motion_vectors,
            curr_frame_num, 
            left_jitter, 
            right_jitter,
            "evaluation"
        )

        pred_left_frame = pred_left_frame.squeeze(0).squeeze(0)
        pred_left_frame = linear_to_gamma(pred_left_frame)
        save_image(pred_left_frame, checkpoints_path / f"left_colour_invalid_{iterations}.png")

        pred_right_frame = pred_right_frame.squeeze(0).squeeze(0)
        pred_right_frame = linear_to_gamma(pred_right_frame)
        save_image(pred_right_frame, checkpoints_path / f"right_colour_invalid_{iterations}.png")
        
        # Blending mask when history is invalid
        left_out_blending_mask = F.pixel_shuffle(left_out_blending_mask, upscale_factor=scale_factor)
        save_image(left_out_blending_mask, checkpoints_path / f"left_blending_mask_invalid_{iterations}.png")

        right_out_blending_mask = F.pixel_shuffle(right_out_blending_mask, upscale_factor=scale_factor)
        save_image(right_out_blending_mask, checkpoints_path / f"right_blending_mask_invalid_{iterations}.png")

        # Output of the network when history is valid
        left_inputs_next = left_inputs_next.to(device).unsqueeze(0).unsqueeze(0)
        left_motion_vectors_next = left_motion_vectors_next.to(device).unsqueeze(0).unsqueeze(0)
        left_jitter_next = left_jitter_next.to(device).unsqueeze(0).unsqueeze(0) if use_jitter else None

        right_inputs_next = right_inputs_next.to(device).unsqueeze(0).unsqueeze(0)
        right_motion_vectors_next = right_motion_vectors_next.to(device).unsqueeze(0).unsqueeze(0)
        right_jitter_next = right_jitter_next.to(device).unsqueeze(0).unsqueeze(0) if use_jitter else None

        c0 = model.num_curr_colour + model.num_curr_depth + model.num_curr_jitter
        c1 = c0 + model.num_prev_colour
        left_inputs_next[:, :, c0:c1] = F.pixel_unshuffle(left_output.squeeze(0).squeeze(0), downscale_factor=scale_factor)
        right_inputs_next[:, :, c0:c1] = F.pixel_unshuffle(right_output.squeeze(0).squeeze(0), downscale_factor=scale_factor)

        c0 = c1
        c1 = c0 + model.num_prev_feature
        left_inputs_next[:, :, c0:c1] = left_features.squeeze(0)
        right_inputs_next[:, :, c0:c1] = right_features.squeeze(0)

        (
            pred_left_frame, 
            pred_right_frame, 
            _, 
            _, 
            left_out_blending_mask, 
            right_out_blending_mask
        ) = model(
            left_inputs_next, 
            right_inputs_next,
            left_motion_vectors_next, 
            right_motion_vectors_next,
            curr_frame_num_next, 
            left_jitter_next,
            right_jitter_next,
            "evaluation"
        )

        pred_left_frame = pred_left_frame.squeeze(0).squeeze(0)
        pred_left_frame = linear_to_gamma(pred_left_frame)
        save_image(pred_left_frame, checkpoints_path / f"left_colour_valid_{iterations}.png")

        pred_right_frame = pred_right_frame.squeeze(0).squeeze(0)
        pred_right_frame = linear_to_gamma(pred_right_frame)
        save_image(pred_right_frame, checkpoints_path / f"right_colour_valid_{iterations}.png")

        # Blending mask when history is valid
        left_out_blending_mask = F.pixel_shuffle(left_out_blending_mask.squeeze(0), upscale_factor=scale_factor)
        save_image(left_out_blending_mask, checkpoints_path / f"left_blending_mask_valid_{iterations}.png")

        right_out_blending_mask = F.pixel_shuffle(right_out_blending_mask.squeeze(0), upscale_factor=scale_factor)
        save_image(right_out_blending_mask, checkpoints_path / f"right_blending_mask_valid_{iterations}.png")


def save_input(
    sanity_checks_output_path: Path,
    model: nn.Module,
    inputs: torch.Tensor,
    motion_vectors: torch.Tensor,
    scale_factor: int
) -> None:
    from utils import linear_to_gamma
    
    c0 = 0
    c1 = c0 + model.num_curr_colour
    curr_colour = inputs[c0:c1]
    save_image(linear_to_gamma(curr_colour), sanity_checks_output_path / "curr_colour.png")

    c0 = c1
    c1 = c0 + model.num_curr_depth
    curr_depth = inputs[c0:c1]
    save_image(linear_to_gamma(curr_depth), sanity_checks_output_path / "curr_depth.png")

    if model.num_curr_jitter != 0:
        c0 = c1
        c1 = c0 + model.num_curr_jitter
        curr_jitter = inputs[c0:c1]
        zeros = torch.zeros(
            1,
            curr_jitter.shape[1],
            curr_jitter.shape[2],
            device=curr_jitter.device,
            dtype=curr_jitter.dtype
        )
        curr_jitter = torch.cat([curr_jitter, zeros], dim=0)
        save_image(curr_jitter, sanity_checks_output_path / "curr_jitter.png")

    c0 = c1
    c1 = c0 + model.num_prev_colour
    prev_colour = inputs[c0:c1]
    save_image(linear_to_gamma(F.pixel_shuffle(prev_colour, upscale_factor=scale_factor)), sanity_checks_output_path / "prev_colour.png")

    c0 = c1
    c1 = c0 + model.num_prev_feature
    prev_feature = inputs[c0:c1]
    save_image(F.pixel_shuffle(prev_feature, upscale_factor=scale_factor), sanity_checks_output_path / "prev_feature.png")
    
    motion_vectors = motion_vectors.squeeze(0)
    motion_vectors = torch.concat([
        torch.zeros((1, motion_vectors.shape[1], motion_vectors.shape[2])),
        motion_vectors
    ])
    save_image(linear_to_gamma(motion_vectors), sanity_checks_output_path / "motion_vectors.png")


def print_parameters(evaluation_output_path: Path, parameters: dict[str, Any]) -> None:
    with open(evaluation_output_path / "model_parameters.txt", "a") as a_writer:
        for layer in parameters:
            a_writer.write(f"\n----------------------{layer}----------------------\n")
            a_writer.write(str(parameters[layer]))
            a_writer.write("\n")


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
    mode: str = "training"
) -> None:
    model.eval()
    with torch.no_grad():
        # frames = [n for n in range(0, 7258) if (n + 1) % 30 != 0]
        # n = random.choice(frames)
        # print(f"{n=}")

        index = 0
        if mode == "training":
            (inputs, motion_vectors, jitter, output, curr_frame_num) = data[(index, 0, 0, input_frame_width, input_frame_height)]
            inputs_next, motion_vectors_next, jitter_next, output_next, curr_frame_num_next = data[(index + 1, 0, 0, input_frame_width, input_frame_height)]
        else:
            inputs, motion_vectors, jitter, output, curr_frame_num = data[index]
            inputs_next, motion_vectors_next, jitter_next, output_next, curr_frame_num_next = data[index + 1]

        if curr_frame_num == 0:
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
