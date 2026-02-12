from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from typing import Any

from utils import linear_to_gamma


def save_input(
    sanity_checks_output_path: Path,
    model: nn.Module,
    inputs: torch.Tensor,
    motion_vectors: torch.Tensor,
    scale_factor: int
) -> None:
    c0 = 0
    c1 = c0 + model.num_curr_colour
    curr_colour = inputs[c0:c1]
    save_image(linear_to_gamma(curr_colour), sanity_checks_output_path / "curr_colour.png")

    c0 = c1
    c1 = c0 + model.num_curr_depth
    curr_depth = inputs[c0:c1]
    save_image(curr_depth, sanity_checks_output_path / "curr_depth.png")

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
    save_image(motion_vectors, sanity_checks_output_path / "motion_vectors.png")


def save_output(
    sanity_checks_output_path: Path,
    output: torch.Tensor
) -> None:
    save_image(output, sanity_checks_output_path / "ground_truth.png")


def print_parameters(evaluation_output_path: Path, parameters: dict[str, Any]) -> None:
    with open(evaluation_output_path / "model_parameters.txt", "a") as a_writer:
        for layer in parameters:
            a_writer.write(f"\n----------------------{layer}----------------------\n")
            a_writer.write(str(parameters[layer]))
            a_writer.write("\n")
