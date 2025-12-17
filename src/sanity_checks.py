import torch
from torch import nn
from torchvision.utils import save_image


def output_input(model: nn.Module, input_imgs: torch.Tensor) -> None:
    c0 = 0
    c1 = c0 + model.num_curr_colour_channels
    curr_colour = input_imgs[:, c0:c1]

    c0 = c1
    c1 = c0 + model.num_curr_depth_channels
    curr_depth = input_imgs[:, c0:c1]

    c0 = c1
    c1 = c0 + model.num_curr_jitter_channels
    curr_jitter = input_imgs[:, c0:c1]

    c0 = c1
    c1 = c0 + model.num_curr_colour_channels
    prev_colour = input_imgs[:, c0:c1]

    c0 = c1
    c1 = c0 + model.num_curr_depth_channels
    prev_depth = input_imgs[:, c0:c1]

    c0 = c1
    c1 = c0 + model.num_curr_jitter_channels
    prev_jitter = input_imgs[:, c0:c1]

    save_image(curr_colour, "sanity_checks_output/curr_colour.png")
    save_image(curr_depth, "sanity_checks_output/curr_depth.png")
    save_image(curr_jitter, "sanity_checks_output/curr_jitter.png")
    save_image(prev_colour, "sanity_checks_output/prev_colour.png")
    save_image(prev_depth, "sanity_checks_output/prev_depth.png")
    save_image(prev_jitter, "sanity_checks_output/prev_jitter.png")
