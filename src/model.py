import torch
from torch import nn


class QualcommNetwork(nn.Module):
    def __init__(
        self,
        num_prev_feature_channels,
        hidden_channels,
        num_blocks,
        upscale_factor,
    ):
        """
        Simplified implementation of the Qualcomm network, adapted for DLAA.
        """
        super().__init__()

        self.num_curr_colour_channels = 3
        self.num_curr_depth_channels = 1
        self.num_curr_jitter_channels = 2  # 2 for displacement in both x and y
        self.num_prev_colour_channels = self.num_curr_colour_channels
        self.num_prev_feature_channels = num_prev_feature_channels
        self.upscale_factor = upscale_factor

        # * 2 to include the previous frame ground truth
        in_channels = (
            self.num_curr_colour_channels +
            self.num_curr_depth_channels +
            self.num_curr_jitter_channels +
            self.num_prev_colour_channels +
            self.num_prev_feature_channels
        )

        # Initial 3×3 Conv + ReLU block
        self.input_conv = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.input_relu = nn.ReLU(inplace=True)

        # m × (3×3 Conv + ReLU) blocks
        body_layers = []
        for _ in range(num_blocks):
            body_layers.append(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                )
            )
            body_layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*body_layers)

        # Feature head (low-res to packed for pixel shuffle)
        self.feature_head = nn.Conv2d(
            hidden_channels,
            num_prev_feature_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )

        # Colour head
        self.colour_head = nn.Conv2d(
            hidden_channels,
            self.num_curr_colour_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )

        # Blending mask head
        self.blending_mask_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                1 * (upscale_factor ** 2),
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.Sigmoid(),
        )

        # Depth-to-space operation (identity when upscale_factor == 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Split inputs (mainly to isolate prev_warped_colour for blending)
        c0 = 0
        c1 = c0 + self.num_curr_colour_channels
        curr_colour = x[:, c0:c1]

        c0 = c1
        c1 = c0 + self.num_curr_depth_channels
        curr_depth = x[:, c0:c1]

        c0 = c1
        c1 = c0 + self.num_curr_jitter_channels
        curr_jitter = x[:, c0:c1]

        c0 = c1
        c1 = c0 + self.num_prev_colour_channels
        prev_warped_colour = x[:, c0:c1]  # To be used in the blend block

        prev_warped_features = x[:, c1:]  # The remaining channels

        # Concatenate everything as input to the conv stack
        net_input = torch.cat(
            [
                curr_colour,
                curr_depth,
                curr_jitter,
                prev_warped_colour,
                prev_warped_features,
            ],
            dim=1,
        )

        # Main conv stack
        h = self.input_relu(self.input_conv(net_input))
        h = self.body(h)

        # Feature branch
        out_features = self.feature_head(h)
        up_out_features = self.pixel_shuffle(out_features)

        # Color + mask branches
        out_colour = self.colour_head(h)
        out_blending_mask = self.blending_mask_head(h)

        up_out_colour = self.pixel_shuffle(out_colour)
        up_out_blending_mask = self.pixel_shuffle(out_blending_mask)

        blended_color = up_out_blending_mask * up_out_colour + (1.0 - up_out_blending_mask) * prev_warped_colour

        return blended_color
