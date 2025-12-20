import torch
from torch import nn


class QualcommNetwork(nn.Module):
    def __init__(self, hidden_channels: int, num_blocks: int):
        """
        Simplified implementation of the Qualcomm network, adapted for DLAA.
        """
        super().__init__()

        self.num_curr_colour = 3
        self.num_curr_depth = 1
        self.num_curr_jitter = 2  # 2 for displacement in both x and y
        self.num_prev_colour = self.num_curr_colour
        self.num_prev_feature = 1

        # * 2 to include the previous frame ground truth
        self.in_channels = (
            self.num_curr_colour +
            self.num_curr_depth +
            self.num_curr_jitter +
            self.num_prev_colour +
            self.num_prev_feature
        )

        # Initial 3 × 3 Conv + ReLU block
        self.input_conv = nn.Conv2d(
            self.in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect"
        )
        self.input_relu = nn.ReLU()

        # num_blocks × (3 × 3 Conv + ReLU) blocks
        body_layers = []
        for _ in range(num_blocks):
            body_layers.append(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect"
                )
            )
            body_layers.append(nn.ReLU())
        self.body = nn.Sequential(*body_layers)

        # Feature head
        self.feature_head = nn.Conv2d(
            hidden_channels,
            self.num_prev_feature,
            kernel_size=3,
            padding=1,
            padding_mode="reflect"
        )

        # Colour head
        self.colour_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                self.num_curr_colour,
                kernel_size=3,
                padding=1,
                padding_mode="reflect"
            ),
            nn.ReLU()
        )

        # Blending mask head
        self.blending_mask_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=3,
                padding=1,
                padding_mode="reflect"
            ),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Split inputs (mainly to isolate prev_colour for blending)
        c0 = 0
        c1 = c0 + self.num_curr_colour
        _ = x[:, c0:c1]

        c0 = c1
        c1 = c0 + self.num_curr_depth
        _ = x[:, c0:c1]

        c0 = c1
        c1 = c0 + self.num_curr_jitter
        _ = x[:, c0:c1]

        c0 = c1
        c1 = c0 + self.num_curr_colour
        prev_colour = x[:, c0:c1]

        c0 = c1
        c1 = c0 + self.num_prev_feature
        _ = x[:, c0:c1]

        # Main conv stack
        h = self.input_relu(self.input_conv(x))
        h = self.body(h)

        # Feature branch
        out_features = self.feature_head(h)

        # Colour + blending mask branches
        out_colour = self.colour_head(h)
        out_blending_mask = self.blending_mask_head(h)

        blended_colour = out_blending_mask * out_colour + (1.0 - out_blending_mask) * prev_colour

        return blended_colour, out_features
