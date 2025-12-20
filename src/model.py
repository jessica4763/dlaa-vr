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
        batch_size, clip_size, C, H, W = x.shape

        # To hold the recurrent colour frame and features
        prev_pred_colour = prev_pred_features = None

        outputs = []
        for clip in range(clip_size):
            clip_frames = x[:, clip].clone()

            # Use recurrent colour frame
            c0 = self.num_curr_colour + self.num_curr_depth + self.num_curr_jitter
            c1 = c0 + self.num_prev_colour
            if prev_pred_colour is not None:
                clip_frames[:, c0:c1] = prev_pred_colour
            prev_colour = clip_frames[:, c0:c1]

            # Use recurrent features
            c0 = c1
            c1 = c0 + self.num_prev_feature
            if prev_pred_features is not None:
                clip_frames[:, c0:c1] = prev_pred_features

            # ------------------------------------------------------------
            # ---------------------- Main conv stack ---------------------
            # ------------------------------------------------------------
            h = self.input_relu(self.input_conv(clip_frames))
            h = self.body(h)

            # ------------------------------------------------------------
            # ---------------------- Feature branch ----------------------
            # ------------------------------------------------------------
            out_features = self.feature_head(h)

            # ------------------------------------------------------------
            # ------------ Colour and blending mask branches -------------
            # ------------------------------------------------------------
            out_colour = self.colour_head(h)
            out_blending_mask = self.blending_mask_head(h)

            # ------------------------------------------------------------
            # -------------------------- Blend ---------------------------
            # ------------------------------------------------------------
            blended_colour = out_blending_mask * out_colour + (1.0 - out_blending_mask) * prev_colour
            outputs.append(blended_colour)

            # Save recurrent colour frame and features
            prev_pred_colour = blended_colour
            prev_pred_features = out_features

        return torch.stack(outputs, dim=1), prev_pred_features
