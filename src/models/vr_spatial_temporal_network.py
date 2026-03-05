import torch
from torch import nn
import torch.nn.functional as F

from qualcomm_network import JitterConditionedConv, QualcommNetwork, kaiming_init_params
from utils import VRConfig


class VRSpatialTemporalNetwork(QualcommNetwork):
    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        scale_factor: int = 1,
        use_jitter: bool = False
    ) -> None:
        nn.Module.__init__(self)

        self.num_curr_left_colour = self.num_curr_right_colour = 3
        self.num_curr_depth = 1
        self.num_curr_jitter = 2 if use_jitter else 0  # 2 for displacement in both x and y
        self.num_prev_left_colour = self.num_prev_right_colour = 3 * (scale_factor ** 2)
        self.num_prev_left_feature = self.num_prev_right_feature = 1 * (scale_factor ** 2)

        self.in_channels = (
            self.num_curr_left_colour +
            self.num_curr_right_colour +
            self.num_curr_depth +
            self.num_curr_jitter +
            self.num_prev_left_colour +
            self.num_prev_right_colour + 
            self.num_prev_left_feature + 
            self.num_prev_right_feature
        )

        self.depth_to_space = nn.PixelShuffle(upscale_factor=scale_factor)
        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=scale_factor)

        if use_jitter:
            self.input_conv = JitterConditionedConv(
                out_channels=hidden_channels,
                in_channels=self.in_channels,
                kernel_height=3,
                kernel_width=3,
                num_hidden_features=2048,
                num_blocks=7
            )
        else:
            # Initial 3 × 3 Conv + ReLU block
            self.input_conv = nn.Conv2d(
                self.in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode="zeros"
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
                    padding_mode="zeros"
                )
            )
            body_layers.append(nn.ReLU())
        self.body = nn.Sequential(*body_layers)

        # Feature head
        if use_jitter:
            self.feature_head = JitterConditionedConv(
                self.num_prev_feature,
                hidden_channels,
                3,
                3,
                num_hidden_features=2048,
                num_blocks=7
            )
        else:
            self.feature_head = nn.Conv2d(
                hidden_channels,
                self.num_prev_feature,
                kernel_size=3,
                padding=1,
                padding_mode="zeros"
            )

        # Colour head
        if use_jitter:
            self.colour_head = JitterConditionedConv(
                self.num_prev_colour,
                hidden_channels,
                3,
                3,
                num_hidden_features=2048,
                num_blocks=7
            )
        else:
            self.colour_head = nn.Conv2d(
                hidden_channels,
                self.num_prev_colour,
                kernel_size=3,
                padding=1,
                padding_mode="zeros"
            )
        self.colour_head_relu = nn.ReLU()

        # Blending mask head
        if use_jitter:
            self.blending_mask_head = JitterConditionedConv(
                3 * (scale_factor ** 2),
                hidden_channels,
                3,
                3,
                num_hidden_features=2048,
                num_blocks=7
            )
        else:
            self.blending_mask_head = nn.Conv2d(
                hidden_channels,
                3 * (scale_factor ** 2),
                kernel_size=3,
                padding=1,
                padding_mode="zeros"
            )

        # Softmax in place of sigmoid to blend three inputs together
        # Compute softmax along the channel dimension 
        self.blending_mask_softmax = nn.Softmax(dim=1)  # (B, C, H, W)

        self.apply(self.kaiming_init_params)

    def forward(
        self,
        left_inputs: torch.Tensor,
        right_inputs: torch.Tensor,
        left_motion_vectors: torch.Tensor,
        right_motion_vectors: torch.Tensor,
        curr_frame_num: int,
        left_jitter: torch.Tensor = None,
        right_jitter: torch.Tensor = None,
        mode: str = "training"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, clip_size, C, H, W = left_inputs.shape

        # To hold the recurrent colour frame and features
        prev_pred_left_frame = prev_pred_right_frame, prev_left_features, prev_right_features = None

        outputs = []
        for clip in range(clip_size):
            left_clip_frames = left_inputs[:, clip].clone()
            left_motion_vector_frames = left_motion_vectors[:, clip]

            right_clip_frames = right_inputs[:, clip].clone()
            right_motion_vector_frames = right_motion_vectors[:, clip]

            if self.num_curr_jitter != 0:
                left_jitter_frames = left_jitter[:, clip]
                right_jitter_frames = right_jitter[:, clip]

            # Use recurrent colour frame
            c0 = self.num_curr_colour + self.num_curr_depth + self.num_curr_jitter
            c1 = c0 + self.num_prev_colour
            if mode == "training":
                if prev_pred_left_frame is not None:
                    # Warp recurrent colour frame
                    left_clip_frames[:, c0:c1] = self.warp(
                        prev_pred_left_frame,
                        left_motion_vector_frames
                    )

                    right_clip_frames[:, c0:c1] = self.warp(
                        VRConfig.right_to_left_warp(prev_pred_right_frame),
                        right_motion_vector_frames
                    )
            else:
                if curr_frame_num > 0:
                    left_clip_frames[:, c0:c1] = self.warp(
                        left_clip_frames[:, c0:c1],
                        left_motion_vector_frames
                    )

                    right_clip_frames[:, c0:c1] = self.warp(
                        VRConfig.right_to_left_warp(right_clip_frames[:, c0:c1]),
                        right_motion_vector_frames
                    )

            prev_left_frame = left_clip_frames[:, c0:c1]  # Save for the blend step
            prev_right_frame = right_clip_frames[:, c0:c1]  # Save for the blend step

            # Use recurrent features
            c0 = c1
            c1 = c0 + self.num_prev_feature
            if mode == "training":
                if prev_left_features is not None:
                    # Warp recurrent features
                    left_clip_frames[:, c0:c1] = self.warp(
                        prev_left_features,
                        left_motion_vector_frames
                    )

                    right_clip_frames[:, c0:c1] = self.warp(
                        VRConfig.right_to_left_warp(prev_right_features),
                        right_motion_vector_frames
                    )
            else:
                if curr_frame_num > 0:
                    left_clip_frames[:, c0:c1] = self.warp(
                        left_clip_frames[:, c0:c1],
                        left_motion_vector_frames
                    )

                    right_clip_frames[:, c0:c1] = self.warp(
                        VRConfig.right_to_left_warp(right_clip_frames[:, c0:c1]),
                        right_motion_vector_frames
                    )

            # ------------------------------------------------------------
            # ---------------- Input convolution and ReLU ----------------
            # ------------------------------------------------------------
            if self.num_curr_jitter != 0:
                left = self.input_conv(left_clip_frames, left_jitter_frames)
            else:
                left = self.input_conv(left_clip_frames)

            left = self.input_relu(left)

            # ------------------------------------------------------------
            # ---------------------- Main conv body ----------------------
            # ------------------------------------------------------------
            left = self.body(left)

            # ------------------------------------------------------------
            # -------- Feature, colour, and blending mask branches -------
            # ------------------------------------------------------------
            if self.num_curr_jitter != 0:
                left_out_features = self.feature_head(left, left_jitter_frames)
                left_out_colour = self.colour_head(left, left_jitter_frames)
                left_out_blending_mask = self.blending_mask_head(left, left_jitter_frames)
            else:
                left_out_features = self.feature_head(left)
                left_out_colour = self.colour_head(left)
                left_out_blending_mask = self.blending_mask_head(left)

            left_out_colour = self.colour_head_relu(left_out_colour)
            # left_out_colour = torch.clamp(left_out_colour, min=0, max=1)

            left_out_blending_mask = self.blending_mask_softmax(left_out_blending_mask)

            # ------------------------------------------------------------
            # -------------------------- Blend ---------------------------
            # ------------------------------------------------------------
            # (B, 12, H, W)
            left_out_colour = left_out_colour.view(B, 3, -1, H, W)
            prev_left_frame = prev_left_frame.view(B, 3, -1, H, W)

            left_out_blending_mask = left_out_blending_mask.view(B, 1, -1, H, W)

            left_blended_colour = left_out_blending_mask * left_out_colour + (1.0 - left_out_blending_mask) * prev_left_frame
            left_blended_colour = torch.clamp(left_blended_colour, min=0.0, max=1.0)
            left_blended_colour = left_blended_colour.view(B, -1, H, W)

            # ------------------------------------------------------------
            # ---------------------- Depth to space ----------------------
            # ------------------------------------------------------------
            full_res_colour = self.depth_to_space(blended_colour)  # (B, 3, H, W)
            outputs.append(full_res_colour)

            # Save recurrent colour frame and features
            prev_pred_colour = blended_colour
            prev_pred_features = out_features

        # prev_pred_features is only used by evaluation
        # out_blending_mask is only used during evaluation for inspection
        return torch.stack(outputs, dim=1), prev_pred_features, out_blending_mask.view(B, -1, H, W)

    

