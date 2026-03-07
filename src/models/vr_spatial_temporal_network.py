import torch
from torch import nn
import torch.nn.functional as F

from .qualcomm_network import JitterConditionedConv, QualcommNetwork
from utils import VRConfig


class VRSpatialTemporalNetwork(QualcommNetwork):
    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        vr_config: VRConfig,
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

    def predict_clip_frames(
        self,
        curr_left_colour: torch.Tensor, 
        curr_right_colour: torch.Tensor,
        curr_depth: torch.Tensor,
        curr_motion_vectors: torch.Tensor,
        curr_jitter: torch.Tensor,
        prev_left_colour: torch.Tensor,
        prev_right_colour: torch.Tensor,
        prev_left_feature: torch.Tensor,
        prev_right_feature: torch.Tensor,
        prev_depth: torch.Tensor,
        jitter_frames: torch.Tensor,
        eye: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        if eye == "left":
            # Current right frame warped onto the current left frame
            curr_warped_right_colour = VRConfig.right_to_left_warp(
                curr_right_colour, 
                curr_depth, 
                self.vr_config.camera_baseline, 
                self.vr_config.focal_length
            )

            # The previous left frame warped temporally onto the current left frame
            prev_warped_left_colour = self.warp(
                prev_left_colour,
                curr_motion_vectors
            )

            # The previous left feature frame warped temporally onto the current left feature frame
            prev_warped_left_feature = self.warp(
                prev_left_feature,
                curr_motion_vectors
            )

            # The previous right frame warped onto the previous left frame, and then warped temporally onto the current left frame
            prev_warped_right_colour = self.warp(
                VRConfig.right_to_left_warp(
                    prev_right_colour, 
                    prev_depth, 
                    self.vr_config.camera_baseline, 
                    self.vr_config.focal_length
                ),
                curr_motion_vectors
            )

            # The previous right feature frame warped onto the previous left feature frame, and then warped temporally onto the current left feature frame
            prev_warped_right_feature = self.warp(
                VRConfig.right_to_left_warp(
                    prev_right_feature, 
                    prev_depth, 
                    self.vr_config.camera_baseline, 
                    self.vr_config.focal_length
                ),
                curr_motion_vectors
            )

            # (B, C, H, W)
            inputs = torch.stack([
                curr_left_colour,          # C = 3
                curr_warped_right_colour,  # C = 3
                curr_depth,                # C = 1
                curr_jitter,               # C = 2
                prev_warped_left_colour,   # C = 12
                prev_warped_right_colour,  # C = 12
                prev_warped_left_feature,  # C = 12
                prev_warped_right_feature  # C = 12
            ], dim=1)

        else:
            # Current left frame warped onto the current right frame
            curr_warped_left_colour = VRConfig.left_to_right_warp(
                curr_left_colour,
                curr_depth,
                self.vr_config.camera_baseline, 
                self.vr_config.focal_length
            )

            # The previous right frame warped temporally onto the current right frame
            prev_warped_right_colour = self.warp(
                prev_right_colour,
                curr_motion_vectors
            )

            # The previous right feature frame warped temporally onto the current right feature frame
            prev_warped_right_feature = self.warp(
                prev_right_feature,
                curr_motion_vectors
            )

            # The previous left frame warped onto the previous right frame, and then warped temporally onto the current right frame
            prev_warped_left_colour = self.warp(
                VRConfig.right_to_left_warp(
                    prev_left_colour, 
                    prev_depth, 
                    self.vr_config.camera_baseline, 
                    self.vr_config.focal_length
                ),
                curr_motion_vectors
            )

            # The previous left feature frame warped onto the previous right feature frame, and then warped temporally onto the current left right feature frame
            prev_warped_left_feature = self.warp(
                VRConfig.right_to_left_warp(
                    prev_left_feature, 
                    prev_depth, 
                    self.vr_config.camera_baseline, 
                    self.vr_config.focal_length
                ),
                curr_motion_vectors
            )

            # (B, C, H, W)
            inputs = torch.stack([
                curr_warped_left_colour,    # C = 3
                curr_right_colour,          # C = 3
                curr_depth,                 # C = 1
                curr_jitter,                # C = 2
                prev_warped_left_colour,    # C = 12
                prev_warped_right_colour,   # C = 12
                prev_warped_left_feature,   # C = 12
                prev_warped_right_feature,  # C = 12
            ], dim=1)

        B, C, H, W = inputs.shape

        # ------------------------------------------------------------
        # ---------------- Input convolution and ReLU ----------------
        # ------------------------------------------------------------
        if self.num_curr_jitter != 0:
            h = self.input_conv(inputs, jitter_frames)
        else:
            h = self.input_conv(inputs)

        h = self.input_relu(h)

        # ------------------------------------------------------------
        # ---------------------- Main conv body ----------------------
        # ------------------------------------------------------------
        h = self.body(h)

        # ------------------------------------------------------------
        # -------- Feature, colour, and blending mask branches -------
        # ------------------------------------------------------------
        if self.num_curr_jitter != 0:
            out_features = self.feature_head(h, jitter_frames)
            out_colour = self.colour_head(h, jitter_frames)
            out_blending_mask = self.blending_mask_head(h, jitter_frames)
        else:
            out_features = self.feature_head(h)
            out_colour = self.colour_head(h)
            out_blending_mask = self.blending_mask_head(h)

        out_colour = self.colour_head_relu(out_colour)
        # left_out_colour = torch.clamp(left_out_colour, min=0, max=1)

        out_blending_mask = self.blending_mask_softmax(out_blending_mask)

        # ------------------------------------------------------------
        # -------------------------- Blend ---------------------------
        # ------------------------------------------------------------
        # (B, 12, H, W) --> (B, 3, 4, H, W)
        out_colour = out_colour.view(B, 3, -1, H, W)
        prev_warped_left_colour = prev_warped_left_colour.view(B, 3, -1, H, W)
        prev_warped_right_colour = prev_warped_right_colour.view(B, 3, -1, H, W)

        # (B, 12, H, W) --> (B, 3, 4, H, W)
        out_blending_mask = out_blending_mask.view(B, 3, -1, H, W)
        blended_colour = (
            out_colour * out_blending_mask[:, 0:1, ...] +
            prev_warped_left_colour * out_blending_mask[:, 1:2, ...] +
            prev_warped_right_colour * out_blending_mask[:, 2:3, ...]
        )
        blended_colour = torch.clamp(blended_colour, min=0.0, max=1.0)
        blended_colour = blended_colour.view(B, -1, H, W)

        return blended_colour, out_features, out_blending_mask

    def forward(
        self,
        left_inputs: torch.Tensor,
        right_inputs: torch.Tensor,
        left_motion_vectors: torch.Tensor,
        right_motion_vectors: torch.Tensor,
        prev_left_depth: torch.Tensor,
        prev_right_depth: torch.Tensor,
        curr_frame_num: int,
        jitter: torch.Tensor = None,
        mode: str = "training"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, clip_size, C, H, W = left_inputs.shape

        # To hold the recurrent colour frame and features
        prev_pred_left_colour, prev_pred_right_colour, prev_pred_left_features, prev_pred_right_features = None

        left_outputs = []
        right_outputs = []
        for clip in range(clip_size):
            left_clip_frames = left_inputs[:, clip].clone()
            left_motion_vector_frames = left_motion_vectors[:, clip]

            right_clip_frames = right_inputs[:, clip].clone()
            right_motion_vector_frames = right_motion_vectors[:, clip]

            if self.num_curr_jitter != 0:
                jitter_frames = jitter[:, clip]

            c0 = 0
            c1 = c0 + self.num_curr_left_colour
            curr_left_colour = left_clip_frames[:, c0:c1]
            curr_right_colour = right_clip_frames[:, c0:c1]

            c0 = c1
            c1 = c0 + self.num_curr_depth
            curr_left_depth = left_clip_frames[:, c0:c1]
            curr_right_depth = right_clip_frames[:, c0:c1]

            c0 = c1
            c1 = c0 + self.num_curr_jitter
            curr_jitter = left_clip_frames[:, c0:c1]

            c0 = c1
            c1 = c0 + self.num_prev_left_colour
            prev_left_colour = prev_pred_left_colour if prev_pred_left_colour is not None else left_clip_frames[:, c0:c1] 
            prev_right_colour = prev_pred_right_colour if prev_pred_right_colour is not None else right_clip_frames[:, c0:c1]
            
            c0 = c1
            c1 = c0 + self.num_prev_left_colour
            prev_left_feature = prev_pred_left_features if prev_pred_left_features is not None else left_clip_frames[:, c0:c1]
            prev_right_feature = prev_pred_right_features if prev_pred_right_features is not None else right_clip_frames[:, c0:c1]
            
            # ------------------------------------------------------------
            # ------------------------- Left eye -------------------------
            # ------------------------------------------------------------
            left_blended_colour, left_out_features, left_out_blending_mask = self.predict_clip_frames(
                curr_left_colour=curr_left_colour,
                curr_right_colour=curr_right_colour,
                curr_depth=curr_left_depth,
                curr_motion_vectors=left_motion_vector_frames,
                curr_jitter=curr_jitter,
                prev_left_colour=prev_left_colour,
                prev_right_colour=prev_right_colour,
                prev_left_feature=prev_left_feature,
                prev_right_feature=prev_right_feature,
                prev_depth=prev_left_depth,
                jitter_frames=jitter_frames,
                eye="left"
            )

            # ------------------------------------------------------------
            # ------------------------- Right eye ------------------------
            # ------------------------------------------------------------
            right_blended_colour, right_out_features, right_out_blending_mask = self.predict_clip_frames(
                curr_left_colour=curr_left_colour,
                curr_right_colour=curr_right_colour,
                curr_depth=curr_right_depth,
                curr_motion_vectors=right_motion_vector_frames,
                curr_jitter=curr_jitter,
                prev_left_colour=prev_left_colour,
                prev_right_colour=prev_right_colour,
                prev_left_feature=prev_left_feature,
                prev_right_feature=prev_right_feature,
                prev_depth=prev_right_depth,
                jitter_frames=jitter_frames,
                eye="right"
            )

            # ------------------------------------------------------------
            # ---------------------- Depth to space ----------------------
            # ------------------------------------------------------------
            left_full_res_colour = self.depth_to_space(left_blended_colour)  # (B, 3, H, W)
            left_outputs.append(left_full_res_colour)

            right_full_res_colour = self.depth_to_space(right_blended_colour)  # (B, 3, H, W)
            right_outputs.append(right_full_res_colour)

            # Save recurrent colour frame and features
            prev_pred_left_colour = left_blended_colour
            prev_pred_left_features = left_out_features

            prev_pred_right_colour = right_blended_colour
            prev_right_feature = right_out_features

        # prev_pred_features is only used by evaluation
        # out_blending_mask is only used during evaluation for inspection
        return (
            torch.stack(left_outputs, dim=1), 
            torch.stack(right_outputs, dim=1), 
            prev_pred_left_features, 
            prev_pred_right_features,
            left_out_blending_mask.view(B, -1, H, W),
            right_out_blending_mask.view(B, -1, H, W),
        )
