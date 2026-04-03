import torch
from torch import nn
import torch.nn.functional as F

from .qualcomm_network import JitterConditionedConv, QualcommNetwork
from utils import VRConfig


class VRNetwork(QualcommNetwork):
    def __init__(
        self,
        vr_config: VRConfig,
        hidden_channels: int,
        num_blocks: int,
        scale_factor: int = 1,
        use_jitter: bool = False
    ) -> None:
        nn.Module.__init__(self)
        
        self.vr_config = vr_config
        self.scale_factor = scale_factor

        self.num_curr_colour = 3
        self.num_curr_depth = 1
        self.num_curr_jitter = 2 if use_jitter else 0  # 2 for displacement in both x and y
        self.num_prev_colour = 3 * (scale_factor ** 2)
        self.num_prev_feature = 1 * (scale_factor ** 2)

        self.input_channels = (
            self.num_curr_colour +
            self.num_curr_depth +
            self.num_curr_jitter +
            self.num_prev_colour +
            self.num_prev_feature
        )

        self.total_channels = (
            self.num_curr_colour +
            self.num_curr_colour +
            self.num_curr_depth +
            self.num_curr_jitter +
            self.num_prev_colour +
            self.num_prev_colour + 
            self.num_prev_feature + 
            self.num_prev_feature
        )

        self.depth_to_space = nn.PixelShuffle(upscale_factor=scale_factor)
        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=scale_factor)

        if use_jitter:
            self.input_conv = JitterConditionedConv(
                output_channels=hidden_channels,
                input_channels=self.total_channels,
                kernel_height=3,
                kernel_width=3,
                num_hidden_features=2048,
                num_blocks=7
            )
        else:
            # Initial 3 × 3 Conv + ReLU block
            self.input_conv = nn.Conv2d(
                self.total_channels,
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
        self.blending_mask_softmax = nn.Softmax(dim=1)  # (B, 3, 4, H, W)

        self.apply(self.kaiming_init_params)

    def warp(
        self,
        input_tensor: torch.Tensor,
        motion_vectors: torch.Tensor
    ) -> torch.Tensor:
        _, _, input_tensor_H, _ = input_tensor.shape
        _, _, motion_vectors_H, _ = motion_vectors.shape

        if input_tensor_H != motion_vectors_H:
            # Depth to space: (B, 12, 132, 132) -> (B, 3, 264, 264)
            input_tensor = self.depth_to_space(input_tensor)

        H = motion_vectors.shape[2]
        W = motion_vectors.shape[3]

        # (B, 2, H, W) --> (B, H, W, 2)
        motion_vectors = torch.permute(motion_vectors, (0, 2, 3, 1))

        # Once motion_vectors is added to base_grid, each location
        # in the grid contains the absolute coordinates of the previous
        # pixel/feature after motion compensation. There is no need to
        # normalise the motion vectors because they are stored in the [-1, 1] range
        y, x = torch.meshgrid(
            torch.linspace(-1 + (1 / H), 1 - (1 / H), H),
            torch.linspace(-1 + (1 / W), 1 - (1 / W), W),
            indexing="ij"
        )
        base_grid = torch.stack((x, y), dim=-1).unsqueeze(0).to(motion_vectors.device)
        warped_grid = base_grid - motion_vectors * 2.0  # base_grid is broadcasted

        warped_input_tensor = F.grid_sample(
            input_tensor,
            warped_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        if input_tensor_H != motion_vectors_H:
            warped_input_tensor = self.space_to_depth(warped_input_tensor)

        return warped_input_tensor

    def left_to_right_warp(
        self,
        left_frame: torch.Tensor,
        right_depth: torch.Tensor,
        camera_baseline: float,
        focal_length: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, left_frame_H, _ = left_frame.shape
        _, _, right_depth_H, _ = right_depth.shape

        if left_frame_H != right_depth_H:
            # Depth to space: (B, 12, 132, 132) -> (B, 3, 264, 264)
            left_frame = self.depth_to_space(left_frame)
            focal_length *= self.scale_factor

        B, _, H, W = left_frame.shape

        ys, xs = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing="ij"
        )
        ys = ys.to(right_depth.device) + 0.5 
        xs = xs.to(right_depth.device) + 0.5

        disparity = (camera_baseline * focal_length) / ((right_depth * 99990.0) + 10.0)
        warped_xs = xs + disparity  # Note xs is broadcast here
        warped_xs = torch.permute(warped_xs, dims=(0, 2, 3, 1))  # (B, C, H, W) --> (B, H, W, C)
        warped_xs = 2.0 * (warped_xs / W) - 1.0  # Normalise to range [-1, 1]. Divide by W rather than W - 1 because we set align_corners=False
        ys = ys.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        ys = torch.permute(ys, dims=(0, 2, 3, 1))  # (B, C, H, W) --> (B, H, W, C)
        ys = 2.0 * (ys / H) - 1.0  # Normalise to range [-1, 1]. Divide by H rather than H - 1 because we set align_corners=False
        warped_grid = torch.cat((warped_xs, ys), dim=-1)

        # Warping the left frame on to the right frame
        warped_left_frame = F.grid_sample(
            left_frame,
            warped_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        if left_frame_H != right_depth_H:
            # Space to depth: (B, 3, 132, 132) -> (B, 12, 264, 264)
            warped_left_frame = self.space_to_depth(warped_left_frame)

        valid_mask = None

        return warped_left_frame, valid_mask

    def right_to_left_warp(
        self,
        right_frame: torch.Tensor,
        left_depth: torch.Tensor,
        camera_baseline: float,
        focal_length: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, right_frame_H, _ = right_frame.shape
        _, _, left_depth_H, _ = left_depth.shape

        if right_frame_H != left_depth_H:
            # Depth to space: (B, 12, 132, 132) -> (B, 3, 264, 264)
            right_frame = self.depth_to_space(right_frame)
            focal_length *= self.scale_factor

        B, _, H, W = right_frame.shape

        ys, xs = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing="ij"
        )
        ys = ys.to(left_depth.device) + 0.5 
        xs = xs.to(left_depth.device) + 0.5

        disparity = (camera_baseline * focal_length) / ((left_depth * 99990.0) + 10.0) 
        warped_xs = xs - disparity  # Note xs is broadcast here
        warped_xs = torch.permute(warped_xs, dims=(0, 2, 3, 1))  # (B, C, H, W) --> (B, H, W, C)
        warped_xs = 2.0 * (warped_xs / W) - 1.0  # Normalise to range [-1, 1]. Divide by W rather than W - 1 because we set align_corners=False
        ys = ys.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        ys = torch.permute(ys, dims=(0, 2, 3, 1))  # (B, C, H, W) --> (B, H, W, C)
        ys = 2.0 * (ys / H) - 1.0  # Normalise to range [-1, 1]. Divide by H rather than H - 1 because we set align_corners=False
        warped_grid = torch.cat((warped_xs, ys), dim=-1)

        # Warping the right frame on to the left frame
        warped_right_frame = F.grid_sample(
            right_frame,
            warped_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        if right_frame_H != left_depth_H:
            # Space to depth: (B, 3, 132, 132) -> (B, 12, 264, 264)
            warped_right_frame = self.space_to_depth(warped_right_frame)

        valid_mask = None

        return warped_right_frame, valid_mask

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
        temporal_warp: int,
        eye: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        if eye == "left":
            # Current right frame warped onto the current left frame
            curr_warped_right_colour, valid_mask = self.right_to_left_warp(
                curr_right_colour, 
                curr_depth,  # left depth
                self.vr_config.camera_baseline, 
                self.vr_config.focal_length
            )

            # The previous left frame warped temporally onto the current left frame
            if temporal_warp:
                prev_left_colour = self.warp(
                    prev_left_colour,
                    curr_motion_vectors  # left motion vectors
                )

            # The previous left feature frame warped temporally onto the current left feature frame
            if temporal_warp:
                prev_left_feature = self.warp(
                    prev_left_feature,
                    curr_motion_vectors  # left motion vectors
                )

            # The previous right frame warped onto the previous left frame, and then warped temporally onto the current left frame
            prev_right_colour, valid_mask = self.right_to_left_warp(
                prev_right_colour, 
                prev_depth,  # left depth
                self.vr_config.camera_baseline, 
                self.vr_config.focal_length
            )
            if temporal_warp:
                prev_right_colour = self.warp(
                    prev_right_colour,
                    curr_motion_vectors  # left motion vectors
                )

            # The previous right feature frame warped onto the previous left feature frame, and then warped temporally onto the current left feature frame
            prev_right_feature, valid_mask = self.right_to_left_warp(
                prev_right_feature, 
                prev_depth,  # left depth
                self.vr_config.camera_baseline, 
                self.vr_config.focal_length
            )
            if temporal_warp:
                prev_right_feature = self.warp(
                    prev_right_feature,
                    curr_motion_vectors   # left motion vectors
                )

            # (B, C, H, W)
            inputs = torch.cat([
                curr_left_colour,          # C = 3
                curr_warped_right_colour,  # C = 3
                curr_depth,                # C = 1
                curr_jitter,               # C = 2
                prev_left_colour,          # C = 12
                prev_right_colour,         # C = 12
                prev_left_feature,         # C = 12
                prev_right_feature         # C = 12
            ], dim=1)

        else:
            # Current left frame warped onto the current right frame
            curr_warped_left_colour, valid_mask = self.left_to_right_warp(
                curr_left_colour,
                curr_depth,  # right depth
                self.vr_config.camera_baseline,
                self.vr_config.focal_length
            )

            # The previous right frame warped temporally onto the current right frame
            if temporal_warp:
                prev_right_colour = self.warp(
                    prev_right_colour,
                    curr_motion_vectors  # right motion vectors
                )

            # The previous right feature frame warped temporally onto the current right feature frame
            if temporal_warp:
                prev_right_feature = self.warp(
                    prev_right_feature,
                    curr_motion_vectors  # right motion vectors
                )

            # The previous left frame warped onto the previous right frame, and then warped temporally onto the current right frame
            prev_left_colour, valid_mask = self.left_to_right_warp(
                prev_left_colour,
                prev_depth,  # right depth
                self.vr_config.camera_baseline, 
                self.vr_config.focal_length
            )
            if temporal_warp:
                prev_left_colour = self.warp(
                    prev_left_colour,
                    curr_motion_vectors  # right motion vectors
                )

            # The previous left feature frame warped onto the previous right feature frame, and then warped temporally onto the current left right feature frame
            prev_left_feature, valid_mask = self.left_to_right_warp(
                prev_left_feature, 
                prev_depth,  # right depth
                self.vr_config.camera_baseline, 
                self.vr_config.focal_length
            )
            if temporal_warp:
                prev_left_feature = self.warp(
                    prev_left_feature,
                    curr_motion_vectors  # right motion vectors
                )

            # (B, C, H, W)
            # (The channel ordering relative to the left eye is important)
            inputs = torch.cat([
                curr_right_colour,        # C = 3
                curr_warped_left_colour,  # C = 3
                curr_depth,               # C = 1
                curr_jitter,              # C = 2
                prev_right_colour,        # C = 12
                prev_left_colour,         # C = 12
                prev_right_feature,       # C = 12
                prev_left_feature,        # C = 12
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

        # ------------------------------------------------------------
        # -------------------------- Blend ---------------------------
        # ------------------------------------------------------------
        # (B, 12, H, W) --> (B, 3, 4, H, W)
        out_colour = out_colour.view(B, 3, -1, H, W)

        prev_left_colour = prev_left_colour.view(B, 3, -1, H, W)
        prev_right_colour = prev_right_colour.view(B, 3, -1, H, W)
        if eye == "left":
            primary_history = prev_left_colour
            secondary_history = prev_right_colour
        else:
            primary_history = prev_right_colour
            secondary_history = prev_left_colour

        # (B, 12, H, W) --> (B, 3, 4, H, W)
        out_blending_mask = out_blending_mask.view(B, 3, -1, H, W)
        out_blending_mask = self.blending_mask_softmax(out_blending_mask)
        blended_colour = (
            out_colour * out_blending_mask[:, 0:1, ...] +
            primary_history * out_blending_mask[:, 1:2, ...] +
            secondary_history * out_blending_mask[:, 2:3, ...]
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
        prev_pred_left_colour = prev_pred_right_colour = prev_pred_left_features = prev_pred_right_features = None

        left_outputs = []
        right_outputs = []
        for clip in range(clip_size):
            left_clip_frames = left_inputs[:, clip].clone()
            left_motion_vector_frames = left_motion_vectors[:, clip]
            prev_left_depth_frames = prev_left_depth[:, clip]

            right_clip_frames = right_inputs[:, clip].clone()
            right_motion_vector_frames = right_motion_vectors[:, clip]
            prev_right_depth_frames = prev_right_depth[:, clip]

            if self.num_curr_jitter != 0:
                jitter_frames = jitter[:, clip]
            else:
                jitter_frames = None

            c0 = 0
            c1 = c0 + self.num_curr_colour
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
            c1 = c0 + self.num_prev_colour
            if mode == "training":
                prev_left_colour = prev_pred_left_colour if prev_pred_left_colour is not None else left_clip_frames[:, c0:c1] 
                prev_right_colour = prev_pred_right_colour if prev_pred_right_colour is not None else right_clip_frames[:, c0:c1]
            else:
                prev_left_colour = left_clip_frames[:, c0:c1] 
                prev_right_colour = right_clip_frames[:, c0:c1]
            
            c0 = c1
            c1 = c0 + self.num_prev_feature
            if mode == "training":
                prev_left_feature = prev_pred_left_features if prev_pred_left_features is not None else left_clip_frames[:, c0:c1] 
                prev_right_feature = prev_pred_right_features if prev_pred_right_features is not None else right_clip_frames[:, c0:c1]
            else:
                prev_left_feature = left_clip_frames[:, c0:c1] 
                prev_right_feature = right_clip_frames[:, c0:c1]

            if (mode == "training" and None not in (prev_pred_left_colour, prev_pred_right_colour, prev_pred_left_features, prev_pred_right_features)) or (mode != "training" and curr_frame_num > 0):
                temporal_warp = True
            else:
                temporal_warp = False
            
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
                prev_depth=prev_left_depth_frames,
                jitter_frames=jitter_frames,
                temporal_warp=temporal_warp,
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
                prev_depth=prev_right_depth_frames,
                jitter_frames=jitter_frames,
                temporal_warp=temporal_warp,
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
            prev_pred_right_features = right_out_features

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
