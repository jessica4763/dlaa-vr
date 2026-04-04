import torch
from torch import nn
import torch.nn.functional as F


class JitterConditionedConv(nn.Module):
    def __init__(
        self,
        output_channels: int,
        input_channels: int,
        kernel_height: int,
        kernel_width: int,
        num_hidden_features: int = 2048,
        num_blocks: int = 7
    ) -> None:
        super().__init__()

        self.output_channels = output_channels
        self.input_channels = input_channels
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        num_outputs = output_channels * input_channels * kernel_height * kernel_width

        self.input_layer = nn.Sequential(
            nn.Linear(2, num_hidden_features),
            nn.ReLU()
        )

        body_layers = []
        for _ in range(num_blocks - 2):
            body_layers.append(
                nn.Linear(num_hidden_features, num_hidden_features)
            )
            body_layers.append(nn.ReLU())
        self.body = nn.Sequential(*body_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(num_hidden_features, num_outputs),
        )

    def forward(self, inputs: torch.Tensor, jitter: torch.Tensor) -> torch.Tensor:
        batch_size, _ = jitter.shape

        h = self.input_layer(jitter)
        h = self.body(h)
        kernel = self.output_layer(h)
        kernel = kernel.view(
            batch_size,
            self.output_channels,
            self.input_channels,
            self.kernel_height,
            self.kernel_width
        )

        # 1. The network doesn't directly update the kernel weights, but instead
        #    updates the weights of the MLP used to calculate the kernel weights
        # 2. The network applies a separate convolutional kernel to each of the frames 
        #    in the batch because they are associated with different camera jitter values.
        #    Although for loops are generally slow, I use them here for readability. 
        outputs = []
        for batch in range(batch_size):
            outputs.append(F.conv2d(inputs[batch:batch + 1], kernel[batch], padding=1))
        return torch.cat(outputs, dim=0)


class QualcommNetwork(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        scale_factor: int = 1,
        use_jitter: bool = False
    ) -> None:
        super().__init__()

        self.num_curr_colour = 3
        self.num_curr_depth = 1
        self.num_curr_jitter = 2 if use_jitter else 0  # 2 for displacement in both x and y
        self.num_prev_colour = self.num_curr_colour * (scale_factor ** 2)
        self.num_prev_feature = 1 * (scale_factor ** 2)

        self.input_channels = (
            self.num_curr_colour +
            self.num_curr_depth +
            self.num_curr_jitter +
            self.num_prev_colour +
            self.num_prev_feature
        )

        self.depth_to_space = nn.PixelShuffle(upscale_factor=scale_factor)
        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=scale_factor)

        if use_jitter:
            self.input_conv = JitterConditionedConv(
                output_channels=hidden_channels,
                input_channels=self.input_channels,
                kernel_height=3,
                kernel_width=3,
                num_hidden_features=2048,
                num_blocks=7
            )
        else:
            # Initial 3 × 3 Conv + ReLU block
            self.input_conv = nn.Conv2d(
                self.input_channels,
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
                scale_factor ** 2,
                hidden_channels,
                3,
                3,
                num_hidden_features=2048,
                num_blocks=7
            )
        else:
            self.blending_mask_head = nn.Conv2d(
                hidden_channels,
                scale_factor ** 2,
                kernel_size=3,
                padding=1,
                padding_mode="zeros"
            )
        self.blending_mask_sigmoid = nn.Sigmoid()

        self.apply(self.kaiming_init_params)

    def kaiming_init_params(self, model):
        if isinstance(model, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(model.weight, mode="fan_out", nonlinearity="relu")
            if model.bias is not None:
                nn.init.constant_(model.bias, 0)

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

    def forward(
        self,
        inputs: torch.Tensor,
        motion_vectors: torch.Tensor,
        curr_frame_num: int,
        jitter: torch.Tensor = None,
        mode: str = "training"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, clip_size, C, H, W = inputs.shape

        # To hold the recurrent colour frame and features
        prev_pred_frame = prev_pred_features = None

        outputs = []
        for clip in range(clip_size):
            clip_frames = inputs[:, clip].clone()
            motion_vector_frames = motion_vectors[:, clip]
            if self.num_curr_jitter != 0:
                jitter_frames = jitter[:, clip]

            # Use recurrent colour frame
            c0 = self.num_curr_colour + self.num_curr_depth + self.num_curr_jitter
            c1 = c0 + self.num_prev_colour
            if mode == "training":
                if prev_pred_frame is not None:
                    # Warp recurrent colour frame
                    clip_frames[:, c0:c1] = self.warp(
                        prev_pred_frame,
                        motion_vector_frames
                    )
            else:
                if curr_frame_num > 0:
                    clip_frames[:, c0:c1] = self.warp(
                        clip_frames[:, c0:c1],
                        motion_vector_frames
                    )

            prev_colour = clip_frames[:, c0:c1]  # Save for the blend step

            # Use recurrent features
            c0 = c1
            c1 = c0 + self.num_prev_feature
            if mode == "training":
                if prev_pred_features is not None:
                    # Warp recurrent features
                    clip_frames[:, c0:c1] = self.warp(
                        prev_pred_features,
                        motion_vector_frames
                    )
            else:
                if curr_frame_num > 0:
                    clip_frames[:, c0:c1] = self.warp(
                        clip_frames[:, c0:c1],
                        motion_vector_frames
                    )

            # ------------------------------------------------------------
            # ---------------- Input convolution and ReLU ----------------
            # ------------------------------------------------------------
            if self.num_curr_jitter != 0:
                h = self.input_conv(clip_frames, jitter_frames)
            else:
                h = self.input_conv(clip_frames)

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
            out_blending_mask = self.blending_mask_sigmoid(out_blending_mask)

            # ------------------------------------------------------------
            # -------------------------- Blend ---------------------------
            # ------------------------------------------------------------
            # (B, 12, H, W) --> (B, 3, 4, H, W)
            out_colour = out_colour.view(B, 3, -1, H, W)
            prev_colour = prev_colour.view(B, 3, -1, H, W)

            # (B, 4, H, W) --> (B, 1, 4, H, W)
            out_blending_mask = out_blending_mask.view(B, 1, -1, H, W)

            blended_colour = out_blending_mask * out_colour + (1.0 - out_blending_mask) * prev_colour
            blended_colour = torch.clamp(blended_colour, min=0.0, max=1.0)
            blended_colour = blended_colour.view(B, -1, H, W)

            # ------------------------------------------------------------
            # ---------------------- Depth to space ----------------------
            # ------------------------------------------------------------
            full_res_colour = self.depth_to_space(blended_colour)  # (B, 3, H, W)
            outputs.append(full_res_colour)

            # Save recurrent colour frame and features
            prev_pred_frame = blended_colour
            prev_pred_features = out_features

        # prev_pred_features is only used by evaluation
        # out_blending_mask is only used during evaluation for inspection
        return torch.stack(outputs, dim=1), prev_pred_features, out_blending_mask.view(B, -1, H, W)
