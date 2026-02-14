import torch
from torch import nn
import torch.nn.functional as F


class JitterConditionedConv(nn.Module):
    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        kernel_height: int,
        kernel_width: int,
        num_hidden_features: int = 2048,
        num_blocks: int = 7
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        num_outputs = out_channels * in_channels * kernel_height * kernel_width

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

    def forward(self, x: torch.Tensor, jitter: torch.Tensor) -> torch.Tensor:
        batch_size, _ = jitter.shape

        h = self.input_layer(jitter)
        h = self.body(h)
        kernel = self.output_layer(h)
        kernel = kernel.view(
            batch_size,
            self.out_channels,
            self.in_channels,
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
            outputs.append(F.conv2d(x[batch:batch + 1], kernel[batch], padding=1))
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

        self.in_channels = (
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
                padding_mode="replicate"
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
                    padding_mode="replicate"
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
                padding_mode="replicate"
            )

        # Colour head
        self.colour_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                self.num_prev_colour,
                kernel_size=3,
                padding=1,
                padding_mode="replicate"
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
                padding_mode="replicate"
            ),
            nn.Sigmoid()
        )

    def warp(
        self,
        input_tensor: torch.Tensor,
        motion_vectors: torch.Tensor
    ) -> torch.Tensor:
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
            indexing='ij'
        )
        base_grid = torch.stack((x, y), dim=-1).unsqueeze(0).to(motion_vectors.device)
        warped_grid = base_grid - motion_vectors * 2.0  # base_grid is broadcasted

        warped_input_tensor = F.grid_sample(
            input_tensor,
            warped_grid,
            mode='bilinear',
            padding_mode='zeros'
        )

        warped_input_tensor = self.space_to_depth(warped_input_tensor)

        return warped_input_tensor

    def forward(
        self,
        x: torch.Tensor,
        motion_vectors: torch.Tensor,
        jitter: torch.Tensor = None,
        mode: str = "training"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, clip_size, C, H, W = x.shape

        # To hold the recurrent colour frame and features
        prev_pred_colour = prev_pred_features = None

        outputs = []
        for clip in range(clip_size):
            clip_frames = x[:, clip].clone()
            motion_vector_frames = motion_vectors[:, clip].clone()
            if self.num_curr_jitter != 0:
                assert jitter is not None
                jitter_frames = jitter[:, clip].clone()

            # Use recurrent colour frame
            c0 = self.num_curr_colour + self.num_curr_depth + self.num_curr_jitter
            c1 = c0 + self.num_prev_colour
            if mode == "training":
                if prev_pred_colour is not None:
                    # Warp recurrent colour frame
                    clip_frames[:, c0:c1] = self.warp(
                        prev_pred_colour,
                        motion_vector_frames
                    )
            else:
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
                clip_frames[:, c0:c1] = self.warp(
                    clip_frames[:, c0:c1],
                    motion_vector_frames
                )

            # ------------------------------------------------------------
            # ---------------- Input convolution and ReLU ----------------
            # ------------------------------------------------------------
            if self.num_curr_jitter != 0:
                assert jitter is not None
                h = self.input_conv(clip_frames, jitter_frames)
            else:
                h = self.input_conv(clip_frames)

            h = self.input_relu(h)

            # ------------------------------------------------------------
            # ---------------------- Main conv body ----------------------
            # ------------------------------------------------------------
            h = self.body(h)

            # ------------------------------------------------------------
            # ---------------------- Feature branch ----------------------
            # ------------------------------------------------------------
            if self.num_curr_jitter != 0:
                assert jitter is not None
                out_features = self.feature_head(h, jitter_frames)
            else:
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
            blended_colour = torch.clamp(blended_colour, min=0.0, max=1.0)

            # ------------------------------------------------------------
            # ---------------------- Depth to space ----------------------
            # ------------------------------------------------------------
            full_res_colour = self.depth_to_space(blended_colour)  # (B, 3, H, W)
            outputs.append(full_res_colour)

            # Save recurrent colour frame and features
            prev_pred_colour = blended_colour
            prev_pred_features = out_features

        # prev_pred_features is only used by evaluation
        return torch.stack(outputs, dim=1), prev_pred_features


class VRSpatialNetwork(nn.Module):
    pass


class VRSpatialTemporalNetwork(nn.Module):
    pass
