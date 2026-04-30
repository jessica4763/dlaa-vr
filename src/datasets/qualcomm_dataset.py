from bisect import bisect_right
import imageio.v3 as iio
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import decode_image
from torch.utils.data import Dataset

from utils import Scene, cumsum


class QualcommDataset(Dataset):
    def __init__(
        self,
        input_imgs_path: str,
        output_imgs_path: str,
        input_frame_height: int,
        input_frame_width: int,
        camera_data_path_suffix: str,
        colour_path_suffix: str,
        depth_path_suffix: str,
        motion_vector_path_suffix: str,
        colour_jittered_path_suffix: str,
        depth_jittered_path_suffix: str,
        motion_vector_jittered_path_suffix: str,
        scene_names: list[str],
        use_jitter: bool = False,
        scale_factor: int = 1,
        dilation_block_size: int = 8,
        transform=None,
        target_transform=None,
        dataset_from: str = "unreal_engine",
        mode: str = "training"
    ) -> None:
        self.input_imgs_path = input_imgs_path
        self.output_imgs_path = output_imgs_path

        self.input_frame_height = input_frame_height
        self.input_frame_width = input_frame_width

        self.camera_data_path_suffix = camera_data_path_suffix

        if use_jitter:
            self.colour_path_suffix = colour_jittered_path_suffix
            self.depth_path_suffix = depth_jittered_path_suffix
            self.motion_vector_path_suffix = motion_vector_jittered_path_suffix
        else:
            self.colour_path_suffix = colour_path_suffix
            self.depth_path_suffix = depth_path_suffix
            self.motion_vector_path_suffix = motion_vector_path_suffix

        self.scenes = []
        for scene_name in scene_names:
            scene_input_imgs_path = Path(self.input_imgs_path.replace("*", scene_name))
            scene_output_imgs_path = Path(self.output_imgs_path.replace("*", scene_name))
            self.scenes.append(Scene(scene_input_imgs_path, scene_output_imgs_path, self.colour_path_suffix, mode=mode, is_vr=False))

        scene_num_instances = [scene.num_instances for scene in self.scenes]
        self.instance_boundaries = cumsum(scene_num_instances)
        self.total_instances = sum(scene_num_instances)

        scene_num_frames = [scene.num_frames for scene in self.scenes]
        self.frame_boundaries = cumsum(scene_num_frames)
        self.total_frames = sum(scene_num_frames)

        self.use_jitter = use_jitter
        self.scale_factor = scale_factor
        self.dilation_block_size = dilation_block_size
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_from = dataset_from
        self.mode = mode

    def get_jitter_offsets(
        self,
        scene: Scene,
        instance: str,
        frame_num: int
    ) -> tuple[float, float]:
        frame = str(frame_num).zfill(4) + ".json"
        json_file_path = scene.scene_input_imgs_path / self.camera_data_path_suffix / instance / frame
        with open(json_file_path, mode="r", encoding="utf-8") as json_file:
            camera_data = json.load(json_file)
            # Negate jitter_offset_y because Unity is Y-up
            # Then, negate both jitter offsets again because the projection matrices are jittered
            jitter_offset_x = -1 * camera_data["jitter_offset"]["x"]
            jitter_offset_y = -1 * -camera_data["jitter_offset"]["y"]
            return jitter_offset_x, jitter_offset_y

    def get_jitter_tensors(
        self,
        depth: int,
        height: int,
        width: int,
        jitter_offset_x: float,
        jitter_offset_y: float,
        device: str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        jitter_tensor_x = torch.full(
            (depth, height, width),
            fill_value=jitter_offset_x,
            device=device,
            dtype=dtype
        )

        jitter_tensor_y = torch.full(
            (depth, height, width),
            fill_value=jitter_offset_y,
            device=device,
            dtype=dtype
        )

        return jitter_tensor_x, jitter_tensor_y

    def get_depth(
        self,
        scene: Scene,
        instance: str,
        curr_frame_num: int
    ) -> torch.Tensor:
        if self.dataset_from == "unreal_engine":
            curr_frame = str(curr_frame_num).zfill(4) + ".exr"
            depth_path = scene.scene_input_imgs_path / self.depth_path_suffix / instance / curr_frame
            depth = iio.imread(depth_path.resolve())

            # (H, W, C) --> (C, H, W)
            depth = torch.permute(torch.from_numpy(depth), (2, 0, 1))

            # (C, H, W) --> (1, H, W)
            depth = depth[0:1]

            return depth
        else:
            curr_frame = str(curr_frame_num).zfill(4) + ".png"
            depth_path = scene.scene_input_imgs_path / self.depth_path_suffix / instance / curr_frame
            depth = decode_image(depth_path.resolve()).float()
            depth = torch.unsqueeze((
                depth[0] / 255 +
                depth[1] / (255 ** 2) +
                depth[2] / (255 ** 3) +
                depth[3] / (255 ** 4)
            ), 0)
            return depth

    def get_motion_vectors(
        self,
        scene: Scene,
        instance: str,
        curr_frame_num: int
    ) -> torch.Tensor:
        curr_frame = str(curr_frame_num).zfill(4) + ".exr"
        motion_vectors_path = scene.scene_input_imgs_path / self.motion_vector_path_suffix / instance / curr_frame
        motion_vectors = iio.imread(motion_vectors_path.resolve())

        # (H, W, C) --> (C, H, W)
        motion_vectors = torch.permute(torch.from_numpy(motion_vectors), (2, 0, 1))

        if self.dataset_from == "unreal_engine":
            # Unreal Engine motion vectors are normalised to the range [0, 1], 
            # where (0.5, 0.5) represents no motion. Convert to the range [-1, 1],
            # where (0, 0) represents no motion. 
            motion_vectors = (motion_vectors - 0.5) * 2.0

        # The horizontal velocity is stored in the first channel and the
        # vertical velocity is stored in the second channel for the Qualcomm dataset, despite what 
        # the paper says; could be an artifact of iio.imread
        motion_vectors = motion_vectors[0:2]

        # The code assumes a Y-down coordinate system
        motion_vectors[1] *= -1

        return motion_vectors

    def apply_jitter_compensation(
        self,
        motion_vectors: torch.Tensor,
        prev_jitter_x: torch.Tensor,
        prev_jitter_y: torch.Tensor,
        curr_jitter_x: torch.Tensor,
        curr_jitter_y: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        motion_vectors[[0]] += (prev_jitter_x - curr_jitter_x) / width
        motion_vectors[[1]] += (prev_jitter_y - curr_jitter_y) / height
        return motion_vectors

    def depth_informed_dilation(
        self,
        depth: torch.Tensor,
        motion_vectors: torch.Tensor
    ) -> torch.Tensor:
        # (C, H / self.scale_factor, W / self.scale_factor) -> (C, H, W) (identity if self.scale_factor = 1)
        depth, motion_vectors = self.upscale_buffer(depth), self.upscale_buffer(motion_vectors)

        # (C, H, W) -> (1, C, H, W)
        depth, motion_vectors = depth.unsqueeze(0), motion_vectors.unsqueeze(0)

        min_pooled_depths, indices = F.max_pool2d(
            -depth,
            kernel_size=self.dilation_block_size,
            stride=self.dilation_block_size,
            return_indices=True
        )
        min_pooled_depths = -min_pooled_depths

        # (1, 2, H, W) --> (1, 2, H * W)
        flattened_motion_vectors = torch.flatten(
            motion_vectors,
            start_dim=2
        )
        flattened_indices = torch.flatten(indices)

        # The motion vectors which correspond to the pixels
        # in each 2 x 2 block with the shallowest depth
        selected_motion_vectors = torch.stack([
            flattened_motion_vectors[:, 0, flattened_indices],
            flattened_motion_vectors[:, 1, flattened_indices]
        ], dim=1)

        # (1, 2, H_new * W_new) --> (1, 2, H_new, W_new)
        seleted_motion_vectors = torch.reshape(
            selected_motion_vectors,
            (1, 2, min_pooled_depths.shape[2], min_pooled_depths.shape[3])
        )

        output_motion_vectors = F.interpolate(
            seleted_motion_vectors,
            scale_factor=self.dilation_block_size,
            mode="nearest"
        )

        return output_motion_vectors.squeeze(0)
    
    def upscale_buffer(self, buffer: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            buffer.unsqueeze(0), 
            scale_factor=self.scale_factor, 
            mode="nearest"
        ).squeeze(0)
    
    def get_patch(
        self, 
        x: torch.Tensor,
        patch_start_x: int, 
        patch_start_y: int, 
        patch_end_x: int, 
        patch_end_y: int, 
    ) -> torch.Tensor:
        # (C, H, W) -> (C, patch_size, patch_size)
        patch = x[:, patch_start_y:patch_end_y, patch_start_x:patch_end_x]

        # .clone() because slicing returns a view of the original frame, and therefore keeps the whole frame in memory
        return patch.clone()

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "training":
            idx, patch_start_x, patch_start_y, patch_end_x, patch_end_y = item
        else: 
            idx = item 
            patch_start_x, patch_start_y, patch_end_x, patch_end_y = (0, 0, self.input_frame_width, self.input_frame_height)

        # Get patch dimensions
        patch_height = patch_end_y - patch_start_y
        patch_width = patch_end_x - patch_start_x

        # Get the scene associated with this index
        scene_idx = bisect_right(self.frame_boundaries, idx)
        scene = self.scenes[scene_idx - 1]

        # Offset the index relative to the scene
        idx_offset = self.frame_boundaries[scene_idx - 1]
        idx -= idx_offset

        instance = str(idx // scene.num_frames_per_instance).zfill(4)

        # -------------------------------------------------------------------
        # -------------------------- Current frame --------------------------
        # -------------------------------------------------------------------
        curr_frame_num = idx % scene.num_frames_per_instance
        curr_frame = str(curr_frame_num).zfill(4) + ".png"
        curr_input_img_path = scene.scene_input_imgs_path / self.colour_path_suffix / instance / curr_frame
        curr_output_img_path = scene.scene_output_imgs_path / instance / curr_frame

        curr_input_img = decode_image(curr_input_img_path.resolve())[0:3].float()
        curr_input_img = self.get_patch(
            curr_input_img,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y
        )

        curr_output_img = decode_image(curr_output_img_path.resolve())[0:3].float()
        curr_output_img = self.get_patch(
            curr_output_img,
            patch_start_x * self.scale_factor,
            patch_start_y * self.scale_factor,
            patch_end_x * self.scale_factor,
            patch_end_y * self.scale_factor
        )
        
        # -------------------------------------------------------------------
        # ------------------- Transforms + previous frame -------------------
        # -------------------------------------------------------------------
        prev_frame_num = max(0, curr_frame_num - 1)

        if self.transform:
            curr_input_img = self.transform(curr_input_img)

        if self.target_transform:
            curr_output_img = self.target_transform(curr_output_img)

        # prev_output_img will be overwritten later, if there was a previous frame output from the model
        prev_output_img = curr_input_img.clone().detach()

        prev_output_img = F.interpolate(  # Interpolate in linear space
            prev_output_img.unsqueeze(0),  # F.interpolate expects a batch dimension
            scale_factor=self.scale_factor, 
            mode="bicubic",
            align_corners=False,
            antialias=True
        ).squeeze(0)  # Remove the batch dimension

        prev_output_img = nn.PixelUnshuffle(downscale_factor=self.scale_factor)(prev_output_img)

        # -------------------------------------------------------------------
        # ----------------------------- Features ----------------------------
        # -------------------------------------------------------------------
        curr_depth = self.get_depth(
            scene,
            instance,
            curr_frame_num
        )
        curr_depth = self.get_patch(
            curr_depth,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y
        )
        
        motion_vectors = self.get_motion_vectors(
            scene,
            instance,
            curr_frame_num
        )
        motion_vectors = self.get_patch(
            motion_vectors,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y
        )
        
        jitter = torch.tensor((0, 0))
        if self.use_jitter:
            curr_jitter_offset_x, curr_jitter_offset_y = self.get_jitter_offsets(
                scene, 
                instance, 
                curr_frame_num
            )

            prev_jitter_offset_x, prev_jitter_offset_y = self.get_jitter_offsets(
                scene, 
                instance, 
                prev_frame_num
            )

            curr_jitter_x, curr_jitter_y = self.get_jitter_tensors(
                1,
                patch_height,
                patch_width,
                curr_jitter_offset_x,
                curr_jitter_offset_y,
                curr_input_img.device,
                curr_input_img.dtype
            )

            motion_vectors = self.apply_jitter_compensation(
                motion_vectors,
                prev_jitter_offset_x,
                prev_jitter_offset_y,
                curr_jitter_offset_x,
                curr_jitter_offset_y,
                self.input_frame_height,  # Still need to scale relative to the dimensions of the frame, not the dimensions of the patch
                self.input_frame_width
            )

            jitter = torch.tensor((curr_jitter_offset_x, curr_jitter_offset_y))

        # Must scale motion vectors after taking a patch 
        motion_vectors[0] *= self.input_frame_width / patch_width
        motion_vectors[1] *= self.input_frame_height / patch_height
        motion_vectors = self.depth_informed_dilation(
            curr_depth,
            motion_vectors
        )

        prev_features = torch.zeros((self.scale_factor ** 2, patch_height, patch_width))

        # -------------------------------------------------------------------------
        # ----------------------------- Prepare input -----------------------------
        # -------------------------------------------------------------------------
        if self.use_jitter:
            input_imgs = torch.cat(
                [
                    curr_input_img,
                    curr_depth,
                    curr_jitter_x,
                    curr_jitter_y,
                    prev_output_img,
                    prev_features
                ],
                dim=0
            )
        else:
            input_imgs = torch.cat(
                [
                    curr_input_img,
                    curr_depth,
                    prev_output_img,
                    prev_features
                ],
                dim=0
            )

        return input_imgs, motion_vectors, jitter, curr_output_img, curr_frame_num
        