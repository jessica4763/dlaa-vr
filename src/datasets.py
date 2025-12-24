import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from bisect import bisect_right
import cv2
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.io import decode_image
from torch.utils.data import Dataset


from utils import Scene, cumsum


class QualcommDataset(Dataset):
    def __init__(
        self,
        input_imgs_path: str,
        output_imgs_path: str,
        ground_truth_path_suffix: str,
        colour_path_suffix: str,
        depth_path_suffix: str,
        camera_data_path_suffix: str,
        motion_vector_path_suffix: str,
        scene_names: list[str],
        jitter: bool = False,
        transform=None,
        target_transform=None,
    ) -> None:
        self.input_imgs_path = input_imgs_path
        self.output_imgs_path = output_imgs_path
        self.ground_truth_path_suffix = ground_truth_path_suffix
        self.colour_path_suffix = colour_path_suffix
        self.depth_path_suffix = depth_path_suffix
        self.camera_data_path_suffix = camera_data_path_suffix
        self.motion_vector_path_suffix = motion_vector_path_suffix
        self.jitter = jitter

        self.scenes = []
        for scene_name in scene_names:
            scene_input_imgs_path = Path(self.input_imgs_path.replace("*", scene_name))
            scene_output_imgs_path = Path(self.output_imgs_path.replace("*", scene_name))
            self.scenes.append(Scene(scene_input_imgs_path, scene_output_imgs_path, colour_path_suffix))

        scene_num_instances = [scene.num_instances for scene in self.scenes]
        self.instance_boundaries = cumsum(scene_num_instances)
        self.total_instances = sum(scene_num_instances)

        scene_num_frames = [scene.num_frames for scene in self.scenes]
        self.frame_boundaries = cumsum(scene_num_frames)
        self.total_frames = sum(scene_num_frames)

        self.transform = transform
        self.target_transform = target_transform

    def get_jitter(
        self,
        depth,
        height,
        width,
        device: str,
        dtype: torch.dtype,
        scene: Scene,
        instance: str,
        curr_frame_num: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        curr_frame = str(curr_frame_num).zfill(4) + ".json"
        json_file_path = scene.scene_input_imgs_path / self.camera_data_path_suffix / instance / curr_frame
        with open(json_file_path, mode="r", encoding="utf-8") as json_file:
            camera_data = json.load(json_file)
            curr_frame_x = camera_data["jitter_offset"]["x"]
            curr_frame_y = camera_data["jitter_offset"]["y"]

        jitter_x = torch.full(
            (depth, height, width),
            fill_value=curr_frame_x,
            device=device,
            dtype=dtype
        )

        jitter_y = torch.full(
            (depth, height, width),
            fill_value=curr_frame_y,
            device=device,
            dtype=dtype
        )

        return jitter_x, jitter_y

    def get_depth(
        self,
        scene: Scene,
        instance: str,
        curr_frame_num: int
    ) -> torch.Tensor:
        curr_frame = str(curr_frame_num).zfill(4) + ".png"

        depth_path = scene.scene_input_imgs_path / self.depth_path_suffix / instance / curr_frame

        depth = decode_image(depth_path.resolve())
        depth = torch.unsqueeze((
            depth[0] / (255 ** 1) +
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
        curr_frame = str(curr_frame_num).zfill(4) + '.exr'

        motion_vectors_path = scene.scene_input_imgs_path / self.motion_vector_path_suffix / instance / curr_frame

        motion_vectors = cv2.imread(motion_vectors_path.resolve(), cv2.IMREAD_UNCHANGED)

        # (H, W, C) --> (C, H, W)
        motion_vectors = torch.permute(torch.from_numpy(motion_vectors), (2, 0, 1))

        # The vertical velocity is stored in the first channel and the
        # horizontal velocity is stored in the second channel
        motion_vectors = motion_vectors[0:2, ...]

        # Although Unity uses a Y-up coordinate system, this code
        # assumes a Y-down coordinate system
        motion_vectors[0, ...] *= -1

        # Let the horizontal velocity be stored in the first channel and the
        # vertical velocity in the second channel
        motion_vectors[0, ...], motion_vectors[1, ...] = motion_vectors[1, ...], motion_vectors[0, ...]

        return motion_vectors

    def apply_jitter_compensation(
        self,
        motion_vectors: torch.Tensor,
        scene: Scene,
        instance: str,
        curr_jitter_x: torch.Tensor,
        curr_jitter_y: torch.Tensor,
        prev_frame_num: int
    ) -> torch.Tensor:
        """Update the motion vectors to account for jitter."""
        depth = 1
        height = motion_vectors.shape[1]
        width = motion_vectors.shape[2]

        prev_jitter_x, prev_jitter_y = self.get_jitter(
            depth,
            height,
            width,
            motion_vectors.device,
            motion_vectors.dtype,
            scene,
            instance,
            prev_frame_num
        )

        motion_vectors[[0], ...] += 2.0 * (prev_jitter_x - curr_jitter_x) / width
        motion_vectors[[1], ...] += 2.0 * (prev_jitter_y - curr_jitter_y) / height

        return motion_vectors

    def depth_informed_dilation(
        self,
        depth: torch.Tensor,
        motion_vectors: torch.Tensor
    ) -> torch.Tensor:
        depth, motion_vectors = depth.unsqueeze(0), motion_vectors.unsqueeze(0)

        min_pooled_depths, indices = F.max_pool2d(
            -depth,
            kernel_size=2,
            stride=2,
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
            scale_factor=2,
            mode='nearest'
        )

        return output_motion_vectors

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        curr_output_img_path = scene.scene_output_imgs_path / self.ground_truth_path_suffix / instance / curr_frame
        curr_input_img = decode_image(curr_input_img_path.resolve())[0:3, ...]
        curr_output_img = decode_image(curr_output_img_path.resolve())[0:3, ...]

        # -------------------------------------------------------------------
        # -------------------------- Previous frame -------------------------
        # -------------------------------------------------------------------

        prev_frame_num = 0 if curr_frame_num == 0 else curr_frame_num - 1
        prev_output_img = curr_input_img.clone().detach()

        # -------------------------------------------------------------------
        # ---------------------------- Transforms ---------------------------
        # -------------------------------------------------------------------

        if self.transform:
            curr_input_img = self.transform(curr_input_img)
            prev_output_img = self.transform(prev_output_img)

        if self.target_transform:
            curr_output_img = self.target_transform(curr_output_img)

        # -------------------------------------------------------------------
        # ----------------------------- Features ----------------------------
        # -------------------------------------------------------------------

        curr_depth = self.get_depth(
            scene,
            instance,
            curr_frame_num
        )

        motion_vectors = self.get_motion_vectors(
            scene,
            instance,
            curr_frame_num
        )

        if self.jitter:
            curr_jitter_x, curr_jitter_y = self.get_jitter(
                1,
                curr_input_img.shape[1],
                curr_input_img.shape[2],
                curr_input_img.device,
                curr_input_img.dtype,
                scene,
                instance,
                curr_frame_num
            )

            motion_vectors = self.apply_jitter_compensation(
                motion_vectors,
                scene,
                instance,
                curr_jitter_x,
                curr_jitter_y,
                prev_frame_num
            )

        motion_vectors = self.depth_informed_dilation(
            curr_depth,
            motion_vectors
        )

        prev_features = torch.zeros((1, curr_input_img.shape[1], curr_input_img.shape[2]))

        if self.jitter:
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

        # Return motion vectors for the current frame to be used for warping
        return input_imgs, curr_output_img, motion_vectors
