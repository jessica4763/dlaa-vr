from bisect import bisect_right
import json
import os
from pathlib import Path
import torch
from torchvision.io import decode_image
from torch.utils.data import Dataset

from utils import cumsum


class Scene:
    def __init__(
        self,
        scene_input_imgs_path: str,
        scene_output_imgs_path: str
    ):
        self.scene_input_imgs_path = scene_input_imgs_path
        self.scene_output_imgs_path = scene_output_imgs_path

        instances = os.listdir(scene_input_imgs_path)
        frames = os.listdir(scene_input_imgs_path / instances[0])

        self.num_instances = len(instances)
        self.num_frames_per_instance = len(frames)
        self.num_frames = self.num_instances * self.num_frames_per_instance


class ToyDataset(Dataset):
    def __init__(
        self,
        scene_names: list[str],
        input_imgs_path: str,
        output_imgs_path: str,
        transform=None,
        target_transform=None,
    ):
        self.input_imgs_path = input_imgs_path
        self.output_imgs_path = output_imgs_path

        self.scenes = []
        for scene_name in scene_names:
            scene_input_imgs_path = Path(self.input_imgs_path.replace('*', scene_name))
            scene_output_imgs_path = Path(self.output_imgs_path.replace('*', scene_name))
            self.scenes.append(Scene(scene_input_imgs_path, scene_output_imgs_path))

        scene_num_frames = [scene.num_frames for scene in self.scenes]

        self.frame_boundaries = cumsum(scene_num_frames)
        self.total_frames = sum(scene_num_frames)

        self.transform = transform
        self.target_transform = target_transform

    def get_jitter(
        self,
        input_image: torch.Tensor,
        scene: Scene,
        instance: str,
        curr_frame_num: str
    ) -> torch.Tensor:
        curr_frame = str(curr_frame_num).zfill(4) + '.json'
        json_file_path = scene.scene_input_imgs_path / "../CameraData" / instance / curr_frame
        with open(json_file_path, mode="r", encoding="utf-8") as json_file:
            camera_data = json.load(json_file)
            curr_frame_x = camera_data["jitter_offset"]["x"]
            curr_frame_y = camera_data["jitter_offset"]["y"]

        jitter_x = torch.full(
            (1, input_image.shape[1], input_image.shape[2]),
            fill_value=curr_frame_x,
            device=input_image.device,
            dtype=input_image.dtype
        )

        jitter_y = torch.full(
            (1, input_image.shape[1], input_image.shape[2]),
            fill_value=curr_frame_y,
            device=input_image.device,
            dtype=input_image.dtype
        )

        return jitter_x, jitter_y

    def get_depth(
        self,
        scene: Scene,
        instance: str,
        curr_frame_num: str
    ) -> torch.Tensor:
        curr_frame = str(curr_frame_num).zfill(4) + '.png'
        depth_path = scene.scene_input_imgs_path / "../DepthMipBiasMinus2Jittered" / instance / curr_frame
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
        curr_frame_num: str
    ) -> torch.Tensor:
        curr_frame = str(curr_frame_num).zfill(4) + '.exr'
        motion_vectors_path = scene.scene_input_imgs_path / "../MotionVectorsMipBiasMinus2Jittered" / instance / curr_frame
        motion_vectors = decode_image(motion_vectors_path.resolve())
        return motion_vectors

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

        curr_frame_num = idx % scene.num_frames_per_instance
        curr_frame = str(curr_frame_num).zfill(4) + '.png'
        curr_input_img_path = scene.scene_input_imgs_path / instance / curr_frame
        curr_output_img_path = scene.scene_input_imgs_path / instance / curr_frame
        curr_input_img = decode_image(curr_input_img_path.resolve())[0:3, ...]
        curr_output_img = decode_image(curr_output_img_path.resolve())[0:3, ...]

        # Teacher forcing: use the gound truth as the previous frame during
        # training. Limitation is that the network never explicitly learns
        # to use its own output as the previous frame during inference.

        # Alternatively, we could gradually introduce using the model's own
        # output as the previous frame during inference.

        # When there is no previous frame, i.e. this is the first frame,
        # duplicate the current frame and use it as the previous frame.
        prev_frame_num = 0 if curr_frame_num == 0 else curr_frame_num - 1
        prev_frame = str(prev_frame_num).zfill(4) + '.png'
        prev_output_img_path = scene.scene_output_imgs_path / instance / prev_frame
        prev_output_img = decode_image(prev_output_img_path.resolve())[0:3, ...]

        if self.transform:
            curr_input_img = self.transform(curr_input_img)
            prev_output_img = self.transform(prev_output_img)
        if self.target_transform:
            curr_output_img = self.target_transform(curr_output_img)

        curr_depth = self.get_depth(
            scene,
            instance,
            curr_frame_num
        )

        curr_jitter_x, curr_jitter_y = self.get_jitter(
            curr_input_img,
            scene,
            instance,
            curr_frame_num
        )

        prev_depth = self.get_depth(
            scene,
            instance,
            prev_frame_num
        )

        prev_jitter_x, prev_jitter_y = self.get_jitter(
            prev_output_img,
            scene,
            instance,
            prev_frame_num
        )

        prev_features = torch.cat(
            [
                prev_depth,
                prev_jitter_x,
                prev_jitter_y
            ],
            dim=0
        )

        input_imgs = torch.cat(
            [
                curr_input_img,
                # curr_depth,
                # curr_jitter_x,
                # curr_jitter_y,
                # prev_output_img,
                # prev_features
            ],
            dim=0
        )

        return input_imgs, curr_output_img
