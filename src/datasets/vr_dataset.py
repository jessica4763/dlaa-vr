from bisect import bisect_right
import imageio.v3 as iio
import json
import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import decode_image
import zarr

from datasets.qualcomm_dataset import QualcommDataset
from utils import Scene, cumsum


class VRDataset(QualcommDataset):
    def __init__(
        self,
        input_imgs_path: str,
        output_imgs_path: str,
        input_frame_height: int,
        input_frame_width: int,
        camera_data_path_suffix: str,
        input_path_suffix: str,
        jittered_input_path_suffix: str,
        colour_path_suffix: str,
        depth_path_suffix: str,
        motion_vector_path_suffix: str,
        scene_names: list[str],
        use_jitter: bool = False,
        scale_factor: int = 1,
        dilation_block_size: int = 8,
        transform=None,
        target_transform=None,
        zarr_walk_root=None,
        dataset_from: str = "unreal_engine",
        mode: str = "training",
        validation_length: int = 600
    ) -> None:
        if mode == "training":
            # Open zarr files
            self.data = {}
            for directory_path, _, _ in os.walk(zarr_walk_root):
                directory_path = Path(directory_path)
                if directory_path.suffix == ".zarr":
                    key = Path(os.path.relpath(directory_path, Path.cwd()))
                    key = key.with_suffix("").as_posix()
                    self.data[key] = zarr.open(directory_path, mode="r")

        self.input_imgs_path = input_imgs_path        # ../data/test_data/VR/*/720x800
        self.output_imgs_path = output_imgs_path      # ../data/test_data/VR/*/1440x1600/Enhanced

        self.input_frame_height = input_frame_height  
        self.input_frame_width = input_frame_width

        if use_jitter:
            self.input_imgs_path += f"/{jittered_input_path_suffix}"             # ../data/test_data/VR/*/720x800/MipBiasMinus1Jittered
        else:
            self.input_imgs_path += f"/{input_path_suffix}"             # ../data/test_data/VR/*/720x800/MipBiasMinus1

        self.camera_data_path_suffix = camera_data_path_suffix      # CameraData
        self.colour_path_suffix = colour_path_suffix                # Colour
        self.depth_path_suffix = depth_path_suffix                  # Depth
        self.motion_vector_path_suffix = motion_vector_path_suffix  # MotionVector

        self.scenes = []
        for scene_name in scene_names:
            scene_input_imgs_path = Path(self.input_imgs_path.replace("*", scene_name))
            scene_output_imgs_path = Path(self.output_imgs_path.replace("*", scene_name))
            path_suffix = "Left" + f"/{self.colour_path_suffix}"  # "Left/Colour" arbitrarily
            scene = Scene(
                scene_input_imgs_path=scene_input_imgs_path, 
                scene_output_imgs_path=scene_output_imgs_path, 
                path_suffix=path_suffix,
                mode=mode,
                is_vr=True
            )
            self.scenes.append(scene)

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
        self.validation_length = validation_length

    def __len__(self) -> int:
        if self.mode == "training":
            return self.total_frames
        else:
            return min(self.total_frames, self.validation_length)
        
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
            jitter_offset_x = -1 * camera_data["jitter_offset"]["x"] / 2
            jitter_offset_y = -1 * -camera_data["jitter_offset"]["y"] / 2
            return jitter_offset_x, jitter_offset_y
        
    def get_depth(
        self,
        scene: Scene,
        instance: str,
        curr_frame_num: int,
        patch_start_x: int,
        patch_start_y: int,
        patch_end_x: int,
        patch_end_y: int,
        eye: str
    ) -> torch.Tensor:
        if self.mode == "training":
            depth_path = scene.scene_input_imgs_path / eye / self.depth_path_suffix / instance
            depth_path = depth_path.as_posix()
            depth = self.data[depth_path][curr_frame_num, :, patch_start_y:patch_end_y, patch_start_x:patch_end_x]
            depth = torch.from_numpy(depth)
        else:
            curr_frame = str(curr_frame_num).zfill(4) + ".exr"
            depth_path = scene.scene_input_imgs_path / eye / self.depth_path_suffix / instance / curr_frame
            depth = iio.imread(depth_path.resolve())

            # (H, W, C) --> (C, H, W)
            depth = torch.permute(torch.from_numpy(depth), (2, 0, 1))

            # (C, H, W) --> (1, H, W)
            depth = depth[0:1]

            depth = self.get_patch(
                depth,
                patch_start_x,
                patch_start_y,
                patch_end_x,
                patch_end_y
            )

        return depth
    
    def get_motion_vectors(
        self,
        scene: Scene,
        instance: str,
        curr_frame_num: int,
        patch_start_x: int,
        patch_start_y: int,
        patch_end_x: int,
        patch_end_y: int,
        eye: str
    ) -> torch.Tensor:
        if self.mode == "training":
            motion_vectors_path = scene.scene_input_imgs_path / eye / self.motion_vector_path_suffix / instance
            motion_vectors_path = motion_vectors_path.as_posix()
            motion_vectors = self.data[motion_vectors_path][curr_frame_num, :, patch_start_y:patch_end_y, patch_start_x:patch_end_x]
            motion_vectors = torch.from_numpy(motion_vectors)
        else:
            curr_frame = str(curr_frame_num).zfill(4) + ".exr"
            motion_vectors_path = scene.scene_input_imgs_path / eye / self.motion_vector_path_suffix / instance / curr_frame
            motion_vectors = iio.imread(motion_vectors_path.resolve())

            # (H, W, C) --> (C, H, W)
            motion_vectors = torch.permute(torch.from_numpy(motion_vectors), (2, 0, 1))

            motion_vectors = self.get_patch(
                motion_vectors,
                patch_start_x,
                patch_start_y,
                patch_end_x,
                patch_end_y
            )

        if self.dataset_from == "unreal_engine" and not self.motion_vector_path_suffix == "MotionVectorRAFT":
            # Unreal Engine motion vectors are normalised to the range [0, 1], 
            # where (0.5, 0.5) represents no motion. Convert to the range [-1, 1],
            # where (0, 0) represents no motion. 
            motion_vectors = (motion_vectors - 0.5) * 2.0

        motion_vectors = motion_vectors[0:2]

        motion_vectors[1] *= -1

        return motion_vectors
        
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

        # ---------------------------- Left frame ---------------------------

        if self.mode == "training":
            left_input_img_path = scene.scene_input_imgs_path / "Left" / self.colour_path_suffix / instance
            left_input_img_path = left_input_img_path.as_posix()
            curr_left_input_img = self.data[left_input_img_path][curr_frame_num, :, patch_start_y:patch_end_y, patch_start_x:patch_end_x]
            curr_left_input_img = torch.from_numpy(curr_left_input_img)
            # print(f"{curr_left_input_img.shape=}")
            # print(f"{curr_left_input_img.dtype=}")
            # print(f"{curr_left_input_img=}")
            # saved_image = torch.from_numpy(curr_left_input_img)
            # saved_image = saved_image.float() / 255.0
            # save_image(saved_image, "sample_output.png")
        else:
            left_input_img_path = scene.scene_input_imgs_path / "Left" / self.colour_path_suffix / instance / curr_frame
            curr_left_input_img = decode_image(left_input_img_path.resolve())[0:3].float()
            curr_left_input_img = self.get_patch(
                curr_left_input_img,
                patch_start_x,
                patch_start_y,
                patch_end_x,
                patch_end_y
            )

        if self.mode == "training":
            left_output_img_path = scene.scene_output_imgs_path / "Left" / self.colour_path_suffix / instance
            left_output_img_path = left_output_img_path.as_posix()
            curr_left_output_img = self.data[left_output_img_path][curr_frame_num, :, patch_start_y * self.scale_factor:patch_end_y * self.scale_factor, patch_start_x * self.scale_factor:patch_end_x * self.scale_factor]
            curr_left_output_img = torch.from_numpy(curr_left_output_img)
        else:
            left_output_img_path = scene.scene_output_imgs_path / "Left" / self.colour_path_suffix / instance / curr_frame
            curr_left_output_img = decode_image(left_output_img_path.resolve())[0:3].float()
            curr_left_output_img = self.get_patch(
                curr_left_output_img,
                patch_start_x * self.scale_factor,
                patch_start_y * self.scale_factor,
                patch_end_x * self.scale_factor,
                patch_end_y * self.scale_factor
            )

        # --------------------------- Right frame ---------------------------
        right_output_img_path = scene.scene_output_imgs_path / "Right" / self.colour_path_suffix / instance / curr_frame

        if self.mode == "training":
            right_input_img_path = scene.scene_input_imgs_path / "Right" / self.colour_path_suffix / instance
            right_input_img_path = right_input_img_path.as_posix()
            curr_right_input_img = self.data[right_input_img_path][curr_frame_num, :, patch_start_y:patch_end_y, patch_start_x:patch_end_x]
            curr_right_input_img = torch.from_numpy(curr_right_input_img)
        else:
            right_input_img_path = scene.scene_input_imgs_path / "Right" / self.colour_path_suffix / instance / curr_frame
            curr_right_input_img = decode_image(right_input_img_path.resolve())[0:3].float()
            curr_right_input_img = self.get_patch(
                curr_right_input_img,
                patch_start_x,
                patch_start_y,
                patch_end_x,
                patch_end_y
            )

        if self.mode == "training":
            right_output_img_path = scene.scene_output_imgs_path / "Right" / self.colour_path_suffix / instance
            right_output_img_path = right_output_img_path.as_posix()
            curr_right_output_img = self.data[right_output_img_path][curr_frame_num, :, patch_start_y * self.scale_factor:patch_end_y * self.scale_factor, patch_start_x * self.scale_factor:patch_end_x * self.scale_factor]
            curr_right_output_img = torch.from_numpy(curr_right_output_img)
        else:
            right_output_img_path = scene.scene_output_imgs_path / "Right" / self.colour_path_suffix / instance / curr_frame
            curr_right_output_img = decode_image(right_output_img_path.resolve())[0:3].float()
            curr_right_output_img = self.get_patch(
                curr_right_output_img,
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
            # ---------------------------- Left frame ---------------------------
            curr_left_input_img = self.transform(curr_left_input_img)

            # prev_left_output_img will be overwritten later, if there was a previous frame output from the model
            prev_left_output_img = curr_left_input_img.clone().detach()

            # --------------------------- Right frame ---------------------------
            curr_right_input_img = self.transform(curr_right_input_img)
            
            # prev_right_output_img will be overwritten later, if there was a previous frame output from the model
            prev_right_output_img = curr_right_input_img.clone().detach()

        if self.target_transform:
            # ---------------------------- Left frame ---------------------------
            curr_left_output_img = self.target_transform(curr_left_output_img)
            
            # --------------------------- Right frame ---------------------------
            curr_right_output_img = self.target_transform(curr_right_output_img)

        # ---------------------------- Left frame ---------------------------
        prev_left_output_img = F.interpolate(  # Interpolate in linear space
            prev_left_output_img.unsqueeze(0),  # F.interpolate expects a batch dimension
            scale_factor=self.scale_factor, 
            mode="bicubic",
            align_corners=False,
            antialias=True
        ).squeeze(0)  # Remove the batch dimension

        prev_left_output_img = nn.PixelUnshuffle(downscale_factor=self.scale_factor)(prev_left_output_img)

        # --------------------------- Right frame ---------------------------
        prev_right_output_img = F.interpolate(  # Interpolate in linear space
            prev_right_output_img.unsqueeze(0),  # F.interpolate expects a batch dimension
            scale_factor=self.scale_factor, 
            mode="bicubic",
            align_corners=False,
            antialias=True
        ).squeeze(0)  # Remove the batch dimension

        prev_right_output_img = nn.PixelUnshuffle(downscale_factor=self.scale_factor)(prev_right_output_img)

        # -------------------------------------------------------------------
        # ----------------------------- Features ----------------------------
        # -------------------------------------------------------------------

        # ---------------------------- Left frame ---------------------------
        curr_left_depth = self.get_depth(
            scene,
            instance,
            curr_frame_num,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y,
            "Left"
        )

        # --------------------------- Right frame ---------------------------
        curr_right_depth = self.get_depth(
            scene,
            instance,
            curr_frame_num,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y,
            "Right"
        )

        # ---------------------------- Left frame ---------------------------
        curr_left_motion_vectors = self.get_motion_vectors(
            scene,
            instance,
            curr_frame_num,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y,
            "Left"
        )

        # --------------------------- Right frame ---------------------------
        curr_right_motion_vectors = self.get_motion_vectors(
            scene,
            instance,
            curr_frame_num,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y,
            "Right"
        )
        
        # Jitter is the same for both eyes, but we create jitter tensors for each eye
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
                curr_left_input_img.device,
                curr_left_input_img.dtype
            )

            # ---------------------------- Left frame ---------------------------
            curr_left_motion_vectors = self.apply_jitter_compensation(
                curr_left_motion_vectors,
                prev_jitter_offset_x,
                prev_jitter_offset_y,
                curr_jitter_offset_x,
                curr_jitter_offset_y,
                self.input_frame_height,  # Still need to scale relative to the dimensions of the frame, not the dimensions of the patch
                self.input_frame_width
            )

            # --------------------------- Right frame ---------------------------
            curr_right_motion_vectors = self.apply_jitter_compensation(
                curr_right_motion_vectors,
                prev_jitter_offset_x,
                prev_jitter_offset_y,
                curr_jitter_offset_x,
                curr_jitter_offset_y,
                self.input_frame_height,  # Still need to scale relative to the dimensions of the frame, not the dimensions of the patch
                self.input_frame_width
            )

            jitter = torch.tensor((curr_jitter_offset_x, curr_jitter_offset_y))


        # ---------------------------- Left frame ---------------------------
        # Must scale motion vectors after taking a patch 
        curr_left_motion_vectors[0] *= self.input_frame_width / patch_width
        curr_left_motion_vectors[1] *= self.input_frame_height / patch_height
        curr_left_motion_vectors = self.depth_informed_dilation(
            curr_left_depth,
            curr_left_motion_vectors
        )

        # --------------------------- Right frame ---------------------------
        # Must scale motion vectors after taking a patch 
        curr_right_motion_vectors[0] *= self.input_frame_width / patch_width
        curr_right_motion_vectors[1] *= self.input_frame_height / patch_height
        curr_right_motion_vectors = self.depth_informed_dilation(
            curr_right_depth,
            curr_right_motion_vectors
        )

        # ---------------------------- Left frame ---------------------------
        prev_left_features = torch.zeros((self.scale_factor ** 2, patch_height, patch_width))

        # --------------------------- Right frame ---------------------------
        prev_right_features = torch.zeros((self.scale_factor ** 2, patch_height, patch_width))

        # -------------------------------------------------------------------------
        # ----------------------------- Prepare input -----------------------------
        # -------------------------------------------------------------------------
        if self.use_jitter:
            left_input_imgs = torch.cat(
                [
                    curr_left_input_img,
                    curr_left_depth,
                    curr_jitter_x,
                    curr_jitter_y,
                    prev_left_output_img,
                    prev_left_features
                ],
                dim=0
            )

            right_input_imgs = torch.cat(
                [
                    curr_right_input_img,
                    curr_right_depth,
                    curr_jitter_x,
                    curr_jitter_y,
                    prev_right_output_img,
                    prev_right_features
                ],
                dim=0
            )
        else:
            left_input_imgs = torch.cat(
                [
                    curr_left_input_img,
                    curr_left_depth,
                    prev_left_output_img,
                    prev_left_features
                ],
                dim=0
            )

            right_input_imgs = torch.cat(
                [
                    curr_right_input_img,
                    curr_right_depth,
                    prev_right_output_img,
                    prev_right_features
                ],
                dim=0
            )

        return (
            left_input_imgs, 
            right_input_imgs, 
            curr_left_motion_vectors,
            curr_right_motion_vectors,
            jitter, 
            curr_left_output_img, 
            curr_right_output_img, 
            curr_frame_num
        )
