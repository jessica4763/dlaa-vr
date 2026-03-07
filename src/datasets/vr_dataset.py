from bisect import bisect_right
import imageio.v3 as iio
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import decode_image

from datasets.qualcomm_dataset import QualcommDataset
from utils import Scene, cumsum


class VRDataset(QualcommDataset):
    def __init__(
        self,
        input_imgs_path: str,
        output_imgs_path: str,
        input_frame_height: int,
        input_frame_width: int,
        input_path_suffix: str,
        jittered_input_path_suffix: str,
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
        mode: str = "training"
    ) -> None:
        self.input_imgs_path = input_imgs_path        # ../data/test_data/VR/*/720x800
        self.output_imgs_path = output_imgs_path      # ../data/test_data/VR/*/1440x1600
        self.input_frame_height = input_frame_height  
        self.input_frame_width = input_frame_width

        if use_jitter:
            self.input_imgs_path += f"/{jittered_input_path_suffix}"             # ../data/test_data/VR/*/720x800/MipBiasMinus1Jittered
            self.colour_path_suffix = colour_jittered_path_suffix                # Colour
            self.depth_path_suffix = depth_jittered_path_suffix                  # Depth
            self.motion_vector_path_suffix = motion_vector_jittered_path_suffix  # MotionVector
        else:
            self.input_imgs_path += f"/{input_path_suffix}"             # ../data/test_data/VR/*/720x800/MipBiasMinus1
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
                path_suffix=path_suffix
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
            # Then, negate both jitter offsets again because it's the projection matrices that are jittered
            jitter_offset_x = -1 * camera_data["jitter_offset"]["x"]
            jitter_offset_y = -1 * -camera_data["jitter_offset"]["y"]
            return jitter_offset_x, jitter_offset_y
        
    def get_depth(
        self,
        scene: Scene,
        instance: str,
        curr_frame_num: int,
        eye: str
    ) -> torch.Tensor:
        curr_frame = str(curr_frame_num).zfill(4) + ".png"
        depth_path = scene.scene_input_imgs_path / eye / self.depth_path_suffix / instance / curr_frame
        depth = iio.imread(depth_path.resolve())

        # (H, W, C) --> (C, H, W)
        depth = torch.permute(torch.from_numpy(depth), (2, 0, 1))

        # (C, H, W) --> (1, H, W)
        depth = depth[0:1]

        return depth
    
    def get_motion_vectors(
        self,
        scene: Scene,
        instance: str,
        curr_frame_num: int,
        eye: str
    ) -> torch.Tensor:
        curr_frame = str(curr_frame_num).zfill(4) + ".exr"
        motion_vectors_path = scene.scene_input_imgs_path / eye / self.motion_vector_path_suffix / instance / curr_frame
        motion_vectors = iio.imread(motion_vectors_path.resolve())

        # (H, W, C) --> (C, H, W)
        motion_vectors = torch.permute(torch.from_numpy(motion_vectors), (2, 0, 1))

        # Unreal Engine motion vectors are normalised to the range [0, 1], 
        # where (0.5, 0.5) represents no motion. Convert to the range [-1, 1],
        # where (0, 0) represents no motion. 
        motion_vectors = (motion_vectors - 0.5) * 2.0

        # The horizontal velocity is stored in the first channel
        motion_vectors = motion_vectors[0:2, ...]

        # Although Unreal Engine uses a Y-up coordinate system, this code
        # assumes a Y-down coordinate system
        motion_vectors[1, ...] *= -1

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
        left_input_img_path = scene.scene_input_imgs_path / "Left" / self.colour_path_suffix / instance / curr_frame
        left_output_img_path = scene.scene_output_imgs_path / "Left" / instance / curr_frame

        curr_left_input_img = decode_image(left_input_img_path.resolve())[0:3, ...].float()
        curr_left_input_img = self.get_patch(
            curr_left_input_img,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y
        )

        curr_left_output_img = decode_image(left_output_img_path.resolve())[0:3, ...].float()
        curr_left_output_img = self.get_patch(
            curr_left_output_img,
            patch_start_x * self.scale_factor,
            patch_start_y * self.scale_factor,
            patch_end_x * self.scale_factor,
            patch_end_y * self.scale_factor
        )

        # --------------------------- Right frame ---------------------------
        right_input_img_path = scene.scene_input_imgs_path / "Right" / self.colour_path_suffix / instance / curr_frame
        right_output_img_path = scene.scene_output_imgs_path / "Right" / instance / curr_frame

        curr_right_input_img = decode_image(right_input_img_path.resolve())[0:3, ...].float()
        curr_right_input_img = self.get_patch(
            curr_right_input_img,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y
        )

        curr_right_output_img = decode_image(right_output_img_path.resolve())[0:3, ...].float()
        curr_right_output_img = self.get_patch(
            curr_right_output_img,
            patch_start_x * self.scale_factor,
            patch_start_y * self.scale_factor,
            patch_end_x * self.scale_factor,
            patch_end_y * self.scale_factor
        )
        
        # -------------------------------------------------------------------
        # -------------------------- Previous frame -------------------------
        # -------------------------------------------------------------------
        prev_frame_num = max(0, curr_frame_num - 1)

        # ---------------------------- Left frame ---------------------------
        # prev_left_output_img will be overwritten later, if there was a previous frame output from the model
        prev_left_output_img = curr_left_input_img.clone().detach()

        # --------------------------- Right frame ---------------------------
        # prev_right_output_img will be overwritten later, if there was a previous frame output from the model
        prev_right_output_img = curr_right_input_img.clone().detach()

        # -------------------------------------------------------------------
        # ---------------------------- Transforms ---------------------------
        # -------------------------------------------------------------------

        if self.transform:
            # ---------------------------- Left frame ---------------------------
            curr_left_input_img = self.transform(curr_left_input_img)
            prev_left_output_img = self.transform(prev_left_output_img)

            # --------------------------- Right frame ---------------------------
            curr_right_input_img = self.transform(curr_right_input_img)
            prev_right_output_img = self.transform(prev_right_output_img)

        if self.target_transform:
            # ---------------------------- Left frame ---------------------------
            curr_left_output_img = self.target_transform(curr_left_output_img)
            
            # --------------------------- Right frame ---------------------------
            curr_right_output_img = self.target_transform(curr_right_output_img)

        # ---------------------------- Left frame ---------------------------
        prev_left_output_img = F.interpolate(  # Interpolate in linear space
            prev_left_output_img.unsqueeze(0),  # F.interpolate expects a batch dimension
            scale_factor=self.scale_factor, 
            mode='bicubic',
            align_corners=False,
            antialias=True
        ).squeeze(0)  # Remove the batch dimension

        prev_output_img = nn.PixelUnshuffle(downscale_factor=self.scale_factor)(prev_left_output_img)

        # --------------------------- Right frame ---------------------------
        prev_right_output_img = F.interpolate(  # Interpolate in linear space
            prev_right_output_img.unsqueeze(0),  # F.interpolate expects a batch dimension
            scale_factor=self.scale_factor, 
            mode='bicubic',
            align_corners=False,
            antialias=True
        ).squeeze(0)  # Remove the batch dimension

        prev_output_img = nn.PixelUnshuffle(downscale_factor=self.scale_factor)(prev_right_output_img)

        # -------------------------------------------------------------------
        # ----------------------------- Features ----------------------------
        # -------------------------------------------------------------------

        # ---------------------------- Left frame ---------------------------
        curr_left_depth = self.get_depth(
            scene,
            instance,
            curr_frame_num,
            "Left"
        )
        curr_left_depth = self.get_patch(
            curr_left_depth,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y
        )

        prev_left_depth = self.get_depth(
            scene,
            instance,
            prev_frame_num,
            "Left"
        )
        prev_left_depth = self.get_patch(
            prev_left_depth,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y
        )

        # --------------------------- Right frame ---------------------------
        curr_right_depth = self.get_depth(
            scene,
            instance,
            curr_frame_num,
            "Right"
        )
        curr_right_depth = self.get_patch(
            curr_right_depth,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y
        )

        prev_right_depth = self.get_depth(
            scene,
            instance,
            prev_frame_num,
            "Right"
        )
        prev_right_depth = self.get_patch(
            prev_right_depth,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y
        )
        
        # ---------------------------- Left frame ---------------------------
        curr_left_motion_vectors = self.get_motion_vectors(
            scene,
            instance,
            curr_frame_num,
            "Left"
        )
        curr_left_motion_vectors = self.get_patch(
            curr_left_motion_vectors,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y
        )

        # --------------------------- Right frame ---------------------------
        curr_right_motion_vectors = self.get_motion_vectors(
            scene,
            instance,
            curr_frame_num,
            "Right"
        )
        curr_right_motion_vectors = self.get_patch(
            curr_right_motion_vectors,
            patch_start_x,
            patch_start_y,
            patch_end_x,
            patch_end_y
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

            # ---------------------------- Left frame ---------------------------
            curr_left_jitter_x, curr_left_jitter_y = self.get_jitter_tensors(
                1,
                patch_height,
                patch_width,
                curr_jitter_offset_x,
                curr_jitter_offset_y,
                curr_left_input_img.device,
                curr_left_input_img.dtype
            )

            # --------------------------- Right frame ---------------------------
            curr_right_jitter_x, curr_right_jitter_y = self.get_jitter_tensors(
                1,
                patch_height,
                patch_width,
                curr_jitter_offset_x,
                curr_jitter_offset_y,
                curr_right_input_img.device,
                curr_right_input_img.dtype
            )

            # ---------------------------- Left frame ---------------------------
            prev_left_jitter_x, prev_left_jitter_y = self.get_jitter_tensors(
                1,
                patch_height,
                patch_width,
                prev_jitter_offset_x,
                prev_jitter_offset_y,
                curr_left_input_img.device,
                curr_left_input_img.dtype
            )

            # --------------------------- Right frame ---------------------------
            prev_right_jitter_x, prev_right_jitter_y = self.get_jitter_tensors(
                1,
                patch_height,
                patch_width,
                prev_jitter_offset_x,
                prev_jitter_offset_y,
                curr_right_input_img.device,
                curr_right_input_img.dtype
            )

            # ---------------------------- Left frame ---------------------------
            curr_left_motion_vectors = self.apply_jitter_compensation(
                curr_left_motion_vectors,
                prev_left_jitter_x,
                prev_left_jitter_y,
                curr_left_jitter_x,
                curr_left_jitter_y,
                self.input_frame_height,  # Still need to scale relative to the dimensions of the frame, not the dimensions of the patch
                self.input_frame_width
            )

            # --------------------------- Right frame ---------------------------
            curr_right_motion_vectors = self.apply_jitter_compensation(
                curr_right_motion_vectors,
                prev_right_jitter_x,
                prev_right_jitter_y,
                curr_right_jitter_x,
                curr_right_jitter_y,
                self.input_frame_height,  # Still need to scale relative to the dimensions of the frame, not the dimensions of the patch
                self.input_frame_width
            )

            jitter = torch.tensor((curr_jitter_offset_x, curr_jitter_offset_y))


        # ---------------------------- Left frame ---------------------------
        # Must scale motion vectors after taking a patch 
        curr_left_motion_vectors[0, ...] *= self.input_frame_width / patch_width
        curr_left_motion_vectors[1, ...] *= self.input_frame_height / patch_height
        curr_left_motion_vectors = self.depth_informed_dilation(
            curr_left_depth,
            curr_left_motion_vectors
        )

        # --------------------------- Right frame ---------------------------
        # Must scale motion vectors after taking a patch 
        curr_right_motion_vectors[0, ...] *= self.input_frame_width / patch_width
        curr_right_motion_vectors[1, ...] *= self.input_frame_height / patch_height
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
                    curr_left_jitter_x,
                    curr_left_jitter_y,
                    prev_left_output_img,
                    prev_left_features
                ],
                dim=0
            )

            right_input_imgs = torch.cat(
                [
                    curr_right_input_img,
                    curr_right_depth,
                    curr_right_jitter_x,
                    curr_right_jitter_y,
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
            prev_left_depth,
            prev_right_depth,
            jitter, 
            curr_left_output_img, 
            curr_right_output_img, 
            curr_frame_num
        )
