import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import zarr
import cv2
import imageio.v3 as iio
import numpy as np
from pathlib import Path
import skimage.io as io
import sys
import torch
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.utils import save_image
from tqdm import tqdm

from metrics.metrics import Metrics
from utils import gamma_to_linear, linear_to_gamma
from models.vr_network import VRConfig


def downsample(
    input_path: Path,
    output_path: Path,
    output_dimensions: tuple[int, int],
) -> None:
    def gamma_to_linear(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        return np.where(
            image <= 0.04045,
            image / 12.92,
            ((image + 0.055) / 1.055) ** 2.4
        )


    def linear_to_gamma(image: np.ndarray) -> np.ndarray:
        image = np.clip(image, 0.0, 1.0)
        return np.where(
            image <= 0.0031308,
            12.92 * image,
            1.055 * (image ** (1.0 / 2.4)) - 0.055
        )

    for instance in os.listdir(input_path):
        input_frames_path = input_path / instance
        output_frames_path = output_path / instance
        output_frames_path.mkdir(parents=True, exist_ok=True)
        for frame in os.listdir(input_frames_path):
            input_image_path = input_frames_path / frame

            image = io.imread(input_image_path, )[:, :, :3]
            image = gamma_to_linear(image)

            downsampled_image = cv2.resize(
                image,
                output_dimensions,
                interpolation=cv2.INTER_AREA
            )
            downsampled_image = linear_to_gamma(downsampled_image)
            downsampled_image = (downsampled_image * 255.0).round().astype(np.uint8)
            downsampled_image = cv2.cvtColor(downsampled_image, cv2.COLOR_RGB2BGR)

            output_image_path = output_frames_path / frame
            cv2.imwrite(output_image_path, downsampled_image)

        print(f"{instance} done.")


def evaluate(pred_path: Path, target_path: Path) -> None:
    # Reproduces the bicubic baseline to confirm metrics are calculated correctly
    pairs = list(zip(sorted(os.listdir(pred_path)), sorted(os.listdir(target_path))))

    metrics = Metrics(
        dataset_size=len(pairs),
        padding=0,
        iterations=0,
        display_name="standard_fhd"
    )

    cuda0 = torch.device("cuda:0")

    for frame_num, (pred_name, target_name) in enumerate(pairs):
        pred = decode_image((pred_path / pred_name).resolve())[0:3, ...]
        pred = pred.to(cuda0).to(torch.float32) / 255.0
        pred = pred.unsqueeze(0)
        pred = F.interpolate(
            pred,
            scale_factor=2, 
            mode="bicubic",
            align_corners=False
        )
        pred = torch.clamp(pred, 0.0, 1.0)

        target = decode_image((target_path / target_name).resolve())[0:3, ...]
        target = target.to(cuda0).to(torch.float32) / 255.0
        target = target.unsqueeze(0)

        metrics.record(pred, target)
        print(f"{frame_num=}")

    metrics.report("SeaPort")


def rename_files(folder_path: str, extension=".png"):
    files = os.listdir(folder_path)
    
    files.sort()

    print(f"Found {len(files)} files. Starting renaming...")

    for index, filename in enumerate(files):
        # Create the new name with 4-digit padding (0000, 0001, etc.)
        new_name = f"{index:04d}{extension}"
        
        # Build full file paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        
    print("Renaming complete.")


def display_depth(input_path: Path, output_path: Path) -> None:
    depth = decode_image(input_path.resolve()).float()
    depth = torch.unsqueeze((
        depth[0] / 255 +
        depth[1] / (255 ** 2) +
        depth[2] / (255 ** 3) +
        depth[3] / (255 ** 4)
    ), 0)
    save_image(linear_to_gamma(depth), output_path / "depth.png")
    return depth


def filter_exr_png(folder_path: Path) -> None:
    for subdirectory in folder_path.iterdir():
        if not subdirectory.is_dir():
            continue

        for directory_path, _, filenames in os.walk(subdirectory):
            for filename in filenames:
                file = Path(directory_path) / filename
                parent_name = file.parent.name
                suffix = file.suffix.lower()

                if suffix == ".exr" and parent_name == "Colour":
                    file.unlink()

                if suffix == ".png" and parent_name in {"Depth", "MotionVector"}:
                    file.unlink()


def subsample_data(folder_path: Path, take: int = 30, skip: int = 60) -> None:
    for file_path in folder_path.iterdir():
        if not file_path.is_dir():
            continue

        files = sorted(f for f in file_path.glob("*.*") if f.is_file())
        kept = []
        for i, f in enumerate(files):
            if i % skip < take:
                kept.append(f)
            else:
                f.unlink()

        for file_index, file_start in enumerate(range(0, len(kept), take)):
            destination_path = file_path / f"{file_index:04d}"
            destination_path.mkdir()
            for local_index, f in enumerate(kept[file_start:file_start + take]):
                f.rename(destination_path / f"{local_index:04d}{f.suffix}")


def prepare_data(folder_path: Path) -> None:
    renames = {
        "BP_VRStereoRig(1)": "Left",
        "BP_VRStereoRig(2)": "Right",
        "FinalImage": "Colour",
        "FinalImageDepth": "Depth",
        "FinalImageMotionVectors": "MotionVector",
    }

    for directory_path, directory_names, _ in os.walk(folder_path, topdown=False):
        for name in directory_names:
            if name in renames:
                Path(directory_path, name).rename(Path(directory_path, renames[name]))

    filter_exr_png(folder_path)


def warp_frames(left_frame_path: Path, right_frame_path: Path, depth_path: Path) -> None:
    def right_to_left_warp(
        right_frame: torch.Tensor,
        left_depth: torch.Tensor,
        camera_baseline: float,
        focal_length: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = right_frame.shape

        ys, xs = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing="ij"
        )
        ys = ys + 0.5 
        xs = xs + 0.5

        disparity = (camera_baseline * focal_length) / ((left_depth * 99990.0) + 10.0) 
        warped_xs = xs - disparity  # Note xs is broadcast here
        warped_xs = torch.permute(warped_xs, dims=(0, 2, 3, 1))  # (B, C, H, W) --> (B, H, W, C)
        warped_xs = 2.0 * (warped_xs / W) - 1.0  # Normalise to range [-1, 1]. Divide by W rather than W - 1 because we set align_corners=False
        ys = ys.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        ys = torch.permute(ys, dims=(0, 2, 3, 1))  # (B, C, H, W) --> (B, H, W, C)
        ys = 2.0 * (ys / H) - 1.0  # Normalise to range [-1, 1]. Divide by H rather than H - 1 because we set align_corners=False
        warped_grid = torch.cat((warped_xs, ys), dim=-1)

        # Warping the right frame on to the left frame
        warped_left_frame = F.grid_sample(
            right_frame,
            warped_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        return warped_left_frame
    
    vr_config = VRConfig(
        camera_baseline=6.4, 
        horizontal_fov=100.0,
        vertical_fov=105.8809,
        horizontal_resolution=1440,
        vertical_resolution=1600
    )

    right_frame = gamma_to_linear(decode_image(right_frame_path.resolve())[0:3, ...].float())
    depth = iio.imread(depth_path.resolve())
    depth = torch.permute(torch.from_numpy(depth), (2, 0, 1))
    depth = depth[0:1]

    warped_left_frame = right_to_left_warp(
        right_frame.unsqueeze(0),
        depth.unsqueeze(0),
        vr_config.camera_baseline,
        vr_config.focal_length
    )

    left_frame = gamma_to_linear(decode_image(left_frame_path.resolve())[0:3, ...].float())

    diff = torch.abs(left_frame - warped_left_frame)
    save_image(diff, "warped_left.png")


def generate_zarr(input_folder_path: Path, zarr_output_path: Path, channels: int) -> None:
    filenames = sorted([filename for filename in os.listdir(input_folder_path)])

    representative_image = cv2.imread(input_folder_path / filenames[0], cv2.IMREAD_UNCHANGED)
    H, W, _ = representative_image.shape

    if H == 1600:
        patch_size = 264
    elif H == 800:
        patch_size = 132
    elif H == 400:
        patch_size = 66

    data = zarr.open(
        zarr_output_path,
        mode="w",
        shape=(len(filenames), channels, H, W),
        chunks=(1, channels, patch_size, patch_size),
        dtype=representative_image.dtype
    )

    for i, filename in enumerate(tqdm(filenames)):
        image = cv2.imread(input_folder_path / filename, cv2.IMREAD_UNCHANGED)
        image = np.transpose(image, (2, 0, 1))

        print(f"{image.dtype=}")

        if channels == 1:
            # Arbitrarily extracting channel B from BGRA
            image = image[[0]]
        elif channels == 2:
            # Extracting channels R, G from BGRA
            image = image[[2, 1]]
        else:  # channels == 3
            # Extracting channels R, G, B from BGRA
            image = image[[2, 1, 0]]

        data[i] = image


def generate_vr_data_zarr(input_root_path: Path, output_root_path: Path):
    for directory_path, _, filenames in os.walk(input_root_path):
        directory_path = Path(directory_path)
        if any(filename.endswith((".png", ".exr")) for filename in filenames):
            relative_path = os.path.relpath(directory_path, input_root_path)

            if "Depth" in relative_path:
                channels = 1
            elif "MotionVector" in relative_path:
                channels = 2
            elif "Colour" in relative_path:
                channels = 3
            else:
                sys.exit("Unexpected directory path.")

            output_path = output_root_path / f"{relative_path}.zarr"
            generate_zarr(directory_path, output_path, channels)


if __name__ == "__main__":
    generate_vr_data_zarr(
        input_root_path=Path("../data/training_data/VR"),
        output_root_path=Path("../data/training_data/VR_zarr")
    )

    # warp_frames(
    #     left_frame_path=Path("0000_stereoscopic_images_visualisation_left"),
    #     right_frame_path=Path("0000_stereoscopic_images_visualisation_right"),
    #     depth_path=Path("0000_stereoscopic_images_visualisation_left_depth"),
    # )

    # folder_path = Path("../data/training_data/VR/FantasticVillage")
    # prepare_data(folder_path)
    # subsample_data(folder_path / "720x800/MipBiasMinus1Jittered/Left")
    # subsample_data(folder_path / "720x800/MipBiasMinus1Jittered/Right")
    # subsample_data(folder_path / "1440x1600/Enhanced/Left")
    # subsample_data(folder_path / "1440x1600/Enhanced/Right")
    # subsample_data(folder_path / "720x800/MipBiasMinus1Jittered")

    # prepare_data(Path("../data/validation_data/VR/FantasticVillage"))

    # prepare_data(Path("../data/test_data/VR/FantasticVillage"))

    # display_depth(
    #     Path("../data/test_data/QRISP/TestSet/SeaPort/540p/DepthMipBiasMinus1Jittered/0000/0000.png"),
    #     Path("checks")
    # )

    # rename_files( "../data/test_data/QRISP/TestSet/AbandonedSchoolStationary/540p/MotionVectorsMipBiasMinus1Jittered/0000", extension=".exr")

    # evaluate(
    #     Path("../data/test_data/QRISP/TestSet/SeaPort/540p/MipBiasMinus1/0000"),
    #     Path("../data/test_data/QRISP/TestSet/SeaPort/1080p/Enhanced/0000")
    # )

    # output_dimensions = (960, 540)

    # training_data_scenes = [
    #     "CBApocalypse",
    #     "FloodedGrounds",
    #     "FloodedGroundsBridges",
    #     "ScifiBase",
    #     "ScifiBaseNightStartStop",
    #     "ScifiBaseStartStop",
    #     "ScifiFacility",
    #     "SunTemple",
    #     "SunTempleBush",
    #     "SunTempleLamps"
    # ]
    # training_data_prefix = Path("../data/training_data/QRISP")
    # training_data_input_suffix = Path("1080p/Enhanced")
    # training_data_output_suffix = Path("540p/Enhanced")
    # for training_data_scene in training_data_scenes:
    #     downsample(
    #         training_data_prefix / training_data_scene / training_data_input_suffix,
    #         training_data_prefix / training_data_scene / training_data_output_suffix,
    #         output_dimensions
    #     )

    # validation_data_scenes = [
    #     "AbandonedSchool",
    #     "SeaPort",
    #     "SpaceShipDemo"
    # ]
    # validation_data_prefix = Path("../data/test_data/QRISP/TestSet")
    # validation_data_input_suffix = Path("1080p/Enhanced")
    # validation_data_output_suffix = Path("540p/Enhanced")
    # for validation_data_scene in validation_data_scenes:
    #     downsample(
    #         validation_data_prefix / validation_data_scene / validation_data_input_suffix,
    #         validation_data_prefix / validation_data_scene / validation_data_output_suffix,
    #         output_dimensions
    #     )
