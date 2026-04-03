import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import zarr
import cv2
import imageio.v3 as iio
import numpy as np
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.utils import save_image
from tqdm import tqdm

from metrics.metrics import Metrics
from utils import gamma_to_linear
from models.vr_network import VRConfig


# -------------------------------------------------------------------------
# ------------------ Reproduce Qualcomm bicubic baseline ------------------
# -------------------------------------------------------------------------

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


def filter_exr_png(folder_path: Path) -> None:
    for subdirectory in folder_path.iterdir():
        if not subdirectory.is_dir():
            continue

        for directory_path, _, filenames in os.walk(subdirectory):
            for filename in filenames:
                file = Path(directory_path) / filename
                parent_name = file.parent.name
                grandparent_name = file.parent.parent.name
                suffix = file.suffix.lower()

                if suffix == ".exr" and (parent_name == "Colour" or grandparent_name == "Colour"):
                    file.unlink()

                if suffix == ".png" and (parent_name in {"Depth", "MotionVector"} or grandparent_name in {"Depth", "MotionVector"}):
                    file.unlink()


def subsample_training_data(folder_path: Path, take: int = 30, skip: int = 60) -> None:
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


def instance_evaluation_data(folder_path: Path) -> None:
    for file_path in folder_path.iterdir():
        if not file_path.is_dir():
            continue
        
        files = sorted(f for f in file_path.glob("*.*") if f.is_file())
        
        if files:
            destination_path = file_path / "0000"
            destination_path.mkdir(exist_ok=True)
            for f in files:
                f.rename(destination_path / f.name)


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
    # generate_vr_data_zarr(
    #     input_root_path=Path("../data/training_data/VR"),
    #     output_root_path=Path("../data/training_data/VR_zarr")
    # )

    # folder_path = Path("../data/training_data/VR/FantasticVillage")
    # prepare_data(folder_path)
    # subsample_training_data(folder_path / "720x800/MipBiasMinus1Jittered/Left")
    # subsample_training_data(folder_path / "720x800/MipBiasMinus1Jittered/Right")
    # subsample_training_data(folder_path / "1440x1600/Enhanced/Left")
    # subsample_training_data(folder_path / "1440x1600/Enhanced/Right")
    # subsample_training_data(folder_path / "720x800/MipBiasMinus1Jittered")  # CameraData
    
    folder_path = Path("../data/validation_data/VR/FantasticVillage")
    prepare_data(folder_path)
    instance_evaluation_data(folder_path / "720x800/MipBiasMinus1Jittered/Left")
    instance_evaluation_data(folder_path / "720x800/MipBiasMinus1Jittered/Right")
    # instance_evaluation_data(folder_path / "1440x1600/Enhanced/Left")
    # instance_evaluation_data(folder_path / "1440x1600/Enhanced/Right")
    instance_evaluation_data(folder_path / "720x800/MipBiasMinus1Jittered")  # Subsample CameraData

    # folder_path = Path("../data/test_data/VR/FantasticVillage")
    # prepare_data(folder_path)
    # instance_evaluation_data(folder_path / "720x800/MipBiasMinus1Jittered/Left")
    # instance_evaluation_data(folder_path / "720x800/MipBiasMinus1Jittered/Right")
    # instance_evaluation_data(folder_path / "1440x1600/Enhanced/Left")
    # instance_evaluation_data(folder_path / "1440x1600/Enhanced/Right")
    # instance_evaluation_data(folder_path / "720x800/MipBiasMinus1Jittered")  # Subsample CameraData

    # evaluate(
    #     Path("../data/test_data/QRISP/TestSet/SeaPort/540p/MipBiasMinus1/0000"),
    #     Path("../data/test_data/QRISP/TestSet/SeaPort/1080p/Enhanced/0000")
    # )
