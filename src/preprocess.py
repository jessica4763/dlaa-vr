import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import zarr
import cv2
import imageio.v3
from natsort import natsorted
import numpy as np
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from torchvision.io import decode_image, ImageReadMode
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import save_image
from tqdm import tqdm

from metrics.metrics import Metrics
from utils import gamma_to_linear
from models.vr_network import VRConfig


def write_video(
    input_path: Path,
    output_path: Path,
    fps: int = 24
) -> None:
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=10,
        pixelformat="yuv420p",
        macro_block_size=8
    )

    for img_name in natsorted(os.listdir(input_path)):
        img_path = input_path / img_name
        img = imageio.v3.imread(img_path)
        writer.append_data(img)

    writer.close()

    
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
        pred = decode_image((pred_path / pred_name).resolve())[0:3]
        pred = pred.to(cuda0).to(torch.float32) / 255.0
        pred = pred.unsqueeze(0)
        pred = F.interpolate(
            pred,
            scale_factor=2, 
            mode="bicubic",
            align_corners=False
        )
        pred = torch.clamp(pred, 0.0, 1.0)

        target = decode_image((target_path / target_name).resolve())[0:3]
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
        warp_xs = xs - disparity  # Note xs is broadcast here
        warp_xs = torch.permute(warp_xs, dims=(0, 2, 3, 1))  # (B, C, H, W) --> (B, H, W, C)
        warp_xs = 2.0 * (warp_xs / W) - 1.0  # Normalise to range [-1, 1]. Divide by W rather than W - 1 because we set align_corners=False
        ys = ys.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        ys = torch.permute(ys, dims=(0, 2, 3, 1))  # (B, C, H, W) --> (B, H, W, C)
        ys = 2.0 * (ys / H) - 1.0  # Normalise to range [-1, 1]. Divide by H rather than H - 1 because we set align_corners=False
        warp_grid = torch.cat((warp_xs, ys), dim=-1)

        # Warping the right frame on to the left frame
        warped_left_frame = F.grid_sample(
            right_frame,
            warp_grid,
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

    right_frame = gamma_to_linear(decode_image(right_frame_path.resolve())[0:3].float())
    depth = imageio.v3.imread(depth_path.resolve())
    depth = torch.permute(torch.from_numpy(depth), (2, 0, 1))
    depth = depth[0:1]

    warped_left_frame = right_to_left_warp(
        right_frame.unsqueeze(0),
        depth.unsqueeze(0),
        vr_config.camera_baseline,
        vr_config.focal_length
    )

    left_frame = gamma_to_linear(decode_image(left_frame_path.resolve())[0:3].float())

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


def generate_vr_data_zarr(input_root_path: Path, output_root_path: Path) -> None:
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


def generate_raft_motion_vectors(parent_path: Path) -> None:
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Use the RAFT model
    raft_weights = Raft_Large_Weights.C_T_SKHT_V2
    raft_model = raft_large(weights=raft_weights, progress=False).to(device)

    for param in raft_model.parameters():
        param.requires_grad = False

    raft_model = raft_model.eval()

    # Write and save RAFT computed motion vectors
    frames_path = parent_path / "Colour"
    motion_vectors_path = parent_path / "MotionVectorRAFT"
    instances = os.listdir(frames_path)
    for instance in instances:
        instance_frames_path = frames_path / instance
        instance_motion_vectors_path = motion_vectors_path / instance
        instance_motion_vectors_path.mkdir(parents=True, exist_ok=True)

        prev_frame = None

        frames = os.listdir(instance_frames_path)
        for frame_num, frame in enumerate(frames):
            frame_path = instance_frames_path / frame
            motion_vector_path = instance_motion_vectors_path / f"{frame_num:04d}.exr"

            frame = decode_image(frame_path.resolve(), mode=ImageReadMode.RGB)[0:3].to(device, dtype=torch.float32)
            frame = 2.0 * (frame / 255.0) - 1.0  # RAFT expects gamma-corrected frames normalised to [-1, 1]
            frame = frame.unsqueeze(0)
            _, _, H, W = frame.shape

            if prev_frame is not None:
                # Compute optical flow with direction prev_frame -> curr_frame
                optical_flow = raft_model(prev_frame, frame)[-1]

                # Normalise optical flow to [-1, 1] because output is in pixels
                optical_flow[:, 0] /= W  
                optical_flow[:, 1] /= H 

                optical_flow = optical_flow.squeeze(0)
                optical_flow = torch.cat([optical_flow, torch.zeros((1, H, W)).to(device)])
                optical_flow = optical_flow.permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
                optical_flow = optical_flow.cpu().numpy()
            else:
                optical_flow = np.zeros((3, H, W), dtype=np.float32)
                optical_flow = np.transpose(optical_flow, (1, 2, 0))  # (3, H, W) -> (H, W, 3)

            imageio.v3.imwrite(motion_vector_path, optical_flow)

            prev_frame = frame

        print(f"Processed instance {instance}")


def generate_raft_motion_vectors_zarr(input_root_path: Path, output_root_path: Path) -> None:
    for directory_path, _, _ in os.walk(input_root_path):
        directory_path = Path(directory_path)
        if directory_path.parent.name == "MotionVectorRAFT":
            relative_path = os.path.relpath(directory_path, input_root_path)
            channels = 2
            output_path = output_root_path / f"{relative_path}.zarr"
            generate_zarr(directory_path, output_path, channels)


if __name__ == "__main__":
    generate_raft_motion_vectors_zarr(
        input_root_path=Path("../data/training_data/VR"),
        output_root_path=Path("../data/training_data/VR_zarr")
    )

    # generate_raft_motion_vectors(parent_path=Path("../data/training_data/VR/FantasticVillage/720x800/MipBiasMinus1Jittered/Left"))
    # generate_raft_motion_vectors(parent_path=Path("../data/training_data/VR/FantasticVillage/720x800/MipBiasMinus1Jittered/Right"))
    # generate_raft_motion_vectors(parent_path=Path("../data/validation_data/VR/FantasticVillage/720x800/MipBiasMinus1Jittered/Left"))
    # generate_raft_motion_vectors(parent_path=Path("../data/validation_data/VR/FantasticVillage/720x800/MipBiasMinus1Jittered/Right"))

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
    
    # folder_path = Path("../data/validation_data/VR/FantasticVillage")
    # prepare_data(folder_path)
    # instance_evaluation_data(folder_path / "720x800/MipBiasMinus1Jittered/Left")
    # instance_evaluation_data(folder_path / "720x800/MipBiasMinus1Jittered/Right")
    # instance_evaluation_data(folder_path / "1440x1600/Enhanced/Left")
    # instance_evaluation_data(folder_path / "1440x1600/Enhanced/Right")
    # instance_evaluation_data(folder_path / "720x800/MipBiasMinus1Jittered")  # Subsample CameraData

    # folder_path = Path("../data/test_data/VR/FantasticVillage")
    # prepare_data(folder_path)
    # instance_evaluation_data(folder_path / "720x800/MipBiasMinus1Jittered/Left")
    # instance_evaluation_data(folder_path / "720x800/MipBiasMinus1Jittered/Right")
    # instance_evaluation_data(folder_path / "1440x1600/Enhanced/Left")
    # instance_evaluation_data(folder_path / "1440x1600/Enhanced/Right")
    # instance_evaluation_data(folder_path / "720x800/MipBiasMinus1Jittered")  # Subsample CameraData

    # write_video(
    #     input_path=Path("../data/validation_data/VR_mono/FantasticVillage/1440x1600/Enhanced/0000"),
    #     output_path=Path("fantastic_village.mp4"),
    #     fps=60
    # )

    # evaluate(
    #     Path("../data/test_data/QRISP/TestSet/SeaPort/540p/MipBiasMinus1/0000"),
    #     Path("../data/test_data/QRISP/TestSet/SeaPort/1080p/Enhanced/0000")
    # )
