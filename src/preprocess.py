import cv2
import os
import numpy as np
from pathlib import Path
import skimage.io as io
import torch
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.utils import save_image

from metrics import Metrics
from utils import linear_to_gamma


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

    cuda0 = torch.device('cuda:0')

    for frame_num, (pred_name, target_name) in enumerate(pairs):
        pred = decode_image((pred_path / pred_name).resolve())[0:3, ...]
        pred = pred.to(cuda0).to(torch.float32) / 255.0
        pred = pred.unsqueeze(0)
        pred = F.interpolate(
            pred,
            scale_factor=2, 
            mode='bicubic',
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


def TAA_benchmarks(pred_path: Path, target_path: Path) -> None:
    # Reproduces the bicubic baseline to confirm metrics are calculated correctly
    pairs = list(zip(sorted(os.listdir(pred_path)), sorted(os.listdir(target_path))))

    metrics = Metrics(
        dataset_size=len(pairs),
        padding=0,
        iterations=0,
        display_name="standard_fhd"
    )

    cuda0 = torch.device('cuda:0')

    for frame_num, (pred_name, target_name) in enumerate(pairs):
        pred = decode_image((pred_path / pred_name).resolve())[0:3, ...]
        pred = pred.to(cuda0).to(torch.float32) / 255.0
        pred = torch.clamp(pred, 0.0, 1.0)
        pred.unsqueeze(0)

        target = decode_image((target_path / target_name).resolve())[0:3, ...]
        target = target.to(cuda0).to(torch.float32) / 255.0
        target = torch.clamp(target, 0.0, 1.0)
        target.unsqueeze(0)

        metrics.record(pred, target)
        print(f"{frame_num=}")

    metrics.report("FantasticVillage")


def filter_exr_png(folder_path: Path) -> None:
    for file in folder_path.rglob("*"):
        if not file.is_file():
            continue

        parent_name = file.parent.parent.name
        suffix = file.suffix.lower()

        if suffix == ".exr" and parent_name == "Colour":
            file.unlink()

        if suffix == ".png" and parent_name in {"Depth", "MotionVector"}:
            file.unlink()


def subsample(folder_path: Path, take: int = 30, skip: int = 60) -> None:
    for directory_name in ("Colour", "Depth", "MotionVector"):
        file_path = folder_path / directory_name
        files = sorted(file_path.glob("*.*"))
        kept = [f for i, f in enumerate(files) if i % skip < take]
        for file_index, file_start in enumerate(range(0, len(kept), take)):
            dest = file_path / f"{file_index:04d}"
            dest.mkdir()
            for f in kept[file_start:file_start + take]:
                f.rename(dest / f.name)


def prepare(folder_path: Path) -> None:
    renames = {
        "BP_VRStereoRig(1)": "Left",
        "BP_VRStereoRig(2)": "Right",
        "FinalImage": "Colcour",
        "FinalImageDepth": "Depth",
        "FinalImageMotionVector": "MotionVector",
    }

    for directory_path, directory_names, _ in os.walk(folder_path, topdown=False):
        for name in directory_names:
            if name in renames:
                Path(directory_path, name).rename(Path(directory_path, renames[name]))

    filter_exr_png(folder_path)
    subsample(folder_path)
    

if __name__ == "__main__":
    # prepare(Path("../data/training_data/VR"))
    prepare(Path("../data/test_data/VR"))

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
