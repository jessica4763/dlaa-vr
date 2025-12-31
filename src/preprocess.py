import cv2
import os
import numpy as np
from pathlib import Path
import skimage.io as io
import torch
from torchvision.io import decode_image

from metrics import Metrics


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


def downsample(
    input_path: Path,
    output_path: Path,
    output_dimensions: tuple[int, int],
) -> None:
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
    pairs = list(zip(os.listdir(pred_path), os.listdir(target_path)))

    metrics = Metrics(
        len(pairs),  # The total number of frames in the dataset
        display_name="standard_fhd"
    )

    cuda0 = torch.device('cuda:0')

    for pred_name, target_name in pairs:
        pred = decode_image((pred_path / pred_name).resolve())[0:3, ...]
        pred = pred.to(cuda0) / 255.0
        target = decode_image((target_path / target_name).resolve())[0:3, ...]
        target = target.to(cuda0) / 255.0
        metrics.record(pred, target)

    metrics.report()


if __name__ == "__main__":
    evaluate(
        Path("../data/test_data/QRISP/TestSet/SeaPort/270p/MipBiasMinus2/0000"),
        Path("../data/test_data/QRISP/TestSet/SeaPort/270p/Enhanced/0000")
    )

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

    # test_data_scenes = [
    #     "AbandonedSchool",
    #     "SeaPort",
    #     "SpaceShipDemo"
    # ]
    # test_data_prefix = Path("../data/test_data/QRISP/TestSet")
    # test_data_input_suffix = Path("1080p/Enhanced")
    # test_data_output_suffix = Path("540p/Enhanced")
    # for test_data_scene in test_data_scenes:
    #     downsample(
    #         test_data_prefix / test_data_scene / test_data_input_suffix,
    #         test_data_prefix / test_data_scene / test_data_output_suffix,
    #         output_dimensions
    #     )
