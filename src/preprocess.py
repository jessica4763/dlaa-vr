import cv2
import os
import numpy as np
from pathlib import Path
import skimage.io as io


def gamma_to_linear(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4
    )


def linear_to_gamma(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    image = np.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * (image ** (1 / 2.4)) - 0.055
    )
    return (image * 255.0).round().astype(np.uint8)


def downsample(input_str, output_str) -> None:
    input_path = Path(input_str)
    output_path = Path(output_str)
    output_resolution = (480, 270)

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
                output_resolution,
                interpolation=cv2.INTER_AREA
            )
            downsampled_image = linear_to_gamma(downsampled_image)
            downsampled_image = cv2.cvtColor(downsampled_image, cv2.COLOR_RGB2BGR)
            output_image_path = output_frames_path / frame
            cv2.imwrite(output_image_path, downsampled_image)

        print(f"{instance} done.")


if __name__ == "__main__":
    training_data_scenes = [
        "CBApocalypse",
        "FloodedGrounds",
        "FloodedGroundsBridges",
        "ScifiBase",
        "ScifiBaseNightStartStop",
        "ScifiBaseStartStop",
        "ScifiFacility",
        "SunTemple",
        "SunTempleBush",
        "SunTempleLamps"
    ]

    training_data_prefix = Path("../data/training_data/QRISP")
    training_data_input_suffix = Path("1080p/Enhanced")
    training_data_output_suffix = Path("270p/Enhanced")
    for training_data_scene in training_data_scenes:
        downsample(
            training_data_prefix / training_data_scene / training_data_input_suffix,
            training_data_prefix / training_data_scene / training_data_output_suffix
        )

    test_data_scenes = [
        "AbandonedSchool",
        "SeaPort",
        "SpaceShipDemo"
    ]

    test_data_prefix = Path("../data/test_data/QRISP/TestSet")
    test_data_input_suffix = Path("1080p/Enhanced")
    test_data_output_suffix = Path("270p/Enhanced")
    for test_data_scene in test_data_scenes:
        downsample(
            test_data_prefix / test_data_scene / test_data_input_suffix,
            test_data_prefix / test_data_scene / test_data_output_suffix,
        )
