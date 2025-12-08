import cv2
import os
from pathlib import Path
import skimage.io as io


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
            image = io.imread(input_image_path)[:, :, :3]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            downsampled_image = cv2.resize(
                image,
                output_resolution,
                interpolation=cv2.INTER_AREA
            )
            output_image_path = output_frames_path / frame
            cv2.imwrite(output_image_path, downsampled_image)

        print(f"{instance} done.")


if __name__ == "__main__":
    downsample(
        "../data/training_data/QRISP/CBApocalypse/1080p/Enhanced",
        "../data/training_data/QRISP/CBApocalypse/270p/Enhanced",
    )
    downsample(
        "../data/test_data/QRISP/TestSet/AbandonedSchool/1080p/Enhanced",
        "../data/test_data/QRISP/TestSet/AbandonedSchool/270p/Enhanced",
    )
