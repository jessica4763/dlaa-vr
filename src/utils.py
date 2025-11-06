import cv2
import os
import skimage.io as io


def downsample() -> None:
    input_path = "../data/training_data/QRISP/FloodedGrounds/1080p/Enhanced"
    output_path = "../data/training_data/QRISP/FloodedGrounds/540p/Enhanced"
    output_resolution = (960, 540)

    for image_name in os.listdir(input_path):
        image = io.imread(os.path.join(input_path, image_name))
        downsampled_image = cv2.resize(
            image,
            output_resolution,
            interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(output_path, downsampled_image)


if __name__ == "__main__":
    downsample()
