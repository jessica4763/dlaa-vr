from dataclasses import dataclass, field
import imageio.v3
import math
from natsort import natsorted
import os
from pathlib import Path
import torch
from torchvision.utils import save_image


class Scene:
    def __init__(
        self,
        scene_input_imgs_path: Path,
        scene_output_imgs_path: Path,
        colour_path_suffix: str,
    ) -> None:
        self.scene_input_imgs_path = scene_input_imgs_path
        self.scene_output_imgs_path = scene_output_imgs_path

        instances = os.listdir(scene_input_imgs_path / colour_path_suffix)
        frames = os.listdir(scene_input_imgs_path / colour_path_suffix / instances[0])

        self.num_instances = len(instances)
        self.num_frames_per_instance = len(frames)
        self.num_frames = self.num_instances * self.num_frames_per_instance


@dataclass
class VRConfig:
    camera_baseline: float = 0.065
    diagonal_fov: float = 110.0
    horizontal_resolution: int = 1440
    vertical_resolution: int = 1600
    focal_length: float = field(init=False)

    @staticmethod
    def get_focal_length(
        diagonal_fov: float, 
        horizontal_resolution: int, 
        vertical_resolution: int
    ) -> float:
        diagonal_fov_rad = math.radians(diagonal_fov)
        diagonal = math.sqrt(horizontal_resolution ** 2 + vertical_resolution ** 2)
        focal_length = diagonal / (2 * math.tan(diagonal_fov_rad / 2))
        return focal_length

    def __post_init__(self):
        self.focal_length = VRConfig.get_focal_length(self.diagonal_fov, self.horizontal_resolution, self.vertical_resolution)


def gamma_to_linear(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32) / 255.0
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4
    )


def linear_to_gamma(image: torch.Tensor) -> torch.Tensor:
    image = torch.clamp(image, 0.0, 1.0)

    # Adding this doesn't change the result of the computation at all, 
    # but sidesteps the fact that PyTorch computes the gradients of 
    # both branches of torch.where, which could result in NaN 
    # if the gradient of one of the branches is NaN even if that 
    # branch wasn't going to be taken anyway
    max_image = torch.maximum(image, torch.tensor(0.0031308, device=image.device))

    return torch.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * (max_image ** (1.0 / 2.4)) - 0.055
    )


def cumsum(xs):
    cumsum_xs = [0]
    for x in xs:
        cumsum_xs.append(cumsum_xs[-1] + x)

    return cumsum_xs


def write_frames(
    evaluation_output_path: Path,
    frames: torch.Tensor,
    batch: int
) -> None:
    for idx, frame in enumerate(frames):
        save_image(frame, evaluation_output_path / f"{batch + idx}.png")


def write_video(
    imgs_path: Path,
    filename: str,
    fps: int = 24
) -> None:
    writer = imageio.get_writer(
        imgs_path / filename,
        fps=fps,
        codec="libx264",
        quality=10,
        pixelformat='yuv420p',
        macro_block_size=8
    )

    imgs_path = imgs_path / "pred"
    for img_name in natsorted(os.listdir(imgs_path)):
        img_path = imgs_path / img_name
        img = imageio.v3.imread(img_path)
        writer.append_data(img)

    writer.close()


if __name__ == "__main__":
    write_video(
        Path("evaluation_outputs"),
        "540pEnhanced.mp4",
        fps=24
    )
