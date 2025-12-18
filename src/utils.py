import imageio.v3
from natsort import natsorted
import os
from pathlib import Path
import torch
from torchvision.utils import save_image


def gamma_to_linear(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32) / 255.0

    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4
    )


def linear_to_gamma(image: torch.Tensor) -> torch.Tensor:
    image = torch.clamp(image, 0.0, 1.0)
    return torch.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * (image ** (1.0 / 2.4)) - 0.055
    )


def cumsum(xs):
    cumsum_xs = [0]
    for x in xs:
        cumsum_xs.append(cumsum_xs[-1] + x)

    return cumsum_xs


def write_frames(
    eval_output_path: Path,
    frames: torch.Tensor,
    batch: int
) -> None:
    for idx, frame in enumerate(frames):
        save_image(frame, eval_output_path / f"{batch + idx}.png")


def write_video(
    imgs_path: Path,
    filename: str,
    fps: int = 24
) -> None:
    writer = imageio.get_writer(imgs_path / filename, fps=fps, codec='libx264', quality=10)

    imgs_path = imgs_path / "pred"
    for img_name in natsorted(os.listdir(imgs_path)):
        img_path = imgs_path / img_name
        img = imageio.v3.imread(img_path)
        writer.append_data(img)

    writer.close()
