import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import imageio.v3
from natsort import natsorted
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.io import decode_image

from metrics.metrics import Metrics
from metrics.vr_metrics import VRMetrics
from network.vr_network import VRConfig


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


# -------------------------------------------------------------------------
# -------------------------- VR bicubic baseline --------------------------
# -------------------------------------------------------------------------

def evaluate_vr(pred_path: Path, target_path: Path) -> None:
    pairs = list(zip(sorted(os.listdir(pred_path / "Left")), sorted(os.listdir(target_path / "Left"))))[:600]

    vr_config = VRConfig(
        camera_baseline=6.4, 
        horizontal_fov=100.0,
        vertical_fov=105.8809,
        horizontal_resolution=1440,
        vertical_resolution=1600
    )

    metrics = VRMetrics(
        dataset_size=len(pairs),
        padding=0,
        iterations=0,
        vr_config=vr_config,
        is_stationary_segment=True,
        display_name="standard_hmd"
    )

    cuda0 = torch.device("cuda:0")

    for frame_num, (pred_name, target_name) in enumerate(pairs):
        # Left eye
        left_pred = decode_image((pred_path / "Left" / pred_name).resolve())[0:3]
        left_pred = left_pred.to(cuda0).to(torch.float32) / 255.0
        left_pred = left_pred.unsqueeze(0)
        left_pred = F.interpolate(
            left_pred,
            scale_factor=4, 
            mode="bicubic",
            align_corners=False
        )
        left_pred = torch.clamp(left_pred, 0.0, 1.0)

        left_target = decode_image((target_path / "Left" / target_name).resolve())[0:3]
        left_target = left_target.to(cuda0).to(torch.float32) / 255.0
        left_target = left_target.unsqueeze(0)

        # Right eye
        right_pred = decode_image((pred_path / "Right" / pred_name).resolve())[0:3]
        right_pred = right_pred.to(cuda0).to(torch.float32) / 255.0
        right_pred = right_pred.unsqueeze(0)
        right_pred = F.interpolate(
            right_pred,
            scale_factor=4, 
            mode="bicubic",
            align_corners=False
        )
        right_pred = torch.clamp(right_pred, 0.0, 1.0)

        right_target = decode_image((target_path / "Right" / target_name).resolve())[0:3]
        right_target = right_target.to(cuda0).to(torch.float32) / 255.0
        right_target = right_target.unsqueeze(0)

        metrics.record(left_pred, left_target, "left")
        metrics.record(right_pred, right_target, "right")
        print(f"{frame_num=}")

    metrics.report("FantasticVillage", len(pairs))


if __name__ == "__main__":
    # write_video(
    #     input_path=Path("../data/validation_data/VR_mono/FantasticVillage/1440x1600/Enhanced/0000"),
    #     output_path=Path("fantastic_village.mp4"),
    #     fps=60
    # )

    # evaluate_filter_exr_png(
    #     Path("../saved/comparison-videos/VR/EnhancedForward")
    # )

    # evaluate(
    #     Path("../data/test_data/QRISP/TestSet/SeaPort/540p/MipBiasMinus1/0000"),
    #     Path("../data/test_data/QRISP/TestSet/SeaPort/1080p/Enhanced/0000")
    # )

    evaluate_vr(
        Path("../saved/comparison-videos/VR/360x400Stationary"),
        Path("../saved/comparison-videos/VR/EnhancedStationary")
    )
