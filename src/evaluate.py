import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig

from datasets import QualcommDataset
from model import QualcommNetwork
from metrics import Metrics
from utils import (
    checkpoint,
    gamma_to_linear,
    linear_to_gamma,
    write_frames,
    write_video,
    VRConfig
)
from sanity_checks import print_parameters


def evaluate(
    device: str,
    model: nn.Module,
    evaluation_dataloader: DataLoader,
    loss_function: nn.Module,
    metrics: Metrics,
    evaluation_output_path: Path,
    scale_factor: int = 1,
    use_jitter: bool = False
) -> None:
    model.eval()
    with torch.no_grad():
        prev_pred_frame = prev_features = None
        for batch, (inputs, motion_vectors, jitter, target, curr_frame_num) in enumerate(evaluation_dataloader):
            inputs = inputs.to(device, non_blocking=True)
            inputs = inputs.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1

            motion_vectors = motion_vectors.to(device, non_blocking=True)
            motion_vectors = motion_vectors.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1

            if use_jitter:
                jitter = jitter.to(device, non_blocking=True)
                jitter = jitter.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1
            else:
                jitter = None

            target = target.to(device, non_blocking=True)

            # Use the previously predicted frame and features during evaluation
            if prev_pred_frame is not None and prev_features is not None and curr_frame_num > 0:
                c0 = model.in_channels - (model.num_prev_colour + model.num_prev_feature)
                c1 = model.in_channels - model.num_prev_feature
                inputs[:, :, c0:c1] = prev_pred_frame
                inputs[:, :, c1:model.in_channels] = prev_features

            output_N, output_C, output_H, output_W = target.shape
            pred_frame, features, _ = model(inputs, motion_vectors, curr_frame_num, jitter, "evaluation")
            pred_frame = pred_frame.view(-1, output_C, output_H, output_W)

            prev_pred_frame = F.pixel_unshuffle(pred_frame, downscale_factor=scale_factor).unsqueeze(0)
            prev_features = features.unsqueeze(0)

            gamma_pred_frame = linear_to_gamma(pred_frame)
            gamma_target = linear_to_gamma(target)

            loss = loss_function(gamma_pred_frame, gamma_target)
            loss, current_img = loss.item(), (batch + 1) * len(inputs)
            print(f"Loss: {loss:>7f}  [{current_img:>5d}/{len(evaluation_dataloader.dataset):>5d}]")

            metrics.record(gamma_pred_frame, gamma_target)
            write_frames(evaluation_output_path / "pred", gamma_pred_frame, batch)
            write_frames(evaluation_output_path / "target", gamma_target, batch)


def run(cfg: DictConfig, writer: SummaryWriter, iterations: int) -> None:
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    eval_output_pred_path = Path(cfg["paths"]["evaluation-output-path"]) / "pred"
    eval_output_pred_path.mkdir(parents=True, exist_ok=True)

    eval_output_y_path = Path(cfg["paths"]["evaluation-output-path"]) / "target"
    eval_output_y_path.mkdir(parents=True, exist_ok=True)

    checkpoints_path = Path(cfg["paths"]["checkpoints-path"])
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    sanity_checks_output_path = Path(cfg["paths"]["sanity-checks-output-path"])
    sanity_checks_output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # ---------------------------- Reproducibility ----------------------------
    # -------------------------------------------------------------------------
    torch.manual_seed(cfg["setup"]["seed"])

    # Deterministically selecting an algorithm reduces efficiency
    torch.backends.cudnn.benchmark = True

    torch.use_deterministic_algorithms(False)

    # Does not use unitialised memory as an input to an operation
    torch.utils.deterministic.fill_uninitialized_memory = False

    # -------------------------------------------------------------------------
    # ------------------------------- Constants -------------------------------
    # -------------------------------------------------------------------------
    scale_factor = cfg["dataset"]["output-frame-height"] // cfg["dataset"]["input-frame-height"]

    # -------------------------------------------------------------------------
    # --------------------------------- Data ----------------------------------
    # -------------------------------------------------------------------------
    evaluation_data = QualcommDataset(
        input_imgs_path=cfg["dataset"]["validation-input-img-path"],
        output_imgs_path=cfg["dataset"]["validation-output-img-path"],
        input_frame_height=cfg["dataset"]["input-frame-height"],
        input_frame_width=cfg["dataset"]["input-frame-width"],
        camera_data_path_suffix=cfg["dataset"]["camera-data-path-suffix"],
        ground_truth_path_suffix=cfg["dataset"]["ground-truth-path-suffix"],
        colour_path_suffix=cfg["dataset"]["colour-path-suffix"],
        depth_path_suffix=cfg["dataset"]["depth-path-suffix"],
        motion_vector_path_suffix=cfg["dataset"]["motion-vector-path-suffix"],
        colour_jittered_path_suffix=cfg["dataset"]["colour-jittered-path-suffix"],
        depth_jittered_path_suffix=cfg["dataset"]["depth-jittered-path-suffix"],
        motion_vector_jittered_path_suffix=cfg["dataset"]["motion-vector-jittered-path-suffix"],
        scene_names=cfg["dataset"]["scene-names"],
        use_jitter=cfg["setup"]["jitter"],
        scale_factor=scale_factor,
        dilation_block_size=cfg["dataset"]["dilation-block-size"],
        transform=gamma_to_linear,
        target_transform=gamma_to_linear,
        mode="evaluation"
    )

    evaluation_dataloader = DataLoader(
        evaluation_data,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
        persistent_workers=True
    )

    # -------------------------------------------------------------------------
    # --------------------------------- Model ---------------------------------
    # -------------------------------------------------------------------------
    model = QualcommNetwork(
        hidden_channels=cfg["model"]["hidden-channels"],
        num_blocks=cfg["model"]["num-blocks"],
        scale_factor=scale_factor,
        use_jitter=cfg["setup"]["jitter"]
    )
    

    model.load_state_dict(
        torch.load(
            cfg["paths"]["saved-models-path"],
            weights_only=True,
            map_location=device
        )
    )

    model = model.to(device)

    print_parameters(
        evaluation_output_path=Path(cfg["paths"]["evaluation-output-path"]),
        parameters=model.state_dict()
    )

    # -------------------------------------------------------------------------
    # ------------------------------ Evaluation -------------------------------
    # -------------------------------------------------------------------------
    loss_function = nn.L1Loss()
    
    vr_config = None
    if cfg["dataset"]["is-vr"]:
        vr_config = VRConfig(
            cfg["dataset"]["camera-baseline"], 
            cfg["dataset"]["diagonal-fov"],
            cfg["dataset"]["output-frame-width"],
            cfg["dataset"]["output-frame-height"]
        )

    metrics = Metrics(
        dataset_size=len(evaluation_dataloader.dataset),  # The total number of frames in the dataset
        padding=cfg["validation"]["padding"],
        iterations=iterations,
        writer=writer,
        vr_config=vr_config,
        is_stationary_segment=cfg["dataset"]["is-stationary-segment"],
        display_name=cfg["setup"]["display-name"],
    )

    evaluate(
        device=device,
        model=model,
        evaluation_dataloader=evaluation_dataloader,
        loss_function=loss_function,
        metrics=metrics,
        evaluation_output_path=Path(cfg["paths"]["evaluation-output-path"]),
        scale_factor=scale_factor,
        use_jitter=cfg["setup"]["jitter"]
    )

    metrics.report(scene_name=cfg["dataset"]["scene-names"][0])

    write_video(
        imgs_path=Path(cfg["paths"]["evaluation-output-path"]),
        filename="evaluation_output.mp4",
        fps=24
    )

    checkpoint(
        checkpoints_path=checkpoints_path,
        sanity_checks_output_path=sanity_checks_output_path,
        device=device,
        model=model,
        data=evaluation_data,
        iterations=0,
        input_frame_height=cfg["dataset"]["input-frame-height"],
        input_frame_width=cfg["dataset"]["input-frame-width"],
        scale_factor=scale_factor,
        use_jitter=cfg["setup"]["jitter"],
        mode="evaluation"
    )


def main() -> None:
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="validation")
        writer = SummaryWriter(log_dir=cfg["paths"]["tensorboard-path"])

        run(cfg=cfg, writer=writer, iterations=0)

        print("\n --------------------- Stationary Segments Evaluation -------------------- \n")
        stationary_segments_cfg = hydra.compose(config_name="validation", overrides=["dataset=validation-upscale-stationary-segment"])
        run(cfg=stationary_segments_cfg, writer=writer, iterations=0)


if __name__ == "__main__":
    main()
