from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig

from datasets.qualcomm_dataset import QualcommDataset
from datasets.vr_dataset import VRDataset
from models.qualcomm_network import QualcommNetwork
from models.vr_network import VRNetwork
from metrics import Metrics
from utils import (
    gamma_to_linear,
    linear_to_gamma,
    write_frames,
    write_video,
    VRConfig
)
from utils import print_parameters


def evaluate(
    device: str,
    model: nn.Module,
    evaluation_dataloader: DataLoader,
    loss_function: nn.Module,
    metrics: Metrics,
    evaluation_output_path_pred: Path,
    evaluation_output_path_target: Path,
    scale_factor: int = 1,
    use_jitter: bool = False
) -> None:
    model.eval()
    with torch.no_grad():
        prev_pred_frame = prev_features = None
        for batch, (inputs, motion_vectors, jitter, target, curr_frame_num) in enumerate(evaluation_dataloader):
            # -------------------------------------------------------------------------
            # ----------------------------- Prepare inputs ----------------------------
            # -------------------------------------------------------------------------
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

            # -------------------------------------------------------------------------
            # -------------------------------- Predict --------------------------------
            # -------------------------------------------------------------------------
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
            write_frames(evaluation_output_path_pred, gamma_pred_frame, batch)
            write_frames(evaluation_output_path_target, gamma_target, batch)


def evaluate_vr(
    device: str,
    model: nn.Module,
    evaluation_dataloader: DataLoader,
    loss_function: nn.Module,
    metrics: Metrics,
    evaluation_output_path_pred: Path,
    evaluation_output_path_target: Path,
    scale_factor: int = 1,
    use_jitter: bool = False
) -> None:
    model.eval()
    with torch.no_grad():
        prev_pred_left_frame = prev_pred_right_frame = prev_left_features = prev_right_features = None
        for batch, data in enumerate(evaluation_dataloader):
            # -------------------------------------------------------------------------
            # ----------------------------- Prepare inputs ----------------------------
            # -------------------------------------------------------------------------
            (
                left_inputs,
                right_inputs, 
                left_motion_vectors, 
                right_motion_vectors, 
                prev_left_depth,
                prev_right_depth,
                jitter,
                left_target, 
                right_target, 
                curr_frame_num
            ) = data
            left_inputs = left_inputs.to(device, non_blocking=True)
            left_inputs = left_inputs.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1
            right_inputs = right_inputs.to(device, non_blocking=True)
            right_inputs = right_inputs.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1

            prev_left_depth = prev_left_depth.to(device, non_blocking=True)
            prev_left_depth = prev_left_depth.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1
            prev_right_depth = prev_right_depth.to(device, non_blocking=True)
            prev_right_depth = prev_right_depth.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1

            left_motion_vectors = left_motion_vectors.to(device, non_blocking=True)
            left_motion_vectors = left_motion_vectors.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1
            right_motion_vectors = right_motion_vectors.to(device, non_blocking=True)
            right_motion_vectors = right_motion_vectors.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1

            if use_jitter:
                jitter = jitter.to(device, non_blocking=True)
                jitter = jitter.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1
            else:
                jitter = None

            left_target = left_target.to(device, non_blocking=True)
            right_target = right_target.to(device, non_blocking=True)

            # Use the previously predicted frame and features during evaluation
            if None not in (prev_pred_left_frame, prev_pred_right_frame, prev_left_features, prev_right_features) and curr_frame_num > 0:
                c0 = model.left_in_channels - (model.num_prev_left_colour + model.num_prev_left_feature)
                c1 = model.left_in_channels - model.num_prev_left_feature

                left_inputs[:, :, c0:c1] = prev_pred_left_frame
                left_inputs[:, :, c1:model.left_in_channels] = prev_left_features
                right_inputs[:, :, c0:c1] = prev_pred_right_frame
                right_inputs[:, :, c1:model.left_in_channels] = prev_right_features

            # -------------------------------------------------------------------------
            # -------------------------------- Predict --------------------------------
            # -------------------------------------------------------------------------
            output_N, output_C, output_H, output_W = left_target.shape
            pred_left_frame, pred_right_frame, left_features, right_features, _, _ = model(
                left_inputs,
                right_inputs,
                left_motion_vectors, 
                right_motion_vectors,
                prev_left_depth,
                prev_right_depth,
                curr_frame_num, 
                jitter,
                "evaluation"
            )
            pred_left_frame = pred_left_frame.view(-1, output_C, output_H, output_W)
            pred_right_frame = pred_right_frame.view(-1, output_C, output_H, output_W)

            prev_pred_left_frame = F.pixel_unshuffle(pred_left_frame, downscale_factor=scale_factor).unsqueeze(0)
            prev_left_features = left_features.unsqueeze(0)
            prev_pred_right_frame = F.pixel_unshuffle(pred_right_frame, downscale_factor=scale_factor).unsqueeze(0)
            prev_right_features = right_features.unsqueeze(0)

            gamma_pred_left_frame = linear_to_gamma(pred_left_frame)
            gamma_left_target = linear_to_gamma(left_target)
            gamma_pred_right_frame = linear_to_gamma(pred_right_frame)
            gamma_right_target = linear_to_gamma(right_target)

            left_frame_loss = loss_function(gamma_pred_left_frame, gamma_left_target)
            left_frame_loss = left_frame_loss.item()
            right_frame_loss = loss_function(gamma_pred_right_frame, gamma_right_target)
            right_frame_loss = right_frame_loss.item()
            current_img = (batch + 1) * len(left_inputs)
            print(f"left_frame_loss: {left_frame_loss:>7f} | right_frame_loss: {right_frame_loss:>7f}  [{current_img:>5d}/{len(evaluation_dataloader.dataset):>5d}]")

            metrics.record(gamma_pred_left_frame, gamma_left_target)
            metrics.record(gamma_pred_right_frame, gamma_right_target)
            write_frames(evaluation_output_path_pred / "left", gamma_pred_left_frame, batch)
            write_frames(evaluation_output_path_target / "left", gamma_left_target, batch)
            write_frames(evaluation_output_path_pred / "right", gamma_pred_right_frame, batch)
            write_frames(evaluation_output_path_target / "right", gamma_right_target, batch)


def run(cfg: DictConfig, writer: SummaryWriter, iterations: int) -> None:
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    evaluation_output_path_pred = Path(cfg["paths"]["evaluation-output-path"]) / "pred"
    (evaluation_output_path_pred / "left").mkdir(parents=True, exist_ok=True)
    (evaluation_output_path_pred / "right").mkdir(parents=True, exist_ok=True)

    evaluation_output_path_target = Path(cfg["paths"]["evaluation-output-path"]) / "target"
    (evaluation_output_path_target / "left").mkdir(parents=True, exist_ok=True)
    (evaluation_output_path_target / "right").mkdir(parents=True, exist_ok=True)

    checkpoints_path = Path(cfg["paths"]["checkpoints-path"])
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    sanity_checks_output_path = Path(cfg["paths"]["sanity-checks-output-path"])
    sanity_checks_output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # ------------------------------ Efficiency -------------------------------
    # -------------------------------------------------------------------------
    torch.backends.cudnn.benchmark = True

    # -------------------------------------------------------------------------
    # ------------------------------- Constants -------------------------------
    # -------------------------------------------------------------------------
    scale_factor = cfg["dataset"]["output-frame-height"] // cfg["dataset"]["input-frame-height"]

    # -------------------------------------------------------------------------
    # --------------------------------- Data ----------------------------------
    # -------------------------------------------------------------------------
    if cfg["dataset"]["is-vr"]:
        evaluation_data = VRDataset(
            input_imgs_path=cfg["dataset"]["validation-input-img-path"],
            output_imgs_path=cfg["dataset"]["validation-output-img-path"],
            input_frame_height=cfg["dataset"]["input-frame-height"],
            input_frame_width=cfg["dataset"]["input-frame-width"],
            camera_data_path_suffix=cfg["dataset"]["camera-data-path-suffix"],
            input_path_suffix=cfg["dataset"]["input-path-suffix"],
            jittered_input_path_suffix=cfg["dataset"]["jittered-input-path-suffix"],
            colour_path_suffix=cfg["dataset"]["colour-path-suffix"],
            depth_path_suffix=cfg["dataset"]["depth-path-suffix"],
            motion_vector_path_suffix=cfg["dataset"]["motion-vector-path-suffix"],
            scene_names=cfg["dataset"]["scene-names"],
            use_jitter=cfg["setup"]["jitter"],
            scale_factor=scale_factor,
            dilation_block_size=cfg["dataset"]["dilation-block-size"],
            transform=gamma_to_linear,
            target_transform=gamma_to_linear,
            mode="evaluation"
        )
    else:
        evaluation_data = QualcommDataset(
            input_imgs_path=cfg["dataset"]["validation-input-img-path"],
            output_imgs_path=cfg["dataset"]["validation-output-img-path"],
            input_frame_height=cfg["dataset"]["input-frame-height"],
            input_frame_width=cfg["dataset"]["input-frame-width"],
            camera_data_path_suffix=cfg["dataset"]["camera-data-path-suffix"],
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

    evaluation_dataloader = DataLoader(evaluation_data)

    # -------------------------------------------------------------------------
    # --------------------------------- Model ---------------------------------
    # -------------------------------------------------------------------------
    if cfg["dataset"]["is-vr"]:
        vr_config = VRConfig(
            camera_baseline=cfg["dataset"]["camera-baseline"], 
            horizontal_fov=cfg["dataset"]["horizontal-fov"],
            vertical_fov=cfg["dataset"]["vertical-fov"],
            horizontal_resolution=cfg["dataset"]["input-frame-width"],
            vertical_resolution=cfg["dataset"]["input-frame-height"]
        )

        model = VRNetwork(
            vr_config=vr_config,
            hidden_channels=cfg["model"]["hidden-channels"],
            num_blocks=cfg["model"]["num-blocks"],
            scale_factor=scale_factor,
            use_jitter=cfg["setup"]["jitter"]
        ).to(device)
    else:
        vr_config = None
        
        model = QualcommNetwork(
            hidden_channels=cfg["model"]["hidden-channels"],
            num_blocks=cfg["model"]["num-blocks"],
            scale_factor=scale_factor,
            use_jitter=cfg["setup"]["jitter"]
        ).to(device)

    model.load_state_dict(
        torch.load(
            cfg["paths"]["saved-models-path"],
            weights_only=True,
            map_location=device
        )
    )

    print_parameters(
        evaluation_output_path=Path(cfg["paths"]["evaluation-output-path"]),
        parameters=model.state_dict()
    )

    # -------------------------------------------------------------------------
    # -------------------------------- Metrics --------------------------------
    # -------------------------------------------------------------------------    
    loss_function = nn.L1Loss()

    metrics = Metrics(
        dataset_size=len(evaluation_dataloader.dataset),  # The total number of frames in the dataset
        padding=cfg["validation"]["padding"],
        iterations=iterations,
        writer=writer,
        vr_config=vr_config,
        is_stationary_segment=cfg["dataset"]["is-stationary-segment"],
        display_name=cfg["setup"]["display-name"],
    )

    # -------------------------------------------------------------------------
    # ------------------------------- Evaluation ------------------------------
    # ------------------------------------------------------------------------- 
    if cfg["dataset"]["is-vr"]:
        evaluate_vr(
            device=device,
            model=model,
            evaluation_dataloader=evaluation_dataloader,
            loss_function=loss_function,
            metrics=metrics,
            evaluation_output_path_pred=evaluation_output_path_pred,
            evaluation_output_path_target=evaluation_output_path_target,
            scale_factor=scale_factor,
            use_jitter=cfg["setup"]["jitter"]
        )
    else:
        evaluate(
            device=device,
            model=model,
            evaluation_dataloader=evaluation_dataloader,
            loss_function=loss_function,
            metrics=metrics,
            evaluation_output_path_pred=evaluation_output_path_pred,
            evaluation_output_path_target=evaluation_output_path_target,
            scale_factor=scale_factor,
            use_jitter=cfg["setup"]["jitter"]
        )

    # -------------------------------------------------------------------------
    # --------------------------------- Output --------------------------------
    # ------------------------------------------------------------------------- 
    metrics.report(scene_name=cfg["dataset"]["scene-names"][0])

    if cfg["dataset"]["is-vr"]:
        write_video(
            imgs_path=evaluation_output_path_pred / "left",
            filename="evaluation_output.mp4",
            fps=24
        )
        write_video(
            imgs_path=evaluation_output_path_pred / "right",
            filename="evaluation_output.mp4",
            fps=24
        )
    else:
        write_video(
            imgs_path=evaluation_output_path_pred,
            filename="evaluation_output.mp4",
            fps=24
        )


def main() -> None:
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="vr-validation")
        writer = SummaryWriter(log_dir=cfg["paths"]["tensorboard-path"])

        run(cfg=cfg, writer=writer, iterations=0)


if __name__ == "__main__":
    main()
