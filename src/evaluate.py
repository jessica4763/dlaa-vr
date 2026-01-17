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
    gamma_to_linear,
    linear_to_gamma,
    write_frames,
    write_video
)
from sanity_checks import print_parameters


def evaluate(
    device: str,
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    metrics: Metrics,
    eval_output_path: Path,
    use_jitter: bool = False
) -> None:
    model.eval()
    with torch.no_grad():
        prev_pred_frame = prev_features = None
        for batch, (inputs, motion_vectors, jitter, target) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1

            output_N, output_C, output_H, output_W = motion_vectors.shape
            motion_vectors = motion_vectors.to(device)
            motion_vectors = motion_vectors.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1

            if use_jitter:
                jitter = jitter.to(device)
                jitter = jitter.unsqueeze(0)  # N = batch_size * clip_size = 1 * 1
            else:
                jitter = None

            target = target.to(device)

            # Use the previously predicted frame and features during test
            if prev_pred_frame is not None and prev_features is not None:
                c0 = model.in_channels - (model.num_prev_colour + model.num_prev_feature)
                c1 = model.in_channels - model.num_prev_feature
                inputs[:, :, c0:c1] = prev_pred_frame
                inputs[:, :, c1:model.in_channels] = prev_features

            pred_frame, features = model(inputs, motion_vectors, jitter)
            pred_frame = pred_frame.view(-1, 3, output_H, output_W)
            loss = loss_fn(pred_frame, target)

            prev_pred_frame = F.pixel_unshuffle(pred_frame, downscale_factor=2)
            prev_features = features.squeeze(0)

            loss, current_img = loss.item(), (batch + 1) * len(inputs)
            print(f"Loss: {loss:>7f}  [{current_img:>5d}/{len(test_dataloader.dataset):>5d}]")

            gamma_pred_frame = linear_to_gamma(pred_frame)
            gamma_target_frame = linear_to_gamma(target)
            metrics.record(gamma_pred_frame, gamma_target_frame)
            write_frames(eval_output_path / "pred", gamma_pred_frame, batch)
            write_frames(eval_output_path / "target", gamma_target_frame, batch)


@hydra.main(version_base=None, config_path="../configs", config_name="test")
def main(cfg: DictConfig) -> None:
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
    # ------------------------------ Diagnostics ------------------------------
    # -------------------------------------------------------------------------
    writer = SummaryWriter(log_dir=cfg["paths"]["tensorboard-path"])

    # -------------------------------------------------------------------------
    # --------------------------------- Data ----------------------------------
    # -------------------------------------------------------------------------
    test_data = QualcommDataset(
        cfg["dataset"]["test-input-img-path"],
        cfg["dataset"]["test-output-img-path"],
        cfg["dataset"]["input-frame-height"],
        cfg["dataset"]["input-frame-width"],
        cfg["dataset"]["output-frame-height"],
        cfg["dataset"]["output-frame-width"],
        cfg["dataset"]["camera-data-path-suffix"],
        cfg["dataset"]["ground-truth-path-suffix"],
        cfg["dataset"]["colour-path-suffix"],
        cfg["dataset"]["depth-path-suffix"],
        cfg["dataset"]["motion-vector-path-suffix"],
        cfg["dataset"]["colour-jittered-path-suffix"],
        cfg["dataset"]["depth-jittered-path-suffix"],
        cfg["dataset"]["motion-vector-jittered-path-suffix"],
        cfg["dataset"]["scene_names"],
        use_jitter=cfg["setup"]["jitter"],
        dilation_block_size=cfg["dataset"]["dilation-block-size"],
        transform=gamma_to_linear,
        target_transform=gamma_to_linear,
        mode="test"
    )

    test_dataloader = DataLoader(
        test_data
    )

    # -------------------------------------------------------------------------
    # --------------------------------- Model ---------------------------------
    # -------------------------------------------------------------------------
    model = QualcommNetwork(
        hidden_channels=cfg["model"]["hidden-channels"],
        num_blocks=cfg["model"]["num-blocks"],
        input_frame_height=cfg["dataset"]["input-frame-height"],
        input_frame_width=cfg["dataset"]["input-frame-width"],
        output_frame_height=cfg["dataset"]["output-frame-height"],
        output_frame_width=cfg["dataset"]["output-frame-width"],
        use_jitter=cfg["setup"]["jitter"]
    ).to(device)

    model.load_state_dict(
        torch.load(
            cfg["paths"]["saved-models-path"],
            weights_only=True,
            map_location=device
        )
    )

    inputs, motion_vectors, jitter, _ = test_data[0]
    inputs = inputs.to(device).unsqueeze(0).unsqueeze(0)
    jitter = jitter.to(device).unsqueeze(0).unsqueeze(0)  # if use_jitter is None, this is a zero tensor, and is ignored during inference
    writer.add_graph(model, input_to_model=(inputs, motion_vectors, jitter))

    print_parameters(Path(cfg["paths"]["evaluation-output-path"]), model.state_dict())

    # -------------------------------------------------------------------------
    # ------------------------------ Evaluation -------------------------------
    # -------------------------------------------------------------------------
    loss_fn = nn.L1Loss()

    metrics = Metrics(
        len(test_dataloader.dataset),  # The total number of frames in the dataset
        display_name=cfg["setup"]["display-name"], 
        writer=writer
    )

    evaluate(
        device,
        model,
        test_dataloader,
        loss_fn,
        metrics,
        Path(cfg["paths"]["evaluation-output-path"]),
        use_jitter=cfg["setup"]["jitter"]
    )

    metrics.report()

    write_video(
        Path(cfg["paths"]["evaluation-output-path"]),
        "evaluation_output.mp4",
        fps=24
    )


if __name__ == "__main__":
    main()
