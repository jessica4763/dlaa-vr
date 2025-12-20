from pathlib import Path
import torch
from torch import nn
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
    writer: SummaryWriter,
    eval_output_path: Path
) -> None:
    model.eval()
    with torch.no_grad():
        dataset_size = len(test_dataloader.dataset)

        num_batches = len(test_dataloader)
        metrics = Metrics(writer, num_batches)
        
        prev_pred_frame = prev_features = None

        for batch, (X, y, _) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)

            # Use the previously predicted frame and features during test
            if prev_pred_frame is not None and prev_features is not None:
                c0 = model.in_channels - (model.num_prev_colour + model.num_prev_feature)
                c1 = model.in_channels - model.num_prev_feature
                X[:, c0:c1] = prev_pred_frame
                X[:, c1:model.in_channels] = prev_features

            pred_frame, features = model(X)
            loss = loss_fn(pred_frame, y)

            prev_pred_frame = pred_frame
            prev_features = features

            loss, current_img = loss.item(), (batch + 1) * len(X)
            print(f"Loss: {loss:>7f}  [{current_img:>5d}/{dataset_size:>5d}]")

            gamma_pred_frame = linear_to_gamma(pred_frame)
            gamma_y_frame = linear_to_gamma(y)
            metrics.record(gamma_pred_frame, gamma_y_frame)
            write_frames(eval_output_path / "pred", gamma_pred_frame, batch)
            write_frames(eval_output_path / "y", gamma_y_frame, batch)

    metrics.report()


@hydra.main(version_base=None, config_path="../configs", config_name="test")
def main(cfg: DictConfig) -> None:
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    eval_output_pred_path = Path("evaluation_output/pred")
    eval_output_pred_path.mkdir(parents=True, exist_ok=True)

    eval_output_y_path = Path("evaluation_output/y")
    eval_output_y_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # ---------------------------- Reproducibility ----------------------------
    # -------------------------------------------------------------------------
    torch.manual_seed(cfg["setup"]["seed"])

    # Deterministically select an algorithm; reduces efficiency
    torch.backends.cudnn.benchmark = False

    # Use only deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Does not use unitialised memory as an input to an operation
    torch.utils.deterministic.fill_uninitialized_memory = False

    # -------------------------------------------------------------------------
    # ------------------------------ Diagnostics ------------------------------
    # -------------------------------------------------------------------------
    writer = SummaryWriter(log_dir=cfg["setup"]["tensorboard-dir"])

    # -------------------------------------------------------------------------
    # --------------------------------- Data ----------------------------------
    # -------------------------------------------------------------------------
    test_data = QualcommDataset(
        cfg["dataset"]["scene_names"],
        cfg["dataset"]["test-input-img-path"],
        cfg["dataset"]["test-output-img-path"],
        transform=gamma_to_linear,
        target_transform=gamma_to_linear
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=1,
    )

    # -------------------------------------------------------------------------
    # --------------------------------- Model ---------------------------------
    # -------------------------------------------------------------------------
    model = QualcommNetwork(
        hidden_channels=cfg["model"]["hidden-channels"],
        num_blocks=cfg["model"]["num-blocks"],
    ).to(device)

    model.load_state_dict(
        torch.load(
            cfg["setup"]["saved-models-path"],
            weights_only=True,
            map_location=device
        )
    )

    example_input_imgs, _, _= test_data[0]
    example_input_imgs = example_input_imgs.to(device).unsqueeze(0)
    writer.add_graph(model, example_input_imgs)

    print_parameters(Path(cfg["setup"]["eval-output-path"]), model.state_dict())

    # -------------------------------------------------------------------------
    # ------------------------------ Evaluation -------------------------------
    # -------------------------------------------------------------------------
    loss_fn = nn.L1Loss()

    evaluate(
        device,
        model,
        test_dataloader,
        loss_fn,
        writer,
        Path(cfg["setup"]["eval-output-path"])
    )

    write_video(
        Path(cfg["setup"]["eval-output-path"]),
        "evaluation_output.avi",
        fps=24
    )


if __name__ == "__main__":
    main()
