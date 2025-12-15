import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import ToyDataset
from model import QualcommNetwork
from metrics import Metrics
from utils import gamma_to_linear


def write_frame(
    eval_output_path: Path,
    frames: torch.Tensor,
    batch: int
) -> None:
    for idx, frame in enumerate(frames):
        save_image(frame, eval_output_path / f"{batch + idx}.png")


def evaluate(
    device: str,
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    eval_output_path: Path
) -> None:
    dataset_size = len(test_dataloader.dataset)

    num_batches = len(test_dataloader)
    metrics = Metrics(num_batches)

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss, current_img = loss.item(), (batch + 1) * len(X)
            print(f"Loss: {loss:>7f}  [{current_img:>5d}/{dataset_size:>5d}]")

            metrics.record(pred, y)

            write_frame(eval_output_path, pred, batch)

    metrics.report()


@hydra.main(version_base=None, config_path="../configs", config_name="test")
def main(cfg: DictConfig) -> None:
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # -------------------------------------------------------------------------
    # ---------------------------- Reproducibility ----------------------------
    # -------------------------------------------------------------------------
    torch.manual_seed(cfg['testing']['seed'])

    # Deterministically select an algorithm; reduces efficiency
    torch.backends.cudnn.benchmark = False

    # Use only deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Does not use unitialised memory as an input to an operation
    torch.utils.deterministic.fill_uninitialized_memory = False

    # -------------------------------------------------------------------------
    # --------------------------------- Data ----------------------------------
    # -------------------------------------------------------------------------
    test_data = ToyDataset(
        cfg['dataset']['scene_names'],
        cfg["dataset"]["test-input-img-path"],
        cfg["dataset"]["test-output-img-path"],
        transform=gamma_to_linear,
        target_transform=gamma_to_linear
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=cfg['testing']['batch-size'],
        shuffle=cfg['testing']['shuffle']  # False for evaluation
    )

    # -------------------------------------------------------------------------
    # --------------------------------- Model ---------------------------------
    # -------------------------------------------------------------------------
    model = QualcommNetwork(
        hidden_channels=cfg['model']['hidden-channels'],
        num_blocks=cfg['model']['num-blocks'],
    ).to(device)

    model.load_state_dict(
        torch.load(
            cfg['testing']['saved-models-path'],
            weights_only=True,
            map_location=device
        )
    )

    # -------------------------------------------------------------------------
    # ------------------------------ Evaluation -------------------------------
    # -------------------------------------------------------------------------
    loss_fn = nn.L1Loss()

    evaluate(
        device,
        model,
        test_dataloader,
        loss_fn,
        cfg['testing']['eval-output-path']
    )


if __name__ == "__main__":
    main()
