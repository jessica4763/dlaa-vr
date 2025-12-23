import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
import sys
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from datasets import QualcommDataset, QualcommDatasetSampler
from model import QualcommNetwork
from sanity_checks import output_input
from utils import gamma_to_linear, linear_to_gamma


def train_epoch(
    device: str,
    model: nn.Module,
    training_dataloader: DataLoader,
    clip_size: int,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    writer: SummaryWriter,
    epoch: int
) -> None:
    dataset_size = len(training_dataloader.dataset)

    model.train()
    for batch, (inputs, targets, motion_vectors) in enumerate(training_dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        motion_vectors = motion_vectors.to(device, non_blocking=True)

        # Sampler gives N = num_batches * clip_size
        N, C, H, W = inputs.shape
        inputs = inputs.view(-1, clip_size, C, H, W)
        motion_vectors = motion_vectors.view(-1, clip_size, 2, H, W)

        # Pass in motion vectors as well, for warping
        pred_frame, _ = model(inputs, motion_vectors)
        pred_frame = pred_frame.view(-1, 3, H, W)

        loss = loss_fn(pred_frame, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar(
            "loss/train",
            loss.item(),
            epoch * len(training_dataloader.dataset) + batch
        )

        loss, current_img = loss.item(), (batch + 1) * N
        print(f"Loss: {loss:>7f}  [{current_img:>5d} / {dataset_size:>5d}]")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    checkpoints_path = Path(cfg["setup"]["checkpoints-path"])
    checkpoints_path.mkdir(parents=True, exist_ok=True)

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
    writer = SummaryWriter(log_dir=cfg["setup"]["tensorboard-dir"])

    # -------------------------------------------------------------------------
    # --------------------------------- Data ----------------------------------
    # -------------------------------------------------------------------------
    training_data = QualcommDataset(
        cfg["dataset"]["training-input-img-path"],
        cfg["dataset"]["training-output-img-path"],
        cfg["dataset"]["ground-truth-path-suffix"],
        cfg["dataset"]["colour-path-suffix"],
        cfg["dataset"]["depth-path-suffix"],
        cfg["dataset"]["camera-data-path-suffix"],
        cfg["dataset"]["motion-vector-path-suffix"],
        cfg["dataset"]["scene_names"],
        jitter=False,
        transform=gamma_to_linear,
        target_transform=gamma_to_linear,
    )

    training_sampler = QualcommDatasetSampler(
        training_data.scenes,
        cfg["optimiser"]["batch-size"],
        cfg["optimiser"]["clip-size"],
    )

    training_dataloader = DataLoader(
        training_data,
        batch_sampler=training_sampler,
        # num_workers=os.cpu_count(),
        # pin_memory=True,
        # persistent_workers=True
    )

    # -------------------------------------------------------------------------
    # --------------------------------- Model ---------------------------------
    # -------------------------------------------------------------------------
    model = QualcommNetwork(
        hidden_channels=cfg["model"]["hidden-channels"],
        num_blocks=cfg["model"]["num-blocks"],
        jitter=False
    ).to(device)

    # Initialise with parameters from a previously trained model if desired
    parameters_path = cfg["model"]["parameters"]
    if parameters_path:
        model.load_state_dict(
            torch.load(
                parameters_path,
                weights_only=True,
                map_location=device
            )
        )

    # model = torch.compile(model)

    # -------------------------------------------------------------------------
    # ----------------------------- Optimisation ------------------------------
    # -------------------------------------------------------------------------
    loss_function = nn.L1Loss()

    if cfg["optimiser"]["name"] == "sgd":
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=cfg["optimiser"]["learning-rate"],
            weight_decay=cfg["optimiser"]["weight-decay"]
        )
    elif cfg["optimiser"]["name"] == "adamw":
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["optimiser"]["learning-rate"],
            betas=cfg["optimiser"]["betas"],
            eps=cfg["optimiser"]["epsilon"],
            weight_decay=cfg["optimiser"]["weight-decay"]
        )
    else:
        sys.exit("Chosen optimiser implementation does not exist.")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimiser,
        milestones=cfg["optimiser"]["learning-rate-milestones"],
        gamma=cfg["optimiser"]["learning-rate-gamma"]
    )

    # -------------------------------------------------------------------------
    # ----------------------------- Training loop -----------------------------
    # -------------------------------------------------------------------------
    for epoch in range(cfg["optimiser"]["epochs"]):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_epoch(
            device,
            model,
            training_dataloader,
            cfg["optimiser"]["clip-size"],
            loss_function,
            optimiser,
            writer,
            epoch
        )
        checkpoint(
            checkpoints_path,
            device,
            model,
            training_data,
            writer,
            epoch
        )
        scheduler.step()

    # Save the model
    torch.save(model.state_dict(), Path(cfg["setup"]["saved-models-path"]))

    # Log the config
    writer.add_text("hyperparams", OmegaConf.to_yaml(cfg))

    writer.flush()
    writer.close()

    print("Done.")


def checkpoint(
    checkpoint_path: Path,
    device: str,
    model: nn.Module,
    training_data: Dataset,
    writer: SummaryWriter,
    epoch: int
) -> None:
    # Strictly a training diagnostic, so it's OK if
    # training data is used here
    model.eval()
    with torch.no_grad():
        X, _, motion_vectors = training_data[0]

        # Verify what exactly goes into the network
        output_input(model, X)

        X = X.to(device).unsqueeze(0).unsqueeze(0)
        motion_vectors = motion_vectors.to(device).unsqueeze(0).unsqueeze(0)
        anti_aliased_img, _ = model(X, motion_vectors)
        anti_aliased_img = anti_aliased_img.squeeze(0).squeeze(0)
        anti_aliased_img = linear_to_gamma(anti_aliased_img)

        save_image(anti_aliased_img, checkpoint_path / f"{epoch}.png")
        if (epoch + 1) % 10 == 0:
            writer.add_image(
                "checkpoint images",
                anti_aliased_img,
                global_step=(epoch + 1)
            )


if __name__ == "__main__":
    train()
