import hydra
from omegaconf import DictConfig
import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.utils import save_image

from datasets import ToyDataset
from model import QualcommNetwork


def train_epoch(
    device: str,
    model: nn.Module,
    training_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    writer: SummaryWriter,
    epoch: int
) -> None:

    dataset_size = len(training_dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(training_dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar(
            "loss/train",
            loss.item(),
            epoch * len(training_dataloader) + batch
        )

        loss, current_img = loss.item(), (batch + 1) * len(X)
        print(f"Loss: {loss:>7f}  [{current_img:>5d}/{dataset_size:>5d}]")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # For reproducibility
    torch.manual_seed(cfg['training']['seed'])
    torch.backends.cudnn.benchmark = False  # Deterministically select an algorithm; reduces efficiency
    torch.use_deterministic_algorithms(True)  # Use only deterministic algorithms

    # Does not use unitialised memory as an input to an operation
    torch.utils.deterministic.fill_uninitialized_memory = False

    training_input_img_dir = Path(cfg['dataset']['training-input-img-path'])
    training_output_img_dir = Path(cfg['dataset']['training-output-img-path'])
    instances = os.listdir(training_input_img_dir)
    frames = os.listdir(training_input_img_dir / instances[0])
    num_instances = len(instances)
    num_frames_per_instance = len(frames)
    training_data = ToyDataset(
        training_input_img_dir,
        training_output_img_dir,
        num_instances,
        num_frames_per_instance,
        transform=v2.ToDtype(torch.float32, scale=True),
        target_transform=v2.ToDtype(torch.float32, scale=True)
    )
    training_dataloader = DataLoader(
        training_data,
        batch_size=cfg['training']['batch-size'],
        shuffle=cfg['training']['shuffle']
    )

    model = QualcommNetwork(
        num_prev_feature_channels=3,
        hidden_channels=32,
        num_blocks=3,
        upscale_factor=1
    ).to(device)

    loss_function = nn.L1Loss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg['training']['learning-rate']
    )

    writer = SummaryWriter(log_dir=cfg['logging']['tensorboard-dir'])

    for epoch in range(cfg['training']['epochs']):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_epoch(
            device,
            model,
            training_dataloader,
            loss_function,
            optimizer,
            writer,
            epoch
        )
        checkpoint(
            device,
            model,
            training_data,
            epoch
        )

    saved_models_path = Path(cfg['training']['saved-models-path'])
    torch.save(model.state_dict(), saved_models_path.resolve())

    print("Done.")

    writer.flush()
    writer.close()


def checkpoint(
    device: str,
    model: nn.Module,
    training_data: Dataset,
    epoch: int
) -> None:
    checkpoint_img, _ = training_data[0]
    checkpoint_img = torch.unsqueeze(checkpoint_img, 0)
    checkpoint_img = checkpoint_img.to(device)
    anti_aliased_img = model(checkpoint_img)
    save_image(anti_aliased_img, f"checkpoints/{epoch}.png")


if __name__ == "__main__":
    train()
