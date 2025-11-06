import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
import torch
from torch import nn, optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import ToyDataset
from model import QualcommNetwork


def train_epoch(
    device: str,
    training_dataloader: DataLoader,
    model: nn.Module,
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

        writer.add_scalar("loss/train", epoch + epoch * len(training_dataloader))

        loss, current_img = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current_img:>5d}/{dataset_size:>5d}]")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

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
    training_dataloader = DataLoader(training_data, batch_size=cfg['training']['batch-size'])

    model = QualcommNetwork().to(device)

    loss_function = nn.L1Loss()

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['training']['learning-rate'])

    writer = SummaryWriter(log_dir="")

    for epoch in range(cfg['training']['epochs']):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_epoch(
            device,
            training_dataloader,
            model,
            loss_function,
            optimizer,
            writer
        )

    print("Done.")

    writer.flush()


if __name__ == "__main__":
    train()
