from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import ToyDataset
from model import QualcommNetwork

import hydra
from omegaconf import DictConfig, OmegaConf


def train_epoch(
    device: str,
    training_dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
) -> None:
    size = len(training_dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(training_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
    training_data = ToyDataset(training_input_img_dir, training_output_img_dir, 21, 30)
    training_dataloader = DataLoader(training_data, batch_size=cfg['loss']['batch_size'])

    model = QualcommNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 3
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_epoch(device, training_dataloader, model, loss_fn, optimizer)
    print("Done.")


if __name__ == "__main__":
    train()
