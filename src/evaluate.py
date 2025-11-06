import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import ToyDataset
from model import QualcommNetwork



def evaluate(device, test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


@hydra.main(version_base=None, config_path="../configs", config_name="test")
def main(cfg: DictConfig) -> None:
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    test_input_img_dir = Path("../data/test_data/QRISP/TestSet/SeaPort/1080p/Native")
    test_output_img_dir = Path("../data/test_data/QRISP/TestSet/SeaPort/1080p/Enhanced")
    test_data = ToyDataset(test_input_img_dir, test_output_img_dir, 1, 300)

    test_dataloader = DataLoader(test_data, batch_size=cfg['loss']['batch_size'])

    # model

    loss_fn = nn.CrossEntropyLoss()

    evaluate(device, test_dataloader, model, loss_fn)


if __name__ == "__main__":
    main()
