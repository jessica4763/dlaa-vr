import hydra
from omegaconf import DictConfig
import os
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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


@hydra.main(version_base=None, config_path="../configs", config_name="test")
def main(cfg: DictConfig) -> None:
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # For reproducibility
    torch.manual_seed(cfg['testing']['seed'])
    torch.backends.cudnn.benchmark = False  # Deterministically select an algorithm; reduces efficiency
    torch.use_deterministic_algorithms(True)  # Use only deterministic algorithms

    # Does not use unitialised memory as an input to an operation
    torch.utils.deterministic.fill_uninitialized_memory = False

    test_input_img_dir = Path(cfg["dataset"]["test-input-img-path"])
    test_output_img_dir = Path(cfg["dataset"]["test-output-img-path"])
    instances = os.listdir(test_input_img_dir)
    frames = os.listdir(test_input_img_dir / instances[0])
    num_instances = len(instances)
    num_frames = len(frames)
    test_data = ToyDataset(
        test_input_img_dir,
        test_output_img_dir,
        num_instances,
        num_frames
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=cfg['testing']['batch-size'],
        shuffle=cfg['testing']['shuffle']  # False for evaluation
    )

    model = QualcommNetwork(
        num_prev_feature_channels=3,
        hidden_channels=32,
        num_blocks=3,
        upscale_factor=1
    ).to(device)
    model.load_state_dict(torch.load(cfg['testing']['saved-models-path'], weights_only=True))

    loss_fn = nn.L1Loss()

    evaluate(device, test_dataloader, model, loss_fn)


if __name__ == "__main__":
    main()
