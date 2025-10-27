from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ImageDataset
from model import NeuralNetwork

batch_size = 64

training_input_img_dir = Path("../data/training_data/QRISP/FloodedGrounds/1080p/Native")
training_output_img_dir = Path("../data/training_data/QRISP/FloodedGrounds/1080p/Enhanced")
training_data = ImageDataset(training_input_img_dir, training_output_img_dir, 21, 30)

test_input_img_dir = Path("../data/test_data/QRISP/TestSet/SeaPort/1080p/Native")
test_output_img_dir = Path("../data/test_data/QRISP/TestSet/SeaPort/1080p/Enhanced")
test_data = ImageDataset(test_input_img_dir, test_output_img_dir, 1, 300)

training_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(training_dataloader, model, loss_fn, optimizer):
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

if __name__ == "__main__":
    epochs = 3
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(training_dataloader, model, loss_fn, optimizer)
    print("Done!")
