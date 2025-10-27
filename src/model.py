from torch import nn

# Consider just the colour buffer the input to the neural network


class QualcommNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1920 * 1080, 1920 * 1080, (3, 3)),
            nn.ReLU(),
            nn.Linear(1920 * 1080, 1920 * 1080, (3, 3)),
            nn.ReLU(),
            nn.Linear(1920 * 1080, 1920 * 1080, (3, 3)),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
