from torch import nn

# Consider just the colour buffer the input to the neural network


class QualcommNetwork(nn.Module):
    def __init__(self):
        """
        Implementation of the Qualcomm approach to neural supersampling, 
        but adapted for DLAA. 

        Input: 
        - Colour information for the current frame
        - Depth for the current frame
        - Jitter offset for the current frame
        - Previous colour output 
        - P

        Bidirectional mechanisms or using a sliding window of input frames,
        like those used in offline video enhancement approaches, are not 
        suitable for gaming applications. 

        """
        super().__init__()
        self.conv2d_relu_stack = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.conv2d_relu_stack(x)
        return logits
