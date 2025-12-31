import pycvvdp
import torch
import torch.nn as nn

from utils import linear_to_gamma


class CVVDPLoss(nn.Module):
    def __init__(self, display_name: pycvvdp.cvvdp) -> None:
        super().__init__()

        self.cvvdp = pycvvdp.cvvdp(display_name=display_name)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # sRGB frames and display-encoded values in the range [0, 1] 
        # are expected if the display model is standard_fhd
        pred, target = linear_to_gamma(pred), linear_to_gamma(target)
        return self.cvvdp.loss(pred, target, dim_order="BCHW").mean()


class L1LossWithCVVDP(nn.Module):
    def __init__(self, display_name: pycvvdp.cvvdp, cvvdp_weight: float = 0.05) -> None:
        super().__init__()

        self.l1_loss = nn.L1Loss()

        self.cvvdp = pycvvdp.cvvdp(display_name=display_name)
        self.cvvdp_weight = cvvdp_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = self.l1_loss(pred, target)

        pred, target = linear_to_gamma(pred), linear_to_gamma(target)
        cvvdp_loss = self.cvvdp_weight * self.cvvdp.loss(pred, target, dim_order="BCHW").mean()

        return l1_loss + cvvdp_loss
