import pycvvdp
import torch
import torch.nn as nn


class CVVDPLoss(nn.Module):
    def __init__(self, display_name: pycvvdp.cvvdp) -> None:
        super().__init__()

        self.cvvdp = pycvvdp.cvvdp(display_name=display_name)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert from linear sRGB colour space
        pred = pycvvdp.display_model.source_2_target_colorspace()

        return self.cvvdp.loss(pred, target, dim_order="BCHW").mean()


class L1LossWithCVVDP(nn.Module):
    def __init__(self, display_name: pycvvdp.cvvdp, cvvdp_weight: float = 0.05) -> None:
        super().__init__()

        self.l1_loss = nn.L1Loss()

        self.cvvdp = pycvvdp.cvvdp(display_name=display_name)
        self.cvvdp_weight = cvvdp_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.l1_loss(pred, target) +
            self.cvvdp_weight * self.cvvdp.loss(pred, target, dim_order="BCHW").mean()
        )
