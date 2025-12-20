import numpy as np
from skimage.metrics import (
    normalized_root_mse,
    peak_signal_noise_ratio,
    structural_similarity,
)
import torch
from torch.utils.tensorboard import SummaryWriter

# TODO: ColourVideoVDP


class Metrics:
    def __init__(self, writer: SummaryWriter, num_batches: int):
        self.writer = writer

        self.num_batches = num_batches

        self.norm_rmse: np.ndarray = np.array([0.0])
        self.psnr: np.ndarray = np.array([0.0])
        self.ssim: np.ndarray = np.array([0.0])
        self.per_pixel_variance: np.ndarray = np.array([0.0])

    def record_rmse(self, pred_frame: np.ndarray, y: np.ndarray) -> None:
        self.norm_rmse += normalized_root_mse(y, pred_frame)

    def record_psnr(self, pred_frame: np.ndarray, y: np.ndarray) -> None:
        self.psnr += peak_signal_noise_ratio(y, pred_frame, data_range=1.0)

    def record_ssim(self, pred_frame: np.ndarray, y: np.ndarray) -> None:
        self.ssim += structural_similarity(
            y,
            pred_frame,
            data_range=1.0,
            channel_axis=0,
            gaussian_weights=True,
        )

    def record_per_pixel_variance(
        self, 
        pred_frame: np.ndarray,
        prev_frame: np.ndarray
    ) -> None:
        difference_frame = pred_frame - prev_frame


    def record(self, pred_frame: torch.Tensor, y: torch.Tensor) -> None:
        y_ndarray = np.squeeze(y.cpu().numpy())
        pred_frame_ndarray = np.squeeze(pred_frame.cpu().numpy())
        self.record_rmse(pred_frame_ndarray, y_ndarray)
        self.record_psnr(pred_frame_ndarray, y_ndarray)
        self.record_ssim(pred_frame_ndarray, y_ndarray)
        self.record_per_pixel_variance(pred_frame_ndarray, y_ndarray)

    def report(self) -> None:
        average_norm_rmse = self.norm_rmse.item() / self.num_batches
        average_psnr = self.psnr.item() / self.num_batches
        average_ssim = self.ssim.item() / self.num_batches
        reported_metrics = f"\n{average_norm_rmse=}\n{average_psnr=}\n{average_ssim=}\n"
        print(reported_metrics)
        self.writer.add_text(
            "reported metrics", 
            reported_metrics
        )
