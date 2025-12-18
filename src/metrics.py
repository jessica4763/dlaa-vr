import torch
import numpy as np
from skimage.metrics import (
    normalized_root_mse,
    peak_signal_noise_ratio,
    structural_similarity,
)

# TODO: ColourVideoVDP


class Metrics:
    def __init__(self, num_batches: int):
        self.num_batches = num_batches

        self.norm_rmse: np.ndarray = np.array([0.0])
        self.psnr: np.ndarray = np.array([0.0])
        self.ssim: np.ndarray = np.array([0.0])

    def record_rmse(self, pred_frame: np.ndarray, y: np.ndarray) -> None:
        self.norm_rmse += normalized_root_mse(pred_frame, y)

    def record_psnr(self, pred_frame: np.ndarray, y: np.ndarray) -> None:
        self.psnr += peak_signal_noise_ratio(pred_frame, y, data_range=255)

    def record_ssim(self, pred_frame: np.ndarray, y: np.ndarray) -> None:
        self.ssim += structural_similarity(
            pred_frame,
            y,
            data_range=255,
            channel_axis=0,
            gaussian_weights=True,
        )

    def record(self, pred_frame: torch.Tensor, y: torch.Tensor) -> None:
        y_ndarray = np.squeeze(y.cpu().numpy())
        pred_frame_ndarray = np.squeeze(pred_frame.cpu().numpy())
        self.record_rmse(pred_frame_ndarray, y_ndarray)
        self.record_psnr(pred_frame_ndarray, y_ndarray)
        self.record_ssim(pred_frame_ndarray, y_ndarray)

    def report(self) -> None:
        average_norm_rmse = self.norm_rmse.item() / self.num_batches
        average_psnr = self.psnr.item() / self.num_batches
        average_ssim = self.ssim.item() / self.num_batches
        print("Reported metrics:")
        print(f"\t {average_norm_rmse=}")
        print(f"\t {average_psnr=}")
        print(f"\t {average_ssim=}")
