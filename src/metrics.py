import numpy as np
import pyvista as pv
from skimage.metrics import (
    normalized_root_mse,
    peak_signal_noise_ratio,
    structural_similarity,
)
import torch
from torch.utils.tensorboard import SummaryWriter

# TODO: ColourVideoVDP


class Metrics:
    def __init__(self, writer: SummaryWriter, dataset_size: int):
        self.writer = writer

        self.dataset_size = dataset_size

        self.norm_rmse = 0
        self.psnr = 0
        self.ssim = 0

        self.metrics = {
            "avg_norm_rmse": 0,
            "avg_psnr": 0,
            "avg_ssim": 0,
            "avg_pixel_wise_std": 0,
        }

        self.pixel_sum = 0
        self.pixel_squared_sum = 0

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

    def record_pixel_wise_std(self, pred_frame: np.ndarray) -> None:
        self.pixel_mean += pred_frame
        self.pixel_mean_squared += np.square(pred_frame)

    def record(self, pred_frame: torch.Tensor, y: torch.Tensor) -> None:
        y_ndarray = np.squeeze(y.cpu().numpy())
        pred_frame_ndarray = np.squeeze(pred_frame.cpu().numpy())
        self.record_rmse(pred_frame_ndarray, y_ndarray)
        self.record_psnr(pred_frame_ndarray, y_ndarray)
        self.record_ssim(pred_frame_ndarray, y_ndarray)
        self.record_pixel_wise_std(pred_frame_ndarray, y_ndarray)

    def report(self) -> None:
        self.metrics["avg_norm_rmse"] = self.norm_rmse.item() / self.dataset_size
        self.metrics["avg_psnr"] = self.psnr.item() / self.dataset_size
        self.metrics["avg_ssim"] = self.ssim.item() / self.dataset_size

        pixel_mean = self.pixel_sum / self.dataset_size
        pixel_squared_mean = self.pixel_squared_sum / self.dataset_size
        self.metrics["avg_pixel_wise_std"] = np.mean(np.sqrt(pixel_squared_mean - np.square(pixel_mean)))

        reported_metrics_strings = []
        for metric_name in self.metrics:
            reported_metrics_strings.append(f"{metric_name}: {self.metrics[metric_name]}")

        reported_metrics = "\n".join(reported_metrics_strings)

        print(reported_metrics)
        self.writer.add_text(
            "reported metrics", 
            reported_metrics
        )
