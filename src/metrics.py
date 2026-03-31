from collections import defaultdict
import lpips
import pycvvdp
import numpy as np
from skimage.metrics import (
    normalized_root_mse,
    peak_signal_noise_ratio,
    structural_similarity,
)
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import VRConfig, rgb_to_y


class Metrics:
    def __init__(
        self, 
        dataset_size: int,
        padding: int,
        iterations: int,
        writer: SummaryWriter = None, 
        vr_config: VRConfig = None,
        is_stationary_segment: bool = False,
        display_name: str = "standard_fhd"
    ) -> None:
        self.dataset_size = dataset_size
        self.padding = padding
        self.iterations = iterations
        self.writer = writer
        self.vr_config = vr_config
        self.is_stationary_segment = is_stationary_segment

        self.norm_rmse_sum = 0
        self.psnr_sum = 0
        self.ssim_sum = 0

        self.lpips_sum = 0
        cuda0 = torch.device('cuda:0')
        self.loss_function_lpips = lpips.LPIPS(net='vgg').to(cuda0)

        self.cvvdp = pycvvdp.cvvdp(display_name=display_name)
        self.cvvdp_jod_sum = 0

        self.pixel_sum = 0
        self.pixel_squared_sum = 0

        self.metrics = defaultdict(float)

    def record_rmse(self, pred: np.ndarray, target: np.ndarray) -> None:
        # Display-encoded values in the range [0, 1]
        self.norm_rmse_sum += normalized_root_mse(target, pred)

    def record_psnr(self, pred: np.ndarray, target: np.ndarray) -> None:
        # Display-encoded values in the range [0, 1]
        self.psnr_sum += peak_signal_noise_ratio(target, pred, data_range=1.0)

    def record_ssim(self, pred: np.ndarray, target: np.ndarray) -> None:
        # Display-encoded values in the range [0, 1]
        self.ssim_sum += structural_similarity(
            target,
            pred,
            data_range=1.0,
            channel_axis=0,
            gaussian_weights=True,
        )

    def record_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # Display-encoded values in the range [-1, 1]
        pred = pred * 2.0 - 1.0
        target = target * 2.0 - 1.0
        self.lpips_sum += self.loss_function_lpips(pred, target).item()

    def record_cvvdp_jod(self, pred: np.ndarray, target: np.ndarray) -> None:
        # sRGB frames and display-encoded values in the range [0, 1] 
        # are expected if the display model is standard_fhd
        Q_jod, _ = self.cvvdp.predict(pred, target, dim_order="CHW")
        self.cvvdp_jod_sum += Q_jod.item()

    def record_pixel_wise_std(self, pred: np.ndarray) -> None:
        self.pixel_sum += pred
        self.pixel_squared_sum += np.square(pred)
    
    def record_photometric_residual(self, left_pred: np.ndarray, left_depth: np.ndarray, right_pred: np.ndarray) -> float:
        pass

    def record(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.round(pred * 255.0) / 255.0
        target = torch.round(target * 255.0) / 255.0

        pred_y = np.squeeze(rgb_to_y(pred).cpu().numpy())
        target_y = np.squeeze(rgb_to_y(target).cpu().numpy())

        pred_ndarray =  np.squeeze(pred.cpu().numpy())
        target_ndarray = np.squeeze(target.cpu().numpy())

        self.record_rmse(pred_y, target_y)
        self.record_psnr(pred_y, target_y)
        self.record_ssim(pred_y, target_y)

        self.record_lpips(pred, target)

        self.record_cvvdp_jod(pred_ndarray, target_ndarray)

        if self.is_stationary_segment:
            self.record_pixel_wise_std(pred_ndarray)

    def report(self, scene_name) -> None:
        self.metrics["avg_norm_rmse"] = self.norm_rmse_sum.item() / self.dataset_size
        self.metrics["avg_psnr"] = self.psnr_sum.item() / self.dataset_size
        self.metrics["avg_ssim"] = self.ssim_sum.item() / self.dataset_size
        self.metrics["avg_lpips"] = self.lpips_sum / self.dataset_size
        self.metrics["avg_cvvdp_jod"] = self.cvvdp_jod_sum / self.dataset_size

        if self.is_stationary_segment:
            pixel_mean = self.pixel_sum / self.dataset_size
            pixel_squared_mean = self.pixel_squared_sum / self.dataset_size
            self.metrics["avg_pixel_wise_std"] = np.mean(np.sqrt(np.maximum(pixel_squared_mean - np.square(pixel_mean), 0)))

        reported_metrics_strings = []
        for metric_name in self.metrics:
            reported_metrics_strings.append(f"{scene_name}/{metric_name}: {self.metrics[metric_name]}")
            if self.writer is not None:
                self.writer.add_scalar(
                    f"{scene_name}/{metric_name}",
                    self.metrics[metric_name],
                    self.iterations
                )

        reported_metrics = "\n".join(reported_metrics_strings)
        print(reported_metrics)
        if self.writer is not None:
            self.writer.add_text(
                "reported metrics", 
                reported_metrics,
                self.iterations
            )
