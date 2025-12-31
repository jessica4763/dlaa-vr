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

# TODO: LPIPS

class Metrics:
    def __init__(
        self, 
        dataset_size: int,
        display_name: str = "standard_fhd", 
        writer: SummaryWriter = None, 
    ) -> None:
        self.writer = writer

        self.dataset_size = dataset_size

        self.norm_rmse_sum = 0
        self.psnr_sum = 0
        self.ssim_sum = 0

        self.lpips_sum = 0
        cuda0 = torch.device('cuda:0')
        self.loss_function_alex = lpips.LPIPS(net='alex').to(cuda0)

        self.cvvdp = pycvvdp.cvvdp(display_name=display_name)
        self.cvvdp_jod_sum = 0

        self.pixel_sum = 0
        self.pixel_squared_sum = 0

        self.metrics = {
            "avg_norm_rmse": 0,
            "avg_psnr": 0,
            "avg_ssim": 0,
            "avg_lpips": 0,
            "avg_cvvdp_jod": 0,
            "avg_pixel_wise_std": 0
        }

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
        pred, target = (pred * 2.0 - 1.0).squeeze(0), (target * 2.0 - 1.0).squeeze(0)
        self.lpips_sum += self.loss_function_alex.forward(pred, target)

    def record_cvvdp_jod(self, pred: np.ndarray, target: np.ndarray) -> None:
        # sRGB frames and display-encoded values in the range [0, 1] 
        # are expected if the display model is standard_fhd
        Q_jod, _ = self.cvvdp.predict(pred, target, dim_order="CHW")
        self.cvvdp_jod_sum += Q_jod

    def record_pixel_wise_std(self, pred: np.ndarray) -> None:
        self.pixel_sum += pred
        self.pixel_squared_sum += np.square(pred)

    def record(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        target_ndarray = np.squeeze(target.cpu().numpy())
        pred_ndarray = np.squeeze(pred.cpu().numpy())
        self.record_rmse(pred_ndarray, target_ndarray)
        self.record_psnr(pred_ndarray, target_ndarray)
        self.record_ssim(pred_ndarray, target_ndarray)
        self.record_lpips(pred, target)
        self.record_cvvdp_jod(pred_ndarray, target_ndarray)
        self.record_pixel_wise_std(pred_ndarray)

    def report(self) -> None:
        self.metrics["avg_norm_rmse"] = self.norm_rmse_sum.item() / self.dataset_size
        self.metrics["avg_psnr"] = self.psnr_sum.item() / self.dataset_size
        self.metrics["avg_ssim"] = self.ssim_sum.item() / self.dataset_size
        self.metrics["avg_lpips"] = self.lpips_sum.item() / self.dataset_size
        self.metrics["avg_cvvdp_jod"] = self.cvvdp_jod_sum / self.dataset_size

        pixel_mean = self.pixel_sum / self.dataset_size
        pixel_squared_mean = self.pixel_squared_sum / self.dataset_size
        self.metrics["avg_pixel_wise_std"] = np.mean(np.sqrt(pixel_squared_mean - np.square(pixel_mean)))

        reported_metrics_strings = []
        for metric_name in self.metrics:
            reported_metrics_strings.append(f"{metric_name}: {self.metrics[metric_name]}")

        reported_metrics = "\n".join(reported_metrics_strings)

        print(reported_metrics)
        if self.writer is not None:
            self.writer.add_text(
                "reported metrics", 
                reported_metrics
            )
