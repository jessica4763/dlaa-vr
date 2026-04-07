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

from models.vr_network import VRNetwork
from utils import VRConfig, rgb_to_y


class VRMetrics:
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

        self.psnr_sum = 0
        self.ssim_sum = 0

        self.lpips_sum = 0
        cuda0 = torch.device("cuda:0")
        self.loss_function_lpips = lpips.LPIPS(net="vgg").to(cuda0)

        self.cvvdp = pycvvdp.cvvdp(display_name=display_name)
        self.cvvdp_jod_sum = 0

        self.pixel_sum = 0
        self.pixel_squared_sum = 0

        self.metrics = defaultdict(lambda: defaultdict(float))

    def record_psnr(self, pred: np.ndarray, target: np.ndarray, eye: str) -> None:
        # Display-encoded values in the range [0, 1]
        self.metrics["avg_psnr"][eye] += peak_signal_noise_ratio(target, pred, data_range=1.0).item()

    def record_ssim(self, pred: np.ndarray, target: np.ndarray, eye: str) -> None:
        # Display-encoded values in the range [0, 1]
        self.metrics["avg_ssim"][eye] += structural_similarity(
            target,
            pred,
            data_range=1.0,
            channel_axis=0,
            gaussian_weights=True,
        ).item()

    def record_lpips(self, pred: torch.Tensor, target: torch.Tensor, eye: str) -> None:
        # Display-encoded values in the range [-1, 1]
        pred = pred * 2.0 - 1.0
        target = target * 2.0 - 1.0
        self.metrics["avg_lpips"][eye] += self.loss_function_lpips(pred, target).item()

    def record_cvvdp_jod(self, pred: np.ndarray, target: np.ndarray, eye: str) -> None:
        # sRGB frames and display-encoded values in the range [0, 1] 
        # are expected if the display model is standard_fhd
        Q_jod, _ = self.cvvdp.predict(pred, target, dim_order="CHW")
        self.metrics["avg_cvvdp_jod"][eye] += Q_jod.item()

    def record_pixel_wise_std(self, pred: np.ndarray, eye: str) -> None:
        self.metrics["pixel_mean"][eye] += pred
        self.metrics["pixel_squared_mean"][eye] += np.square(pred)
    
    def record_photometric_residual(
        self, 
        model: VRNetwork, 
        left_pred: np.ndarray, 
        left_depth: np.ndarray, 
        right_pred: np.ndarray,
        right_depth: np.ndarray
    ) -> float:
        _, right_to_left_warp_grid = model.right_to_left_warp(
            right_pred,
            left_depth,
            self.vr_config.camera_baseline,
            self.vr_config.focal_length
        )

        left_warped_curr, left_to_right_warp_grid = model.left_to_right_warp(
            left_pred,
            right_depth,
            self.vr_config.camera_baseline,
            self.vr_config.focal_length
        )

        between_eye_warp_mask = model.get_between_eye_warp_mask(
            warp_from=left_to_right_warp_grid, 
            warp_onto=right_to_left_warp_grid
        )

        diff = torch.abs(left_warped_curr - left_pred)
        diff = diff[between_eye_warp_mask]
        return 

    def record(self, model: VRNetwork, pred: torch.Tensor, target: torch.Tensor, eye: str) -> None:
        pred = torch.round(pred * 255.0) / 255.0
        target = torch.round(target * 255.0) / 255.0

        pred_y = np.squeeze(rgb_to_y(pred).cpu().numpy())
        target_y = np.squeeze(rgb_to_y(target).cpu().numpy())

        pred_ndarray =  np.squeeze(pred.cpu().numpy())
        target_ndarray = np.squeeze(target.cpu().numpy())

        self.record_psnr(pred_y, target_y, eye)
        self.record_ssim(pred_y, target_y, eye)

        self.record_lpips(pred, target, eye)

        self.record_cvvdp_jod(pred_ndarray, target_ndarray, eye)

        if self.is_stationary_segment:
            self.record_pixel_wise_std(pred_ndarray, eye)

    def report(self, scene_name: str, evaluation_length: int) -> None:
        # --------------------------------------------------------------------
        # ----------------------------- Left eye -----------------------------
        # --------------------------------------------------------------------
        self.metrics["avg_psnr"]["left"] /= self.dataset_size
        self.metrics["avg_ssim"]["left"] /= self.dataset_size
        self.metrics["avg_lpips"]["left"] /= self.dataset_size
        self.metrics["avg_cvvdp_jod"]["left"] /= self.dataset_size

        if self.is_stationary_segment:
            self.metrics["pixel_mean"]["left"] /= self.dataset_size
            self.metrics["pixel_squared_mean"]["left"] /= self.dataset_size
            self.metrics["avg_pixel_wise_std"]["left"] = np.mean(np.sqrt(np.maximum(self.metrics["pixel_squared_mean"]["left"] - np.square(self.metrics["pixel_mean"]["left"]), 0)))

        # --------------------------------------------------------------------
        # ----------------------------- Right eye ----------------------------
        # --------------------------------------------------------------------
        self.metrics["avg_psnr"]["right"] /= self.dataset_size
        self.metrics["avg_ssim"]["right"] /= self.dataset_size
        self.metrics["avg_lpips"]["right"] /= self.dataset_size
        self.metrics["avg_cvvdp_jod"]["right"] /= self.dataset_size

        if self.is_stationary_segment:
            self.metrics["pixel_mean"]["right"] /= self.dataset_size
            self.metrics["pixel_squared_mean"]["right"] /= self.dataset_size
            self.metrics["avg_pixel_wise_std"]["right"] = np.mean(np.sqrt(np.maximum(self.metrics["pixel_squared_mean"]["right"] - np.square(self.metrics["pixel_mean"]["right"]), 0)))

        reported_metrics_strings = []
        for metric_name in self.metrics:
            for eye in self.metrics[metric_name]: 
                metric_id = f"{scene_name}/{'Fast' if evaluation_length <= 300 else 'Slow'}/{metric_name}/{eye}"
                reported_metrics_strings.append(f"{metric_id}: {self.metrics[metric_name][eye]}")

                if self.writer is not None:
                    self.writer.add_scalar(
                        metric_id,
                        self.metrics[metric_name][eye],
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
