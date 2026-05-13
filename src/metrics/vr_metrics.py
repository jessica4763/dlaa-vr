from collections import defaultdict
import lpips
from pathlib import Path
import pycvvdp
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from network.vr_network import VRNetwork
from utils import VRConfig, rgb_to_y, linear_to_gamma


class VRMetrics:
    def __init__(
        self, 
        dataset_size: int,
        padding: int,
        iterations: int,
        writer: SummaryWriter = None, 
        vr_config: VRConfig = None,
        is_stationary_segment: bool = False,
        display_name: str = "standard_fhd",
        evaluation_output_path: Path = None,
    ) -> None:
        self.dataset_size = dataset_size
        self.padding = padding
        self.iterations = iterations
        self.writer = writer
        self.vr_config = vr_config
        self.is_stationary_segment = is_stationary_segment
        self.evaluation_output_path = evaluation_output_path

        self.right_to_left_photometric_error = 0
        self.left_to_right_photometric_error = 0
        self.photometric_error_all = defaultdict(list)

        cuda0 = torch.device("cuda:0")
        self.loss_function_lpips = lpips.LPIPS(net="vgg").to(cuda0)

        self.cvvdp = pycvvdp.cvvdp(display_name=display_name)

        self.metrics = defaultdict(lambda: defaultdict(float))
        self.metrics_all = defaultdict(lambda: defaultdict(list))

        self.profiling_times = defaultdict(list)

    def record_psnr(self, pred: np.ndarray, target: np.ndarray, eye: str) -> None:
        # Display-encoded values in the range [0, 1]
        psnr = peak_signal_noise_ratio(target, pred, data_range=1.0).item()
        self.metrics["avg_psnr"][eye] += psnr
        self.metrics_all["avg_psnr"][eye].append(psnr)

    def record_ssim(self, pred: np.ndarray, target: np.ndarray, eye: str) -> None:
        # Display-encoded values in the range [0, 1]
        ssim = structural_similarity(
            target,
            pred,
            data_range=1.0,
            channel_axis=0,
            gaussian_weights=True,
        ).item()
        self.metrics["avg_ssim"][eye] += ssim
        self.metrics_all["avg_ssim"][eye].append(ssim)

    def record_lpips(self, pred: torch.Tensor, target: torch.Tensor, eye: str) -> None:
        # Display-encoded values in the range [-1, 1]
        pred = pred * 2.0 - 1.0
        target = target * 2.0 - 1.0
        lpips = self.loss_function_lpips(pred, target).item()
        self.metrics["avg_lpips"][eye] += lpips
        self.metrics_all["avg_lpips"][eye].append(lpips)

    def record_cvvdp_jod(self, pred: np.ndarray, target: np.ndarray, eye: str) -> None:
        # sRGB frames and display-encoded values in the range [0, 1] 
        # are expected if the display model is standard_fhd
        Q_jod, _ = self.cvvdp.predict(pred, target, dim_order="CHW")
        Q_jod = Q_jod.item()
        self.metrics["avg_cvvdp_jod"][eye] += Q_jod
        self.metrics_all["avg_cvvdp_jod"][eye].append(Q_jod)

    def record_pixel_wise_std(self, pred: np.ndarray, eye: str) -> None:
        self.metrics["pixel_mean"][eye] += pred
        self.metrics["pixel_squared_mean"][eye] += np.square(pred)
    
    def record_photometric_error(
        self, 
        model: VRNetwork, 
        left_pred: np.ndarray, 
        left_depth: np.ndarray, 
        right_pred: np.ndarray,
        right_depth: np.ndarray,
        curr_frame_num: int
    ) -> None:
        right_to_left_warped_curr, right_to_left_warp_grid = model.right_to_left_warp(
            right_pred,
            left_depth,
            self.vr_config.camera_baseline,
            self.vr_config.focal_length
        )

        left_to_right_warped_curr, left_to_right_warp_grid = model.left_to_right_warp(
            left_pred,
            right_depth,
            self.vr_config.camera_baseline,
            self.vr_config.focal_length
        )

        right_to_left_between_eye_warp_mask = model.get_between_eye_warp_mask(
            warp_from=left_to_right_warp_grid, 
            warp_onto=right_to_left_warp_grid
        )  # (B, 1, H, W)

        left_to_right_between_eye_warp_mask = model.get_between_eye_warp_mask(
            warp_from=right_to_left_warp_grid, 
            warp_onto=left_to_right_warp_grid
        )  # (B, 1, H, W)

        left_diff = torch.abs(left_pred - right_to_left_warped_curr)
        masked_left_diff = left_diff * right_to_left_between_eye_warp_mask
        mean_masked_left_diff = torch.sum(masked_left_diff) / (torch.sum(right_to_left_between_eye_warp_mask) + 1e-8)
        mean_masked_left_diff = mean_masked_left_diff.item()
        self.right_to_left_photometric_error += mean_masked_left_diff
        self.photometric_error_all["right_to_left_photometric_error"].append(mean_masked_left_diff)

        right_diff = torch.abs(right_pred - left_to_right_warped_curr)
        masked_right_diff = right_diff * left_to_right_between_eye_warp_mask
        mean_masked_right_diff = torch.sum(masked_right_diff) / (torch.sum(left_to_right_between_eye_warp_mask) + 1e-8)
        mean_masked_right_diff = mean_masked_right_diff.item()
        self.left_to_right_photometric_error += mean_masked_right_diff
        self.photometric_error_all["left_to_right_photometric_error"].append(mean_masked_right_diff)

        # if curr_frame_num == 30:
        #     save_image(right_to_left_between_eye_warp_mask, "right_to_left_between_eye_warp_mask.png")
        #     save_image(left_to_right_between_eye_warp_mask, "left_to_right_between_eye_warp_mask.png")

        #     left_diff = torch.abs(linear_to_gamma(left_pred) - linear_to_gamma(right_to_left_warped_curr))
        #     save_image(left_diff, "left_diff.png")

    def profile(self, name: str, elapsed_time: float, frame_num: int) -> None:
        self.profiling_times[name].append(elapsed_time)
        print(f"{elapsed_time=}")

    def record(self, pred: torch.Tensor, target: torch.Tensor, eye: str, model: VRNetwork = None) -> None:
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
            if metric_name not in ("pixel_mean", "pixel_squared_mean"):
                for eye in self.metrics[metric_name]: 
                    metric_id = f"{scene_name}/{'Fast' if evaluation_length <= 300 else 'Slow'}/{metric_name}/{eye}"
                    reported_metrics_strings.append(f"{metric_id}: {self.metrics[metric_name][eye]}")

                    if self.writer is not None:
                        print(f"{metric_name=}")
                        self.writer.add_scalar(
                            metric_id,
                            self.metrics[metric_name][eye],
                            self.iterations
                        )

                    with open(self.evaluation_output_path / f"{metric_id.replace('/', '_')}.txt", "w") as f:
                        for value in self.metrics_all[metric_name][eye]:
                            f.write(f"{value}\n")

        reported_metrics = "\n".join(reported_metrics_strings)
        print(reported_metrics)
        if self.writer is not None:
            self.writer.add_text(
                "reported metrics", 
                reported_metrics,
                self.iterations
            )

        self.right_to_left_photometric_error /= self.dataset_size
        self.left_to_right_photometric_error /= self.dataset_size
        print(f"{self.right_to_left_photometric_error=}")
        print(f"{self.left_to_right_photometric_error=}")
        for photometric_error_key in self.photometric_error_all:
            photometric_error_values = self.photometric_error_all[photometric_error_key]
            with open(self.evaluation_output_path / f"{photometric_error_key}.txt", "w") as f:
                for value in photometric_error_values:
                    f.write(f"{value}\n")
        
        print(f"Average time (ms): {sum(self.profiling_times['total'][30:]) / len(self.profiling_times['total'][30:])}")
        for profiling_time_key in self.profiling_times:
            with open(self.evaluation_output_path / f"{profiling_time_key}_times.txt", "w") as f:
                for value in self.profiling_times[profiling_time_key][30:]:
                    f.write(f"{value}\n")

