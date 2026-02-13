import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import torch

# Create a random RGB image, values in [0,1]
H, W, C = 128, 128, 3
np.random.seed(0)
image_hwc = np.random.rand(H, W, C).astype(np.float32)

# Make a “noisy” version
noise = np.random.normal(scale=0.05, size=(H, W, C)).astype(np.float32)
image_noisy_hwc = np.clip(image_hwc + noise, 0.0, 1.0)

# Case 1: HWC (correct)
psnr_hwc = peak_signal_noise_ratio(image_hwc, image_noisy_hwc, data_range=1.0)
print(f"PSNR HWC: {psnr_hwc:.4f} dB")

# Case 2: CHW (incorrect)
image_chw = np.transpose(image_hwc, (2, 0, 1))
image_noisy_chw = np.transpose(image_noisy_hwc, (2, 0, 1))
psnr_chw = peak_signal_noise_ratio(image_chw, image_noisy_chw, data_range=1.0)
print(f"PSNR CHW: {psnr_chw:.4f} dB")
