import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

def plot_difference_map(pred_path, target_path, scale_factor=5.0):
    pred_pil = Image.open(pred_path).convert("RGB")
    target_pil = Image.open(target_path).convert("RGB")
    
    pred = TF.to_tensor(pred_pil)
    target = TF.to_tensor(target_pil)
    
    diff = torch.abs(pred - target).mean(dim=0) 
    
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    pred_im = pred.permute(1, 2, 0)
    target_im = target.permute(1, 2, 0)
    
    axes[0].imshow(target_im)
    axes[0].set_title("ground truth")
    
    axes[1].imshow(pred_im)
    axes[1].set_title("prediction")
    
    im_diff = axes[2].imshow(diff * scale_factor, cmap='magma', vmin=0, vmax=1)
    axes[2].set_title(f"difference map (scaled x{scale_factor})")
    
    plt.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
    plt.show()

plot_difference_map("predicted.png", "ground-truth.png")
