from typing import Tuple, List
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import torch
from pytorch_msssim import ssim


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Tuple[List[float], List[float]]:
    """
    Compute PSNR and SSIM metrics for pairs of images from two batches.
    
    Args:
    - original: A batch of original images with shape (batch_size, channels, width, height).
    - reconstructed: A batch of reconstructed images with shape (batch_size, channels, width, height).
    
    Returns:
    - A tuple containing two lists: (psnr_values, ssim_values)
    """
    
    psnr_values = []
    ssim_values = []
    mae_values = []
    rmse_values = []
    vis_metrics = {'psnr':[], 'ssim':[], 'rmse':[], 'mae':[]}

    for  orig, recon in zip(original, reconstructed):
        ssim_value = ssim(orig.unsqueeze(0), recon.unsqueeze(0), data_range=1.0).item()
        ssim_values.append(ssim_value)
    
        orig_np = orig.permute(1, 2, 0).cpu().numpy()
        recon_np = recon.permute(1, 2, 0).cpu().numpy()
        psnr_value = psnr(orig_np, recon_np, data_range=1.0)  # The data range is 1.0 because your images are normalized to [0,1]
        psnr_values.append(psnr_value)

        mse_value = np.mean((orig_np - recon_np) ** 2)
        rmse_values.append(np.sqrt(mse_value))

        mae_value = np.mean(np.abs(orig_np - recon_np))
        mae_values.append(mae_value)

    vis_metrics['psnr'] = psnr_values
    vis_metrics['ssim'] = ssim_values
    vis_metrics['rmse'] = rmse_values
    vis_metrics['mae'] = mae_values
    
    return vis_metrics
