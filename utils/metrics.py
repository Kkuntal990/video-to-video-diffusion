"""
Video quality metrics

Implements:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
"""

import torch
import torch.nn.functional as F
import numpy as np


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images/videos

    Args:
        img1: first image/video tensor (*, H, W) or (*, C, H, W)
        img2: second image/video tensor (*, H, W) or (*, C, H, W)
        max_val: maximum pixel value (1.0 for normalized, 255 for uint8)

    Returns:
        psnr: PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf')

    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    """
    Calculate SSIM between two images/videos

    Simplified version - for full implementation, use pytorch-msssim library

    Args:
        img1: first image/video tensor (B, C, H, W)
        img2: second image/video tensor (B, C, H, W)
        window_size: size of Gaussian window
        max_val: maximum pixel value

    Returns:
        ssim: SSIM value in [0, 1]
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Mean
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Variance
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def calculate_video_metrics(video1, video2, max_val=1.0):
    """
    Calculate multiple metrics for videos

    Args:
        video1: first video tensor (B, C, T, H, W) or (C, T, H, W)
        video2: second video tensor (B, C, T, H, W) or (C, T, H, W)
        max_val: maximum pixel value

    Returns:
        metrics: dict with PSNR and SSIM
    """
    # Handle single video (add batch dim)
    if video1.dim() == 4:
        video1 = video1.unsqueeze(0)
        video2 = video2.unsqueeze(0)

    B, C, T, H, W = video1.shape

    # Calculate metrics frame by frame
    psnr_values = []
    ssim_values = []

    for t in range(T):
        frame1 = video1[:, :, t, :, :]  # (B, C, H, W)
        frame2 = video2[:, :, t, :, :]

        psnr = calculate_psnr(frame1, frame2, max_val)
        ssim = calculate_ssim(frame1, frame2, max_val=max_val)

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    # Average over time
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'psnr_per_frame': psnr_values,
        'ssim_per_frame': ssim_values
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing video metrics...")

    # Create dummy videos
    video1 = torch.randn(2, 3, 16, 256, 256)
    video2 = video1 + torch.randn(2, 3, 16, 256, 256) * 0.1  # Similar but with noise

    print(f"Video 1 shape: {video1.shape}")
    print(f"Video 2 shape: {video2.shape}")

    # Test PSNR
    psnr = calculate_psnr(video1, video2)
    print(f"\nPSNR: {psnr:.2f} dB")

    # Test SSIM (on single frame)
    frame1 = video1[:, :, 0, :, :]
    frame2 = video2[:, :, 0, :, :]
    ssim = calculate_ssim(frame1, frame2)
    print(f"SSIM (frame 0): {ssim:.4f}")

    # Test video metrics
    metrics = calculate_video_metrics(video1, video2)
    print(f"\nVideo metrics:")
    print(f"  Average PSNR: {metrics['psnr']:.2f} dB")
    print(f"  Average SSIM: {metrics['ssim']:.4f}")

    print("\nMetrics test successful!")
