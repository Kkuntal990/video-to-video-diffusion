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

    # FIXED: Clamp MSE to prevent numerical issues
    # - Prevents sqrt(0) which can cause numerical errors
    # - Prevents division by near-zero which creates inf
    eps = 1e-8
    mse = torch.clamp(mse, min=eps)

    # Handle perfect or near-perfect reconstruction
    if mse < eps:
        return 100.0  # Very high PSNR (instead of inf)

    # Calculate PSNR
    psnr = 20 * torch.log10(torch.tensor(max_val, device=mse.device) / torch.sqrt(mse))

    # FIXED: Clamp output to reasonable range [0, 100] to prevent inf/NaN propagation
    psnr = torch.clamp(psnr, min=0.0, max=100.0)

    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    """
    Calculate SSIM between two images/videos

    Simplified version - for full implementation, use pytorch-msssim library

    Args:
        img1: first image/video tensor (B, C, H, W) or (B, C, D, H, W)
        img2: second image/video tensor (B, C, H, W) or (B, C, D, H, W)
        window_size: size of Gaussian window
        max_val: maximum pixel value

    Returns:
        ssim: SSIM value in [0, 1]

    Note:
        For 5D tensors (3D volumes), SSIM is computed on the middle slice
    """
    # Handle 5D tensors (3D volumes) by using middle slice
    if img1.ndim == 5:  # (B, C, D, H, W)
        mid_slice = img1.shape[2] // 2
        img1 = img1[:, :, mid_slice, :, :]  # Extract middle slice → (B, C, H, W)
        img2 = img2[:, :, mid_slice, :, :]

    # FIXED: Add numerical stability constants
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    eps = 1e-8  # Additional epsilon for numerical stability

    # Mean
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Variance with numerical stability
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    # FIXED: Clamp variances to prevent negative values from floating point errors
    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

    # SSIM formula with additional epsilon for stability
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps

    ssim_map = numerator / denominator

    # FIXED: Clamp SSIM values to valid range [0, 1] and check for NaN
    ssim_map = torch.clamp(ssim_map, min=0.0, max=1.0)

    # Check if result contains NaN
    if torch.isnan(ssim_map).any():
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"SSIM calculation produced {torch.isnan(ssim_map).sum().item()} NaN values, returning 0.0")
        return 0.0

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
    # FIXED: Add input validation to check for NaN values
    if torch.isnan(video1).any():
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"NaN values detected in predicted video: {torch.isnan(video1).sum().item()} NaN values")
        return {'psnr': 0.0, 'ssim': 0.0, 'psnr_per_frame': [], 'ssim_per_frame': []}

    if torch.isnan(video2).any():
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"NaN values detected in ground truth video: {torch.isnan(video2).sum().item()} NaN values")
        return {'psnr': 0.0, 'ssim': 0.0, 'psnr_per_frame': [], 'ssim_per_frame': []}

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

        # FIXED: Check for NaN in individual frame metrics
        if np.isnan(psnr) or np.isnan(ssim):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"NaN detected in frame {t}: PSNR={psnr}, SSIM={ssim}")
            # Skip this frame instead of propagating NaN
            continue

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    # FIXED: Use nanmean to handle any remaining NaN values gracefully
    # If all frames are NaN, return 0.0 instead of NaN
    if len(psnr_values) == 0:
        avg_psnr = 0.0
        avg_ssim = 0.0
    else:
        avg_psnr = np.nanmean(psnr_values) if len(psnr_values) > 0 else 0.0
        avg_ssim = np.nanmean(ssim_values) if len(ssim_values) > 0 else 0.0

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
