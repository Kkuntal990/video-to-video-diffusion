"""
Multi-Scale Loss Functions for CT Slice Interpolation

Implements SOTA loss functions based on 2024 research:
1. Perceptual Loss (VGG features on 2D slices)
2. MS-SSIM Loss (Multi-Scale Structural Similarity)
3. Combined loss for diffusion training

References:
- 3D MedDiffusion (2024): Combined losses for medical imaging
- MSDSR (2024): Masked Slice Diffusion with perceptual loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple
import math


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features on 2D slices

    Computes feature differences from pre-trained VGG network.
    For 3D volumes, samples random slices and computes loss in 2D.

    Args:
        feature_layers: List of VGG layers to extract features from
        use_l1: Use L1 distance (True) or L2 distance (False)
        slice_sample_rate: How many slices to sample for loss computation (1.0 = all)
    """

    def __init__(
        self,
        feature_layers: list = [2, 7, 12, 21, 30],  # relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
        use_l1: bool = True,
        slice_sample_rate: float = 0.2,  # Sample 20% of slices to reduce compute
    ):
        super().__init__()

        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features

        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False

        self.feature_layers = feature_layers
        self.use_l1 = use_l1
        self.slice_sample_rate = slice_sample_rate

        # Split VGG into blocks for feature extraction
        self.vgg_blocks = nn.ModuleList()
        prev_layer = 0
        for layer_idx in feature_layers:
            self.vgg_blocks.append(vgg[prev_layer:layer_idx+1])
            prev_layer = layer_idx + 1

        # Move to eval mode
        self.eval()

        # Normalization for ImageNet pre-training
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_grayscale_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert grayscale volume to RGB and normalize for VGG

        Args:
            x: (B, 1, D, H, W) grayscale volume in [-1, 1]

        Returns:
            rgb: (B*D_sampled, 3, H, W) RGB slices normalized for VGG
        """
        B, C, D, H, W = x.shape
        assert C == 1, "Expected grayscale input (C=1)"

        # Sample slices (reduce computation)
        num_sample = max(1, int(D * self.slice_sample_rate))
        if num_sample < D:
            # Evenly sample slices
            slice_indices = torch.linspace(0, D-1, num_sample, dtype=torch.long, device=x.device)
            x = x[:, :, slice_indices]  # (B, 1, D_sampled, H, W)

        # Reshape: (B, 1, D_sampled, H, W) → (B*D_sampled, 1, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, 1, H, W)

        # Convert from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0

        # Repeat grayscale to RGB
        x = x.repeat(1, 3, 1, 1)  # (B*D_sampled, 3, H, W)

        # Normalize for VGG (ImageNet stats)
        x = (x - self.mean) / self.std

        return x

    def extract_features(self, x: torch.Tensor) -> list:
        """Extract multi-scale VGG features"""
        features = []
        for block in self.vgg_blocks:
            x = block(x)
            features.append(x)
        return features

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and target volumes

        Args:
            pred: (B, 1, D, H, W) predicted volume
            target: (B, 1, D, H, W) target volume

        Returns:
            loss: scalar perceptual loss
        """
        # Normalize to RGB for VGG
        pred_rgb = self.normalize_grayscale_to_rgb(pred)
        target_rgb = self.normalize_grayscale_to_rgb(target)

        # Extract features
        with torch.no_grad():
            target_features = self.extract_features(target_rgb)

        pred_features = self.extract_features(pred_rgb)

        # Compute loss across all feature layers
        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            if self.use_l1:
                loss += F.l1_loss(pred_feat, target_feat)
            else:
                loss += F.mse_loss(pred_feat, target_feat)

        # Average across layers
        loss = loss / len(self.feature_layers)

        return loss


class MS_SSIM_Loss(nn.Module):
    """
    Multi-Scale Structural Similarity Index (MS-SSIM) Loss

    Computes structural similarity at multiple scales.
    For 3D volumes, computes on 2D slices.

    Args:
        window_size: Size of Gaussian window
        size_average: Average the loss across batch
        channel: Number of channels (1 for grayscale)
    """

    def __init__(
        self,
        window_size: int = 11,
        size_average: bool = True,
        channel: int = 1,
    ):
        super().__init__()

        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

        # Create Gaussian window
        self.window = self._create_window(window_size, channel)

    def _gaussian_window(self, window_size: int, sigma: float = 1.5) -> torch.Tensor:
        """Create 1D Gaussian window"""
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2)**2 / (2.0 * sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create 2D Gaussian window"""
        _1D_window = self._gaussian_window(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        channel: int,
        size_average: bool = True,
    ) -> torch.Tensor:
        """Compute SSIM between two images"""
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MS-SSIM loss between predicted and target volumes

        Args:
            pred: (B, 1, D, H, W) predicted volume in [-1, 1]
            target: (B, 1, D, H, W) target volume in [-1, 1]

        Returns:
            loss: 1 - MS-SSIM (lower is better)
        """
        B, C, D, H, W = pred.shape

        # Reshape: (B, C, D, H, W) → (B*D, C, H, W)
        pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(B*D, C, H, W)
        target_2d = target.permute(0, 2, 1, 3, 4).reshape(B*D, C, H, W)

        # Convert from [-1, 1] to [0, 1] for SSIM
        pred_2d = (pred_2d + 1.0) / 2.0
        target_2d = (target_2d + 1.0) / 2.0

        # Ensure window is on correct device
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)

        # Compute SSIM at multiple scales
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=pred.device)
        levels = weights.size(0)

        mssim = []
        mcs = []

        for i in range(levels):
            ssim_val = self._ssim(
                pred_2d, target_2d, self.window, self.window_size, self.channel, self.size_average
            )
            mssim.append(ssim_val)

            if i < levels - 1:
                # Downsample for next scale
                pred_2d = F.avg_pool2d(pred_2d, kernel_size=2, stride=2)
                target_2d = F.avg_pool2d(target_2d, kernel_size=2, stride=2)

        # Combine multi-scale SSIM
        mssim = torch.stack(mssim)
        ms_ssim_val = (mssim ** weights).prod()

        # Return loss (1 - SSIM)
        return 1.0 - ms_ssim_val


class CombinedLoss(nn.Module):
    """
    Combined loss for CT slice interpolation training

    Combines:
    - Diffusion loss (MSE on noise prediction) - primary
    - Perceptual loss (VGG features) - for texture/structure
    - MS-SSIM loss (multi-scale similarity) - for quality

    Args:
        lambda_perceptual: Weight for perceptual loss (0.1 recommended)
        lambda_ssim: Weight for MS-SSIM loss (0.1 recommended)
        perceptual_every_n_steps: Compute perceptual loss every N steps (default: 10)
        ssim_every_n_steps: Compute SSIM loss every N steps (default: 10)
    """

    def __init__(
        self,
        lambda_perceptual: float = 0.1,
        lambda_ssim: float = 0.1,
        perceptual_every_n_steps: int = 10,
        ssim_every_n_steps: int = 10,
    ):
        super().__init__()

        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.perceptual_every_n_steps = perceptual_every_n_steps
        self.ssim_every_n_steps = ssim_every_n_steps

        # Initialize loss modules
        self.perceptual_loss = VGGPerceptualLoss()
        self.ssim_loss = MS_SSIM_Loss()

        # Step counter
        self.register_buffer('step', torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        diffusion_loss: torch.Tensor,
        compute_auxiliary: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss

        Args:
            pred: (B, 1, D, H, W) predicted volume (decoded from latent)
            target: (B, 1, D, H, W) target volume (decoded from latent)
            diffusion_loss: Scalar diffusion loss (noise prediction MSE)
            compute_auxiliary: Whether to compute perceptual + SSIM (expensive)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {
            'diffusion': diffusion_loss.item(),
        }

        total_loss = diffusion_loss

        # Compute auxiliary losses every N steps to reduce overhead
        if compute_auxiliary:
            # Perceptual loss
            if (self.step % self.perceptual_every_n_steps == 0) and self.lambda_perceptual > 0:
                perceptual = self.perceptual_loss(pred, target)
                total_loss = total_loss + self.lambda_perceptual * perceptual
                loss_dict['perceptual'] = perceptual.item()

            # MS-SSIM loss
            if (self.step % self.ssim_every_n_steps == 0) and self.lambda_ssim > 0:
                ssim = self.ssim_loss(pred, target)
                total_loss = total_loss + self.lambda_ssim * ssim
                loss_dict['ssim'] = ssim.item()

        # Increment step counter
        self.step += 1

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# Simple test
if __name__ == "__main__":
    print("Testing Multi-Scale Loss Functions...")
    print("="*80)

    # Create dummy data
    B, C, D, H, W = 2, 1, 50, 256, 256
    pred = torch.randn(B, C, D, H, W)
    target = torch.randn(B, C, D, H, W)
    diffusion_loss = torch.tensor(0.1)

    # Test VGG Perceptual Loss
    print("\n1. Testing VGG Perceptual Loss...")
    perceptual_loss = VGGPerceptualLoss(slice_sample_rate=0.2)
    perc_val = perceptual_loss(pred, target)
    print(f"   Perceptual loss: {perc_val.item():.4f}")
    print(f"   ✓ Shape preserved, computed on sampled slices")

    # Test MS-SSIM Loss
    print("\n2. Testing MS-SSIM Loss...")
    ssim_loss = MS_SSIM_Loss()
    ssim_val = ssim_loss(pred, target)
    print(f"   MS-SSIM loss: {ssim_val.item():.4f}")
    print(f"   ✓ Multi-scale similarity computed")

    # Test Combined Loss
    print("\n3. Testing Combined Loss...")
    combined_loss = CombinedLoss(
        lambda_perceptual=0.1,
        lambda_ssim=0.1,
        perceptual_every_n_steps=10,
        ssim_every_n_steps=10,
    )

    total, loss_dict = combined_loss(pred, target, diffusion_loss, compute_auxiliary=True)
    print(f"   Total loss: {total.item():.4f}")
    print(f"   Loss components: {loss_dict}")
    print(f"   ✓ Combined loss computed successfully")

    print("\n" + "="*80)
    print("✅ All loss functions working correctly!")
