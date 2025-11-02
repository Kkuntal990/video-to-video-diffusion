#!/usr/bin/env python3
"""
Test MAISI VAE Integration

This script tests the NVIDIA MAISI VAE integration with the video diffusion pipeline.
Tests include:
1. MAISI VAE loading
2. Encode/decode cycle
3. Reconstruction quality
4. Latent space statistics
5. Integration with diffusion model

Usage:
    python scripts/test_maisi_vae.py \
        --maisi-checkpoint ./pretrained/maisi_vae/models/autoencoder.pt \
        --config config/cloud_train_config_a100.yaml
"""

import argparse
import yaml
import torch
import logging
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import VideoVAE, VideoToVideoDiffusion
from utils.metrics import calculate_video_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_maisi_vae_loading(maisi_checkpoint):
    """
    Test 1: MAISI VAE loading

    Args:
        maisi_checkpoint: Path to MAISI checkpoint
    """
    logger.info(f"\\n{'='*70}")
    logger.info(f"Test 1: MAISI VAE Loading")
    logger.info(f"{'='*70}")

    try:
        vae = VideoVAE(
            in_channels=1,
            latent_dim=3,
            use_maisi=True,
            maisi_checkpoint=maisi_checkpoint
        )
        logger.info("✓ MAISI VAE loaded successfully")
        logger.info(f"  Latent dim: {vae.latent_dim}")
        logger.info(f"  Scaling factor: {vae.scaling_factor}")
        logger.info(f"  Input channels: {vae.in_channels}")
        return vae
    except Exception as e:
        logger.error(f"✗ Failed to load MAISI VAE: {e}")
        raise


def test_encode_decode(vae, device='cuda'):
    """
    Test 2: Encode/Decode cycle

    Args:
        vae: VideoVAE instance
        device: Device to run on
    """
    logger.info(f"\\n{'='*70}")
    logger.info(f"Test 2: Encode/Decode Cycle")
    logger.info(f"{'='*70}")

    vae = vae.to(device)
    vae.eval()

    # Create synthetic grayscale video (simulating CT scan)
    # Range: [-1, 1] (normalized CT)
    batch_size = 2
    num_frames = 16
    height, width = 256, 256

    logger.info(f"\\nCreating synthetic CT video:")
    logger.info(f"  Shape: ({batch_size}, 1, {num_frames}, {height}, {width})")
    logger.info(f"  Range: [-1, 1]")

    video = torch.randn(batch_size, 1, num_frames, height, width).to(device)
    video = torch.clamp(video, -1, 1)  # Ensure in range

    logger.info(f"  Min: {video.min():.4f}, Max: {video.max():.4f}, Mean: {video.mean():.4f}")

    # Encode
    logger.info(f"\\nEncoding...")
    with torch.no_grad():
        latent = vae.encode(video)

    logger.info(f"  Latent shape: {latent.shape}")
    logger.info(f"  Latent range: [{latent.min():.4f}, {latent.max():.4f}]")
    logger.info(f"  Latent mean: {latent.mean():.4f}, std: {latent.std():.4f}")

    # Decode
    logger.info(f"\\nDecoding...")
    with torch.no_grad():
        recon = vae.decode(latent)

    logger.info(f"  Reconstruction shape: {recon.shape}")
    logger.info(f"  Reconstruction range: [{recon.min():.4f}, {recon.max():.4f}]")

    # Calculate reconstruction error
    mse = torch.nn.functional.mse_loss(recon, video).item()
    mae = torch.nn.functional.l1_loss(recon, video).item()

    logger.info(f"\\nReconstruction Quality:")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")

    # Expected: MSE < 0.01 for good reconstruction (pretrained VAE)
    #          MSE > 0.5 for random initialization
    if mse < 0.05:
        logger.info(f"  ✓ Excellent reconstruction (pretrained weights likely loaded)")
    elif mse < 0.2:
        logger.info(f"  ✓ Good reconstruction")
    else:
        logger.warning(f"  ⚠ Poor reconstruction - MAISI may be randomly initialized")
        logger.warning(f"     This is normal if you haven't downloaded pretrained weights")

    return video, recon


def test_latent_statistics(vae, num_samples=20, device='cuda'):
    """
    Test 3: Latent space statistics

    Args:
        vae: VideoVAE instance
        num_samples: Number of samples to test
        device: Device to run on
    """
    logger.info(f"\\n{'='*70}")
    logger.info(f"Test 3: Latent Space Statistics")
    logger.info(f"{'='*70}")

    vae = vae.to(device)
    vae.eval()

    latents = []

    logger.info(f"\\nGenerating {num_samples} synthetic CT volumes...")

    with torch.no_grad():
        for i in range(num_samples):
            video = torch.randn(1, 1, 16, 256, 256).to(device)
            video = torch.clamp(video, -1, 1)

            latent = vae.encode(video)
            latents.append(latent.cpu())

    all_latents = torch.cat(latents).flatten()

    mean = all_latents.mean().item()
    std = all_latents.std().item()
    min_val = all_latents.min().item()
    max_val = all_latents.max().item()

    logger.info(f"\\nLatent Statistics (over {num_samples} samples):")
    logger.info(f"  Mean: {mean:.6f}")
    logger.info(f"  Std: {std:.6f}")
    logger.info(f"  Min: {min_val:.6f}")
    logger.info(f"  Max: {max_val:.6f}")

    # Check if latents are in reasonable range for diffusion
    if abs(mean) < 0.5 and 0.5 < std < 2.0:
        logger.info(f"  ✓ Latent statistics are in good range for diffusion")
    else:
        logger.warning(f"  ⚠ Latent statistics may be suboptimal for diffusion")
        logger.warning(f"     Expected: mean ≈ 0, std ≈ 1")


def test_diffusion_integration(config_path, maisi_checkpoint, device='cuda'):
    """
    Test 4: Integration with full diffusion model

    Args:
        config_path: Path to config file
        maisi_checkpoint: Path to MAISI checkpoint
        device: Device to run on
    """
    logger.info(f"\\n{'='*70}")
    logger.info(f"Test 4: Diffusion Model Integration")
    logger.info(f"{'='*70}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override MAISI checkpoint path
    if 'pretrained' not in config:
        config['pretrained'] = {}
    if 'vae' not in config['pretrained']:
        config['pretrained']['vae'] = {}

    config['pretrained']['use_pretrained'] = True
    config['pretrained']['vae']['enabled'] = True
    config['pretrained']['vae']['use_maisi'] = True
    config['pretrained']['vae']['checkpoint_path'] = maisi_checkpoint

    logger.info(f"\\nCreating full diffusion model...")
    model = VideoToVideoDiffusion(config['model'], load_pretrained=True)
    model = model.to(device)
    model.eval()

    logger.info(f"  ✓ Model created successfully")
    logger.info(f"  VAE latent dim: {model.vae.latent_dim}")
    logger.info(f"  U-Net latent dim: {model.unet.latent_dim}")

    # Test forward pass
    logger.info(f"\\nTesting forward pass...")
    batch_size = 1
    video_in = torch.randn(batch_size, 1, 16, 256, 256).to(device)
    video_gt = torch.randn(batch_size, 1, 16, 256, 256).to(device)

    video_in = torch.clamp(video_in, -1, 1)
    video_gt = torch.clamp(video_gt, -1, 1)

    with torch.no_grad():
        try:
            loss, metrics = model(video_in, video_gt)
            logger.info(f"  ✓ Forward pass successful")
            logger.info(f"  Loss: {loss.item():.6f}")
            logger.info(f"  Metrics: {metrics}")
        except Exception as e:
            logger.error(f"  ✗ Forward pass failed: {e}")
            raise

    # Test generation
    logger.info(f"\\nTesting generation...")
    with torch.no_grad():
        try:
            generated = model.generate(video_in, sampler='ddim', num_inference_steps=10)
            logger.info(f"  ✓ Generation successful")
            logger.info(f"  Generated shape: {generated.shape}")
            logger.info(f"  Generated range: [{generated.min():.4f}, {generated.max():.4f}]")
        except Exception as e:
            logger.error(f"  ✗ Generation failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Test MAISI VAE integration')
    parser.add_argument(
        '--maisi-checkpoint',
        type=str,
        default='./pretrained/maisi_vae/models/autoencoder.pt',
        help='Path to MAISI VAE checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/cloud_train_config_a100.yaml',
        help='Path to model config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run tests on'
    )

    args = parser.parse_args()

    logger.info(f"{'='*70}")
    logger.info(f"MAISI VAE Integration Test Suite")
    logger.info(f"{'='*70}")
    logger.info(f"MAISI checkpoint: {args.maisi_checkpoint}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {args.device}")

    try:
        # Test 1: Loading
        vae = test_maisi_vae_loading(args.maisi_checkpoint)

        # Test 2: Encode/Decode
        video, recon = test_encode_decode(vae, args.device)

        # Test 3: Latent statistics
        test_latent_statistics(vae, num_samples=10, device=args.device)

        # Test 4: Diffusion integration
        test_diffusion_integration(args.config, args.maisi_checkpoint, args.device)

        logger.info(f"\\n{'='*70}")
        logger.info(f"✓ ALL TESTS PASSED")
        logger.info(f"{'='*70}")
        logger.info(f"\\nMAISI VAE integration is working correctly!")
        logger.info(f"You can now proceed with training.")

    except Exception as e:
        logger.error(f"\\n{'='*70}")
        logger.error(f"✗ TESTS FAILED")
        logger.error(f"{'='*70}")
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
