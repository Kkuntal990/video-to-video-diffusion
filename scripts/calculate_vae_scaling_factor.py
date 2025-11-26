#!/usr/bin/env python3
"""
Calculate Optimal VAE Scaling Factor

This script calculates the appropriate scaling factor for the VAE latent space
by measuring the standard deviation of latent activations on the training dataset.

The scaling factor is used to normalize latent representations to a range
suitable for diffusion training (typically ~[-1, 1]).

Usage:
    python scripts/calculate_vae_scaling_factor.py \
        --config config/cloud_train_config_a100.yaml \
        --checkpoint /path/to/checkpoint.pt \
        --num-samples 100
"""

import argparse
import yaml
import torch
import logging
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import VideoToVideoDiffusion
from data import get_unified_dataloader as get_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_scaling_factor(
    checkpoint_path: str,
    config_path: str,
    num_samples: int = 100,
    device: str = 'cuda'
):
    """
    Calculate the optimal VAE scaling factor

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        num_samples: Number of samples to use for calculation
        device: Device to run on
    """
    logger.info(f"{'='*70}")
    logger.info(f"VAE Scaling Factor Calculation")
    logger.info(f"{'='*70}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Samples: {num_samples}")

    # Load model (standard PyTorch pattern)
    logger.info("\\nLoading model from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VideoToVideoDiffusion(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Current VAE scaling factor: {model.vae.scaling_factor}")

    # Load dataloader
    logger.info(f"\\nLoading training dataset...")
    dataloader = get_dataloader(config['data'], split='train')

    # Collect latent statistics
    logger.info(f"\\nCalculating latent statistics...")

    latent_values = []
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if sample_count >= num_samples:
                break

            v_gt = batch['target'].to(device)

            # Encode to latent space WITHOUT scaling
            # We need to temporarily disable scaling
            original_scaling = model.vae.scaling_factor
            model.vae.scaling_factor = 1.0  # Disable scaling temporarily

            z_gt = model.vae.encoder(v_gt)  # Use encoder directly

            # Restore original scaling
            model.vae.scaling_factor = original_scaling

            # Collect latent values
            latent_values.append(z_gt.cpu().flatten())

            sample_count += v_gt.size(0)

    # Concatenate all latent values
    all_latents = torch.cat(latent_values)

    # Calculate statistics
    mean = all_latents.mean().item()
    std = all_latents.std().item()
    min_val = all_latents.min().item()
    max_val = all_latents.max().item()

    # Calculate optimal scaling factor
    # Standard approach: scale = 1 / std
    # This makes the latent space have unit std (similar to normalized data)
    optimal_scaling_factor = 1.0 / std

    logger.info(f"\\n{'='*70}")
    logger.info(f"Latent Statistics (Unscaled)")
    logger.info(f"{'='*70}")
    logger.info(f"Mean: {mean:.6f}")
    logger.info(f"Std: {std:.6f}")
    logger.info(f"Min: {min_val:.6f}")
    logger.info(f"Max: {max_val:.6f}")

    logger.info(f"\\n{'='*70}")
    logger.info(f"Recommended Scaling Factor")
    logger.info(f"{'='*70}")
    logger.info(f"Optimal scaling factor: {optimal_scaling_factor:.6f}")
    logger.info(f"  (This is 1/std = 1/{std:.6f})")

    # Show what the scaled statistics would be
    logger.info(f"\\nWith this scaling factor, latents will have:")
    logger.info(f"  Mean: {mean * optimal_scaling_factor:.6f}")
    logger.info(f"  Std: {std * optimal_scaling_factor:.6f} ≈ 1.0")
    logger.info(f"  Min: {min_val * optimal_scaling_factor:.6f}")
    logger.info(f"  Max: {max_val * optimal_scaling_factor:.6f}")

    logger.info(f"\\n{'='*70}")
    logger.info(f"Usage")
    logger.info(f"{'='*70}")
    logger.info(f"Update your VAE initialization with:")
    logger.info(f"  vae = VideoVAE(..., scaling_factor={optimal_scaling_factor:.6f})")
    logger.info(f"")
    logger.info(f"Or update the config file:")
    logger.info(f"  model:")
    logger.info(f"    vae_scaling_factor: {optimal_scaling_factor:.6f}")

    # Compare with Stable Diffusion scaling factor
    sd_scaling = 0.18215
    logger.info(f"\\nComparison with Stable Diffusion:")
    logger.info(f"  SD VAE scaling factor: {sd_scaling:.6f}")
    logger.info(f"  Your optimal factor: {optimal_scaling_factor:.6f}")
    logger.info(f"  Ratio: {optimal_scaling_factor / sd_scaling:.2f}x")

    if abs(optimal_scaling_factor - sd_scaling) / sd_scaling > 0.5:
        logger.warning(f"\\n⚠️  WARNING: Your scaling factor differs significantly from SD VAE!")
        logger.warning(f"  This confirms the domain gap between natural images and CT scans.")
        logger.warning(f"  Using the optimal factor ({optimal_scaling_factor:.6f}) is recommended.")

    return optimal_scaling_factor


def main():
    parser = argparse.ArgumentParser(description='Calculate optimal VAE scaling factor')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/cloud_train_config_a100.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to use for calculation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (cuda or cpu)'
    )

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    # Check config exists
    if not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        return

    # Calculate scaling factor
    scaling_factor = calculate_scaling_factor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_samples=args.num_samples,
        device=args.device
    )

    # Save to file
    output_file = Path('vae_scaling_factor.txt')
    with open(output_file, 'w') as f:
        f.write(f"Optimal VAE Scaling Factor: {scaling_factor:.6f}\\n")
        f.write(f"\\n")
        f.write(f"Usage in code:\\n")
        f.write(f"vae = VideoVAE(..., scaling_factor={scaling_factor:.6f})\\n")

    logger.info(f"\\nScaling factor saved to: {output_file}")


if __name__ == "__main__":
    main()
