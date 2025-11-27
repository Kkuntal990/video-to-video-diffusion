"""
VAE Reconstruction Quality Evaluation

Evaluates the reconstruction quality of a trained VAE on validation patches.
Calculates PSNR and SSIM metrics and creates visualizations.

Usage:
    python scripts/evaluate_vae_reconstruction.py \\
        --checkpoint /path/to/vae_best.pt \\
        --config config/vae_training.yaml \\
        --slice-type thin \\
        --num-samples 50 \\
        --output-dir results/vae_eval/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data import get_unified_dataloader
from models.vae import VideoVAE
from utils.metrics import calculate_psnr, calculate_ssim

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate VAE reconstruction quality')

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to VAE checkpoint (e.g., vae_best.pt)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to VAE training config YAML')

    # Evaluation settings
    parser.add_argument('--slice-type', type=str, default='thin', choices=['thin', 'thick'],
                       help='Which slices to evaluate (thin or thick)')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of patches to evaluate (-1 for all)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for evaluation')

    # Output settings
    parser.add_argument('--output-dir', type=str, default='results/vae_eval',
                       help='Directory to save results')
    parser.add_argument('--save-reconstructions', action='store_true',
                       help='Save reconstruction tensors as .pt files')
    parser.add_argument('--visualize-slices', type=str, default='0,24,47',
                       help='Comma-separated slice indices to visualize (for thin slices)')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def load_vae_checkpoint(checkpoint_path: str, config: Dict, device: str) -> VideoVAE:
    """Load VAE model from checkpoint"""

    logger.info(f"Loading VAE from {checkpoint_path}...")

    # Create VAE model
    vae = VideoVAE(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim'],
        base_channels=config['model']['vae_base_channels']
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        best_loss = checkpoint.get('best_loss', 'unknown')
        logger.info(f"  Checkpoint epoch: {epoch}, Best loss: {best_loss}")
    else:
        state_dict = checkpoint
        logger.info("  Loaded state dict directly")

    # Load weights
    missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=True)
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")

    # Move to device and set eval mode
    vae = vae.to(device)
    vae.eval()

    logger.info("✓ VAE loaded successfully")

    return vae


def evaluate_reconstruction(
    vae: VideoVAE,
    input_tensor: torch.Tensor,
    device: str
) -> Dict:
    """
    Evaluate VAE reconstruction on a single patch.

    Args:
        vae: VAE model
        input_tensor: Input tensor (B, C, D, H, W) range [-1, 1]
        device: Device

    Returns:
        Dict with 'reconstruction' and metrics
    """
    with torch.no_grad():
        input_tensor = input_tensor.to(device)

        # Encode-decode in FP32 for stability
        with torch.cuda.amp.autocast(enabled=False):
            # Encode
            latent = vae.encode(input_tensor)

            # Check for NaN
            if torch.isnan(latent).any():
                logger.warning("NaN detected in latent! Replacing with zeros.")
                latent = torch.nan_to_num(latent, nan=0.0)

            # Decode
            reconstruction = vae.decode(latent)

            # Check for NaN
            if torch.isnan(reconstruction).any():
                logger.warning("NaN detected in reconstruction! Replacing with zeros.")
                reconstruction = torch.nan_to_num(reconstruction, nan=0.0)

        # Clamp to valid range and normalize to [0, 1] for metrics (match VAE training)
        reconstruction = torch.clamp(reconstruction, -1, 1)
        input_clamped = torch.clamp(input_tensor, -1, 1)
        reconstruction_norm = (reconstruction + 1.0) / 2.0  # [-1, 1] → [0, 1]
        input_norm = (input_clamped + 1.0) / 2.0            # [-1, 1] → [0, 1]

        # Calculate metrics with max_val=1.0 for [0, 1] range (consistent with VAE training)
        psnr = calculate_psnr(reconstruction_norm, input_norm, max_val=1.0)
        ssim = calculate_ssim(reconstruction_norm, input_norm, max_val=1.0)

        metrics = {
            'psnr': psnr.item() if isinstance(psnr, torch.Tensor) else psnr,
            'ssim': ssim.item() if isinstance(ssim, torch.Tensor) else ssim
        }

        return reconstruction, metrics


def create_reconstruction_visualization(
    input_patch: torch.Tensor,
    reconstruction: torch.Tensor,
    slice_indices: List[int],
    output_path: Path,
    metadata: Dict
):
    """
    Create 2-row comparison visualization: Input (top) vs Reconstruction (bottom)

    Args:
        input_patch: (1, 1, D, H, W) range [-1, 1]
        reconstruction: (1, 1, D, H, W) range [-1, 1]
        slice_indices: Which slices to show
        output_path: Where to save
        metadata: Dict with patient_id, category, psnr, ssim
    """
    # Move to CPU and convert to numpy
    input_patch = input_patch.cpu().squeeze().numpy()  # (D, H, W)
    reconstruction = reconstruction.cpu().squeeze().numpy()  # (D, H, W)

    # Normalize to [0, 1] for display
    input_patch = (input_patch + 1) / 2
    reconstruction = (reconstruction + 1) / 2

    n_slices = len(slice_indices)
    fig, axes = plt.subplots(2, n_slices, figsize=(4*n_slices, 8))

    # Handle single slice case
    if n_slices == 1:
        axes = axes.reshape(2, 1)

    for col_idx, slice_idx in enumerate(slice_indices):
        # Ensure slice index is valid
        if slice_idx >= input_patch.shape[0]:
            slice_idx = input_patch.shape[0] - 1

        # Row 0: Input
        axes[0, col_idx].imshow(input_patch[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, col_idx].set_title(f'Input - Slice {slice_idx}')
        axes[0, col_idx].axis('off')

        # Row 1: Reconstruction
        axes[1, col_idx].imshow(reconstruction[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[1, col_idx].set_title(f'Reconstruction - Slice {slice_idx}')
        axes[1, col_idx].axis('off')

    # Overall title with metrics
    fig.suptitle(
        f"Patient: {metadata['patient_id']} | {metadata['category']} | "
        f"PSNR: {metadata['psnr']:.2f} dB | SSIM: {metadata['ssim']:.4f}",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    metrics_dir = output_dir / 'metrics'
    metrics_dir.mkdir(exist_ok=True)

    if args.save_reconstructions:
        recon_dir = output_dir / 'reconstructions'
        recon_dir.mkdir(exist_ok=True)

    # Banner
    print("=" * 80)
    print("VAE Reconstruction Quality Evaluation")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Slice type: {args.slice_type}")
    print()

    # Load config
    print(f"Loading config from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Print patch configuration
    patch_depth = config['data']['patch_depth_thin'] if args.slice_type == 'thin' else config['data']['patch_depth_thick']
    patch_spatial = config['data']['patch_size']
    print(f"\nPatch Configuration:")
    print(f"  Slices: {patch_depth} ({args.slice_type})")
    print(f"  Spatial: {patch_spatial[0]}×{patch_spatial[1]}")
    print()

    # Load VAE
    print("Loading VAE model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    vae = load_vae_checkpoint(args.checkpoint, config, device)
    print()

    # Create dataloader
    print(f"Creating {args.split} dataloader...")

    # Override batch_size in config for evaluation
    config['data']['batch_size'] = args.batch_size

    # get_unified_dataloader expects just the 'data' portion of config
    dataloader = get_unified_dataloader(
        config=config['data'],
        split=args.split
    )

    num_samples = len(dataloader) if args.num_samples == -1 else args.num_samples
    print(f"  Dataset size: {len(dataloader.dataset)} patients")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Evaluating {num_samples} patches")
    print()

    # Parse visualization slice indices
    vis_slices = [int(x.strip()) for x in args.visualize_slices.split(',')]

    # Evaluation loop
    print("=" * 80)
    print("Starting Evaluation...")
    print("=" * 80)
    print()

    all_metrics = []

    pbar = tqdm(enumerate(dataloader), total=num_samples, desc='Evaluating patches')

    for batch_idx, batch in pbar:
        if batch_idx >= num_samples:
            break

        # Get input based on slice type
        if args.slice_type == 'thin':
            input_tensor = batch['target']  # (B, 1, D_thin, H, W)
        else:
            input_tensor = batch['input']  # (B, 1, D_thick, H, W)

        # Get metadata
        patient_id = batch['patient_id'][0] if 'patient_id' in batch else f'patch_{batch_idx:04d}'
        category = batch['category'][0] if 'category' in batch else 'unknown'

        # Evaluate reconstruction
        reconstruction, metrics = evaluate_reconstruction(vae, input_tensor, device)

        # Update progress bar
        pbar.set_postfix({
            'PSNR': f"{metrics['psnr']:.2f}",
            'SSIM': f"{metrics['ssim']:.4f}"
        })

        # Save metrics
        metrics_entry = {
            'patch_idx': batch_idx,
            'patient_id': patient_id,
            'category': category,
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim']
        }
        all_metrics.append(metrics_entry)

        # Create visualization
        vis_metadata = {
            'patient_id': patient_id,
            'category': category,
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim']
        }

        vis_path = vis_dir / f"patch_{batch_idx:04d}_{patient_id}.png"
        create_reconstruction_visualization(
            input_tensor,
            reconstruction,
            vis_slices,
            vis_path,
            vis_metadata
        )

        # Save reconstruction tensor if requested
        if args.save_reconstructions:
            recon_path = recon_dir / f"patch_{batch_idx:04d}_reconstruction.pt"
            torch.save(reconstruction.cpu(), recon_path)

    print()

    # Calculate statistics
    print("=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print()

    psnr_values = [m['psnr'] for m in all_metrics]
    ssim_values = [m['ssim'] for m in all_metrics]

    print(f"Overall Statistics ({len(all_metrics)} patches):")
    print(f"  PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
    print(f"    Min: {np.min(psnr_values):.2f}, Max: {np.max(psnr_values):.2f}")
    print(f"  SSIM: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
    print(f"    Min: {np.min(ssim_values):.4f}, Max: {np.max(ssim_values):.4f}")
    print()

    # Per-category statistics
    categories = set(m['category'] for m in all_metrics)
    if len(categories) > 1:
        print("Per-Category Statistics:")
        for category in sorted(categories):
            cat_metrics = [m for m in all_metrics if m['category'] == category]
            cat_psnr = [m['psnr'] for m in cat_metrics]
            cat_ssim = [m['ssim'] for m in cat_metrics]

            print(f"  {category} ({len(cat_metrics)} patches):")
            print(f"    PSNR: {np.mean(cat_psnr):.2f} ± {np.std(cat_psnr):.2f} dB")
            print(f"    SSIM: {np.mean(cat_ssim):.4f} ± {np.std(cat_ssim):.4f}")
        print()

    # Best/worst patches
    sorted_by_psnr = sorted(all_metrics, key=lambda x: x['psnr'], reverse=True)

    print("Best Patches (by PSNR):")
    for i, m in enumerate(sorted_by_psnr[:5], 1):
        print(f"  {i}. Patch {m['patch_idx']:04d} ({m['patient_id']}): "
              f"PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}")
    print()

    print("Worst Patches (by PSNR):")
    for i, m in enumerate(sorted_by_psnr[-5:][::-1], 1):
        print(f"  {i}. Patch {m['patch_idx']:04d} ({m['patient_id']}): "
              f"PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}")
    print()

    # Save metrics
    results = {
        'config': {
            'checkpoint': str(args.checkpoint),
            'slice_type': args.slice_type,
            'num_samples': len(all_metrics),
            'split': args.split
        },
        'overall': {
            'psnr_mean': float(np.mean(psnr_values)),
            'psnr_std': float(np.std(psnr_values)),
            'psnr_min': float(np.min(psnr_values)),
            'psnr_max': float(np.max(psnr_values)),
            'ssim_mean': float(np.mean(ssim_values)),
            'ssim_std': float(np.std(ssim_values)),
            'ssim_min': float(np.min(ssim_values)),
            'ssim_max': float(np.max(ssim_values))
        },
        'per_patch': all_metrics
    }

    json_path = metrics_dir / 'vae_reconstruction_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Metrics saved to: {json_path}")

    # Save CSV
    import csv
    csv_path = metrics_dir / 'vae_reconstruction_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patch_idx', 'patient_id', 'category', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"CSV saved to: {csv_path}")
    print()

    # Summary
    print("=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print()
    print(f"Results saved to: {output_dir}")
    print(f"  Visualizations: {len(all_metrics)} images")
    print(f"  Metrics: JSON + CSV")
    if args.save_reconstructions:
        print(f"  Reconstructions: {len(all_metrics)} .pt files")
    print()


if __name__ == '__main__':
    main()
