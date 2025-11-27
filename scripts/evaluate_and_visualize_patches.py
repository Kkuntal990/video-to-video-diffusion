"""
Comprehensive Patch Evaluation and Visualization Script

Evaluates and visualizes CT slice interpolation model on patch-based validation data.
- Loads checkpoint with model_suffix support
- Evaluates patches (8 thick → 48 thin slices @ 192×192)
- Calculates PSNR & SSIM per patch
- Generates comparison visualizations
- Saves detailed metrics (JSON + CSV)
- Shows best/worst patches and statistics

Usage:
    python scripts/evaluate_and_visualize_patches.py \
        --checkpoint /path/to/checkpoint_best_epoch_24_slice_interp_full3.pt \
        --config config/slice_interpolation_full_medium.yaml \
        --num-samples 20 \
        --output-dir results/patch_eval/
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import VideoToVideoDiffusion
from data.patch_slice_interpolation_dataset import get_patch_dataloader
from utils.checkpoint import load_model_from_checkpoint, find_best_checkpoint
from utils.metrics import calculate_video_metrics
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate and visualize patch-based CT slice interpolation')

    # Model and data
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='config/slice_interpolation_full_medium.yaml',
                        help='Path to config file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')

    # Evaluation settings
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of patches to evaluate (default: 20)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation (default: 1)')
    parser.add_argument('--num-inference-steps', type=int, default=20,
                        help='Number of DDIM steps for generation (default: 20)')
    parser.add_argument('--sampler', type=str, default='ddim', choices=['ddpm', 'ddim'],
                        help='Sampling method (default: ddim)')

    # Output settings
    parser.add_argument('--output-dir', type=str, default='results/patch_eval',
                        help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save prediction tensors (.pt files)')
    parser.add_argument('--visualize-slices', type=str, default='0,24,47',
                        help='Slice indices to visualize (comma-separated, default: 0,24,47)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def setup_output_dirs(output_dir: str) -> Dict[str, Path]:
    """Create output directory structure."""
    output_dir = Path(output_dir)

    dirs = {
        'root': output_dir,
        'visualizations': output_dir / 'visualizations',
        'metrics': output_dir / 'metrics',
        'predictions': output_dir / 'predictions',
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_comparison_visualization(
    input_patch: np.ndarray,
    target_patch: np.ndarray,
    pred_patch: np.ndarray,
    slice_indices: List[int],
    output_path: str,
    metadata: Dict
):
    """
    Create side-by-side comparison visualization.

    Args:
        input_patch: Input thick slices (8, 192, 192)
        target_patch: Ground truth thin slices (48, 192, 192)
        pred_patch: Predicted thin slices (48, 192, 192)
        slice_indices: Which slices to visualize from thin patches
        output_path: Where to save the visualization
        metadata: Additional info (PSNR, SSIM, patient ID, etc.)
    """
    # Create figure with 3 rows (input, target, prediction) × N columns (slices)
    n_slices = len(slice_indices)
    fig, axes = plt.subplots(3, n_slices, figsize=(4*n_slices, 12))

    if n_slices == 1:
        axes = axes.reshape(3, 1)

    # Row labels (TEMPORARY DEBUG: Testing VAE reconstruction)
    row_labels = ['Input (Thick)', 'Target (Thin)', 'VAE Reconstruction']

    for col_idx, slice_idx in enumerate(slice_indices):
        # Input: map thin slice index to thick slice index (6× ratio)
        thick_idx = min(slice_idx // 6, input_patch.shape[0] - 1)

        # Row 0: Input thick slice
        axes[0, col_idx].imshow(input_patch[thick_idx], cmap='gray', vmin=-1, vmax=1)
        axes[0, col_idx].set_title(f'Thick Slice {thick_idx}')
        axes[0, col_idx].axis('off')

        # Row 1: Target thin slice
        axes[1, col_idx].imshow(target_patch[slice_idx], cmap='gray', vmin=-1, vmax=1)
        axes[1, col_idx].set_title(f'Target Slice {slice_idx}')
        axes[1, col_idx].axis('off')

        # Row 2: VAE Reconstruction (TEMPORARY DEBUG)
        axes[2, col_idx].imshow(pred_patch[slice_idx], cmap='gray', vmin=-1, vmax=1)
        axes[2, col_idx].set_title(f'VAE Recon Slice {slice_idx}')
        axes[2, col_idx].axis('off')

    # Add row labels
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=14, fontweight='bold')

    # Add title with metrics (TEMPORARY DEBUG: VAE Reconstruction Test)
    title = f"[DEBUG: VAE RECONSTRUCTION TEST] Patient: {metadata.get('patient_id', 'N/A')} | Category: {metadata.get('category', 'N/A')}\n"
    title += f"PSNR: {metadata.get('psnr', 0):.2f} dB | SSIM: {metadata.get('ssim', 0):.4f}"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_patch(
    model: VideoToVideoDiffusion,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    device: str,
    sampler: str = 'ddim',
    num_inference_steps: int = 20
) -> Tuple[torch.Tensor, Dict]:
    """
    Evaluate a single patch.

    Args:
        model: Trained model
        input_tensor: Input thick slices (1, 1, 8, 192, 192)
        target_tensor: Target thin slices (1, 1, 48, 192, 192)
        device: Device to use
        sampler: Sampling method
        num_inference_steps: Number of diffusion steps

    Returns:
        Tuple of (prediction_tensor, metrics_dict)
    """
    with torch.no_grad():
        # Move to device
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        # Get target depth
        target_depth = target_tensor.shape[2]

        # TEMPORARY DEBUG: Bypass diffusion, test VAE reconstruction only
        # Generate prediction (FP32 for stability)
        with torch.cuda.amp.autocast(enabled=False):
            # Original diffusion generation (commented out for debugging)
            # prediction = model.generate(
            #     input_tensor,
            #     sampler=sampler,
            #     num_inference_steps=num_inference_steps,
            #     target_depth=target_depth
            # )

            # DEBUG: Test VAE reconstruction with skip connections
            # CRITICAL: Use forward() to enable skip connections (U-Net style)
            prediction, _ = model.vae(target_tensor)  # Encode-decode with skip connections

        # Clamp values and normalize to [0, 1] for metrics (consistent with VAE training)
        prediction = torch.clamp(prediction, -1, 1)
        target_clamped = torch.clamp(target_tensor, -1, 1)
        prediction_norm = (prediction + 1.0) / 2.0      # [-1, 1] → [0, 1]
        target_norm = (target_clamped + 1.0) / 2.0      # [-1, 1] → [0, 1]

        # Calculate metrics with max_val=1.0 for [0, 1] range
        metrics = calculate_video_metrics(prediction_norm, target_norm, max_val=1.0)

        return prediction, metrics


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup output directories
    print(f"\n{'='*80}")
    print(f"CT Slice Interpolation - Patch Evaluation & Visualization")
    print(f"{'='*80}\n")

    # TEMPORARY DEBUG MODE
    print(f"\n{'!'*80}")
    print(f"! DEBUG MODE: BYPASSING DIFFUSION - TESTING VAE RECONSTRUCTION ONLY")
    print(f"! This will encode target patches with VAE and decode them back")
    print(f"! Expected PSNR: 38-45 dB if VAE is working correctly")
    print(f"{'!'*80}\n")

    output_dirs = setup_output_dirs(args.output_dir)
    print(f"Output directory: {output_dirs['root']}")

    # Load config
    print(f"\nLoading config from {args.config}...")
    config = load_config(args.config)

    # Check if patch mode is enabled
    use_patches = config['data'].get('use_patches', False)
    if not use_patches:
        print("WARNING: Config has use_patches=False. This script is designed for patch-based evaluation.")
        print("Continuing anyway...")

    # Print patch configuration
    patch_depth_thin = config['data'].get('patch_depth_thin', 48)
    patch_depth_thick = config['data'].get('patch_depth_thick', 8)
    patch_size = config['data'].get('patch_size', [192, 192])

    print(f"\nPatch Configuration:")
    print(f"  Input:  {patch_depth_thick} slices @ {patch_size[0]}×{patch_size[1]} (thick)")
    print(f"  Output: {patch_depth_thin} slices @ {patch_size[0]}×{patch_size[1]} (thin)")
    print(f"  Ratio:  {patch_depth_thin/patch_depth_thick:.1f}× depth interpolation")

    # Load model
    print(f"\nLoading model...")
    model = VideoToVideoDiffusion(config)
    print(f"  Model created: {model.__class__.__name__}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    model, checkpoint_metadata = load_model_from_checkpoint(
        model, args.checkpoint, device=args.device, strict=True
    )

    epoch = checkpoint_metadata.get('epoch', 'unknown')
    best_loss = checkpoint_metadata.get('best_loss', 'unknown')
    print(f"  Checkpoint epoch: {epoch}")
    print(f"  Best loss: {best_loss}")

    model.eval()

    # Create dataloader
    print(f"\nCreating {args.split} dataloader...")
    dataloader = get_patch_dataloader(
        processed_dir=config['data']['processed_dir'],
        config=config['data'],
        split=args.split
    )

    print(f"  Dataset size: {len(dataloader.dataset)} patients")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Evaluating {args.num_samples} patches")

    # Prepare for evaluation
    all_metrics = []
    slice_indices = [int(x) for x in args.visualize_slices.split(',')]

    print(f"\n{'='*80}")
    print(f"Starting Evaluation...")
    print(f"{'='*80}\n")

    # Evaluate patches
    pbar = tqdm(dataloader, desc="Evaluating patches", total=min(args.num_samples, len(dataloader)))

    for idx, batch in enumerate(pbar):
        if idx >= args.num_samples:
            break

        # Extract data
        input_patch = batch['input']  # (B, 1, 8, 192, 192)
        target_patch = batch['target']  # (B, 1, 48, 192, 192)
        patient_id = batch.get('patient_id', ['unknown'])[0]
        category = batch.get('category', ['unknown'])[0]

        # Evaluate
        prediction, metrics = evaluate_patch(
            model, input_patch, target_patch,
            device=args.device,
            sampler=args.sampler,
            num_inference_steps=args.num_inference_steps
        )

        # Store metrics
        patch_metrics = {
            'patch_id': idx,
            'patient_id': patient_id,
            'category': category,
            'psnr': float(metrics['psnr']),
            'ssim': float(metrics['ssim']),
        }
        all_metrics.append(patch_metrics)

        # Update progress bar
        pbar.set_postfix({
            'PSNR': f"{metrics['psnr']:.2f}",
            'SSIM': f"{metrics['ssim']:.4f}"
        })

        # Create visualization
        vis_path = output_dirs['visualizations'] / f"patch_{idx:04d}_{patient_id}.png"
        create_comparison_visualization(
            input_patch[0, 0].cpu().numpy(),  # (8, 192, 192)
            target_patch[0, 0].cpu().numpy(),  # (48, 192, 192)
            prediction[0, 0].cpu().numpy(),  # (48, 192, 192)
            slice_indices,
            str(vis_path),
            metadata={
                'patient_id': patient_id,
                'category': category,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim']
            }
        )

        # Save predictions if requested
        if args.save_predictions:
            pred_path = output_dirs['predictions'] / f"patch_{idx:04d}_prediction.pt"
            torch.save({
                'prediction': prediction.cpu(),
                'input': input_patch.cpu(),
                'target': target_patch.cpu(),
                'patient_id': patient_id,
                'category': category,
                'metrics': metrics
            }, pred_path)

    # Calculate statistics
    print(f"\n{'='*80}")
    print(f"Evaluation Results")
    print(f"{'='*80}\n")

    psnr_values = [m['psnr'] for m in all_metrics]
    ssim_values = [m['ssim'] for m in all_metrics]

    print(f"Overall Statistics ({len(all_metrics)} patches):")
    print(f"  PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
    print(f"    Min: {np.min(psnr_values):.2f}, Max: {np.max(psnr_values):.2f}")
    print(f"  SSIM: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
    print(f"    Min: {np.min(ssim_values):.4f}, Max: {np.max(ssim_values):.4f}")

    # Per-category statistics
    categories = set(m['category'] for m in all_metrics)
    print(f"\nPer-Category Statistics:")
    for cat in sorted(categories):
        cat_metrics = [m for m in all_metrics if m['category'] == cat]
        cat_psnr = [m['psnr'] for m in cat_metrics]
        cat_ssim = [m['ssim'] for m in cat_metrics]
        print(f"  {cat} ({len(cat_metrics)} patches):")
        print(f"    PSNR: {np.mean(cat_psnr):.2f} ± {np.std(cat_psnr):.2f} dB")
        print(f"    SSIM: {np.mean(cat_ssim):.4f} ± {np.std(cat_ssim):.4f}")

    # Best and worst patches
    print(f"\nBest Patches (by PSNR):")
    sorted_by_psnr = sorted(all_metrics, key=lambda x: x['psnr'], reverse=True)
    for i, m in enumerate(sorted_by_psnr[:5]):
        print(f"  {i+1}. Patch {m['patch_id']:04d} ({m['patient_id']}): PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}")

    print(f"\nWorst Patches (by PSNR):")
    for i, m in enumerate(sorted_by_psnr[-5:]):
        print(f"  {i+1}. Patch {m['patch_id']:04d} ({m['patient_id']}): PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}")

    # Save metrics to JSON
    metrics_json_path = output_dirs['metrics'] / 'patch_metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump({
            'config': {
                'checkpoint': args.checkpoint,
                'split': args.split,
                'num_samples': args.num_samples,
                'sampler': args.sampler,
                'num_inference_steps': args.num_inference_steps,
                'patch_config': {
                    'depth_thick': patch_depth_thick,
                    'depth_thin': patch_depth_thin,
                    'spatial_size': patch_size
                }
            },
            'overall': {
                'psnr_mean': float(np.mean(psnr_values)),
                'psnr_std': float(np.std(psnr_values)),
                'psnr_min': float(np.min(psnr_values)),
                'psnr_max': float(np.max(psnr_values)),
                'ssim_mean': float(np.mean(ssim_values)),
                'ssim_std': float(np.std(ssim_values)),
                'ssim_min': float(np.min(ssim_values)),
                'ssim_max': float(np.max(ssim_values)),
            },
            'per_patch': all_metrics
        }, f, indent=2)

    print(f"\nMetrics saved to: {metrics_json_path}")

    # Save metrics to CSV
    metrics_csv_path = output_dirs['metrics'] / 'patch_metrics.csv'
    with open(metrics_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patch_id', 'patient_id', 'category', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"CSV saved to: {metrics_csv_path}")

    # Summary
    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dirs['root']}")
    print(f"  Visualizations: {len(list(output_dirs['visualizations'].glob('*.png')))} images")
    print(f"  Metrics: JSON + CSV")
    if args.save_predictions:
        print(f"  Predictions: {len(list(output_dirs['predictions'].glob('*.pt')))} .pt files")
    print()


if __name__ == '__main__':
    main()
