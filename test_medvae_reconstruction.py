#!/usr/bin/env python3
"""
MedVAE Compatibility Test for CT Slice Interpolation

Tests whether MedVAE 3D model is compatible with our CT preprocessing pipeline.
Runs on multiple samples and generates comprehensive compatibility report.

Usage:
    python test_medvae_reconstruction.py \
        --cache_dir /workspace/storage_a100/.cache/processed \
        --num_samples 10 \
        --output_dir ./medvae_test_results
"""

import argparse
import sys
import math
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Try importing MedVAE, install if needed
try:
    from medvae import MVAE
except ImportError:
    print("MedVAE not found, installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "medvae"], check=False)
    try:
        from medvae import MVAE
    except ImportError:
        print("ERROR: Failed to install MedVAE. Please install manually: pip install medvae")
        sys.exit(1)

# Try importing pytorch-msssim, install if needed
try:
    from pytorch_msssim import ssim
except ImportError:
    print("pytorch-msssim not found, installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pytorch-msssim"], check=False)
    try:
        from pytorch_msssim import ssim
    except ImportError:
        print("ERROR: Failed to install pytorch-msssim. Please install manually: pip install pytorch-msssim")
        sys.exit(1)


def load_preprocessed_sample(pt_file: Path, use_thick: bool = False):
    """
    Load preprocessed CT volume matching repo conventions

    Args:
        pt_file: Path to .pt file
        use_thick: If True, use thick slices; if False, use thin slices

    Returns:
        x: Tensor of shape (1, D, 512, 512) in range [-1, 1]
        case_id: Patient/case identifier
    """
    sample = torch.load(pt_file, weights_only=False)

    # Use thick or thin slices
    key = 'input' if use_thick else 'target'
    x = sample[key]  # Shape: (1, D, 512, 512), range [-1, 1]

    case_id = sample.get('patient_id', pt_file.stem)
    num_slices = sample.get('num_thick_slices' if use_thick else 'num_thin_slices', x.shape[1])

    return x, case_id, num_slices


def compute_metrics(x_original, x_reconstructed):
    """
    Compute reconstruction metrics

    Args:
        x_original: Original tensor (B, C, D, H, W)
        x_reconstructed: Reconstructed tensor (B, C, D, H, W)

    Returns:
        dict: Metrics including MAE, MSE, PSNR, SSIM
    """
    # MAE
    mae = (x_original - x_reconstructed).abs().mean().item()

    # MSE
    mse = ((x_original - x_reconstructed) ** 2).mean().item()

    # PSNR (data range is 2.0 for [-1, 1])
    if mse > 1e-10:
        psnr = -10 * math.log10(mse)
    else:
        psnr = 100.0  # Perfect reconstruction

    # SSIM (for 3D volumes, compute on middle slices to avoid edge effects)
    # Extract middle 16 slices for SSIM computation (or fewer if volume is small)
    D = x_original.shape[2]
    if D >= 32:
        mid_idx = D // 2
        slice_range = 16
        x_mid = x_original[:, :, mid_idx-slice_range//2:mid_idx+slice_range//2, :, :]
        xr_mid = x_reconstructed[:, :, mid_idx-slice_range//2:mid_idx+slice_range//2, :, :]
    else:
        # Use all slices if volume is small
        x_mid = x_original
        xr_mid = x_reconstructed

    ssim_val = ssim(x_mid, xr_mid, data_range=2.0, size_average=True).item()

    return {
        'mae': mae,
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim_val
    }


def test_medvae(vae, x, normalization='direct'):
    """
    Test MedVAE with specified normalization

    Args:
        vae: MedVAE model
        x: Input tensor (1, D, 512, 512) in range [-1, 1]
        normalization: 'direct' (keep [-1,1]) or 'to_01' (convert to [0,1])

    Returns:
        x_rec: Reconstructed tensor in range [-1, 1]
        z: Latent representation
    """
    # Add channel dimension: (1, D, 512, 512) -> (1, 1, D, 512, 512)
    if x.dim() == 4:
        x = x.unsqueeze(1)

    # Apply normalization
    if normalization == 'to_01':
        # Convert [-1, 1] -> [0, 1]
        x_input = (x + 1.0) / 2.0
    else:
        # Keep [-1, 1]
        x_input = x

    with torch.no_grad():
        # Encode
        z = vae.encode(x_input)

        # Decode
        x_rec = vae.decode(z)

        # Convert back to [-1, 1] if needed
        if normalization == 'to_01':
            x_rec = x_rec * 2.0 - 1.0

    return x_rec, z


def save_slice_comparison(x_orig, x_rec, slice_idx, output_path):
    """
    Save side-by-side comparison of original vs reconstructed slice

    Args:
        x_orig: Original tensor (1, 1, D, H, W)
        x_rec: Reconstructed tensor (1, 1, D, H, W)
        slice_idx: Index of slice to visualize
        output_path: Path to save PNG
    """
    # Extract slices: (1, 1, D, H, W) -> (H, W)
    orig_slice = x_orig[0, 0, slice_idx].cpu().numpy()
    rec_slice = x_rec[0, 0, slice_idx].cpu().numpy()

    # Convert [-1, 1] -> [0, 255]
    orig_img = ((orig_slice + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    rec_img = ((rec_slice + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

    # Compute difference map (absolute error)
    diff = np.abs(orig_slice - rec_slice)
    diff_img = (diff / 2.0 * 255).clip(0, 255).astype(np.uint8)  # Scale to [0, 255]

    # Create side-by-side image: Original | Reconstruction | Difference
    combined = np.hstack([orig_img, rec_img, diff_img])
    Image.fromarray(combined).save(output_path)


def main():
    parser = argparse.ArgumentParser(description='MedVAE Compatibility Test')
    parser.add_argument('--cache_dir', type=str,
                       default='/workspace/storage_a100/.cache/processed',
                       help='Preprocessed cache directory')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to test')
    parser.add_argument('--output_dir', type=str, default='./medvae_test_results',
                       help='Output directory')
    parser.add_argument('--use_thick', action='store_true',
                       help='Test on thick slices instead of thin')
    parser.add_argument('--model_name', type=str, default='medvae_4x_1c_3d',
                       help='MedVAE model name')
    args = parser.parse_args()

    # Setup
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MedVAE Compatibility Test for CT Slice Interpolation")
    print("=" * 60)
    print(f"Cache directory: {cache_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Slice type: {'thick' if args.use_thick else 'thin'}")
    print(f"Output directory: {output_dir}")
    print()

    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU (will be slow)")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print()

    # Load MedVAE model
    print(f"Loading MedVAE model: {args.model_name} (modality=ct)")
    try:
        vae = MVAE(model_name=args.model_name, modality="ct")
        vae = vae.eval().to(device)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"ERROR: Failed to load MedVAE model: {e}")
        print("\nAvailable models: Try 'medvae_4x_1c_3d' or check MedVAE documentation")
        sys.exit(1)

    # Get sample files
    pt_files = sorted(cache_dir.glob("case_*.pt"))
    if len(pt_files) == 0:
        print(f"ERROR: No .pt files found in {cache_dir}")
        sys.exit(1)

    pt_files = pt_files[:args.num_samples]
    print(f"Found {len(pt_files)} samples to test")
    print()

    # Test both normalization approaches
    results = {'direct': [], 'to_01': []}

    for idx, pt_file in enumerate(tqdm(pt_files, desc="Testing samples")):
        try:
            # Load sample
            x, case_id, num_slices = load_preprocessed_sample(pt_file, args.use_thick)
            x = x.to(device)

            if idx == 0:
                print(f"\nFirst sample info:")
                print(f"  Case ID: {case_id}")
                print(f"  Shape: {x.shape}")
                print(f"  Num slices: {num_slices}")
                print(f"  Value range: [{x.min():.3f}, {x.max():.3f}]")
                print(f"  Data type: {x.dtype}")
                print()

            # Test both normalizations
            for norm_type in ['direct', 'to_01']:
                x_rec, z = test_medvae(vae, x, normalization=norm_type)
                metrics = compute_metrics(x.unsqueeze(1), x_rec)  # Add channel dim for metrics
                metrics['case_id'] = case_id
                metrics['num_slices'] = num_slices
                metrics['latent_shape'] = tuple(z.shape)
                metrics['compression_ratio'] = np.prod(x.shape) / np.prod(z.shape)
                results[norm_type].append(metrics)

            # Save visualization for first sample
            if idx == 0:
                mid_idx = x.shape[1] // 2  # Middle slice
                for norm in ['direct', 'to_01']:
                    x_rec, _ = test_medvae(vae, x, normalization=norm)
                    save_slice_comparison(
                        x.unsqueeze(1), x_rec, mid_idx,
                        output_dir / f"comparison_{norm}_slice_{mid_idx}.png"
                    )
                print(f"✓ Saved visualizations for case {case_id}")

        except Exception as e:
            print(f"\nWARNING: Failed to process {pt_file.name}: {e}")
            continue

    # Aggregate results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    summary = {}

    for norm_type in ['direct', 'to_01']:
        if len(results[norm_type]) == 0:
            print(f"\nNo results for normalization: {norm_type}")
            continue

        print(f"\n--- Normalization: {norm_type} ({'[-1, 1]' if norm_type == 'direct' else '[0, 1]'}) ---")

        mae_vals = [r['mae'] for r in results[norm_type]]
        mse_vals = [r['mse'] for r in results[norm_type]]
        psnr_vals = [r['psnr'] for r in results[norm_type]]
        ssim_vals = [r['ssim'] for r in results[norm_type]]

        avg_mae = np.mean(mae_vals)
        avg_mse = np.mean(mse_vals)
        avg_psnr = np.mean(psnr_vals)
        avg_ssim = np.mean(ssim_vals)
        std_psnr = np.std(psnr_vals)

        print(f"  Samples processed: {len(results[norm_type])}")
        print(f"  Average MAE:  {avg_mae:.4f} (±{np.std(mae_vals):.4f})")
        print(f"  Average MSE:  {avg_mse:.6f}")
        print(f"  Average PSNR: {avg_psnr:.2f} dB (±{std_psnr:.2f})")
        print(f"    Min PSNR: {np.min(psnr_vals):.2f} dB")
        print(f"    Max PSNR: {np.max(psnr_vals):.2f} dB")
        print(f"  Average SSIM: {avg_ssim:.4f} (±{np.std(ssim_vals):.4f})")

        if len(results[norm_type]) > 0:
            print(f"  Latent shape: {results[norm_type][0]['latent_shape']}")
            print(f"  Compression ratio: {results[norm_type][0]['compression_ratio']:.1f}x")

        # Verdict
        if avg_mae < 0.02:
            verdict = "✅ Excellent match - MedVAE is compatible"
            verdict_code = "excellent"
        elif avg_mae < 0.06:
            verdict = "⚠️  OK but may need finetuning"
            verdict_code = "ok"
        else:
            verdict = "❌ Not compatible with MedVAE"
            verdict_code = "incompatible"

        print(f"\n  Verdict: {verdict}")

        summary[norm_type] = {
            'num_samples': len(results[norm_type]),
            'avg_mae': avg_mae,
            'avg_mse': avg_mse,
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'std_psnr': std_psnr,
            'min_psnr': np.min(psnr_vals),
            'max_psnr': np.max(psnr_vals),
            'verdict': verdict,
            'verdict_code': verdict_code
        }

    # Determine best approach
    if len(results['direct']) > 0 and len(results['to_01']) > 0:
        best = 'direct' if np.mean([r['mae'] for r in results['direct']]) < \
                           np.mean([r['mae'] for r in results['to_01']]) else 'to_01'

        print(f"\n{'=' * 60}")
        print(f"🎯 Best approach: {best} ({'[-1, 1]' if best == 'direct' else '[0, 1]'})")
        print(f"   Use this normalization for MedVAE with your pipeline")
        print(f"{'=' * 60}")

        summary['best_approach'] = best

    # Save results
    results_file = output_dir / "summary_report.txt"
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MedVAE Compatibility Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Samples tested: {len(pt_files)}\n")
        f.write(f"Slice type: {'thick' if args.use_thick else 'thin'}\n\n")

        for norm_type, metrics in summary.items():
            if norm_type == 'best_approach':
                continue
            f.write(f"\n--- {norm_type} normalization ---\n")
            f.write(f"Average MAE:  {metrics['avg_mae']:.4f}\n")
            f.write(f"Average PSNR: {metrics['avg_psnr']:.2f} dB (±{metrics['std_psnr']:.2f})\n")
            f.write(f"Average SSIM: {metrics['avg_ssim']:.4f}\n")
            f.write(f"Verdict: {metrics['verdict']}\n")

        if 'best_approach' in summary:
            f.write(f"\nBest approach: {summary['best_approach']}\n")

    # Save JSON results
    json_file = output_dir / "metrics.json"
    with open(json_file, 'w') as f:
        json.dump({
            'summary': summary,
            'detailed_results': results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - {results_file.name}")
    print(f"  - {json_file.name}")
    print(f"  - comparison_*.png (visualizations)")

    print("\nTest completed successfully!")


if __name__ == '__main__':
    main()
