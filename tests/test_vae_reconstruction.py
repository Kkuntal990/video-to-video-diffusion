#!/usr/bin/env python3
"""
Test MAISI VAE Reconstruction Quality

This script validates that the pretrained MAISI VAE is working correctly by:
1. Loading preprocessed CT data
2. Encoding to latent space
3. Decoding back to image space
4. Comparing reconstruction to original (PSNR/SSIM)
5. Visualizing results

Expected results:
- PSNR > 35 dB (high quality reconstruction)
- SSIM > 0.95 (very similar structure)
- No NaN in latents or reconstructions
- Latent range ≈[-5, +5] (MAISI VAE expected range)

Usage:
    python tests/test_vae_reconstruction.py
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import VideoToVideoDiffusion
from utils.metrics import calculate_psnr, calculate_ssim


def test_single_slice(model, sample_data, slice_idx=149):
    """
    Test VAE reconstruction on a single CT slice

    Args:
        model: VideoToVideoDiffusion with pretrained VAE
        sample_data: Preprocessed patient data
        slice_idx: Which slice to test (default: middle)

    Returns:
        dict with metrics and tensors
    """
    print(f"\n{'='*70}")
    print(f"TEST 1: Single Slice Reconstruction (Slice {slice_idx})")
    print(f"{'='*70}")

    # Get 4 consecutive slices (MAISI VAE has 4:1 depth compression)
    # So we need 4 input slices to get 4 output slices for proper comparison
    thin_volume = sample_data['target']  # (1, 300, 512, 512)
    start_idx = slice_idx
    end_idx = min(slice_idx + 4, thin_volume.shape[1])
    test_slices = thin_volume[:, start_idx:end_idx, :, :]  # (1, 4, 512, 512)

    # Add channel dimension for VAE: (B, D, H, W) -> (B, C, D, H, W)
    test_slices = test_slices.unsqueeze(1)  # (1, 1, 4, 512, 512)

    print(f"\nInput slices:")
    print(f"  Shape: {test_slices.shape}")
    print(f"  Range: [{test_slices.min():.3f}, {test_slices.max():.3f}]")
    print(f"  Mean: {test_slices.mean():.3f}")
    print(f"  Std: {test_slices.std():.3f}")

    # VAE encode
    with torch.no_grad():
        latent = model.vae.encode(test_slices)

    print(f"\nLatent representation:")
    print(f"  Shape: {latent.shape}")
    print(f"  Range: [{latent.min():.3f}, {latent.max():.3f}]")
    print(f"  Mean: {latent.mean():.3f}")
    print(f"  Std: {latent.std():.3f}")

    # Check for NaN in latent
    if torch.isnan(latent).any():
        nan_count = torch.isnan(latent).sum().item()
        print(f"  ❌ WARNING: Latent contains {nan_count} NaN values!")
    else:
        print(f"  ✓ No NaN in latent")

    # VAE decode
    with torch.no_grad():
        reconstructed = model.vae.decode(latent)

    print(f"\nReconstructed slice:")
    print(f"  Shape: {reconstructed.shape}")
    print(f"  Range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    print(f"  Mean: {reconstructed.mean():.3f}")
    print(f"  Std: {reconstructed.std():.3f}")

    # Check for NaN in reconstruction
    if torch.isnan(reconstructed).any():
        nan_count = torch.isnan(reconstructed).sum().item()
        print(f"  ❌ WARNING: Reconstruction contains {nan_count} NaN values!")
    else:
        print(f"  ✓ No NaN in reconstruction")

    # Squeeze depth dimension for metrics (5D -> 4D): (B, C, D, H, W) -> (B, C*D, H, W)
    # Flatten depth into channels for comparison
    rec_squeezed = reconstructed.reshape(reconstructed.shape[0], -1, reconstructed.shape[3], reconstructed.shape[4])
    inp_squeezed = test_slices.reshape(test_slices.shape[0], -1, test_slices.shape[3], test_slices.shape[4])

    # Compute metrics
    psnr = calculate_psnr(rec_squeezed, inp_squeezed, max_val=2.0)  # Range is [-1, 1], so max_val=2.0
    ssim = calculate_ssim(rec_squeezed, inp_squeezed, max_val=2.0)

    print(f"\nReconstruction Quality:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")

    # Assess quality
    if psnr > 35 and ssim > 0.95:
        print(f"  ✅ EXCELLENT quality (VAE working correctly)")
    elif psnr > 30 and ssim > 0.90:
        print(f"  ⚠️  GOOD quality (minor issues)")
    elif psnr > 25 and ssim > 0.80:
        print(f"  ⚠️  FAIR quality (check VAE configuration)")
    else:
        print(f"  ❌ POOR quality (VAE may have issues)")

    return {
        'original': test_slices,
        'latent': latent,
        'reconstructed': reconstructed,
        'psnr': psnr,
        'ssim': ssim
    }


def test_full_volume(model, sample_data):
    """
    Test VAE reconstruction on full CT volume

    Args:
        model: VideoToVideoDiffusion with pretrained VAE
        sample_data: Preprocessed patient data

    Returns:
        dict with metrics
    """
    print(f"\n{'='*70}")
    print(f"TEST 2: Full Volume Reconstruction (All {sample_data['num_thin_slices']} Slices)")
    print(f"{'='*70}")

    thin_volume = sample_data['target']  # (1, 300, 512, 512)

    # Add channel dimension for VAE: (B, D, H, W) -> (B, C, D, H, W)
    thin_volume = thin_volume.unsqueeze(1)  # (1, 1, 300, 512, 512)

    print(f"\nInput volume:")
    print(f"  Shape: {thin_volume.shape}")
    print(f"  Range: [{thin_volume.min():.3f}, {thin_volume.max():.3f}]")

    # VAE encode (may use chunking for large volumes)
    with torch.no_grad():
        latent = model.vae.encode(thin_volume)

    print(f"\nLatent volume:")
    print(f"  Shape: {latent.shape}")
    print(f"  Range: [{latent.min():.3f}, {latent.max():.3f}]")
    print(f"  Compression: {thin_volume.shape[1]} slices → {latent.shape[2]} latent slices (depth)")
    print(f"  Compression: 512×512 → {latent.shape[3]}×{latent.shape[4]} (spatial)")

    # Check for NaN
    if torch.isnan(latent).any():
        nan_count = torch.isnan(latent).sum().item()
        print(f"  ❌ WARNING: Latent contains {nan_count} NaN values!")
        return {'psnr': 0.0, 'ssim': 0.0, 'nan_detected': True}
    else:
        print(f"  ✓ No NaN in latent")

    # VAE decode
    with torch.no_grad():
        reconstructed = model.vae.decode(latent)

    print(f"\nReconstructed volume:")
    print(f"  Shape: {reconstructed.shape}")
    print(f"  Range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

    # Check for NaN
    if torch.isnan(reconstructed).any():
        nan_count = torch.isnan(reconstructed).sum().item()
        print(f"  ❌ WARNING: Reconstruction contains {nan_count} NaN values!")
        return {'psnr': 0.0, 'ssim': 0.0, 'nan_detected': True}
    else:
        print(f"  ✓ No NaN in reconstruction")

    # Compute metrics on full volume
    # Flatten 5D tensors to 4D for metrics: (B, C, D, H, W) -> (B, C*D, H, W)
    rec_flat = reconstructed.reshape(reconstructed.shape[0], -1, reconstructed.shape[3], reconstructed.shape[4])
    inp_flat = thin_volume.reshape(thin_volume.shape[0], -1, thin_volume.shape[3], thin_volume.shape[4])

    psnr = calculate_psnr(rec_flat, inp_flat, max_val=2.0)
    ssim = calculate_ssim(rec_flat, inp_flat, max_val=2.0)

    print(f"\nFull Volume Reconstruction Quality:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")

    if psnr > 35 and ssim > 0.95:
        print(f"  ✅ EXCELLENT (VAE handles full volumes correctly)")
    elif psnr > 30 and ssim > 0.90:
        print(f"  ⚠️  GOOD (minor chunking artifacts possible)")
    else:
        print(f"  ❌ POOR (check chunking logic or VAE configuration)")

    return {
        'psnr': psnr,
        'ssim': ssim,
        'nan_detected': False
    }


def test_multiple_patients(model, processed_dir, num_samples=5):
    """
    Test VAE on multiple patients for consistency

    Args:
        model: VideoToVideoDiffusion with pretrained VAE
        processed_dir: Directory with preprocessed .pt files
        num_samples: How many patients to test

    Returns:
        list of results per patient
    """
    print(f"\n{'='*70}")
    print(f"TEST 3: Multiple Patient Consistency ({num_samples} patients)")
    print(f"{'='*70}")

    # Get device from model
    device = next(model.parameters()).device

    processed_dir = Path(processed_dir)
    pt_files = sorted(list(processed_dir.glob('*.pt')))[:num_samples]

    if len(pt_files) == 0:
        print(f"❌ No .pt files found in {processed_dir}")
        return []

    results = []

    for i, pt_file in enumerate(pt_files):
        print(f"\n--- Patient {i+1}/{num_samples}: {pt_file.name} ---")

        # Load data
        sample = torch.load(pt_file, weights_only=False)

        # Test 4 middle slices (MAISI VAE requires 4:1 compression)
        thin_volume = sample['target']
        mid_idx = thin_volume.shape[1] // 2
        test_slices = thin_volume[:, mid_idx:mid_idx+4, :, :]  # (1, 4, 512, 512)

        # Add channel dimension: (B, D, H, W) -> (B, C, D, H, W)
        test_slices = test_slices.unsqueeze(1).to(device)  # (1, 1, 4, 512, 512)

        # Encode → Decode
        with torch.no_grad():
            latent = model.vae.encode(test_slices)
            reconstructed = model.vae.decode(latent)

        # Flatten for metrics
        rec_flat = reconstructed.reshape(reconstructed.shape[0], -1, reconstructed.shape[3], reconstructed.shape[4])
        inp_flat = test_slices.reshape(test_slices.shape[0], -1, test_slices.shape[3], test_slices.shape[4])

        # Metrics
        psnr = calculate_psnr(rec_flat, inp_flat, max_val=2.0)
        ssim = calculate_ssim(rec_flat, inp_flat, max_val=2.0)

        # Check NaN
        has_nan = torch.isnan(latent).any() or torch.isnan(reconstructed).any()

        print(f"  PSNR: {psnr:.1f} dB, SSIM: {ssim:.3f}, NaN: {has_nan}")

        results.append({
            'patient_id': sample['patient_id'],
            'category': sample['category'],
            'psnr': psnr,
            'ssim': ssim,
            'has_nan': has_nan
        })

    # Summary statistics
    print(f"\n{'='*70}")
    print(f"Summary Statistics:")
    print(f"{'='*70}")
    psnrs = [r['psnr'] for r in results if not r['has_nan']]
    ssims = [r['ssim'] for r in results if not r['has_nan']]
    nan_count = sum(r['has_nan'] for r in results)

    if len(psnrs) > 0:
        print(f"  PSNR: mean={np.mean(psnrs):.1f}, std={np.std(psnrs):.1f}, range=[{np.min(psnrs):.1f}, {np.max(psnrs):.1f}]")
        print(f"  SSIM: mean={np.mean(ssims):.3f}, std={np.std(ssims):.3f}, range=[{np.min(ssims):.3f}, {np.max(ssims):.3f}]")
        print(f"  NaN count: {nan_count}/{len(results)}")

        if np.mean(psnrs) > 35 and np.mean(ssims) > 0.95 and nan_count == 0:
            print(f"\n  ✅ VAE is CONSISTENT across all patients")
        elif nan_count > 0:
            print(f"\n  ❌ NaN detected in {nan_count} patients - VAE has issues!")
        else:
            print(f"\n  ⚠️  VAE quality varies - check configuration")
    else:
        print(f"  ❌ All samples had NaN - VAE is broken!")

    return results


def visualize_reconstruction(results, save_path='vae_reconstruction_test.png'):
    """
    Create visualization comparing original, latent, and reconstruction

    Args:
        results: Dict from test_single_slice()
        save_path: Where to save visualization
    """
    print(f"\n{'='*70}")
    print(f"Creating Visualization")
    print(f"{'='*70}")

    # Extract middle slice from the 4-slice volume for visualization
    # original: (B, C, D, H, W) -> take middle slice
    original = results['original'][0, 0, results['original'].shape[2]//2].cpu().numpy()  # (512, 512)
    latent = results['latent'][0, 0, 0].cpu().numpy()  # (128, 128) - first latent slice, first channel
    reconstructed = results['reconstructed'][0, 0, results['reconstructed'].shape[2]//2].cpu().numpy()  # (512, 512)

    # Denormalize from [-1, 1] to [0, 1] for visualization
    original_vis = (original + 1.0) / 2.0
    reconstructed_vis = (reconstructed + 1.0) / 2.0

    # Compute difference
    diff = np.abs(original - reconstructed)
    diff_vis = diff / diff.max()  # Normalize difference

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original
    axes[0, 0].imshow(original_vis, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original CT Slice\n(Ground Truth)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Latent (first channel)
    axes[0, 1].imshow(latent, cmap='viridis')
    axes[0, 1].set_title(f'VAE Latent (Channel 0)\nRange: [{results["latent"].min():.2f}, {results["latent"].max():.2f}]',
                        fontsize=12)
    axes[0, 1].axis('off')

    # Reconstructed
    axes[1, 0].imshow(reconstructed_vis, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'VAE Reconstruction\nPSNR: {results["psnr"]:.1f} dB, SSIM: {results["ssim"]:.3f}',
                        fontsize=12, fontweight='bold',
                        color='green' if results['psnr'] > 35 else 'orange')
    axes[1, 0].axis('off')

    # Difference map
    im = axes[1, 1].imshow(diff_vis, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Absolute Difference\n(Reconstruction Error)', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    plt.suptitle('MAISI VAE Reconstruction Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {save_path}")


def main():
    print("=" * 70)
    print("MAISI VAE RECONSTRUCTION QUALITY TEST")
    print("=" * 70)
    print("\nThis test validates that the pretrained MAISI VAE can:")
    print("  1. Encode CT images to latent space")
    print("  2. Decode latents back to images")
    print("  3. Produce high-quality reconstructions (PSNR > 35 dB)")
    print("  4. Handle full volumes without NaN")
    print("  5. Work consistently across different patients")

    # Load config
    config_path = Path('config/slice_interpolation_full_medium.yaml')
    print(f"\nLoading config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model with pretrained VAE
    print(f"\nLoading model with pretrained MAISI VAE...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VideoToVideoDiffusion(config, load_pretrained=True).to(device)
    model.eval()

    print(f"✓ Model loaded on {device}")
    print(f"  VAE type: {'Custom MAISI' if hasattr(model.vae, 'maisi_vae') else 'Standard'}")

    # Load test data
    # Check if running on cluster or locally
    cluster_dir = Path('/workspace/storage_a100/.cache/processed')
    local_dir = Path('/tmp/processed_cache')

    if cluster_dir.exists():
        processed_dir = cluster_dir
        print(f"Running on cluster, using: {processed_dir}")
    else:
        processed_dir = local_dir
        print(f"Running locally, using: {processed_dir}")

    test_file = processed_dir / 'case_115.pt'

    if not test_file.exists():
        print(f"\n⚠️  Test file not found: {test_file}")
        print(f"Please ensure preprocessed data is available.")
        return

    print(f"\nLoading test data: {test_file}")
    sample_data = torch.load(test_file, weights_only=False)
    print(f"✓ Loaded patient: {sample_data['patient_id']} ({sample_data['category']})")

    # Move data to device
    sample_data['input'] = sample_data['input'].to(device)
    sample_data['target'] = sample_data['target'].to(device)

    # Run tests
    test1_results = test_single_slice(model, sample_data, slice_idx=149)
    test2_results = test_full_volume(model, sample_data)

    # Visualize
    visualize_reconstruction(test1_results, save_path='/tmp/vae_reconstruction_test.png')

    # Test multiple patients if available
    if processed_dir.exists() and len(list(processed_dir.glob('*.pt'))) > 1:
        test3_results = test_multiple_patients(model, processed_dir, num_samples=5)
    else:
        print(f"\n⚠️  Skipping multi-patient test (only one .pt file available)")
        print(f"To test more patients, copy them to: {processed_dir}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL VERDICT")
    print(f"{'='*70}")

    single_ok = test1_results['psnr'] > 35 and test1_results['ssim'] > 0.95
    volume_ok = test2_results['psnr'] > 35 and test2_results['ssim'] > 0.95
    no_nan = not test2_results['nan_detected']

    if single_ok and volume_ok and no_nan:
        print(f"✅ VAE IS WORKING CORRECTLY!")
        print(f"   - Single slice reconstruction: EXCELLENT")
        print(f"   - Full volume reconstruction: EXCELLENT")
        print(f"   - No NaN detected")
        print(f"\n✅ Ready to proceed with DDIM sampler fixes")
    elif no_nan:
        print(f"⚠️  VAE HAS MINOR ISSUES")
        print(f"   - Single slice PSNR: {test1_results['psnr']:.1f} dB (target: >35)")
        print(f"   - Single slice SSIM: {test1_results['ssim']:.3f} (target: >0.95)")
        print(f"   - Full volume PSNR: {test2_results['psnr']:.1f} dB (target: >35)")
        print(f"   - Check VAE configuration before proceeding")
    else:
        print(f"❌ VAE HAS CRITICAL ISSUES!")
        print(f"   - NaN detected in latents or reconstructions")
        print(f"   - DO NOT proceed with DDIM fixes until VAE is fixed")
        print(f"   - Check VAE checkpoint loading and configuration")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
