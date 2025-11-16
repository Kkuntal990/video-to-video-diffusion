#!/usr/bin/env python3
"""
Quick Forward Pass Test

Tests model initialization and forward pass with synthetic data.
"""

import argparse
import yaml
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import VideoToVideoDiffusion


def test_forward_pass(config_path, device='cpu'):
    """Test model initialization and forward pass"""

    print('='*70)
    print('Model Forward Pass Test')
    print('='*70)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f'\nLoading model from config: {config_path}')

    # Create model
    model = VideoToVideoDiffusion(config)
    model = model.to(device)
    model.eval()

    # Model info
    params = model.count_parameters()
    print(f'\nModel Statistics:')
    print(f'  Total parameters: {params["total"]:,}')
    print(f'  VAE parameters: {params["vae"]:,}')
    print(f'  U-Net parameters: {params["unet"]:,}')

    # Create synthetic data
    B, C, T, H, W = 1, 1, 8, 128, 128  # Smaller for CPU testing
    print(f'\nCreating synthetic CT video:')
    print(f'  Batch size: {B}')
    print(f'  Channels: {C} (grayscale)')
    print(f'  Frames: {T}')
    print(f'  Resolution: {H}x{W}')

    v_in = torch.randn(B, C, T, H, W, device=device)
    v_gt = torch.randn(B, C, T, H, W, device=device)

    # Test VAE encode
    print(f'\n[1/4] Testing VAE encode...')
    with torch.no_grad():
        z = model.vae.encode(v_in)
        print(f'  ✓ Encoded shape: {z.shape}')
        print(f'    Spatial compression: {H}x{W} → {z.shape[3]}x{z.shape[4]}')
        print(f'    Latent statistics: mean={z.mean():.4f}, std={z.std():.4f}')

    # Test VAE decode
    print(f'\n[2/4] Testing VAE decode...')
    with torch.no_grad():
        v_recon = model.vae.decode(z)
        print(f'  ✓ Decoded shape: {v_recon.shape}')
        mse = torch.mean((v_in - v_recon) ** 2).item()
        print(f'    Reconstruction MSE: {mse:.6f} (random weights, expected high)')

    # Test diffusion forward pass
    print(f'\n[3/4] Testing diffusion forward pass...')
    with torch.no_grad():
        loss, metrics = model(v_in, v_gt)
        print(f'  ✓ Loss: {loss.item():.6f}')
        print(f'    Metrics: {metrics}')

    # Test generation
    print(f'\n[4/4] Testing generation (DDIM sampling)...')
    with torch.no_grad():
        v_gen = model.generate(v_in, sampler='ddim', num_inference_steps=10)
        print(f'  ✓ Generated shape: {v_gen.shape}')

    print('\n' + '='*70)
    print('✓✓✓ ALL TESTS PASSED ✓✓✓')
    print('='*70)
    print('\nModel is ready for training!')
    print(f'\nConfiguration:')
    print(f'  - Architecture: MAISI-inspired 2D VAE')
    print(f'  - Input: Grayscale CT ({C} channel)')
    print(f'  - VAE: {params["vae"]:,} parameters')
    print(f'  - U-Net: {params["unet"]:,} parameters')
    print(f'  - Latent scaling: {model.vae.scaling_factor}')


def main():
    parser = argparse.ArgumentParser(description='Test model forward pass')
    parser.add_argument(
        '--config',
        type=str,
        default='config/cloud_train_config_a100.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on'
    )

    args = parser.parse_args()

    # Check config exists
    if not Path(args.config).exists():
        print(f"Error: Config not found: {args.config}")
        return

    # Run test
    test_forward_pass(args.config, args.device)


if __name__ == "__main__":
    main()
