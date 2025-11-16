"""
Simple VAE compatibility test - verifies VAE works with main diffusion model.
Lightweight test that can run locally without heavy dependencies.

Usage:
    python tests/test_vae_compatibility.py
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.vae import VideoVAE


def test_vae_architecture():
    """Test VAE architecture matches what main model expects"""

    print("\n" + "="*70)
    print("VAE COMPATIBILITY TEST")
    print("Testing VAE architecture for diffusion model integration")
    print("="*70 + "\n")

    device = torch.device('cpu')  # Use CPU for local testing
    print(f"Device: {device}\n")

    # Configuration matching vae_training.yaml
    config = {
        'in_channels': 1,          # Grayscale CT
        'latent_dim': 4,           # 4 latent channels
        'base_channels': 64,       # Base channels
        'scaling_factor': 0.18215  # Standard VAE scaling
    }

    print("Creating VAE with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Create VAE (same as train_vae.py)
    try:
        vae = VideoVAE(**config).to(device)
        num_params = sum(p.numel() for p in vae.parameters())
        print(f"✓ VAE created successfully")
        print(f"  Parameters: {num_params:,}\n")
    except Exception as e:
        print(f"✗ FAILED to create VAE: {e}\n")
        return False

    # Test 1: Basic encode-decode (what main model uses)
    print("Test 1: Encode-Decode Pipeline")
    print("-" * 70)

    B, C, D, H, W = 1, 1, 20, 128, 128  # Small size for local testing
    print(f"  Input shape: ({B}, {C}, {D}, {H}, {W})")

    try:
        vae.eval()
        with torch.no_grad():
            # Create sample input (normalized CT data range)
            v_in = torch.randn(B, C, D, H, W, device=device) * 0.5  # [-1, 1] range

            # Test encode (what diffusion model calls)
            z = vae.encode(v_in)
            print(f"  Latent shape: {z.shape}")

            # Verify latent properties
            expected_latent_depth = D  # No temporal compression in VideoVAE
            expected_latent_spatial = H // 8  # 8× spatial compression
            expected_shape = (B, 4, expected_latent_depth, expected_latent_spatial, expected_latent_spatial)

            if z.shape != expected_shape:
                print(f"  ✗ FAILED: Expected latent shape {expected_shape}, got {z.shape}")
                return False

            # Test decode (what diffusion model calls)
            v_out = vae.decode(z)
            print(f"  Output shape: {v_out.shape}")

            # Verify reconstruction shape
            if v_out.shape != v_in.shape:
                print(f"  ✗ FAILED: Expected output shape {v_in.shape}, got {v_out.shape}")
                return False

            # Check for NaN/Inf
            if torch.isnan(z).any() or torch.isinf(z).any():
                print(f"  ✗ FAILED: Latent contains NaN or Inf")
                return False

            if torch.isnan(v_out).any() or torch.isinf(v_out).any():
                print(f"  ✗ FAILED: Output contains NaN or Inf")
                return False

            print(f"  Latent range: [{z.min():.3f}, {z.max():.3f}]")
            print(f"  Output range: [{v_out.min():.3f}, {v_out.max():.3f}]")
            print("  ✓ PASSED\n")

    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")
        return False

    # Test 2: Slice interpolation scenario (thick -> thin)
    print("Test 2: Slice Interpolation Scenario")
    print("-" * 70)

    try:
        # Thick slices (input)
        B, C, D_thick, H, W = 1, 1, 10, 128, 128
        v_thick = torch.randn(B, C, D_thick, H, W, device=device) * 0.5
        print(f"  Thick slices: {v_thick.shape}")

        # Encode thick slices
        z_thick = vae.encode(v_thick)
        print(f"  Thick latent: {z_thick.shape}")

        # Upsample latent to thin depth (what main model does)
        D_thin = 60  # 6× interpolation
        D_latent_thin = D_thin  # VideoVAE has no temporal compression

        import torch.nn.functional as F
        z_thin = F.interpolate(
            z_thick,
            size=(D_latent_thin, z_thick.shape[3], z_thick.shape[4]),
            mode='trilinear',
            align_corners=False
        )
        print(f"  Upsampled latent: {z_thin.shape}")

        # Decode to thin slices
        v_thin = vae.decode(z_thin)
        print(f"  Thin slices: {v_thin.shape}")

        # Verify thin depth
        if v_thin.shape[2] != D_thin:
            print(f"  ✗ FAILED: Expected depth {D_thin}, got {v_thin.shape[2]}")
            return False

        print("  ✓ PASSED\n")

    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")
        return False

    # Test 3: Integration with main model config
    print("Test 3: Main Model Config Compatibility")
    print("-" * 70)

    try:
        # Simulate main model initialization
        main_model_config = {
            'model': {
                'in_channels': 1,
                'latent_dim': 4,
                'vae_base_channels': 64,
                'vae_scaling_factor': 0.18215
            }
        }

        # Extract config (same as model.py lines 88-92)
        model_config = main_model_config.get('model', main_model_config)
        in_channels = model_config.get('in_channels', 3)
        base_channels = model_config.get('vae_base_channels', 64)
        latent_dim = model_config.get('latent_dim', 4)
        scaling_factor = model_config.get('vae_scaling_factor', 0.18215)

        # Create VAE same way as model.py
        vae_main = VideoVAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            scaling_factor=scaling_factor,
        ).to(device)

        print(f"  ✓ VAE created with main model config")
        print(f"    in_channels: {in_channels}")
        print(f"    latent_dim: {latent_dim}")
        print(f"    base_channels: {base_channels}")
        print(f"    scaling_factor: {scaling_factor}")

        # Test forward pass
        v_test = torch.randn(1, 1, 10, 128, 128, device=device)
        with torch.no_grad():
            z_test = vae_main.encode(v_test)
            v_recon = vae_main.decode(z_test)

        print(f"  ✓ Forward pass successful")
        print("  ✓ PASSED\n")

    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")
        return False

    # Test 4: Batch processing
    print("Test 4: Batch Processing")
    print("-" * 70)

    try:
        for batch_size in [1, 2]:
            v_batch = torch.randn(batch_size, 1, 10, 128, 128, device=device)
            with torch.no_grad():
                z_batch = vae.encode(v_batch)
                v_recon = vae.decode(z_batch)

            if v_recon.shape != v_batch.shape:
                print(f"  ✗ FAILED: Batch size {batch_size} shape mismatch")
                return False

            print(f"  Batch size {batch_size}: ✓")

        print("  ✓ PASSED\n")

    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")
        return False

    # Test 5: Save and load checkpoint (for VAE training)
    print("Test 5: Checkpoint Save/Load")
    print("-" * 70)

    try:
        import tempfile
        import os

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "vae_test.pt")

            checkpoint = {
                'model_state_dict': vae.state_dict(),
                'config': config
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Checkpoint saved")

            # Load checkpoint
            checkpoint_loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            vae_loaded = VideoVAE(**checkpoint_loaded['config']).to(device)
            vae_loaded.load_state_dict(checkpoint_loaded['model_state_dict'])
            print(f"  ✓ Checkpoint loaded")

            # Test loaded model
            v_test = torch.randn(1, 1, 10, 128, 128, device=device)
            with torch.no_grad():
                z_orig = vae.encode(v_test)
                z_loaded = vae_loaded.encode(v_test)

            # Should be identical
            if not torch.allclose(z_orig, z_loaded, atol=1e-6):
                print(f"  ✗ FAILED: Loaded model produces different outputs")
                return False

            print("  ✓ PASSED\n")

    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")
        return False

    # Summary
    print("="*70)
    print("ALL COMPATIBILITY TESTS PASSED ✓")
    print("="*70)
    print("\nVAE Architecture Summary:")
    print(f"  - Input: (B, 1, D, 512, 512) - grayscale CT slices")
    print(f"  - Latent: (B, 4, D, 64, 64) - 8× spatial compression, no temporal")
    print(f"  - Output: (B, 1, D, 512, 512) - reconstructed slices")
    print(f"  - Parameters: {num_params:,}")
    print(f"\nCompatible with main diffusion model: YES ✓")
    print(f"Ready for VAE training: YES ✓")
    print()

    return True


if __name__ == '__main__':
    success = test_vae_architecture()
    sys.exit(0 if success else 1)
