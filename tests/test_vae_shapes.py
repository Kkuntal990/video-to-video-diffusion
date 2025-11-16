"""
Standalone test script to verify VAE input/output shapes.
Can run without pytest - just uses Python and PyTorch.

Usage:
    python tests/test_vae_shapes.py
"""

import torch
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.vae import VideoVAE


def load_config():
    """Load VAE training config"""
    config_path = Path(__file__).parent.parent / "config" / "vae_training.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_vae_shapes():
    """Test VAE input/output shape integrity"""

    print("\n" + "="*70)
    print("VAE SHAPE INTEGRITY TESTS")
    print("="*70 + "\n")

    # Load config
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create model
    print("Creating VAE model...")
    vae = VideoVAE(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim'],
        base_channels=config['model']['vae_base_channels'],
        scaling_factor=config['model']['vae_scaling_factor'],
    ).to(device)

    num_params = sum(p.numel() for p in vae.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Test 1: Correct 5D input shape
    print("Test 1: 5D input (B, C, D, H, W)")
    print("-" * 70)
    B, C, D, H, W = 2, 1, 50, 512, 512
    x = torch.randn(B, C, D, H, W, device=device)
    print(f"  Input shape: {x.shape}")

    try:
        vae.eval()
        with torch.no_grad():
            recon, z = vae(x)

        print(f"  Reconstruction shape: {recon.shape}")
        print(f"  Latent shape: {z.shape}")

        # Verify shapes
        assert recon.shape == x.shape, f"Shape mismatch: {recon.shape} != {x.shape}"
        assert z.shape == (B, 4, D, H//8, W//8), f"Latent shape unexpected: {z.shape}"

        # Verify no NaN/Inf
        assert not torch.isnan(recon).any(), "Reconstruction contains NaN"
        assert not torch.isinf(recon).any(), "Reconstruction contains Inf"

        print(f"  Output range: [{recon.min():.3f}, {recon.max():.3f}]")
        print("  ✓ PASSED\n")

    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")
        return False

    # Test 2: Reject 4D input (missing channel dimension)
    print("Test 2: 4D input (B, D, H, W) - should REJECT")
    print("-" * 70)
    B, D, H, W = 2, 50, 512, 512
    x_4d = torch.randn(B, D, H, W, device=device)
    print(f"  Input shape: {x_4d.shape}")

    try:
        with torch.no_grad():
            recon, z = vae(x_4d)
        print(f"  ✗ FAILED: Model accepted 4D input (should reject)\n")
        return False
    except RuntimeError as e:
        if "5D" in str(e) or "Expected" in str(e):
            print(f"  Correctly rejected: {str(e)[:80]}...")
            print("  ✓ PASSED\n")
        else:
            print(f"  ✗ FAILED: Wrong error: {e}\n")
            return False

    # Test 3: Reject 6D input (extra dimension)
    print("Test 3: 6D input (B, C1, C2, D, H, W) - should REJECT")
    print("-" * 70)
    B, C1, C2, D, H, W = 2, 1, 1, 50, 512, 512
    x_6d = torch.randn(B, C1, C2, D, H, W, device=device)
    print(f"  Input shape: {x_6d.shape}")

    try:
        with torch.no_grad():
            recon, z = vae(x_6d)
        print(f"  ✗ FAILED: Model accepted 6D input (should reject)\n")
        return False
    except RuntimeError as e:
        if "5D" in str(e) or "Expected" in str(e):
            print(f"  Correctly rejected: {str(e)[:80]}...")
            print("  ✓ PASSED\n")
        else:
            print(f"  ✗ FAILED: Wrong error: {e}\n")
            return False

    # Test 4: Variable batch sizes
    print("Test 4: Variable batch sizes")
    print("-" * 70)
    C, D, H, W = 1, 20, 256, 256

    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, C, D, H, W, device=device)
        try:
            with torch.no_grad():
                recon, z = vae(x)

            assert recon.shape[0] == batch_size
            assert z.shape[0] == batch_size
            print(f"  Batch size {batch_size}: ✓")
        except Exception as e:
            print(f"  Batch size {batch_size}: ✗ {e}")
            return False

    print("  ✓ PASSED\n")

    # Test 5: Variable depth (slice count)
    print("Test 5: Variable depth (slice count)")
    print("-" * 70)
    B, C, H, W = 2, 1, 256, 256

    for depth in [10, 20, 50, 100]:
        x = torch.randn(B, C, depth, H, W, device=device)
        try:
            with torch.no_grad():
                recon, z = vae(x)

            assert recon.shape[2] == depth, f"Depth mismatch: {recon.shape[2]} != {depth}"
            print(f"  Depth {depth:3d}: ✓ (latent depth: {z.shape[2]})")
        except Exception as e:
            print(f"  Depth {depth}: ✗ {e}")
            return False

    print("  ✓ PASSED\n")

    # Test 6: Encode-decode cycle
    print("Test 6: Encode-decode cycle")
    print("-" * 70)
    B, C, D, H, W = 2, 1, 20, 256, 256
    x = torch.randn(B, C, D, H, W, device=device)

    try:
        with torch.no_grad():
            z = vae.encode(x)
            recon = vae.decode(z)

        print(f"  Input:  {x.shape}")
        print(f"  Latent: {z.shape}")
        print(f"  Recon:  {recon.shape}")

        assert recon.shape == x.shape, "Encode-decode cycle broke shape"
        print("  ✓ PASSED\n")
    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")
        return False

    # Test 7: Deterministic forward pass
    print("Test 7: Deterministic forward pass")
    print("-" * 70)
    B, C, D, H, W = 2, 1, 20, 256, 256
    x = torch.randn(B, C, D, H, W, device=device)

    try:
        vae.eval()
        with torch.no_grad():
            recon1, z1 = vae(x)
            recon2, z2 = vae(x)

        diff_recon = (recon1 - recon2).abs().max().item()
        diff_z = (z1 - z2).abs().max().item()

        print(f"  Max recon difference: {diff_recon:.2e}")
        print(f"  Max latent difference: {diff_z:.2e}")

        assert diff_recon < 1e-5, f"Reconstruction not deterministic: {diff_recon}"
        assert diff_z < 1e-5, f"Latent not deterministic: {diff_z}"

        print("  ✓ PASSED\n")
    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")
        return False

    # Test 8: Mixed precision (if CUDA available)
    if device.type == 'cuda':
        print("Test 8: Mixed precision (BF16/FP16)")
        print("-" * 70)
        B, C, D, H, W = 2, 1, 20, 256, 256
        x = torch.randn(B, C, D, H, W, device=device)

        try:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    recon, z = vae(x)

            print(f"  Reconstruction dtype: {recon.dtype}")
            print(f"  Latent dtype: {z.dtype}")

            assert not torch.isnan(recon).any(), "NaN in mixed precision"
            assert not torch.isinf(recon).any(), "Inf in mixed precision"

            print("  ✓ PASSED\n")
        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")
            return False

    print("="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70 + "\n")

    return True


if __name__ == '__main__':
    success = test_vae_shapes()
    sys.exit(0 if success else 1)
