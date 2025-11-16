"""
Test Gradient Checkpointing Implementation

Verifies that:
1. Gradient checkpointing produces identical results to normal mode
2. Memory usage is reduced with checkpointing
3. Model can train successfully with checkpointing enabled

This test ensures the OOM fix works correctly.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from models.unet3d import UNet3D


def test_gradient_checkpointing():
    """Test that gradient checkpointing works and saves memory"""
    print("="*80)
    print("TEST: Gradient Checkpointing")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cpu':
        print("⚠️  CUDA not available - memory savings test skipped")
        print("Will verify functional correctness only")

    # Create small test inputs
    B, latent_dim, T, h, w = 1, 4, 16, 32, 32
    x = torch.randn(B, latent_dim, T, h, w, device=device, requires_grad=True)
    c = torch.randn(B, latent_dim, T, h, w, device=device)
    t = torch.randint(0, 1000, (B,), device=device)

    print(f"\nInput shape: {x.shape}")
    print(f"Conditioning shape: {c.shape}")

    # Test 1: Without checkpointing
    print("\n" + "-"*80)
    print("Test 1: WITHOUT Gradient Checkpointing")
    print("-"*80)

    model_no_ckpt = UNet3D(
        latent_dim=latent_dim,
        model_channels=64,  # Small for testing
        num_res_blocks=2,
        attention_levels=[1],
        channel_mult=(1, 2, 4),
        num_heads=4,
        use_checkpoint=False  # ← Disabled
    ).to(device)

    model_no_ckpt.train()

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Forward pass
    output_no_ckpt = model_no_ckpt(x, t, c)
    loss_no_ckpt = output_no_ckpt.pow(2).mean()

    # Backward pass
    loss_no_ckpt.backward()

    if device == 'cuda':
        mem_no_ckpt = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"✓ Forward + Backward complete")
        print(f"  Peak memory: {mem_no_ckpt:.2f} MB")
    else:
        print(f"✓ Forward + Backward complete")
        print(f"  Loss: {loss_no_ckpt.item():.6f}")

    # Test 2: With checkpointing
    print("\n" + "-"*80)
    print("Test 2: WITH Gradient Checkpointing")
    print("-"*80)

    model_with_ckpt = UNet3D(
        latent_dim=latent_dim,
        model_channels=64,
        num_res_blocks=2,
        attention_levels=[1],
        channel_mult=(1, 2, 4),
        num_heads=4,
        use_checkpoint=True  # ← Enabled
    ).to(device)

    # Copy weights from first model for fair comparison
    model_with_ckpt.load_state_dict(model_no_ckpt.state_dict())
    model_with_ckpt.train()

    # Clear gradient from previous test
    x.grad = None

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Forward pass
    output_with_ckpt = model_with_ckpt(x, t, c)
    loss_with_ckpt = output_with_ckpt.pow(2).mean()

    # Backward pass
    loss_with_ckpt.backward()

    if device == 'cuda':
        mem_with_ckpt = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"✓ Forward + Backward complete")
        print(f"  Peak memory: {mem_with_ckpt:.2f} MB")
    else:
        print(f"✓ Forward + Backward complete")
        print(f"  Loss: {loss_with_ckpt.item():.6f}")

    # Test 3: Verify correctness
    print("\n" + "-"*80)
    print("Test 3: Verify Correctness")
    print("-"*80)

    # Check outputs are identical (or very close due to numerical precision)
    output_diff = (output_no_ckpt - output_with_ckpt).abs().max().item()
    loss_diff = abs(loss_no_ckpt.item() - loss_with_ckpt.item())

    print(f"\nOutput difference: {output_diff:.2e}")
    print(f"Loss difference: {loss_diff:.2e}")

    if output_diff < 1e-5 and loss_diff < 1e-6:
        print("✓ Outputs are identical (within numerical precision)")
    else:
        print(f"⚠️  Outputs differ more than expected")
        print(f"   This may indicate a bug in checkpointing implementation")

    # Test 4: Memory savings
    if device == 'cuda':
        print("\n" + "-"*80)
        print("Test 4: Memory Savings")
        print("-"*80)

        mem_saved = mem_no_ckpt - mem_with_ckpt
        mem_saved_pct = (mem_saved / mem_no_ckpt) * 100

        print(f"\nMemory without checkpointing: {mem_no_ckpt:.2f} MB")
        print(f"Memory with checkpointing:    {mem_with_ckpt:.2f} MB")
        print(f"Memory saved:                 {mem_saved:.2f} MB ({mem_saved_pct:.1f}%)")

        if mem_saved > 0:
            print(f"\n✓ Gradient checkpointing REDUCES memory by {mem_saved_pct:.1f}%")
        else:
            print(f"\n⚠️  Gradient checkpointing did NOT reduce memory")
            print(f"   (May be because test model is too small)")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    results = {
        'Functional correctness': output_diff < 1e-5 and loss_diff < 1e-6,
    }

    if device == 'cuda':
        results['Memory savings'] = mem_saved > 0

    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:30s}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("Gradient checkpointing is working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("Review implementation for potential issues.")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = test_gradient_checkpointing()
    sys.exit(0 if success else 1)
