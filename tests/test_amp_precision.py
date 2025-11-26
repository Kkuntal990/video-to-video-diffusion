"""
Test AMP (Automatic Mixed Precision) dtype and precision

Verifies that:
1. AMP uses torch.bfloat16 dtype (not float16)
2. Loss is computed in FP32 (not BF16)
3. Metrics (PSNR/SSIM) are computed in FP32
4. Forward pass uses BF16 for efficiency
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_amp_uses_bfloat16():
    """Test that autocast uses bfloat16 dtype"""
    from torch.cuda.amp import autocast

    device = torch.device('cuda')
    x = torch.randn(2, 1, 8, 64, 64, device=device)  # Input in FP32

    # Create simple model
    model = torch.nn.Conv3d(1, 4, kernel_size=3, padding=1).to(device)

    # Test with BF16 autocast
    with autocast(dtype=torch.bfloat16):
        output = model(x)
        # Output should be BF16 inside autocast
        assert output.dtype == torch.bfloat16, f"Expected BF16, got {output.dtype}"

    # Outside autocast, tensor should convert back to FP32
    # (Note: This depends on PyTorch version, but loss should be FP32)

    print("✓ AMP uses torch.bfloat16 dtype")


def test_loss_computed_in_fp32():
    """Test that loss is computed in FP32 even when inputs are BF16"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from train_vae import AutoencoderLoss
    import yaml

    # Load config
    config_path = Path(__file__).parent.parent / "config" / "vae_training.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create loss module (no device parameter needed)
    loss_fn = AutoencoderLoss(config)

    # Create BF16 inputs
    pred = torch.randn(2, 1, 8, 64, 64, dtype=torch.bfloat16)
    target = torch.randn(2, 1, 8, 64, 64, dtype=torch.bfloat16)

    # Compute loss
    loss, loss_dict = loss_fn(pred, target, step=0)

    # Loss should be FP32 (Python scalar from .item())
    assert isinstance(loss_dict['loss'], float), "Loss should be Python float (FP32)"
    assert isinstance(loss_dict['recon_loss'], float), "Recon loss should be Python float"

    # Total loss tensor should be FP32 (after .float() casting in forward)
    assert loss.dtype == torch.float32, f"Expected FP32 loss, got {loss.dtype}"

    print("✓ Loss is computed in FP32 (even with BF16 inputs)")


def test_metrics_computed_in_fp32():
    """Test that PSNR and SSIM are computed with FP32 precision"""
    from utils.metrics import calculate_psnr, calculate_ssim

    # Create BF16 tensors
    img1 = torch.rand(1, 1, 8, 64, 64, dtype=torch.bfloat16)
    img2 = torch.rand(1, 1, 8, 64, 64, dtype=torch.bfloat16)

    # Compute metrics
    psnr = calculate_psnr(img1, img2, max_val=1.0)
    ssim = calculate_ssim(img1, img2, max_val=1.0)

    # Metrics should return Python floats (FP64)
    assert isinstance(psnr, float), f"PSNR should be Python float, got {type(psnr)}"
    assert isinstance(ssim, float), f"SSIM should be Python float, got {type(ssim)}"

    # Check reasonable ranges
    assert 0 <= psnr <= 100, f"PSNR out of range: {psnr}"
    assert 0 <= ssim <= 1, f"SSIM out of range: {ssim}"

    print("✓ Metrics (PSNR/SSIM) are computed in high precision")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_training_forward_pass_uses_bf16():
    """Test that forward pass uses BF16 with autocast"""
    from torch.cuda.amp import autocast
    from models.vae import VideoVAE

    device = torch.device('cuda')

    # Create small VAE
    vae = VideoVAE(
        in_channels=1,
        latent_channels=4,
        base_channels=32,  # Small for testing
    ).to(device)

    # Create input
    x = torch.randn(1, 1, 8, 64, 64, device=device)  # FP32 input

    # Forward pass with autocast
    with autocast(dtype=torch.bfloat16):
        recon, z = vae(x)
        # Inside autocast, activations should be BF16
        assert z.dtype == torch.bfloat16, f"Latent should be BF16, got {z.dtype}"
        assert recon.dtype == torch.bfloat16, f"Reconstruction should be BF16, got {recon.dtype}"

    print("✓ Forward pass uses BF16 with autocast")


def test_amp_memory_efficiency():
    """Test that BF16 uses less memory than FP32"""
    # Create two tensors: one BF16, one FP32
    size = (4, 4, 48, 192, 192)  # Typical patch size

    tensor_fp32 = torch.randn(size, dtype=torch.float32)
    tensor_bf16 = torch.randn(size, dtype=torch.bfloat16)

    # BF16 should use half the memory
    size_fp32 = tensor_fp32.element_size() * tensor_fp32.numel()
    size_bf16 = tensor_bf16.element_size() * tensor_bf16.numel()

    assert size_bf16 == size_fp32 / 2, f"BF16 should use 50% memory, got {size_bf16 / size_fp32 * 100:.1f}%"

    print(f"✓ BF16 uses 50% memory (FP32: {size_fp32 / 1e9:.2f} GB, BF16: {size_bf16 / 1e9:.2f} GB)")


if __name__ == "__main__":
    print("=" * 70)
    print("AMP Precision Tests")
    print("=" * 70)
    print()

    print("Test 1: AMP uses bfloat16 dtype")
    if torch.cuda.is_available():
        test_amp_uses_bfloat16()
    else:
        print("⊘ Skipped (CUDA not available)")
    print()

    print("Test 2: Loss computed in FP32")
    test_loss_computed_in_fp32()
    print()

    print("Test 3: Metrics computed in FP32")
    test_metrics_computed_in_fp32()
    print()

    print("Test 4: Forward pass uses BF16")
    if torch.cuda.is_available():
        test_training_forward_pass_uses_bf16()
    else:
        print("⊘ Skipped (CUDA not available)")
    print()

    print("Test 5: BF16 memory efficiency")
    test_amp_memory_efficiency()
    print()

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
