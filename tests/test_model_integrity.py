"""
Test suite to verify model integrity for VAE training.

Tests cover:
1. Data shape validation
2. Model forward pass
3. Loss computation
4. Gradient flow
5. Memory constraints
6. Numerical stability
"""

import pytest
import torch
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.vae import VideoVAE
from train_vae import AutoencoderLoss


@pytest.fixture
def config():
    """Load VAE training config"""
    config_path = Path(__file__).parent.parent / "config" / "vae_training.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def device():
    """Get available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def vae_model(config, device):
    """Create VAE model"""
    model = VideoVAE(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim'],
        base_channels=config['model']['vae_base_channels'],
        scaling_factor=config['model']['vae_scaling_factor'],
    ).to(device)
    return model


@pytest.fixture
def loss_fn(config, device):
    """Create loss function"""
    return AutoencoderLoss(config).to(device)


class TestDataShapes:
    """Test data shape integrity"""

    def test_input_shape_5d(self, vae_model, device):
        """Test that model accepts 5D input (B, C, D, H, W)"""
        B, C, D, H, W = 2, 1, 50, 512, 512
        x = torch.randn(B, C, D, H, W, device=device)

        # Should not raise
        recon, z = vae_model(x)

        assert recon.shape == x.shape, f"Reconstruction shape {recon.shape} != input shape {x.shape}"
        print(f"✓ Input shape test passed: {x.shape} -> {recon.shape}")

    def test_reject_4d_input(self, vae_model, device):
        """Test that model rejects 4D input (B, D, H, W) - missing channel"""
        B, D, H, W = 2, 50, 512, 512
        x = torch.randn(B, D, H, W, device=device)

        with pytest.raises(RuntimeError, match="Expected.*5D"):
            recon, z = vae_model(x)
        print("✓ 4D input rejection test passed")

    def test_reject_6d_input(self, vae_model, device):
        """Test that model rejects 6D input (extra dimension)"""
        B, C1, C2, D, H, W = 2, 1, 1, 50, 512, 512
        x = torch.randn(B, C1, C2, D, H, W, device=device)

        with pytest.raises(RuntimeError, match="Expected.*5D"):
            recon, z = vae_model(x)
        print("✓ 6D input rejection test passed")

    def test_latent_shape(self, vae_model, device):
        """Test latent space dimensionality"""
        B, C, D, H, W = 2, 1, 50, 512, 512
        x = torch.randn(B, C, D, H, W, device=device)

        recon, z = vae_model(x)

        # Latent should be 8× smaller spatially (512->64)
        # and same depth (no temporal compression in VideoVAE)
        expected_latent = (B, 4, D, H//8, W//8)  # 4 latent channels
        assert z.shape == expected_latent, f"Latent shape {z.shape} != expected {expected_latent}"
        print(f"✓ Latent shape test passed: {z.shape}")

    def test_batch_size_flexibility(self, vae_model, device):
        """Test model works with different batch sizes"""
        C, D, H, W = 1, 50, 512, 512

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, C, D, H, W, device=device)
            recon, z = vae_model(x)

            assert recon.shape[0] == batch_size
            assert z.shape[0] == batch_size

        print("✓ Batch size flexibility test passed")

    def test_variable_depth(self, vae_model, device):
        """Test model handles variable depth (slice count)"""
        B, C, H, W = 2, 1, 512, 512

        for depth in [10, 20, 50, 100]:
            x = torch.randn(B, C, depth, H, W, device=device)
            recon, z = vae_model(x)

            assert recon.shape[2] == depth, f"Depth mismatch: {recon.shape[2]} != {depth}"

        print("✓ Variable depth test passed")


class TestModelForward:
    """Test model forward pass integrity"""

    def test_forward_no_nan(self, vae_model, device):
        """Test forward pass produces no NaN values"""
        B, C, D, H, W = 2, 1, 50, 512, 512
        x = torch.randn(B, C, D, H, W, device=device)

        recon, z = vae_model(x)

        assert not torch.isnan(recon).any(), "Reconstruction contains NaN"
        assert not torch.isnan(z).any(), "Latent contains NaN"
        print("✓ No NaN test passed")

    def test_forward_no_inf(self, vae_model, device):
        """Test forward pass produces no Inf values"""
        B, C, D, H, W = 2, 1, 50, 512, 512
        x = torch.randn(B, C, D, H, W, device=device)

        recon, z = vae_model(x)

        assert not torch.isinf(recon).any(), "Reconstruction contains Inf"
        assert not torch.isinf(z).any(), "Latent contains Inf"
        print("✓ No Inf test passed")

    def test_output_range(self, vae_model, device):
        """Test reconstruction output is in reasonable range"""
        B, C, D, H, W = 2, 1, 50, 512, 512
        # Input in [-1, 1] range (normalized CT data)
        x = torch.randn(B, C, D, H, W, device=device).clamp(-1, 1)

        recon, z = vae_model(x)

        # Reconstruction should be roughly in [-2, 2] range
        # (allowing some overshoot during training)
        assert recon.min() > -5.0, f"Reconstruction too negative: {recon.min()}"
        assert recon.max() < 5.0, f"Reconstruction too positive: {recon.max()}"
        print(f"✓ Output range test passed: [{recon.min():.3f}, {recon.max():.3f}]")

    def test_deterministic_forward(self, vae_model, device):
        """Test forward pass is deterministic (no dropout/stochastic layers)"""
        B, C, D, H, W = 2, 1, 50, 512, 512
        x = torch.randn(B, C, D, H, W, device=device)

        vae_model.eval()
        with torch.no_grad():
            recon1, z1 = vae_model(x)
            recon2, z2 = vae_model(x)

        assert torch.allclose(recon1, recon2, atol=1e-6), "Forward pass not deterministic"
        assert torch.allclose(z1, z2, atol=1e-6), "Latent encoding not deterministic"
        print("✓ Deterministic forward test passed")

    def test_encode_decode_cycle(self, vae_model, device):
        """Test encode->decode cycle maintains shapes"""
        B, C, D, H, W = 2, 1, 50, 512, 512
        x = torch.randn(B, C, D, H, W, device=device)

        # Manual encode-decode
        z = vae_model.encode(x)
        recon = vae_model.decode(z)

        assert recon.shape == x.shape, "Encode-decode cycle broke shape"
        print("✓ Encode-decode cycle test passed")


class TestLossComputation:
    """Test loss computation integrity"""

    def test_loss_computation(self, vae_model, loss_fn, device):
        """Test loss can be computed without errors"""
        B, C, D, H, W = 2, 1, 50, 512, 512
        x = torch.randn(B, C, D, H, W, device=device)

        recon, z = vae_model(x)
        loss, loss_dict = loss_fn(recon, x, step=0)

        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
        assert loss.item() >= 0, "Loss is negative"
        print(f"✓ Loss computation test passed: {loss.item():.4f}")

    def test_loss_components(self, vae_model, loss_fn, device):
        """Test individual loss components"""
        B, C, D, H, W = 2, 1, 50, 512, 512
        x = torch.randn(B, C, D, H, W, device=device)

        recon, z = vae_model(x)
        loss, loss_dict = loss_fn(recon, x, step=0)

        # Check all components exist and are reasonable
        assert 'loss' in loss_dict
        assert 'recon_loss' in loss_dict
        assert 'perceptual_loss' in loss_dict
        assert 'ssim_loss' in loss_dict

        assert loss_dict['recon_loss'] >= 0, "Reconstruction loss negative"
        print(f"✓ Loss components test passed: {loss_dict}")

    def test_loss_backward(self, vae_model, loss_fn, device):
        """Test loss backward pass"""
        B, C, D, H, W = 2, 1, 10, 256, 256  # Smaller for speed
        x = torch.randn(B, C, D, H, W, device=device)

        vae_model.zero_grad()
        recon, z = vae_model(x)
        loss, loss_dict = loss_fn(recon, x, step=0)
        loss.backward()

        # Check gradients exist and are not NaN
        grad_count = 0
        nan_count = 0
        for name, param in vae_model.named_parameters():
            if param.requires_grad:
                grad_count += 1
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        nan_count += 1
                        print(f"  NaN gradient in: {name}")

        assert grad_count > 0, "No gradients computed"
        assert nan_count == 0, f"NaN gradients in {nan_count}/{grad_count} parameters"
        print(f"✓ Backward pass test passed: {grad_count} parameters with gradients")


class TestGradientFlow:
    """Test gradient flow integrity"""

    def test_gradient_flow_encoder(self, vae_model, device):
        """Test gradients flow through encoder"""
        B, C, D, H, W = 1, 1, 10, 256, 256
        x = torch.randn(B, C, D, H, W, device=device)

        vae_model.zero_grad()
        z = vae_model.encode(x)
        loss = z.mean()
        loss.backward()

        # Check encoder has gradients
        encoder_has_grad = False
        for name, param in vae_model.encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                encoder_has_grad = True
                break

        assert encoder_has_grad, "No gradients in encoder"
        print("✓ Encoder gradient flow test passed")

    def test_gradient_flow_decoder(self, vae_model, device):
        """Test gradients flow through decoder"""
        B, C, D, H, W = 1, 1, 10, 256, 256
        x = torch.randn(B, C, D, H, W, device=device)

        vae_model.zero_grad()
        z = vae_model.encode(x)
        recon = vae_model.decode(z)
        loss = recon.mean()
        loss.backward()

        # Check decoder has gradients
        decoder_has_grad = False
        for name, param in vae_model.decoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                decoder_has_grad = True
                break

        assert decoder_has_grad, "No gradients in decoder"
        print("✓ Decoder gradient flow test passed")

    def test_gradient_magnitude(self, vae_model, loss_fn, device):
        """Test gradient magnitudes are reasonable"""
        B, C, D, H, W = 2, 1, 10, 256, 256
        x = torch.randn(B, C, D, H, W, device=device)

        vae_model.zero_grad()
        recon, z = vae_model(x)
        loss, _ = loss_fn(recon, x, step=0)
        loss.backward()

        # Check gradient magnitudes
        max_grad = 0.0
        for param in vae_model.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())

        assert max_grad < 1000.0, f"Gradient too large: {max_grad}"
        assert max_grad > 1e-10, f"Gradient too small: {max_grad}"
        print(f"✓ Gradient magnitude test passed: max_grad={max_grad:.6f}")


class TestMemoryConstraints:
    """Test memory usage constraints"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_usage_v100(self, vae_model, device):
        """Test memory usage fits in V100 16GB"""
        if device.type != 'cuda':
            pytest.skip("CUDA required for memory test")

        B, C, D, H, W = 2, 1, 50, 512, 512
        x = torch.randn(B, C, D, H, W, device=device)

        torch.cuda.reset_peak_memory_stats()

        recon, z = vae_model(x)
        loss = (recon - x).pow(2).mean()
        loss.backward()

        peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB

        # Should fit in 16GB V100 with headroom
        assert peak_memory < 12.0, f"Peak memory {peak_memory:.2f}GB exceeds 12GB limit"
        print(f"✓ Memory test passed: {peak_memory:.2f}GB peak")

    def test_no_memory_leak(self, vae_model, device):
        """Test multiple forward passes don't leak memory"""
        if device.type != 'cuda':
            pytest.skip("CUDA required for memory test")

        B, C, D, H, W = 1, 1, 10, 256, 256

        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        for _ in range(10):
            x = torch.randn(B, C, D, H, W, device=device)
            with torch.no_grad():
                recon, z = vae_model(x)
            del x, recon, z
            torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()
        memory_growth = (final_memory - initial_memory) / 1e6  # MB

        assert memory_growth < 100, f"Memory leak detected: {memory_growth:.2f}MB growth"
        print(f"✓ No memory leak test passed: {memory_growth:.2f}MB growth")


class TestNumericalStability:
    """Test numerical stability"""

    def test_extreme_input_values(self, vae_model, device):
        """Test model handles extreme input values"""
        B, C, D, H, W = 1, 1, 10, 256, 256

        # Test very negative values
        x_neg = torch.full((B, C, D, H, W), -10.0, device=device)
        recon_neg, z_neg = vae_model(x_neg)
        assert not torch.isnan(recon_neg).any(), "NaN with extreme negative input"

        # Test very positive values
        x_pos = torch.full((B, C, D, H, W), 10.0, device=device)
        recon_pos, z_pos = vae_model(x_pos)
        assert not torch.isnan(recon_pos).any(), "NaN with extreme positive input"

        print("✓ Extreme input values test passed")

    def test_zero_input(self, vae_model, device):
        """Test model handles zero input"""
        B, C, D, H, W = 1, 1, 10, 256, 256
        x = torch.zeros(B, C, D, H, W, device=device)

        recon, z = vae_model(x)

        assert not torch.isnan(recon).any(), "NaN with zero input"
        assert not torch.isnan(z).any(), "NaN latent with zero input"
        print("✓ Zero input test passed")

    def test_mixed_precision_stability(self, vae_model, device):
        """Test model stable with mixed precision"""
        if device.type != 'cuda':
            pytest.skip("CUDA required for mixed precision test")

        B, C, D, H, W = 2, 1, 10, 256, 256
        x = torch.randn(B, C, D, H, W, device=device)

        with torch.cuda.amp.autocast():
            recon, z = vae_model(x)

        assert not torch.isnan(recon).any(), "NaN with mixed precision"
        assert recon.dtype == torch.float16 or recon.dtype == torch.bfloat16, \
            f"Expected fp16/bf16, got {recon.dtype}"
        print(f"✓ Mixed precision test passed: dtype={recon.dtype}")


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*70)
    print("MODEL INTEGRITY TEST SUITE")
    print("="*70 + "\n")

    pytest.main([__file__, "-v", "--tb=short", "-s"])


if __name__ == '__main__':
    run_all_tests()
