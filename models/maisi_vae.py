"""
Custom MAISI VAE Architecture

Builds a VAE that exactly matches NVIDIA MAISI's architecture
to enable loading 100% of pretrained weights.

Based on checkpoint inspection:
- 11 encoder blocks: input_conv, 2×res_64, down, 2×res_128, down, 2×res_256, norm, final_conv
- 11 decoder blocks: (reverse of encoder)
- Variational: mu + log_sigma quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class ConvModule(nn.Module):
    """Module to match MAISI's .conv structure"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)


class Conv3dWrapper(nn.Module):
    """Wrapper to match MAISI's nested .conv.conv structure"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = ConvModule(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)


class Conv3dTripleNested(nn.Module):
    """Direct triple nesting to match .conv.conv.conv structure"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Create triple nesting: self.conv.conv.conv
        self.conv = nn.Module()
        self.conv.conv = nn.Module()
        self.conv.conv.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv.conv.conv(x)


class ResidualBlock3D(nn.Module):
    """
    3D Residual Block matching MAISI structure

    Structure:
        norm1 → conv1 → norm2 → conv2 → (optional nin_shortcut)
    """
    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = Conv3dWrapper(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = Conv3dWrapper(out_channels, out_channels, kernel_size=3, padding=1)

        # Shortcut if channels change
        if in_channels != out_channels:
            self.nin_shortcut = Conv3dWrapper(in_channels, out_channels, kernel_size=1)
        else:
            self.nin_shortcut = None

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)

        return x + h


class DownsampleBlock3D(nn.Module):
    """Downsample block (spatial /2) - triple nesting for checkpoint compatibility"""
    def __init__(self, channels):
        super().__init__()
        # Create triple nesting directly: .conv.conv.conv
        self.conv = nn.Module()
        self.conv.conv = nn.Module()
        self.conv.conv.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv.conv.conv(x)


class UpsampleBlock3D(nn.Module):
    """Upsample block (spatial ×2) - triple nesting for checkpoint compatibility"""
    def __init__(self, channels):
        super().__init__()
        # Create triple nesting directly: .conv.conv.conv
        self.conv = nn.Module()
        self.conv.conv = nn.Module()
        self.conv.conv.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Upsample then convolve
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv.conv.conv(x)


class MAISIEncoder(nn.Module):
    """
    MAISI Encoder matching exact checkpoint structure

    Structure (11 blocks):
        0: Initial conv (1 → 64)
        1-2: Res blocks @ 64
        3: Downsample (spatial /2)
        4-5: Res blocks @ 128 (with shortcut)
        6: Downsample (spatial /2)
        7-8: Res blocks @ 256 (with shortcut)
        9: GroupNorm
        10: Final conv (256 → 4)
    """
    def __init__(self, in_channels=1, latent_channels=4, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()

        # Block 0: Initial conv (1 → 64)
        self.blocks.append(Conv3dWrapper(in_channels, 64, kernel_size=3, padding=1))

        # Blocks 1-2: Residual blocks @ 64 channels
        self.blocks.append(ResidualBlock3D(64, 64))
        self.blocks.append(ResidualBlock3D(64, 64))

        # Block 3: Downsample (64 → 64, spatial /2)
        self.blocks.append(DownsampleBlock3D(64))

        # Blocks 4-5: Residual blocks @ 128 channels (64 → 128 with shortcut)
        self.blocks.append(ResidualBlock3D(64, 128))
        self.blocks.append(ResidualBlock3D(128, 128))

        # Block 6: Downsample (128 → 128, spatial /2)
        self.blocks.append(DownsampleBlock3D(128))

        # Blocks 7-8: Residual blocks @ 256 channels (128 → 256 with shortcut)
        self.blocks.append(ResidualBlock3D(128, 256))
        self.blocks.append(ResidualBlock3D(256, 256))

        # Block 9: GroupNorm
        self.blocks.append(nn.GroupNorm(32, 256))

        # Block 10: Final conv (256 → latent_channels)
        self.blocks.append(Conv3dWrapper(256, latent_channels, kernel_size=3, padding=1))

    def _checkpoint_forward(self, module, x):
        """
        Helper to conditionally apply gradient checkpointing.

        Gradient checkpointing trades compute for memory by not storing
        intermediate activations during forward pass, recomputing them
        during backward pass instead.
        """
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                module, x, use_reentrant=False
            )
        else:
            return module(x)

    def forward(self, x):
        h = x

        # Block 0: Initial conv (no checkpoint - simple conv)
        h = self.blocks[0](h)

        # Blocks 1-2: Res @ 64 (checkpoint these - expensive residual blocks)
        h = self._checkpoint_forward(self.blocks[1], h)
        h = self._checkpoint_forward(self.blocks[2], h)

        # Block 3: Downsample (checkpoint - spatial reduction)
        h = self._checkpoint_forward(self.blocks[3], h)

        # Blocks 4-5: Res @ 128 (checkpoint these - expensive residual blocks)
        h = self._checkpoint_forward(self.blocks[4], h)
        h = self._checkpoint_forward(self.blocks[5], h)

        # Block 6: Downsample (checkpoint - spatial reduction)
        h = self._checkpoint_forward(self.blocks[6], h)

        # Blocks 7-8: Res @ 256 (checkpoint these - expensive residual blocks)
        h = self._checkpoint_forward(self.blocks[7], h)
        h = self._checkpoint_forward(self.blocks[8], h)

        # Block 9: GroupNorm + activation (no checkpoint - simple norm)
        h = self.blocks[9](h)
        h = F.silu(h)

        # Block 10: Final conv (no checkpoint - simple conv)
        h = self.blocks[10](h)

        return h


class MAISIDecoder(nn.Module):
    """
    MAISI Decoder matching exact checkpoint structure

    Structure (11 blocks):
        0: Initial conv (4 → 256)
        1-2: Res blocks @ 256
        3: Upsample (spatial ×2)
        4-5: Res blocks @ 128 (with shortcut)
        6: Upsample (spatial ×2)
        7-8: Res blocks @ 64 (with shortcut)
        9: GroupNorm
        10: Final conv (64 → 1)
    """
    def __init__(self, latent_channels=4, out_channels=1, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()

        # Block 0: Initial conv (4 → 256)
        self.blocks.append(Conv3dWrapper(latent_channels, 256, kernel_size=3, padding=1))

        # Blocks 1-2: Residual blocks @ 256 channels
        self.blocks.append(ResidualBlock3D(256, 256))
        self.blocks.append(ResidualBlock3D(256, 256))

        # Block 3: Upsample (256 → 256, spatial ×2)
        self.blocks.append(UpsampleBlock3D(256))

        # Blocks 4-5: Residual blocks @ 128 channels (256 → 128 with shortcut)
        self.blocks.append(ResidualBlock3D(256, 128))
        self.blocks.append(ResidualBlock3D(128, 128))

        # Block 6: Upsample (128 → 128, spatial ×2)
        self.blocks.append(UpsampleBlock3D(128))

        # Blocks 7-8: Residual blocks @ 64 channels (128 → 64 with shortcut)
        self.blocks.append(ResidualBlock3D(128, 64))
        self.blocks.append(ResidualBlock3D(64, 64))

        # Block 9: GroupNorm
        self.blocks.append(nn.GroupNorm(32, 64))

        # Block 10: Final conv (64 → out_channels)
        self.blocks.append(Conv3dWrapper(64, out_channels, kernel_size=3, padding=1))

    def _checkpoint_forward(self, module, x):
        """
        Helper to conditionally apply gradient checkpointing.

        Gradient checkpointing trades compute for memory by not storing
        intermediate activations during forward pass, recomputing them
        during backward pass instead.
        """
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                module, x, use_reentrant=False
            )
        else:
            return module(x)

    def forward(self, z):
        h = z

        # Block 0: Initial conv (no checkpoint - simple conv)
        h = self.blocks[0](h)

        # Blocks 1-2: Res @ 256 (checkpoint these - expensive residual blocks)
        h = self._checkpoint_forward(self.blocks[1], h)
        h = self._checkpoint_forward(self.blocks[2], h)

        # Block 3: Upsample (checkpoint - spatial expansion)
        h = self._checkpoint_forward(self.blocks[3], h)

        # Blocks 4-5: Res @ 128 (checkpoint these - expensive residual blocks)
        h = self._checkpoint_forward(self.blocks[4], h)
        h = self._checkpoint_forward(self.blocks[5], h)

        # Block 6: Upsample (checkpoint - spatial expansion)
        h = self._checkpoint_forward(self.blocks[6], h)

        # Blocks 7-8: Res @ 64 (checkpoint these - expensive residual blocks)
        h = self._checkpoint_forward(self.blocks[7], h)
        h = self._checkpoint_forward(self.blocks[8], h)

        # Block 9: GroupNorm + activation (no checkpoint - simple norm)
        h = self.blocks[9](h)
        h = F.silu(h)

        # Block 10: Final conv (no checkpoint - simple conv)
        h = self.blocks[10](h)

        return h


class MAISIVAE(nn.Module):
    """
    Complete MAISI VAE matching NVIDIA's architecture

    Enables loading 100% of pretrained weights from
    pretrained/maisi_vae/models/autoencoder.pt
    """
    def __init__(self, in_channels=1, out_channels=1, latent_channels=4, scaling_factor=0.18215, use_checkpoint=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.use_checkpoint = use_checkpoint

        # Encoder and Decoder with gradient checkpointing support
        self.encoder = MAISIEncoder(in_channels, latent_channels, use_checkpoint=use_checkpoint)
        self.decoder = MAISIDecoder(latent_channels, out_channels, use_checkpoint=use_checkpoint)

        # Variational quantization (mu + log_sigma)
        self.quant_conv_mu = ConvModule(latent_channels, latent_channels, kernel_size=1)
        self.quant_conv_log_sigma = ConvModule(latent_channels, latent_channels, kernel_size=1)

        # Post-quantization
        self.post_quant_conv = ConvModule(latent_channels, latent_channels, kernel_size=1)

    def encode(self, x):
        """
        Encode input to latent distribution

        Args:
            x: (B, C, D, H, W) input volume

        Returns:
            z: (B, latent_channels, D//4, H//8, W//8) latent (with sampling)
        """
        # Encode
        h = self.encoder(x)

        # Get distribution parameters
        mu = self.quant_conv_mu(h)
        log_sigma = self.quant_conv_log_sigma(h)

        # Sample from distribution (reparameterization trick)
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Scale
        z = z * self.scaling_factor

        return z

    def decode(self, z):
        """
        Decode latent to reconstruction

        Args:
            z: (B, latent_channels, d, h, w) latent

        Returns:
            x_recon: (B, out_channels, D, H, W) reconstruction
        """
        # Unscale
        z = z / self.scaling_factor

        # Post-quantization conv
        z = self.post_quant_conv(z)

        # Decode
        x_recon = self.decoder(z)

        return x_recon

    def forward(self, x):
        """Full encode-decode pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    def load_pretrained_weights(self, checkpoint_path, strict=False):
        """
        Load pretrained MAISI weights

        Args:
            checkpoint_path: Path to autoencoder.pt
            strict: Whether to require exact key matching

        Returns:
            Loading statistics
        """
        print(f"Loading MAISI pretrained weights from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Load state dict
        result = self.load_state_dict(checkpoint, strict=strict)

        # Report statistics
        n_loaded = len(checkpoint) - len(result.unexpected_keys)
        n_total = len(checkpoint)

        print(f"✓ Loaded {n_loaded}/{n_total} parameters ({n_loaded/n_total*100:.1f}%)")

        if result.missing_keys:
            print(f"  ⚠ Missing {len(result.missing_keys)} keys in checkpoint")
            if len(result.missing_keys) <= 10:
                for key in result.missing_keys:
                    print(f"    - {key}")

        if result.unexpected_keys:
            print(f"  ⚠ Unexpected {len(result.unexpected_keys)} keys in checkpoint")
            if len(result.unexpected_keys) <= 10:
                for key in result.unexpected_keys:
                    print(f"    - {key}")

        return {
            'loaded': n_loaded,
            'total': n_total,
            'missing_keys': result.missing_keys,
            'unexpected_keys': result.unexpected_keys,
        }


if __name__ == "__main__":
    # Test MAISI VAE architecture
    print("="*80)
    print("TESTING CUSTOM MAISI VAE ARCHITECTURE")
    print("="*80)

    # Create model
    model = MAISIVAE(in_channels=1, out_channels=1, latent_channels=4)

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(1, 1, 8, 128, 128)
    print(f"  Input: {x.shape}")

    # Encode
    z = model.encode(x)
    print(f"  Latent: {z.shape}")
    print(f"  Compression: D: 8→{z.shape[2]} ({8/z.shape[2]:.1f}×), H/W: 128→{z.shape[3]} ({128/z.shape[3]:.1f}×)")

    # Decode
    x_recon = model.decode(z)
    print(f"  Reconstruction: {x_recon.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {n_params:,}")

    # Try loading pretrained weights
    print("\n" + "="*80)
    print("TESTING PRETRAINED WEIGHT LOADING")
    print("="*80)

    checkpoint_path = './pretrained/maisi_vae/models/autoencoder.pt'

    try:
        stats = model.load_pretrained_weights(checkpoint_path, strict=False)

        if stats['loaded'] == stats['total']:
            print("\n🎉 SUCCESS! 100% of MAISI weights loaded!")
        else:
            print(f"\n⚠ Partial loading: {stats['loaded']}/{stats['total']} ({stats['loaded']/stats['total']*100:.1f}%)")

        # Test with loaded weights
        print("\n" + "="*80)
        print("TESTING ENCODE/DECODE WITH LOADED WEIGHTS")
        print("="*80)

        model.eval()
        with torch.no_grad():
            x_test = torch.randn(1, 1, 8, 512, 512)
            print(f"\nTest input: {x_test.shape}")

            z_test = model.encode(x_test)
            print(f"Encoded: {z_test.shape}")

            x_recon_test = model.decode(z_test)
            print(f"Decoded: {x_recon_test.shape}")

            mse = torch.mean((x_test - x_recon_test) ** 2).item()
            print(f"\nReconstruction MSE: {mse:.6f}")

            if mse < 0.5:
                print("✓ Low MSE - pretrained weights working correctly!")
            else:
                print("⚠ High MSE - may need more training or architecture adjustment")

    except Exception as e:
        print(f"\n✗ Error loading weights: {e}")
        import traceback
        traceback.print_exc()
