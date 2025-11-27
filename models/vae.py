"""
3D Slice Interpolation VAE (Variational Autoencoder)

Spatial compression: 4× (H,W → H//4, W//4)
Temporal: Preserved (T → T, no depth compression)

Encoder: Compresses CT volumes (B, C, T, H, W) into latent space (B, latent_dim, T, H//4, W//4)
Decoder: Reconstructs CT volumes from latent representation

Architecture:
- 2 downsample stages (spatial 2×2 each = 4× total)
- No skip connections (designed for latent diffusion)
- Depth preserved throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Conv3DBlock(nn.Module):
    """3D Convolutional block with GroupNorm and SiLU activation"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResBlock3D(nn.Module):
    """3D Residual block with two conv layers"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv3DBlock(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.act(x)
        return x


class DownsampleBlock(nn.Module):
    """Downsampling block with 3D convolution"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Downsample spatial dimensions (H, W) by 2, keep temporal (T) same
        self.conv = nn.Conv3d(in_channels, out_channels,
                             kernel_size=(3, 4, 4),
                             stride=(1, 2, 2),
                             padding=(1, 1, 1))
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class UpsampleBlock(nn.Module):
    """Upsampling block with 3D transposed convolution"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Upsample spatial dimensions (H, W) by 2, keep temporal (T) same
        self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                                      kernel_size=(3, 4, 4),
                                      stride=(1, 2, 2),
                                      padding=(1, 1, 1))
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class VideoEncoder(nn.Module):
    """
    3D Video Encoder
    Input: (B, 3, T, H, W) - video clip
    Output: (B, latent_dim, T, H//4, W//4) - latent representation
    """

    def __init__(self, in_channels=3, latent_dim=4, base_channels=64):
        super().__init__()

        # Initial conv
        self.conv_in = Conv3DBlock(in_channels, base_channels)

        # Encoder blocks with downsampling (2 stages, downsample by 4)
        self.down1 = nn.Sequential(
            ResBlock3D(base_channels),
            ResBlock3D(base_channels),
            DownsampleBlock(base_channels, base_channels * 2)
        )

        self.down2 = nn.Sequential(
            ResBlock3D(base_channels * 2),
            ResBlock3D(base_channels * 2),
            DownsampleBlock(base_channels * 2, base_channels * 4)
        )

        # Middle blocks
        self.mid = nn.Sequential(
            ResBlock3D(base_channels * 4),
            ResBlock3D(base_channels * 4),
        )

        # Output projection to latent space (match SD VAE architecture)
        # SD VAE: outputs 8 channels, then quantizes to latent_dim
        self.conv_out = nn.Conv3d(base_channels * 4, 8, kernel_size=3, padding=1)

        # Quantization layer: 8 → latent_dim (typically 4)
        self.quant_conv = nn.Conv3d(8, latent_dim, kernel_size=1)

    def forward(self, x):
        # Input: (B, 3, T, H, W)
        h0 = self.conv_in(x)       # (B, 64, T, H, W)         - spatial: H×W
        h1 = self.down1(h0)        # (B, 128, T, H//2, W//2)  - spatial: H/2×W/2
        h2 = self.down2(h1)        # (B, 256, T, H//4, W//4)  - spatial: H/4×W/4
        h = self.mid(h2)           # (B, 256, T, H//4, W//4)
        h = self.conv_out(h)       # (B, 8, T, H//4, W//4)
        z = self.quant_conv(h)     # (B, latent_dim, T, H//4, W//4)
        return z


class VideoDecoder(nn.Module):
    """
    3D Slice Interpolation Decoder
    Input: (B, latent_dim, T, H//4, W//4) - latent representation
    Output: (B, 3, T, H, W) - reconstructed CT slices
    """

    def __init__(self, latent_dim=4, out_channels=3, base_channels=64):
        super().__init__()

        # Post-quantization layer: latent_dim → 8 (match SD VAE architecture)
        self.post_quant_conv = nn.Conv3d(latent_dim, 8, kernel_size=1)

        # Input projection from latent space
        self.conv_in = Conv3DBlock(8, base_channels * 4)

        # Middle blocks
        self.mid = nn.Sequential(
            ResBlock3D(base_channels * 4),
            ResBlock3D(base_channels * 4),
        )

        # Decoder blocks with upsampling (2 stages, upsample by 4)
        # up2: 256 → 128
        self.up2_upsample = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.up2_res = nn.Sequential(
            ResBlock3D(base_channels * 2),
            ResBlock3D(base_channels * 2),
        )

        # up3: 128 → 64
        self.up3_upsample = UpsampleBlock(base_channels * 2, base_channels)
        self.up3_res = nn.Sequential(
            ResBlock3D(base_channels),
            ResBlock3D(base_channels),
        )

        # Output conv
        self.conv_out = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Input: (B, latent_dim, T, H//4, W//4)
        x = self.post_quant_conv(x)  # (B, 8, T, H//4, W//4)
        x = self.conv_in(x)          # (B, 256, T, H//4, W//4)
        x = self.mid(x)              # (B, 256, T, H//4, W//4)

        # Decoder upsampling
        x = self.up2_upsample(x)     # (B, 128, T, H//2, W//2)
        x = self.up2_res(x)
        x = self.up3_upsample(x)     # (B, 64, T, H, W)
        x = self.up3_res(x)

        x = self.conv_out(x)      # (B, 1, T, H, W)
        x = torch.tanh(x)         # Bound output to [-1, 1]
        return x


class SliceInterpolationVAE(nn.Module):
    """
    3D VAE for CT Slice Interpolation

    Compresses 3D CT volumes into latent space and reconstructs them.
    Designed for latent diffusion models - encoder and decoder work independently.
    Input/Output format: (B, C, T, H, W) where T is depth (number of slices)
    """

    def __init__(self, in_channels=3, latent_dim=4, base_channels=64, scaling_factor=0.18215,
                 gradient_checkpointing=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.gradient_checkpointing = gradient_checkpointing

        # Initialize encoder and decoder
        self.encoder = VideoEncoder(in_channels, latent_dim, base_channels)
        self.decoder = VideoDecoder(latent_dim, in_channels, base_channels)

        # Scaling factor for latent space normalization
        # Standard SD VAE uses 0.18215, but this should be calculated from your dataset
        # To calculate: scale = 1 / std(latent_activations)
        self.scaling_factor = scaling_factor
        print(f"✓ Initialized Slice Interpolation VAE")
        print(f"  Latent dim: {latent_dim}, Base channels: {base_channels}")

    def encode(self, x):
        """
        Encode CT volume to latent space with scaling

        Args:
            x: (B, C, T, H, W) CT volume tensor in range [-1, 1]
        Returns:
            z: (B, latent_dim, T, h, w) scaled latent tensor
        """
        z = self.encoder(x)
        # Scale latents to normalize magnitude for diffusion
        z = z * self.scaling_factor
        return z

    def decode(self, z):
        """
        Decode latent to CT volume with unscaling

        Args:
            z: (B, latent_dim, T, h, w) scaled latent tensor
        Returns:
            x: (B, C, T, H, W) reconstructed CT volume in range [-1, 1]
        """
        # Unscale latents before decoding
        z = z / self.scaling_factor
        return self.decoder(z)

    def encode_with_posterior(self, x):
        """
        Encode CT volume and return posterior distribution parameters (mu, logvar).

        For VAE training from scratch.
        Splits the encoder output into mean and log variance.

        Args:
            x: (B, C, T, H, W) CT volume tensor in range [-1, 1]

        Returns:
            mu: (B, latent_dim//2, T, h, w) mean of latent distribution
            logvar: (B, latent_dim//2, T, h, w) log variance of latent distribution
        """
        # Get encoder output (before scaling)
        z = self.encoder(x)  # (B, latent_dim, T, H//8, W//8)

        # Split into mu and logvar (half channels each)
        # Standard VAE approach: encoder outputs 2*latent_dim channels
        # We need to output twice as many channels from conv_out

        # For now, split the latent_dim channels in half
        # mu = first half, logvar = second half
        mu, logvar = torch.chunk(z, 2, dim=1)

        return mu, logvar

    def forward(self, x):
        """
        Full forward pass: encode then decode

        Args:
            x: (B, C, T, H, W) CT volume tensor in range [-1, 1]
        Returns:
            recon: (B, C, T, H, W) reconstructed CT volume in range [-1, 1]
            z: (B, latent_dim, T, h, w) scaled latent representation
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def get_latent_shape(self, volume_shape):
        """Calculate latent shape given CT volume shape"""
        B, C, T, H, W = volume_shape
        return (B, self.latent_dim, T, H // 4, W // 4)

    @classmethod
    def from_pretrained(cls, model_name_or_path, method='auto', inflate_method='central',
                       strict=True, device='cpu', **kwargs):
        """
        Load Slice Interpolation VAE from pretrained weights (DISABLED)

        This method is not implemented. Use custom trained VAE instead.
        Load weights manually via torch.load() and model.load_state_dict().
        """
        raise NotImplementedError(
            "from_pretrained() is not available. "
            "Pretrained VAE loading has been removed. "
            "Use custom trained VAE and load weights manually in train.py"
        )


# Backward compatibility alias
VideoVAE = SliceInterpolationVAE


if __name__ == "__main__":
    # Test the Slice Interpolation VAE (4× spatial compression)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with increased latent capacity
    vae = SliceInterpolationVAE(in_channels=1, latent_dim=16, base_channels=128).to(device)

    # Test with random CT volume (batch=2, channels=1, slices=48, height=192, width=192)
    volume = torch.randn(2, 1, 48, 192, 192).to(device)

    print(f"Input shape: {volume.shape}")

    # Encode (192×192 → 48×48, 4× compression)
    latent = vae.encode(volume)
    print(f"Latent shape: {latent.shape}")  # Expected: (2, 16, 48, 48, 48)

    # Decode (48×48 → 192×192)
    recon = vae.decode(latent)
    print(f"Reconstruction shape: {recon.shape}")

    # Full forward
    recon, latent = vae(volume)
    print(f"Forward pass - Recon: {recon.shape}, Latent: {latent.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"Total parameters: {total_params:,}")
