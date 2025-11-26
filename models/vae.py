"""
3D Video VAE (Variational Autoencoder)

Encoder: Compresses video (T×H×W×3) into latent space (T×h×w×c)
Decoder: Reconstructs video from latent representation
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
    """Upsampling block with 3D transposed convolution and optional skip connections"""

    def __init__(self, in_channels, out_channels, use_skip=False):
        super().__init__()
        self.use_skip = use_skip

        # Upsample spatial dimensions (H, W) by 2, keep temporal (T) same
        self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                                      kernel_size=(3, 4, 4),
                                      stride=(1, 2, 2),
                                      padding=(1, 1, 1))
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU()

        # If using skip connections, add 1x1 conv to merge concatenated features
        # After concat: out_channels (upsampled) + out_channels (skip) = 2 * out_channels
        if use_skip:
            self.skip_conv = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, skip=None):
        # Upsample
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        # Merge skip connection if provided
        if self.use_skip and skip is not None:
            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=1)  # (B, 2*C, T, H, W)
            # Merge with 1x1 conv
            x = self.skip_conv(x)            # (B, C, T, H, W)

        return x


class VideoEncoder(nn.Module):
    """
    3D Video Encoder
    Input: (B, 3, T, H, W) - video clip
    Output: (B, latent_dim, T, H//8, W//8) - latent representation
    """

    def __init__(self, in_channels=3, latent_dim=4, base_channels=64):
        super().__init__()

        # Initial conv
        self.conv_in = Conv3DBlock(in_channels, base_channels)

        # Encoder blocks with downsampling (3 stages, downsample by 8)
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

        self.down3 = nn.Sequential(
            ResBlock3D(base_channels * 4),
            ResBlock3D(base_channels * 4),
            DownsampleBlock(base_channels * 4, base_channels * 4)
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

    def forward(self, x, return_skips=False):
        # Input: (B, 3, T, H, W)
        h0 = self.conv_in(x)       # (B, 64, T, H, W)         - spatial: H×W
        h1 = self.down1(h0)        # (B, 128, T, H//2, W//2)  - spatial: H/2×W/2
        h2 = self.down2(h1)        # (B, 256, T, H//4, W//4)  - spatial: H/4×W/4
        h3 = self.down3(h2)        # (B, 256, T, H//8, W//8)  - spatial: H/8×W/8
        h = self.mid(h3)           # (B, 256, T, H//8, W//8)
        h = self.conv_out(h)       # (B, 8, T, H//8, W//8)
        z = self.quant_conv(h)     # (B, latent_dim, T, H//8, W//8)

        if return_skips:
            # Return skip connections from each encoder level
            # Order: [h0, h1, h2] - we don't need h3 (bottleneck level)
            # Decoder upsamples then concatenates, so skip must match upsampled size:
            # - up1 upsamples to H/4×W/4, use h2 (H/4×W/4)
            # - up2 upsamples to H/2×W/2, use h1 (H/2×W/2)
            # - up3 upsamples to H×W, use h0 (H×W)
            return z, [h0, h1, h2]
        return z


class VideoDecoder(nn.Module):
    """
    3D Video Decoder
    Input: (B, latent_dim, T, H//8, W//8) - latent representation
    Output: (B, 3, T, H, W) - reconstructed video
    """

    def __init__(self, latent_dim=4, out_channels=3, base_channels=64, use_skip_connections=True):
        super().__init__()
        self.use_skip_connections = use_skip_connections

        # Post-quantization layer: latent_dim → 8 (match SD VAE architecture)
        self.post_quant_conv = nn.Conv3d(latent_dim, 8, kernel_size=1)

        # Input projection from latent space
        self.conv_in = Conv3DBlock(8, base_channels * 4)

        # Middle blocks
        self.mid = nn.Sequential(
            ResBlock3D(base_channels * 4),
            ResBlock3D(base_channels * 4),
        )

        # Decoder blocks with upsampling (3 stages, upsample by 8)
        # up1: 256 → 256 (with skip from encoder down3: 256 channels)
        self.up1_upsample = UpsampleBlock(base_channels * 4, base_channels * 4, use_skip=use_skip_connections)
        self.up1_res = nn.Sequential(
            ResBlock3D(base_channels * 4),
            ResBlock3D(base_channels * 4),
        )

        # up2: 256 → 128 (with skip from encoder down2: 256 channels)
        self.up2_upsample = UpsampleBlock(base_channels * 4, base_channels * 2, use_skip=use_skip_connections)
        self.up2_res = nn.Sequential(
            ResBlock3D(base_channels * 2),
            ResBlock3D(base_channels * 2),
        )

        # up3: 128 → 64 (with skip from encoder down1: 128 channels)
        self.up3_upsample = UpsampleBlock(base_channels * 2, base_channels, use_skip=use_skip_connections)
        self.up3_res = nn.Sequential(
            ResBlock3D(base_channels),
            ResBlock3D(base_channels),
        )

        # Output conv
        self.conv_out = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skips=None):
        # Input: (B, latent_dim, T, H//8, W//8)
        x = self.post_quant_conv(x)  # (B, 8, T, H//8, W//8)
        x = self.conv_in(x)          # (B, 256, T, H//8, W//8)
        x = self.mid(x)              # (B, 256, T, H//8, W//8)

        # Decoder with skip connections (U-Net style)
        # Skips are in order: [h0, h1, h2] from encoder (conv_in, down1, down2)
        # Spatial resolutions: h0=H×W, h1=H/2×W/2, h2=H/4×W/4
        # UpsampleBlock upsamples FIRST, then concatenates, so skip must match upsampled size
        if self.use_skip_connections and skips is not None:
            # up1: upsample 24×24 → 48×48, then concat with h2 (48×48)
            x = self.up1_upsample(x, skip=skips[2])  # (B, 256, T, H//4, W//4)
            x = self.up1_res(x)

            # up2: upsample 48×48 → 96×96, then concat with h1 (96×96)
            x = self.up2_upsample(x, skip=skips[1])  # (B, 128, T, H//2, W//2)
            x = self.up2_res(x)

            # up3: upsample 96×96 → 192×192, then concat with h0 (192×192)
            x = self.up3_upsample(x, skip=skips[0])  # (B, 64, T, H, W)
            x = self.up3_res(x)
        else:
            # No skip connections (backward compatibility)
            x = self.up1_upsample(x)     # (B, 256, T, H//4, W//4)
            x = self.up1_res(x)
            x = self.up2_upsample(x)     # (B, 128, T, H//2, W//2)
            x = self.up2_res(x)
            x = self.up3_upsample(x)     # (B, 64, T, H, W)
            x = self.up3_res(x)

        x = self.conv_out(x)      # (B, 1, T, H, W)
        x = torch.tanh(x)         # ✅ CRITICAL FIX: Bound output to [-1, 1]
        return x


class VideoVAE(nn.Module):
    """
    Complete Video VAE with Encoder and Decoder

    Compresses videos into latent space and reconstructs them.
    Input/Output format: (B, C, T, H, W)
    """

    def __init__(self, in_channels=3, latent_dim=4, base_channels=64, scaling_factor=0.18215,
                 gradient_checkpointing=False, use_skip_connections=True):
        super().__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.use_skip_connections = use_skip_connections  # Enable U-Net style skip connections

        # Use custom VideoVAE architecture
        self.encoder = VideoEncoder(in_channels, latent_dim, base_channels)
        self.decoder = VideoDecoder(latent_dim, in_channels, base_channels, use_skip_connections=use_skip_connections)

        # Scaling factor for latent space normalization
        # Standard SD VAE uses 0.18215, but this should be calculated from your dataset
        # To calculate: scale = 1 / std(latent_activations)
        self.scaling_factor = scaling_factor
        print(f"✓ Initialized custom VideoVAE")
        print(f"  Skip connections: {'Enabled (U-Net style)' if use_skip_connections else 'Disabled'}")

    def encode(self, x):
        """
        Encode video to latent space with scaling

        Args:
            x: (B, C, T, H, W) video tensor in range [-1, 1]
        Returns:
            z: (B, latent_dim, T, h, w) scaled latent tensor
        """
        z = self.encoder(x)
        # Scale latents to normalize magnitude for diffusion
        z = z * self.scaling_factor
        return z

    def decode(self, z):
        """
        Decode latent to video with unscaling

        Args:
            z: (B, latent_dim, T, h, w) scaled latent tensor
        Returns:
            x: (B, C, T, H, W) reconstructed video in range [-1, 1]
        """
        # Unscale latents before decoding
        z = z / self.scaling_factor
        return self.decoder(z)

    def encode_with_posterior(self, x):
        """
        Encode video and return posterior distribution parameters (mu, logvar).

        For VAE training from scratch.
        Splits the encoder output into mean and log variance.

        Args:
            x: (B, C, T, H, W) video tensor in range [-1, 1]

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
        Full forward pass (encode then decode) with skip connections
        Args:
            x: (B, C, T, H, W) video tensor
        Returns:
            recon: (B, C, T, H, W) reconstructed video
            z: (B, latent_dim, T, h, w) latent representation
        """
        # Use skip connections if enabled
        if self.use_skip_connections:
            # Encode with skip connections
            z, skips = self.encoder(x, return_skips=True)
            z = z * self.scaling_factor

            # Decode with skip connections
            z_unscaled = z / self.scaling_factor
            recon = self.decoder(z_unscaled, skips=skips)

            return recon, z
        else:
            # Standard forward pass (no skips)
            z = self.encode(x)
            recon = self.decode(z)
            return recon, z

    def get_latent_shape(self, video_shape):
        """Calculate latent shape given video shape"""
        B, C, T, H, W = video_shape
        return (B, self.latent_dim, T, H // 8, W // 8)

    @classmethod
    def from_pretrained(cls, model_name_or_path, method='auto', inflate_method='central',
                       strict=True, device='cpu', **kwargs):
        """
        Load VideoVAE from pretrained weights (DISABLED)

        This method is not implemented. Use custom trained VAE instead.
        Load weights manually via torch.load() and model.load_state_dict().
        """
        raise NotImplementedError(
            "from_pretrained() is not available. "
            "Pretrained VAE loading has been removed. "
            "Use custom trained VAE and load weights manually in train.py"
        )


if __name__ == "__main__":
    # Test the VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    vae = VideoVAE(in_channels=3, latent_dim=4, base_channels=64).to(device)

    # Test with random video (batch=2, channels=3, frames=16, height=256, width=256)
    video = torch.randn(2, 3, 16, 256, 256).to(device)

    print(f"Input shape: {video.shape}")

    # Encode
    latent = vae.encode(video)
    print(f"Latent shape: {latent.shape}")

    # Decode
    recon = vae.decode(latent)
    print(f"Reconstruction shape: {recon.shape}")

    # Full forward
    recon, latent = vae(video)
    print(f"Forward pass - Recon: {recon.shape}, Latent: {latent.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"Total parameters: {total_params:,}")
