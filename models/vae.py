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

    def forward(self, x):
        # Input: (B, 3, T, H, W)
        x = self.conv_in(x)       # (B, 64, T, H, W)
        x = self.down1(x)         # (B, 128, T, H//2, W//2)
        x = self.down2(x)         # (B, 256, T, H//4, W//4)
        x = self.down3(x)         # (B, 256, T, H//8, W//8)
        x = self.mid(x)           # (B, 256, T, H//8, W//8)
        x = self.conv_out(x)      # (B, 8, T, H//8, W//8)
        x = self.quant_conv(x)    # (B, latent_dim, T, H//8, W//8)
        return x


class VideoDecoder(nn.Module):
    """
    3D Video Decoder
    Input: (B, latent_dim, T, H//8, W//8) - latent representation
    Output: (B, 3, T, H, W) - reconstructed video
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

        # Decoder blocks with upsampling (3 stages, upsample by 8)
        self.up1 = nn.Sequential(
            UpsampleBlock(base_channels * 4, base_channels * 4),
            ResBlock3D(base_channels * 4),
            ResBlock3D(base_channels * 4),
        )

        self.up2 = nn.Sequential(
            UpsampleBlock(base_channels * 4, base_channels * 2),
            ResBlock3D(base_channels * 2),
            ResBlock3D(base_channels * 2),
        )

        self.up3 = nn.Sequential(
            UpsampleBlock(base_channels * 2, base_channels),
            ResBlock3D(base_channels),
            ResBlock3D(base_channels),
        )

        # Output conv
        self.conv_out = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Input: (B, latent_dim, T, H//8, W//8)
        x = self.post_quant_conv(x)  # (B, 8, T, H//8, W//8)
        x = self.conv_in(x)       # (B, 256, T, H//8, W//8)
        x = self.mid(x)           # (B, 256, T, H//8, W//8)
        x = self.up1(x)           # (B, 256, T, H//4, W//4)
        x = self.up2(x)           # (B, 128, T, H//2, W//2)
        x = self.up3(x)           # (B, 64, T, H, W)
        x = self.conv_out(x)      # (B, 3, T, H, W)
        return x


class VideoVAE(nn.Module):
    """
    Complete Video VAE with Encoder and Decoder

    Compresses videos into latent space and reconstructs them.
    Input/Output format: (B, C, T, H, W)
    """

    def __init__(self, in_channels=3, latent_dim=4, base_channels=64):
        super().__init__()
        self.encoder = VideoEncoder(in_channels, latent_dim, base_channels)
        self.decoder = VideoDecoder(latent_dim, in_channels, base_channels)
        self.latent_dim = latent_dim

    def encode(self, x):
        """
        Encode video to latent space
        Args:
            x: (B, C, T, H, W) video tensor
        Returns:
            z: (B, latent_dim, T, h, w) latent tensor
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent to video
        Args:
            z: (B, latent_dim, T, h, w) latent tensor
        Returns:
            x: (B, C, T, H, W) reconstructed video
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Full forward pass (encode then decode)
        Args:
            x: (B, C, T, H, W) video tensor
        Returns:
            recon: (B, C, T, H, W) reconstructed video
            z: (B, latent_dim, T, h, w) latent representation
        """
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
        Load VideoVAE from pretrained weights

        Supports:
        - Open-Sora VAE (3D, direct loading)
        - CogVideoX VAE (3D, direct loading)
        - Stable Diffusion VAE (2D, inflate to 3D)
        - HunyuanVideo VAE (3D, direct loading)
        - Local checkpoint files

        Args:
            model_name_or_path: HuggingFace model name or local path
            method: 'auto', 'opensora', 'cogvideox', 'sd', or 'local'
            inflate_method: For SD VAE, inflation method ('central', 'replicate', 'average')
            strict: Whether to strictly enforce state dict matching
            device: Device to load model on
            **kwargs: Additional arguments for model initialization

        Returns:
            VideoVAE instance with loaded weights

        Examples:
            >>> # Load Open-Sora VAE (recommended)
            >>> vae = VideoVAE.from_pretrained('hpcai-tech/OpenSora-VAE-v1.2')

            >>> # Load CogVideoX VAE
            >>> vae = VideoVAE.from_pretrained('THUDM/CogVideoX-5b', method='cogvideox')

            >>> # Load and inflate SD VAE
            >>> vae = VideoVAE.from_pretrained('stabilityai/sd-vae-ft-mse', method='sd')

            >>> # Load from local checkpoint
            >>> vae = VideoVAE.from_pretrained('./checkpoints/vae.pt', method='local')
        """
        from pathlib import Path
        from ..utils.pretrained import (
            load_pretrained_opensora_vae,
            load_pretrained_cogvideox_vae,
            load_pretrained_sd_vae,
            map_sd_vae_to_video_vae,
            load_state_dict_from_file
        )

        # Auto-detect method
        if method == 'auto':
            model_lower = str(model_name_or_path).lower()
            if 'opensora' in model_lower or 'hpcai' in model_lower:
                method = 'opensora'
            elif 'cogvideo' in model_lower or 'thudm' in model_lower:
                method = 'cogvideox'
            elif 'sd' in model_lower or 'stable' in model_lower or 'stability' in model_lower:
                method = 'sd'
            elif Path(model_name_or_path).exists():
                method = 'local'
            else:
                raise ValueError(f"Cannot auto-detect method for: {model_name_or_path}")

        print(f"Loading VideoVAE using method: {method}")

        # Load state dict based on method
        if method == 'opensora':
            state_dict = load_pretrained_opensora_vae(model_name_or_path)
        elif method == 'cogvideox':
            state_dict = load_pretrained_cogvideox_vae(model_name_or_path)
        elif method == 'sd':
            # Load SD VAE and inflate 2D->3D
            sd_state_dict = load_pretrained_sd_vae(model_name_or_path)
            state_dict = map_sd_vae_to_video_vae(sd_state_dict, inflate_method=inflate_method)
            print(f"Inflated 2D SD VAE to 3D using method: {inflate_method}")
        elif method == 'local':
            state_dict = load_state_dict_from_file(model_name_or_path)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create model instance
        # Try to infer parameters from state dict
        if 'encoder.conv_in.conv.weight' in state_dict:
            in_channels = state_dict['encoder.conv_in.conv.weight'].shape[1]
        else:
            in_channels = kwargs.get('in_channels', 3)

        if 'encoder.conv_out.weight' in state_dict:
            latent_dim = state_dict['encoder.conv_out.weight'].shape[0]
        else:
            latent_dim = kwargs.get('latent_dim', 4)

        if 'encoder.conv_in.conv.weight' in state_dict:
            base_channels = state_dict['encoder.conv_in.conv.weight'].shape[0]
        else:
            base_channels = kwargs.get('base_channels', 64)

        # Create model
        model = cls(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels
        )

        # Load weights
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

            if missing_keys:
                print(f"Warning: Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys: {unexpected_keys}")

            print(f"✓ Successfully loaded pretrained VAE weights")

        except Exception as e:
            print(f"Error loading state dict: {e}")
            if strict:
                raise
            print("Trying non-strict loading...")
            model.load_state_dict(state_dict, strict=False)

        model = model.to(device)
        model.eval()  # Set to eval mode by default

        return model


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
