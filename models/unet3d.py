"""
3D U-Net Denoiser for Video Diffusion

Predicts noise in latent space conditioned on:
- Noisy latent z_t
- Input video conditioning c
- Timestep t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """Project timestep to embedding dimension"""

    def __init__(self, dim, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, t):
        return self.time_mlp(t)


class Conv3DBlock(nn.Module):
    """3D Convolutional block with GroupNorm and SiLU"""

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
    """
    3D Residual block with time embedding injection
    """

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = Conv3DBlock(in_channels, out_channels)

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels)
        )

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x, time_emb):
        residual = self.residual_conv(x)

        # First conv
        x = self.conv1(x)

        # Add time embedding (broadcast across spatial/temporal dims)
        time_emb = self.time_mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
        x = x + time_emb

        # Second conv
        x = self.conv2(x)

        # Residual
        x = x + residual
        x = self.act(x)
        return x


class TemporalAttention(nn.Module):
    """
    Temporal self-attention along the time dimension
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        residual = x

        x = self.norm(x)

        # Get Q, K, V
        qkv = self.qkv(x)  # (B, 3C, T, H, W)
        qkv = rearrange(qkv, 'b (three c) t h w -> three b c t h w', three=3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Reshape for attention along temporal dimension
        q = rearrange(q, 'b (head c) t h w -> (b h w) head t c', head=self.num_heads)
        k = rearrange(k, 'b (head c) t h w -> (b h w) head t c', head=self.num_heads)
        v = rearrange(v, 'b (head c) t h w -> (b h w) head t c', head=self.num_heads)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhqc,bhkc->bhqk', q, k) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.einsum('bhqk,bhvc->bhqc', attn, v)

        # Reshape back
        out = rearrange(out, '(b h w) head t c -> b (head c) t h w',
                       b=B, h=H, w=W, head=self.num_heads)

        # Project
        out = self.proj_out(out)

        return out + residual


class Downsample3D(nn.Module):
    """Downsample spatial dimensions by 2"""

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Conv3d(in_channels, out_channels,
                             kernel_size=(3, 4, 4),
                             stride=(1, 2, 2),
                             padding=(1, 1, 1))

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """Upsample spatial dimensions by 2"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(channels, channels,
                                      kernel_size=(3, 4, 4),
                                      stride=(1, 2, 2),
                                      padding=(1, 1, 1))

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for video diffusion denoising

    Input:
        - x: noisy latent (B, latent_dim, T, h, w)
        - c: conditioning from input video (B, latent_dim, T, h, w)
        - t: timestep (B,)

    Output:
        - noise prediction (B, latent_dim, T, h, w)
    """

    def __init__(self, latent_dim=4, model_channels=128, num_res_blocks=2,
                 attention_levels=[1, 2], channel_mult=(1, 2, 4, 4),
                 num_heads=4, time_embed_dim=512):
        super().__init__()

        self.latent_dim = latent_dim
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.channel_mult = channel_mult
        self.num_levels = len(channel_mult)

        # Time embedding
        self.time_embed = TimeEmbedding(model_channels, time_embed_dim)

        # Input projection (concatenate noisy latent + conditioning)
        self.conv_in = nn.Conv3d(latent_dim * 2, model_channels, kernel_size=3, padding=1)

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        ch = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            # Group all blocks for this level
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                layers = [ResBlock3D(ch, out_ch, time_embed_dim)]

                # Add temporal attention at specified levels
                if level in attention_levels:
                    layers.append(TemporalAttention(out_ch, num_heads))

                level_blocks.append(nn.ModuleList(layers))
                ch = out_ch

            self.down_blocks.append(level_blocks)

            # Downsample (except last level) - keeps same channels
            if level < self.num_levels - 1:
                self.down_samples.append(Downsample3D(ch, ch))
            else:
                self.down_samples.append(nn.Identity())

        # Middle blocks
        self.mid_block1 = ResBlock3D(ch, ch, time_embed_dim)
        self.mid_attn = TemporalAttention(ch, num_heads)
        self.mid_block2 = ResBlock3D(ch, ch, time_embed_dim)

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult

            # Group all blocks for this level
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                # First block in level gets concatenated skip connection
                if i == 0:
                    in_ch = ch + (model_channels * channel_mult[self.num_levels - 1 - level])
                else:
                    in_ch = ch

                layers = [ResBlock3D(in_ch, out_ch, time_embed_dim)]

                # Add temporal attention at specified levels
                if (self.num_levels - 1 - level) in attention_levels:
                    layers.append(TemporalAttention(out_ch, num_heads))

                level_blocks.append(nn.ModuleList(layers))
                ch = out_ch

            self.up_blocks.append(level_blocks)

            # Upsample (except last level)
            if level < self.num_levels - 1:
                self.up_samples.append(Upsample3D(ch))
            else:
                self.up_samples.append(nn.Identity())

        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=ch),
            nn.SiLU(),
            nn.Conv3d(ch, latent_dim, kernel_size=3, padding=1)
        )

    def forward(self, x, t, c):
        """
        Args:
            x: noisy latent (B, latent_dim, T, h, w)
            t: timestep (B,)
            c: conditioning (B, latent_dim, T, h, w)

        Returns:
            noise prediction (B, latent_dim, T, h, w)
        """

        # Time embedding
        time_emb = self.time_embed(t)

        # Concatenate noisy latent with conditioning
        x = torch.cat([x, c], dim=1)  # (B, latent_dim*2, T, h, w)
        x = self.conv_in(x)  # (B, model_channels, T, h, w)

        # Encoder
        skip_connections = []
        for i, (level_blocks, downsample) in enumerate(zip(self.down_blocks, self.down_samples)):
            # Process all blocks at this level
            for block_list in level_blocks:
                for layer in block_list:
                    if isinstance(layer, ResBlock3D):
                        x = layer(x, time_emb)
                    else:  # TemporalAttention
                        x = layer(x)

            skip_connections.append(x)
            x = downsample(x)

        # Middle
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        # Decoder
        for i, (level_blocks, upsample) in enumerate(zip(self.up_blocks, self.up_samples)):
            # Process all blocks at this level
            for j, block_list in enumerate(level_blocks):
                # Concatenate skip connection only before first block of the level
                if j == 0:
                    skip = skip_connections.pop()
                    x = torch.cat([x, skip], dim=1)

                for layer in block_list:
                    if isinstance(layer, ResBlock3D):
                        x = layer(x, time_emb)
                    else:  # TemporalAttention
                        x = layer(x)

            x = upsample(x)

        # Output
        x = self.conv_out(x)
        return x


if __name__ == "__main__":
    # Test the U-Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    unet = UNet3D(latent_dim=4, model_channels=128, num_res_blocks=2,
                  attention_levels=[1, 2], channel_mult=(1, 2, 4, 4),
                  num_heads=4).to(device)

    # Test inputs
    B, T, h, w = 2, 16, 32, 32
    noisy_latent = torch.randn(B, 4, T, h, w).to(device)
    conditioning = torch.randn(B, 4, T, h, w).to(device)
    timesteps = torch.randint(0, 1000, (B,)).to(device)

    print(f"Input shape: {noisy_latent.shape}")
    print(f"Conditioning shape: {conditioning.shape}")
    print(f"Timesteps shape: {timesteps.shape}")

    # Forward pass
    noise_pred = unet(noisy_latent, timesteps, conditioning)
    print(f"Output shape: {noise_pred.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Total parameters: {total_params:,}")
