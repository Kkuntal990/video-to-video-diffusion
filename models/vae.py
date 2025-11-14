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

    def __init__(self, in_channels=3, latent_dim=4, base_channels=64, scaling_factor=0.18215,
                 use_maisi=False, maisi_checkpoint=None, use_maisi_arch=False, maisi_stack_size=4,
                 use_custom_maisi=False, gradient_checkpointing=False):
        super().__init__()

        self.use_maisi = (use_maisi and maisi_checkpoint is not None)  # Only True if loading MONAI MAISI
        self.use_custom_maisi = use_custom_maisi  # Use custom MAISIVAE with 100% weight loading
        self.use_maisi_arch = use_maisi_arch  # Using MAISI-like architecture (but 2D, not 3D)
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.maisi_stack_size = maisi_stack_size  # Frames to stack as 3D volume for MAISI
        self.gradient_checkpointing = gradient_checkpointing

        if self.use_custom_maisi:
            # Use custom MAISIVAE with 100% pretrained weight loading
            print("Initializing custom MAISI VAE (100% pretrained weights)...")
            if gradient_checkpointing:
                print("  ✓ Gradient checkpointing enabled for MAISI VAE")
            from .maisi_vae import MAISIVAE

            self.maisi_vae = MAISIVAE(
                in_channels=1,  # Grayscale CT
                out_channels=1,
                latent_channels=4,
                scaling_factor=0.18215,
                use_checkpoint=gradient_checkpointing
            )

            # Load pretrained weights if checkpoint provided
            if maisi_checkpoint is not None:
                print(f"Loading MAISI checkpoint: {maisi_checkpoint}")
                stats = self.maisi_vae.load_pretrained_weights(maisi_checkpoint, strict=False)

                if stats['loaded'] == stats['total']:
                    print(f"🎉 SUCCESS! {stats['loaded']}/{stats['total']} MAISI weights loaded (100%)!")
                else:
                    print(f"⚠ Loaded {stats['loaded']}/{stats['total']} ({stats['loaded']/stats['total']*100:.1f}%) weights")
            else:
                print("⚠ No checkpoint provided - using random initialization")

            # CRITICAL: Freeze MAISI VAE weights (pretrained, should not be trained)
            for param in self.maisi_vae.parameters():
                param.requires_grad = False
            print("  ✓ MAISI VAE weights frozen (requires_grad=False)")

            self.latent_dim = 4  # MAISI uses 4 latent channels
            self.scaling_factor = 0.18215

        elif self.use_maisi:
            # Load pretrained MAISI VAE (3D volumetric) - MONAI version (partial loading)
            print("Initializing MONAI MAISI VAE with pretrained weights...")
            print(f"  Temporal stacking: {maisi_stack_size} frames per 3D volume")
            self._load_maisi_vae(maisi_checkpoint)
            self.scaling_factor = 0.18215  # Use standard scaling for compatibility
        elif use_maisi_arch:
            # Use MAISI-inspired 2D architecture (grayscale, medical imaging optimized)
            print("Initializing MAISI-inspired 2D architecture (grayscale, training from scratch)...")
            # Use our 2D+time VideoEncoder/VideoDecoder with MAISI-like depth
            # Channels progression: 64 → 128 → 256 → 256 (3-level like MAISI)
            self.encoder = VideoEncoder(in_channels, latent_dim, base_channels)
            self.decoder = VideoDecoder(latent_dim, in_channels, base_channels)
            self.scaling_factor = scaling_factor
            print(f"✓ Initialized MAISI-inspired 2D VAE (medical imaging optimized)")
            print(f"  Architecture: 3-level encoder/decoder ({base_channels}→{base_channels*2}→{base_channels*4})")
            print(f"  Grayscale input: {in_channels} channel(s)")
            print(f"  Latent dim: {latent_dim}")
            print(f"  Scaling factor: {scaling_factor}")
        else:
            # Use custom VideoVAE architecture
            self.encoder = VideoEncoder(in_channels, latent_dim, base_channels)
            self.decoder = VideoDecoder(latent_dim, in_channels, base_channels)

            # Scaling factor for latent space normalization
            # Standard SD VAE uses 0.18215, but this should be calculated from your dataset
            # To calculate: scale = 1 / std(latent_activations)
            self.scaling_factor = scaling_factor

    def _load_maisi_vae(self, checkpoint_path):
        """
        Load NVIDIA MAISI VAE for medical CT imaging

        Args:
            checkpoint_path: Path to MAISI VAE checkpoint
        """
        try:
            from monai.networks.nets import AutoencoderKL
            print("Loading MAISI VAE from MONAI...")

            # MAISI VAE configuration (3D AutoencoderKL)
            # Based on actual MAISI model: 3-level architecture 64→128→256
            self.maisi_vae = AutoencoderKL(
                spatial_dims=3,
                in_channels=1,  # Grayscale CT
                out_channels=1,
                channels=(64, 128, 256),  # 3-level progression: 64→128→256
                latent_channels=4,  # MAISI uses 4 latent channels
                num_res_blocks=(2, 2, 2),  # 2 residual blocks per level
                attention_levels=(False, False, True),  # Attention at deepest level only
                with_encoder_nonlocal_attn=True,
                with_decoder_nonlocal_attn=True,
            )

            # Update latent_dim to match MAISI
            self.latent_dim = 4

            # Load checkpoint if provided
            if checkpoint_path is not None:
                from pathlib import Path
                checkpoint_path = Path(checkpoint_path)

                if not checkpoint_path.exists():
                    print(f"Warning: MAISI checkpoint not found at {checkpoint_path}")
                    print("Using randomly initialized MAISI architecture (will need training)")
                else:
                    print(f"Loading MAISI weights from {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')

                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        elif 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint

                    # Load state dict with manual filtering for incompatible shapes
                    model_state = self.maisi_vae.state_dict()
                    filtered_state = {}
                    incompatible_keys = []

                    for key, param in state_dict.items():
                        if key in model_state:
                            if param.shape == model_state[key].shape:
                                filtered_state[key] = param
                            else:
                                incompatible_keys.append(
                                    f"{key}: checkpoint{list(param.shape)} vs model{list(model_state[key].shape)}"
                                )
                        # else: key not in model (will be handled by load_state_dict)

                    # Load only compatible weights
                    missing_keys, unexpected_keys = self.maisi_vae.load_state_dict(filtered_state, strict=False)

                    # Report loading status
                    loaded_keys = len(filtered_state)
                    total_model_keys = len(model_state)

                    print(f"✓ Loaded {loaded_keys}/{total_model_keys} compatible weights from MAISI checkpoint")
                    if incompatible_keys:
                        print(f"  ⚠ Skipped {len(incompatible_keys)} incompatible keys (shape mismatch)")
                        if len(incompatible_keys) <= 5:
                            for key_info in incompatible_keys:
                                print(f"    - {key_info}")
                    if missing_keys:
                        print(f"  ℹ {len(missing_keys)} keys initialized randomly (not in checkpoint)")
                    if unexpected_keys:
                        print(f"  ℹ {len(unexpected_keys)} unexpected keys in checkpoint (ignored)")

        except ImportError:
            raise ImportError(
                "MONAI is required for MAISI VAE. Install with: pip install 'monai[all]>=1.4.0'"
            )

    def _init_maisi_architecture(self):
        """
        Initialize MAISI-like architecture without loading pretrained weights.
        This creates the same medical imaging optimized architecture but trains from scratch.
        """
        try:
            from monai.networks.nets import AutoencoderKL

            # MAISI VAE configuration (3D AutoencoderKL for medical imaging)
            # - Grayscale input (1 channel)
            # - 3D convolutions (process each frame as D=1 depth)
            # - 3-level encoder/decoder: channels=(64, 128, 256)
            # - Latent channels: 4 (standard for diffusion)
            self.maisi_vae = AutoencoderKL(
                spatial_dims=3,
                in_channels=1,  # Grayscale CT scans
                out_channels=1,
                channels=(64, 128, 256),  # 3-level progression (MAISI-like)
                latent_channels=4,  # Standard for diffusion models
                num_res_blocks=(2, 2, 2),  # 2 residual blocks per level
                attention_levels=(False, False, True),  # Attention only at deepest level
                with_encoder_nonlocal_attn=True,  # Global attention in encoder
                with_decoder_nonlocal_attn=True,  # Global attention in decoder
            )

            self.latent_dim = 4
            print(f"✓ Initialized MAISI-like architecture (grayscale, medical imaging optimized)")
            print(f"  Architecture: 3-level VAE with channels (64→128→256)")
            print(f"  Latent dim: {self.latent_dim}")
            print(f"  Training from scratch (no pretrained weights)")

        except ImportError:
            raise ImportError(
                "MONAI is required for MAISI architecture. Install with: pip install 'monai[all]>=1.4.0'"
            )

    def encode(self, x, chunk_size=32):
        """
        Encode video to latent space with scaling

        For large volumes (D > chunk_size), uses chunked encoding to reduce memory usage.

        Args:
            x: (B, C, T, H, W) video tensor in range [-1, 1]
               C=1 for MAISI (grayscale), C=3 for custom VAE (RGB)
            chunk_size: Process volumes in chunks of this many slices (default: 32)
                       Only used for custom MAISI VAE with large volumes
        Returns:
            z: (B, latent_dim, T, h, w) scaled latent tensor
        """
        if self.use_custom_maisi:
            # Use custom MAISIVAE with chunked encoding for large volumes
            # Input: (B, C, D, H, W) where C=1 (grayscale), D is depth (variable)
            # Output: (B, 4, D/4, H/8, W/8) - already scaled by MAISIVAE

            # CRITICAL: Convert from [-1, 1] to [0, 1] for MAISI VAE
            # MAISI was trained on HU [-1000,1000] normalized to [0,1] (confirmed from paper)
            x = (x + 1.0) / 2.0

            # For large volumes, process in chunks to reduce memory usage
            B, C, D, H, W = x.shape
            if D > chunk_size:
                return self._encode_chunked(x, chunk_size)
            else:
                return self.maisi_vae.encode(x)

        elif self.use_maisi:
            return self._encode_with_maisi(x)
        else:
            z = self.encoder(x)
            # Scale latents to normalize magnitude for diffusion
            z = z * self.scaling_factor
            return z

    def decode(self, z, chunk_size=8):
        """
        Decode latent to video with unscaling

        For large volumes (d > chunk_size), uses chunked decoding to reduce memory usage.

        Args:
            z: (B, latent_dim, T, h, w) scaled latent tensor
            chunk_size: Process latents in chunks (default: 8 latent slices = 32 output slices)
                       Only used for custom MAISI VAE with large volumes
        Returns:
            x: (B, C, T, H, W) reconstructed video in range [-1, 1]
               C=1 for MAISI (grayscale), C=3 for custom VAE (RGB)
        """
        if self.use_custom_maisi:
            # Use custom MAISIVAE with chunked decoding for large volumes
            # Input: (B, 4, d, h, w) - scaled latent (d = D/4 where D is original depth)
            # Output: (B, 1, D, H, W) - grayscale CT

            # For large latents, process in chunks to reduce memory usage
            B, C, d, h, w = z.shape
            if d > chunk_size:
                x_recon = self._decode_chunked(z, chunk_size)
            else:
                x_recon = self.maisi_vae.decode(z)

            # CRITICAL: Convert from [0, 1] to [-1, 1] for diffusion
            # MAISI outputs [0,1] range (confirmed from paper), but our diffusion uses [-1, 1]

            # DEBUG: Log MAISI output range before conversion
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"MAISI VAE decode output range BEFORE conversion: [{x_recon.min():.3f}, {x_recon.max():.3f}]")

            x_recon = x_recon * 2.0 - 1.0

            logger.info(f"MAISI VAE decode output range AFTER conversion: [{x_recon.min():.3f}, {x_recon.max():.3f}]")

            return x_recon

        elif self.use_maisi:
            return self._decode_with_maisi(z)
        else:
            # Unscale latents before decoding
            z = z / self.scaling_factor
            return self.decoder(z)

    def _encode_chunked(self, x, chunk_size=32):
        """
        Encode large volume in chunks to reduce memory usage

        Processes the volume in non-overlapping chunks along the depth dimension,
        then concatenates the latent representations.

        Args:
            x: (B, C, D, H, W) input volume (C=1 for grayscale CT)
            chunk_size: Number of slices to process at once (default: 32)

        Returns:
            z: (B, 4, d, h, w) latent representation (d = D/4)
        """
        import logging
        logger = logging.getLogger(__name__)

        B, C, D, H, W = x.shape

        # Note: Conversion from [-1, 1] to [0, 1] is done in encode() before calling this
        # Do NOT convert again here to avoid double conversion

        chunks = []

        # Log chunking info (only once per batch)
        if not hasattr(self, '_logged_chunking'):
            num_chunks = (D + chunk_size - 1) // chunk_size
            logger.info(f"Using chunked encoding: {D} slices → {num_chunks} chunks of max {chunk_size} slices")
            logger.info(f"  Memory savings: ~{100 * (1 - chunk_size/D):.0f}% peak reduction vs full volume")
            self._logged_chunking = True

        # Process volume in non-overlapping chunks
        for i in range(0, D, chunk_size):
            end_idx = min(i + chunk_size, D)
            chunk = x[:, :, i:end_idx, :, :]  # (B, C, chunk_size, H, W)

            # Encode this chunk
            with torch.cuda.amp.autocast(enabled=False):  # Ensure consistent precision
                z_chunk = self.maisi_vae.encode(chunk)  # (B, 4, chunk_size/4, H/8, W/8)

            chunks.append(z_chunk)

        # Concatenate all chunks along depth dimension
        z = torch.cat(chunks, dim=2)  # (B, 4, D/4, H/8, W/8)

        return z

    def _decode_chunked(self, z, chunk_size=8):
        """
        Decode large latent in chunks to reduce memory usage

        Processes the latent in chunks along the depth dimension,
        then concatenates the reconstructed volumes.

        Args:
            z: (B, 4, d, h, w) latent representation (d = D/4 where D is original depth)
            chunk_size: Number of latent slices to process at once (default: 8)
                       Note: 8 latent slices → 32 output slices (4x upsampling)

        Returns:
            x: (B, 1, D, H, W) reconstructed volume
        """
        import logging
        logger = logging.getLogger(__name__)

        B, C, d, h, w = z.shape
        chunks = []

        # Log chunking info (only once per batch)
        if not hasattr(self, '_logged_decode_chunking'):
            logger.info(f"Using chunked decoding: {d} latent slices → {(d + chunk_size - 1) // chunk_size} chunks of {chunk_size} slices")
            self._logged_decode_chunking = True

        # Process latent in non-overlapping chunks
        for i in range(0, d, chunk_size):
            end_idx = min(i + chunk_size, d)
            chunk = z[:, :, i:end_idx, :, :]  # (B, 4, chunk_size, h, w)

            # Decode this chunk
            with torch.cuda.amp.autocast(enabled=False):  # Ensure consistent precision
                x_chunk = self.maisi_vae.decode(chunk)  # (B, 1, chunk_size*4, H, W)

            chunks.append(x_chunk)

        # Concatenate all chunks along depth dimension
        x = torch.cat(chunks, dim=2)  # (B, 1, D, H, W)

        # Note: Conversion from [0, 1] to [-1, 1] is done in decode() after calling this
        # Do NOT convert again here to avoid double conversion

        return x

    def _encode_with_maisi(self, x):
        """
        Encode video using MAISI VAE with temporal frame stacking

        Strategy: Group consecutive frames into 3D volumes to satisfy MAISI's
        3D convolution requirements (kernel_size=3×3×3 needs depth >= 3)

        Args:
            x: (B, C, T, H, W) video tensor, C should be 1 for grayscale
        Returns:
            z: (B, latent_dim, T_latent, H//8, W//8) latent tensor
               where T_latent = T // stack_size
        """
        B, C, T, H, W = x.shape

        # Ensure single channel (MAISI expects grayscale)
        if C != 1:
            raise ValueError(f"MAISI VAE expects 1-channel grayscale input, got {C} channels")

        stack_size = self.maisi_stack_size  # e.g., 4 frames

        # Pad temporal dimension if not divisible by stack_size
        T_original = T
        if T % stack_size != 0:
            pad_size = stack_size - (T % stack_size)
            # Pad by repeating last frame
            last_frames = x[:, :, -1:, :, :].repeat(1, 1, pad_size, 1, 1)
            x = torch.cat([x, last_frames], dim=2)
            T = x.shape[2]

        # Reshape: (B, C, T, H, W) → (B*num_volumes, C, stack_size, H, W)
        num_volumes = T // stack_size
        x_volumes = x.reshape(B, C, num_volumes, stack_size, H, W)
        x_volumes = x_volumes.permute(0, 2, 1, 3, 4, 5)  # (B, num_volumes, C, stack_size, H, W)
        x_volumes = x_volumes.reshape(B * num_volumes, C, stack_size, H, W)

        # Encode all volumes through MAISI VAE
        z_dist = self.maisi_vae.encode(x_volumes)
        if hasattr(z_dist, 'sample'):
            z_volumes = z_dist.sample()
        elif isinstance(z_dist, tuple):
            z_volumes = z_dist[0]
        else:
            z_volumes = z_dist

        # z_volumes shape: (B*num_volumes, latent_channels, D_latent, h, w)
        # MAISI compresses depth too, typically D_latent = stack_size // 8 = 1 for stack_size=4
        # Reshape back: (B*num_volumes, latent_channels, D_latent, h, w) → (B, latent_channels, num_volumes*D_latent, h, w)
        latent_channels = z_volumes.shape[1]
        D_latent = z_volumes.shape[2]
        h, w = z_volumes.shape[3], z_volumes.shape[4]

        z_volumes = z_volumes.reshape(B, num_volumes, latent_channels, D_latent, h, w)
        z_volumes = z_volumes.permute(0, 2, 1, 3, 4, 5)  # (B, latent_channels, num_volumes, D_latent, h, w)
        z = z_volumes.reshape(B, latent_channels, num_volumes * D_latent, h, w)

        # If we padded, crop back to original temporal size (in latent space)
        if T_original != T:
            # Calculate original latent temporal size
            T_latent_original = (T_original + stack_size - 1) // stack_size  # ceiling division
            T_latent_original = min(T_latent_original, z.shape[2])
            z = z[:, :, :T_latent_original, :, :]

        # Apply scaling factor
        z = z * self.scaling_factor

        return z

    def _decode_with_maisi(self, z):
        """
        Decode latent using MAISI VAE with temporal frame unstacking

        This is the inverse of _encode_with_maisi()

        Args:
            z: (B, latent_dim, T_latent, h, w) latent tensor
        Returns:
            x: (B, 1, T, H, W) reconstructed video tensor
               where T ≈ T_latent * stack_size
        """
        # Unscale latents
        z = z / self.scaling_factor

        B, latent_channels, T_latent, h, w = z.shape
        stack_size = self.maisi_stack_size

        # MAISI compresses depth, so we need to infer original structure
        # Assume D_latent = 1 for stack_size=4 (typical 8× depth compression)
        # So num_volumes = T_latent / D_latent = T_latent

        # For now, assume D_latent = 1 (MAISI heavily compresses depth dimension)
        D_latent = 1
        num_volumes = T_latent // D_latent

        # Reshape to volumes: (B, latent_channels, num_volumes, D_latent, h, w)
        z_volumes = z.reshape(B, latent_channels, num_volumes, D_latent, h, w)
        z_volumes = z_volumes.permute(0, 2, 1, 3, 4, 5)  # (B, num_volumes, latent_channels, D_latent, h, w)
        z_volumes = z_volumes.reshape(B * num_volumes, latent_channels, D_latent, h, w)

        # Decode all volumes through MAISI VAE
        x_volumes = self.maisi_vae.decode(z_volumes)  # (B*num_volumes, 1, stack_size, H, W)

        # x_volumes shape: (B*num_volumes, C_out, D_out, H, W)
        # where D_out = stack_size (MAISI upsamples depth back to original)
        C_out = x_volumes.shape[1]
        D_out = x_volumes.shape[2]
        H, W = x_volumes.shape[3], x_volumes.shape[4]

        # Reshape back: (B*num_volumes, C_out, D_out, H, W) → (B, C_out, T, H, W)
        x_volumes = x_volumes.reshape(B, num_volumes, C_out, D_out, H, W)
        x_volumes = x_volumes.permute(0, 2, 1, 3, 4, 5)  # (B, C_out, num_volumes, D_out, H, W)
        x = x_volumes.reshape(B, C_out, num_volumes * D_out, H, W)

        return x

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
        if self.use_custom_maisi or self.use_maisi:
            raise NotImplementedError(
                "encode_with_posterior is only for training custom VAE from scratch. "
                "MAISI VAE is pretrained and frozen."
            )

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
