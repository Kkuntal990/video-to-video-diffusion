"""
Complete Video-to-Video Diffusion Model

Integrates VAE, U-Net, and Diffusion process into a single model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae import VideoVAE
from .unet3d import UNet3D
from .diffusion import GaussianDiffusion


class VideoToVideoDiffusion(nn.Module):
    """
    Complete Video-to-Video Diffusion Model

    Architecture:
        1. VAE encodes input and ground truth videos to latent space
        2. Diffusion process adds noise to ground truth latent
        3. U-Net predicts noise conditioned on input latent
        4. VAE decoder reconstructs video from denoised latent

    Training:
        - Input: v_in (input video), v_gt (ground truth video)
        - Encode: z_in = VAE.encode(v_in), z_gt = VAE.encode(v_gt)
        - Noise: z_t = diffusion.q_sample(z_gt, t)
        - Predict: noise_pred = UNet(z_t, t, z_in)
        - Loss: MSE(noise_pred, noise)

    Inference:
        - Input: v_in (input video)
        - Encode: z_in = VAE.encode(v_in)
        - Sample: z_0 = sampler.sample(z_T, z_in)
        - Decode: v_out = VAE.decode(z_0)
    """

    def __init__(self, config, load_pretrained=False):
        super().__init__()

        # Check if pretrained weights should be loaded
        pretrained_config = config.get('pretrained', {})
        use_pretrained = pretrained_config.get('use_pretrained', False) or load_pretrained

        # Get gradient checkpointing flag from hardware section or top-level
        gradient_checkpointing = config.get('hardware', {}).get('gradient_checkpointing',
                                                                config.get('gradient_checkpointing', False))

        # VAE for encoding/decoding videos
        if use_pretrained and pretrained_config.get('vae', {}).get('enabled', False):
            vae_config = pretrained_config['vae']

            # Check if using custom MAISI VAE (100% weight loading)
            if vae_config.get('use_custom_maisi', False):
                print(f"Loading Custom MAISI VAE (100% pretrained weights)...")
                self.vae = VideoVAE(
                    in_channels=config.get('in_channels', 1),  # Grayscale for medical
                    latent_dim=4,  # MAISI uses 4 latent channels
                    use_custom_maisi=True,
                    maisi_checkpoint=vae_config.get('checkpoint_path', None),
                    gradient_checkpointing=gradient_checkpointing
                )
            # Check if using MAISI VAE (medical imaging specific) - MONAI version
            elif vae_config.get('use_maisi', False):
                print(f"Loading MONAI MAISI VAE for medical CT imaging...")
                stack_size = vae_config.get('stack_size', 4)  # Frames per 3D volume
                self.vae = VideoVAE(
                    in_channels=config.get('in_channels', 1),  # Grayscale for medical
                    latent_dim=4,  # MAISI uses 4 latent channels (verified from checkpoint)
                    use_maisi=True,
                    maisi_checkpoint=vae_config.get('checkpoint_path', None),
                    maisi_stack_size=stack_size,
                    gradient_checkpointing=gradient_checkpointing
                )
            else:
                # Load other pretrained VAE (SD VAE, etc.)
                print(f"Loading pretrained VAE from {vae_config['model_name']}...")
                self.vae = VideoVAE.from_pretrained(
                    vae_config['model_name'],
                    method=vae_config.get('method', 'auto'),
                    inflate_method=vae_config.get('inflate_method', 'central'),
                    device='cpu'  # Will be moved to correct device later
                )
        else:
            # Train VAE from scratch
            # Check if using MAISI-like architecture (grayscale + specific channel config)
            model_config = config.get('model', config)  # Handle nested model key
            in_channels = model_config.get('in_channels', config.get('in_channels', 3))
            base_channels = model_config.get('vae_base_channels', config.get('vae_base_channels', 64))
            latent_dim = model_config.get('latent_dim', config.get('latent_dim', 4))
            scaling_factor = model_config.get('vae_scaling_factor', config.get('vae_scaling_factor', 0.18215))

            use_maisi_arch = (in_channels == 1 and base_channels == 64)  # MAISI pattern

            self.vae = VideoVAE(
                in_channels=in_channels,
                latent_dim=latent_dim,
                base_channels=base_channels,
                scaling_factor=scaling_factor,
                use_maisi_arch=use_maisi_arch,  # Use MAISI architecture without pretrained weights
                gradient_checkpointing=gradient_checkpointing
            )

        # U-Net for denoising
        # Use latent_dim from VAE (3 for MAISI, 4 for custom)
        actual_latent_dim = self.vae.latent_dim
        self.unet = UNet3D(
            latent_dim=actual_latent_dim,
            model_channels=config.get('unet_model_channels', 128),
            num_res_blocks=config.get('unet_num_res_blocks', 2),
            attention_levels=config.get('unet_attention_levels', [1, 2]),
            channel_mult=tuple(config.get('unet_channel_mult', [1, 2, 4, 4])),
            num_heads=config.get('unet_num_heads', 4),
            time_embed_dim=config.get('unet_time_embed_dim', 512),
            use_checkpoint=gradient_checkpointing  # Enable gradient checkpointing
        )

        # Diffusion process
        self.diffusion = GaussianDiffusion(
            noise_schedule=config.get('noise_schedule', 'cosine'),
            timesteps=config.get('diffusion_timesteps', 1000),
            beta_start=config.get('beta_start', 0.0001),
            beta_end=config.get('beta_end', 0.02)
        )

        self.config = config
        self.use_pretrained = use_pretrained

    def encode_videos(self, v_in, v_gt=None):
        """
        Encode videos to latent space

        Args:
            v_in: input video (B, 3, T, H, W)
            v_gt: ground truth video (B, 3, T, H, W), optional

        Returns:
            z_in: input latent (B, latent_dim, T, h, w)
            z_gt: ground truth latent (B, latent_dim, T, h, w), if provided
        """
        with torch.no_grad():
            z_in = self.vae.encode(v_in)

        if v_gt is not None:
            z_gt = self.vae.encode(v_gt)
            return z_in, z_gt
        else:
            return z_in

    def decode_latent(self, z):
        """
        Decode latent to video

        Args:
            z: latent (B, latent_dim, T, h, w)

        Returns:
            v: video (B, 3, T, H, W)
        """
        return self.vae.decode(z)

    def forward(self, v_in, v_gt, mask=None):
        """
        Forward pass for training

        Args:
            v_in: input video/volume (B, C, T_in, H, W)
                  For slice interpolation: thick slices (T_in < T_gt)
            v_gt: ground truth video/volume (B, C, T_gt, H, W)
                  For slice interpolation: thin slices (T_gt > T_in)
            mask: optional padding mask (B, C, T_gt) where 1=real data, 0=padding
                  CRITICAL for variable-depth batches to avoid learning on padding

        Returns:
            loss: training loss (scalar)
            metrics: dict with additional metrics
        """
        # Encode videos to latent space
        z_in = self.vae.encode(v_in)  # Conditioning latent
        z_gt = self.vae.encode(v_gt)  # Target latent for diffusion

        # For slice interpolation: z_in may have different depth than z_gt
        # Example: z_in = (B, 4, 12, 64, 64) for thick @ 5.0mm
        #          z_gt = (B, 4, 62, 64, 64) for thin @ 1.0mm
        # Need to upsample z_in to match z_gt depth for proper conditioning
        if z_in.shape[2] != z_gt.shape[2]:
            # Upsample z_in along depth dimension to match z_gt
            z_in_upsampled = F.interpolate(
                z_in,
                size=z_gt.shape[2:],  # (D_gt, H, W)
                mode='trilinear',
                align_corners=False
            )
        else:
            z_in_upsampled = z_in

        # If mask provided, downsample it to match latent depth (VAE reduces depth by factor)
        z_mask = None
        if mask is not None:
            # VAE downsamples depth by a factor (typically 4-6x for MAISI)
            # Downsample mask to match z_gt depth: (B, C, T_gt) -> (B, C, T_latent)
            z_mask = F.interpolate(
                mask.float().unsqueeze(-1).unsqueeze(-1),  # (B, C, T, 1, 1)
                size=(z_gt.shape[2], 1, 1),  # Match latent depth
                mode='trilinear',
                align_corners=False
            ).squeeze(-1).squeeze(-1)  # (B, C, T_latent)
            # Binarize: threshold at 0.5
            z_mask = (z_mask > 0.5).float()

        # Compute diffusion loss (with optional MS-SSIM)
        # Note: use_ssim and ssim_weight should be set in config
        # For now, defaulting to MSE only to maintain backward compatibility
        loss, loss_dict = self.diffusion.training_loss(
            self.unet, z_gt, z_in_upsampled,  # Use upsampled conditioning
            mask=z_mask,  # Pass latent-space mask
            vae=self.vae,
            v_gt=v_gt,
            use_ssim=False,  # Can be enabled via config
            ssim_weight=0.0   # Can be set via config (e.g., 0.2)
        )

        # Additional metrics
        metrics = {
            'loss': loss.item(),
            **loss_dict  # Include individual loss components
        }

        return loss, metrics

    def generate(self, v_in, sampler, num_inference_steps=20, guidance_scale=1.0,
                 target_depth=None):
        """
        Generate output video from input video

        Args:
            v_in: input video/volume (B, C, T_in, H, W)
                  For slice interpolation: thick slices
            sampler: sampling method ('ddpm' or 'ddim')
            num_inference_steps: number of denoising steps
            guidance_scale: classifier-free guidance scale (not implemented in simple version)
            target_depth: target depth dimension for output (for slice interpolation)
                         If None, output will have same depth as input

        Returns:
            v_out: generated video/volume (B, C, T_out, H, W)
                   For slice interpolation: thin slices (T_out > T_in)
        """
        device = v_in.device
        B, C, T_in, H, W = v_in.shape

        # Encode input video to get conditioning
        with torch.no_grad():
            z_in = self.vae.encode(v_in)  # (B, latent_dim, T_latent_in, h, w)

            # Determine target latent shape
            if target_depth is not None:
                # For slice interpolation: need to determine latent depth from target depth
                # MAISI compresses depth by 4×: D_latent = D / 4
                latent_depth_target = target_depth // 4

                # Upsample z_in to target latent depth
                z_in_upsampled = F.interpolate(
                    z_in,
                    size=(latent_depth_target, z_in.shape[3], z_in.shape[4]),
                    mode='trilinear',
                    align_corners=False
                )

                latent_shape = z_in_upsampled.shape
            else:
                # Standard video-to-video: same depth
                z_in_upsampled = z_in
                latent_shape = z_in.shape

            # Initialize with random noise
            z_t = torch.randn(latent_shape, device=device)

            # Denoise using specified sampler
            if sampler == 'ddpm':
                # Use built-in DDPM sampling from diffusion
                z_0 = self.diffusion.p_sample_loop(
                    self.unet,
                    latent_shape,
                    z_in_upsampled,
                    device,
                    progress=True
                )
            elif sampler == 'ddim':
                # DDIM sampling (will be implemented in sampler.py)
                from inference.sampler import DDIMSampler
                ddim_sampler = DDIMSampler(self.diffusion, self.unet)
                z_0 = ddim_sampler.sample(
                    latent_shape,
                    z_in_upsampled,
                    num_inference_steps,
                    device
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

            # Decode latent to video
            v_out = self.vae.decode(z_0)

        return v_out

    def save_checkpoint(self, path, optimizer=None, scheduler=None, scaler=None, epoch=None, global_step=None,
                       current_phase=None, best_loss=None, **kwargs):
        """
        Save model checkpoint

        Args:
            path: save path
            optimizer: optimizer state (optional)
            scheduler: scheduler state (optional)
            scaler: GradScaler state for mixed precision training (optional)
                    CRITICAL for stable checkpoint resume with AMP!
            epoch: current epoch (optional)
            global_step: current global step (optional)
            current_phase: current training phase for two-phase training (optional)
            best_loss: best validation loss (optional)
            **kwargs: additional state to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if scaler is not None:
            # Save GradScaler state for mixed precision training
            # Critical: preserves scale factor, growth/backoff counters
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if global_step is not None:
            checkpoint['global_step'] = global_step
        if current_phase is not None:
            checkpoint['current_phase'] = current_phase
        if best_loss is not None:
            checkpoint['best_loss'] = best_loss

        # Add any additional kwargs
        checkpoint.update(kwargs)

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        """
        Load model from checkpoint

        Args:
            path: checkpoint path
            device: device to load model on

        Returns:
            model: loaded model
            checkpoint: full checkpoint dict (contains optimizer state, etc.)
        """
        # weights_only=False is required to load optimizer/scheduler states
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Create model from config
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        print(f"Model loaded from {path}")
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'global_step' in checkpoint:
            print(f"Checkpoint global step: {checkpoint['global_step']}")

        return model, checkpoint

    def count_parameters(self):
        """
        Count model parameters (both total and trainable)

        Returns dict with:
            - total_params: All parameters (including frozen)
            - trainable_params: Only trainable parameters
            - vae_params: VAE parameters (all, including frozen for pretrained)
            - unet_params: U-Net parameters (trainable only)
        """
        # Count all parameters in the model
        total_params = sum(p.numel() for p in self.parameters())

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Handle custom MAISI VAE which is nested in self.vae.maisi_vae
        # Count ALL MAISI parameters (even if frozen) to show the full model size
        if hasattr(self.vae, 'maisi_vae') and self.vae.maisi_vae is not None:
            vae_params = sum(p.numel() for p in self.vae.maisi_vae.parameters())
            vae_trainable = sum(p.numel() for p in self.vae.maisi_vae.parameters() if p.requires_grad)
        else:
            # Regular VAE - count all parameters
            vae_params = sum(p.numel() for p in self.vae.parameters())
            vae_trainable = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)

        # U-Net parameters (trainable only)
        unet_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'vae': vae_params,
            'vae_trainable': vae_trainable,
            'unet': unet_params,
            'diffusion': 0  # Diffusion has no trainable params
        }


if __name__ == "__main__":
    # Test the complete model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    config = {
        'in_channels': 3,
        'latent_dim': 4,
        'vae_base_channels': 64,
        'unet_model_channels': 128,
        'unet_num_res_blocks': 2,
        'unet_attention_levels': [1, 2],
        'unet_channel_mult': [1, 2, 4, 4],
        'unet_num_heads': 4,
        'unet_time_embed_dim': 512,
        'noise_schedule': 'cosine',
        'diffusion_timesteps': 1000
    }

    # Create model
    model = VideoToVideoDiffusion(config).to(device)

    # Test forward pass (training)
    v_in = torch.randn(2, 3, 16, 256, 256).to(device)
    v_gt = torch.randn(2, 3, 16, 256, 256).to(device)

    print("Testing forward pass (training)...")
    loss, metrics = model(v_in, v_gt)
    print(f"Loss: {loss.item():.6f}")
    print(f"Metrics: {metrics}")

    # Count parameters
    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Total: {params['total']:,}")
    print(f"  VAE: {params['vae']:,}")
    print(f"  U-Net: {params['unet']:,}")

    print("\nModel test successful!")
