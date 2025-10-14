"""
Complete Video-to-Video Diffusion Model

Integrates VAE, U-Net, and Diffusion process into a single model.
"""

import torch
import torch.nn as nn
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

    def __init__(self, config):
        super().__init__()

        # VAE for encoding/decoding videos
        self.vae = VideoVAE(
            in_channels=config.get('in_channels', 3),
            latent_dim=config.get('latent_dim', 4),
            base_channels=config.get('vae_base_channels', 64)
        )

        # U-Net for denoising
        self.unet = UNet3D(
            latent_dim=config.get('latent_dim', 4),
            model_channels=config.get('unet_model_channels', 128),
            num_res_blocks=config.get('unet_num_res_blocks', 2),
            attention_levels=config.get('unet_attention_levels', [1, 2]),
            channel_mult=tuple(config.get('unet_channel_mult', [1, 2, 4, 4])),
            num_heads=config.get('unet_num_heads', 4),
            time_embed_dim=config.get('unet_time_embed_dim', 512)
        )

        # Diffusion process
        self.diffusion = GaussianDiffusion(
            noise_schedule=config.get('noise_schedule', 'cosine'),
            timesteps=config.get('diffusion_timesteps', 1000),
            beta_start=config.get('beta_start', 0.0001),
            beta_end=config.get('beta_end', 0.02)
        )

        self.config = config

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

    def forward(self, v_in, v_gt):
        """
        Forward pass for training

        Args:
            v_in: input video (B, 3, T, H, W)
            v_gt: ground truth video (B, 3, T, H, W)

        Returns:
            loss: training loss (scalar)
            metrics: dict with additional metrics
        """
        # Encode videos to latent space
        z_in = self.vae.encode(v_in)  # Conditioning
        z_gt = self.vae.encode(v_gt)  # Target for diffusion

        # Compute diffusion loss
        loss = self.diffusion.training_loss(self.unet, z_gt, z_in)

        # Additional metrics (optional)
        metrics = {
            'loss': loss.item()
        }

        return loss, metrics

    def generate(self, v_in, sampler, num_inference_steps=20, guidance_scale=1.0):
        """
        Generate output video from input video

        Args:
            v_in: input video (B, 3, T, H, W)
            sampler: sampling method ('ddpm' or 'ddim')
            num_inference_steps: number of denoising steps
            guidance_scale: classifier-free guidance scale (not implemented in simple version)

        Returns:
            v_out: generated video (B, 3, T, H, W)
        """
        device = v_in.device
        B, C, T, H, W = v_in.shape

        # Encode input video to get conditioning
        with torch.no_grad():
            z_in = self.vae.encode(v_in)

            # Get latent shape
            latent_shape = z_in.shape

            # Initialize with random noise
            z_t = torch.randn(latent_shape, device=device)

            # Denoise using specified sampler
            if sampler == 'ddpm':
                # Use built-in DDPM sampling from diffusion
                z_0 = self.diffusion.p_sample_loop(
                    self.unet,
                    latent_shape,
                    z_in,
                    device,
                    progress=True
                )
            elif sampler == 'ddim':
                # DDIM sampling (will be implemented in sampler.py)
                from inference.sampler import DDIMSampler
                ddim_sampler = DDIMSampler(self.diffusion, self.unet)
                z_0 = ddim_sampler.sample(
                    latent_shape,
                    z_in,
                    num_inference_steps,
                    device
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

            # Decode latent to video
            v_out = self.vae.decode(z_0)

        return v_out

    def save_checkpoint(self, path, optimizer=None, epoch=None, global_step=None):
        """
        Save model checkpoint

        Args:
            path: save path
            optimizer: optimizer state (optional)
            epoch: current epoch (optional)
            global_step: current global step (optional)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if global_step is not None:
            checkpoint['global_step'] = global_step

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
        checkpoint = torch.load(path, map_location=device)

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
        """Count total trainable parameters"""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        vae = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        unet = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)

        return {
            'total': total,
            'vae': vae,
            'unet': unet,
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
