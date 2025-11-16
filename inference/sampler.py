"""
Sampling algorithms for video generation

Implements:
- DDPM sampling (original diffusion sampling)
- DDIM sampling (faster deterministic sampling)
"""

import torch
import numpy as np
from tqdm import tqdm


class DDPMSampler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) Sampler

    Implements the original DDPM sampling algorithm.
    Slower but stochastic (generates diverse outputs).
    """

    def __init__(self, diffusion, model):
        """
        Args:
            diffusion: GaussianDiffusion instance
            model: denoising model (U-Net)
        """
        self.diffusion = diffusion
        self.model = model
        self.timesteps = diffusion.timesteps

    @torch.no_grad()
    def sample(self, shape, conditioning, device, progress=True):
        """
        Generate samples using DDPM

        Args:
            shape: shape of latent to generate (B, C, T, H, W)
            conditioning: conditioning latent (B, C, T, H, W)
            device: torch device
            progress: whether to show progress bar

        Returns:
            z_0: generated clean latent
        """
        B = shape[0]
        z = torch.randn(shape, device=device)

        if progress:
            timesteps = tqdm(reversed(range(self.timesteps)), desc='DDPM Sampling', total=self.timesteps)
        else:
            timesteps = reversed(range(self.timesteps))

        for t_idx in timesteps:
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            z = self.diffusion.p_sample(self.model, z, t, conditioning)

        return z


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) Sampler

    Implements the DDIM sampling algorithm.
    Faster (requires fewer steps) and deterministic.

    Reference: https://arxiv.org/abs/2010.02502
    """

    def __init__(self, diffusion, model):
        """
        Args:
            diffusion: GaussianDiffusion instance
            model: denoising model (U-Net)
        """
        self.diffusion = diffusion
        self.model = model
        self.timesteps = diffusion.timesteps

    def _get_timesteps(self, num_inference_steps):
        """
        Get subset of timesteps for DDIM

        Args:
            num_inference_steps: number of denoising steps

        Returns:
            timesteps: list of timestep indices
        """
        # Uniformly sample timesteps
        step = self.timesteps // num_inference_steps
        timesteps = np.arange(0, self.timesteps, step)

        # Ensure we include timestep 0
        if timesteps[-1] != self.timesteps - 1:
            timesteps = np.append(timesteps, self.timesteps - 1)

        return timesteps[::-1]  # Reverse (start from T, go to 0)

    @torch.no_grad()
    def sample(self, shape, conditioning, num_inference_steps, device, eta=0.0, progress=True):
        """
        Generate samples using DDIM

        Args:
            shape: shape of latent to generate (B, C, T, H, W)
            conditioning: conditioning latent (B, C, T, H, W)
            num_inference_steps: number of denoising steps (e.g., 20, 50)
            device: torch device
            eta: stochasticity parameter (0 = deterministic, 1 = DDPM-like)
            progress: whether to show progress bar

        Returns:
            z_0: generated clean latent
        """
        B = shape[0]

        # Get subset of timesteps
        timesteps = self._get_timesteps(num_inference_steps)

        # Start with random noise
        z = torch.randn(shape, device=device)

        if progress:
            pbar = tqdm(timesteps, desc='DDIM Sampling', total=len(timesteps))
        else:
            pbar = timesteps

        for i, t_idx in enumerate(pbar):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(z, t, conditioning)

            # Get alpha values
            alpha_t = self.diffusion.alphas_cumprod[t_idx]
            alpha_t_prev = self.diffusion.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)

            # Predict z_0 from z_t and noise
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            z_0_pred = (z - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

            # DO NOT clamp latents - VAE latent space is NOT bounded to [-1, 1]
            # MAISI VAE uses scaling_factor=0.18215, so latents can range [-5, 5] or wider
            # Clamping destroys the latent structure and causes poor reconstruction quality
            # z_0_pred = torch.clamp(z_0_pred, -1.0, 1.0)  # REMOVED - corrupts latents

            # Compute direction pointing to z_t
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)

            # DDIM formula
            dir_zt = sqrt_one_minus_alpha_t_prev * noise_pred

            # Add stochasticity (eta parameter)
            if eta > 0:
                sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
                noise = torch.randn_like(z)
                z = sqrt_alpha_t_prev * z_0_pred + dir_zt + sigma_t * noise
            else:
                z = sqrt_alpha_t_prev * z_0_pred + dir_zt

        return z


class EDMSampler:
    """
    EDM (Elucidating the Design Space of Diffusion-Based Generative Models) Sampler

    Advanced sampler with better noise schedules and sampling strategies.
    For future implementation.
    """

    def __init__(self, diffusion, model):
        self.diffusion = diffusion
        self.model = model
        raise NotImplementedError("EDM sampler not yet implemented")


if __name__ == "__main__":
    # Test samplers
    from models import VideoToVideoDiffusion
    import numpy as np

    print("Testing samplers...")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        'in_channels': 3,
        'latent_dim': 4,
        'vae_base_channels': 32,
        'unet_model_channels': 64,
        'unet_num_res_blocks': 1,
        'unet_attention_levels': [1],
        'unet_channel_mult': [1, 2],
        'unet_num_heads': 2,
        'noise_schedule': 'cosine',
        'diffusion_timesteps': 100  # Fewer for testing
    }

    model = VideoToVideoDiffusion(config).to(device)
    model.eval()

    # Create dummy conditioning
    B, C, T, H, W = 1, 4, 16, 32, 32
    conditioning = torch.randn(B, C, T, H, W).to(device)
    shape = (B, C, T, H, W)

    print(f"Shape: {shape}")
    print(f"Conditioning shape: {conditioning.shape}")

    # Test DDPM sampler
    print("\nTesting DDPM sampler...")
    ddpm_sampler = DDPMSampler(model.diffusion, model.unet)

    # Note: This will be slow without actually running it
    # z_ddpm = ddpm_sampler.sample(shape, conditioning, device, progress=False)
    # print(f"DDPM output shape: {z_ddpm.shape}")

    # Test DDIM sampler
    print("\nTesting DDIM sampler...")
    ddim_sampler = DDIMSampler(model.diffusion, model.unet)

    z_ddim = ddim_sampler.sample(shape, conditioning, num_inference_steps=10, device=device, progress=True)
    print(f"DDIM output shape: {z_ddim.shape}")

    print("\nSampler test successful!")
