"""
Sampling algorithms for video generation

Implements:
- DDPM sampling (original diffusion sampling)
- DDIM sampling (faster deterministic sampling)
- Sliding-window patch stitching for full-volume inference
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional


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

    @torch.no_grad()
    def sample_with_stitching(
        self,
        v_thick_full: torch.Tensor,
        vae,
        patch_size: Tuple[int, int, int] = (8, 192, 192),  # (D_thick, H, W)
        target_patch_size: Tuple[int, int, int] = (48, 192, 192),  # (D_thin, H, W)
        stride: Tuple[int, int, int] = (4, 96, 96),  # 50% overlap
        device: str = 'cuda',
        progress: bool = True
    ) -> torch.Tensor:
        """
        Generate full thin volume using sliding-window patch-based sampling.

        Args:
            v_thick_full: Full thick volume (B, 1, D_thick, H_full, W_full)
                         e.g., (1, 1, 50, 512, 512)
            vae: VAE model for encoding/decoding patches
            patch_size: Size of thick patches (D, H, W) to extract
            target_patch_size: Size of thin patches to generate (D, H, W)
            stride: Stride for sliding window (D, H, W) - smaller = more overlap
            device: torch device
            progress: whether to show progress bar

        Returns:
            v_thin_full: Generated full thin volume (B, 1, D_thin, H_full, W_full)
                        e.g., (1, 1, 300, 512, 512)
        """
        B, C, D_thick, H_full, W_full = v_thick_full.shape
        patch_d, patch_h, patch_w = patch_size
        target_d, target_h, target_w = target_patch_size
        stride_d, stride_h, stride_w = stride

        # Calculate output depth based on ratio
        depth_ratio = target_d / patch_d  # e.g., 48/8 = 6
        D_thin = int(D_thick * depth_ratio)

        # Initialize output accumulator and weight map
        v_thin_full = torch.zeros(B, C, D_thin, H_full, W_full, device=device)
        weight_map = torch.zeros(B, C, D_thin, H_full, W_full, device=device)

        # Create Gaussian weight for blending overlapping patches
        weight_patch = self._create_gaussian_weight(target_d, target_h, target_w).to(device)
        weight_patch = weight_patch.view(1, 1, target_d, target_h, target_w)

        # Generate patch grid
        patches_d = list(range(0, D_thick - patch_d + 1, stride_d)) + [max(0, D_thick - patch_d)]
        patches_h = list(range(0, H_full - patch_h + 1, stride_h)) + [max(0, H_full - patch_h)]
        patches_w = list(range(0, W_full - patch_w + 1, stride_w)) + [max(0, W_full - patch_w)]

        # Remove duplicates
        patches_d = sorted(set(patches_d))
        patches_h = sorted(set(patches_h))
        patches_w = sorted(set(patches_w))

        total_patches = len(patches_d) * len(patches_h) * len(patches_w)

        if progress:
            pbar = tqdm(total=total_patches, desc='Patch-based inference')

        # Process each patch
        for d_start in patches_d:
            for h_start in patches_h:
                for w_start in patches_w:
                    # Extract thick patch
                    v_thick_patch = v_thick_full[
                        :, :,
                        d_start:d_start + patch_d,
                        h_start:h_start + patch_h,
                        w_start:w_start + patch_w
                    ]

                    # Encode thick patch
                    z_thick_patch = vae.encode(v_thick_patch)  # (B, 4, D_latent, h_latent, w_latent)

                    # Generate thin latent patch via diffusion
                    latent_shape = (B, z_thick_patch.shape[1], z_thick_patch.shape[2],
                                   z_thick_patch.shape[3], z_thick_patch.shape[4])
                    z_thin_patch = self.sample(
                        latent_shape,
                        z_thick_patch,
                        device,
                        progress=False
                    )

                    # Decode to thin patch
                    v_thin_patch = vae.decode(z_thin_patch)  # (B, 1, D_patch, H_patch, W_patch)

                    # Compute output patch location
                    d_thin_start = int(d_start * depth_ratio)
                    d_thin_end = d_thin_start + target_d
                    h_end = h_start + target_h
                    w_end = w_start + target_w

                    # Add to accumulator with Gaussian weighting
                    v_thin_full[:, :, d_thin_start:d_thin_end, h_start:h_end, w_start:w_end] += (
                        v_thin_patch * weight_patch
                    )
                    weight_map[:, :, d_thin_start:d_thin_end, h_start:h_end, w_start:w_end] += weight_patch

                    if progress:
                        pbar.update(1)

        if progress:
            pbar.close()

        # Normalize by weight map
        v_thin_full = v_thin_full / (weight_map + 1e-8)

        return v_thin_full

    def _create_gaussian_weight(self, d: int, h: int, w: int) -> torch.Tensor:
        """
        Create 3D Gaussian weight for smooth patch blending.

        Args:
            d, h, w: patch dimensions

        Returns:
            weight: (d, h, w) Gaussian weight tensor
        """
        # Create 1D Gaussian for each dimension
        sigma_d, sigma_h, sigma_w = d / 6, h / 6, w / 6

        z = torch.arange(d).float() - (d - 1) / 2
        y = torch.arange(h).float() - (h - 1) / 2
        x = torch.arange(w).float() - (w - 1) / 2

        gauss_d = torch.exp(-(z ** 2) / (2 * sigma_d ** 2))
        gauss_h = torch.exp(-(y ** 2) / (2 * sigma_h ** 2))
        gauss_w = torch.exp(-(x ** 2) / (2 * sigma_w ** 2))

        # Outer product to get 3D Gaussian
        weight = gauss_d[:, None, None] * gauss_h[None, :, None] * gauss_w[None, None, :]

        return weight


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
        import logging
        logger = logging.getLogger(__name__)

        B = shape[0]

        # Get subset of timesteps
        timesteps = self._get_timesteps(num_inference_steps)

        # Start with random noise
        z = torch.randn(shape, device=device)

        # CHECKPOINT 1: Check initial noise
        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.error(f"NaN/Inf in initial noise z! NaN: {torch.isnan(z).sum()}, Inf: {torch.isinf(z).sum()}")
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)

        # CHECKPOINT 2: Check conditioning
        if torch.isnan(conditioning).any() or torch.isinf(conditioning).any():
            logger.error(f"NaN/Inf in conditioning! NaN: {torch.isnan(conditioning).sum()}, Inf: {torch.isinf(conditioning).sum()}")

        if progress:
            pbar = tqdm(timesteps, desc='DDIM Sampling', total=len(timesteps))
        else:
            pbar = timesteps

        for i, t_idx in enumerate(pbar):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(z, t, conditioning)

            # CHECKPOINT 3: Check UNet prediction for NaN
            if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                logger.error(f"[Step {i}/{len(timesteps)}] NaN/Inf in noise_pred! NaN: {torch.isnan(noise_pred).sum()}, Inf: {torch.isinf(noise_pred).sum()}")
                # Replace NaN/Inf with zeros to prevent propagation
                noise_pred = torch.nan_to_num(noise_pred, nan=0.0, posinf=1.0, neginf=-1.0)

            # Get alpha values
            alpha_t = self.diffusion.alphas_cumprod[t_idx]
            alpha_t_prev = self.diffusion.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)

            # Predict z_0 from z_t and noise with numerical stability
            sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)  # Add epsilon to prevent division by zero
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)

            # NUMERICAL STABILITY: Add epsilon to denominator
            z_0_pred = (z - sqrt_one_minus_alpha_t * noise_pred) / (sqrt_alpha_t + 1e-8)

            # CHECKPOINT 4: Check z_0_pred for NaN
            if torch.isnan(z_0_pred).any() or torch.isinf(z_0_pred).any():
                logger.error(f"[Step {i}/{len(timesteps)}] NaN/Inf in z_0_pred! NaN: {torch.isnan(z_0_pred).sum()}, Inf: {torch.isinf(z_0_pred).sum()}")
                logger.error(f"  alpha_t={alpha_t:.6f}, sqrt_alpha_t={sqrt_alpha_t:.6f}")
                z_0_pred = torch.nan_to_num(z_0_pred, nan=0.0, posinf=1.0, neginf=-1.0)

            # Clamp z_0_pred to reasonable latent range to prevent explosion
            # Custom VAE latent space typically in [-10, 10] range (not [-1, 1])
            z_0_pred = torch.clamp(z_0_pred, -10.0, 10.0)

            # Compute direction pointing to z_t
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev + 1e-8)
            sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev + 1e-8)

            # DDIM formula
            dir_zt = sqrt_one_minus_alpha_t_prev * noise_pred

            # Add stochasticity (eta parameter)
            if eta > 0:
                # Add epsilon for numerical stability
                sigma_t = eta * torch.sqrt((1 - alpha_t_prev + 1e-8) / (1 - alpha_t + 1e-8) * (1 - alpha_t / (alpha_t_prev + 1e-8)))
                noise = torch.randn_like(z)
                z = sqrt_alpha_t_prev * z_0_pred + dir_zt + sigma_t * noise
            else:
                z = sqrt_alpha_t_prev * z_0_pred + dir_zt

            # CHECKPOINT 5: Check z for NaN after update
            if torch.isnan(z).any() or torch.isinf(z).any():
                logger.error(f"[Step {i}/{len(timesteps)}] NaN/Inf in z after update! NaN: {torch.isnan(z).sum()}, Inf: {torch.isinf(z).sum()}")
                z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)

        return z

    @torch.no_grad()
    def sample_with_stitching(
        self,
        v_thick_full: torch.Tensor,
        vae,
        num_inference_steps: int = 20,
        patch_size: Tuple[int, int, int] = (8, 192, 192),  # (D_thick, H, W)
        target_patch_size: Tuple[int, int, int] = (48, 192, 192),  # (D_thin, H, W)
        stride: Tuple[int, int, int] = (4, 96, 96),  # 50% overlap
        device: str = 'cuda',
        eta: float = 0.0,
        progress: bool = True
    ) -> torch.Tensor:
        """
        Generate full thin volume using sliding-window patch-based DDIM sampling.

        Args:
            v_thick_full: Full thick volume (B, 1, D_thick, H_full, W_full)
                         e.g., (1, 1, 50, 512, 512)
            vae: VAE model for encoding/decoding patches
            num_inference_steps: number of DDIM steps (e.g., 20, 50)
            patch_size: Size of thick patches (D, H, W) to extract
            target_patch_size: Size of thin patches to generate (D, H, W)
            stride: Stride for sliding window (D, H, W) - smaller = more overlap
            device: torch device
            eta: stochasticity parameter (0 = deterministic)
            progress: whether to show progress bar

        Returns:
            v_thin_full: Generated full thin volume (B, 1, D_thin, H_full, W_full)
                        e.g., (1, 1, 300, 512, 512)
        """
        B, C, D_thick, H_full, W_full = v_thick_full.shape
        patch_d, patch_h, patch_w = patch_size
        target_d, target_h, target_w = target_patch_size
        stride_d, stride_h, stride_w = stride

        # Calculate output depth based on ratio
        depth_ratio = target_d / patch_d  # e.g., 48/8 = 6
        D_thin = int(D_thick * depth_ratio)

        # Initialize output accumulator and weight map
        v_thin_full = torch.zeros(B, C, D_thin, H_full, W_full, device=device)
        weight_map = torch.zeros(B, C, D_thin, H_full, W_full, device=device)

        # Create Gaussian weight for blending overlapping patches
        weight_patch = self._create_gaussian_weight(target_d, target_h, target_w).to(device)
        weight_patch = weight_patch.view(1, 1, target_d, target_h, target_w)

        # Generate patch grid
        patches_d = list(range(0, D_thick - patch_d + 1, stride_d)) + [max(0, D_thick - patch_d)]
        patches_h = list(range(0, H_full - patch_h + 1, stride_h)) + [max(0, H_full - patch_h)]
        patches_w = list(range(0, W_full - patch_w + 1, stride_w)) + [max(0, W_full - patch_w)]

        # Remove duplicates
        patches_d = sorted(set(patches_d))
        patches_h = sorted(set(patches_h))
        patches_w = sorted(set(patches_w))

        total_patches = len(patches_d) * len(patches_h) * len(patches_w)

        if progress:
            pbar = tqdm(total=total_patches, desc=f'Patch-based DDIM inference ({num_inference_steps} steps)')

        # Process each patch
        for d_start in patches_d:
            for h_start in patches_h:
                for w_start in patches_w:
                    # Extract thick patch
                    v_thick_patch = v_thick_full[
                        :, :,
                        d_start:d_start + patch_d,
                        h_start:h_start + patch_h,
                        w_start:w_start + patch_w
                    ]

                    # Encode thick patch
                    z_thick_patch = vae.encode(v_thick_patch)  # (B, 4, D_latent, h_latent, w_latent)

                    # Generate thin latent patch via DDIM
                    latent_shape = (B, z_thick_patch.shape[1], z_thick_patch.shape[2],
                                   z_thick_patch.shape[3], z_thick_patch.shape[4])
                    z_thin_patch = self.sample(
                        latent_shape,
                        z_thick_patch,
                        num_inference_steps,
                        device,
                        eta=eta,
                        progress=False
                    )

                    # Decode to thin patch
                    v_thin_patch = vae.decode(z_thin_patch)  # (B, 1, D_patch, H_patch, W_patch)

                    # Compute output patch location
                    d_thin_start = int(d_start * depth_ratio)
                    d_thin_end = d_thin_start + target_d
                    h_end = h_start + target_h
                    w_end = w_start + target_w

                    # Add to accumulator with Gaussian weighting
                    v_thin_full[:, :, d_thin_start:d_thin_end, h_start:h_end, w_start:w_end] += (
                        v_thin_patch * weight_patch
                    )
                    weight_map[:, :, d_thin_start:d_thin_end, h_start:h_end, w_start:w_end] += weight_patch

                    if progress:
                        pbar.update(1)

        if progress:
            pbar.close()

        # Normalize by weight map
        v_thin_full = v_thin_full / (weight_map + 1e-8)

        return v_thin_full

    def _create_gaussian_weight(self, d: int, h: int, w: int) -> torch.Tensor:
        """
        Create 3D Gaussian weight for smooth patch blending.

        Args:
            d, h, w: patch dimensions

        Returns:
            weight: (d, h, w) Gaussian weight tensor
        """
        # Create 1D Gaussian for each dimension
        sigma_d, sigma_h, sigma_w = d / 6, h / 6, w / 6

        z = torch.arange(d).float() - (d - 1) / 2
        y = torch.arange(h).float() - (h - 1) / 2
        x = torch.arange(w).float() - (w - 1) / 2

        gauss_d = torch.exp(-(z ** 2) / (2 * sigma_d ** 2))
        gauss_h = torch.exp(-(y ** 2) / (2 * sigma_h ** 2))
        gauss_w = torch.exp(-(x ** 2) / (2 * sigma_w ** 2))

        # Outer product to get 3D Gaussian
        weight = gauss_d[:, None, None] * gauss_h[None, :, None] * gauss_w[None, None, :]

        return weight


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
