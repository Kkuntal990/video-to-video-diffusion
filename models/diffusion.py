"""
Gaussian Diffusion Process

Implements:
- Forward diffusion: Add noise to clean latents
- Training loss: MSE between predicted and actual noise
- Reverse diffusion: Denoise from noise to clean (handled by sampler)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process for Video-to-Video model

    Implements the forward process:
        z_t = alpha_t * z + sigma_t * epsilon

    Training objective:
        L = E[||epsilon - epsilon_theta(z_t, t, c)||^2]
    """

    def __init__(self, noise_schedule='cosine', timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()

        self.timesteps = timesteps
        self.noise_schedule = noise_schedule

        # Generate noise schedule
        if noise_schedule == 'linear':
            betas = self._linear_beta_schedule(timesteps, beta_start, beta_end)
        elif noise_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")

        # Pre-compute diffusion constants
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (not parameters, but saved with model)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for forward diffusion q(z_t | z_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Calculations for posterior q(z_{t-1} | z_t, z_0) - used in DDPM sampling
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _linear_beta_schedule(self, timesteps, beta_start, beta_end):
        """Linear noise schedule"""
        return torch.linspace(beta_start, beta_end, timesteps)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, z_0, t, noise=None):
        """
        Forward diffusion: Sample from q(z_t | z_0)

        z_t = sqrt(alpha_cumprod_t) * z_0 + sqrt(1 - alpha_cumprod_t) * epsilon

        Args:
            z_0: clean latent (B, C, T, H, W)
            t: timestep (B,)
            noise: optional pre-sampled noise

        Returns:
            z_t: noisy latent at timestep t
            noise: the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        # Get coefficients for timestep t
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, z_0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, z_0.shape)

        # Apply noise
        z_t = sqrt_alpha_cumprod_t * z_0 + sqrt_one_minus_alpha_cumprod_t * noise

        return z_t, noise

    def training_loss(self, model, z_0, c, mask=None, vae=None, v_gt=None, use_ssim=False, ssim_weight=0.0):
        """
        Compute training loss for the denoising model

        L = (1-λ) * E[||epsilon - epsilon_theta(z_t, t, c)||^2] + λ * L_SSIM

        Args:
            model: denoising model (epsilon_theta)
            z_0: clean latent (B, C, T, H, W)
            c: conditioning latent (B, C, T, H, W)
            mask: optional padding mask (B, C, T) where 1=real, 0=padding
                  CRITICAL: Prevents learning on zero-padded regions!
            vae: VAE model for decoding (optional, for SSIM loss)
            v_gt: ground truth video (optional, for SSIM loss)
            use_ssim: whether to use MS-SSIM loss component
            ssim_weight: weight for SSIM loss (default 0.0 = MSE only)

        Returns:
            loss: Combined loss (MSE + optional MS-SSIM)
            loss_dict: Dictionary with individual loss components
        """
        B = z_0.shape[0]
        device = z_0.device

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device, dtype=torch.long)

        # Sample noise
        noise = torch.randn_like(z_0)

        # Get noisy latent
        z_t, _ = self.q_sample(z_0, t, noise)

        # Predict noise
        noise_pred = model(z_t, t, c)

        # Compute MSE loss (base diffusion loss)
        if mask is not None:
            # Masked MSE loss: only compute loss on real (non-padded) regions
            # Expand mask to match noise shape: (B, C, T) -> (B, C, T, H, W)
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # (B, C, T, 1, 1)
            mask_expanded = mask_expanded.expand_as(noise_pred)  # (B, C, T, H, W)

            # Compute element-wise MSE and apply mask
            mse_per_element = (noise_pred - noise) ** 2
            masked_mse = mse_per_element * mask_expanded

            # Average over masked elements only (not all elements)
            num_valid_elements = mask_expanded.sum()
            if num_valid_elements > 0:
                loss_mse = masked_mse.sum() / num_valid_elements
            else:
                # Fallback if somehow all masked out (shouldn't happen)
                loss_mse = masked_mse.mean()
        else:
            # No mask: standard MSE loss
            loss_mse = F.mse_loss(noise_pred, noise)

        # Initialize loss dict
        loss_dict = {'mse': loss_mse.item()}

        # Optionally add MS-SSIM loss on reconstructed videos
        if use_ssim and ssim_weight > 0.0 and vae is not None and v_gt is not None:
            try:
                from pytorch_msssim import ms_ssim

                # Predict clean latent (denoise)
                z_0_pred = self._predict_z_0_from_noise(z_t, t, noise_pred)

                # Decode to video space
                with torch.no_grad():
                    v_pred = vae.decode(z_0_pred)

                # Compute MS-SSIM loss
                # MS-SSIM works on 4D tensors, so we process each frame
                ssim_losses = []
                B, C, T, H, W = v_gt.shape
                for i in range(T):
                    frame_gt = v_gt[:, :, i, :, :]  # (B, C, H, W)
                    frame_pred = v_pred[:, :, i, :, :]  # (B, C, H, W)

                    # MS-SSIM expects values in [0, 1], but we have [-1, 1]
                    frame_gt_norm = (frame_gt + 1.0) / 2.0
                    frame_pred_norm = (frame_pred + 1.0) / 2.0

                    ssim_val = ms_ssim(frame_pred_norm, frame_gt_norm, data_range=1.0, size_average=True)
                    ssim_losses.append(1.0 - ssim_val)  # Convert to loss (lower is better)

                loss_ssim = torch.stack(ssim_losses).mean()
                loss_dict['ssim'] = loss_ssim.item()

                # Combined loss
                total_loss = (1.0 - ssim_weight) * loss_mse + ssim_weight * loss_ssim
                loss_dict['total'] = total_loss.item()

                return total_loss, loss_dict

            except ImportError:
                print("Warning: pytorch-msssim not installed. Falling back to MSE-only loss.")
            except Exception as e:
                print(f"Warning: MS-SSIM calculation failed: {e}. Using MSE-only loss.")

        # If no SSIM, return MSE only
        loss_dict['total'] = loss_mse.item()
        return loss_mse, loss_dict

    def _predict_z_0_from_noise(self, z_t, t, noise_pred):
        """
        Predict clean latent z_0 from noisy latent z_t and predicted noise

        z_0 = (z_t - sqrt(1-alpha_cumprod) * noise) / sqrt(alpha_cumprod)

        Args:
            z_t: noisy latent
            t: timestep
            noise_pred: predicted noise

        Returns:
            z_0_pred: predicted clean latent
        """
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, z_t.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)

        z_0_pred = (z_t - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t

        return z_0_pred

    def p_mean_variance(self, model, z_t, t, c, clip_denoised=True):
        """
        Compute the mean and variance of the posterior distribution q(z_{t-1} | z_t, z_0)

        Used for DDPM sampling.

        Args:
            model: denoising model
            z_t: noisy latent at timestep t
            t: timestep
            c: conditioning
            clip_denoised: whether to clip the denoised latent to [-1, 1]

        Returns:
            mean, variance, log_variance
        """
        # Predict noise
        noise_pred = model(z_t, t, c)

        # Predict z_0 from z_t and noise
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, z_t.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)

        z_0_pred = (z_t - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t

        if clip_denoised:
            z_0_pred = torch.clamp(z_0_pred, -1.0, 1.0)

        # Compute posterior mean
        posterior_mean_coef1 = self._extract(self.posterior_mean_coef1, t, z_t.shape)
        posterior_mean_coef2 = self._extract(self.posterior_mean_coef2, t, z_t.shape)

        mean = posterior_mean_coef1 * z_0_pred + posterior_mean_coef2 * z_t

        # Compute posterior variance
        variance = self._extract(self.posterior_variance, t, z_t.shape)
        log_variance = self._extract(self.posterior_log_variance_clipped, t, z_t.shape)

        return mean, variance, log_variance

    def p_sample(self, model, z_t, t, c, clip_denoised=True):
        """
        Sample z_{t-1} from p(z_{t-1} | z_t) using the model

        DDPM sampling step.

        Args:
            model: denoising model
            z_t: noisy latent at timestep t
            t: timestep
            c: conditioning
            clip_denoised: whether to clip denoised latent

        Returns:
            z_{t-1}: less noisy latent
        """
        # Get mean and variance
        mean, _, log_variance = self.p_mean_variance(model, z_t, t, c, clip_denoised)

        # Sample noise
        noise = torch.randn_like(z_t)

        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(z_t.shape) - 1)))

        # Sample z_{t-1}
        z_t_minus_1 = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

        return z_t_minus_1

    def p_sample_loop(self, model, shape, c, device, progress=True):
        """
        Generate samples using DDPM sampling

        Args:
            model: denoising model
            shape: shape of latent to generate (B, C, T, H, W)
            c: conditioning
            device: torch device
            progress: whether to show progress bar

        Returns:
            z_0: generated clean latent
        """
        B = shape[0]
        z = torch.randn(shape, device=device)

        if progress:
            from tqdm import tqdm
            timesteps = tqdm(reversed(range(self.timesteps)), desc='DDPM Sampling')
        else:
            timesteps = reversed(range(self.timesteps))

        for t_idx in timesteps:
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            z = self.p_sample(model, z, t, c)

        return z

    def _extract(self, a, t, x_shape):
        """
        Extract coefficients at specified timesteps t and reshape to match x_shape

        Args:
            a: tensor of coefficients (timesteps,)
            t: timestep indices (B,)
            x_shape: target shape (B, C, T, H, W)

        Returns:
            extracted values reshaped to (B, 1, 1, 1, 1)
        """
        B = t.shape[0]
        out = a.gather(-1, t).float()
        return out.reshape(B, *((1,) * (len(x_shape) - 1)))


if __name__ == "__main__":
    # Test diffusion process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create diffusion
    diffusion = GaussianDiffusion(noise_schedule='cosine', timesteps=1000).to(device)

    # Test forward diffusion
    B, C, T, H, W = 2, 4, 16, 32, 32
    z_0 = torch.randn(B, C, T, H, W).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)

    print(f"Clean latent shape: {z_0.shape}")
    print(f"Timesteps: {t}")

    # Forward process
    z_t, noise = diffusion.q_sample(z_0, t)
    print(f"Noisy latent shape: {z_t.shape}")
    print(f"Noise shape: {noise.shape}")

    # Check noise schedule
    print(f"\nNoise schedule: {diffusion.noise_schedule}")
    print(f"Beta range: [{diffusion.betas.min():.6f}, {diffusion.betas.max():.6f}]")
    print(f"Alpha cumprod range: [{diffusion.alphas_cumprod.min():.6f}, {diffusion.alphas_cumprod.max():.6f}]")
