"""
Video generation utilities

Helper functions for generating videos from trained models.
"""

import torch
from pathlib import Path
from .sampler import DDIMSampler, DDPMSampler
from data.transforms import save_video, load_video, VideoTransform, DenormalizeVideo


def generate_video(
    model,
    input_video_path,
    output_path,
    sampler_type='ddim',
    num_inference_steps=20,
    device='cuda',
    num_frames=16,
    resolution=(256, 256)
):
    """
    Generate output video from input video using trained model

    Args:
        model: trained VideoToVideoDiffusion model
        input_video_path: path to input video file
        output_path: path to save output video
        sampler_type: 'ddim' or 'ddpm'
        num_inference_steps: number of denoising steps
        device: torch device
        num_frames: number of frames to process
        resolution: (height, width)

    Returns:
        output_video: generated video tensor
    """
    model.eval()
    model.to(device)

    # Load and preprocess input video
    print(f"Loading input video from {input_video_path}...")
    input_frames = load_video(input_video_path, num_frames=num_frames)

    # Transform
    transform = VideoTransform(resolution=resolution, num_frames=num_frames)
    input_video = transform(input_frames)  # (3, T, H, W) in [-1, 1]
    input_video = input_video.unsqueeze(0).to(device)  # (1, 3, T, H, W)

    print(f"Input video shape: {input_video.shape}")

    # Encode input to latent space
    print("Encoding input video...")
    with torch.no_grad():
        z_in = model.vae.encode(input_video)

    print(f"Latent shape: {z_in.shape}")

    # Sample using specified sampler
    print(f"Generating video using {sampler_type} sampler ({num_inference_steps} steps)...")

    if sampler_type == 'ddim':
        sampler = DDIMSampler(model.diffusion, model.unet)
        z_0 = sampler.sample(
            z_in.shape,
            z_in,
            num_inference_steps,
            device,
            progress=True
        )
    elif sampler_type == 'ddpm':
        sampler = DDPMSampler(model.diffusion, model.unet)
        z_0 = sampler.sample(
            z_in.shape,
            z_in,
            device,
            progress=True
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

    # Decode latent to video
    print("Decoding latent to video...")
    with torch.no_grad():
        output_video = model.vae.decode(z_0)  # (1, 3, T, H, W)

    output_video = output_video.squeeze(0)  # (3, T, H, W)

    # Save output video
    print(f"Saving output video to {output_path}...")
    save_video(output_video, output_path, fps=8)

    return output_video


def generate_batch(
    model,
    input_videos,
    sampler_type='ddim',
    num_inference_steps=20,
    device='cuda'
):
    """
    Generate multiple videos in batch

    Args:
        model: trained VideoToVideoDiffusion model
        input_videos: batch of input videos (B, 3, T, H, W)
        sampler_type: 'ddim' or 'ddpm'
        num_inference_steps: number of denoising steps
        device: torch device

    Returns:
        output_videos: batch of generated videos (B, 3, T, H, W)
    """
    model.eval()
    model.to(device)

    input_videos = input_videos.to(device)
    B = input_videos.shape[0]

    print(f"Generating batch of {B} videos...")

    # Encode inputs
    with torch.no_grad():
        z_in = model.vae.encode(input_videos)

    # Sample
    if sampler_type == 'ddim':
        sampler = DDIMSampler(model.diffusion, model.unet)
        z_0 = sampler.sample(
            z_in.shape,
            z_in,
            num_inference_steps,
            device,
            progress=True
        )
    elif sampler_type == 'ddpm':
        sampler = DDPMSampler(model.diffusion, model.unet)
        z_0 = sampler.sample(
            z_in.shape,
            z_in,
            device,
            progress=True
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

    # Decode
    with torch.no_grad():
        output_videos = model.vae.decode(z_0)

    return output_videos


def interpolate_videos(
    model,
    video_a,
    video_b,
    num_interpolations=5,
    sampler_type='ddim',
    num_inference_steps=20,
    device='cuda'
):
    """
    Interpolate between two videos in latent space

    Args:
        model: trained VideoToVideoDiffusion model
        video_a: first video (3, T, H, W)
        video_b: second video (3, T, H, W)
        num_interpolations: number of interpolation steps
        sampler_type: 'ddim' or 'ddpm'
        num_inference_steps: number of denoising steps
        device: torch device

    Returns:
        interpolated_videos: list of interpolated videos
    """
    model.eval()
    model.to(device)

    video_a = video_a.unsqueeze(0).to(device)
    video_b = video_b.unsqueeze(0).to(device)

    # Encode to latent
    with torch.no_grad():
        z_a = model.vae.encode(video_a)
        z_b = model.vae.encode(video_b)

    # Interpolate in latent space
    alphas = torch.linspace(0, 1, num_interpolations).to(device)
    interpolated_videos = []

    for alpha in alphas:
        # Linear interpolation
        z_interp = (1 - alpha) * z_a + alpha * z_b

        # Generate from interpolated latent
        if sampler_type == 'ddim':
            sampler = DDIMSampler(model.diffusion, model.unet)
            z_0 = sampler.sample(
                z_interp.shape,
                z_interp,
                num_inference_steps,
                device,
                progress=False
            )
        else:
            sampler = DDPMSampler(model.diffusion, model.unet)
            z_0 = sampler.sample(
                z_interp.shape,
                z_interp,
                device,
                progress=False
            )

        # Decode
        with torch.no_grad():
            output = model.vae.decode(z_0).squeeze(0)

        interpolated_videos.append(output)

    return interpolated_videos


if __name__ == "__main__":
    # Test generation
    from models import VideoToVideoDiffusion
    import numpy as np
    import tempfile
    from data.transforms import save_video as save_vid

    print("Testing video generation...")

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
        'diffusion_timesteps': 100
    }

    model = VideoToVideoDiffusion(config).to(device)
    model.eval()

    # Create dummy input video
    input_video = torch.randn(1, 3, 16, 256, 256).to(device)

    print(f"Input video shape: {input_video.shape}")

    # Test batch generation
    print("\nTesting batch generation...")
    output_videos = generate_batch(
        model,
        input_video,
        sampler_type='ddim',
        num_inference_steps=10,
        device=device
    )

    print(f"Output videos shape: {output_videos.shape}")

    print("\nGeneration test successful!")
