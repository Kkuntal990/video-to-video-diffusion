"""
Main inference script for Video-to-Video Diffusion Model
"""

import torch
import argparse
import yaml
from pathlib import Path

from models import VideoToVideoDiffusion
from inference.generate import generate_video
from utils import setup_logger


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = {}

    # Setup logger
    logger = setup_logger('inference')

    logger.info("=" * 80)
    logger.info("Video-to-Video Diffusion Model Inference")
    logger.info("=" * 80)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model, checkpoint = VideoToVideoDiffusion.load_checkpoint(args.checkpoint, device)

    # Model info
    param_counts = model.count_parameters()
    logger.info(f"Total parameters: {param_counts['total']:,}")

    # Get inference settings
    sampler_type = args.sampler if args.sampler else config.get('inference', {}).get('sampler_type', 'ddim')
    num_steps = args.steps if args.steps else config.get('inference', {}).get('num_inference_steps', 20)
    num_frames = args.num_frames if args.num_frames else config.get('data', {}).get('num_frames', 16)
    resolution = tuple(args.resolution) if args.resolution else tuple(config.get('data', {}).get('resolution', [256, 256]))

    logger.info(f"Sampler: {sampler_type}")
    logger.info(f"Inference steps: {num_steps}")
    logger.info(f"Frames: {num_frames}")
    logger.info(f"Resolution: {resolution}")

    # Generate video
    logger.info(f"\nProcessing input video: {args.input}")

    output_video = generate_video(
        model=model,
        input_video_path=args.input,
        output_path=args.output,
        sampler_type=sampler_type,
        num_inference_steps=num_steps,
        device=device,
        num_frames=num_frames,
        resolution=resolution
    )

    logger.info(f"✓ Output video saved to: {args.output}")
    logger.info("\nInference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video-to-Video Diffusion Inference')

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save output video')

    # Optional arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--sampler', type=str, default=None, choices=['ddim', 'ddpm'],
                       help='Sampling method (default: from config or ddim)')
    parser.add_argument('--steps', type=int, default=None,
                       help='Number of inference steps (default: from config or 20)')
    parser.add_argument('--num-frames', type=int, default=None,
                       help='Number of frames to process (default: from config or 16)')
    parser.add_argument('--resolution', type=int, nargs=2, default=None,
                       help='Resolution [height width] (default: from config or [256 256])')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    main(args)
