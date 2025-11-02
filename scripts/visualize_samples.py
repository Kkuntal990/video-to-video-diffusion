#!/usr/bin/env python3
"""
Generate and Visualize Sample Predictions

This script:
1. Loads a trained model checkpoint
2. Generates predictions on a few validation/test samples
3. Creates side-by-side visualizations (input, ground truth, prediction)
4. Saves images for visual quality inspection

Usage:
    python scripts/visualize_samples.py \
        --checkpoint /workspace/storage_a100/checkpoints/ape_v2v_diffusion/checkpoint_best.pt \
        --config config/cloud_train_config_a100.yaml \
        --output-dir /workspace/storage_a100/visualizations \
        --split val \
        --num-samples 5
"""

import argparse
import yaml
import torch
import logging
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import VideoToVideoDiffusion
from data import get_unified_dataloader as get_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def tensor_to_image(tensor):
    """
    Convert tensor to numpy image

    Args:
        tensor: (C, H, W) tensor in range [0, 1]
               C=1 for grayscale (medical imaging), C=3 for RGB

    Returns:
        image: (H, W, C) numpy array in range [0, 255]
               For grayscale, C=3 (repeated for visualization)
    """
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy: (C, H, W) -> (H, W, C)
    image = tensor.cpu().numpy().transpose(1, 2, 0)

    # If grayscale (C=1), repeat to RGB for visualization
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)  # (H, W, 1) -> (H, W, 3)

    # Convert to [0, 255]
    image = (image * 255).astype(np.uint8)

    return image


def save_video_frames(video_tensor, save_path, prefix="frame"):
    """
    Save all frames from a video tensor

    Args:
        video_tensor: (C, T, H, W) tensor
        save_path: Directory to save frames
        prefix: Prefix for frame filenames
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    C, T, H, W = video_tensor.shape

    for t in range(T):
        frame = video_tensor[:, t, :, :]  # (C, H, W)
        image = tensor_to_image(frame)

        # Save frame
        frame_path = save_path / f"{prefix}_{t:03d}.png"
        Image.fromarray(image).save(frame_path)

    logger.info(f"Saved {T} frames to {save_path}")


def create_comparison_grid(input_video, gt_video, pred_video, save_path, sample_idx):
    """
    Create a comparison grid showing input, ground truth, and prediction

    Args:
        input_video: (C, T, H, W) input video tensor
        gt_video: (C, T, H, W) ground truth video tensor
        pred_video: (C, T, H, W) prediction video tensor
        save_path: Path to save the grid
        sample_idx: Sample index for filename
    """
    C, T, H, W = input_video.shape

    # Select middle frame and first/last for visualization
    frame_indices = [0, T//2, T-1]

    # Create figure with 3 columns (input, gt, pred) and 3 rows (start, middle, end)
    fig, axes = plt.subplots(len(frame_indices), 3, figsize=(12, 4*len(frame_indices)))

    row_labels = ['First Frame', 'Middle Frame', 'Last Frame']
    col_labels = ['Input (Low-Res)', 'Ground Truth', 'Prediction']

    for row_idx, t in enumerate(frame_indices):
        # Input frame
        input_img = tensor_to_image(input_video[:, t, :, :])
        axes[row_idx, 0].imshow(input_img, cmap='gray' if C == 1 else None)
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_labels[0], fontsize=14, fontweight='bold')
        axes[row_idx, 0].set_ylabel(row_labels[row_idx], fontsize=12, fontweight='bold')

        # Ground truth frame
        gt_img = tensor_to_image(gt_video[:, t, :, :])
        axes[row_idx, 1].imshow(gt_img, cmap='gray' if C == 1 else None)
        axes[row_idx, 1].axis('off')
        if row_idx == 0:
            axes[row_idx, 1].set_title(col_labels[1], fontsize=14, fontweight='bold')

        # Prediction frame
        pred_img = tensor_to_image(pred_video[:, t, :, :])
        axes[row_idx, 2].imshow(pred_img, cmap='gray' if C == 1 else None)
        axes[row_idx, 2].axis('off')
        if row_idx == 0:
            axes[row_idx, 2].set_title(col_labels[2], fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved comparison grid to {save_path}")


def visualize_samples(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    split: str = 'val',
    num_samples: int = 5,
    num_inference_steps: int = 50,
    device: str = 'cuda'
):
    """
    Generate and visualize sample predictions

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        output_dir: Directory to save visualizations
        split: Dataset split ('val' or 'test')
        num_samples: Number of samples to visualize
        num_inference_steps: Number of diffusion steps
        device: Device to run on
    """
    logger.info(f"{'='*70}")
    logger.info(f"Sample Visualization")
    logger.info(f"{'='*70}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Split: {split}")
    logger.info(f"Samples to generate: {num_samples}")
    logger.info(f"Inference steps: {num_inference_steps}")

    # Load model
    logger.info("\nLoading model from checkpoint...")
    model, checkpoint = VideoToVideoDiffusion.load_checkpoint(checkpoint_path, device=device)
    model.eval()

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load dataloader
    logger.info(f"\nLoading {split} dataset...")
    dataloader = get_dataloader(config['data'], split=split)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nGenerating predictions...")

    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break

            v_in = batch['input'].to(device)
            v_gt = batch['target'].to(device)
            categories = batch['category']
            patient_ids = batch['patient_id']

            # Generate predictions
            logger.info(f"\nProcessing batch {batch_idx + 1}...")
            v_pred = model.generate(v_in, sampler='ddim', num_inference_steps=num_inference_steps)

            # Process each sample in batch
            batch_size = v_in.size(0)
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break

                # Get sample data
                input_video = v_in[i]  # (C, T, H, W)
                gt_video = v_gt[i]
                pred_video = v_pred[i]
                category = categories[i] if isinstance(categories, list) else categories
                patient_id = patient_ids[i] if isinstance(patient_ids, list) else patient_ids

                logger.info(f"  Sample {sample_count + 1}: {patient_id} ({category})")

                # Create sample directory
                sample_dir = output_path / f"sample_{sample_count:03d}_{patient_id}_{category.replace(' ', '_')}"
                sample_dir.mkdir(exist_ok=True)

                # Save comparison grid
                grid_path = sample_dir / "comparison_grid.png"
                create_comparison_grid(input_video, gt_video, pred_video, grid_path, sample_count)

                # Save all frames as individual images
                save_video_frames(input_video, sample_dir / "input", "input")
                save_video_frames(gt_video, sample_dir / "ground_truth", "gt")
                save_video_frames(pred_video, sample_dir / "prediction", "pred")

                # Save metadata
                metadata_path = sample_dir / "metadata.txt"
                with open(metadata_path, 'w') as f:
                    f.write(f"Patient ID: {patient_id}\n")
                    f.write(f"Category: {category}\n")
                    f.write(f"Split: {split}\n")
                    f.write(f"Checkpoint Epoch: {checkpoint.get('epoch', 'unknown')}\n")
                    f.write(f"Inference Steps: {num_inference_steps}\n")
                    f.write(f"Video Shape: {input_video.shape}\n")

                sample_count += 1

    logger.info(f"\n{'='*70}")
    logger.info(f"Visualization Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Generated {sample_count} samples")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"\nEach sample contains:")
    logger.info(f"  - comparison_grid.png: Side-by-side comparison")
    logger.info(f"  - input/: Individual input frames")
    logger.info(f"  - ground_truth/: Individual ground truth frames")
    logger.info(f"  - prediction/: Individual prediction frames")
    logger.info(f"  - metadata.txt: Sample information")


def main():
    parser = argparse.ArgumentParser(description='Generate and visualize sample predictions')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/cloud_train_config_a100.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['val', 'test'],
        help='Dataset split to visualize'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--inference-steps',
        type=int,
        default=50,
        help='Number of diffusion inference steps'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (cuda or cpu)'
    )

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    # Check config exists
    if not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        return

    # Run visualization
    visualize_samples(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        split=args.split,
        num_samples=args.num_samples,
        num_inference_steps=args.inference_steps,
        device=args.device
    )


if __name__ == "__main__":
    main()
