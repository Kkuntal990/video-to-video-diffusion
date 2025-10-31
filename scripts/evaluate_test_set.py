#!/usr/bin/env python3
"""
Evaluate Model on Test Set

This script:
1. Loads a trained model checkpoint
2. Runs inference on the held-out test set (10% of data)
3. Calculates SSIM, PSNR metrics
4. Saves results for model comparison

Usage:
    python scripts/evaluate_test_set.py \
        --checkpoint /path/to/checkpoint_best.pt \
        --config config/cloud_train_config_a100.yaml \
        --output results/test_evaluation.json
"""

import argparse
import yaml
import torch
import json
import logging
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import VideoToVideoDiffusion
from data import get_unified_dataloader as get_dataloader
from utils.metrics import calculate_video_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_test_set(
    checkpoint_path: str,
    config_path: str,
    output_path: str,
    num_inference_steps: int = 50,
    device: str = 'cuda'
):
    """
    Evaluate model on test set

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        output_path: Path to save results JSON
        num_inference_steps: Number of diffusion steps for inference
        device: Device to run on
    """
    logger.info(f"{'='*70}")
    logger.info(f"Test Set Evaluation")
    logger.info(f"{'='*70}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Device: {device}")

    # Load model
    logger.info("Loading model from checkpoint...")
    model, checkpoint = VideoToVideoDiffusion.load_checkpoint(checkpoint_path, device=device)
    model.eval()

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Checkpoint training loss: {checkpoint.get('best_loss', 'unknown'):.4f}")

    # Load test dataloader
    logger.info("Loading test dataset...")
    test_dataloader = get_dataloader(config['data'], split='test')

    logger.info(f"Test set size: {len(test_dataloader.dataset)} samples")
    logger.info(f"Test batches: {len(test_dataloader)}")

    # Run evaluation
    logger.info(f"\nRunning inference on test set...")
    logger.info(f"Inference steps: {num_inference_steps}")

    results = {
        'checkpoint': str(checkpoint_path),
        'epoch': checkpoint.get('epoch', None),
        'training_loss': float(checkpoint.get('best_loss', 0)),
        'num_inference_steps': num_inference_steps,
        'test_samples': len(test_dataloader.dataset),
        'per_sample_results': [],
        'category_results': {}
    }

    all_psnr = []
    all_ssim = []
    category_metrics = {}

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            v_in = batch['input'].to(device)
            v_gt = batch['target'].to(device)
            categories = batch['category']
            patient_ids = batch['patient_id']

            # Generate predictions
            v_pred = model.generate(v_in, sampler='ddim', num_inference_steps=num_inference_steps)

            # Calculate metrics for each sample in batch
            for i in range(v_in.size(0)):
                pred = v_pred[i:i+1]
                gt = v_gt[i:i+1]
                category = categories[i] if isinstance(categories, list) else categories
                patient_id = patient_ids[i] if isinstance(patient_ids, list) else patient_ids

                # Clamp to [0, 1]
                pred_clamped = torch.clamp(pred, 0, 1)
                gt_clamped = torch.clamp(gt, 0, 1)

                # Calculate metrics
                metrics = calculate_video_metrics(pred_clamped, gt_clamped, max_val=1.0)

                all_psnr.append(metrics['psnr'])
                all_ssim.append(metrics['ssim'])

                # Track by category
                if category not in category_metrics:
                    category_metrics[category] = {'psnr': [], 'ssim': []}
                category_metrics[category]['psnr'].append(metrics['psnr'])
                category_metrics[category]['ssim'].append(metrics['ssim'])

                # Store per-sample result
                results['per_sample_results'].append({
                    'patient_id': patient_id,
                    'category': category,
                    'psnr': float(metrics['psnr']),
                    'ssim': float(metrics['ssim'])
                })

    # Calculate aggregate statistics
    results['overall'] = {
        'psnr_mean': float(sum(all_psnr) / len(all_psnr)),
        'psnr_std': float(torch.tensor(all_psnr).std().item()),
        'ssim_mean': float(sum(all_ssim) / len(all_ssim)),
        'ssim_std': float(torch.tensor(all_ssim).std().item()),
        'num_samples': len(all_psnr)
    }

    # Calculate per-category statistics
    for category, metrics in category_metrics.items():
        results['category_results'][category] = {
            'psnr_mean': float(sum(metrics['psnr']) / len(metrics['psnr'])),
            'psnr_std': float(torch.tensor(metrics['psnr']).std().item()),
            'ssim_mean': float(sum(metrics['ssim']) / len(metrics['ssim'])),
            'ssim_std': float(torch.tensor(metrics['ssim']).std().item()),
            'num_samples': len(metrics['psnr'])
        }

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"Test Set Results")
    logger.info(f"{'='*70}")
    logger.info(f"Overall:")
    logger.info(f"  PSNR: {results['overall']['psnr_mean']:.2f} ± {results['overall']['psnr_std']:.2f} dB")
    logger.info(f"  SSIM: {results['overall']['ssim_mean']:.4f} ± {results['overall']['ssim_std']:.4f}")

    logger.info(f"\nBy Category:")
    for category, stats in results['category_results'].items():
        logger.info(f"  {category}:")
        logger.info(f"    PSNR: {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f} dB ({stats['num_samples']} samples)")
        logger.info(f"    SSIM: {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}")

    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
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
        '--output',
        type=str,
        default='results/test_evaluation.json',
        help='Path to save results JSON'
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

    # Run evaluation
    evaluate_test_set(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        num_inference_steps=args.inference_steps,
        device=args.device
    )


if __name__ == "__main__":
    main()
