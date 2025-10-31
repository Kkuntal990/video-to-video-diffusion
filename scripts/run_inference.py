#!/usr/bin/env python3
"""
Run Inference on Validation and Test Sets

This script:
1. Loads a trained model checkpoint
2. Runs inference on both validation (15%) and test (10%) sets
3. Calculates SSIM, PSNR metrics
4. Saves results for model evaluation

Usage:
    python scripts/run_inference.py \
        --checkpoint /workspace/storage_a100/checkpoints/checkpoint_best.pt \
        --config config/cloud_train_config_a100.yaml \
        --output-dir /workspace/storage_a100/inference_results
"""

import argparse
import yaml
import torch
import json
import logging
from pathlib import Path
from tqdm import tqdm
import sys
from datetime import datetime

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


def evaluate_split(
    model,
    dataloader,
    split_name: str,
    num_inference_steps: int,
    device: str
):
    """
    Evaluate model on a specific split

    Args:
        model: Trained model
        dataloader: DataLoader for the split
        split_name: Name of split ('val' or 'test')
        num_inference_steps: Number of diffusion steps for inference
        device: Device to run on

    Returns:
        results: Dictionary with evaluation results
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating {split_name} set...")
    logger.info(f"{'='*70}")
    logger.info(f"Dataset size: {len(dataloader.dataset)} samples")
    logger.info(f"Batches: {len(dataloader)}")
    logger.info(f"Inference steps: {num_inference_steps}")

    results = {
        'split': split_name,
        'num_samples': len(dataloader.dataset),
        'num_inference_steps': num_inference_steps,
        'per_sample_results': [],
        'category_results': {}
    }

    all_psnr = []
    all_ssim = []
    category_metrics = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
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

    # Log results
    logger.info(f"\n{split_name.upper()} Set Results:")
    logger.info(f"{'='*70}")
    logger.info(f"Overall:")
    logger.info(f"  PSNR: {results['overall']['psnr_mean']:.2f} ± {results['overall']['psnr_std']:.2f} dB")
    logger.info(f"  SSIM: {results['overall']['ssim_mean']:.4f} ± {results['overall']['ssim_std']:.4f}")

    logger.info(f"\nBy Category:")
    for category, stats in results['category_results'].items():
        logger.info(f"  {category}:")
        logger.info(f"    PSNR: {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f} dB ({stats['num_samples']} samples)")
        logger.info(f"    SSIM: {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}")

    return results


def run_inference(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    num_inference_steps: int = 50,
    device: str = 'cuda',
    splits: list = ['val', 'test']
):
    """
    Run inference on validation and test sets

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        output_dir: Directory to save results
        num_inference_steps: Number of diffusion steps for inference
        device: Device to run on
        splits: List of splits to evaluate ('val', 'test')
    """
    logger.info(f"{'='*70}")
    logger.info(f"Inference Evaluation")
    logger.info(f"{'='*70}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Splits: {splits}")

    # Load model
    logger.info("\nLoading model from checkpoint...")
    model, checkpoint = VideoToVideoDiffusion.load_checkpoint(checkpoint_path, device=device)
    model.eval()

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'best_loss' in checkpoint:
        logger.info(f"Checkpoint training loss: {checkpoint.get('best_loss'):.4f}")

    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Store overall results
    all_results = {
        'checkpoint': str(checkpoint_path),
        'epoch': checkpoint.get('epoch', None),
        'training_loss': float(checkpoint.get('best_loss', 0)) if 'best_loss' in checkpoint else None,
        'num_inference_steps': num_inference_steps,
        'timestamp': datetime.now().isoformat(),
        'splits': {}
    }

    # Evaluate each split
    for split in splits:
        logger.info(f"\nLoading {split} dataset...")
        try:
            dataloader = get_dataloader(config['data'], split=split)

            # Run evaluation
            split_results = evaluate_split(
                model=model,
                dataloader=dataloader,
                split_name=split,
                num_inference_steps=num_inference_steps,
                device=device
            )

            all_results['splits'][split] = split_results

            # Save individual split results
            split_output = output_path / f'{split}_results.json'
            with open(split_output, 'w') as f:
                json.dump(split_results, f, indent=2)
            logger.info(f"\n{split} results saved to: {split_output}")

        except Exception as e:
            logger.error(f"Error evaluating {split} set: {e}")
            import traceback
            traceback.print_exc()

    # Save combined results
    combined_output = output_path / 'combined_results.json'
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"Inference Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Combined results saved to: {combined_output}")
    logger.info(f"Output directory: {output_path}")

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Summary:")
    logger.info(f"{'='*70}")
    for split, split_results in all_results['splits'].items():
        overall = split_results.get('overall', {})
        logger.info(f"\n{split.upper()}:")
        logger.info(f"  Samples: {overall.get('num_samples', 'N/A')}")
        logger.info(f"  PSNR: {overall.get('psnr_mean', 0):.2f} ± {overall.get('psnr_std', 0):.2f} dB")
        logger.info(f"  SSIM: {overall.get('ssim_mean', 0):.4f} ± {overall.get('ssim_std', 0):.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run inference on validation and test sets')
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
        default='results/inference',
        help='Directory to save results'
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
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['val', 'test'],
        choices=['val', 'test'],
        help='Splits to evaluate (default: val test)'
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

    # Run inference
    run_inference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        num_inference_steps=args.inference_steps,
        device=args.device,
        splits=args.splits
    )


if __name__ == "__main__":
    main()
