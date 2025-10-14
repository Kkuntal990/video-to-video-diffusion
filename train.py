"""
Main training script for Video-to-Video Diffusion Model
"""

import torch
import argparse
import yaml
from pathlib import Path
import random
import numpy as np

from models import VideoToVideoDiffusion
from data import get_dataloader
from training import Trainer, get_scheduler
from utils import setup_logger


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_seed(config.get('seed', 42))

    # Setup logger
    log_dir = Path(config['training']['log_dir']) / config['experiment_name']
    logger = setup_logger('training', log_file=log_dir / 'train.log')

    logger.info("=" * 80)
    logger.info("Video-to-Video Diffusion Model Training")
    logger.info("=" * 80)

    # Device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Create model
    logger.info("Creating model...")
    model = VideoToVideoDiffusion(config['model'])

    # Count parameters
    param_counts = model.count_parameters()
    logger.info(f"Total parameters: {param_counts['total']:,}")
    logger.info(f"  VAE: {param_counts['vae']:,}")
    logger.info(f"  U-Net: {param_counts['unet']:,}")

    model = model.to(device)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = get_dataloader(config['data'], split='train')
    val_dataloader = None  # Add validation dataloader if needed

    logger.info(f"Training batches: {len(train_dataloader)}")

    # Create optimizer
    optimizer_name = config['training'].get('optimizer', 'adam').lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.0)

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    logger.info(f"Optimizer: {optimizer_name}, LR: {lr}")

    # Create scheduler
    scheduler_config = {
        'scheduler_type': config['training'].get('scheduler_type', 'cosine'),
        'num_epochs': config['training']['num_epochs'],
        'warmup_epochs': config['training'].get('warmup_epochs', 5),
        'min_lr': config['training'].get('min_lr', 1e-6)
    }
    scheduler = get_scheduler(optimizer, scheduler_config)
    logger.info(f"Scheduler: {scheduler_config['scheduler_type']}")

    # Create trainer
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / config['experiment_name']

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config['training'],
        device=device,
        log_dir=str(log_dir),
        checkpoint_dir=str(checkpoint_dir)
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Video-to-Video Diffusion Model')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    main(args)
