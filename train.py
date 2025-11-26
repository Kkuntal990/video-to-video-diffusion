"""
Main training script for Video-to-Video Diffusion Model
"""

import warnings
import logging

# Suppress pydicom warnings about character encodings (GB18030, etc.)
# These are harmless warnings from DICOM files with Chinese text
warnings.filterwarnings('ignore', message='.*cannot be used as code extension.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pydicom')

# Also suppress pydicom logging warnings
logging.getLogger('pydicom').setLevel(logging.ERROR)

import torch
import argparse
import yaml
import os
from pathlib import Path
import random
import numpy as np

from models import VideoToVideoDiffusion
from data import get_unified_dataloader as get_dataloader
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
    log_dir = Path(config['training']['log_dir']) / config['training']['experiment_name']
    logger = setup_logger('training', log_file=log_dir / 'train.log')

    logger.info("=" * 80)
    logger.info("Video-to-Video Diffusion Model Training")
    logger.info("=" * 80)

    # Device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Create model
    logger.info("Creating model...")
    model = VideoToVideoDiffusion(config)

    # Load pretrained VAE weights if configured
    if config.get('pretrained', {}).get('use_pretrained', False):
        vae_config = config['pretrained'].get('vae', {})
        if vae_config.get('enabled', False):
            checkpoint_path = vae_config.get('checkpoint_path', None)

            # Check if loading from local checkpoint file (custom trained VAE)
            if checkpoint_path and os.path.exists(checkpoint_path):
                logger.info(f"Loading custom trained VAE from local checkpoint: {checkpoint_path}")

                try:
                    # Load checkpoint
                    vae_checkpoint = torch.load(checkpoint_path, map_location='cpu')

                    # Handle different checkpoint formats
                    if 'model_state_dict' in vae_checkpoint:
                        vae_state_dict = vae_checkpoint['model_state_dict']
                    elif 'vae_state_dict' in vae_checkpoint:
                        vae_state_dict = vae_checkpoint['vae_state_dict']
                    elif 'state_dict' in vae_checkpoint:
                        vae_state_dict = vae_checkpoint['state_dict']
                    else:
                        # Assume checkpoint is a direct state dict
                        vae_state_dict = vae_checkpoint

                    # Load weights into model's VAE
                    missing_keys, unexpected_keys = model.vae.load_state_dict(vae_state_dict, strict=False)

                    logger.info(f"✓ Loaded custom VAE checkpoint: {len(vae_state_dict)} keys")
                    if len(missing_keys) > 0:
                        logger.warning(f"  Missing keys: {len(missing_keys)}")
                    if len(unexpected_keys) > 0:
                        logger.warning(f"  Unexpected keys: {len(unexpected_keys)}")

                except Exception as e:
                    logger.error(f"Failed to load VAE checkpoint: {e}")
                    logger.info("Continuing with randomly initialized VAE")
            else:
                logger.warning("No checkpoint_path specified for VAE, using random initialization")

        # Freeze VAE parameters (more efficient than lr=0.0)
        # This saves memory (no gradient storage) and speeds up training
        for param in model.vae.parameters():
            param.requires_grad = False
        logger.info("✓ Froze all VAE parameters (requires_grad=False)")

    # Count parameters AFTER freezing VAE to show correct trainable counts
    param_counts = model.count_parameters()
    logger.info(f"Total parameters: {param_counts['total']:,}")
    logger.info(f"  Trainable: {param_counts['trainable']:,}")
    logger.info(f"  VAE: {param_counts['vae']:,} ({param_counts['vae_trainable']:,} trainable)")
    logger.info(f"  U-Net: {param_counts['unet']:,}")

    # Note: Model will be moved to device in Trainer.__init__ for better encapsulation

    # Create dataloaders for multi-tier validation strategy
    logger.info("Creating dataloaders...")
    use_patches = config['data'].get('use_patches', False)

    # Training dataloader (uses config as-is)
    if use_patches:
        logger.info("Using PATCH-BASED training mode")
        logger.info(f"  Patch size: {config['data'].get('patch_depth_thin', 48)} thin × {config['data'].get('patch_depth_thick', 8)} thick @ {config['data'].get('patch_size', [192, 192])}")
    else:
        logger.info("Using FULL-VOLUME training mode")

    train_dataloader = get_dataloader(config['data'], split='train')

    # Patch validation dataloader (Tier 1 & 2: loss-only and patch-based validation)
    logger.info("Creating patch validation dataloader...")
    patch_val_config = config['data'].copy()
    patch_val_config['use_patches'] = True  # Always use patches for Tier 1 & 2
    patch_val_config['batch_size'] = config['data'].get('batch_size', 8)
    patch_val_dataloader = get_dataloader(patch_val_config, split='val')

    # Full-volume validation dataloader (Tier 3: full-volume validation)
    # Only create if full validation is enabled (interval < 1000)
    full_val_interval = config['training'].get('full_val_interval', 999999)
    if full_val_interval < 1000:
        logger.info("Creating full-volume validation dataloader...")
        full_val_config = config['data'].copy()
        full_val_config['use_patches'] = False  # Full volumes for Tier 3
        full_val_config['batch_size'] = 1  # Must use small batch for full volumes
        full_val_dataloader = get_dataloader(full_val_config, split='val')
    else:
        logger.info(f"Full-volume validation disabled (interval={full_val_interval}), skipping dataloader creation")
        full_val_dataloader = None

    # Note: Streaming datasets don't support len()
    try:
        logger.info(f"Training batches: {len(train_dataloader)}")
    except TypeError:
        logger.info("Training batches: Unknown (streaming dataset)")

    # Create optimizer with layer-wise learning rates
    optimizer_name = config['training'].get('optimizer', 'adam').lower()
    base_lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training'].get('weight_decay', 0.0))

    # Setup parameter groups with layer-wise learning rates
    param_groups = []

    # Get layer-wise LR multipliers from config
    lr_multipliers = config.get('pretrained', {}).get('layer_lr_multipliers', {})
    vae_encoder_mult = lr_multipliers.get('vae_encoder', 1.0)
    vae_decoder_mult = lr_multipliers.get('vae_decoder', 1.0)
    unet_mult = lr_multipliers.get('unet', 1.0)

    # VAE encoder/decoder parameters
    # CRITICAL: Only include parameters with requires_grad=True to avoid GradScaler errors
    vae_encoder_params = [p for p in model.vae.encoder.parameters() if p.requires_grad]
    if vae_encoder_params:
        param_groups.append({
            'params': vae_encoder_params,
            'lr': base_lr * vae_encoder_mult,
            'name': 'vae_encoder'
        })

    vae_decoder_params = [p for p in model.vae.decoder.parameters() if p.requires_grad]
    if vae_decoder_params:
        param_groups.append({
            'params': vae_decoder_params,
            'lr': base_lr * vae_decoder_mult,
            'name': 'vae_decoder'
        })

    # U-Net parameters - filter to trainable only for consistency
    unet_params = [p for p in model.unet.parameters() if p.requires_grad]
    if unet_params:
        param_groups.append({
            'params': unet_params,
            'lr': base_lr * unet_mult,
            'name': 'unet'
        })

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    logger.info(f"Optimizer: {optimizer_name}, Base LR: {base_lr}")
    logger.info(f"Optimizer param groups: {len(param_groups)}")
    for i, pg in enumerate(param_groups):
        num_params = sum(p.numel() for p in pg['params'])
        logger.info(f"  Group {i} ({pg.get('name', 'unnamed')}): {num_params:,} params, LR: {pg['lr']:.2e}")
    logger.info(f"Note: VAE frozen param groups are excluded from optimizer")

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
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / config['training']['experiment_name']

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=patch_val_dataloader,  # Patch-based validation (Tier 1 & 2)
        full_val_dataloader=full_val_dataloader,  # Full-volume validation (Tier 3)
        config=config['training'],
        device=device,
        log_dir=str(log_dir),
        checkpoint_dir=str(checkpoint_dir),
        pretrained_config=config.get('pretrained', {})
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    else:
        # Auto-detect best checkpoint in checkpoint directory
        # If model_suffix is specified, look for checkpoints with that suffix
        checkpoint_dir_path = Path(checkpoint_dir)
        model_suffix = config['training'].get('model_suffix', '')

        if model_suffix:
            # Look for checkpoints with this specific suffix
            pattern = f'checkpoint_best_*_{model_suffix}.pt'
            logger.info(f"Looking for checkpoints with suffix: {model_suffix}")
        else:
            # Look for any best checkpoint
            pattern = 'checkpoint_best_*.pt'

        best_checkpoints = list(checkpoint_dir_path.glob(pattern))
        if best_checkpoints:
            # Find the best checkpoint (most recent if multiple)
            best_checkpoint = max(best_checkpoints, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found existing best checkpoint: {best_checkpoint}")
            logger.info(f"Resuming from best checkpoint...")
            trainer.load_checkpoint(str(best_checkpoint))
        else:
            logger.info("No existing checkpoint found. Starting training from scratch.")

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Final comprehensive validation on all validation samples
    if config['training'].get('final_val_enabled', True):
        logger.info("=" * 80)
        logger.info("Running FINAL validation on all validation samples...")
        logger.info("=" * 80)

        use_full_volumes = config['training'].get('final_val_full_volumes', True)
        final_dataloader = full_val_dataloader if use_full_volumes else patch_val_dataloader

        trainer.final_validate(
            dataloader=final_dataloader,
            use_full_volumes=use_full_volumes,
            max_samples=None  # No limit - use ALL validation data
        )

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Video-to-Video Diffusion Model')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    main(args)
