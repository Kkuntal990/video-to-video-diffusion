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

    # Count parameters
    param_counts = model.count_parameters()
    logger.info(f"Total parameters: {param_counts['total']:,}")
    logger.info(f"  Trainable: {param_counts['trainable']:,}")
    logger.info(f"  VAE: {param_counts['vae']:,} ({param_counts['vae_trainable']:,} trainable)")
    logger.info(f"  U-Net: {param_counts['unet']:,}")

    # Load pretrained VAE weights if configured
    if config.get('pretrained', {}).get('use_pretrained', False):
        from utils.pretrained import (
            load_pretrained_opensora_vae,
            load_pretrained_cogvideox_vae,
            load_pretrained_sd_vae,
            map_sd_vae_to_video_vae
        )

        vae_config = config['pretrained'].get('vae', {})
        # Skip if using custom MAISI VAE (weights already loaded in model constructor)
        if vae_config.get('enabled', False) and not vae_config.get('use_custom_maisi', False):
            model_name = vae_config.get('model_name', 'hpcai-tech/OpenSora-VAE-v1.2')
            inflate_method = vae_config.get('inflate_method', 'central')

            logger.info(f"Loading pretrained VAE from {model_name}...")

            try:
                # Determine which loader to use
                if 'OpenSora' in model_name or 'opensora' in model_name.lower():
                    vae_state_dict = load_pretrained_opensora_vae(model_name)
                elif 'CogVideo' in model_name:
                    vae_state_dict = load_pretrained_cogvideox_vae(model_name)
                elif 'stable' in model_name.lower() or 'sd-' in model_name:
                    vae_state_dict = load_pretrained_sd_vae(model_name)
                    # Inflate 2D->3D for Stable Diffusion VAE
                    vae_state_dict = map_sd_vae_to_video_vae(vae_state_dict, inflate_method)
                else:
                    logger.warning(f"Unknown VAE model: {model_name}, skipping pretrained loading")
                    vae_state_dict = None

                if vae_state_dict is not None:
                    # Load weights into model's VAE
                    missing_keys, unexpected_keys = model.vae.load_state_dict(vae_state_dict, strict=False)

                    if len(missing_keys) > 0:
                        logger.warning(f"Missing keys in VAE: {len(missing_keys)} keys")
                    if len(unexpected_keys) > 0:
                        logger.warning(f"Unexpected keys in VAE: {len(unexpected_keys)} keys")

                    logger.info("✓ Successfully loaded pretrained VAE weights")

            except Exception as e:
                logger.error(f"Failed to load pretrained VAE: {e}")
                logger.info("Continuing with randomly initialized VAE")

    model = model.to(device)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = get_dataloader(config['data'], split='train')

    # Create validation dataloader (15% of data, stratified by category)
    logger.info("Creating validation dataloader...")
    val_dataloader = get_dataloader(config['data'], split='val')

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

    # VAE encoder/decoder parameters - handle custom MAISI VAE structure
    # CRITICAL: Only include parameters with requires_grad=True to avoid GradScaler errors
    if hasattr(model.vae, 'maisi_vae') and model.vae.maisi_vae is not None:
        # Custom MAISI VAE: encoder/decoder are nested in maisi_vae
        vae_encoder_params = [p for p in model.vae.maisi_vae.encoder.parameters() if p.requires_grad]
        if vae_encoder_params:
            param_groups.append({
                'params': vae_encoder_params,
                'lr': base_lr * vae_encoder_mult,
                'name': 'vae_encoder'
            })

        vae_decoder_params = [p for p in model.vae.maisi_vae.decoder.parameters() if p.requires_grad]
        if vae_decoder_params:
            param_groups.append({
                'params': vae_decoder_params,
                'lr': base_lr * vae_decoder_mult,
                'name': 'vae_decoder'
            })
    else:
        # Regular VAE: encoder/decoder are direct attributes
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
        val_dataloader=val_dataloader,
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

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Video-to-Video Diffusion Model')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    main(args)
