"""
Train a custom 3D VAE from scratch for CT slice interpolation.

This script trains ONLY the VAE encoder/decoder without the diffusion model.
Target: PSNR >35 dB reconstruction quality on validation set.

Usage:
    python train_vae.py --config config/vae_training.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.slice_interpolation_dataset import SliceInterpolationDataset, collate_variable_depth
from models.vae import VideoVAE
from utils.metrics import calculate_psnr, calculate_ssim

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoencoderLoss(nn.Module):
    """Combined Autoencoder loss: Reconstruction + Perceptual + MS-SSIM (no KL divergence)"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lambda_recon = config['losses'].get('lambda_recon', 1.0)
        self.lambda_perceptual = config['losses'].get('lambda_perceptual', 0.1)
        self.lambda_ssim = config['losses'].get('lambda_ssim', 0.1)

        self.use_perceptual = config['losses'].get('use_perceptual_loss', True)
        self.use_ssim = config['losses'].get('use_ms_ssim_loss', True)

        # Lazy-load perceptual network (VGG) only if needed
        self.perceptual_net = None
        if self.use_perceptual:
            try:
                import lpips
                self.perceptual_net = lpips.LPIPS(net='vgg').eval()
                logger.info("Loaded LPIPS perceptual loss (VGG)")
            except ImportError:
                logger.warning("lpips not available, skipping perceptual loss")
                self.use_perceptual = False

    def reconstruction_loss(self, pred, target):
        """MSE reconstruction loss"""
        return F.mse_loss(pred, target)

    def perceptual_loss(self, pred, target):
        """LPIPS perceptual loss (VGG features)"""
        if not self.use_perceptual or self.perceptual_net is None:
            return torch.tensor(0.0, device=pred.device)

        # LPIPS expects 4D input (B, C, H, W) with RGB channels
        # For 3D volumes, compute on middle slice
        B, C, D, H, W = pred.shape
        mid_slice = D // 2

        pred_slice = pred[:, :, mid_slice, :, :]  # (B, C, H, W)
        target_slice = target[:, :, mid_slice, :, :]

        # Convert grayscale to RGB (repeat channel 3 times)
        if pred_slice.shape[1] == 1:
            pred_slice = pred_slice.repeat(1, 3, 1, 1)
            target_slice = target_slice.repeat(1, 3, 1, 1)

        # LPIPS expects [-1, 1] range (our data is already in this range)
        with torch.no_grad():
            self.perceptual_net = self.perceptual_net.to(pred.device)

        loss = self.perceptual_net(pred_slice, target_slice).mean()
        return loss

    def ssim_loss(self, pred, target):
        """MS-SSIM loss (multi-scale structural similarity)"""
        if not self.use_ssim:
            return torch.tensor(0.0, device=pred.device)

        # For 3D volumes, compute SSIM on middle slice
        B, C, D, H, W = pred.shape
        mid_slice = D // 2

        pred_slice = pred[:, :, mid_slice, :, :]  # (B, C, H, W)
        target_slice = target[:, :, mid_slice, :, :]

        # Convert to [0, 1] range for SSIM (from [-1, 1])
        pred_slice = (pred_slice + 1.0) / 2.0
        target_slice = (target_slice + 1.0) / 2.0

        # Compute SSIM
        ssim_val = calculate_ssim(pred_slice, target_slice, max_val=1.0)

        # Convert to loss (1 - SSIM)
        loss = 1.0 - ssim_val
        return loss

    def forward(self, pred, target, step):
        """
        Compute combined autoencoder loss.

        Args:
            pred: Reconstructed output (B, C, D, H, W)
            target: Ground truth input (B, C, D, H, W)
            step: Current training step

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        # Reconstruction loss (always computed)
        recon_loss = self.reconstruction_loss(pred, target)

        # Perceptual loss (computed every N steps)
        perceptual_every_n = self.config['losses'].get('perceptual_every_n_steps', 10)
        if step % perceptual_every_n == 0:
            perceptual_loss_val = self.perceptual_loss(pred, target)
        else:
            perceptual_loss_val = torch.tensor(0.0, device=pred.device)

        # SSIM loss (computed every N steps)
        ssim_every_n = self.config['losses'].get('ssim_every_n_steps', 10)
        if step % ssim_every_n == 0:
            ssim_loss_val = self.ssim_loss(pred, target)
        else:
            ssim_loss_val = torch.tensor(0.0, device=pred.device)

        # Combined loss
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_perceptual * perceptual_loss_val +
            self.lambda_ssim * ssim_loss_val
        )

        loss_dict = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'perceptual_loss': to_scalar(perceptual_loss_val),
            'ssim_loss': to_scalar(ssim_loss_val),
        }

        return total_loss, loss_dict


def to_scalar(val):
    """Convert tensor or float to Python scalar"""
    if isinstance(val, torch.Tensor):
        return val.item()
    return float(val)


class VAETrainer:
    """Trainer for custom autoencoder (deterministic VAE)"""

    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Create model
        logger.info("Creating autoencoder model...")
        self.vae = VideoVAE(
            in_channels=config['model']['in_channels'],
            latent_dim=config['model']['latent_dim'],
            base_channels=config['model']['vae_base_channels'],
            scaling_factor=config['model']['vae_scaling_factor'],
        ).to(device)

        logger.info(f"Autoencoder parameters: {sum(p.numel() for p in self.vae.parameters()):,}")

        # Create loss function
        self.criterion = AutoencoderLoss(config).to(device)

        # Create optimizer
        lr = config['training']['learning_rate']
        weight_decay = config['training']['weight_decay']
        self.optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Create scheduler
        self.scheduler = None
        if config['training'].get('scheduler') == 'cosine':
            warmup_steps = config['training'].get('warmup_steps', 500)
            total_steps = config['training']['num_epochs'] * 1000  # Rough estimate
            min_lr = config['training'].get('min_lr', 1e-6)

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=min_lr
            )

        # Mixed precision
        self.use_amp = config['training'].get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        # Output directories
        self.output_dir = Path(config['training']['output_dir'])
        self.log_dir = Path(config['training']['log_dir'])
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])

        # Create experiment subdirectory
        exp_name = config['training']['experiment_name']
        self.output_dir = self.output_dir / exp_name
        self.log_dir = self.log_dir / exp_name
        self.checkpoint_dir = self.checkpoint_dir / exp_name

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.vae.train()

        epoch_losses = []
        epoch_psnr = []
        epoch_ssim = []

        grad_accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        log_interval = self.config['training'].get('log_interval', 50)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Get thick slices as reconstruction target
            # For VAE training, we reconstruct the input
            # Dataset returns: (B, C, D, H, W) where C=1 (already has channel dim)
            thick_slices = batch['thick'].to(self.device)  # (B, 1, D, H, W)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    # Encode → Decode (autoencoder forward pass)
                    recon, z = self.vae(thick_slices)

                    # Compute loss
                    loss, loss_dict = self.criterion(
                        recon, thick_slices, self.global_step
                    )

                    # Scale loss for gradient accumulation
                    loss = loss / grad_accum_steps

                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                # Encode → Decode (autoencoder forward pass)
                recon, z = self.vae(thick_slices)

                # Compute loss
                loss, loss_dict = self.criterion(
                    recon, thick_slices, self.global_step
                )

                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps

                # Backward pass
                loss.backward()

            # Optimizer step (every grad_accum_steps)
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            # Compute metrics
            with torch.no_grad():
                # Convert to [0, 1] for PSNR and SSIM
                recon_norm = (recon + 1.0) / 2.0
                target_norm = (thick_slices + 1.0) / 2.0

                psnr = calculate_psnr(recon_norm, target_norm, max_val=1.0)
                ssim = calculate_ssim(recon_norm, target_norm, max_val=1.0)

                psnr_scalar = to_scalar(psnr)
                ssim_scalar = to_scalar(ssim)

                epoch_psnr.append(psnr_scalar)
                epoch_ssim.append(ssim_scalar)

            epoch_losses.append(loss_dict['loss'])

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['loss']:.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'psnr': f"{psnr_scalar:.2f}",
                'ssim': f"{ssim_scalar:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            # Log
            if self.global_step % log_interval == 0:
                logger.info(
                    f"Step {self.global_step} | "
                    f"Loss: {loss_dict['loss']:.4f} | "
                    f"Recon: {loss_dict['recon_loss']:.4f} | "
                    f"PSNR: {psnr_scalar:.2f} dB | "
                    f"SSIM: {ssim_scalar:.4f}"
                )

        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_psnr = sum(epoch_psnr) / len(epoch_psnr)
        avg_ssim = sum(epoch_ssim) / len(epoch_ssim)

        logger.info(
            f"Epoch {epoch} Summary | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Avg PSNR: {avg_psnr:.2f} dB | "
            f"Avg SSIM: {avg_ssim:.4f}"
        )

        return avg_loss, avg_psnr, avg_ssim

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """Validate reconstruction quality"""
        self.vae.eval()

        val_losses = []
        val_psnr = []
        val_ssim = []

        num_samples = self.config['training'].get('num_validation_samples', 10)

        logger.info(f"Validating on {num_samples} samples...")

        pbar = tqdm(enumerate(val_loader), total=num_samples, desc="Validation")
        for batch_idx, batch in pbar:
            if batch_idx >= num_samples:
                break

            # Get thick slices
            # Dataset returns: (B, C, D, H, W) where C=1 (already has channel dim)
            thick_slices = batch['thick'].to(self.device)  # (B, 1, D, H, W)

            # Forward pass (deterministic)
            recon, z = self.vae(thick_slices)

            # Compute loss
            loss, loss_dict = self.criterion(
                recon, thick_slices, self.global_step
            )

            val_losses.append(loss_dict['loss'])

            # Compute metrics
            recon_norm = (recon + 1.0) / 2.0
            target_norm = (thick_slices + 1.0) / 2.0

            psnr = calculate_psnr(recon_norm, target_norm, max_val=1.0)
            ssim = calculate_ssim(recon_norm, target_norm, max_val=1.0)

            psnr_scalar = to_scalar(psnr)
            ssim_scalar = to_scalar(ssim)

            val_psnr.append(psnr_scalar)
            val_ssim.append(ssim_scalar)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['loss']:.4f}",
                'psnr': f"{psnr_scalar:.2f}",
                'ssim': f"{ssim_scalar:.4f}"
            })

        # Compute averages
        avg_loss = sum(val_losses) / len(val_losses)
        avg_psnr = sum(val_psnr) / len(val_psnr)
        avg_ssim = sum(val_ssim) / len(val_ssim)

        logger.info(
            f"Validation Epoch {epoch} | "
            f"Loss: {avg_loss:.4f} | "
            f"PSNR: {avg_psnr:.2f} dB | "
            f"SSIM: {avg_ssim:.4f}"
        )

        return avg_loss, avg_psnr, avg_ssim

    def save_checkpoint(self, epoch, val_psnr, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_psnr': val_psnr,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"vae_epoch_{epoch}_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "vae_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path} (PSNR: {val_psnr:.2f} dB)")

        # Keep only last N checkpoints
        keep_last_n = self.config['training'].get('keep_last_n_checkpoints', 3)
        checkpoints = sorted(self.checkpoint_dir.glob("vae_epoch_*.pt"))
        if len(checkpoints) > keep_last_n:
            for old_ckpt in checkpoints[:-keep_last_n]:
                old_ckpt.unlink()
                logger.info(f"Deleted old checkpoint: {old_ckpt}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.vae.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✓ Loaded VAE model state")

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("✓ Loaded optimizer state")

        # Load scheduler state if exists
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("✓ Loaded scheduler state")

        # Load scaler state if exists
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info("✓ Loaded gradient scaler state")

        # Restore training state
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)

        val_psnr = checkpoint.get('val_psnr', 0.0)

        logger.info(f"✓ Resumed from epoch {self.current_epoch}, step {self.global_step}")
        logger.info(f"  Previous best PSNR: {val_psnr:.2f} dB")

        return val_psnr

    def train(self, train_loader, val_loader, initial_best_psnr=0.0):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']
        val_interval = self.config['training'].get('val_interval', 1)

        best_psnr = initial_best_psnr

        start_epoch = self.current_epoch + 1 if self.current_epoch > 0 else 1

        if start_epoch > 1:
            logger.info(f"Resuming VAE training from epoch {start_epoch} to {num_epochs}...")
            logger.info(f"Previous best PSNR: {best_psnr:.2f} dB")
        else:
            logger.info(f"Starting VAE training for {num_epochs} epochs...")

        logger.info(f"Total training samples: {len(train_loader.dataset)}")
        logger.info(f"Total validation samples: {len(val_loader.dataset)}")

        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch

            # Train
            train_loss, train_psnr, train_ssim = self.train_epoch(train_loader, epoch)

            # Validate
            if epoch % val_interval == 0:
                logger.info(f"Running validation for epoch {epoch}...")
                val_loss, val_psnr, val_ssim = self.validate(val_loader, epoch)

                # Save checkpoint
                is_best = val_psnr > best_psnr
                if is_best:
                    best_psnr = val_psnr
                    logger.info(f"✓ New best PSNR: {val_psnr:.2f} dB (was {best_psnr:.2f} dB)")

                self.save_checkpoint(epoch, val_psnr, is_best=is_best)

                # Check if target reached
                if val_psnr >= 35.0:
                    logger.info(f"🎉 Target PSNR >35 dB reached! (PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f})")
                    logger.info("VAE training successful. You can stop early if desired.")
            else:
                logger.info(f"Skipping validation for epoch {epoch} (runs every {val_interval} epochs)")

        logger.info(f"Training complete! Best PSNR: {best_psnr:.2f} dB")
        logger.info(f"Best checkpoint saved at: {self.checkpoint_dir / 'vae_best.pt'}")


def main():
    parser = argparse.ArgumentParser(description='Train custom VAE for CT slice interpolation')
    parser.add_argument('--config', type=str, default='config/vae_training.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., /workspace/storage_a100/checkpoints/vae_training/vae_best.pt)')
    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create datasets
    logger.info("Creating datasets...")

    train_dataset = SliceInterpolationDataset(
        data_dir=config['data']['dataset_path'],
        extract_dir=config['data']['extract_dir'],
        categories=config['data']['categories'],
        max_thick_slices=config['data']['max_thick_slices'],
        max_thin_slices=config['data']['max_thin_slices'],
        resolution=config['data']['resolution'],
        window_center=config['data']['window_center'],
        window_width=config['data']['window_width'],
        split='train',
        val_ratio=config['data']['val_split'],
        test_ratio=config['data']['test_split'],
        seed=config['data']['seed'],
    )

    val_dataset = SliceInterpolationDataset(
        data_dir=config['data']['dataset_path'],
        extract_dir=config['data']['extract_dir'],
        categories=config['data']['categories'],
        max_thick_slices=config['data']['max_thick_slices'],
        max_thin_slices=config['data']['max_thin_slices'],
        resolution=config['data']['resolution'],
        window_center=config['data']['window_center'],
        window_width=config['data']['window_width'],
        split='val',
        val_ratio=config['data']['val_split'],
        test_ratio=config['data']['test_split'],
        seed=config['data']['seed'],
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create dataloaders with custom collate function for variable depths
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=config['data']['drop_last'],
        collate_fn=collate_variable_depth,  # Handle variable slice depths
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_variable_depth,  # Handle variable slice depths
    )

    # Create trainer
    trainer = VAETrainer(config, device)

    # Load checkpoint if resuming
    initial_best_psnr = 0.0
    if args.resume:
        if not os.path.exists(args.resume):
            logger.error(f"Checkpoint not found: {args.resume}")
            logger.error("Please provide a valid checkpoint path with --resume")
            return

        initial_best_psnr = trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(train_loader, val_loader, initial_best_psnr=initial_best_psnr)


if __name__ == '__main__':
    main()
