"""
Training loop for Video-to-Video Diffusion Model
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from pathlib import Path
import time


class Trainer:
    """
    Trainer for Video-to-Video Diffusion Model

    Features:
    - Mixed precision training
    - Gradient accumulation
    - Checkpoint saving
    - TensorBoard logging
    - Validation
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader=None,
        config=None,
        device='cuda',
        log_dir='logs',
        checkpoint_dir='checkpoints',
        pretrained_config=None
    ):
        """
        Args:
            model: VideoToVideoDiffusion model
            optimizer: optimizer
            scheduler: learning rate scheduler
            train_dataloader: training data loader
            val_dataloader: validation data loader (optional)
            config: training config dict
            device: training device
            log_dir: directory for TensorBoard logs
            checkpoint_dir: directory for checkpoints
            pretrained_config: pretrained weights config dict
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.config = config or {}
        self.pretrained_config = pretrained_config or {}

        # Training settings
        self.num_epochs = self.config.get('num_epochs', 100)

        # Two-phase training settings
        self.use_two_phase = self.pretrained_config.get('two_phase_training', False)
        self.phase1_epochs = self.pretrained_config.get('phase1_epochs', 1)
        self.vae_freeze_epochs = self.pretrained_config.get('vae', {}).get('freeze_epochs', 0)
        self.current_phase = 1 if self.use_two_phase else 0  # 0=no phases, 1=phase1, 2=phase2

        # Log phase information
        if self.use_two_phase:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Two-phase training enabled:")
            logger.info(f"  Phase 1: {self.phase1_epochs} epochs (VAE frozen)")
            logger.info(f"  Phase 2: {self.num_epochs - self.phase1_epochs} epochs (fine-tune all)")
        elif self.vae_freeze_epochs > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"VAE will be frozen for first {self.vae_freeze_epochs} epochs")
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.use_amp = self.config.get('use_amp', True)
        self.log_interval = self.config.get('log_interval', 10)
        self.val_interval = self.config.get('val_interval', 1000)
        self.checkpoint_interval = self.config.get('checkpoint_interval', 5000)

        # Setup directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Num epochs: {self.num_epochs}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Log dir: {self.log_dir}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")

    def freeze_vae(self):
        """Freeze VAE parameters (encoder and decoder)"""
        for param in self.model.vae.parameters():
            param.requires_grad = False

        import logging
        logger = logging.getLogger(__name__)
        logger.info("🔒 VAE frozen (parameters set to requires_grad=False)")
        print("🔒 VAE frozen")

    def unfreeze_vae(self):
        """Unfreeze VAE parameters (encoder and decoder)"""
        for param in self.model.vae.parameters():
            param.requires_grad = True

        import logging
        logger = logging.getLogger(__name__)
        logger.info("🔓 VAE unfrozen (parameters set to requires_grad=True)")
        print("🔓 VAE unfrozen")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            v_in = batch['input'].to(self.device)
            v_gt = batch['target'].to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    loss, metrics = self.model(v_in, v_gt)
                    loss = loss / self.gradient_accumulation_steps
            else:
                loss, metrics = self.model(v_in, v_gt)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Update global step
            self.global_step += 1

            # Log metrics
            loss_value = loss.item() * self.gradient_accumulation_steps
            epoch_losses.append(loss_value)

            if self.global_step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/loss', loss_value, self.global_step)
                self.writer.add_scalar('train/lr', lr, self.global_step)

                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'lr': f'{lr:.2e}',
                    'step': self.global_step
                })

            # Validation
            if self.val_dataloader and self.global_step % self.val_interval == 0:
                val_loss = self.validate()
                self.writer.add_scalar('val/loss', val_loss, self.global_step)
                self.model.train()

            # Save checkpoint
            if self.global_step % self.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

        # Average epoch loss
        if len(epoch_losses) > 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
        else:
            logger.warning("No batches were processed in this epoch!")
            avg_loss = 0.0
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Run validation"""
        self.model.eval()
        val_losses = []

        pbar = tqdm(self.val_dataloader, desc="Validation")

        for batch in pbar:
            v_in = batch['input'].to(self.device)
            v_gt = batch['target'].to(self.device)

            # Forward pass
            if self.use_amp:
                with autocast():
                    loss, metrics = self.model(v_in, v_gt)
            else:
                loss, metrics = self.model(v_in, v_gt)

            val_losses.append(loss.item())

            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation loss: {avg_val_loss:.4f}")

        return avg_val_loss

    def train(self):
        """Main training loop with two-phase training support"""
        print(f"\nStarting training for {self.num_epochs} epochs...")
        try:
            print(f"Total steps: {len(self.train_dataloader) * self.num_epochs}")
        except TypeError:
            print(f"Total steps: Unknown (streaming dataset)")

        # Freeze VAE at start if configured
        if self.use_two_phase or self.vae_freeze_epochs > 0:
            self.freeze_vae()
            if self.use_two_phase:
                print(f"\n{'='*60}")
                print(f"PHASE 1: Training U-Net only (VAE frozen)")
                print(f"Duration: {self.phase1_epochs} epoch(s)")
                print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Check for phase transition (two-phase training)
            if self.use_two_phase and self.current_phase == 1 and epoch >= self.phase1_epochs:
                # Transition to Phase 2
                self.current_phase = 2
                self.unfreeze_vae()
                print(f"\n{'='*60}")
                print(f"PHASE 2: Fine-tuning entire model (VAE unfrozen)")
                print(f"Duration: {self.num_epochs - self.phase1_epochs} epoch(s)")
                print(f"{'='*60}\n")

            # Check for simple freeze mode (not two-phase)
            elif not self.use_two_phase and self.vae_freeze_epochs > 0 and epoch == self.vae_freeze_epochs:
                self.unfreeze_vae()
                print(f"\n{'='*60}")
                print(f"VAE unfrozen after {self.vae_freeze_epochs} epoch(s)")
                print(f"{'='*60}\n")

            # Train epoch
            avg_loss = self.train_epoch()

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log epoch metrics
            self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)

            print(f"Epoch {epoch} - Average loss: {avg_loss:.4f}")

            # Save checkpoint at end of epoch
            self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {elapsed_time / 3600:.2f} hours")

        # Save final checkpoint
        self.save_checkpoint('checkpoint_final.pt')

        # Close writer
        self.writer.close()

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = self.checkpoint_dir / filename

        self.model.save_checkpoint(
            path,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            global_step=self.global_step
        )

    def load_checkpoint(self, path):
        """Load model checkpoint and resume training"""
        print(f"Loading checkpoint from {path}...")

        _, checkpoint = self.model.load_checkpoint(path, self.device)

        # Restore optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore training state
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch'] + 1
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']

        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")


if __name__ == "__main__":
    # Test trainer setup
    from models import VideoToVideoDiffusion
    from data import get_dataloader

    print("Testing trainer...")

    # Create dummy config
    model_config = {
        'in_channels': 3,
        'latent_dim': 4,
        'vae_base_channels': 32,  # Smaller for testing
        'unet_model_channels': 64,
        'unet_num_res_blocks': 1,
        'unet_attention_levels': [1],
        'unet_channel_mult': [1, 2],
        'unet_num_heads': 2,
        'noise_schedule': 'cosine',
        'diffusion_timesteps': 100
    }

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoToVideoDiffusion(model_config)

    # Create dummy dataset
    import numpy as np
    video_pairs = [
        {'input': np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8),
         'target': np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8)}
        for _ in range(8)
    ]

    data_config = {
        'data_source': video_pairs,
        'source_type': 'list',
        'batch_size': 2,
        'num_workers': 0,
        'num_frames': 16,
        'resolution': [256, 256]
    }

    train_dataloader = get_dataloader(data_config, split='train')

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = None

    # Training config
    train_config = {
        'num_epochs': 2,
        'gradient_accumulation_steps': 1,
        'use_amp': False,
        'log_interval': 1,
        'checkpoint_interval': 10
    }

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        config=train_config,
        device=device
    )

    print("\nTrainer setup successful!")
    print("Run trainer.train() to start training")
