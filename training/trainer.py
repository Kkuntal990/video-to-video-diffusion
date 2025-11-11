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
import gc
from utils.metrics import calculate_video_metrics


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
        self.model_suffix = self.config.get('model_suffix', '')  # Optional suffix for multi-model training

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
        self.best_loss = float('inf')  # Track best validation/training loss
        self.best_checkpoint_path = None  # Track path to best checkpoint

        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Num epochs: {self.num_epochs}")
        print(f"  Model suffix: {self.model_suffix if self.model_suffix else 'None'}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Log dir: {self.log_dir}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")

    def _get_checkpoint_filename(self, base_name):
        """
        Generate checkpoint filename with model suffix

        Args:
            base_name: Base filename (e.g., 'checkpoint_best_epoch_10.pt')

        Returns:
            Filename with suffix (e.g., 'checkpoint_best_epoch_10_pretrained.pt')
        """
        if not self.model_suffix:
            return base_name

        # Insert suffix before .pt extension
        if base_name.endswith('.pt'):
            return base_name[:-3] + f'_{self.model_suffix}.pt'
        else:
            return f'{base_name}_{self.model_suffix}'

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

        # Clear any leftover accumulated gradients from previous epoch
        # This is critical when gradient_accumulation_steps doesn't divide evenly into num_batches
        # Example: 243 batches % 8 steps = 3 leftover batches that never called optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        epoch_losses = []  # Store loss tensors to avoid repeated CPU-GPU sync
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            v_in = batch['input'].to(self.device)
            v_gt = batch['target'].to(self.device)

            # Extract padding mask for target (thin slices) - CRITICAL for variable depths!
            # Prevents model from learning on zero-padded regions
            mask = batch.get('thin_mask', None)
            if mask is not None:
                mask = mask.to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    loss, metrics = self.model(v_in, v_gt, mask=mask)
                    loss = loss / self.gradient_accumulation_steps
            else:
                loss, metrics = self.model(v_in, v_gt, mask=mask)
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

                self.optimizer.zero_grad(set_to_none=True)  # Faster memory release

            # Update global step
            self.global_step += 1

            # Track loss for epoch average (minimize CPU-GPU sync)
            with torch.no_grad():
                epoch_loss_sum += (loss * self.gradient_accumulation_steps).detach()
                epoch_loss_count += 1

            # Log metrics only at intervals
            if self.global_step % self.log_interval == 0:
                loss_value = loss.item() * self.gradient_accumulation_steps

                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/loss', loss_value, self.global_step)
                self.writer.add_scalar('train/lr', lr, self.global_step)

                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'lr': f'{lr:.2e}',
                    'step': self.global_step
                })

                # Periodic garbage collection to prevent RAM buildup
                gc.collect()

            # Validation
            if self.val_dataloader and self.global_step % self.val_interval == 0:
                val_loss = self.validate()
                self.writer.add_scalar('val/loss', val_loss, self.global_step)
                self.model.train()

            # Save checkpoint
            if self.global_step % self.checkpoint_interval == 0:
                step_checkpoint_filename = self._get_checkpoint_filename(f'checkpoint_step_{self.global_step}.pt')
                self.save_checkpoint(step_checkpoint_filename)

        # Handle any remaining accumulated gradients at end of epoch
        # If num_batches % gradient_accumulation_steps != 0, there will be leftover gradients
        # Example: 243 batches % 8 = 3 leftover batches that accumulated but never stepped
        num_batches = len(self.train_dataloader)
        if num_batches % self.gradient_accumulation_steps != 0:
            # Step the optimizer with remaining accumulated gradients
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        # Average epoch loss (convert to scalar only once at end of epoch)
        if epoch_loss_count > 0:
            avg_loss = (epoch_loss_sum / epoch_loss_count).item()
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("No batches were processed in this epoch!")
            avg_loss = 0.0

        # Force garbage collection after epoch completes to free RAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Run validation with SSIM and PSNR metrics"""
        self.model.eval()

        # Use streaming average instead of list accumulation to save RAM
        val_loss_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        val_count = 0

        # Limit validation samples for speed (config parameter)
        max_samples = self.config.get('num_validation_samples', 2)
        pbar = tqdm(self.val_dataloader, desc="Validation", total=min(max_samples, len(self.val_dataloader)))

        for batch in pbar:
            # Stop after max_samples to speed up validation
            if val_count >= max_samples:
                break
            v_in = batch['input'].to(self.device)
            v_gt = batch['target'].to(self.device)

            # Forward pass for loss
            if self.use_amp:
                with autocast():
                    loss, metrics = self.model(v_in, v_gt)
            else:
                loss, metrics = self.model(v_in, v_gt)

            # Generate prediction for SSIM/PSNR calculation
            # Sample from the diffusion model using generate() method
            try:
                # For slice interpolation, pass target depth from ground truth
                target_depth = v_gt.shape[2]  # Get depth from ground truth (300 thin slices)

                if self.use_amp:
                    with autocast():
                        v_pred = self.model.generate(v_in, sampler='ddim', num_inference_steps=20, target_depth=target_depth)
                else:
                    v_pred = self.model.generate(v_in, sampler='ddim', num_inference_steps=20, target_depth=target_depth)

                # Calculate SSIM and PSNR metrics
                # Clamp values to [-1, 1] range (matching dataset normalization)
                v_pred_clamped = torch.clamp(v_pred, -1, 1)
                v_gt_clamped = torch.clamp(v_gt, -1, 1)

                # PSNR max_val should be range size: 2.0 for [-1, 1]
                video_metrics = calculate_video_metrics(v_pred_clamped, v_gt_clamped, max_val=2.0)
                psnr_sum += video_metrics['psnr']
                ssim_sum += video_metrics['ssim']
            except Exception as e:
                # If sampling fails, skip metrics
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not calculate SSIM/PSNR: {e}")
                psnr_sum += 0.0
                ssim_sum += 0.0

            # Accumulate loss sum
            val_loss_sum += loss.item()
            val_count += 1

            pbar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'psnr': f'{psnr_sum/val_count:.2f}' if val_count > 0 else 'N/A',
                'ssim': f'{ssim_sum/val_count:.4f}' if val_count > 0 else 'N/A'
            })

        # Calculate averages
        avg_val_loss = val_loss_sum / val_count if val_count > 0 else 0.0
        avg_psnr = psnr_sum / val_count if val_count > 0 else 0.0
        avg_ssim = ssim_sum / val_count if val_count > 0 else 0.0

        print(f"Validation - Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        # Log metrics to TensorBoard
        self.writer.add_scalar('val/psnr', avg_psnr, self.global_step)
        self.writer.add_scalar('val/ssim', avg_ssim, self.global_step)

        # Clean up after validation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_val_loss

    def train(self):
        """Main training loop with two-phase training support"""
        print(f"\nStarting training for {self.num_epochs} epochs...")
        try:
            print(f"Total steps: {len(self.train_dataloader) * self.num_epochs}")
        except TypeError:
            print(f"Total steps: Unknown (streaming dataset)")

        # Freeze VAE at start if configured (only if starting fresh, not resuming)
        # Note: When resuming, VAE freeze state is restored in load_checkpoint()
        if (self.use_two_phase or self.vae_freeze_epochs > 0) and self.current_epoch == 0:
            self.freeze_vae()
            if self.use_two_phase:
                print(f"\n{'='*60}")
                print(f"PHASE 1: Training U-Net only (VAE frozen)")
                print(f"Duration: {self.phase1_epochs} epoch(s)")
                print(f"{'='*60}\n")

        start_time = time.time()

        # Start from current_epoch (handles resumption correctly)
        for epoch in range(self.current_epoch, self.num_epochs):
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

            # Save best checkpoint if this epoch has lowest loss
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss

                # Delete previous best checkpoint to save space
                if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
                    os.remove(self.best_checkpoint_path)
                    print(f"  Deleted previous best checkpoint")

                # Save new best checkpoint
                best_checkpoint_filename = self._get_checkpoint_filename(f'checkpoint_best_epoch_{epoch}.pt')
                self.best_checkpoint_path = self.checkpoint_dir / best_checkpoint_filename
                self.save_checkpoint(best_checkpoint_filename)
                print(f"  ✅ New best checkpoint saved: {best_checkpoint_filename} (Loss: {avg_loss:.4f})")

            # Also save latest checkpoint (for resuming if training crashes)
            latest_checkpoint_filename = self._get_checkpoint_filename('checkpoint_latest.pt')
            latest_checkpoint_path = self.checkpoint_dir / latest_checkpoint_filename
            if os.path.exists(latest_checkpoint_path):
                os.remove(latest_checkpoint_path)
            self.save_checkpoint(latest_checkpoint_filename)

        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {elapsed_time / 3600:.2f} hours")

        # Run final validation to get end-of-training metrics
        if self.val_dataloader:
            print("\nRunning final validation...")
            final_val_loss = self.validate()
            self.writer.add_scalar('val/final_loss', final_val_loss, self.global_step)
            print(f"Final validation loss: {final_val_loss:.4f}")

        # Save final checkpoint
        final_checkpoint_filename = self._get_checkpoint_filename('checkpoint_final.pt')
        self.save_checkpoint(final_checkpoint_filename)
        print(f"Final checkpoint saved: {final_checkpoint_filename}")

        # Close writer
        self.writer.close()

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = self.checkpoint_dir / filename

        self.model.save_checkpoint(
            path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            global_step=self.global_step,
            current_phase=self.current_phase,
            best_loss=self.best_loss
        )

        # Force garbage collection after checkpoint saving to free RAM
        gc.collect()

    def load_checkpoint(self, path):
        """Load model checkpoint and resume training"""
        print(f"Loading checkpoint from {path}...")

        # Load checkpoint and get the loaded model
        loaded_model, checkpoint = self.model.load_checkpoint(path, self.device)

        # Replace the trainer's model with the loaded model
        self.model = loaded_model
        print(f"✓ Model weights loaded successfully")

        # Restore optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✓ Optimizer state restored")

        # Restore scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"✓ Scheduler state restored")

        # Restore training state
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch'] + 1
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
            print(f"Restored best loss: {self.best_loss:.4f}")

        # Restore phase state for two-phase training
        if 'current_phase' in checkpoint:
            self.current_phase = checkpoint['current_phase']
            print(f"Restored to phase {self.current_phase}")

            # Set VAE freeze/unfreeze based on restored phase
            if self.use_two_phase:
                if self.current_phase == 1:
                    self.freeze_vae()
                    print("🔒 VAE frozen (Phase 1)")
                elif self.current_phase == 2:
                    self.unfreeze_vae()
                    print("🔓 VAE unfrozen (Phase 2)")
        else:
            # Infer phase from epoch if not saved in checkpoint (backwards compatibility)
            if self.use_two_phase and self.current_epoch >= self.phase1_epochs:
                self.current_phase = 2
                self.unfreeze_vae()
                print(f"⚠️ Phase not in checkpoint, inferred Phase 2 from epoch {self.current_epoch}")
                print("🔓 VAE unfrozen")

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
