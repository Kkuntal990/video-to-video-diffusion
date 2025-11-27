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
        full_val_dataloader=None,
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
            val_dataloader: patch-based validation data loader (for Tier 1 & 2)
            full_val_dataloader: full-volume validation data loader (for Tier 3) (optional)
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
        self.full_val_dataloader = full_val_dataloader
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

        # Configure precision dtype for AMP
        precision_str = self.config.get('precision', 'fp16')
        if precision_str == 'bf16':
            self.amp_dtype = torch.bfloat16
        elif precision_str == 'fp16':
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float16  # Default to FP16

        self.log_interval = self.config.get('log_interval', 10)
        self.val_interval = self.config.get('val_interval', 1000)

        # Multi-tier validation settings
        self.patch_val_interval = self.config.get('patch_val_interval', 5)  # Every N epochs
        self.patch_val_samples = self.config.get('patch_val_samples', 10)  # Number of patches
        self.patch_val_generate = self.config.get('patch_val_generate', True)  # Generate predictions
        self.full_val_interval = self.config.get('full_val_interval', 15)  # Every N epochs
        self.full_val_samples = self.config.get('full_val_samples', 2)  # Number of full volumes

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
        self.just_resumed_checkpoint = False  # Flag to clear CUDA cache after checkpoint resume

        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Num epochs: {self.num_epochs}")
        print(f"  Model suffix: {self.model_suffix if self.model_suffix else 'None'}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Mixed precision: {self.use_amp}")
        if self.use_amp:
            print(f"  AMP dtype: {self.amp_dtype}")
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

        # Clear CUDA cache if this is first epoch after checkpoint resume
        # This ensures checkpoint loading artifacts don't consume memory
        if self.just_resumed_checkpoint:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"✓ CUDA cache cleared after checkpoint resume")
            self.just_resumed_checkpoint = False  # Reset flag

        # Clear any leftover accumulated gradients from previous epoch
        # This is critical when gradient_accumulation_steps doesn't divide evenly into num_batches
        # Example: 243 batches % 8 steps = 3 leftover batches that never called optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        # Timer instrumentation to measure data loading vs compute time
        data_time = 0.0
        step_time = 0.0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            t0 = time.time()

            # Move data to device (non-blocking for async transfer)
            v_in = batch['input'].to(self.device, non_blocking=True)
            v_gt = batch['target'].to(self.device, non_blocking=True)

            # Extract padding mask (if available)
            # Full-volume mode: mask prevents learning on padded regions (variable depths)
            # Patch-based mode: mask=None (all patches have fixed size, no padding)
            mask = batch.get('thin_mask', None)
            if mask is not None:
                mask = mask.to(self.device, non_blocking=True)

            t1 = time.time()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
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

                # Update global step (only after optimizer step, not per micro-batch)
                self.global_step += 1

            # Track loss for epoch average (minimize CPU-GPU sync)
            with torch.no_grad():
                epoch_loss_sum += (loss * self.gradient_accumulation_steps).detach()
                epoch_loss_count += 1

            t2 = time.time()
            data_time += (t1 - t0)
            step_time += (t2 - t1)

            # Log metrics only at intervals
            if self.global_step % self.log_interval == 0:
                loss_value = loss.item() * self.gradient_accumulation_steps

                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/loss', loss_value, self.global_step)
                self.writer.add_scalar('train/lr', lr, self.global_step)

                # Update progress bar with timing info every 50 batches
                if batch_idx > 0 and batch_idx % 50 == 0:
                    avg_data = data_time / batch_idx
                    avg_step = step_time / batch_idx
                    pbar.set_postfix({
                        'loss': f'{loss_value:.4f}',
                        'lr': f'{lr:.2e}',
                        'data_t': f'{avg_data:.3f}s',
                        'step_t': f'{avg_step:.3f}s',
                    })
                else:
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

        # Handle any remaining accumulated gradients at end of epoch
        # If num_batches % gradient_accumulation_steps != 0, there will be leftover gradients
        # Example: 243 batches % 8 = 3 leftover batches that accumulated but never stepped
        try:
            num_batches = len(self.train_dataloader)
        except TypeError:
            # Streaming dataset without __len__(), no leftover gradients possible
            num_batches = None

        if num_batches is not None and num_batches % self.gradient_accumulation_steps != 0:
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

        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Run validation with SSIM and PSNR metrics"""
        self.model.eval()

        # Accumulate losses as tensors on GPU (avoid GPU-CPU sync)
        val_losses = []
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
            v_in = batch['input'].to(self.device, non_blocking=True)
            v_gt = batch['target'].to(self.device, non_blocking=True)

            # Forward pass for loss
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    loss, metrics = self.model(v_in, v_gt)
            else:
                loss, metrics = self.model(v_in, v_gt)

            # Generate prediction for SSIM/PSNR calculation
            # Note: For patch-based training, we skip generation metrics during validation
            # (patches are too small for meaningful PSNR/SSIM, and we'd need stitching for full volumes)
            try:
                # Detect if we're in patch mode (small spatial dimensions = patches)
                _, _, _, H, W = v_gt.shape
                is_patch_mode = (H < 256) or (W < 256)  # Patches are typically 192×192 or smaller

                if not is_patch_mode:
                    # Full-volume mode: generate prediction and compute metrics
                    target_depth = v_gt.shape[2]  # Get depth from ground truth

                    # CRITICAL: Always generate in FP32 for numerical stability
                    with torch.cuda.amp.autocast(enabled=False):
                        v_pred = self.model.generate(v_in, sampler='ddim', num_inference_steps=20, target_depth=target_depth)

                    # Calculate SSIM and PSNR metrics
                    # Normalize to [0, 1] range to match VAE training metrics
                    v_pred_clamped = torch.clamp(v_pred, -1, 1)
                    v_gt_clamped = torch.clamp(v_gt, -1, 1)
                    v_pred_norm = (v_pred_clamped + 1.0) / 2.0  # [-1, 1] → [0, 1]
                    v_gt_norm = (v_gt_clamped + 1.0) / 2.0      # [-1, 1] → [0, 1]

                    # Use max_val=1.0 for [0, 1] range (consistent with VAE training)
                    video_metrics = calculate_video_metrics(v_pred_norm, v_gt_norm, max_val=1.0)
                    psnr_sum += video_metrics['psnr']
                    ssim_sum += video_metrics['ssim']
                else:
                    # Patch-based mode: skip generation metrics (use loss only)
                    # For full-volume metrics, use separate inference script with stitching
                    psnr_sum += 0.0
                    ssim_sum += 0.0
            except Exception as e:
                # If sampling fails, skip metrics
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not calculate SSIM/PSNR: {e}")
                psnr_sum += 0.0
                ssim_sum += 0.0

            # Accumulate loss tensor (avoid GPU-CPU sync)
            val_losses.append(loss.detach())
            val_count += 1

            pbar.set_postfix({
                'psnr': f'{psnr_sum/val_count:.2f}' if val_count > 0 else 'N/A',
                'ssim': f'{ssim_sum/val_count:.4f}' if val_count > 0 else 'N/A'
            })

        # Calculate averages (single GPU-CPU sync at end)
        avg_val_loss = torch.stack(val_losses).mean().item() if len(val_losses) > 0 else 0.0
        avg_psnr = psnr_sum / val_count if val_count > 0 else 0.0
        avg_ssim = ssim_sum / val_count if val_count > 0 else 0.0

        print(f"Validation - Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        # Log metrics to TensorBoard
        self.writer.add_scalar('val/psnr', avg_psnr, self.global_step)
        self.writer.add_scalar('val/ssim', avg_ssim, self.global_step)

        # Clean up after validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_val_loss

    @torch.no_grad()
    def validate_loss_only(self):
        """
        Tier 1: Quick loss-only validation (no generation, no metrics)
        Runs every epoch for fast monitoring
        """
        self.model.eval()

        val_losses = []
        max_samples = 5  # Only check a few batches for speed

        for batch in self.val_dataloader:
            if len(val_losses) >= max_samples:
                break

            v_in = batch['input'].to(self.device, non_blocking=True)
            v_gt = batch['target'].to(self.device, non_blocking=True)

            # Compute loss only (no generation)
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    loss, _ = self.model(v_in, v_gt)
            else:
                loss, _ = self.model(v_in, v_gt)

            val_losses.append(loss.detach())

        avg_loss = torch.stack(val_losses).mean().item() if len(val_losses) > 0 else 0.0
        print(f"  [Tier 1] Val Loss (quick): {avg_loss:.4f}")
        self.writer.add_scalar('val/loss_only', avg_loss, self.current_epoch)

        self.model.train()
        return avg_loss

    @torch.no_grad()
    def validate_patches(self, max_samples=10, generate=True):
        """
        Tier 2: Patch-based validation with generation
        Computes PSNR/SSIM on patches, used for checkpoint selection
        """
        self.model.eval()

        val_losses = []
        psnr_sum = 0.0
        ssim_sum = 0.0
        val_count = 0

        pbar = tqdm(self.val_dataloader, desc="[Tier 2] Patch Validation", total=max_samples)

        for batch in pbar:
            if val_count >= max_samples:
                break

            v_in = batch['input'].to(self.device, non_blocking=True)
            v_gt = batch['target'].to(self.device, non_blocking=True)

            # Compute loss
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    loss, _ = self.model(v_in, v_gt)
            else:
                loss, _ = self.model(v_in, v_gt)

            val_losses.append(loss.detach())

            # Generate predictions and compute metrics (if enabled)
            if generate:
                try:
                    target_depth = v_gt.shape[2]

                    # CRITICAL: Always generate in FP32 for numerical stability
                    # model.generate() internally disables autocast, but we disable it here too
                    # to ensure no mixed precision artifacts from outer context
                    with torch.cuda.amp.autocast(enabled=False):
                        v_pred = self.model.generate(v_in, sampler='ddim', num_inference_steps=20, target_depth=target_depth)

                    # Compute metrics on patches
                    # Normalize to [0, 1] range to match VAE training metrics
                    v_pred_clamped = torch.clamp(v_pred, -1, 1)
                    v_gt_clamped = torch.clamp(v_gt, -1, 1)
                    v_pred_norm = (v_pred_clamped + 1.0) / 2.0  # [-1, 1] → [0, 1]
                    v_gt_norm = (v_gt_clamped + 1.0) / 2.0      # [-1, 1] → [0, 1]
                    video_metrics = calculate_video_metrics(v_pred_norm, v_gt_norm, max_val=1.0)

                    psnr_sum += video_metrics['psnr']
                    ssim_sum += video_metrics['ssim']
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Patch generation failed: {e}")
                    psnr_sum += 0.0
                    ssim_sum += 0.0

            val_count += 1
            pbar.set_postfix({
                'psnr': f'{psnr_sum/val_count:.2f}' if generate and val_count > 0 else 'N/A',
                'ssim': f'{ssim_sum/val_count:.4f}' if generate and val_count > 0 else 'N/A'
            })

        # Calculate averages (single GPU-CPU sync at end)
        avg_loss = torch.stack(val_losses).mean().item() if len(val_losses) > 0 else 0.0
        avg_psnr = psnr_sum / val_count if val_count > 0 else 0.0
        avg_ssim = ssim_sum / val_count if val_count > 0 else 0.0

        print(f"  [Tier 2] Patch Val - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        # Log to TensorBoard
        self.writer.add_scalar('val/patch_loss', avg_loss, self.current_epoch)
        if generate:
            self.writer.add_scalar('val/patch_psnr', avg_psnr, self.current_epoch)
            self.writer.add_scalar('val/patch_ssim', avg_ssim, self.current_epoch)

        self.model.train()
        return {'loss': avg_loss, 'psnr': avg_psnr, 'ssim': avg_ssim}

    @torch.no_grad()
    def validate_full_volumes(self, max_samples=2):
        """
        Tier 3: Full-volume validation with generation
        Expensive, runs rarely during training
        """
        self.model.eval()

        val_losses = []
        psnr_sum = 0.0
        ssim_sum = 0.0
        val_count = 0

        print(f"\n  [Tier 3] Full-Volume Validation (n={max_samples})...")

        for batch in tqdm(self.full_val_dataloader, desc="Full-Volume Val", total=max_samples):
            if val_count >= max_samples:
                break

            v_in = batch['input'].to(self.device, non_blocking=True)  # (B, 1, 50, 512, 512)
            v_gt = batch['target'].to(self.device, non_blocking=True)  # (B, 1, 300, 512, 512)

            # Compute loss
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    loss, _ = self.model(v_in, v_gt)
            else:
                loss, _ = self.model(v_in, v_gt)

            val_losses.append(loss.detach())

            # Generate full volumes with DDIM
            try:
                target_depth = v_gt.shape[2]

                # CRITICAL: Always generate in FP32 for numerical stability
                with torch.cuda.amp.autocast(enabled=False):
                    v_pred = self.model.generate(v_in, sampler='ddim', num_inference_steps=20, target_depth=target_depth)

                # Compute metrics on full volumes
                # Normalize to [0, 1] range to match VAE training metrics
                v_pred_clamped = torch.clamp(v_pred, -1, 1)
                v_gt_clamped = torch.clamp(v_gt, -1, 1)
                v_pred_norm = (v_pred_clamped + 1.0) / 2.0  # [-1, 1] → [0, 1]
                v_gt_norm = (v_gt_clamped + 1.0) / 2.0      # [-1, 1] → [0, 1]
                video_metrics = calculate_video_metrics(v_pred_norm, v_gt_norm, max_val=1.0)

                psnr_sum += video_metrics['psnr']
                ssim_sum += video_metrics['ssim']
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Full-volume generation failed: {e}")
                psnr_sum += 0.0
                ssim_sum += 0.0

            val_count += 1

        # Calculate averages (single GPU-CPU sync at end)
        avg_loss = torch.stack(val_losses).mean().item() if len(val_losses) > 0 else 0.0
        avg_psnr = psnr_sum / val_count if val_count > 0 else 0.0
        avg_ssim = ssim_sum / val_count if val_count > 0 else 0.0

        print(f"  [Tier 3] Full-Volume Val - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}\n")

        # Log to TensorBoard
        self.writer.add_scalar('val/full_loss', avg_loss, self.current_epoch)
        self.writer.add_scalar('val/full_psnr', avg_psnr, self.current_epoch)
        self.writer.add_scalar('val/full_ssim', avg_ssim, self.current_epoch)

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.train()
        return {'loss': avg_loss, 'psnr': avg_psnr, 'ssim': avg_ssim}

    @torch.no_grad()
    def final_validate(self, dataloader, use_full_volumes=True, max_samples=None):
        """
        Final comprehensive validation at end of training
        No sample limit - validates on ALL validation data
        """
        self.model.eval()

        val_losses = []
        psnr_sum = 0.0
        ssim_sum = 0.0
        val_count = 0

        mode_str = "Full-Volume" if use_full_volumes else "Patch"
        print(f"\n[FINAL] Validation Mode: {mode_str}")
        print(f"[FINAL] Processing ALL validation samples (no limit)...\n")

        pbar = tqdm(dataloader, desc=f"[FINAL] {mode_str} Validation")

        for batch in pbar:
            # No max_samples limit for final validation!
            v_in = batch['input'].to(self.device, non_blocking=True)
            v_gt = batch['target'].to(self.device, non_blocking=True)

            # Compute loss
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    loss, _ = self.model(v_in, v_gt)
            else:
                loss, _ = self.model(v_in, v_gt)

            val_losses.append(loss.detach())

            # Generate predictions and compute metrics
            try:
                target_depth = v_gt.shape[2]

                # CRITICAL: Always generate in FP32 for numerical stability
                with torch.cuda.amp.autocast(enabled=False):
                    v_pred = self.model.generate(v_in, sampler='ddim', num_inference_steps=20, target_depth=target_depth)

                # Compute metrics
                # Normalize to [0, 1] range to match VAE training metrics
                v_pred_clamped = torch.clamp(v_pred, -1, 1)
                v_gt_clamped = torch.clamp(v_gt, -1, 1)
                v_pred_norm = (v_pred_clamped + 1.0) / 2.0  # [-1, 1] → [0, 1]
                v_gt_norm = (v_gt_clamped + 1.0) / 2.0      # [-1, 1] → [0, 1]
                video_metrics = calculate_video_metrics(v_pred_norm, v_gt_norm, max_val=1.0)

                psnr_sum += video_metrics['psnr']
                ssim_sum += video_metrics['ssim']
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Final validation generation failed: {e}")
                psnr_sum += 0.0
                ssim_sum += 0.0

            val_count += 1
            pbar.set_postfix({
                'psnr': f'{psnr_sum/val_count:.2f}',
                'ssim': f'{ssim_sum/val_count:.4f}'
            })

        # Calculate averages (single GPU-CPU sync at end)
        avg_loss = torch.stack(val_losses).mean().item() if len(val_losses) > 0 else 0.0
        avg_psnr = psnr_sum / val_count if val_count > 0 else 0.0
        avg_ssim = ssim_sum / val_count if val_count > 0 else 0.0

        print(f"\n{'='*80}")
        print(f"[FINAL] Validation Results ({val_count} samples):")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  PSNR: {avg_psnr:.2f} dB")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"{'='*80}\n")

        # Log to TensorBoard
        self.writer.add_scalar('val/final_loss', avg_loss, self.current_epoch)
        self.writer.add_scalar('val/final_psnr', avg_psnr, self.current_epoch)
        self.writer.add_scalar('val/final_ssim', avg_ssim, self.current_epoch)

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.train()
        return {'loss': avg_loss, 'psnr': avg_psnr, 'ssim': avg_ssim}

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

            # Multi-tier validation strategy
            # Tier 1: Loss-only validation (every epoch, fast)
            if self.val_dataloader:
                self.validate_loss_only()

            # Tier 2: Patch-based validation with generation (periodic, checkpoints based on this)
            if self.val_dataloader and (epoch + 1) % self.patch_val_interval == 0:
                patch_val_metrics = self.validate_patches(
                    max_samples=self.patch_val_samples,
                    generate=self.patch_val_generate
                )
                # Update best checkpoint based on patch validation
                if patch_val_metrics and 'loss' in patch_val_metrics:
                    patch_val_loss = patch_val_metrics['loss']
                    if patch_val_loss < self.best_loss:
                        self.best_loss = patch_val_loss
                        # Delete previous best checkpoint
                        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
                            os.remove(self.best_checkpoint_path)
                            print(f"  Deleted previous best checkpoint")

                        # Save new best checkpoint
                        best_checkpoint_filename = self._get_checkpoint_filename(f'checkpoint_best_epoch_{epoch}.pt')
                        self.best_checkpoint_path = self.checkpoint_dir / best_checkpoint_filename
                        self.save_checkpoint(best_checkpoint_filename)
                        print(f"  ✅ New best checkpoint saved: {best_checkpoint_filename} (Patch Val Loss: {patch_val_loss:.4f})")

            # Tier 3: Full-volume validation (rare, expensive)
            if self.full_val_dataloader and (epoch + 1) % self.full_val_interval == 0:
                self.validate_full_volumes(max_samples=self.full_val_samples)

        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {elapsed_time / 3600:.2f} hours")

        # Note: Final validation is now handled in train.py after this method returns

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
            scaler=self.scaler,  # CRITICAL: Save GradScaler state for stable AMP resume
            epoch=self.current_epoch,
            global_step=self.global_step,
            current_phase=self.current_phase,
            best_loss=self.best_loss
        )

        # Force garbage collection after checkpoint saving to free RAM
        gc.collect()

    def load_checkpoint(self, path):
        """
        Load model checkpoint and resume training.

        Follows PyTorch best practices:
        1. Load checkpoint dict
        2. Load state dict into EXISTING model (no duplication)
        3. Load optimizer/scheduler/scaler states
        4. Restore training state

        This avoids creating a new model instance (which caused 2× memory usage).
        """
        print(f"Loading checkpoint from {path}...")

        # Load checkpoint dict
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load model state dict into EXISTING model (standard PyTorch pattern)
        # This avoids creating a duplicate model in memory
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"✓ Model weights loaded successfully")
        except RuntimeError as e:
            print(f"⚠ WARNING: Model state dict mismatch: {e}")
            print(f"  Attempting non-strict loading...")
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            if missing_keys:
                print(f"  Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys[:5]}...")
            print(f"✓ Model weights loaded (non-strict)")

        # Restore optimizer state
        # The optimizer already has correct param_groups from train.py initialization
        # We just need to restore the state (momentum buffers, etc.)
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✓ Optimizer state restored")
            except (ValueError, KeyError) as e:
                print(f"⚠ WARNING: Optimizer state mismatch: {e}")
                print(f"  This can happen if model architecture changed or VAE was frozen/unfrozen")
                print(f"  Skipping optimizer state - will use fresh optimizer with current LRs")
                print(f"  Training will continue normally!")

        # Restore scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✓ Scheduler state restored")
            except (ValueError, KeyError) as e:
                print(f"⚠ WARNING: Scheduler state mismatch: {e}")
                print(f"  Skipping scheduler state - will use fresh scheduler")

        # Restore GradScaler state (CRITICAL for mixed precision training!)
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print(f"✓ GradScaler state restored (scale={self.scaler.get_scale():.0f})")
            except (ValueError, RuntimeError) as e:
                print(f"⚠ WARNING: GradScaler state mismatch: {e}")
                print(f"  Skipping GradScaler state - will use fresh scaler")

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

        # Clear any stale gradients
        self.optimizer.zero_grad(set_to_none=True)
        print(f"✓ Gradients cleared after checkpoint load")

        # Set flag to clear CUDA cache at start of next epoch
        # This prevents checkpoint loading artifacts from consuming memory
        self.just_resumed_checkpoint = True


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
