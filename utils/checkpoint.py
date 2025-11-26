"""
Checkpoint Loading Utilities

Handles loading checkpoints with model_suffix support for CT slice interpolation.
"""

import os
import glob
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


def find_best_checkpoint(checkpoint_dir: str, model_suffix: Optional[str] = None) -> Optional[str]:
    """
    Find the best checkpoint file in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model_suffix: Optional model suffix (e.g., 'slice_interp_full3')

    Returns:
        Path to best checkpoint, or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None

    # Build pattern based on suffix
    if model_suffix:
        pattern = f"checkpoint_best_epoch_*_{model_suffix}.pt"
    else:
        pattern = "checkpoint_best_epoch_*.pt"

    # Find matching checkpoints
    checkpoints = sorted(checkpoint_dir.glob(pattern))

    if not checkpoints:
        logger.warning(f"No checkpoints found matching pattern: {pattern}")
        return None

    # Return the latest one (highest epoch number)
    best_checkpoint = checkpoints[-1]
    logger.info(f"Found best checkpoint: {best_checkpoint.name}")

    return str(best_checkpoint)


def find_latest_checkpoint(checkpoint_dir: str, model_suffix: Optional[str] = None) -> Optional[str]:
    """
    Find the latest checkpoint file (final or best).

    Args:
        checkpoint_dir: Directory containing checkpoints
        model_suffix: Optional model suffix

    Returns:
        Path to latest checkpoint, or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None

    # Try final checkpoint first
    if model_suffix:
        final_pattern = f"checkpoint_final_epoch_*_{model_suffix}.pt"
    else:
        final_pattern = "checkpoint_final_epoch_*.pt"

    final_checkpoints = sorted(checkpoint_dir.glob(final_pattern))
    if final_checkpoints:
        latest = final_checkpoints[-1]
        logger.info(f"Found final checkpoint: {latest.name}")
        return str(latest)

    # Fall back to best checkpoint
    return find_best_checkpoint(checkpoint_dir, model_suffix)


def load_checkpoint(
    checkpoint_path: str,
    device: str = 'cpu',
    weights_only: bool = False
) -> Dict:
    """
    Load a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
        weights_only: If True, only load weights (safer, recommended for inference)

    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)

        # Log checkpoint info
        if isinstance(checkpoint, dict):
            epoch = checkpoint.get('epoch', 'unknown')
            best_loss = checkpoint.get('best_loss', 'unknown')
            logger.info(f"Checkpoint info: Epoch {epoch}, Best loss: {best_loss}")

        return checkpoint

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


def load_model_from_checkpoint(
    model,
    checkpoint_path: str,
    device: str = 'cuda',
    strict: bool = True
) -> Tuple[object, Dict]:
    """
    Load model weights from checkpoint.

    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Tuple of (loaded_model, checkpoint_metadata)
    """
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device='cpu')

    # Extract model state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'best_loss': checkpoint.get('best_loss', float('inf')),
            'config': checkpoint.get('config', {})
        }
    else:
        # Old format: checkpoint is the state dict directly
        state_dict = checkpoint
        metadata = {}

    # Load state dict into model
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        logger.info("Model weights loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        raise

    # Move model to device
    model = model.to(device)
    model.eval()  # Set to eval mode for inference

    return model, metadata


def extract_model_suffix_from_path(checkpoint_path: str) -> Optional[str]:
    """
    Extract model suffix from checkpoint filename.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Model suffix if found, else None

    Example:
        'checkpoint_best_epoch_24_slice_interp_full3.pt' -> 'slice_interp_full3'
    """
    filename = Path(checkpoint_path).stem  # Remove .pt extension

    # Pattern: checkpoint_<type>_epoch_<num>_<suffix>
    # Split by '_' and find the suffix after epoch number
    parts = filename.split('_')

    # Find 'epoch' index
    try:
        epoch_idx = parts.index('epoch')
        # Suffix is everything after epoch_<number>
        if len(parts) > epoch_idx + 2:
            suffix = '_'.join(parts[epoch_idx + 2:])
            return suffix
    except (ValueError, IndexError):
        pass

    return None


def list_all_checkpoints(checkpoint_dir: str, model_suffix: Optional[str] = None) -> list:
    """
    List all checkpoints in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model_suffix: Optional model suffix to filter by

    Returns:
        List of checkpoint paths, sorted by modification time (newest first)
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return []

    # Build pattern
    if model_suffix:
        pattern = f"checkpoint_*_{model_suffix}.pt"
    else:
        pattern = "checkpoint_*.pt"

    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob(pattern))

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    logger.info(f"Found {len(checkpoints)} checkpoints")

    return [str(p) for p in checkpoints]
