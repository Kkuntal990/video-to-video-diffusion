"""
CT Slice Interpolation Dataset - Patch-Based Version

This dataset extracts 3D patches from full volumes for efficient patch-based training.
Unlike the full-volume dataset, this version:
- Extracts random 3D patches (D×H×W) from cached volumes
- Returns fixed-size patches (no padding needed)
- Enables larger batch sizes and faster training

Task: Anisotropic super-resolution in depth dimension using patches
Input: 8 thick slices @ 5.0mm → Output: 48 thin slices @ 1.0mm
Patch size: (8, 192, 192) thick → (48, 192, 192) thin
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatchSliceInterpolationDataset(Dataset):
    """
    Patch-based CT Slice Interpolation Dataset

    Extracts random 3D patches from preprocessed full volumes for memory-efficient training.

    Patch dimensions:
    - HR (thin) patch: (1, 48, 192, 192) in range [-1, 1]
    - LR (thick) patch: (1, 8, 192, 192) in range [-1, 1]
    - 6× depth ratio: 48 thin slices ↔ 8 thick slices

    Args:
        processed_dir: Directory containing preprocessed .pt files
        patch_depth_thin: Number of thin slices in HR patch (default: 48)
        patch_depth_thick: Number of thick slices in LR patch (default: 8)
        patch_size: (H, W) spatial size of patches (default: (192, 192))
        split: 'train', 'val', or 'test'
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        seed: Random seed for reproducible splits
        augment: Whether to apply data augmentation (random flips/rotations)
    """

    def __init__(
        self,
        processed_dir: str,
        patch_depth_thin: int = 48,
        patch_depth_thick: int = 8,
        patch_size: Tuple[int, int] = (192, 192),
        split: str = 'train',
        val_ratio: float = 0.15,
        test_ratio: float = 0.10,
        seed: int = 42,
        augment: bool = True,
    ):
        super().__init__()

        self.processed_dir = Path(processed_dir)
        self.patch_depth_thin = patch_depth_thin
        self.patch_depth_thick = patch_depth_thick
        self.patch_size = patch_size  # (H, W)
        self.split = split
        self.augment = augment and (split == 'train')  # Only augment training

        # Validate processed directory
        if not self.processed_dir.exists():
            raise ValueError(f"Processed directory not found: {processed_dir}")

        # Find all preprocessed .pt files
        all_files = sorted(list(self.processed_dir.glob("case_*.pt")))

        if len(all_files) == 0:
            raise ValueError(f"No preprocessed .pt files found in {processed_dir}")

        logger.info(f"Found {len(all_files)} preprocessed patient files")

        # Create deterministic train/val/test split
        random.seed(seed)
        random.shuffle(all_files)

        n_total = len(all_files)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val - n_test

        if split == 'train':
            self.patient_files = all_files[:n_train]
        elif split == 'val':
            self.patient_files = all_files[n_train:n_train + n_val]
        elif split == 'test':
            self.patient_files = all_files[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {split}. Choose from ['train', 'val', 'test']")

        logger.info(f"Split '{split}': {len(self.patient_files)} patients")
        logger.info(f"Patch config: thin={patch_depth_thin}, thick={patch_depth_thick}, spatial={patch_size}")

        # Verify depth ratio is approximately 6×
        depth_ratio = patch_depth_thin / patch_depth_thick
        if not (5.5 <= depth_ratio <= 6.5):
            logger.warning(f"Depth ratio {depth_ratio:.2f} is not close to expected 6× ratio")

    def __len__(self) -> int:
        return len(self.patient_files)

    def extract_random_patch(
        self,
        thick_volume: torch.Tensor,
        thin_volume: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract spatially-aligned random 3D patches from thick and thin volumes.

        Args:
            thick_volume: (1, D_thick, 512, 512) - typically (1, 50, 512, 512)
            thin_volume: (1, D_thin, 512, 512) - typically (1, 300, 512, 512)

        Returns:
            thick_patch: (1, patch_depth_thick, patch_h, patch_w) - (1, 8, 192, 192)
            thin_patch: (1, patch_depth_thin, patch_h, patch_w) - (1, 48, 192, 192)
        """
        _, D_thick, H, W = thick_volume.shape
        _, D_thin, _, _ = thin_volume.shape
        patch_h, patch_w = self.patch_size

        # 1. Sample random spatial crop (same for both thick and thin)
        # Ensure patch fits within 512×512
        if H < patch_h or W < patch_w:
            raise ValueError(f"Volume size ({H}, {W}) smaller than patch size ({patch_h}, {patch_w})")

        y0 = random.randint(0, H - patch_h)
        x0 = random.randint(0, W - patch_w)

        # 2. Sample random depth window for thin (HR) patch
        if D_thin < self.patch_depth_thin:
            # If volume is smaller than patch depth, pad or use full volume
            z_thin_start = 0
            thin_patch_depth = D_thin
        else:
            z_thin_start = random.randint(0, D_thin - self.patch_depth_thin)
            thin_patch_depth = self.patch_depth_thin

        z_thin_end = z_thin_start + thin_patch_depth

        # 3. Extract thin (HR) patch
        thin_patch = thin_volume[:, z_thin_start:z_thin_end, y0:y0+patch_h, x0:x0+patch_w]

        # 4. Map thin depth range to thick depth range
        # Thick and thin have ~6× ratio (e.g., 50 thick @ 5mm vs 300 thin @ 1mm)
        # For each thin slice index, the corresponding thick slice is: thick_idx ≈ thin_idx / 6

        # Compute thick slice range that covers the thin patch
        thick_z_start = int(z_thin_start * D_thick / D_thin)  # Map start
        thick_z_end = int(z_thin_end * D_thick / D_thin)      # Map end

        # Ensure we have at least 1 slice
        if thick_z_end <= thick_z_start:
            thick_z_end = thick_z_start + 1

        # Clamp to valid range
        thick_z_start = max(0, thick_z_start)
        thick_z_end = min(D_thick, thick_z_end)

        # 5. Extract thick slices and resample to exactly patch_depth_thick slices
        thick_sub = thick_volume[:, thick_z_start:thick_z_end, y0:y0+patch_h, x0:x0+patch_w]
        # thick_sub shape: (1, n_thick_sub, patch_h, patch_w) where n_thick_sub varies

        # Resample along depth to get exactly patch_depth_thick slices
        # F.interpolate for 3D (trilinear) requires 5D input: (B, C, D, H, W)
        thick_patch = F.interpolate(
            thick_sub.unsqueeze(0),  # Add batch dim: (1, 1, n_thick_sub, patch_h, patch_w)
            size=(self.patch_depth_thick, patch_h, patch_w),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dim: (1, 8, 192, 192)

        # 6. Ensure thin patch is also correct size (pad if needed for edge cases)
        if thin_patch.shape[1] < self.patch_depth_thin:
            # Pad depth if volume was too small
            pad_depth = self.patch_depth_thin - thin_patch.shape[1]
            thin_patch = F.pad(thin_patch, (0, 0, 0, 0, 0, pad_depth), value=-1.0)

        return thick_patch, thin_patch

    def augment_patch(
        self,
        thick_patch: torch.Tensor,
        thin_patch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random data augmentation to patches.

        Augmentations:
        - Random horizontal flip (50%)
        - Random vertical flip (50%)
        - Random 90° rotation (0°, 90°, 180°, 270°)

        Args:
            thick_patch: (1, D, H, W)
            thin_patch: (1, D, H, W)

        Returns:
            Augmented patches with same shapes
        """
        # Random horizontal flip
        if random.random() > 0.5:
            thick_patch = torch.flip(thick_patch, dims=[3])  # Flip W
            thin_patch = torch.flip(thin_patch, dims=[3])

        # Random vertical flip
        if random.random() > 0.5:
            thick_patch = torch.flip(thick_patch, dims=[2])  # Flip H
            thin_patch = torch.flip(thin_patch, dims=[2])

        # Random 90° rotation (0, 1, 2, 3 rotations)
        k = random.randint(0, 3)
        if k > 0:
            # torch.rot90 operates on last 2 dims (H, W)
            thick_patch = torch.rot90(thick_patch, k=k, dims=[2, 3])
            thin_patch = torch.rot90(thin_patch, k=k, dims=[2, 3])

        return thick_patch, thin_patch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load preprocessed volume and extract random patch.

        Returns:
            dict with:
                'x_hr': thin patch (1, 48, 192, 192) in [-1, 1]
                'x_lr': thick patch (1, 8, 192, 192) in [-1, 1]
                'category': 'APE' or 'non-APE'
                'patient_id': patient identifier
        """
        patient_file = self.patient_files[idx]

        # Load preprocessed full volume
        try:
            sample_dict = torch.load(patient_file, weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load {patient_file}: {e}")
            # Return dummy sample
            return self._get_dummy_sample(patient_file.stem)

        thick_volume = sample_dict.get('input', sample_dict.get('thick'))  # (1, D_thick, 512, 512)
        thin_volume = sample_dict.get('target', sample_dict.get('thin'))   # (1, D_thin, 512, 512)

        # Extract random patch
        thick_patch, thin_patch = self.extract_random_patch(thick_volume, thin_volume)

        # Apply data augmentation (only for training)
        if self.augment:
            thick_patch, thin_patch = self.augment_patch(thick_patch, thin_patch)

        return {
            'x_lr': thick_patch,      # (1, 8, 192, 192) - Low-resolution (thick slices)
            'x_hr': thin_patch,       # (1, 48, 192, 192) - High-resolution (thin slices)
            'input': thick_patch,     # Alias for compatibility
            'target': thin_patch,     # Alias for compatibility
            'category': sample_dict.get('category', 'unknown'),
            'patient_id': sample_dict.get('patient_id', patient_file.stem),
        }

    def _get_dummy_sample(self, patient_id: str) -> Dict[str, torch.Tensor]:
        """Create dummy sample for error cases"""
        patch_h, patch_w = self.patch_size
        dummy_thick = torch.zeros(1, self.patch_depth_thick, patch_h, patch_w)
        dummy_thin = torch.zeros(1, self.patch_depth_thin, patch_h, patch_w)

        return {
            'x_lr': dummy_thick,
            'x_hr': dummy_thin,
            'input': dummy_thick,
            'target': dummy_thin,
            'category': 'unknown',
            'patient_id': patient_id,
        }


def get_patch_dataloader(
    processed_dir: str,
    config: dict,
    split: str = 'train'
) -> torch.utils.data.DataLoader:
    """
    Create dataloader for patch-based CT slice interpolation.

    Args:
        processed_dir: Path to directory with preprocessed .pt files
        config: Config dict with patch parameters
        split: 'train', 'val', or 'test'

    Returns:
        dataloader: PyTorch DataLoader with fixed-size patches (no custom collation needed)
    """
    from torch.utils.data import DataLoader, Subset

    # Extract patch configuration
    patch_depth_thin = config.get('patch_depth_thin', 48)
    patch_depth_thick = config.get('patch_depth_thick', 8)
    patch_size = tuple(config.get('patch_size', [192, 192]))

    dataset = PatchSliceInterpolationDataset(
        processed_dir=processed_dir,
        patch_depth_thin=patch_depth_thin,
        patch_depth_thick=patch_depth_thick,
        patch_size=patch_size,
        split=split,
        val_ratio=config.get('val_split', 0.15),
        test_ratio=config.get('test_split', 0.10),
        seed=config.get('seed', 42),
        augment=config.get('augment', True),
    )

    # Limit dataset size for debugging (if specified)
    if split == 'train':
        max_samples = config.get('max_train_samples')
    elif split == 'val':
        max_samples = config.get('max_val_samples')
    else:
        max_samples = None

    if max_samples is not None and isinstance(max_samples, int) and max_samples > 0:
        dataset_size = len(dataset)
        max_samples = min(max_samples, dataset_size)  # Don't exceed dataset size
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)
        print(f"[DEBUG] Limiting {split} dataset to {max_samples}/{dataset_size} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=(split == 'train'),
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        drop_last=config.get('drop_last', True) and (split == 'train'),
        persistent_workers=True if config.get('num_workers', 4) > 0 else False,
        prefetch_factor=2 if config.get('num_workers', 4) > 0 else None,  # Prefetch 2 batches per worker (30-40% faster)
        # No custom collate_fn needed - all patches are same size!
    )

    return dataloader
