"""
Unified DataLoader Interface for CT Slice Interpolation

This module provides a single function that routes to the appropriate
dataset implementation based on configuration:
- Full-volume mode: SliceInterpolationDataset (variable depth, padding)
- Patch-based mode: PatchSliceInterpolationDataset (fixed size, efficient)

Usage:
    from data import get_unified_dataloader

    # Patch-based training (VAE, diffusion)
    config = {
        'data_source': 'slice_interpolation',
        'use_patches': True,
        'processed_dir': '/path/to/cache',
        'patch_depth_thick': 8,
        'patch_depth_thin': 48,
        'patch_size': [192, 192],
        ...
    }
    dataloader = get_unified_dataloader(config, split='train')

    # Full-volume evaluation
    config = {
        'data_source': 'slice_interpolation',
        'use_patches': False,
        'dataset_path': '/path/to/dataset',
        'max_thick_slices': 50,
        'max_thin_slices': 300,
        ...
    }
    dataloader = get_unified_dataloader(config, split='train')
"""

from typing import Optional, Dict, Any
from torch.utils.data import DataLoader


def get_dataloader(config: Dict[str, Any], split: str = 'train') -> DataLoader:
    """
    Get appropriate dataloader for CT slice interpolation

    Args:
        config: Configuration dictionary with dataset parameters
        split: Dataset split ('train', 'val', 'test')

    Returns:
        DataLoader instance

    Configuration Keys:
        data_source: Must be 'slice_interpolation'
        use_patches: bool (True for patch-based, False for full-volume)

        For patch-based mode (use_patches=True):
            - processed_dir: Path to preprocessed .pt cache files
            - patch_depth_thick: Thick patch depth (e.g., 8)
            - patch_depth_thin: Thin patch depth (e.g., 48)
            - patch_size: [H, W] spatial patch size (e.g., [192, 192])
            - augment: bool (enable random flips/rotations for training)

        For full-volume mode (use_patches=False):
            - dataset_path: Path to dataset with .zip files
            - max_thick_slices: Max thick slices to keep (e.g., 50)
            - max_thin_slices: Max thin slices to keep (e.g., 300)
            - extract_dir: Temp extraction directory

        Common:
            - categories: ['APE', 'non-APE']
            - resolution: [H, W]
            - batch_size: int
            - num_workers: int
            - window_center: int (default: 40)
            - window_width: int (default: 400)
            - val_split: float (default: 0.1)
            - test_split: float (default: 0.1)
            - seed: int (default: 42)

    Example:
        >>> # Patch-based training
        >>> config = {
        ...     'data_source': 'slice_interpolation',
        ...     'use_patches': True,
        ...     'processed_dir': '/workspace/storage/.cache/processed',
        ...     'patch_depth_thick': 8,
        ...     'patch_depth_thin': 48,
        ...     'patch_size': [192, 192],
        ...     'batch_size': 4,
        ...     'augment': True
        ... }
        >>> loader = get_dataloader(config, split='train')
    """

    data_source = config.get('data_source', 'slice_interpolation')

    if data_source != 'slice_interpolation':
        raise ValueError(
            f"Only 'slice_interpolation' data source is supported. Got: {data_source}"
        )

    print(f"Using data source: {data_source}")
    return _get_slice_interpolation_dataloader(config, split)


def _get_slice_interpolation_dataloader(config: Dict[str, Any], split: str) -> DataLoader:
    """
    Get dataloader for CT slice interpolation

    Supports two modes:
    1. Full-volume mode (use_patches=False):
       - Input: Full thick slices (~50 @ 5.0mm)
       - Output: Full thin slices (~300 @ 1.0mm)
       - Variable depth handling with padding

    2. Patch-based mode (use_patches=True):
       - Input: 3D patches (8 thick slices)
       - Output: 3D patches (48 thin slices)
       - Fixed size, no padding, larger batch sizes
    """
    use_patches = config.get('use_patches', False)

    if use_patches:
        # Patch-based mode
        from .patch_slice_interpolation_dataset import get_patch_dataloader

        # Use processed_dir (cache directory with .pt files) for patches
        processed_dir = config.get('processed_dir', config.get('extract_dir'))
        if not processed_dir:
            raise ValueError(
                "processed_dir or extract_dir is required for patch-based slice_interpolation. "
                "This should point to the directory with preprocessed .pt files."
            )

        print(f"Loading CT Slice Interpolation data (PATCH-BASED MODE)")
        print(f"Processed cache: {processed_dir}")
        print(f"Patch config: {config.get('patch_depth_thick', 8)} thick → {config.get('patch_depth_thin', 48)} thin @ {config.get('patch_size', [192, 192])}")
        print(f"Mode: 3D patches (fixed size, no padding, larger batches)")

        return get_patch_dataloader(
            processed_dir=processed_dir,
            config=config,
            split=split
        )
    else:
        # Full-volume mode (original)
        from .slice_interpolation_dataset import get_slice_interpolation_dataloader

        dataset_path = config.get('dataset_path')
        if not dataset_path:
            raise ValueError("dataset_path is required for slice_interpolation data source")

        print(f"Loading CT Slice Interpolation data (FULL-VOLUME MODE)")
        print(f"Dataset path: {dataset_path}")
        print(f"Task: Thick slices (50 @ 5.0mm) → Thin slices (300 @ 1.0mm)")
        print(f"Mode: Full volumes (NO patches, NO downsampling)")

        return get_slice_interpolation_dataloader(
            data_dir=dataset_path,
            config=config,
            split=split
        )


def create_training_config(
    dataset_path: str,
    use_patches: bool = True,
    batch_size: int = 4,
    num_workers: int = 4,
    resolution: tuple = (512, 512),
    **kwargs
) -> Dict[str, Any]:
    """
    Helper function to create a CT slice interpolation training configuration

    Args:
        dataset_path: Path to dataset directory or processed cache
        use_patches: Use patch-based mode (True) or full-volume mode (False)
        batch_size: Batch size
        num_workers: Number of data loading workers
        resolution: (H, W) resolution
        **kwargs: Additional config parameters

    Returns:
        Configuration dictionary

    Example:
        >>> # Patch-based training
        >>> config = create_training_config(
        ...     dataset_path='/workspace/storage/.cache/processed',
        ...     use_patches=True,
        ...     batch_size=4,
        ...     num_workers=4
        ... )

        >>> # Full-volume evaluation
        >>> config = create_training_config(
        ...     dataset_path='/workspace/storage/dataset',
        ...     use_patches=False,
        ...     batch_size=1,
        ...     num_workers=4
        ... )
    """

    config = {
        'data_source': 'slice_interpolation',
        'use_patches': use_patches,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'resolution': list(resolution),
        'categories': ['APE', 'non-APE'],
        'window_center': 40,
        'window_width': 400,
        'val_split': 0.1,
        'test_split': 0.1,
        'seed': 42,
    }

    if use_patches:
        # Patch-based configuration
        config.update({
            'processed_dir': dataset_path,
            'patch_depth_thick': kwargs.get('patch_depth_thick', 8),
            'patch_depth_thin': kwargs.get('patch_depth_thin', 48),
            'patch_size': kwargs.get('patch_size', [192, 192]),
            'augment': kwargs.get('augment', True),
        })
    else:
        # Full-volume configuration
        config.update({
            'dataset_path': dataset_path,
            'max_thick_slices': kwargs.get('max_thick_slices', 50),
            'max_thin_slices': kwargs.get('max_thin_slices', 300),
            'extract_dir': kwargs.get('extract_dir', '/tmp/slice_cache'),
        })

    # Add any additional kwargs
    config.update(kwargs)

    return config


if __name__ == "__main__":
    print("=" * 70)
    print("CT Slice Interpolation DataLoader Configuration")
    print("=" * 70)
    print()

    # Example 1: Patch-based training
    print("Example 1: Patch-Based Training (VAE/Diffusion)")
    print("-" * 70)
    patch_config = create_training_config(
        dataset_path='/workspace/storage/.cache/processed',
        use_patches=True,
        batch_size=4,
        num_workers=4,
        patch_depth_thick=8,
        patch_depth_thin=48,
        patch_size=[192, 192]
    )
    print("Config:")
    for key, value in patch_config.items():
        print(f"  {key}: {value}")
    print()

    # Example 2: Full-volume evaluation
    print("Example 2: Full-Volume Evaluation")
    print("-" * 70)
    full_config = create_training_config(
        dataset_path='/workspace/storage/dataset',
        use_patches=False,
        batch_size=1,
        num_workers=4,
        max_thick_slices=50,
        max_thin_slices=300
    )
    print("Config:")
    for key, value in full_config.items():
        print(f"  {key}: {value}")
    print()

    print("=" * 70)
    print()
    print("Usage in training script:")
    print("-" * 70)
    print("""
from data import get_unified_dataloader, create_training_config

# Create config
config = create_training_config(
    dataset_path='/workspace/storage/.cache/processed',
    use_patches=True,
    batch_size=4
)

# Get dataloader
train_loader = get_unified_dataloader(config, split='train')

# Training loop
for batch in train_loader:
    thick_slices = batch['input']   # (B, C, D_thick, H, W)
    thin_slices = batch['target']   # (B, C, D_thin, H, W)
    # ... train your model
    """)
