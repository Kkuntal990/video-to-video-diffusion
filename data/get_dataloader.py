"""
Unified DataLoader Interface for APE-Data

This module provides a single function that automatically chooses
the right dataloader based on configuration:
- Local files: Uses APEDataset (for testing with downloaded samples)
- HuggingFace: Uses APEHuggingFaceDataset (for cloud training)
- Slice Interpolation: Uses SliceInterpolationDataset (for CT slice interpolation with full volumes)

Usage:
    from data.get_dataloader import get_dataloader

    # For local testing
    config = {
        'data_source': 'local',
        'data_dir': '/path/to/local/data',
        ...
    }
    dataloader = get_dataloader(config, split='train')

    # For cloud training with HuggingFace
    config = {
        'data_source': 'huggingface',
        'dataset_name': 't2ance/APE-data',
        ...
    }
    dataloader = get_dataloader(config, split='train')

    # For CT slice interpolation (full volumes)
    config = {
        'data_source': 'slice_interpolation',
        'dataset_path': '/path/to/dataset',
        'resolution': [512, 512],
        'max_thick_slices': 50,
        'max_thin_slices': 300,
        ...
    }
    dataloader = get_dataloader(config, split='train')
"""

from typing import Optional, Dict, Any
from torch.utils.data import DataLoader


def get_dataloader(config: Dict[str, Any], split: str = 'train') -> DataLoader:
    """
    Get appropriate dataloader based on configuration

    Args:
        config: Configuration dictionary with dataset parameters
        split: Dataset split ('train', 'val', 'test')

    Returns:
        DataLoader instance

    Configuration Keys:
        data_source: 'local', 'huggingface', 'slice_interpolation', or 'auto'

        For local:
            - data_dir: Path to local data directory
            - categories: ['APE', 'non-APE']
            - cache_extracted: bool

        For HuggingFace:
            - dataset_name: HuggingFace dataset identifier
            - streaming: bool (default: True)
            - cache_dir: Optional cache directory

        For slice_interpolation:
            - dataset_path: Path to dataset with .zip files
            - max_thick_slices: Max thick slices to keep (e.g., 50)
            - max_thin_slices: Max thin slices to keep (e.g., 300)
            - categories: ['APE', 'non-APE']

        Common:
            - num_frames: int (not used for slice_interpolation)
            - resolution: [H, W]
            - batch_size: int
            - num_workers: int
            - window_center: int
            - window_width: int

    Examples:
        >>> # Local data
        >>> config = {
        ...     'data_source': 'local',
        ...     'data_dir': '/data/APE-data',
        ...     'num_frames': 16,
        ...     'resolution': [256, 256],
        ...     'batch_size': 4
        ... }
        >>> loader = get_dataloader(config, split='train')

        >>> # HuggingFace streaming
        >>> config = {
        ...     'data_source': 'huggingface',
        ...     'dataset_name': 't2ance/APE-data',
        ...     'streaming': True,
        ...     'num_frames': 16,
        ...     'resolution': [256, 256],
        ...     'batch_size': 4
        ... }
        >>> loader = get_dataloader(config, split='train')
    """

    data_source = config.get('data_source', 'auto')

    # Auto-detect if not specified
    if data_source == 'auto':
        if 'data_dir' in config:
            data_source = 'local'
        elif 'dataset_name' in config:
            data_source = 'huggingface'
        else:
            raise ValueError(
                "Could not auto-detect data source. "
                "Please specify 'data_source' as 'local' or 'huggingface', "
                "or provide 'data_dir' (local) or 'dataset_name' (HuggingFace)"
            )

    print(f"Using data source: {data_source}")

    if data_source == 'local':
        return _get_local_dataloader(config, split)
    elif data_source == 'huggingface' or data_source == 'hf':
        return _get_hf_dataloader(config, split)
    elif data_source == 'slice_interpolation':
        return _get_slice_interpolation_dataloader(config, split)
    else:
        raise ValueError(f"Unknown data_source: {data_source}. Must be 'local', 'huggingface', or 'slice_interpolation'")


def _get_local_dataloader(config: Dict[str, Any], split: str) -> DataLoader:
    """Get dataloader for local files"""
    from .ape_dataset import get_ape_dataloader

    data_dir = config.get('data_dir')
    if not data_dir:
        raise ValueError("data_dir is required for local data source")

    print(f"Loading data from local directory: {data_dir}")

    return get_ape_dataloader(data_dir, config, split=split)


def _get_hf_dataloader(config: Dict[str, Any], split: str) -> DataLoader:
    """Get dataloader for HuggingFace dataset"""
    dataset_name = config.get('dataset_name', 't2ance/APE-data')
    streaming = config.get('streaming', True)
    use_cache = config.get('use_cache', True)

    print(f"Loading data from HuggingFace: {dataset_name}")

    # Use cached preprocessing (recommended for persistent storage)
    if use_cache and not streaming:
        from .ape_cached_dataset import get_ape_cached_dataloader

        print(f"Using CACHED mode (download + preprocess once, then load tensors)")
        print(f"Cache directory: {config.get('cache_dir', '/workspace/storage/ape_cache')}")

        return get_ape_cached_dataloader(
            dataset_name=dataset_name,
            cache_dir=config.get('cache_dir', '/workspace/storage/ape_cache'),
            num_frames=config.get('num_frames', 16),
            resolution=tuple(config.get('resolution', [256, 256])),
            batch_size=config.get('batch_size', 4),
            num_workers=config.get('num_workers', 4),
            categories=config.get('categories'),
            window_center=config.get('window_center', 40),
            window_width=config.get('window_width', 400),
            force_reprocess=config.get('force_reprocess', False),
            split=split,
            val_split=config.get('val_split', 0.15),
            test_split=config.get('test_split', 0.10),
            seed=config.get('seed', 42),
            local_zip_dir=config.get('local_zip_dir')
        )

    # Use streaming mode (old behavior)
    else:
        from .ape_hf_dataset import get_ape_hf_dataloader

        print(f"Using STREAMING mode (downloads on-the-fly, preprocesses every epoch)")
        print(f"⚠ WARNING: This is slow! Consider using use_cache=True with streaming=False")

        return get_ape_hf_dataloader(
            dataset_name=dataset_name,
            split=split,
            num_frames=config.get('num_frames', 16),
            resolution=tuple(config.get('resolution', [256, 256])),
            batch_size=config.get('batch_size', 4),
            num_workers=config.get('num_workers', 4),
            categories=config.get('categories'),
            streaming=streaming,
            window_center=config.get('window_center', 40),
            window_width=config.get('window_width', 400),
            cache_dir=config.get('cache_dir'),
            max_samples=config.get('max_samples')
        )


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
    data_source: str = 'huggingface',
    dataset_name: str = 't2ance/APE-data',
    data_dir: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    num_frames: int = 16,
    resolution: tuple = (256, 256),
    streaming: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Helper function to create a training configuration

    Args:
        data_source: 'local' or 'huggingface'
        dataset_name: HuggingFace dataset name (if using HF)
        data_dir: Local data directory (if using local)
        batch_size: Batch size
        num_workers: Number of data loading workers
        num_frames: Number of frames per video
        resolution: (H, W) resolution
        streaming: Use streaming for HuggingFace
        **kwargs: Additional config parameters

    Returns:
        Configuration dictionary

    Example:
        >>> # For cloud training with HuggingFace
        >>> config = create_training_config(
        ...     data_source='huggingface',
        ...     batch_size=4,
        ...     num_workers=4
        ... )

        >>> # For local testing
        >>> config = create_training_config(
        ...     data_source='local',
        ...     data_dir='/path/to/data',
        ...     batch_size=1,
        ...     num_workers=0
        ... )
    """

    config = {
        'data_source': data_source,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_frames': num_frames,
        'resolution': list(resolution),
        'categories': ['APE', 'non-APE'],
        'window_center': 40,
        'window_width': 400,
    }

    if data_source == 'huggingface':
        config.update({
            'dataset_name': dataset_name,
            'streaming': streaming,
        })
    elif data_source == 'local':
        if not data_dir:
            raise ValueError("data_dir is required for local data source")
        config.update({
            'data_dir': data_dir,
            'cache_extracted': True,
        })

    # Add any additional kwargs
    config.update(kwargs)

    return config


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("DataLoader Configuration Helper")
    print("=" * 70)
    print()

    # Example 1: Local data
    print("Example 1: Local Data Configuration")
    print("-" * 70)
    local_config = create_training_config(
        data_source='local',
        data_dir='/path/to/local/data',
        batch_size=2,
        num_workers=0,
        num_frames=8,
        resolution=(128, 128)
    )
    print("Config:")
    for key, value in local_config.items():
        print(f"  {key}: {value}")
    print()

    # Example 2: HuggingFace data
    print("Example 2: HuggingFace Data Configuration")
    print("-" * 70)
    hf_config = create_training_config(
        data_source='huggingface',
        dataset_name='t2ance/APE-data',
        batch_size=4,
        num_workers=4,
        num_frames=16,
        resolution=(256, 256),
        streaming=True
    )
    print("Config:")
    for key, value in hf_config.items():
        print(f"  {key}: {value}")
    print()

    print("=" * 70)
    print()
    print("Usage in training script:")
    print("-" * 70)
    print("""
from data.get_dataloader import get_dataloader, create_training_config

# Create config
config = create_training_config(
    data_source='huggingface',  # or 'local'
    batch_size=4,
    num_frames=16,
    resolution=(256, 256)
)

# Get dataloader
train_loader = get_dataloader(config, split='train')

# Training loop
for batch in train_loader:
    inputs = batch['input']   # (B, 3, T, H, W)
    targets = batch['target']  # (B, 3, T, H, W)
    # ... train your model
    """)
