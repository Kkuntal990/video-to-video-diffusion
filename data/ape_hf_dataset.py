"""
APE-Data HuggingFace Dataset Loader

This module provides dataset loaders that work with HuggingFace's datasets library
to download and stream the full APE-data dataset for cloud training.

Usage:
    from data.ape_hf_dataset import get_ape_hf_dataloader

    dataloader = get_ape_hf_dataloader(
        dataset_name='t2ance/APE-data',
        split='train',
        num_frames=16,
        resolution=(256, 256),
        batch_size=4
    )
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import os
from pathlib import Path
from typing import Optional, List, Tuple
import tempfile
import shutil

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not installed. Install with: pip install pydicom")

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: datasets library not installed. Install with: pip install datasets")

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")

from .transforms import VideoTransform


class APEHuggingFaceDataset(IterableDataset):
    """
    APE-Data HuggingFace Streaming Dataset

    Streams APE-data directly from HuggingFace without downloading everything upfront.
    Ideal for cloud training with large datasets.

    Features:
    - Streaming: Downloads data on-the-fly
    - No disk space needed upfront
    - Supports both APE and non-APE categories
    - Automatic DICOM processing
    """

    def __init__(
        self,
        dataset_name: str = 't2ance/APE-data',
        split: str = 'train',
        num_frames: int = 16,
        resolution: Tuple[int, int] = (256, 256),
        categories: Optional[List[str]] = None,
        transform=None,
        normalize: bool = True,
        window_center: int = 40,
        window_width: int = 400,
        streaming: bool = True,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name (e.g., 't2ance/APE-data')
            split: Dataset split ('train', 'validation', 'test')
            num_frames: Number of slices to sample from each volume
            resolution: (H, W) to resize each slice
            categories: List of categories ['APE', 'non-APE'] or None for all
            transform: Additional transforms
            normalize: Apply HU windowing and normalization
            window_center: HU window center for CT windowing
            window_width: HU window width for CT windowing
            streaming: Use streaming mode (recommended for large datasets)
            cache_dir: Directory to cache downloaded files (None = default)
        """
        super().__init__()

        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required. Install with: pip install pydicom")

        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library is required. Install with: pip install datasets")

        self.dataset_name = dataset_name
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.categories = categories if categories else ['APE', 'non-APE']
        self.normalize = normalize
        self.window_center = window_center
        self.window_width = window_width
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.max_samples = max_samples

        # Blacklist of verified corrupted cases (empty DICOM data at source)
        # Verified by manual download - these cases have only admin files
        self.corrupted_cases = {
            'case_190', 'case_191', 'case_192', 'case_193', 'case_194',
            'case_195', 'case_196', 'case_197', 'case_198', 'case_199',
            'case_200', 'case_201', 'case_202', 'case_203', 'case_204',
            'case_205', 'case_206'
        }

        # Default transform
        if transform is None:
            self.transform = VideoTransform(resolution=resolution, num_frames=num_frames)
        else:
            self.transform = transform

        # Load dataset from HuggingFace
        print(f"Loading APE-data from HuggingFace: {dataset_name}")
        print(f"Split: {split}")
        print(f"Streaming: {streaming}")
        print(f"Categories: {self.categories}")

        try:
            # Since APE-data contains ZIP files, we need to list and download them directly
            if not HF_HUB_AVAILABLE:
                raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

            print("Listing files in repository...")
            all_files = list(list_repo_files(dataset_name, repo_type="dataset"))

            # Filter ZIP files by category
            self.zip_files = []
            for category in self.categories:
                category_files = [f for f in all_files if f.startswith(f"{category}/") and f.endswith(".zip")]
                self.zip_files.extend(category_files)

            # Filter out corrupted cases
            valid_zip_files = []
            corrupted_count = 0
            for zip_file in self.zip_files:
                zip_hash = Path(zip_file).stem
                if zip_hash not in self.corrupted_cases:
                    valid_zip_files.append(zip_file)
                else:
                    corrupted_count += 1

            self.zip_files = valid_zip_files

            print(f"✓ Found {len(self.zip_files)} valid ZIP files to process")
            if corrupted_count > 0:
                print(f"  ⚠ Skipping {corrupted_count} corrupted cases (no DICOM data)")
            print(f"  Categories: {', '.join(self.categories)}")

            # Store dataset name for downloading files later
            self.dataset_name = dataset_name
            self.hf_dataset = None  # Not using load_dataset for ZIP files

        except Exception as e:
            print(f"✗ Failed to load dataset: {e}")
            print("\nPossible solutions:")
            print("1. Check dataset name: t2ance/APE-data")
            print("2. Login to HuggingFace: huggingface-cli login")
            print("3. Request access to the dataset if it's private")
            raise

    def _load_dicom_from_hf_sample(self, sample: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load DICOM volumes from a HuggingFace dataset sample

        Args:
            sample: HuggingFace dataset sample containing patient data

        Returns:
            (baseline_volume, followup_volume): Tuple of numpy arrays or (None, None) on error
        """
        # The exact structure depends on how APE-data is stored in HuggingFace
        # This is a generic implementation that should work with common formats

        try:
            # Method 0: If we extracted from ZIP and have a case_dir
            if 'case_dir' in sample:
                case_dir = Path(sample['case_dir'])
                # Look for subdirectories containing DICOM files (1/ and 2/ or baseline/ and followup/)
                # Filter out hidden directories (starting with '.')
                subdirs = [d for d in case_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

                if len(subdirs) >= 2:
                    # Sort to ensure consistent ordering (by name)
                    subdirs = sorted(subdirs, key=lambda x: x.name)
                    baseline_volume = self._load_dicom_directory(subdirs[0])
                    followup_volume = self._load_dicom_directory(subdirs[1])
                    return baseline_volume, followup_volume
                elif len(subdirs) == 1:
                    # If only 1 subdirectory, look for subdirectories inside it (patient folder structure)
                    inner_dir = subdirs[0]
                    # Filter out hidden directories and look for numeric folders (1, 2, etc.)
                    inner_subdirs = [d for d in inner_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    if len(inner_subdirs) >= 2:
                        # Sort numerically if possible, otherwise alphabetically
                        try:
                            inner_subdirs = sorted(inner_subdirs, key=lambda x: int(x.name))
                        except ValueError:
                            inner_subdirs = sorted(inner_subdirs, key=lambda x: x.name)
                        baseline_volume = self._load_dicom_directory(inner_subdirs[0])
                        followup_volume = self._load_dicom_directory(inner_subdirs[1])
                        return baseline_volume, followup_volume
                    else:
                        # Debug: List what's actually in the directory
                        all_items = list(inner_dir.iterdir())
                        print(f"Debug: Contents of {inner_dir.name}: {[item.name for item in all_items]}")
                        print(f"Warning: Found 1 subdirectory but it contains {len(inner_subdirs)} subdirectories")
                        return None, None
                else:
                    # Debug: List what's actually in the case directory
                    all_items = list(case_dir.iterdir())
                    print(f"Debug: Contents of {case_dir.name}: {[item.name for item in all_items]}")
                    print(f"Warning: Expected at least 1 subdirectory in {case_dir}, found {len(subdirs)}")
                    return None, None

            # Method 1: If data is stored as files
            elif 'baseline' in sample and 'followup' in sample:
                baseline_volume = self._process_dicom_data(sample['baseline'])
                followup_volume = self._process_dicom_data(sample['followup'])
                return baseline_volume, followup_volume

            # Method 2: If data is stored in nested structure
            elif 'data' in sample:
                data = sample['data']
                if isinstance(data, dict) and '1' in data and '2' in data:
                    baseline_volume = self._process_dicom_data(data['1'])
                    followup_volume = self._process_dicom_data(data['2'])
                    return baseline_volume, followup_volume

            # Method 3: If data is stored as paths that need to be downloaded
            elif 'baseline_path' in sample and 'followup_path' in sample:
                # Download and process files
                baseline_volume = self._load_from_path(sample['baseline_path'])
                followup_volume = self._load_from_path(sample['followup_path'])
                return baseline_volume, followup_volume

            # If none of the above work, print structure for debugging
            print(f"Warning: Unknown sample structure. Keys: {sample.keys()}")
            return None, None

        except Exception as e:
            print(f"Error loading DICOM from HuggingFace sample: {e}")
            return None, None

    def _load_dicom_directory(self, directory: Path) -> Optional[np.ndarray]:
        """
        Load all DICOM files from a directory and stack them into a 3D volume

        Args:
            directory: Path to directory containing DICOM files

        Returns:
            3D numpy array of shape (num_slices, H, W) or None on error
        """
        if not PYDICOM_AVAILABLE:
            print("Error: pydicom is required to load DICOM files")
            return None

        try:
            # Get all DICOM files in the directory (recursively)
            # DICOM files might be nested in subdirectories
            # First try direct files (faster), then recursive if needed
            # IMPORTANT: Filter out known non-DICOM files (LOCKFILE, VERSION, .DS_Store, etc.)
            non_dicom_files = {'LOCKFILE', 'VERSION', 'DICOMDIR', '.DS_Store', 'Thumbs.db'}

            dicom_files = [f for f in directory.iterdir()
                          if f.is_file()
                          and not f.name.startswith('.')
                          and f.name not in non_dicom_files]

            # If no files in root, search recursively
            if not dicom_files:
                dicom_files = [f for f in directory.rglob('*')
                              if f.is_file()
                              and not f.name.startswith('.')
                              and f.name not in non_dicom_files]

            dicom_files = sorted(dicom_files, key=lambda x: x.name)

            if not dicom_files:
                # Debug: Show what files ARE in the directory
                all_files = list(directory.rglob('*'))
                file_list = [f.name for f in all_files if f.is_file()][:10]  # First 10 files
                print(f"No DICOM files found in {directory}")
                print(f"  Directory contains {len([f for f in all_files if f.is_file()])} total files")
                if file_list:
                    print(f"  Sample files: {file_list}")
                return None

            # Load all DICOM slices
            slices = []
            errors_encountered = []
            for dicom_file in dicom_files:
                try:
                    ds = pydicom.dcmread(str(dicom_file))
                    if hasattr(ds, 'pixel_array'):
                        slices.append(ds.pixel_array)
                except Exception as e:
                    # Track errors for debugging (only show first 3)
                    if len(errors_encountered) < 3:
                        errors_encountered.append(f"{dicom_file.name}: {str(e)[:50]}")
                    continue

            if not slices:
                if errors_encountered:
                    print(f"No valid DICOM slices found in {directory}")
                    print(f"  Sample errors: {errors_encountered}")
                    print(f"  Total files tried: {len(dicom_files)}")
                else:
                    print(f"No DICOM files with pixel_array found in {directory}")
                return None

            # Stack into 3D volume
            volume = np.stack(slices, axis=0)
            return volume

        except Exception as e:
            print(f"Error loading DICOM directory {directory}: {e}")
            return None

    def _process_dicom_data(self, data):
        """Process DICOM data (could be files, bytes, or arrays)"""
        # This needs to be adapted based on actual HF dataset format
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, bytes):
            # Process bytes as DICOM
            return self._load_dicom_from_bytes(data)
        elif isinstance(data, (str, Path)):
            # Load from file path
            return self._load_dicom_from_path(data)
        else:
            print(f"Unknown data type: {type(data)}")
            return None

    def _load_dicom_from_bytes(self, data_bytes: bytes) -> Optional[np.ndarray]:
        """Load DICOM from bytes"""
        try:
            import io
            dcm = pydicom.dcmread(io.BytesIO(data_bytes))
            if hasattr(dcm, 'pixel_array'):
                return dcm.pixel_array
        except Exception as e:
            print(f"Error loading DICOM from bytes: {e}")
        return None

    def _load_dicom_from_path(self, path) -> Optional[np.ndarray]:
        """Load DICOM from file path"""
        try:
            dcm = pydicom.dcmread(path)
            if hasattr(dcm, 'pixel_array'):
                return dcm.pixel_array
        except Exception as e:
            print(f"Error loading DICOM from path: {e}")
        return None

    def _apply_ct_windowing(self, volume: np.ndarray) -> np.ndarray:
        """Apply CT windowing to convert HU values to display range"""
        lower = self.window_center - (self.window_width / 2)
        upper = self.window_center + (self.window_width / 2)

        windowed = np.clip(volume, lower, upper)
        windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)

        return windowed

    def _sample_frames(self, volume: np.ndarray) -> np.ndarray:
        """Sample num_frames evenly from the volume"""
        if len(volume.shape) == 2:
            # Single slice, repeat it
            volume = volume[np.newaxis, ...]

        depth = volume.shape[0]

        if depth <= self.num_frames:
            sampled = np.zeros((self.num_frames, *volume.shape[1:]), dtype=volume.dtype)
            sampled[:depth] = volume
            sampled[depth:] = volume[-1]
        else:
            indices = np.linspace(0, depth - 1, self.num_frames, dtype=int)
            sampled = volume[indices]

        return sampled

    def _process_sample(self, sample: dict) -> dict:
        """Process a HuggingFace sample into model input format"""
        # Filter by category if specified
        category = sample.get('category', 'unknown')
        if self.categories and category not in self.categories:
            return None

        # Load DICOM volumes
        baseline_volume, followup_volume = self._load_dicom_from_hf_sample(sample)

        if baseline_volume is None or followup_volume is None:
            return None

        # Apply CT windowing
        if self.normalize:
            baseline_volume = self._apply_ct_windowing(baseline_volume)
            followup_volume = self._apply_ct_windowing(followup_volume)
        else:
            baseline_volume = ((baseline_volume - baseline_volume.min()) /
                             (baseline_volume.max() - baseline_volume.min() + 1e-8) * 255).astype(np.uint8)
            followup_volume = ((followup_volume - followup_volume.min()) /
                             (followup_volume.max() - followup_volume.min() + 1e-8) * 255).astype(np.uint8)

        # Sample frames
        baseline_sampled = self._sample_frames(baseline_volume)
        followup_sampled = self._sample_frames(followup_volume)

        # Convert grayscale to RGB
        baseline_rgb = np.stack([baseline_sampled] * 3, axis=-1)
        followup_rgb = np.stack([followup_sampled] * 3, axis=-1)

        # Apply transforms
        input_tensor = self.transform(baseline_rgb)
        target_tensor = self.transform(followup_rgb)

        return {
            'input': input_tensor,
            'target': target_tensor,
            'category': category,
            'patient_id': sample.get('patient_id', 'unknown')
        }

    def __iter__(self):
        """Iterate over the dataset"""
        import zipfile

        # Track number of samples yielded
        samples_yielded = 0

        # If using old load_dataset approach (won't work for ZIP files)
        if self.hf_dataset is not None:
            for sample in self.hf_dataset:
                if self.max_samples is not None and samples_yielded >= self.max_samples:
                    break
                processed = self._process_sample(sample)
                if processed is not None:
                    yield processed
                    samples_yielded += 1
        # New approach: Download and process ZIP files one by one
        else:
            # Only process the first max_samples ZIP files
            zip_files_to_process = self.zip_files
            if self.max_samples is not None:
                zip_files_to_process = self.zip_files[:self.max_samples]
                print(f"Processing only first {self.max_samples} ZIP files (max_samples limit)")

            # Create persistent cache directory for extracted files
            # Always use HuggingFace cache location to ensure persistence across pod restarts
            if self.cache_dir:
                extract_cache_dir = Path(self.cache_dir) / "extracted_zips"
            else:
                # Use same base as HuggingFace downloads (persistent storage)
                # This ensures extracted files survive pod restarts
                hf_cache_home = os.environ.get('HF_HOME', os.environ.get('HUGGINGFACE_HUB_CACHE', Path.home() / '.cache' / 'huggingface'))
                extract_cache_dir = Path(hf_cache_home) / "extracted_zips"
            extract_cache_dir.mkdir(parents=True, exist_ok=True)

            for zip_file_path in zip_files_to_process:
                # Double-check limit in case processing fails for some files
                if self.max_samples is not None and samples_yielded >= self.max_samples:
                    print(f"Reached max_samples limit ({self.max_samples}), stopping dataset iteration")
                    break
                try:
                    # Create a unique cache directory for this ZIP file
                    zip_hash = Path(zip_file_path).stem  # e.g., "case_001"

                    # Skip corrupted cases immediately (verified to have no DICOM data)
                    if zip_hash in self.corrupted_cases:
                        continue

                    cached_extract_dir = extract_cache_dir / zip_hash

                    # Check if extraction exists FIRST (avoid unnecessary network calls)
                    # Also validate that extraction has actual DICOM files
                    needs_extraction = False
                    if not cached_extract_dir.exists():
                        needs_extraction = True
                    else:
                        # Validate that cached extraction has subdirectories AND actual files
                        subdirs = [d for d in cached_extract_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                        if len(subdirs) == 0:
                            print(f"Warning: Cached extraction {zip_hash} has no subdirectories, re-extracting...")
                            needs_extraction = True
                        else:
                            # Check if there are any files at all (recursively)
                            non_admin_files = {'LOCKFILE', 'VERSION', 'DICOMDIR', '.DS_Store', 'Thumbs.db'}
                            all_files = [f for f in cached_extract_dir.rglob('*')
                                        if f.is_file()
                                        and not f.name.startswith('.')
                                        and f.name not in non_admin_files]
                            if len(all_files) == 0:
                                print(f"Warning: Cached extraction {zip_hash} has no DICOM files, re-extracting...")
                                needs_extraction = True

                    if needs_extraction:
                        # Download and extract
                        print(f"Extracting {zip_file_path}...")
                        local_zip_path = hf_hub_download(
                            repo_id=self.dataset_name,
                            filename=zip_file_path,
                            repo_type="dataset",
                            cache_dir=self.cache_dir
                        )

                        # Clear existing directory if it exists
                        if cached_extract_dir.exists():
                            import shutil
                            shutil.rmtree(cached_extract_dir)

                        cached_extract_dir.mkdir(parents=True, exist_ok=True)

                        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                            zip_ref.extractall(cached_extract_dir)

                        # Delete ZIP file after successful extraction to save disk space
                        try:
                            os.remove(local_zip_path)
                            print(f"  ✓ Deleted ZIP after extraction: {Path(local_zip_path).name}")
                        except Exception as e:
                            print(f"  Warning: Could not delete ZIP file: {e}")
                    # If cached and validated, skip download entirely (no network call!)

                    # Process the extracted case (similar to local APEDataset)
                    case_dir = cached_extract_dir
                    # Extract category from ZIP file path (e.g., "APE/case_001.zip" -> "APE")
                    category = zip_file_path.split('/')[0]
                    # Extract patient ID from filename
                    patient_id = Path(zip_file_path).stem  # e.g., "case_001"
                    sample = {
                        'case_dir': str(case_dir),
                        'category': category,
                        'patient_id': patient_id
                    }
                    processed = self._process_sample(sample)
                    if processed is not None:
                        yield processed
                        samples_yielded += 1

                except Exception as e:
                    print(f"Error processing {zip_file_path}: {e}")
                    continue


def get_ape_hf_dataloader(
    dataset_name: str = 't2ance/APE-data',
    split: str = 'train',
    num_frames: int = 16,
    resolution: Tuple[int, int] = (256, 256),
    batch_size: int = 4,
    num_workers: int = 4,
    categories: Optional[List[str]] = None,
    streaming: bool = True,
    max_samples: Optional[int] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for APE-data from HuggingFace

    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split
        num_frames: Number of frames per video
        resolution: (H, W) resolution
        batch_size: Batch size
        num_workers: Number of worker processes
        categories: List of categories to include
        streaming: Use streaming mode
        **kwargs: Additional arguments for APEHuggingFaceDataset

    Returns:
        DataLoader instance

    Example:
        >>> dataloader = get_ape_hf_dataloader(
        ...     dataset_name='t2ance/APE-data',
        ...     split='train',
        ...     batch_size=4,
        ...     streaming=True
        ... )
        >>> for batch in dataloader:
        ...     inputs = batch['input']  # (B, 3, T, H, W)
        ...     targets = batch['target']  # (B, 3, T, H, W)
    """

    dataset = APEHuggingFaceDataset(
        dataset_name=dataset_name,
        split=split,
        num_frames=num_frames,
        resolution=resolution,
        categories=categories,
        streaming=streaming,
        max_samples=max_samples,
        **kwargs
    )

    # Custom collate function
    def collate_fn(batch):
        inputs = torch.stack([item['input'] for item in batch])
        targets = torch.stack([item['target'] for item in batch])
        categories = [item['category'] for item in batch]
        patient_ids = [item['patient_id'] for item in batch]

        return {
            'input': inputs,
            'target': targets,
            'category': categories,
            'patient_id': patient_ids
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers if not streaming else 0,  # Streaming doesn't work well with multiprocessing
        collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader


def inspect_hf_dataset(dataset_name: str = 't2ance/APE-data', split: str = 'train'):
    """
    Inspect the structure of the HuggingFace APE-data dataset

    This helper function prints the structure of the dataset to help
    adapt the loader to the actual format.

    Usage:
        >>> from data.ape_hf_dataset import inspect_hf_dataset
        >>> inspect_hf_dataset('t2ance/APE-data')
    """
    print("=" * 70)
    print(f"Inspecting HuggingFace Dataset: {dataset_name}")
    print("=" * 70)
    print()

    try:
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
        print("✓ Dataset loaded\n")

        # Get first sample
        print("Fetching first sample...")
        first_sample = next(iter(dataset))
        print("✓ Sample retrieved\n")

        # Print structure
        print("Dataset Structure:")
        print("-" * 70)
        print(f"Keys: {list(first_sample.keys())}\n")

        for key, value in first_sample.items():
            print(f"{key}:")
            print(f"  Type: {type(value)}")

            if hasattr(value, 'shape'):
                print(f"  Shape: {value.shape}")
            elif hasattr(value, '__len__') and not isinstance(value, str):
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  First item type: {type(value[0])}")
            elif isinstance(value, str):
                preview = value[:100] + "..." if len(value) > 100 else value
                print(f"  Value: {preview}")
            elif isinstance(value, dict):
                print(f"  Sub-keys: {list(value.keys())}")
            else:
                print(f"  Value: {value}")
            print()

        print("=" * 70)
        print("\nUse this information to adapt the APEHuggingFaceDataset loader")
        print("if the default implementation doesn't work.")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPossible issues:")
        print("1. Dataset doesn't exist or name is wrong")
        print("2. Need to login: huggingface-cli login")
        print("3. Dataset is private and you don't have access")
        print("4. Network issues")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--inspect':
        # Inspect mode
        inspect_hf_dataset('t2ance/APE-data')
    else:
        # Test mode
        print("Testing HuggingFace APE-data loader...")
        print()
        print("Note: This requires access to the APE-data dataset on HuggingFace")
        print("Run with --inspect to see the dataset structure first")
        print()

        try:
            dataloader = get_ape_hf_dataloader(
                dataset_name='t2ance/APE-data',
                split='train',
                num_frames=8,
                resolution=(128, 128),
                batch_size=1,
                streaming=True
            )

            print("✓ DataLoader created")
            print("\nFetching first batch...")

            batch = next(iter(dataloader))

            print("✓ Batch loaded successfully")
            print(f"\nBatch structure:")
            print(f"  Input shape: {batch['input'].shape}")
            print(f"  Target shape: {batch['target'].shape}")
            print(f"  Categories: {batch['category']}")
            print(f"  Patient IDs: {batch['patient_id']}")

            print("\n✓ HuggingFace loader test successful!")

        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            print("\nRun with --inspect flag to examine dataset structure:")
            print("  python data/ape_hf_dataset.py --inspect")
