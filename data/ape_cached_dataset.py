"""
APE-Data Cached Dataset (Minimal Storage Version)

This dataset:
1. Downloads and extracts ZIPs ONCE to persistent storage
2. Preprocesses all DICOM data ONCE and caches as .pt files
3. DELETES ZIPs and extracted DICOM files after preprocessing (saves ~40-50GB!)
4. Loads preprocessed tensors directly during training (very fast)

Storage Requirements:
- During preprocessing: ~20-30GB (temporary)
- After preprocessing: ~15-20GB (only .pt tensors)

Usage:
    dataset = APECachedDataset(
        dataset_name='t2ance/APE-data',
        cache_dir='/workspace/storage/ape_cache',
        num_frames=24,
        resolution=(256, 256)
    )

    # First run: Downloads, extracts, preprocesses, deletes intermediates (slow, one-time only)
    # Subsequent runs: Loads cached tensors (very fast)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import shutil
import gc
import signal
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Exception raised when operation times out"""
    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal"""
    raise TimeoutException("Operation timed out")

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from .transforms import VideoTransform


class APECachedDataset(Dataset):
    """
    APE-Data with preprocessing caching (Minimal Storage)

    Directory structure:
    cache_dir/
    ├── raw/              # Downloaded ZIPs (deleted after extraction)
    ├── extracted/        # Extracted DICOM files (deleted after preprocessing)
    └── processed/        # Preprocessed tensors (.pt files) - KEPT
        ├── case_001.pt
        ├── case_002.pt
        └── ...

    Note: Only the processed/ directory remains after initial setup (~15-20GB).
    ZIPs and extracted files are automatically deleted to save space.
    """

    def __init__(
        self,
        dataset_name: str = 't2ance/APE-data',
        cache_dir: str = '/workspace/storage/ape_cache',
        num_frames: int = 24,
        resolution: Tuple[int, int] = (256, 256),
        categories: Optional[List[str]] = None,
        window_center: int = 40,
        window_width: int = 400,
        force_reprocess: bool = False,
        split: str = 'train',
        val_split: float = 0.15,
        test_split: float = 0.10,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            dataset_name: HuggingFace dataset identifier
            cache_dir: Directory for caching (should be persistent storage!)
            num_frames: Number of frames per sample
            resolution: (H, W) resolution
            categories: List of categories to include
            window_center: CT window center (HU)
            window_width: CT window width (HU)
            force_reprocess: If True, reprocess all data even if cached
            split: 'train', 'val', or 'test'
            val_split: Fraction for validation (default 0.15 = 15%)
            test_split: Fraction for test (default 0.10 = 10%)
            seed: Random seed for reproducible splits
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom required: pip install pydicom")
        if not HF_HUB_AVAILABLE:
            raise ImportError("huggingface_hub required: pip install huggingface_hub")

        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        # Note: HuggingFace folder is "non APE" (with space), not "non-APE"
        self.categories = categories if categories else ['APE', 'non APE']
        self.window_center = window_center
        self.window_width = window_width
        self.force_reprocess = force_reprocess
        self.split = split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        # Blacklist of corrupted cases (verified to have no DICOM data)
        self.corrupted_cases = {
            'case_190', 'case_191', 'case_192', 'case_193', 'case_194',
            'case_195', 'case_196', 'case_197', 'case_198', 'case_199',
            'case_200', 'case_201', 'case_202', 'case_203', 'case_204',
            'case_205', 'case_206'
        }

        # Create cache directories
        self.raw_dir = self.cache_dir / 'raw'
        self.extracted_dir = self.cache_dir / 'extracted'
        self.processed_dir = self.cache_dir / 'processed'

        for dir_path in [self.raw_dir, self.extracted_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Transform for final preprocessing
        self.transform = VideoTransform(resolution=resolution, num_frames=num_frames)

        # Initialize dataset
        print(f"\n{'='*70}")
        print(f"APE Cached Dataset Initialization")
        print(f"{'='*70}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Resolution: {resolution}, Frames: {num_frames}")
        print(f"Categories: {', '.join(self.categories)}")

        self._prepare_dataset()

    def _prepare_dataset(self):
        """Download, extract, and preprocess dataset (only if needed)"""

        # Step 1: Get list of ZIP files from HuggingFace
        print(f"\n[Step 1/4] Fetching file list from HuggingFace...")
        all_files = list(list_repo_files(self.dataset_name, repo_type="dataset"))

        # Collect files by category
        zip_files = []
        category_counts = {}
        for category in self.categories:
            category_files = [f for f in all_files if f.startswith(f"{category}/") and f.endswith(".zip")]
            category_counts[category] = len(category_files)
            zip_files.extend(category_files)
            print(f"  Found {len(category_files)} cases in '{category}' category")

        # Filter out corrupted cases
        valid_zip_files = []
        corrupted_count = 0
        valid_by_category = {cat: 0 for cat in self.categories}

        for zip_file in zip_files:
            case_id = Path(zip_file).stem
            category = zip_file.split('/')[0]
            if case_id not in self.corrupted_cases:
                valid_zip_files.append(zip_file)
                if category in valid_by_category:
                    valid_by_category[category] += 1
            else:
                corrupted_count += 1

        zip_files = valid_zip_files
        print(f"\n✓ Found {len(zip_files)} valid cases total")
        for category, count in valid_by_category.items():
            print(f"  ├─ {category}: {count} valid cases")
        if corrupted_count > 0:
            print(f"  └─ Skipped {corrupted_count} corrupted cases")

        # Step 2: Check if we need to download/extract/preprocess
        # If all cases are already preprocessed, skip download/extract entirely
        all_preprocessed = all(
            (self.processed_dir / f"{Path(zf).stem}.pt").exists()
            for zf in zip_files
        )

        if all_preprocessed and not self.force_reprocess:
            print(f"\n[Step 2-3/4] ✓ All {len(zip_files)} cases already preprocessed, skipping download/extract/preprocess")
        else:
            # Step 2: Download and extract ZIPs
            print(f"\n[Step 2/4] Downloading and extracting DICOM files...")
            self._download_and_extract(zip_files)

            # Step 3: Preprocess all cases
            print(f"\n[Step 3/4] Preprocessing DICOM data...")
            self._preprocess_all_cases(zip_files)

        # Step 4: Load metadata with train/val/test split
        print(f"\n[Step 4/4] Loading metadata and creating {self.split} split...")
        self._load_metadata(
            split=self.split,
            val_split=self.val_split,
            test_split=self.test_split,
            seed=self.seed
        )

        print(f"\n{'='*70}")
        print(f"✓ Dataset ready! {len(self.samples)} samples in {self.split} split")
        print(f"{'='*70}")

        # Log final summary
        logger.info(f"Dataset initialization complete: {len(self.samples)} samples ready")
        logger.info(f"Cache directory: {self.cache_dir}")

        # Check if there's a failure report
        failure_report_path = self.cache_dir / 'preprocessing_failures.txt'
        if failure_report_path.exists():
            print(f"\n⚠️  Some cases failed during preprocessing.")
            print(f"   See detailed report: {failure_report_path}")
        print()

    def _download_and_extract(self, zip_files: List[str]):
        """Download and extract all ZIP files"""
        import zipfile

        to_download = []
        for zip_file in zip_files:
            case_id = Path(zip_file).stem
            extracted_case_dir = self.extracted_dir / case_id
            processed_file = self.processed_dir / f"{case_id}.pt"

            # Skip if already preprocessed (extracted dirs are deleted after preprocessing)
            if processed_file.exists() and not self.force_reprocess:
                continue  # Already preprocessed, no need to download/extract

            # Check if already extracted but not yet preprocessed
            if extracted_case_dir.exists() and not self.force_reprocess:
                # Verify extraction has files
                subdirs = list(extracted_case_dir.glob('*/'))
                if len(subdirs) > 0:
                    continue  # Already extracted and valid

            to_download.append(zip_file)

        if len(to_download) == 0:
            print(f"✓ All {len(zip_files)} cases already extracted, skipping downloads")
            return

        print(f"Downloading and extracting {len(to_download)} cases...")
        for zip_file in tqdm(to_download, desc="Download & Extract"):
            case_id = Path(zip_file).stem
            extracted_case_dir = self.extracted_dir / case_id

            try:
                # Download ZIP
                local_zip_path = hf_hub_download(
                    repo_id=self.dataset_name,
                    filename=zip_file,
                    repo_type="dataset",
                    cache_dir=str(self.raw_dir)
                )

                # Extract
                if extracted_case_dir.exists():
                    shutil.rmtree(extracted_case_dir)
                extracted_case_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_case_dir)

                # Delete ZIP to save space
                os.remove(local_zip_path)

            except Exception as e:
                print(f"\n  ✗ Error with {case_id}: {e}")
                continue

        print(f"✓ Downloaded and extracted {len(to_download)} cases")

    def _preprocess_all_cases(self, zip_files: List[str]):
        """Preprocess all cases and save as .pt files"""
        import json

        # Load or create metadata cache
        metadata_file = self.cache_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        to_process = []
        for zip_file in zip_files:
            case_id = Path(zip_file).stem
            processed_file = self.processed_dir / f"{case_id}.pt"

            # Check if already processed
            if processed_file.exists() and not self.force_reprocess:
                # Ensure metadata exists for already-processed files
                if case_id not in metadata:
                    category = zip_file.split('/')[0]
                    metadata[case_id] = category
                continue

            to_process.append((zip_file, case_id))

        # Save metadata for already-processed cases
        if metadata:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        if len(to_process) == 0:
            print(f"✓ All {len(zip_files)} cases already preprocessed, skipping")
            return

        print(f"Preprocessing {len(to_process)} cases (this may take a while)...")
        successful = 0
        failed = []

        for zip_file, case_id in tqdm(to_process, desc="Preprocess"):
            try:
                # Get category from ZIP path
                category = zip_file.split('/')[0]
                extracted_case_dir = self.extracted_dir / case_id

                logger.info(f"Processing {case_id} (category: {category})...")

                # Set timeout (15 minutes max per case - increased from 5 min)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(900)  # 900 seconds = 15 minutes timeout

                try:
                    # Load and preprocess DICOM data
                    logger.debug(f"  - Loading DICOM data from {extracted_case_dir}")
                    sample_dict = self._load_and_preprocess_case(extracted_case_dir, category, case_id)
                finally:
                    # Cancel timeout
                    signal.alarm(0)

                if sample_dict is not None:
                    # Save preprocessed tensors
                    processed_file = self.processed_dir / f"{case_id}.pt"
                    torch.save(sample_dict, processed_file)
                    successful += 1

                    # Save metadata (category mapping)
                    metadata[case_id] = category
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    logger.info(f"  ✓ {case_id}: Successfully preprocessed and saved")

                    # Delete extracted DICOM files to save space (Option 2: Minimal Cache)
                    if extracted_case_dir.exists():
                        shutil.rmtree(extracted_case_dir)

                    # Force memory cleanup to prevent OOM
                    del sample_dict
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    logger.error(f"  ✗ {case_id}: Preprocessing returned None (no valid data)")
                    failed.append((case_id, "No valid data returned"))

            except TimeoutException:
                error_msg = f"Timeout (>15 min)"
                logger.error(f"  ✗ {case_id}: {error_msg}")
                print(f"\n  ✗ Timeout preprocessing {case_id} (>15 min) - skipping")
                failed.append((case_id, error_msg))
                # Clean up memory and extracted files
                if extracted_case_dir.exists():
                    shutil.rmtree(extracted_case_dir)
                gc.collect()
                continue
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"  ✗ {case_id}: {error_msg}")
                logger.debug(f"  Traceback: {traceback.format_exc()}")
                print(f"\n  ✗ Error preprocessing {case_id}: {e}")
                failed.append((case_id, error_msg))
                # Clean up memory even on error
                gc.collect()
                continue

        print(f"✓ Preprocessed {successful} cases successfully")
        if len(failed) > 0:
            print(f"  ⚠ Failed to preprocess {len(failed)} cases")

            # Save detailed failure report
            failure_report_path = self.cache_dir / 'preprocessing_failures.txt'
            with open(failure_report_path, 'w') as f:
                f.write(f"Preprocessing Failure Report\n")
                f.write(f"{'='*70}\n")
                f.write(f"Total failed: {len(failed)}\n")
                f.write(f"Total successful: {successful}\n")
                f.write(f"Success rate: {successful/(successful+len(failed))*100:.1f}%\n\n")
                f.write(f"Failed cases:\n")
                f.write(f"{'-'*70}\n")
                for case_id, error in failed:
                    f.write(f"{case_id}: {error}\n")

            logger.info(f"Detailed failure report saved to: {failure_report_path}")
            print(f"  Detailed failure report saved to: {failure_report_path}")

            # Group errors by type
            error_types = {}
            for case_id, error in failed:
                error_type = error.split(':')[0] if ':' in error else error
                error_types[error_type] = error_types.get(error_type, 0) + 1

            print(f"\n  Error breakdown:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {error_type}: {count} cases")

        # Final cleanup: Remove any remaining extracted directories to save space
        extracted_dirs = list(self.extracted_dir.glob('*'))
        if len(extracted_dirs) > 0:
            print(f"Cleaning up {len(extracted_dirs)} remaining extracted directories...")
            for extracted_dir in extracted_dirs:
                if extracted_dir.is_dir():
                    shutil.rmtree(extracted_dir)
            print(f"✓ Cleaned up extracted directories, saved ~{len(extracted_dirs) * 0.2:.1f}GB")

    def _load_and_preprocess_case(self, case_dir: Path, category: str, case_id: str) -> Optional[dict]:
        """Load DICOM files and apply all preprocessing for one case"""

        # Find subdirectories with DICOM files
        subdirs = [d for d in case_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        logger.debug(f"    Found {len(subdirs)} subdirectories in {case_dir.name}")

        if len(subdirs) == 0:
            logger.warning(f"    No subdirectories found in {case_id}")
            return None
        elif len(subdirs) == 1:
            # Check for nested structure
            inner_subdirs = [d for d in subdirs[0].iterdir() if d.is_dir() and not d.name.startswith('.')]
            logger.debug(f"    Found {len(inner_subdirs)} inner subdirectories")
            if len(inner_subdirs) >= 2:
                subdirs = sorted(inner_subdirs, key=lambda x: (x.name.isdigit(), x.name))
            else:
                logger.warning(f"    Only 1 subdirectory with insufficient inner dirs in {case_id}")
                return None

        if len(subdirs) < 2:
            logger.warning(f"    Less than 2 subdirectories ({len(subdirs)}) in {case_id}")
            return None

        subdirs = sorted(subdirs, key=lambda x: x.name)
        logger.debug(f"    Loading baseline from: {subdirs[0].name}")
        logger.debug(f"    Loading follow-up from: {subdirs[1].name}")

        # Load DICOM volumes
        baseline_volume = self._load_dicom_directory(subdirs[0])
        if baseline_volume is None:
            logger.error(f"    Failed to load baseline volume from {subdirs[0].name}")
            return None
        logger.debug(f"    Baseline volume shape: {baseline_volume.shape}")

        followup_volume = self._load_dicom_directory(subdirs[1])
        if followup_volume is None:
            logger.error(f"    Failed to load follow-up volume from {subdirs[1].name}")
            return None
        logger.debug(f"    Follow-up volume shape: {followup_volume.shape}")

        # Apply CT windowing (in-place modification to save memory)
        baseline_windowed = self._apply_ct_windowing(baseline_volume)
        del baseline_volume  # Free original volume
        followup_windowed = self._apply_ct_windowing(followup_volume)
        del followup_volume  # Free original volume

        # Sample frames
        baseline_sampled = self._sample_frames(baseline_windowed)
        del baseline_windowed  # Free windowed volume
        followup_sampled = self._sample_frames(followup_windowed)
        del followup_windowed  # Free windowed volume

        # Convert to RGB
        baseline_rgb = np.stack([baseline_sampled] * 3, axis=-1)
        del baseline_sampled  # Free grayscale
        followup_rgb = np.stack([followup_sampled] * 3, axis=-1)
        del followup_sampled  # Free grayscale

        # Apply transforms (resize, normalize, to tensor)
        input_tensor = self.transform(baseline_rgb)
        del baseline_rgb  # Free RGB array
        target_tensor = self.transform(followup_rgb)
        del followup_rgb  # Free RGB array

        return {
            'input': input_tensor,
            'target': target_tensor,
            'category': category,
            'patient_id': case_id
        }

    def _load_dicom_directory(self, directory: Path) -> Optional[np.ndarray]:
        """Load all DICOM files from a directory"""
        non_dicom_files = {'LOCKFILE', 'VERSION', 'DICOMDIR', '.DS_Store', 'Thumbs.db'}

        dicom_files = [f for f in directory.rglob('*')
                      if f.is_file()
                      and not f.name.startswith('.')
                      and f.name not in non_dicom_files]

        logger.debug(f"      Found {len(dicom_files)} potential DICOM files in {directory.name}")

        if not dicom_files:
            logger.warning(f"      No DICOM files found in {directory.name}")
            return None

        dicom_files = sorted(dicom_files, key=lambda x: x.name)

        slices = []
        failed_reads = 0
        for dicom_file in dicom_files:
            try:
                ds = pydicom.dcmread(str(dicom_file))
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds.pixel_array)
                else:
                    failed_reads += 1
            except Exception as e:
                failed_reads += 1
                logger.debug(f"      Failed to read {dicom_file.name}: {e}")
                continue

        logger.debug(f"      Successfully loaded {len(slices)} slices, {failed_reads} failed")

        if not slices:
            return None

        volume = np.stack(slices, axis=0)
        return volume

    def _apply_ct_windowing(self, volume: np.ndarray) -> np.ndarray:
        """Apply CT windowing"""
        lower = self.window_center - (self.window_width / 2)
        upper = self.window_center + (self.window_width / 2)

        windowed = np.clip(volume, lower, upper)
        windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)

        return windowed

    def _sample_frames(self, volume: np.ndarray) -> np.ndarray:
        """Sample num_frames evenly from volume"""
        if len(volume.shape) == 2:
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

    def _load_metadata(self, split: str = 'train', val_split: float = 0.15, test_split: float = 0.10, seed: int = 42):
        """
        Load list of available preprocessed samples with train/val/test split

        Args:
            split: 'train', 'val', or 'test'
            val_split: Fraction for validation (default 0.15 = 15%)
            test_split: Fraction for test (default 0.10 = 10%)
            seed: Random seed for reproducible splits
        """
        import random
        import json

        metadata_file = self.cache_dir / 'metadata.json'

        # Load metadata from cache (fast!)
        if metadata_file.exists():
            logger.info("Loading metadata from cache...")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            # Fallback: Generate metadata from existing .pt files (slow, first time only)
            logger.warning("Metadata cache not found. Generating from .pt files (this may take a while)...")
            metadata = {}
            for processed_file in sorted(self.processed_dir.glob('*.pt')):
                try:
                    sample = torch.load(processed_file, weights_only=False)
                    category = sample.get('category', 'unknown')
                    metadata[processed_file.stem] = category
                except Exception as e:
                    logger.warning(f"Failed to load {processed_file} for metadata: {e}")
                    continue

            # Save generated metadata for next time
            if metadata:
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Saved metadata cache to {metadata_file}")

        if len(metadata) == 0:
            raise RuntimeError("No preprocessed samples found! Check preprocessing step.")

        # Group samples by category (using sorted keys for deterministic order)
        samples_by_category = {}
        for case_id in sorted(metadata.keys()):  # SORTED for deterministic splits!
            category = metadata[case_id]
            processed_file = self.processed_dir / f"{case_id}.pt"

            # Verify file exists
            if not processed_file.exists():
                logger.warning(f"Metadata references {case_id} but file not found, skipping")
                continue

            if category not in samples_by_category:
                samples_by_category[category] = []
            samples_by_category[category].append(processed_file)

        if len(samples_by_category) == 0:
            raise RuntimeError("No valid samples found after metadata loading!")

        # Perform stratified split (balanced by category)
        self.samples = []
        train_count = 0
        val_count = 0
        test_count = 0

        random.seed(seed)

        logger.info(f"Creating {split} split with {val_split*100:.0f}% val, {test_split*100:.0f}% test")

        # Iterate categories in sorted order for deterministic splits across runs
        for category in sorted(samples_by_category.keys()):
            category_samples = samples_by_category[category]
            n_samples = len(category_samples)
            n_val = int(n_samples * val_split)
            n_test = int(n_samples * test_split)
            n_train = n_samples - n_val - n_test

            # Shuffle samples for this category
            shuffled = category_samples.copy()
            random.shuffle(shuffled)

            # Split
            train_samples = shuffled[:n_train]
            val_samples = shuffled[n_train:n_train + n_val]
            test_samples = shuffled[n_train + n_val:]

            logger.info(f"  {category}: {n_train} train, {n_val} val, {n_test} test (total: {n_samples})")

            # Add appropriate samples based on requested split
            if split == 'train':
                self.samples.extend(train_samples)
                train_count += len(train_samples)
            elif split == 'val':
                self.samples.extend(val_samples)
                val_count += len(val_samples)
            elif split == 'test':
                self.samples.extend(test_samples)
                test_count += len(test_samples)
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        # Log final counts
        if split == 'train':
            logger.info(f"Training set: {train_count} samples")
        elif split == 'val':
            logger.info(f"Validation set: {val_count} samples")
        elif split == 'test':
            logger.info(f"Test set: {test_count} samples")

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples in {split} split! Check split ratios.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load preprocessed tensor directly (very fast!)"""
        sample_path = self.samples[idx]

        try:
            # Load preprocessed tensors directly from disk
            # weights_only=False is safe here because we control the data source
            sample_dict = torch.load(sample_path, weights_only=False)
            return sample_dict
        except Exception as e:
            print(f"Error loading {sample_path}: {e}")
            # Return next sample on error
            return self.__getitem__((idx + 1) % len(self.samples))


def get_ape_cached_dataloader(
    dataset_name: str = 't2ance/APE-data',
    cache_dir: str = '/workspace/storage/ape_cache',
    num_frames: int = 24,
    resolution: Tuple[int, int] = (256, 256),
    batch_size: int = 1,
    num_workers: int = 4,
    categories: Optional[List[str]] = None,
    force_reprocess: bool = False,
    split: str = 'train',
    val_split: float = 0.15,
    test_split: float = 0.10,
    seed: int = 42,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader with cached preprocessing

    Args:
        dataset_name: HuggingFace dataset identifier
        cache_dir: Cache directory (should be persistent storage!)
        num_frames: Frames per sample
        resolution: (H, W) resolution
        batch_size: Batch size
        num_workers: Number of workers
        categories: Categories to include
        force_reprocess: Reprocess even if cached
        split: 'train', 'val', or 'test'
        val_split: Fraction for validation (default 0.15 = 15%)
        test_split: Fraction for test (default 0.10 = 10%)
        seed: Random seed for reproducible splits
    """

    dataset = APECachedDataset(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        num_frames=num_frames,
        resolution=resolution,
        categories=categories,
        force_reprocess=force_reprocess,
        split=split,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        **kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle since we're using regular Dataset now
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader
