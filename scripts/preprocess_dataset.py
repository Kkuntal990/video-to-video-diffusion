#!/usr/bin/env python3
"""
Preprocess Downloaded APE Dataset

This script preprocesses the downloaded APE dataset ZIP files into cached tensors.
It incorporates all learnings from previous preprocessing runs:
- Handles corrupted cases (blacklist)
- Handles wrong folder structures (nested directories)
- Timeout handling (15 min per case)
- Memory management (cleanup after each case)
- Detailed failure reporting

Usage:
    python scripts/preprocess_dataset.py \
        --data_dir /workspace/storage_a100/dataset \
        --cache_dir /workspace/storage_a100/.cache/ape_preprocessed \
        --num_frames 24 \
        --resolution 256 256

    # Resume failed cases only
    python scripts/preprocess_dataset.py --resume
"""

import argparse
import torch
import numpy as np
import os
import shutil
import gc
import signal
import logging
import traceback
import json
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm
import zipfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import dependencies
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logger.error("pydicom not installed. Install with: pip install pydicom")

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.transforms import VideoTransform


class TimeoutException(Exception):
    """Exception raised when operation times out"""
    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal"""
    raise TimeoutException("Operation timed out")


class APEPreprocessor:
    """
    APE Dataset Preprocessor

    Handles all edge cases learned from previous preprocessing:
    - Corrupted cases (no DICOM data)
    - Wrong folder structures
    - Slow processing files (timeout)
    - Memory management
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        num_frames: int = 24,
        resolution: Tuple[int, int] = (256, 256),
        window_center: int = 40,
        window_width: int = 400,
        categories: List[str] = None
    ):
        """
        Args:
            data_dir: Directory with APE/ and non-APE/ folders containing ZIPs
            cache_dir: Directory to store preprocessed tensors
            num_frames: Number of frames per sample
            resolution: (H, W) resolution
            window_center: CT window center (HU)
            window_width: CT window width (HU)
            categories: Categories to process (default: both APE and non-APE)
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom required: pip install pydicom")

        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        self.window_center = window_center
        self.window_width = window_width

        # Categories (note: folder might be "non-APE" but HF uses "non APE")
        if categories is None:
            categories = ['APE', 'non-APE']
        self.categories = categories

        # Blacklist of corrupted cases (verified to have no DICOM data)
        # From previous preprocessing runs
        self.corrupted_cases = {
            'case_190', 'case_191', 'case_192', 'case_193', 'case_194',
            'case_195', 'case_196', 'case_197', 'case_198', 'case_199',
            'case_200', 'case_201', 'case_202', 'case_203', 'case_204',
            'case_205', 'case_206'
        }

        # Create cache directories
        self.extracted_dir = self.cache_dir / 'extracted'
        self.processed_dir = self.cache_dir / 'processed'

        for dir_path in [self.extracted_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Transform for preprocessing
        self.transform = VideoTransform(resolution=resolution, num_frames=num_frames)

        logger.info(f"APE Preprocessor initialized")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info(f"  Resolution: {resolution}, Frames: {num_frames}")
        logger.info(f"  Categories: {', '.join(self.categories)}")

    def find_zip_files(self) -> List[Tuple[Path, str]]:
        """Find all ZIP files in data directory"""
        zip_files = []

        for category in self.categories:
            category_dir = self.data_dir / category

            if not category_dir.exists():
                logger.warning(f"Category directory not found: {category_dir}")
                continue

            # Find all ZIPs
            category_zips = list(category_dir.glob('*.zip'))

            # Filter out corrupted cases
            valid_zips = []
            corrupted_count = 0

            for zip_path in category_zips:
                case_id = zip_path.stem
                if case_id not in self.corrupted_cases:
                    valid_zips.append((zip_path, category))
                else:
                    corrupted_count += 1

            logger.info(f"Found {len(valid_zips)} valid ZIPs in {category}/ (skipped {corrupted_count} corrupted)")
            zip_files.extend(valid_zips)

        return zip_files

    def preprocess_all(self, resume: bool = False):
        """Preprocess all ZIP files"""

        # Find all ZIP files
        logger.info("\n" + "="*70)
        logger.info("Finding ZIP files...")
        logger.info("="*70)

        zip_files = self.find_zip_files()

        if len(zip_files) == 0:
            logger.error("No ZIP files found!")
            return

        logger.info(f"Found {len(zip_files)} valid ZIP files to process")

        # Load metadata
        metadata_file = self.cache_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Filter to only unprocessed cases (if resume mode)
        if resume:
            to_process = []
            for zip_path, category in zip_files:
                case_id = zip_path.stem
                processed_file = self.processed_dir / f"{case_id}.pt"
                if not processed_file.exists():
                    to_process.append((zip_path, category))

            logger.info(f"Resume mode: {len(to_process)} cases remaining (already processed: {len(zip_files) - len(to_process)})")
        else:
            to_process = zip_files

        if len(to_process) == 0:
            logger.info("All cases already preprocessed!")
            return

        # Preprocess all cases
        logger.info("\n" + "="*70)
        logger.info(f"Preprocessing {len(to_process)} cases...")
        logger.info("="*70)

        successful = 0
        failed = []

        for zip_path, category in tqdm(to_process, desc="Preprocess"):
            case_id = zip_path.stem

            try:
                logger.info(f"\nProcessing {case_id} (category: {category})...")

                # Extract ZIP
                extracted_case_dir = self.extracted_dir / case_id
                if extracted_case_dir.exists():
                    shutil.rmtree(extracted_case_dir)
                extracted_case_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_case_dir)

                # Set timeout (15 minutes max per case)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(900)  # 900 seconds = 15 minutes

                try:
                    # Load and preprocess DICOM data
                    sample_dict = self._load_and_preprocess_case(
                        extracted_case_dir,
                        category,
                        case_id
                    )
                finally:
                    # Cancel timeout
                    signal.alarm(0)

                if sample_dict is not None:
                    # Save preprocessed tensors
                    processed_file = self.processed_dir / f"{case_id}.pt"
                    torch.save(sample_dict, processed_file)
                    successful += 1

                    # Save metadata
                    # Map "non-APE" folder to "non APE" category (HF format)
                    category_name = "non APE" if category == "non-APE" else category
                    metadata[case_id] = category_name
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    logger.info(f"  ✓ {case_id}: Successfully preprocessed")

                    # Delete extracted files to save space
                    if extracted_case_dir.exists():
                        shutil.rmtree(extracted_case_dir)

                    # Force memory cleanup
                    del sample_dict
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    logger.error(f"  ✗ {case_id}: No valid data returned")
                    failed.append((case_id, "No valid data returned"))

            except TimeoutException:
                error_msg = "Timeout (>15 min)"
                logger.error(f"  ✗ {case_id}: {error_msg}")
                failed.append((case_id, error_msg))
                # Cleanup
                if extracted_case_dir.exists():
                    shutil.rmtree(extracted_case_dir)
                gc.collect()
                continue

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"  ✗ {case_id}: {error_msg}")
                logger.debug(f"  Traceback: {traceback.format_exc()}")
                failed.append((case_id, error_msg))
                # Cleanup
                if extracted_case_dir.exists():
                    shutil.rmtree(extracted_case_dir)
                gc.collect()
                continue

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("Preprocessing Complete!")
        logger.info("="*70)
        logger.info(f"✓ Successfully preprocessed: {successful} cases")

        if len(failed) > 0:
            logger.info(f"✗ Failed to preprocess: {len(failed)} cases")

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

            logger.info(f"  Detailed report saved: {failure_report_path}")

            # Error breakdown
            error_types = {}
            for case_id, error in failed:
                error_type = error.split(':')[0] if ':' in error else error
                error_types[error_type] = error_types.get(error_type, 0) + 1

            logger.info("\nError breakdown:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {error_type}: {count} cases")

        # Final cleanup
        extracted_dirs = list(self.extracted_dir.glob('*'))
        if len(extracted_dirs) > 0:
            logger.info(f"\nCleaning up {len(extracted_dirs)} remaining extracted directories...")
            for extracted_dir in extracted_dirs:
                if extracted_dir.is_dir():
                    shutil.rmtree(extracted_dir)

        logger.info(f"\nPreprocessed data saved to: {self.processed_dir}")
        logger.info(f"Total preprocessed files: {len(list(self.processed_dir.glob('*.pt')))}")

    def _load_and_preprocess_case(self, case_dir: Path, category: str, case_id: str) -> Optional[dict]:
        """Load DICOM files and preprocess one case"""

        # Find subdirectories with DICOM files
        # Filter out hidden directories
        subdirs = [d for d in case_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        logger.debug(f"  Found {len(subdirs)} subdirectories")

        if len(subdirs) == 0:
            logger.warning(f"  No subdirectories found in {case_id}")
            return None
        elif len(subdirs) == 1:
            # Handle nested structure (common in APE dataset)
            # Some ZIPs have: case_dir/patient_folder/1/ and case_dir/patient_folder/2/
            inner_subdirs = [d for d in subdirs[0].iterdir() if d.is_dir() and not d.name.startswith('.')]
            logger.debug(f"  Found {len(inner_subdirs)} inner subdirectories")
            if len(inner_subdirs) >= 2:
                # Sort numerically if possible, else alphabetically
                try:
                    subdirs = sorted(inner_subdirs, key=lambda x: int(x.name))
                except ValueError:
                    subdirs = sorted(inner_subdirs, key=lambda x: x.name)
            else:
                logger.warning(f"  Only 1 subdirectory with insufficient inner dirs in {case_id}")
                return None

        if len(subdirs) < 2:
            logger.warning(f"  Less than 2 subdirectories ({len(subdirs)}) in {case_id}")
            return None

        # Sort subdirectories (baseline=first, followup=second)
        subdirs = sorted(subdirs, key=lambda x: x.name)
        logger.debug(f"  Loading baseline from: {subdirs[0].name}")
        logger.debug(f"  Loading followup from: {subdirs[1].name}")

        # Load DICOM volumes
        baseline_volume = self._load_dicom_directory(subdirs[0])
        if baseline_volume is None:
            logger.error(f"  Failed to load baseline volume")
            return None
        logger.debug(f"  Baseline shape: {baseline_volume.shape}")

        followup_volume = self._load_dicom_directory(subdirs[1])
        if followup_volume is None:
            logger.error(f"  Failed to load followup volume")
            return None
        logger.debug(f"  Followup shape: {followup_volume.shape}")

        # Apply CT windowing (in-place to save memory)
        baseline_windowed = self._apply_ct_windowing(baseline_volume)
        del baseline_volume
        followup_windowed = self._apply_ct_windowing(followup_volume)
        del followup_volume

        # Sample frames
        baseline_sampled = self._sample_frames(baseline_windowed)
        del baseline_windowed
        followup_sampled = self._sample_frames(followup_windowed)
        del followup_windowed

        # Convert to RGB (stack grayscale 3 times)
        baseline_rgb = np.stack([baseline_sampled] * 3, axis=-1)
        del baseline_sampled
        followup_rgb = np.stack([followup_sampled] * 3, axis=-1)
        del followup_sampled

        # Apply transforms (resize, normalize, to tensor)
        input_tensor = self.transform(baseline_rgb)
        del baseline_rgb
        target_tensor = self.transform(followup_rgb)
        del followup_rgb

        return {
            'input': input_tensor,
            'target': target_tensor,
            'category': category,
            'patient_id': case_id
        }

    def _load_dicom_directory(self, directory: Path) -> Optional[np.ndarray]:
        """Load all DICOM files from a directory"""
        # Known non-DICOM files to skip (from previous experience)
        non_dicom_files = {'LOCKFILE', 'VERSION', 'DICOMDIR', '.DS_Store', 'Thumbs.db'}

        # Find all files recursively, excluding non-DICOM files
        dicom_files = [f for f in directory.rglob('*')
                      if f.is_file()
                      and not f.name.startswith('.')
                      and f.name not in non_dicom_files]

        if not dicom_files:
            logger.warning(f"  No DICOM files found in {directory.name}")
            return None

        dicom_files = sorted(dicom_files, key=lambda x: x.name)
        logger.debug(f"  Found {len(dicom_files)} potential DICOM files")

        slices = []
        failed_reads = 0

        for dicom_file in dicom_files:
            try:
                ds = pydicom.dcmread(str(dicom_file))
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds.pixel_array)
                else:
                    failed_reads += 1
            except Exception:
                failed_reads += 1
                continue

        logger.debug(f"  Loaded {len(slices)} slices ({failed_reads} failed)")

        if not slices:
            return None

        volume = np.stack(slices, axis=0)
        return volume

    def _apply_ct_windowing(self, volume: np.ndarray) -> np.ndarray:
        """Apply CT windowing to convert HU to display range"""
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
            # Pad with last frame if not enough
            sampled = np.zeros((self.num_frames, *volume.shape[1:]), dtype=volume.dtype)
            sampled[:depth] = volume
            sampled[depth:] = volume[-1]
        else:
            # Evenly sample frames
            indices = np.linspace(0, depth - 1, self.num_frames, dtype=int)
            sampled = volume[indices]

        return sampled


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess APE Dataset from Downloaded ZIPs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess all data
  python scripts/preprocess_dataset.py \\
      --data_dir /workspace/storage_a100/dataset \\
      --cache_dir /workspace/storage_a100/.cache/ape_preprocessed

  # Resume failed cases only
  python scripts/preprocess_dataset.py --resume

  # Custom resolution and frames
  python scripts/preprocess_dataset.py \\
      --num_frames 16 \\
      --resolution 128 128
        """
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='/workspace/storage_a100/dataset',
        help='Directory containing APE/ and non-APE/ folders with ZIPs'
    )

    parser.add_argument(
        '--cache_dir',
        type=str,
        default='/workspace/storage_a100/.cache/ape_preprocessed',
        help='Directory to store preprocessed tensors'
    )

    parser.add_argument(
        '--num_frames',
        type=int,
        default=24,
        help='Number of frames per sample (default: 24)'
    )

    parser.add_argument(
        '--resolution',
        type=int,
        nargs=2,
        default=[256, 256],
        help='Resolution (H W) (default: 256 256)'
    )

    parser.add_argument(
        '--window_center',
        type=int,
        default=40,
        help='CT window center in HU (default: 40)'
    )

    parser.add_argument(
        '--window_width',
        type=int,
        default=400,
        help='CT window width in HU (default: 400)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume preprocessing (only process unfinished cases)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create preprocessor
    preprocessor = APEPreprocessor(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        num_frames=args.num_frames,
        resolution=tuple(args.resolution),
        window_center=args.window_center,
        window_width=args.window_width
    )

    # Run preprocessing
    preprocessor.preprocess_all(resume=args.resume)


if __name__ == "__main__":
    main()
