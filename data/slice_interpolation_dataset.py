"""
CT Slice Interpolation Dataset - Full Volume Version

This dataset correctly handles the APE data structure for slice interpolation:
- patient/1/ contains THICK slices (~50 @ 5.0mm spacing)
- patient/2/ contains THIN slices (~300-400 @ 1.0mm spacing)

Task: Anisotropic super-resolution in depth dimension
Input: 50 thick slices → Output: 300 thin slices

IMPORTANT: Loads FULL volumes without patching for maximum quality.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import zipfile
import tempfile
import shutil
import random
import logging
import warnings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress pydicom warnings about character encodings (GB18030, etc.)
# These are harmless warnings from DICOM files with Chinese text
warnings.filterwarnings('ignore', message='.*cannot be used as code extension.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pydicom')

# IMPORTANT: Suppress pydicom logger warnings (GB18030 encoding)
logging.getLogger('pydicom').setLevel(logging.ERROR)
logging.getLogger('pydicom.charset').setLevel(logging.ERROR)

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not installed. Install with: pip install pydicom")


class SliceInterpolationDataset(Dataset):
    """
    CT Slice Interpolation Dataset for Anisotropic Super-Resolution

    Loads FULL-VOLUME CT scans without patching:
    - Input: Thick slices (patient/1/) - typically 50 slices @ 5.0mm
    - Target: Thin slices (patient/2/) - typically 300-400 slices @ 1.0mm

    Returns variable-depth volumes for training diffusion models.

    Args:
        data_dir: Root directory containing APE/ and non-APE/ folders with .zip files
        resolution: (H, W) to resize each slice (default: 512x512)
        categories: List of categories to include ['APE', 'non-APE']
        extract_dir: Directory to extract zip files (None = temp dir)
        cache_extracted: Keep extracted files for faster loading
        window_center: CT window center in HU (40 for soft tissue)
        window_width: CT window width in HU (400 for soft tissue+vessels)
        max_thick_slices: Maximum number of thick slices to keep (None = all)
        max_thin_slices: Maximum number of thin slices to keep (None = all)
        split: 'train', 'val', or 'test'
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        seed: Random seed for reproducible splits
    """

    def __init__(
        self,
        data_dir: str,
        resolution: Tuple[int, int] = (512, 512),
        categories: List[str] = ['APE', 'non-APE'],
        extract_dir: Optional[str] = None,
        cache_extracted: bool = True,
        window_center: float = 40,
        window_width: float = 400,
        max_thick_slices: Optional[int] = None,
        max_thin_slices: Optional[int] = None,
        split: str = 'train',
        val_ratio: float = 0.15,
        test_ratio: float = 0.10,
        seed: int = 42,
    ):
        super().__init__()

        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required. Install with: pip install pydicom")

        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.categories = categories
        self.window_center = window_center
        self.window_width = window_width
        self.cache_extracted = cache_extracted
        self.max_thick_slices = max_thick_slices
        self.max_thin_slices = max_thin_slices
        self.split = split

        # Blacklist of corrupted cases (verified to have no DICOM data or missing ZIPs)
        # From previous preprocessing runs - these cases have only admin files or are missing
        self.corrupted_cases = {
            'case_091', 'case_094',  # Missing ZIP files
            'case_190', 'case_191', 'case_192', 'case_193', 'case_194',
            'case_195', 'case_196', 'case_197', 'case_198', 'case_199',
            'case_200', 'case_201', 'case_202', 'case_203', 'case_204',
            'case_205', 'case_206'
        }

        # Setup extraction and preprocessing directories
        if extract_dir is None:
            self.extract_dir = Path(tempfile.mkdtemp(prefix='slice_interp_extract_'))
            self._temp_extract_dir = True
        else:
            self.extract_dir = Path(extract_dir)
            self.extract_dir.mkdir(parents=True, exist_ok=True)
            self._temp_extract_dir = False

        # Preprocessing cache directory (persistent)
        self.processed_dir = self.extract_dir.parent / 'processed' if extract_dir else Path(tempfile.gettempdir()) / 'slice_interp_processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Find all patient zip files
        all_patient_files = self._find_patient_files()

        # Preprocess all data ONCE (extract, process, cache, cleanup)
        logger.info(f"\n{'='*70}")
        logger.info(f"Preprocessing CT Slice Interpolation Dataset")
        logger.info(f"{'='*70}")
        successfully_preprocessed = self._preprocess_all_patients(all_patient_files)

        # CRITICAL: Only split files that were successfully preprocessed
        # This ensures train/val/test sets contain ONLY valid, usable files
        valid_patient_files = [p for p in all_patient_files if p['patient_id'] in successfully_preprocessed]

        logger.info(f"\n{'='*70}")
        logger.info(f"Train-Test Split Summary")
        logger.info(f"{'='*70}")
        logger.info(f"Total files found: {len(all_patient_files)}")
        logger.info(f"Successfully preprocessed: {len(valid_patient_files)}")
        logger.info(f"Excluded (failed preprocessing): {len(all_patient_files) - len(valid_patient_files)}")
        logger.info(f"{'='*70}\n")

        # Split into train/val/test (ONLY using successfully preprocessed files)
        self.patient_files = self._split_patients(valid_patient_files, val_ratio, test_ratio, seed)

        print(f"\n✓ {split.capitalize()} set: {len(self.patient_files)} patients")
        print(f"  Categories: {categories}")
        print(f"  Resolution: {resolution}")
        print(f"  Slice limits: thick={max_thick_slices or 'all'}, thin={max_thin_slices or 'all'}")

    def _find_patient_files(self) -> List[Dict]:
        """Find all patient files (ZIP or already-extracted directories) in specified categories"""
        patient_files = []
        corrupted_count = 0

        for category in self.categories:
            category_dir = self.data_dir / category
            if not category_dir.exists():
                logger.warning(f"Category directory not found: {category_dir}")
                continue

            # Look for both ZIP files and extracted directories
            zip_files = sorted(category_dir.glob('*.zip'))
            # Filter to only valid, existing ZIP files
            zip_files = [z for z in zip_files if z.exists() and z.is_file()]

            # Look for already-extracted directories (case_*)
            case_dirs = sorted([d for d in category_dir.iterdir()
                               if d.is_dir() and d.name.startswith('case_')])

            # Log what we found for debugging
            logger.info(f"Category '{category}': Found {len(zip_files)} .zip files, {len(case_dirs)} extracted directories")

            # Prefer extracted directories over ZIP files (faster, no extraction needed)
            if len(case_dirs) > 0:
                logger.info(f"Using {len(case_dirs)} extracted directories from {category}")

                for case_dir in case_dirs:
                    # Skip corrupted cases
                    if case_dir.name in self.corrupted_cases:
                        corrupted_count += 1
                        logger.debug(f"Skipping corrupted case: {case_dir.name}")
                        continue

                    # Create entry pointing to extracted directory
                    patient_files.append({
                        'zip_path': case_dir,  # Point to extracted directory instead of ZIP
                        'category': category,
                        'patient_id': case_dir.name,
                        'already_extracted': True  # Flag to skip extraction
                    })

            elif len(zip_files) > 0:
                logger.info(f"Using {len(zip_files)} .zip files from {category}")

                # Fall back to ZIP files only if no extracted directories found
                for zip_path in zip_files:
                    # Skip corrupted cases
                    if zip_path.stem in self.corrupted_cases:
                        corrupted_count += 1
                        logger.debug(f"Skipping corrupted case: {zip_path.stem}")
                        continue

                    patient_files.append({
                        'zip_path': zip_path,
                        'category': category,
                        'patient_id': zip_path.stem
                    })
            else:
                logger.warning(f"Category '{category}': No ZIP files or extracted directories found!")

        if corrupted_count > 0:
            logger.info(f"Skipped {corrupted_count} corrupted cases (no DICOM data)")

        return patient_files

    def _split_patients(self, valid_patients: List[Dict], val_ratio: float, test_ratio: float, seed: int) -> List[Dict]:
        """
        Split patients into train/val/test

        NOTE: Input is already filtered to only successfully preprocessed patients
        """
        if len(valid_patients) == 0:
            logger.error("No valid patients to split!")
            return []

        random.seed(seed)
        shuffled = valid_patients.copy()
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_test - n_val

        if self.split == 'train':
            return shuffled[:n_train]
        elif self.split == 'val':
            return shuffled[n_train:n_train+n_val]
        elif self.split == 'test':
            return shuffled[n_train+n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def _preprocess_all_patients(self, all_patients: List[Dict]) -> set:
        """
        Preprocess all patients ONCE and cache results

        Returns:
            set: Patient IDs that were successfully preprocessed (either now or previously)
        """
        from tqdm import tqdm

        # First, find which patients already have cached data
        already_cached = set()
        to_process = []

        for patient in all_patients:
            processed_file = self.processed_dir / f"{patient['patient_id']}.pt"
            if processed_file.exists():
                already_cached.add(patient['patient_id'])
            else:
                to_process.append(patient)

        if len(to_process) == 0:
            logger.info(f"✓ All {len(all_patients)} patients already preprocessed")
            logger.info(f"✓ Loading from cache: {self.processed_dir}")
            logger.info(f"✓ Future epochs will be FAST with NO warnings!")
            return already_cached  # Return all already-cached patient IDs

        logger.info(f"⚙ FIRST-TIME PREPROCESSING: {len(to_process)} patients")
        logger.info(f"⚙ This happens ONCE - extracting DICOMs and caching...")
        logger.info(f"⚙ Output cache: {self.processed_dir}")
        logger.info(f"⚙ Note: GB18030 warnings are normal (Chinese DICOM encoding)")
        logger.info(f"⚙ After this completes, future epochs will load from cache (fast!)")

        successful = 0
        failed = []

        for patient_info in tqdm(to_process, desc="Preprocessing"):
            case_id = patient_info['patient_id']
            try:
                # Extract patient data (or use already-extracted directory)
                patient_dir = self._extract_patient_data(patient_info)

                if patient_dir is None:
                    failed.append((case_id, "Failed to extract ZIP"))
                    continue

                # Load volumes
                thick_dir = patient_dir / '1'
                thin_dir = patient_dir / '2'

                if not thick_dir.exists() or not thin_dir.exists():
                    failed.append((case_id, f"Missing directories: 1={thick_dir.exists()}, 2={thin_dir.exists()}"))
                    # Cleanup
                    if patient_dir.exists():
                        shutil.rmtree(patient_dir, ignore_errors=True)
                    continue

                thick_volume = self._load_dicom_volume(thick_dir, max_slices=self.max_thick_slices)
                thin_volume = self._load_dicom_volume(thin_dir, max_slices=self.max_thin_slices)

                if thick_volume is None or thin_volume is None:
                    failed.append((case_id, f"Failed to load volumes: thick={thick_volume is not None}, thin={thin_volume is not None}"))
                    # Cleanup
                    if patient_dir.exists():
                        shutil.rmtree(patient_dir, ignore_errors=True)
                    continue

                # Apply preprocessing
                thick_volume = self._apply_ct_windowing(thick_volume)
                thin_volume = self._apply_ct_windowing(thin_volume)

                # Resize if needed
                if thick_volume.shape[1:] != self.resolution:
                    thick_volume = self._resize_slices(thick_volume, self.resolution)
                if thin_volume.shape[1:] != self.resolution:
                    thin_volume = self._resize_slices(thin_volume, self.resolution)

                # Convert to tensors
                thick_tensor = torch.from_numpy(thick_volume).unsqueeze(0).float()
                thin_tensor = torch.from_numpy(thin_volume).unsqueeze(0).float()

                # Normalize to [-1, 1]
                thick_tensor = thick_tensor * 2.0 - 1.0
                thin_tensor = thin_tensor * 2.0 - 1.0

                # Save preprocessed data
                # Use 'input' and 'target' keys to match training script expectations
                processed_file = self.processed_dir / f"{case_id}.pt"
                torch.save({
                    'input': thick_tensor,   # Thick slices (input)
                    'target': thin_tensor,   # Thin slices (target/ground truth)
                    'thick': thick_tensor,   # Also keep for compatibility
                    'thin': thin_tensor,
                    'category': patient_info['category'],
                    'patient_id': case_id,
                    'num_thick_slices': thick_tensor.shape[1],
                    'num_thin_slices': thin_tensor.shape[1],
                }, processed_file)

                successful += 1

                # IMPORTANT: Delete extracted DICOM files to save storage
                if patient_dir.exists():
                    shutil.rmtree(patient_dir, ignore_errors=True)
                    logger.debug(f"✓ {case_id}: Preprocessed and cleaned up")

            except Exception as e:
                logger.error(f"✗ {case_id}: {e}")
                failed.append((case_id, str(e)))
                # Cleanup on error
                try:
                    patient_dir = self.extract_dir / case_id
                    if patient_dir.exists():
                        shutil.rmtree(patient_dir, ignore_errors=True)
                except:
                    pass
                continue

        # Calculate total successful (already cached + newly processed)
        total_successful = len(already_cached) + successful

        logger.info(f"\n{'='*70}")
        logger.info(f"✓ PREPROCESSING COMPLETE:")
        logger.info(f"  Already cached: {len(already_cached)} patients")
        logger.info(f"  Newly processed: {successful} successful, {len(failed)} failed")
        logger.info(f"  Total available: {total_successful} patients")
        logger.info(f"✓ Cached to: {self.processed_dir}")
        logger.info(f"✓ Cache size: ~{successful * 0.3:.1f} GB (new data)")
        logger.info(f"✓ Future epochs will load from cache - NO more DICOM reading!")
        logger.info(f"✓ Expected speedup: 40% faster per epoch")
        logger.info(f"{'='*70}\n")

        # Save failure report if any
        if len(failed) > 0:
            failure_report = self.processed_dir.parent / 'preprocessing_failures.txt'
            with open(failure_report, 'w') as f:
                f.write(f"Slice Interpolation Preprocessing Failures\n")
                f.write(f"{'='*70}\n")
                f.write(f"Total failed: {len(failed)}\n")
                f.write(f"Total successful (new): {successful}\n")
                f.write(f"Total available (cached + new): {total_successful}\n\n")
                f.write(f"Failed cases:\n")
                f.write(f"{'-'*70}\n")
                for case_id, error in failed:
                    f.write(f"{case_id}: {error}\n")
            logger.warning(f"⚠ Failure report saved: {failure_report}")

        # Return set of ALL successfully preprocessed patient IDs (cached + new)
        # Collect newly successful patient IDs
        newly_successful = set()
        for patient_info in to_process:
            case_id = patient_info['patient_id']
            processed_file = self.processed_dir / f"{case_id}.pt"
            if processed_file.exists():
                newly_successful.add(case_id)

        return already_cached | newly_successful  # Union of cached and new

    def _extract_patient_data(self, patient_data: Dict) -> Optional[Path]:
        """Extract patient zip file if needed, or use already-extracted directory"""
        zip_path = patient_data['zip_path']
        already_extracted = patient_data.get('already_extracted', False)

        # If already extracted, use the directory directly
        if already_extracted:
            patient_dir = zip_path  # zip_path is actually the extracted directory
            logger.debug(f"Using already-extracted directory: {patient_dir}")

            # Find the actual patient data directory (handle nested structure)
            # Check if directory contains 1/ and 2/ directly
            if (patient_dir / '1').exists() or (patient_dir / '2').exists():
                return patient_dir

            # Otherwise look for nested structure (one level down)
            subdirs = [d for d in patient_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if len(subdirs) == 1:
                inner_dir = subdirs[0]
                if (inner_dir / '1').exists() or (inner_dir / '2').exists():
                    return inner_dir

            # Search recursively
            all_dirs = [d for d in patient_dir.rglob('*') if d.is_dir() and not d.name.startswith('.')]
            for d in all_dirs:
                if (d / '1').exists() or (d / '2').exists():
                    return d

            logger.error(f"Could not find patient data structure in {patient_dir.name}")
            return None

        # Otherwise extract from ZIP
        patient_dir = self.extract_dir / zip_path.stem

        if patient_dir.exists() and self.cache_extracted:
            # Verify extraction has subdirectories
            subdirs = [d for d in patient_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if len(subdirs) > 0:
                return patient_dir
            else:
                logger.warning(f"Cached extraction {zip_path.stem} has no subdirectories, re-extracting...")
                shutil.rmtree(patient_dir)

        # Extract zip file
        try:
            if patient_dir.exists():
                shutil.rmtree(patient_dir)
            patient_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(patient_dir)

            # Find the actual patient directory (handle nested structure)
            # Check if extraction created subdirectories 1/ and 2/ directly
            if (patient_dir / '1').exists() or (patient_dir / '2').exists():
                return patient_dir

            # Otherwise look for nested structure (one level down)
            subdirs = [d for d in patient_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if len(subdirs) == 1:
                # Check if this single subdir contains 1/ and 2/
                inner_dir = subdirs[0]
                if (inner_dir / '1').exists() or (inner_dir / '2').exists():
                    return inner_dir

            # If still not found, search recursively (rare case)
            all_dirs = [d for d in patient_dir.rglob('*') if d.is_dir() and not d.name.startswith('.')]
            for d in all_dirs:
                if (d / '1').exists() or (d / '2').exists():
                    return d

            logger.error(f"Could not find patient data structure in {zip_path.stem}")
            return None

        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            return None

    def _load_dicom_volume(self, dicom_dir: Path, max_slices: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Load all DICOM slices from a directory and stack into volume

        Args:
            dicom_dir: Directory containing DICOM files
            max_slices: Maximum number of slices to keep (None = all)

        Returns:
            volume: numpy array of shape (num_slices, H, W) or None if error
        """
        # Known non-DICOM files to skip (from previous preprocessing experience)
        non_dicom_files = {'LOCKFILE', 'VERSION', 'DICOMDIR', '.DS_Store', 'Thumbs.db'}

        try:
            # Find all DICOM files (filter out known non-DICOM files)
            dicom_files = []
            for root, dirs, files in os.walk(dicom_dir):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]

                for file in files:
                    # Skip hidden files and known non-DICOM files
                    if file.startswith('.') or file in non_dicom_files:
                        continue

                    filepath = Path(root) / file
                    try:
                        # Quick check if it's a DICOM file
                        pydicom.dcmread(filepath, stop_before_pixels=True)
                        dicom_files.append(filepath)
                    except:
                        # Not a DICOM file, skip silently
                        continue

            if not dicom_files:
                logger.warning(f"No DICOM files found in {dicom_dir}")
                return None

            logger.debug(f"Found {len(dicom_files)} DICOM files in {dicom_dir.name}")

            # Load and sort slices
            slices = []
            failed_reads = 0
            for dcm_path in dicom_files:
                try:
                    dcm = pydicom.dcmread(dcm_path)
                    if hasattr(dcm, 'pixel_array'):
                        slices.append({
                            'image': dcm.pixel_array.astype(np.float32),
                            'position': float(dcm.ImagePositionPatient[2]) if hasattr(dcm, 'ImagePositionPatient') else 0,
                            'instance': int(dcm.InstanceNumber) if hasattr(dcm, 'InstanceNumber') else 0,
                        })
                    else:
                        failed_reads += 1
                except Exception as e:
                    failed_reads += 1
                    logger.debug(f"Failed to read {dcm_path.name}: {e}")
                    continue

            if not slices:
                logger.error(f"No valid DICOM slices with pixel_array in {dicom_dir}")
                if failed_reads > 0:
                    logger.error(f"  Failed to read {failed_reads}/{len(dicom_files)} files")
                return None

            logger.debug(f"Loaded {len(slices)} slices ({failed_reads} failed)")

            # Sort by position (z-coordinate) and instance number
            slices.sort(key=lambda x: (x['position'], x['instance']))

            # Optionally limit number of slices (centered crop)
            if max_slices is not None and len(slices) > max_slices:
                start_idx = (len(slices) - max_slices) // 2
                slices = slices[start_idx:start_idx + max_slices]
                logger.debug(f"Limited to {max_slices} slices (centered crop)")

            # Stack into volume
            volume = np.stack([s['image'] for s in slices], axis=0)

            return volume

        except Exception as e:
            logger.error(f"Error loading DICOM volume from {dicom_dir}: {e}")
            return None

    def _apply_ct_windowing(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply CT windowing to convert HU values to normalized range

        Args:
            volume: CT volume in Hounsfield Units (float32)

        Returns:
            windowed_volume: Float32 values in range [0, 1]
        """
        lower = self.window_center - (self.window_width / 2)
        upper = self.window_center + (self.window_width / 2)

        # Clip and normalize to [0, 1]
        windowed = np.clip(volume, lower, upper)
        windowed = ((windowed - lower) / (upper - lower)).astype(np.float32)

        return windowed

    def _resize_slices(self, volume: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize all slices in volume to target size

        Args:
            volume: (D, H, W) numpy array
            target_size: (H_new, W_new)

        Returns:
            resized_volume: (D, H_new, W_new) numpy array
        """
        D, H, W = volume.shape
        H_new, W_new = target_size

        # Convert to tensor for efficient resizing
        volume_tensor = torch.from_numpy(volume).unsqueeze(1)  # (D, 1, H, W)

        # Resize all slices at once
        resized_tensor = torch.nn.functional.interpolate(
            volume_tensor,
            size=(H_new, W_new),
            mode='bilinear',
            align_corners=False
        )

        resized_volume = resized_tensor.squeeze(1).numpy()  # (D, H_new, W_new)

        return resized_volume

    def __len__(self) -> int:
        return len(self.patient_files)

    def __getitem__(self, idx: int, _retry_count: int = 0) -> Dict[str, torch.Tensor]:
        """
        Load preprocessed patient data (FAST - just loads cached .pt file)

        Returns:
            dict with:
                'thick': thick slices tensor (1, D_thick, H, W) - typically (1, 50, 512, 512)
                'thin': thin slices tensor (1, D_thin, H, W) - typically (1, 300, 512, 512)
                'category': 'APE' or 'non-APE'
                'patient_id': patient identifier
                'num_thick_slices': actual number of thick slices
                'num_thin_slices': actual number of thin slices
        """
        # Prevent infinite recursion if many files are corrupted
        if _retry_count > 10:
            logger.warning(f"Too many corrupted files encountered, returning dummy sample")
            return self._get_dummy_sample(self.patient_files[idx])

        patient_info = self.patient_files[idx]
        processed_file = self.processed_dir / f"{patient_info['patient_id']}.pt"

        # Log on first sample to confirm cache loading
        if idx == 0 and not hasattr(self, '_first_load_logged'):
            logger.info(f"✓ Loading from preprocessed cache (fast, no DICOM warnings!)")
            logger.info(f"✓ Cache location: {self.processed_dir}")
            self._first_load_logged = True

        try:
            # Load preprocessed tensors directly (very fast!)
            # weights_only=False is safe here because we control the data source
            sample_dict = torch.load(processed_file, weights_only=False)
            return sample_dict
        except Exception as e:
            # Log error only on first retry to avoid spam
            if _retry_count == 0:
                logger.warning(f"Skipping corrupted file {patient_info['patient_id']}: {str(e)[:100]}")

            # Skip to next sample (needed because DataLoader requires valid return)
            next_idx = (idx + 1) % len(self.patient_files)
            return self.__getitem__(next_idx, _retry_count=_retry_count + 1)

    def _get_dummy_sample(self, patient_info: Dict) -> Dict[str, torch.Tensor]:
        """Create dummy sample for error cases"""
        H, W = self.resolution
        dummy_thick = torch.zeros(1, 50, H, W)
        dummy_thin = torch.zeros(1, 300, H, W)

        return {
            'thick': dummy_thick,
            'thin': dummy_thin,
            'category': patient_info['category'],
            'patient_id': patient_info['patient_id'],
            'num_thick_slices': 50,
            'num_thin_slices': 300,
        }

    def __del__(self):
        """Cleanup temporary extraction directory"""
        if self._temp_extract_dir and self.extract_dir.exists():
            shutil.rmtree(self.extract_dir, ignore_errors=True)


def collate_variable_depth(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle variable depth volumes

    Pads volumes to the maximum depth in the batch and stacks them into tensors.
    This allows batch_size > 1 while handling variable depths.

    Args:
        batch: List of samples from __getitem__

    Returns:
        dict with batched tensors (all volumes padded to max depth in batch)
    """
    import torch.nn.functional as F

    # Get all thick and thin tensors
    thick_tensors = [item['thick'] for item in batch]  # List of (1, D_thick, H, W)
    thin_tensors = [item['thin'] for item in batch]    # List of (1, D_thin, H, W)

    # Find maximum depths in this batch
    max_thick_depth = max(t.shape[1] for t in thick_tensors)  # Max D_thick
    max_thin_depth = max(t.shape[1] for t in thin_tensors)    # Max D_thin

    # Pad all tensors to max depth (pad at the end of depth dimension)
    # Also create padding masks: 1 = real data, 0 = padded (CRITICAL for correct loss!)
    thick_padded = []
    thin_padded = []
    thick_masks = []
    thin_masks = []

    for thick, thin in zip(thick_tensors, thin_tensors):
        # Pad thick: shape (1, D, H, W) -> (1, max_D, H, W)
        thick_depth = thick.shape[1]
        thick_padding = max_thick_depth - thick_depth
        if thick_padding > 0:
            # Pad along dimension 1 (depth): (pad_left, pad_right) for last 3 dims
            # For 4D: (W_left, W_right, H_left, H_right, D_left, D_right, C_left, C_right)
            # We want to pad depth (dim 1), so: (0, 0, 0, 0, 0, thick_padding, 0, 0)
            # FIXED: Pad with -1.0 (air/background in CT) instead of 0 (mid-range intensity)
            thick_padded.append(F.pad(thick, (0, 0, 0, 0, 0, thick_padding), value=-1.0))
        else:
            thick_padded.append(thick)

        # Create mask: 1 for real data, 0 for padding (1, D)
        thick_mask = torch.cat([
            torch.ones(1, thick_depth),
            torch.zeros(1, thick_padding)
        ], dim=1)
        thick_masks.append(thick_mask)

        # Pad thin: shape (1, D, H, W) -> (1, max_D, H, W)
        thin_depth = thin.shape[1]
        thin_padding = max_thin_depth - thin_depth
        if thin_padding > 0:
            # FIXED: Pad with -1.0 (air/background in CT) instead of 0 (mid-range intensity)
            thin_padded.append(F.pad(thin, (0, 0, 0, 0, 0, thin_padding), value=-1.0))
        else:
            thin_padded.append(thin)

        # Create mask: 1 for real data, 0 for padding (1, D)
        thin_mask = torch.cat([
            torch.ones(1, thin_depth),
            torch.zeros(1, thin_padding)
        ], dim=1)
        thin_masks.append(thin_mask)

    # Stack into batch tensors: (B, 1, D, H, W)
    thick_batch = torch.stack(thick_padded, dim=0)  # (B, 1, max_D_thick, H, W)
    thin_batch = torch.stack(thin_padded, dim=0)    # (B, 1, max_D_thin, H, W)
    thick_mask_batch = torch.stack(thick_masks, dim=0)  # (B, 1, max_D_thick)
    thin_mask_batch = torch.stack(thin_masks, dim=0)    # (B, 1, max_D_thin)

    return {
        'thick': thick_batch,
        'thin': thin_batch,
        'input': thick_batch,    # Alias for trainer compatibility
        'target': thin_batch,    # Alias for trainer compatibility
        'thick_mask': thick_mask_batch,  # Padding mask for thick slices
        'thin_mask': thin_mask_batch,    # Padding mask for thin slices
        'category': [item['category'] for item in batch],
        'patient_id': [item['patient_id'] for item in batch],
        'num_thick_slices': [item['num_thick_slices'] for item in batch],
        'num_thin_slices': [item['num_thin_slices'] for item in batch],
    }


def get_slice_interpolation_dataloader(
    data_dir: str,
    config: dict,
    split: str = 'train'
) -> torch.utils.data.DataLoader:
    """
    Create dataloader for CT slice interpolation with full volumes

    Args:
        data_dir: Path to APE-data root directory
        config: Config dict with dataset parameters
        split: 'train', 'val', or 'test'

    Returns:
        dataloader: PyTorch DataLoader with variable-depth volumes
    """
    from torch.utils.data import DataLoader

    # Determine categories based on split
    categories = config.get('categories', ['APE', 'non-APE'])

    dataset = SliceInterpolationDataset(
        data_dir=data_dir,
        resolution=tuple(config.get('resolution', [512, 512])),
        categories=categories,
        extract_dir=config.get('extract_dir', None),  # Use config-specified directory for preprocessing cache
        cache_extracted=config.get('cache_extracted', False),  # No longer needed - we delete after preprocessing
        window_center=config.get('window_center', 40),
        window_width=config.get('window_width', 400),
        max_thick_slices=config.get('max_thick_slices', None),
        max_thin_slices=config.get('max_thin_slices', None),
        split=split,
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.10),
        seed=config.get('seed', 42),
    )

    batch_size = config.get('batch_size', 1)  # Typically 1-2 for full volumes
    num_workers = config.get('num_workers', 0)  # Use 0 for DICOM to avoid issues
    shuffle = (split == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=collate_variable_depth  # Handle variable depths
    )

    return dataloader


if __name__ == "__main__":
    # Test the slice interpolation dataset with full volumes
    print("Testing CT Slice Interpolation Dataset (Full Volumes)...\n")
    print("="*80)

    data_dir = "/Users/kuntalkokate/Desktop/LLM_agents_projects/dataset"

    if not Path(data_dir).exists():
        print(f"Data directory not found: {data_dir}")
        print("Please update the path in the script")
    else:
        try:
            # Create test dataset
            dataset = SliceInterpolationDataset(
                data_dir=data_dir,
                resolution=(512, 512),
                categories=['APE'],  # Test with APE only
                cache_extracted=False,
                max_thick_slices=50,    # Limit to 50 thick slices
                max_thin_slices=300,    # Limit to 300 thin slices
                split='train',
                val_ratio=0.15,
                test_ratio=0.10,
                seed=42,
            )

            print(f"\n✓ Dataset created: {len(dataset)} patients")

            if len(dataset) > 0:
                print("\nLoading first sample...")
                sample = dataset[0]

                print(f"\nSample structure:")
                print(f"  Keys: {sample.keys()}")
                print(f"\n  Thick slices: {sample['thick'].shape} {sample['thick'].dtype}")
                print(f"  Thin slices:  {sample['thin'].shape} {sample['thin'].dtype}")
                print(f"  Category: {sample['category']}")
                print(f"  Patient ID: {sample['patient_id']}")
                print(f"  Num thick slices: {sample['num_thick_slices']}")
                print(f"  Num thin slices: {sample['num_thin_slices']}")

                print(f"\nValue ranges:")
                print(f"  Thick: [{sample['thick'].min():.3f}, {sample['thick'].max():.3f}]")
                print(f"  Thin:  [{sample['thin'].min():.3f}, {sample['thin'].max():.3f}]")

                # Check interpolation factor
                interp_factor = sample['num_thin_slices'] / sample['num_thick_slices']
                print(f"\nInterpolation:")
                print(f"  {sample['num_thick_slices']} thick → {sample['num_thin_slices']} thin")
                print(f"  Factor: {interp_factor:.1f}×")

                # Test dataloader with custom collate function
                print("\n" + "-"*80)
                print("Testing dataloader with batch_size=1...")

                config = {
                    'batch_size': 1,
                    'num_workers': 0,
                    'categories': ['APE'],
                    'resolution': [512, 512],
                    'max_thick_slices': 50,
                    'max_thin_slices': 300,
                    'cache_extracted': False,
                }

                dataloader = get_slice_interpolation_dataloader(
                    data_dir=data_dir,
                    config=config,
                    split='train'
                )

                print(f"✓ Dataloader created: {len(dataloader)} batches")

                # Load one batch
                batch = next(iter(dataloader))
                print(f"\nBatch structure:")
                print(f"  Thick: {batch['thick'].shape}")
                print(f"  Thin: {batch['thin'].shape}")
                print(f"  Patient IDs: {batch['patient_id']}")

                print("\n" + "="*80)
                print("✅ Full-volume dataset test successful!")

            else:
                print("No data found in dataset")

        except Exception as e:
            print(f"\n✗ Dataset test failed: {e}")
            import traceback
            traceback.print_exc()
