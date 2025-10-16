"""
APE-Data DICOM Dataset Loader

This dataset handles the specific structure of APE-data:
- Each patient has two timepoints (baseline and followup)
- Data is in DICOM format (medical CT scans)
- Structure: patient_folder/1/ (input) and patient_folder/2/ (target)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional
import zipfile
import tempfile
import shutil

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not installed. Install with: pip install pydicom")

from .transforms import VideoTransform


class APEDataset(Dataset):
    """
    APE-Data DICOM Dataset

    Loads CT scan pairs (baseline -> followup) for video-to-video diffusion

    Data structure:
        data_dir/
            APE/
                patient1.zip -> patient1/1/ (input), patient1/2/ (target)
                patient2.zip -> patient2/1/ (input), patient2/2/ (target)
            non-APE/
                patient3.zip -> patient3/1/ (input), patient3/2/ (target)
    """

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 16,
        resolution: Tuple[int, int] = (256, 256),
        categories: List[str] = ['APE', 'non-APE'],
        transform=None,
        extract_dir: Optional[str] = None,
        cache_extracted: bool = True,
        normalize: bool = True,
        window_center: int = 40,  # HU window center for soft tissue
        window_width: int = 400,  # HU window width
    ):
        """
        Args:
            data_dir: Root directory containing APE/ and non-APE/ folders
            num_frames: Number of slices to sample from each volume
            resolution: (H, W) to resize each slice
            categories: List of categories to include ['APE', 'non-APE']
            transform: Additional transforms
            extract_dir: Directory to extract zip files (None = temp dir)
            cache_extracted: Keep extracted files for faster loading
            normalize: Apply HU windowing and normalization
            window_center: HU window center for CT windowing
            window_width: HU window width for CT windowing
        """
        super().__init__()

        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required for APEDataset. Install with: pip install pydicom")

        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        self.categories = categories
        self.normalize = normalize
        self.window_center = window_center
        self.window_width = window_width
        self.cache_extracted = cache_extracted

        # Setup extraction directory
        if extract_dir is None:
            self.extract_dir = Path(tempfile.mkdtemp(prefix='ape_data_'))
            self._temp_extract_dir = True
        else:
            self.extract_dir = Path(extract_dir)
            self.extract_dir.mkdir(parents=True, exist_ok=True)
            self._temp_extract_dir = False

        # Default transform
        if transform is None:
            self.transform = VideoTransform(resolution=resolution, num_frames=num_frames)
        else:
            self.transform = transform

        # Find all patient zip files
        self.patient_files = self._find_patient_files()

        print(f"Found {len(self.patient_files)} patient studies")
        print(f"Categories: {categories}")
        print(f"Extraction directory: {self.extract_dir}")

    def _find_patient_files(self) -> List[dict]:
        """Find all patient zip files in specified categories"""
        patient_files = []

        for category in self.categories:
            category_dir = self.data_dir / category
            if not category_dir.exists():
                print(f"Warning: Category directory not found: {category_dir}")
                continue

            # Find all zip files
            zip_files = list(category_dir.glob('*.zip'))

            for zip_path in zip_files:
                patient_files.append({
                    'zip_path': zip_path,
                    'category': category,
                    'patient_id': zip_path.stem
                })

        return patient_files

    def _extract_patient_data(self, zip_path: Path) -> Optional[Path]:
        """Extract patient zip file if needed"""
        # Check if already extracted
        patient_dir = self.extract_dir / zip_path.stem

        if patient_dir.exists() and self.cache_extracted:
            return patient_dir

        # Extract zip file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)

            # Find the actual patient directory (might be nested)
            extracted_dirs = list(self.extract_dir.glob(f"{zip_path.stem}*"))
            if extracted_dirs:
                return extracted_dirs[0]
            else:
                # Check for nested structure
                all_dirs = [d for d in self.extract_dir.rglob('*') if d.is_dir()]
                for d in all_dirs:
                    if (d / '1').exists() and (d / '2').exists():
                        return d

            return None

        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
            return None

    def _load_dicom_volume(self, dicom_dir: Path) -> Optional[np.ndarray]:
        """
        Load all DICOM slices from a directory

        Returns:
            volume: numpy array of shape (num_slices, H, W) or None if error
        """
        try:
            # Find all DICOM files
            dicom_files = []
            for root, dirs, files in os.walk(dicom_dir):
                for file in files:
                    if not file.startswith('.'):
                        filepath = Path(root) / file
                        try:
                            # Quick check if it's a DICOM file
                            pydicom.dcmread(filepath, stop_before_pixels=True)
                            dicom_files.append(filepath)
                        except:
                            continue

            if not dicom_files:
                return None

            # Load and sort slices
            slices = []
            for dcm_path in dicom_files:
                try:
                    dcm = pydicom.dcmread(dcm_path)
                    if hasattr(dcm, 'pixel_array'):
                        slices.append({
                            'image': dcm.pixel_array,
                            'position': float(dcm.ImagePositionPatient[2]) if hasattr(dcm, 'ImagePositionPatient') else 0,
                            'instance': int(dcm.InstanceNumber) if hasattr(dcm, 'InstanceNumber') else 0
                        })
                except Exception as e:
                    continue

            if not slices:
                return None

            # Sort by position or instance number
            slices.sort(key=lambda x: (x['position'], x['instance']))

            # Stack into volume
            volume = np.stack([s['image'] for s in slices], axis=0)

            return volume

        except Exception as e:
            print(f"Error loading DICOM volume from {dicom_dir}: {e}")
            return None

    def _apply_ct_windowing(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply CT windowing to convert HU values to display range

        Args:
            volume: CT volume in Hounsfield Units

        Returns:
            windowed_volume: Values in range [0, 255]
        """
        lower = self.window_center - (self.window_width / 2)
        upper = self.window_center + (self.window_width / 2)

        # Clip and normalize
        windowed = np.clip(volume, lower, upper)
        windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)

        return windowed

    def _sample_frames(self, volume: np.ndarray) -> np.ndarray:
        """
        Sample num_frames evenly from the volume

        Args:
            volume: (D, H, W) array

        Returns:
            sampled: (num_frames, H, W) array
        """
        depth = volume.shape[0]

        if depth <= self.num_frames:
            # Repeat last frame if not enough
            sampled = np.zeros((self.num_frames, *volume.shape[1:]), dtype=volume.dtype)
            sampled[:depth] = volume
            sampled[depth:] = volume[-1]
        else:
            # Evenly sample frames
            indices = np.linspace(0, depth - 1, self.num_frames, dtype=int)
            sampled = volume[indices]

        return sampled

    def __len__(self):
        return len(self.patient_files)

    def __getitem__(self, idx):
        """
        Get a patient study pair

        Returns:
            dict with:
                'input': baseline scan tensor (3, T, H, W)
                'target': followup scan tensor (3, T, H, W)
                'category': 'APE' or 'non-APE'
                'patient_id': patient identifier
        """
        patient_info = self.patient_files[idx]

        # Extract patient data
        patient_dir = self._extract_patient_data(patient_info['zip_path'])

        if patient_dir is None:
            # Return dummy data on error
            print(f"Warning: Could not load patient {patient_info['patient_id']}")
            H, W = self.resolution
            dummy_volume = np.zeros((self.num_frames, H, W, 3), dtype=np.uint8)
            dummy_tensor = self.transform(dummy_volume)
            return {
                'input': dummy_tensor,
                'target': dummy_tensor,
                'category': patient_info['category'],
                'patient_id': patient_info['patient_id']
            }

        # Load baseline (timepoint 1) and followup (timepoint 2)
        baseline_dir = patient_dir / '1'
        followup_dir = patient_dir / '2'

        baseline_volume = self._load_dicom_volume(baseline_dir)
        followup_volume = self._load_dicom_volume(followup_dir)

        # Handle loading errors
        if baseline_volume is None or followup_volume is None:
            print(f"Warning: Could not load volumes for {patient_info['patient_id']}")
            H, W = self.resolution
            dummy_volume = np.zeros((self.num_frames, H, W, 3), dtype=np.uint8)
            dummy_tensor = self.transform(dummy_volume)
            return {
                'input': dummy_tensor,
                'target': dummy_tensor,
                'category': patient_info['category'],
                'patient_id': patient_info['patient_id']
            }

        # Apply CT windowing
        if self.normalize:
            baseline_volume = self._apply_ct_windowing(baseline_volume)
            followup_volume = self._apply_ct_windowing(followup_volume)
        else:
            # Just normalize to 0-255
            baseline_volume = ((baseline_volume - baseline_volume.min()) /
                             (baseline_volume.max() - baseline_volume.min() + 1e-8) * 255).astype(np.uint8)
            followup_volume = ((followup_volume - followup_volume.min()) /
                             (followup_volume.max() - followup_volume.min() + 1e-8) * 255).astype(np.uint8)

        # Sample frames
        baseline_sampled = self._sample_frames(baseline_volume)  # (T, H, W)
        followup_sampled = self._sample_frames(followup_volume)  # (T, H, W)

        # Convert grayscale to RGB (repeat channel 3 times)
        baseline_rgb = np.stack([baseline_sampled] * 3, axis=-1)  # (T, H, W, 3)
        followup_rgb = np.stack([followup_sampled] * 3, axis=-1)  # (T, H, W, 3)

        # Apply transforms (resize and normalize to [-1, 1])
        input_tensor = self.transform(baseline_rgb)
        target_tensor = self.transform(followup_rgb)

        # Cleanup if not caching
        if not self.cache_extracted and patient_dir.exists():
            shutil.rmtree(patient_dir, ignore_errors=True)

        return {
            'input': input_tensor,
            'target': target_tensor,
            'category': patient_info['category'],
            'patient_id': patient_info['patient_id']
        }

    def __del__(self):
        """Cleanup temporary extraction directory"""
        if self._temp_extract_dir and self.extract_dir.exists():
            shutil.rmtree(self.extract_dir, ignore_errors=True)


def get_ape_dataloader(data_dir, config, split='train'):
    """
    Create APE-data dataloader

    Args:
        data_dir: Path to APE-data root directory
        config: Config dict with dataset parameters
        split: 'train', 'val', or 'test' (for future train/val splits)

    Returns:
        dataloader: PyTorch DataLoader
    """
    from torch.utils.data import DataLoader

    # Determine categories based on split
    # For now, use both categories for all splits
    # You can implement proper splits later
    categories = config.get('categories', ['APE', 'non-APE'])

    dataset = APEDataset(
        data_dir=data_dir,
        num_frames=config.get('num_frames', 16),
        resolution=tuple(config.get('resolution', [256, 256])),
        categories=categories,
        cache_extracted=config.get('cache_extracted', True),
        window_center=config.get('window_center', 40),
        window_width=config.get('window_width', 400),
    )

    batch_size = config.get('batch_size', 1)
    num_workers = config.get('num_workers', 0)  # Use 0 for DICOM to avoid multiprocessing issues
    shuffle = (split == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


if __name__ == "__main__":
    # Test the APE dataset
    print("Testing APE dataset...")

    # You need to update this path to your actual data directory
    data_dir = "/Users/kuntalkokate/Desktop/LLM_agents_projects/dataset"

    if not Path(data_dir).exists():
        print(f"Data directory not found: {data_dir}")
        print("Please update the path in the script")
    else:
        dataset = APEDataset(
            data_dir=data_dir,
            num_frames=8,
            resolution=(128, 128),
            categories=['APE'],  # Test with APE only
            cache_extracted=False  # Don't cache for testing
        )

        print(f"\nDataset size: {len(dataset)}")

        if len(dataset) > 0:
            print("\nLoading first sample...")
            sample = dataset[0]

            print(f"Sample keys: {sample.keys()}")
            print(f"Input shape: {sample['input'].shape}")
            print(f"Target shape: {sample['target'].shape}")
            print(f"Category: {sample['category']}")
            print(f"Patient ID: {sample['patient_id']}")
            print(f"Input range: [{sample['input'].min():.2f}, {sample['input'].max():.2f}]")

            print("\nDataset test successful!")
        else:
            print("No data found in dataset")
