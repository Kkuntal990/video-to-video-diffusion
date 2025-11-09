"""
DICOM Utility Functions for Slice Interpolation

Provides utilities for:
- Loading DICOM series
- Matching thick/thin slice pairs spatially
- Extracting metadata
"""

import os
import numpy as np
import warnings
import logging

# Suppress pydicom warnings about character encodings (GB18030, etc.)
# These are harmless warnings from DICOM files with Chinese text
warnings.filterwarnings('ignore', message='.*cannot be used as code extension.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pydicom')

# IMPORTANT: Suppress pydicom logger warnings (GB18030 encoding)
logging.getLogger('pydicom').setLevel(logging.ERROR)
logging.getLogger('pydicom.charset').setLevel(logging.ERROR)

import pydicom
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def find_dicom_files(directory: str, max_files: Optional[int] = None) -> List[str]:
    """
    Recursively find all DICOM files in directory

    Args:
        directory: Root directory to search
        max_files: Maximum number of files to return (None = all)

    Returns:
        List of DICOM file paths
    """
    dicom_files = []

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # Skip hidden files and known non-DICOM files
            if filename.startswith('.') or filename in ['DICOMDIR', 'LOCKFILE', 'VERSION']:
                continue

            filepath = os.path.join(dirpath, filename)

            # Try to read as DICOM
            try:
                pydicom.dcmread(filepath, stop_before_pixels=True)
                dicom_files.append(filepath)

                if max_files and len(dicom_files) >= max_files:
                    return dicom_files
            except:
                # Not a DICOM file, skip
                pass

    return dicom_files


def organize_by_series(dicom_files: List[str]) -> Dict[str, List[str]]:
    """
    Organize DICOM files by SeriesInstanceUID

    Args:
        dicom_files: List of DICOM file paths

    Returns:
        Dictionary mapping series_uid -> list of file paths
    """
    series_dict = defaultdict(list)

    for filepath in dicom_files:
        try:
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
            series_uid = dcm.SeriesInstanceUID
            series_dict[series_uid].append(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return dict(series_dict)


def get_series_metadata(dicom_files: List[str]) -> Dict:
    """
    Extract metadata from a DICOM series

    Args:
        dicom_files: List of DICOM file paths in series

    Returns:
        Dictionary with series metadata
    """
    if not dicom_files:
        return None

    # Read first file for series-level metadata
    first_dcm = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)

    metadata = {
        'num_slices': len(dicom_files),
        'series_description': getattr(first_dcm, 'SeriesDescription', 'Unknown'),
        'series_number': getattr(first_dcm, 'SeriesNumber', 'Unknown'),
        'modality': getattr(first_dcm, 'Modality', 'Unknown'),
        'rows': getattr(first_dcm, 'Rows', None),
        'columns': getattr(first_dcm, 'Columns', None),
        'pixel_spacing': getattr(first_dcm, 'PixelSpacing', None),
        'slice_thickness': getattr(first_dcm, 'SliceThickness', None),
        'spacing_between_slices': getattr(first_dcm, 'SpacingBetweenSlices', None),
    }

    # Calculate actual spacing between slices from ImagePositionPatient
    try:
        positions = []
        for filepath in dicom_files[:min(10, len(dicom_files))]:
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
            if hasattr(dcm, 'ImagePositionPatient'):
                positions.append(float(dcm.ImagePositionPatient[2]))  # Z position

        if len(positions) >= 2:
            positions = sorted(positions)
            spacings = [abs(positions[i+1] - positions[i]) for i in range(len(positions)-1)]
            metadata['measured_slice_spacing'] = np.mean(spacings)
        else:
            metadata['measured_slice_spacing'] = None
    except:
        metadata['measured_slice_spacing'] = None

    return metadata


def load_dicom_series_with_positions(dicom_files: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load DICOM series as 3D volume with Z-positions

    Args:
        dicom_files: List of DICOM file paths

    Returns:
        volume: 3D numpy array (D, H, W)
        z_positions: 1D numpy array of Z-coordinates for each slice
    """
    # Sort by instance number and slice location
    sorted_files = []
    for filepath in dicom_files:
        dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
        instance_num = getattr(dcm, 'InstanceNumber', 0)
        slice_loc = getattr(dcm, 'SliceLocation', 0)

        # Get Z position from ImagePositionPatient
        if hasattr(dcm, 'ImagePositionPatient'):
            z_pos = float(dcm.ImagePositionPatient[2])
        else:
            z_pos = slice_loc

        sorted_files.append((z_pos, instance_num, slice_loc, filepath))

    # Sort by Z position primarily, then instance number
    sorted_files.sort(key=lambda x: (x[0], x[1]))

    # Load pixel data
    slices = []
    z_positions = []

    for z_pos, _, _, filepath in sorted_files:
        dcm = pydicom.dcmread(filepath)
        slices.append(dcm.pixel_array)
        z_positions.append(z_pos)

    volume = np.stack(slices, axis=0)
    z_positions = np.array(z_positions)

    return volume, z_positions


def match_thick_thin_slices(thick_z_positions: np.ndarray,
                            thin_z_positions: np.ndarray,
                            tolerance: float = 0.5) -> List[Tuple[int, List[int]]]:
    """
    Match thick slices with corresponding thin slices based on Z-positions

    Args:
        thick_z_positions: Z-coordinates of thick slices (N_thick,)
        thin_z_positions: Z-coordinates of thin slices (N_thin,)
        tolerance: Maximum distance (mm) to consider slices as matching

    Returns:
        List of (thick_idx, [thin_indices]) pairs
    """
    matches = []

    for thick_idx in range(len(thick_z_positions) - 1):
        # Get range of Z-positions covered by this thick slice pair
        z_start = thick_z_positions[thick_idx]
        z_end = thick_z_positions[thick_idx + 1]

        # Find all thin slices within this range
        thin_indices = []
        for thin_idx, thin_z in enumerate(thin_z_positions):
            if z_start - tolerance <= thin_z <= z_end + tolerance:
                thin_indices.append(thin_idx)

        if thin_indices:
            matches.append((thick_idx, thin_indices))

    return matches


def identify_thick_thin_series(series_dict: Dict[str, List[str]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify which series is thick-slice and which is thin-slice

    Args:
        series_dict: Dictionary of series_uid -> file_list

    Returns:
        (thick_series_uid, thin_series_uid)
    """
    # Get metadata for all series
    series_metadata = {}
    for series_uid, files in series_dict.items():
        metadata = get_series_metadata(files)
        if metadata:
            series_metadata[series_uid] = metadata

    if len(series_metadata) < 2:
        print(f"Warning: Found only {len(series_metadata)} series, need at least 2")
        return None, None

    # Identify thick vs thin based on slice thickness
    series_with_thickness = []
    for series_uid, metadata in series_metadata.items():
        thickness = metadata.get('slice_thickness') or metadata.get('measured_slice_spacing')
        if thickness:
            series_with_thickness.append((series_uid, float(thickness), metadata['num_slices']))

    if len(series_with_thickness) < 2:
        print("Warning: Could not determine slice thickness for series")
        return None, None

    # Sort by thickness
    series_with_thickness.sort(key=lambda x: x[1])

    # Thin series has smallest thickness, thick series has largest
    thin_series_uid = series_with_thickness[0][0]
    thick_series_uid = series_with_thickness[-1][0]

    thin_thickness = series_with_thickness[0][1]
    thick_thickness = series_with_thickness[-1][1]

    print(f"Identified series:")
    print(f"  Thin series: {thin_series_uid[:8]}... ({thin_thickness}mm, {series_with_thickness[0][2]} slices)")
    print(f"  Thick series: {thick_series_uid[:8]}... ({thick_thickness}mm, {series_with_thickness[-1][2]} slices)")

    return thick_series_uid, thin_series_uid


def apply_ct_windowing(volume: np.ndarray,
                      window_center: float = 200,
                      window_width: float = 1800) -> np.ndarray:
    """
    Apply CT windowing (HU to intensity mapping)

    Args:
        volume: 3D volume in HU units
        window_center: Window center in HU (200 for pulmonary)
        window_width: Window width in HU (1800 for lung+vessels)

    Returns:
        Windowed volume in [0, 1] range
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2

    # Clip and normalize
    volume_windowed = np.clip(volume, lower, upper)
    volume_windowed = (volume_windowed - lower) / (upper - lower)

    return volume_windowed.astype(np.float32)


def normalize_to_range(volume: np.ndarray,
                       target_min: float = -1.0,
                       target_max: float = 1.0) -> np.ndarray:
    """
    Normalize volume to target range

    Args:
        volume: Input volume (any range)
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized volume in [target_min, target_max]
    """
    v_min = volume.min()
    v_max = volume.max()

    if v_max - v_min < 1e-8:
        return np.full_like(volume, (target_min + target_max) / 2)

    # Normalize to [0, 1]
    volume_norm = (volume - v_min) / (v_max - v_min)

    # Scale to target range
    volume_scaled = volume_norm * (target_max - target_min) + target_min

    return volume_scaled.astype(np.float32)


def extract_3d_patch(volume: np.ndarray,
                     start_slice: int,
                     num_slices: int,
                     spatial_crop: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Extract 3D patch from volume

    Args:
        volume: 3D volume (D, H, W)
        start_slice: Starting slice index
        num_slices: Number of slices to extract
        spatial_crop: Optional (top, left, height, width) for spatial cropping

    Returns:
        Extracted patch (num_slices, height, width)
    """
    # Extract along depth
    end_slice = min(start_slice + num_slices, volume.shape[0])
    patch = volume[start_slice:end_slice]

    # Spatial crop if specified
    if spatial_crop is not None:
        top, left, height, width = spatial_crop
        patch = patch[:, top:top+height, left:left+width]

    return patch


def get_patient_directories(raw_data_dir: str) -> List[str]:
    """
    Get all patient directories from raw data folder

    Args:
        raw_data_dir: Path to raw_data directory

    Returns:
        List of patient directory paths
    """
    patient_dirs = []

    for entry in os.listdir(raw_data_dir):
        path = os.path.join(raw_data_dir, entry)
        if os.path.isdir(path) and not entry.startswith('.'):
            patient_dirs.append(path)

    return sorted(patient_dirs)


if __name__ == "__main__":
    # Test utilities
    raw_data_dir = '/Users/kuntalkokate/Desktop/LLM_agents_projects/raw_data'

    print("Testing DICOM utilities...")

    # Find patients
    patient_dirs = get_patient_directories(raw_data_dir)
    print(f"\nFound {len(patient_dirs)} patients")

    # Test first patient
    if patient_dirs:
        patient_dir = patient_dirs[0]
        print(f"\nTesting with: {os.path.basename(patient_dir)}")

        # Find DICOM files
        dicom_files = find_dicom_files(patient_dir)
        print(f"  Found {len(dicom_files)} DICOM files")

        # Organize by series
        series_dict = organize_by_series(dicom_files)
        print(f"  Found {len(series_dict)} series")

        # Identify thick/thin
        thick_uid, thin_uid = identify_thick_thin_series(series_dict)

        if thick_uid and thin_uid:
            print("\n✓ Successfully identified thick/thin series!")
        else:
            print("\n✗ Failed to identify series")
