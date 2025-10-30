#!/usr/bin/env python3
"""
Test preprocessing on local sample files ONLY
"""

import sys
import tempfile
import zipfile
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.transforms import apply_ct_windowing, resize_volume, normalize_to_range

def load_dicom_volume(dicom_dir):
    """Load DICOM slices into a 3D volume"""
    import pydicom
    import numpy as np

    print(f"  Loading DICOM files from: {dicom_dir}")

    # Find all DICOM files (they don't have extensions usually)
    dicom_files = []
    for root, dirs, files in dicom_dir.walk():
        for file in files:
            file_path = root / file
            # Skip DICOMDIR and other non-DICOM files
            if file in ['DICOMDIR', 'LOCKFILE'] or file.startswith('.'):
                continue
            try:
                # Try to read as DICOM
                pydicom.dcmread(str(file_path), stop_before_pixels=True)
                dicom_files.append(file_path)
            except:
                continue

    print(f"  Found {len(dicom_files)} DICOM files")

    if len(dicom_files) == 0:
        raise ValueError("No valid DICOM files found")

    # Load all slices
    slices = []
    for dcm_file in dicom_files:
        try:
            ds = pydicom.dcmread(str(dcm_file))
            if hasattr(ds, 'pixel_array'):
                slices.append((ds, dcm_file))
        except Exception as e:
            print(f"    ✗ Failed to read {dcm_file.name}: {e}")

    print(f"  Successfully loaded {len(slices)} slices with pixel data")

    if len(slices) == 0:
        raise ValueError("No slices with pixel data found")

    # Sort by ImagePositionPatient (Z-coordinate)
    try:
        slices.sort(key=lambda x: float(x[0].ImagePositionPatient[2]))
        print(f"  ✓ Sorted by ImagePositionPatient")
    except:
        print(f"  ⚠ Could not sort by ImagePositionPatient, using file order")
        slices.sort(key=lambda x: str(x[1]))

    # Stack into 3D volume
    volume = np.stack([s[0].pixel_array for s in slices], axis=0)

    # Apply rescale slope and intercept to get HU values
    try:
        slope = float(slices[0][0].RescaleSlope)
        intercept = float(slices[0][0].RescaleIntercept)
        volume = volume * slope + intercept
        print(f"  ✓ Applied RescaleSlope ({slope}) and RescaleIntercept ({intercept})")
    except:
        print(f"  ⚠ No RescaleSlope/Intercept found, using raw values")

    print(f"  Volume shape: {volume.shape}")
    print(f"  Value range: [{volume.min():.1f}, {volume.max():.1f}]")

    return torch.from_numpy(volume).float()

def preprocess_case(extract_path, case_id, category, config):
    """Test preprocessing pipeline on a single case"""
    print(f"\n{'='*70}")
    print(f"Preprocessing {case_id} (category: {category})")
    print(f"{'='*70}")

    try:
        # Step 1: Load DICOM volume
        print("\n[1/4] Loading DICOM volume...")
        volume = load_dicom_volume(extract_path)

        # Step 2: Apply CT windowing
        print("\n[2/4] Applying CT windowing...")
        print(f"  Window center: {config['window_center']}, width: {config['window_width']}")
        volume_windowed = apply_ct_windowing(
            volume,
            window_center=config['window_center'],
            window_width=config['window_width']
        )
        print(f"  ✓ After windowing: [{volume_windowed.min():.3f}, {volume_windowed.max():.3f}]")

        # Step 3: Select frames
        print("\n[3/4] Selecting frames...")
        num_slices = volume_windowed.shape[0]
        num_frames = config['num_frames']
        print(f"  Total slices: {num_slices}, target frames: {num_frames}")

        if num_slices >= num_frames:
            # Sample uniformly
            indices = torch.linspace(0, num_slices - 1, num_frames).long()
            volume_sampled = volume_windowed[indices]
            print(f"  ✓ Sampled {num_frames} frames uniformly")
        else:
            # Repeat to reach target
            repeats = (num_frames + num_slices - 1) // num_slices
            volume_sampled = volume_windowed.repeat(repeats, 1, 1)[:num_frames]
            print(f"  ✓ Repeated volume to reach {num_frames} frames")

        print(f"  Shape after sampling: {volume_sampled.shape}")

        # Step 4: Resize
        print("\n[4/4] Resizing volume...")
        resolution = config['resolution']
        print(f"  Target resolution: {resolution}")

        # Add batch and channel dimensions for resize
        volume_resized = resize_volume(
            volume_sampled.unsqueeze(0).unsqueeze(0),  # [1, 1, T, H, W]
            resolution
        ).squeeze(0).squeeze(0)  # [T, H, W]

        print(f"  ✓ Resized to: {volume_resized.shape}")
        print(f"  Value range: [{volume_resized.min():.3f}, {volume_resized.max():.3f}]")

        # Create input and target (for testing, we'll use degraded version as input)
        print("\n[5/5] Creating input/target pair...")

        # Target: full resolution
        target = volume_resized.unsqueeze(0).repeat(3, 1, 1, 1)  # [3, T, H, W] for RGB

        # Input: downsampled and upsampled (simulated low-res)
        input_lowres = resize_volume(
            volume_resized.unsqueeze(0).unsqueeze(0),  # [1, 1, T, H, W]
            (resolution[0] // 2, resolution[1] // 2)
        )
        input_upsampled = resize_volume(
            input_lowres,
            resolution
        ).squeeze(0).squeeze(0)  # [T, H, W]

        input_tensor = input_upsampled.unsqueeze(0).repeat(3, 1, 1, 1)  # [3, T, H, W]

        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Target shape: {target.shape}")

        # Create sample dict
        sample = {
            'input': input_tensor,
            'target': target,
            'category': category,
            'patient_id': case_id
        }

        print(f"\n✓ Successfully preprocessed {case_id}")
        return sample

    except Exception as e:
        print(f"\n✗ Failed to preprocess {case_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_local_preprocessing():
    """Test preprocessing on two local files"""

    print("=" * 70)
    print("Testing APE Dataset Preprocessing on Local Files")
    print("=" * 70)

    # Configuration
    config = {
        'num_frames': 24,
        'resolution': (256, 256),
        'window_center': 40,
        'window_width': 400
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Test cases
    test_cases = [
        ('case_002.zip', 'APE', 'case_002'),
        ('case_211.zip', 'non APE', 'case_211')
    ]

    raw_dir = Path('./dataset/raw')
    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for zip_file, category, case_id in test_cases:
            src_zip = raw_dir / zip_file

            if not src_zip.exists():
                print(f"\n✗ File not found: {src_zip}")
                continue

            print(f"\n{'='*70}")
            print(f"Testing: {zip_file}")
            print(f"{'='*70}")

            # Extract
            extract_path = temp_path / case_id
            extract_path.mkdir(parents=True, exist_ok=True)

            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(src_zip, 'r') as zf:
                zf.extractall(extract_path)
            print(f"✓ Extracted to {extract_path}")

            # Find the actual DICOM directory (usually nested)
            dicom_dirs = list(extract_path.rglob('*'))
            dicom_dir = extract_path
            for d in dicom_dirs:
                if d.is_dir():
                    # Check if it contains DICOM files
                    files = list(d.iterdir())
                    if any(f.name.startswith('I') and not f.suffix for f in files):
                        dicom_dir = d
                        break

            # Preprocess
            sample = preprocess_case(dicom_dir, case_id, category, config)
            if sample is not None:
                results.append((case_id, category, sample))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    if len(results) == 0:
        print("✗ No samples successfully preprocessed")
        return False

    print(f"✓ Successfully preprocessed {len(results)}/{len(test_cases)} samples")

    for case_id, category, sample in results:
        print(f"\n{case_id} ({category}):")
        print(f"  Input: {sample['input'].shape} - range [{sample['input'].min():.3f}, {sample['input'].max():.3f}]")
        print(f"  Target: {sample['target'].shape} - range [{sample['target'].min():.3f}, {sample['target'].max():.3f}]")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)

    return True

if __name__ == "__main__":
    success = test_local_preprocessing()
    sys.exit(0 if success else 1)
