"""
Visualize Raw DICOM Data to Understand Super-Resolution Task

This script inspects raw DICOM files to determine:
1. Actual input and ground truth resolutions
2. Pixel spacing and slice thickness
3. Whether there are multiple series (thin-slice vs thick-slice)
4. The exact super-resolution task (spatial upsampling, slice interpolation, or both)
"""

import os
import numpy as np
import pydicom
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def find_dicom_files(root_dir, max_files=None):
    """
    Recursively find all DICOM files in directory

    Args:
        root_dir: Root directory to search
        max_files: Maximum number of files to return (None = all)

    Returns:
        List of DICOM file paths
    """
    dicom_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
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


def organize_by_series(dicom_files):
    """
    Organize DICOM files by SeriesInstanceUID

    Returns:
        dict: {series_uid: [dicom_files]}
    """
    series_dict = defaultdict(list)

    for filepath in dicom_files:
        try:
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
            series_uid = dcm.SeriesInstanceUID
            series_dict[series_uid].append(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return series_dict


def extract_series_metadata(dicom_files):
    """
    Extract metadata from a series of DICOM files

    Returns:
        dict with series metadata
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
            spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            metadata['measured_slice_spacing'] = np.mean(spacings)
        else:
            metadata['measured_slice_spacing'] = None
    except:
        metadata['measured_slice_spacing'] = None

    return metadata


def load_dicom_volume(dicom_files, max_slices=None):
    """
    Load DICOM files as 3D volume

    Returns:
        3D numpy array (slices, height, width)
    """
    # Sort by instance number or slice location
    sorted_files = []
    for filepath in dicom_files:
        dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
        instance_num = getattr(dcm, 'InstanceNumber', 0)
        slice_loc = getattr(dcm, 'SliceLocation', 0)
        sorted_files.append((instance_num, slice_loc, filepath))

    sorted_files.sort(key=lambda x: (x[0], x[1]))

    if max_slices:
        sorted_files = sorted_files[:max_slices]

    # Load pixel data
    slices = []
    for _, _, filepath in sorted_files:
        dcm = pydicom.dcmread(filepath)
        slices.append(dcm.pixel_array)

    return np.stack(slices, axis=0)


def visualize_series_comparison(patient_dir, output_dir='visualizations/raw_dicom'):
    """
    Visualize all series in a patient directory
    """
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(patient_dir)}")
    print(f"{'='*80}")

    # Find all DICOM files
    print("Finding DICOM files...")
    dicom_files = find_dicom_files(patient_dir)
    print(f"Found {len(dicom_files)} DICOM files")

    if not dicom_files:
        print("No DICOM files found!")
        return

    # Organize by series
    print("\nOrganizing by series...")
    series_dict = organize_by_series(dicom_files)
    print(f"Found {len(series_dict)} series")

    # Extract metadata for each series
    series_info = []
    for series_uid, files in series_dict.items():
        metadata = extract_series_metadata(files)
        metadata['series_uid'] = series_uid
        metadata['files'] = files
        series_info.append(metadata)

        print(f"\nSeries {metadata['series_number']}: {metadata['series_description']}")
        print(f"  Slices: {metadata['num_slices']}")
        print(f"  Resolution: {metadata['rows']}x{metadata['columns']}")
        print(f"  Pixel Spacing: {metadata['pixel_spacing']} mm")
        print(f"  Slice Thickness: {metadata['slice_thickness']} mm")
        print(f"  Spacing Between Slices: {metadata['spacing_between_slices']} mm")
        print(f"  Measured Slice Spacing: {metadata['measured_slice_spacing']:.3f} mm" if metadata['measured_slice_spacing'] else "  Measured Slice Spacing: N/A")

    # Sort by series number
    series_info.sort(key=lambda x: x['series_number'] if x['series_number'] != 'Unknown' else 999)

    # Identify potential input/ground truth pairs
    print(f"\n{'='*80}")
    print("ANALYSIS: Identifying Super-Resolution Task")
    print(f"{'='*80}")

    # Check for spatial resolution differences
    resolutions = [(s['rows'], s['columns']) for s in series_info]
    unique_resolutions = set(resolutions)

    print(f"\nUnique spatial resolutions found: {unique_resolutions}")

    if len(unique_resolutions) > 1:
        print("✓ Multiple spatial resolutions detected!")
        print("  → Task likely includes SPATIAL super-resolution")
        low_res = min(unique_resolutions, key=lambda x: x[0]*x[1])
        high_res = max(unique_resolutions, key=lambda x: x[0]*x[1])
        print(f"  → Low-res: {low_res[0]}×{low_res[1]}")
        print(f"  → High-res: {high_res[0]}×{high_res[1]}")
    else:
        print("✗ All series have same spatial resolution")
        print(f"  → Spatial resolution: {resolutions[0][0]}×{resolutions[0][1]}")

    # Check for slice thickness differences
    slice_thicknesses = [s['slice_thickness'] for s in series_info if s['slice_thickness'] is not None]
    unique_thicknesses = set(slice_thicknesses)

    print(f"\nUnique slice thicknesses found: {unique_thicknesses}")

    if len(unique_thicknesses) > 1:
        print("✓ Multiple slice thicknesses detected!")
        print("  → Task likely includes SLICE INTERPOLATION")
        thin = min(unique_thicknesses)
        thick = max(unique_thicknesses)
        print(f"  → Thin slices: {thin} mm")
        print(f"  → Thick slices: {thick} mm")
    else:
        print("✗ All series have same slice thickness")
        if slice_thicknesses:
            print(f"  → Slice thickness: {slice_thicknesses[0]} mm")

    # Visualize samples from each series
    os.makedirs(output_dir, exist_ok=True)
    patient_name = os.path.basename(patient_dir)

    print(f"\n{'='*80}")
    print("Creating Visualizations")
    print(f"{'='*80}")

    for i, series in enumerate(series_info):
        print(f"\nLoading series {series['series_number']} ({series['num_slices']} slices)...")

        # Load up to 32 slices
        volume = load_dicom_volume(series['files'], max_slices=32)

        # Normalize for visualization
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        # Create multi-slice visualization
        num_display = min(16, volume.shape[0])
        step = max(1, volume.shape[0] // num_display)

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle(f"Series {series['series_number']}: {series['series_description']}\n"
                    f"Resolution: {series['rows']}×{series['columns']}, "
                    f"Slices: {series['num_slices']}, "
                    f"Thickness: {series['slice_thickness']} mm",
                    fontsize=14)

        for idx, ax in enumerate(axes.flat):
            slice_idx = idx * step
            if slice_idx < volume.shape[0]:
                ax.imshow(volume[slice_idx], cmap='gray')
                ax.set_title(f"Slice {slice_idx}")
                ax.axis('off')
            else:
                ax.axis('off')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{patient_name}_series_{series['series_number']}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")

    # Create comparison visualization if multiple series
    if len(series_info) >= 2:
        print("\nCreating series comparison...")

        fig, axes = plt.subplots(len(series_info), 4, figsize=(16, 4*len(series_info)))
        if len(series_info) == 1:
            axes = axes.reshape(1, -1)

        for series_idx, series in enumerate(series_info):
            volume = load_dicom_volume(series['files'], max_slices=16)
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

            # Show 4 evenly spaced slices
            slice_indices = [volume.shape[0]//5, 2*volume.shape[0]//5, 3*volume.shape[0]//5, 4*volume.shape[0]//5]

            for col_idx, slice_idx in enumerate(slice_indices):
                if slice_idx < volume.shape[0]:
                    axes[series_idx, col_idx].imshow(volume[slice_idx], cmap='gray')
                    axes[series_idx, col_idx].set_title(
                        f"Series {series['series_number']}, Slice {slice_idx}\n"
                        f"{series['rows']}×{series['columns']}, {series['slice_thickness']}mm"
                    )
                    axes[series_idx, col_idx].axis('off')

        plt.suptitle(f"Series Comparison: {patient_name}", fontsize=16)
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{patient_name}_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")

    return series_info


def main():
    """Main execution"""
    raw_data_dir = '/Users/kuntalkokate/Desktop/LLM_agents_projects/raw_data'
    output_dir = '/Users/kuntalkokate/Desktop/LLM_agents_projects/LLM_agent_v2v/visualizations/raw_dicom'

    print("="*80)
    print("RAW DICOM DATA VISUALIZATION AND ANALYSIS")
    print("="*80)

    # Find all patient directories
    patient_dirs = [
        os.path.join(raw_data_dir, d)
        for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d)) and not d.startswith('.')
    ]

    print(f"\nFound {len(patient_dirs)} patient directories:")
    for pdir in patient_dirs:
        print(f"  - {os.path.basename(pdir)}")

    # Process each patient
    all_series_info = []
    for patient_dir in patient_dirs:
        series_info = visualize_series_comparison(patient_dir, output_dir)
        if series_info:
            all_series_info.extend(series_info)

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    all_resolutions = set((s['rows'], s['columns']) for s in all_series_info)
    all_thicknesses = set(s['slice_thickness'] for s in all_series_info if s['slice_thickness'] is not None)
    all_slice_counts = [s['num_slices'] for s in all_series_info]

    print(f"\nTotal series analyzed: {len(all_series_info)}")
    print(f"Spatial resolutions: {sorted(all_resolutions)}")
    print(f"Slice thicknesses: {sorted(all_thicknesses)} mm")
    print(f"Slice counts: min={min(all_slice_counts)}, max={max(all_slice_counts)}, mean={np.mean(all_slice_counts):.1f}")

    print(f"\n{'='*80}")
    print("RECOMMENDED SUPER-RESOLUTION CONFIGURATION")
    print(f"{'='*80}")

    if len(all_resolutions) > 1:
        low_res = min(all_resolutions, key=lambda x: x[0]*x[1])
        high_res = max(all_resolutions, key=lambda x: x[0]*x[1])
        print(f"\n✓ Spatial Super-Resolution Task:")
        print(f"  Input: {low_res[0]}×{low_res[1]}")
        print(f"  Output: {high_res[0]}×{high_res[1]}")
        print(f"  Upsampling Factor: {high_res[0]//low_res[0]}×")

    if len(all_thicknesses) > 1:
        thin = min(all_thicknesses)
        thick = max(all_thicknesses)
        print(f"\n✓ Slice Interpolation Task:")
        print(f"  Input thickness: {thick} mm")
        print(f"  Output thickness: {thin} mm")
        print(f"  Interpolation Factor: {thick/thin:.1f}×")

    print(f"\nVisualizations saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review visualizations to confirm input/ground truth pairs")
    print("2. Update model architecture for identified super-resolution task")
    print("3. Configure data loading pipeline for multi-series processing")


if __name__ == "__main__":
    main()
