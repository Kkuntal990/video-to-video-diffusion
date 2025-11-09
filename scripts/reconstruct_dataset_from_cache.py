#!/usr/bin/env python3
"""
Reconstruct APE Dataset from HuggingFace Cache

This script reconstructs the dataset from the existing HuggingFace cache blobs
using the metadata.json file to map case IDs to categories.
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

def reconstruct_dataset_from_cache():
    # Paths
    cache_dir = Path('/workspace/storage_a100/ape_cache')
    blobs_dir = cache_dir / 'raw' / 'datasets--t2ance--APE-data' / 'blobs'
    metadata_file = cache_dir / 'metadata.json'
    output_dir = Path('/workspace/storage_a100/dataset')

    # Create output directories
    ape_dir = output_dir / 'APE'
    non_ape_dir = output_dir / 'non-APE'
    ape_dir.mkdir(parents=True, exist_ok=True)
    non_ape_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading metadata from: {metadata_file}")

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"Found {len(metadata)} cases in metadata")
    print(f"Cache blobs directory: {blobs_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Get all blob files
    blob_files = list(blobs_dir.glob('*'))
    print(f"Found {len(blob_files)} files in blobs directory")
    print()

    # Process each case in metadata
    stats = {'total': 0, 'ape': 0, 'non_ape': 0, 'found': 0, 'not_found': 0}

    print("Reconstructing dataset...")
    print("=" * 70)

    for case_id, category in tqdm(metadata.items(), desc="Processing"):
        stats['total'] += 1

        # Determine target directory
        if 'APE' in category or category == 'APE':
            target_dir = ape_dir
            stats['ape'] += 1
        else:
            target_dir = non_ape_dir
            stats['non_ape'] += 1

        # Target filename
        target_file = target_dir / f'{case_id}.zip'

        # Skip if already exists
        if target_file.exists():
            file_size_mb = target_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 1:  # Valid zip
                stats['found'] += 1
                continue

        # Search for this case in blobs (need to find by content inspection)
        # Since blobs are named by hash, we need to look at snapshot references
        # For now, try to find .zip files in the snapshot directory
        snapshot_dir = cache_dir / 'raw' / 'datasets--t2ance--APE-data' / 'snapshots'

        # Try to find the file in snapshots
        found = False
        for snapshot in snapshot_dir.rglob('*'):
            if snapshot.name == f'{case_id}.zip' and snapshot.is_symlink():
                # Follow symlink
                try:
                    source = snapshot.resolve()
                    if source.exists():
                        shutil.copy2(source, target_file)
                        stats['found'] += 1
                        found = True
                        break
                except Exception as e:
                    pass

        if not found:
            stats['not_found'] += 1

    print("=" * 70)
    print()
    print("✓ Reconstruction complete!")
    print()
    print("Statistics:")
    print(f"  Total cases in metadata: {stats['total']}")
    print(f"  APE cases: {stats['ape']}")
    print(f"  non-APE cases: {stats['non_ape']}")
    print(f"  Found and copied: {stats['found']}")
    print(f"  Not found: {stats['not_found']}")
    print()

    # Verify files on disk
    ape_files = list(ape_dir.glob('*.zip'))
    non_ape_files = list(non_ape_dir.glob('*.zip'))

    print("Files on disk:")
    print(f"  {ape_dir}: {len(ape_files)} files")
    print(f"  {non_ape_dir}: {len(non_ape_files)} files")
    print(f"  Total: {len(ape_files) + len(non_ape_files)} files")

    # Calculate total size
    total_size_bytes = sum(f.stat().st_size for f in ape_files + non_ape_files)
    total_size_gb = total_size_bytes / (1024 ** 3)
    print(f"  Total size: {total_size_gb:.2f} GB")
    print()

if __name__ == '__main__':
    reconstruct_dataset_from_cache()
