#!/usr/bin/env python3
"""
Test preprocessing on local sample files
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.ape_cached_dataset import APECachedDataset

def test_preprocessing():
    """Test preprocessing on two sample files"""

    print("=" * 70)
    print("Testing APE Dataset Preprocessing")
    print("=" * 70)

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_cache:
        print(f"\nUsing temporary cache: {temp_cache}")

        # Configuration
        config = {
            'dataset_name': 't2ance/APE-data',
            'cache_dir': temp_cache,
            'categories': ['APE', 'non APE'],
            'num_frames': 24,
            'resolution': (256, 256),
            'window_center': 40,
            'window_width': 400,
            'force_reprocess': True
        }

        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # Copy sample files to cache for testing
        print("\n" + "=" * 70)
        print("Copying sample files to cache...")
        print("=" * 70)

        raw_dir = Path('./dataset/raw')
        cache_path = Path(temp_cache)
        extracted_dir = cache_path / 'extracted'
        extracted_dir.mkdir(parents=True, exist_ok=True)

        # Map case files
        test_cases = [
            ('case_002.zip', 'APE', 'case_002'),
            ('case_211.zip', 'non APE', 'case_211')
        ]

        for zip_file, category, case_id in test_cases:
            src_zip = raw_dir / zip_file
            if src_zip.exists():
                print(f"\nProcessing: {zip_file} (category: {category})")

                # Extract to temporary location
                extract_path = extracted_dir / case_id
                extract_path.mkdir(parents=True, exist_ok=True)

                import zipfile
                print(f"  Extracting to: {extract_path}")
                try:
                    with zipfile.ZipFile(src_zip, 'r') as zf:
                        zf.extractall(extract_path)
                    print(f"  ✓ Extracted successfully")

                    # List extracted contents
                    print(f"  Contents:")
                    for item in extract_path.iterdir():
                        print(f"    - {item.name}")
                        if item.is_dir():
                            for subitem in item.iterdir():
                                print(f"      - {subitem.name}")
                                if subitem.is_dir():
                                    dcm_count = len(list(subitem.rglob('*')))
                                    print(f"        ({dcm_count} files)")
                                    break
                                break

                except Exception as e:
                    print(f"  ✗ Extraction failed: {e}")
                    continue

        # Now test the preprocessing pipeline
        print("\n" + "=" * 70)
        print("Running Preprocessing Pipeline")
        print("=" * 70)

        try:
            # Initialize dataset (this will trigger preprocessing)
            dataset = APECachedDataset(
                dataset_name='t2ance/APE-data',
                cache_dir=temp_cache,
                num_frames=config['num_frames'],
                resolution=config['resolution'],
                categories=config['categories'],
                window_center=config['window_center'],
                window_width=config['window_width'],
                force_reprocess=config['force_reprocess'],
                split='train'
            )

            print(f"\n✓ Dataset created successfully")
            print(f"  Total samples: {len(dataset)}")

            # Try loading a sample
            if len(dataset) > 0:
                print("\n" + "=" * 70)
                print("Testing Sample Loading")
                print("=" * 70)

                sample = dataset[0]
                print(f"\nLoaded sample 0:")
                print(f"  Input shape: {sample['input'].shape}")
                print(f"  Target shape: {sample['target'].shape}")
                print(f"  Category: {sample['category']}")
                print(f"  Patient ID: {sample['patient_id']}")
                print(f"  Input range: [{sample['input'].min():.3f}, {sample['input'].max():.3f}]")
                print(f"  Target range: [{sample['target'].min():.3f}, {sample['target'].max():.3f}]")

        except Exception as e:
            print(f"\n✗ Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_preprocessing()
    sys.exit(0 if success else 1)
