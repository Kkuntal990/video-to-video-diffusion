#!/usr/bin/env python3
"""
Direct test of preprocessing on the two local sample files
"""

import sys
import tempfile
import zipfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_direct_preprocessing():
    """Test preprocessing directly on extracted local files"""

    print("=" * 70)
    print("Direct Preprocessing Test on Local Samples")
    print("=" * 70)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        extracted_dir = temp_path / 'extracted'
        extracted_dir.mkdir()

        # Extract sample files
        test_cases = [
            ('case_002.zip', 'APE'),
            ('case_211.zip', 'non APE')
        ]

        raw_dir = Path('./dataset/raw')

        print("\n" + "=" * 70)
        print("Step 1: Extracting Files")
        print("=" * 70)

        for zip_file, category in test_cases:
            src_zip = raw_dir / zip_file
            case_id = zip_file.replace('.zip', '')
            extract_path = extracted_dir / case_id

            print(f"\n{case_id} ({category}):")
            extract_path.mkdir()

            with zipfile.ZipFile(src_zip, 'r') as zf:
                zf.extractall(extract_path)

            # Show structure
            subdirs = list(extract_path.iterdir())
            print(f"  Extracted {len(list(extract_path.rglob('*')))} files")
            print(f"  Root directory: {subdirs[0].name if subdirs else 'None'}")

            # Look deeper
            if subdirs:
                inner = list(subdirs[0].iterdir())
                print(f"  Inner directories: {[d.name for d in inner if d.is_dir()]}")

        # Now call preprocessing
        print("\n" + "=" * 70)
        print("Step 2: Running Preprocessing")
        print("=" * 70)

        # Create a minimal APECachedDataset instance
        from data.ape_cached_dataset import APECachedDataset
        from data.transforms import VideoTransform

        # Create dataset object with minimal initialization
        dataset = APECachedDataset.__new__(APECachedDataset)
        dataset.cache_dir = temp_path
        dataset.extracted_dir = extracted_dir
        dataset.num_frames = 24
        dataset.resolution = (256, 256)
        dataset.window_center = 40
        dataset.window_width = 400
        dataset.transform = VideoTransform(resolution=(256, 256), num_frames=24, normalize=False)

        # Import logging
        import logging
        logging.basicConfig(level=logging.INFO)

        # Test preprocessing on each case
        for zip_file, category in test_cases:
            case_id = zip_file.replace('.zip', '')
            case_dir = extracted_dir / case_id

            print(f"\n{'='*70}")
            print(f"Testing: {case_id} ({category})")
            print(f"{'='*70}")

            try:
                # Call the preprocessing method
                result = dataset._load_and_preprocess_case(case_dir, category, case_id)

                if result is None:
                    print(f"\n✗ Preprocessing returned None")
                    print(f"  This means the data structure doesn't match what's expected!")
                else:
                    print(f"\n✓ Preprocessing succeeded!")
                    print(f"  Input shape: {result['input'].shape}")
                    print(f"  Target shape: {result['target'].shape}")
                    print(f"  Category: {result['category']}")
                    print(f"  Patient ID: {result['patient_id']}")
                    print(f"  Input range: [{result['input'].min():.3f}, {result['input'].max():.3f}]")
                    print(f"  Target range: [{result['target'].min():.3f}, {result['target'].max():.3f}]")

            except Exception as e:
                print(f"\n✗ Preprocessing failed with error:")
                print(f"  {type(e).__name__}: {e}")
                print(f"\n  Full traceback:")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_direct_preprocessing()
