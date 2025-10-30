#!/usr/bin/env python3
"""
Test preprocessing directly on the two local sample files
"""

import sys
import tempfile
import zipfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the actual dataset class
from data.ape_cached_dataset import APECachedDataset

def test_local_samples():
    """Test preprocessing on local sample files"""

    print("=" * 70)
    print("Testing Preprocessing on Local Samples")
    print("=" * 70)

    # Create a temporary cache directory
    with tempfile.TemporaryDirectory() as temp_cache:
        cache_path = Path(temp_cache)
        extracted_dir = cache_path / 'extracted'
        extracted_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nTemporary cache: {cache_path}")

        # Extract the two sample files
        raw_dir = Path('./dataset/raw')
        test_cases = [
            ('case_002.zip', 'APE'),
            ('case_211.zip', 'non APE')
        ]

        print("\n" + "=" * 70)
        print("Step 1: Extracting sample files")
        print("=" * 70)

        for zip_file, category in test_cases:
            src_zip = raw_dir / zip_file
            case_id = zip_file.replace('.zip', '')

            if not src_zip.exists():
                print(f"✗ File not found: {src_zip}")
                continue

            extract_path = extracted_dir / case_id

            print(f"\n{case_id} ({category}):")
            print(f"  Source: {src_zip}")
            print(f"  Extracting to: {extract_path}")

            try:
                extract_path.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(src_zip, 'r') as zf:
                    zf.extractall(extract_path)

                # Count extracted files
                all_files = list(extract_path.rglob('*'))
                print(f"  ✓ Extracted {len(all_files)} files/directories")

                # Show structure
                subdirs = [d for d in extract_path.iterdir() if d.is_dir()]
                if subdirs:
                    print(f"  Top-level directories:")
                    for d in subdirs:
                        print(f"    - {d.name}/")
                        # Look one level deeper
                        inner = [x for x in d.iterdir() if x.is_dir()]
                        if inner:
                            for i in inner[:3]:  # Show first 3
                                print(f"      - {i.name}/")

            except Exception as e:
                print(f"  ✗ Extraction failed: {e}")
                continue

        # Now instantiate the dataset class - it should use the extracted files
        print("\n" + "=" * 70)
        print("Step 2: Running APECachedDataset preprocessing")
        print("=" * 70)
        print("\n⚠️  NOTE: The dataset class tries to download from HuggingFace.")
        print("We'll let it detect the files, but it won't use our local extracts directly.")
        print("This is showing the issue with the preprocessing!")

        # Let's directly call the preprocessing method on our extracted files
        print("\n" + "=" * 70)
        print("Step 3: Direct preprocessing test")
        print("=" * 70)

        # Create a minimal dataset instance just to use its preprocessing methods
        from data.ape_cached_dataset import APECachedDataset
        import torch
        import numpy as np
        import pydicom

        print("\nManually calling preprocessing methods...")

        # Create dataset instance (this will fail but we can still use methods)
        try:
            # Use a fake dataset name to prevent downloads
            dataset = APECachedDataset.__new__(APECachedDataset)
            dataset.cache_dir = cache_path
            dataset.extracted_dir = extracted_dir
            dataset.processed_dir = cache_path / 'processed'
            dataset.processed_dir.mkdir(exist_ok=True)
            dataset.num_frames = 24
            dataset.resolution = (256, 256)
            dataset.window_center = 40
            dataset.window_width = 400
            dataset.force_reprocess = True
            dataset.dataset_name = 'fake'  # Prevent HF access

            # Import transforms
            from torchvision import transforms
            from data.ape_cached_dataset import VideoResizeTransform

            dataset.transform = transforms.Compose([
                VideoResizeTransform(dataset.resolution),
                transforms.Lambda(lambda x: torch.from_numpy(x).float()),
                transforms.Lambda(lambda x: x.permute(3, 0, 1, 2) / 255.0),
            ])

            print("✓ Dataset instance created for testing")

            # Now test preprocessing on each extracted case
            for zip_file, category in test_cases:
                case_id = zip_file.replace('.zip', '')
                case_dir = extracted_dir / case_id

                print(f"\n{'='*70}")
                print(f"Testing: {case_id} ({category})")
                print(f"{'='*70}")

                try:
                    # Call the actual preprocessing method
                    result = dataset._load_and_preprocess_case(case_dir, category, case_id)

                    if result is None:
                        print(f"✗ Preprocessing returned None")
                    else:
                        print(f"✓ Preprocessing succeeded!")
                        print(f"  Input shape: {result['input'].shape}")
                        print(f"  Target shape: {result['target'].shape}")
                        print(f"  Category: {result['category']}")
                        print(f"  Patient ID: {result['patient_id']}")
                        print(f"  Input range: [{result['input'].min():.3f}, {result['input'].max():.3f}]")
                        print(f"  Target range: [{result['target'].min():.3f}, {result['target'].max():.3f}]")

                        # Save to check
                        output_file = cache_path / f"{case_id}_test.pt"
                        torch.save(result, output_file)
                        print(f"  Saved to: {output_file}")

                except Exception as e:
                    print(f"✗ Preprocessing failed: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"✗ Failed to create dataset instance: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_local_samples()
