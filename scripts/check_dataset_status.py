#!/usr/bin/env python3
"""
Check APE Dataset Status - Quick Diagnostic

This script checks:
1. How many .pt files exist (successfully preprocessed)
2. How many cases should exist (from HuggingFace)
3. Which categories are represented
4. Detailed breakdown by category
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from huggingface_hub import list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  huggingface_hub not available - install with: pip install huggingface_hub")

def check_dataset_status(cache_dir: str):
    """Check current dataset preprocessing status"""
    cache_path = Path(cache_dir)
    processed_dir = cache_path / 'processed'

    print(f"\n{'='*70}")
    print(f"APE Dataset Status Check")
    print(f"{'='*70}\n")

    print(f"Cache directory: {cache_dir}")

    # Check if processed directory exists
    if not processed_dir.exists():
        print(f"\n❌ No processed directory found at: {processed_dir}")
        print(f"   Dataset has not been preprocessed yet!")
        return

    # Count .pt files
    pt_files = list(processed_dir.glob('*.pt'))
    print(f"\n✅ Processed directory exists")
    print(f"   Location: {processed_dir}")
    print(f"   Preprocessed cases: {len(pt_files)}")

    # Analyze case IDs
    case_ids = [f.stem for f in pt_files]
    case_numbers = []
    for case_id in case_ids:
        try:
            # Extract number from case_XXX
            num = int(case_id.split('_')[1])
            case_numbers.append(num)
        except:
            pass

    if case_numbers:
        case_numbers.sort()
        print(f"   Case range: {min(case_numbers)} - {max(case_numbers)}")

    # Check what should exist on HuggingFace
    if HF_AVAILABLE:
        print(f"\n📊 Checking HuggingFace dataset...")
        try:
            all_files = list(list_repo_files('t2ance/APE-data', repo_type="dataset"))

            # Count by category
            ape_files = [f for f in all_files if f.startswith('APE/') and f.endswith('.zip')]
            non_ape_files = [f for f in all_files if f.startswith('non-APE/') and f.endswith('.zip')]

            # Corrupted cases blacklist
            corrupted_cases = {
                'case_190', 'case_191', 'case_192', 'case_193', 'case_194',
                'case_195', 'case_196', 'case_197', 'case_198', 'case_199',
                'case_200', 'case_201', 'case_202', 'case_203', 'case_204',
                'case_205', 'case_206'
            }

            ape_valid = [f for f in ape_files if Path(f).stem not in corrupted_cases]
            non_ape_valid = [f for f in non_ape_files if Path(f).stem not in corrupted_cases]

            print(f"\n   HuggingFace dataset contents:")
            print(f"   ├─ APE category:")
            print(f"   │  ├─ Total ZIP files: {len(ape_files)}")
            print(f"   │  └─ Valid (non-corrupted): {len(ape_valid)}")
            print(f"   ├─ non-APE category:")
            print(f"   │  ├─ Total ZIP files: {len(non_ape_files)}")
            print(f"   │  └─ Valid (non-corrupted): {len(non_ape_valid)}")
            print(f"   └─ Total valid cases: {len(ape_valid) + len(non_ape_valid)}")

            total_expected = len(ape_valid) + len(non_ape_valid)
            success_rate = (len(pt_files) / total_expected * 100) if total_expected > 0 else 0

            print(f"\n📈 Preprocessing Success Rate:")
            print(f"   Processed: {len(pt_files)}/{total_expected} ({success_rate:.1f}%)")
            print(f"   Missing: {total_expected - len(pt_files)} cases")

            if success_rate < 50:
                print(f"\n⚠️  WARNING: Less than 50% success rate!")
                print(f"   This is unusually low and should be investigated.")
                print(f"   Run: python scripts/reprocess_ape_dataset.py --analyze")

        except Exception as e:
            print(f"   ❌ Error checking HuggingFace: {e}")

    # Check for failure report
    failure_report = cache_path / 'preprocessing_failures.txt'
    if failure_report.exists():
        print(f"\n📄 Failure report found:")
        print(f"   {failure_report}")
        print(f"\n   First 20 lines:")
        with open(failure_report, 'r') as f:
            lines = f.readlines()[:20]
            for line in lines:
                print(f"   {line.rstrip()}")

    # Recommendations
    print(f"\n{'='*70}")
    print(f"Recommendations:")
    print(f"{'='*70}")

    if len(pt_files) < 400:
        print(f"1. Your dataset has fewer cases than expected")
        print(f"   Run: python scripts/reprocess_ape_dataset.py --analyze")
        print(f"   Then: python scripts/reprocess_ape_dataset.py --reprocess")
    else:
        print(f"1. ✓ Dataset looks good! {len(pt_files)} cases ready for training")

    print(f"\n2. To see detailed breakdown:")
    print(f"   python scripts/reprocess_ape_dataset.py --analyze")

    print(f"\n3. To reprocess failed cases:")
    print(f"   python scripts/reprocess_ape_dataset.py --reprocess")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Check APE dataset status')
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='/workspace/storage_a100/ape_cache',
        help='Cache directory path'
    )
    args = parser.parse_args()

    check_dataset_status(args.cache_dir)
