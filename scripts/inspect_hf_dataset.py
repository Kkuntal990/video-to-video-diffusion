#!/usr/bin/env python3
"""
Inspect HuggingFace APE Dataset

Check what actually exists on HuggingFace and why we're only seeing 189 cases.
"""

from huggingface_hub import list_repo_files
from pathlib import Path

def inspect_dataset():
    print(f"\n{'='*70}")
    print(f"HuggingFace Dataset Inspection: t2ance/APE-data")
    print(f"{'='*70}\n")

    # Get all files
    print("Fetching file list from HuggingFace...")
    all_files = list(list_repo_files('t2ance/APE-data', repo_type="dataset"))

    # Filter by category
    ape_files = [f for f in all_files if f.startswith('APE/') and f.endswith('.zip')]
    non_ape_files = [f for f in all_files if f.startswith('non-APE/') and f.endswith('.zip')]
    other_files = [f for f in all_files if not f.endswith('.zip')]

    print(f"\n📊 Dataset Contents:")
    print(f"   Total files: {len(all_files)}")
    print(f"   ZIP files: {len(ape_files) + len(non_ape_files)}")
    print(f"   Other files: {len(other_files)}")

    print(f"\n📁 By Category:")
    print(f"   APE/     → {len(ape_files)} ZIP files")
    print(f"   non-APE/ → {len(non_ape_files)} ZIP files")

    # Show case ranges
    def get_case_numbers(file_list):
        numbers = []
        for f in file_list:
            try:
                case_id = Path(f).stem
                num = int(case_id.split('_')[1])
                numbers.append(num)
            except:
                pass
        return sorted(numbers)

    ape_numbers = get_case_numbers(ape_files)
    non_ape_numbers = get_case_numbers(non_ape_files)

    if ape_numbers:
        print(f"\n   APE cases:")
        print(f"      Range: case_{ape_numbers[0]:03d} to case_{ape_numbers[-1]:03d}")
        print(f"      Count: {len(ape_numbers)}")

        # Check for gaps
        expected = set(range(ape_numbers[0], ape_numbers[-1] + 1))
        actual = set(ape_numbers)
        missing = expected - actual
        if missing:
            print(f"      Missing: {len(missing)} cases")
            print(f"      Missing IDs: {sorted(list(missing))[:10]}..." if len(missing) > 10 else f"      Missing IDs: {sorted(list(missing))}")

    if non_ape_numbers:
        print(f"\n   non-APE cases:")
        print(f"      Range: case_{non_ape_numbers[0]:03d} to case_{non_ape_numbers[-1]:03d}")
        print(f"      Count: {len(non_ape_numbers)}")

        # Check for gaps
        expected = set(range(non_ape_numbers[0], non_ape_numbers[-1] + 1))
        actual = set(non_ape_numbers)
        missing = expected - actual
        if missing:
            print(f"      Missing: {len(missing)} cases")
            print(f"      Missing IDs: {sorted(list(missing))[:10]}..." if len(missing) > 10 else f"      Missing IDs: {sorted(list(missing))}")

    # Check overlaps
    if ape_numbers and non_ape_numbers:
        overlap = set(ape_numbers) & set(non_ape_numbers)
        if overlap:
            print(f"\n⚠️  WARNING: {len(overlap)} cases exist in BOTH categories!")
            print(f"   Overlapping cases: {sorted(list(overlap))[:10]}")

    # Corrupted cases blacklist
    corrupted_cases = {
        'case_190', 'case_191', 'case_192', 'case_193', 'case_194',
        'case_195', 'case_196', 'case_197', 'case_198', 'case_199',
        'case_200', 'case_201', 'case_202', 'case_203', 'case_204',
        'case_205', 'case_206'
    }

    corrupted_numbers = [int(c.split('_')[1]) for c in corrupted_cases]

    # Count valid cases
    all_case_numbers = sorted(set(ape_numbers + non_ape_numbers))
    valid_numbers = [n for n in all_case_numbers if f'case_{n:03d}' not in corrupted_cases]

    print(f"\n📈 Summary:")
    print(f"   Total unique cases: {len(all_case_numbers)}")
    print(f"   Corrupted (blacklisted): {len([n for n in all_case_numbers if f'case_{n:03d}' in corrupted_cases])}")
    print(f"   Valid cases: {len(valid_numbers)}")

    print(f"\n🔍 Analysis:")
    if len(all_case_numbers) < 300:
        print(f"   ⚠️  Dataset is smaller than expected!")
        print(f"   Expected: ~500 cases")
        print(f"   Found: {len(all_case_numbers)} cases")
        print(f"\n   This explains why you're only seeing 189 processed cases.")
        print(f"   The HuggingFace dataset may be incomplete or still being uploaded.")
    else:
        print(f"   ✓ Dataset size looks reasonable")

    # Show first few files from each category
    print(f"\n📋 Sample Files:")
    print(f"\n   APE (first 5):")
    for f in sorted(ape_files)[:5]:
        print(f"      {f}")

    print(f"\n   non-APE (first 5):")
    for f in sorted(non_ape_files)[:5]:
        print(f"      {f}")

    print()

if __name__ == "__main__":
    inspect_dataset()
