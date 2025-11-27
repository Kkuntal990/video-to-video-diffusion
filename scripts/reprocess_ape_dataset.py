#!/usr/bin/env python3
"""
Reprocess APE Dataset - Analyze and Fix Preprocessing Failures

This script helps you:
1. Analyze existing preprocessing failures
2. Reprocess failed cases with increased timeout
3. Generate detailed diagnostic reports

Usage:
    # Analyze failures only (no reprocessing)
    python scripts/reprocess_ape_dataset.py --analyze

    # Reprocess failed cases
    python scripts/reprocess_ape_dataset.py --reprocess

    # Force reprocess ALL cases (use with caution!)
    python scripts/reprocess_ape_dataset.py --force-all
"""

import argparse
import yaml
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_unified_dataloader as get_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_failures(cache_dir: Path):
    """Analyze preprocessing failures and print summary"""
    failure_report_path = cache_dir / 'preprocessing_failures.txt'

    print(f"\n{'='*70}")
    print(f"APE Dataset Preprocessing Analysis")
    print(f"{'='*70}\n")

    # Count processed files
    processed_dir = cache_dir / 'processed'
    if not processed_dir.exists():
        print(f"❌ No processed directory found at: {processed_dir}")
        print(f"   Run training first to trigger preprocessing!")
        return

    processed_files = list(processed_dir.glob('*.pt'))
    print(f"✓ Successfully preprocessed: {len(processed_files)} cases")

    # Check for failure report
    if not failure_report_path.exists():
        print(f"\n✓ No failure report found - all cases processed successfully!")
        return

    print(f"\n⚠️  Failure report found: {failure_report_path}")
    print(f"\nReading failure details...\n")

    with open(failure_report_path, 'r') as f:
        content = f.read()
        print(content)

    # Parse failures
    failures = []
    in_failed_section = False
    for line in content.split('\n'):
        if line.startswith('Failed cases:'):
            in_failed_section = True
            continue
        if in_failed_section and ':' in line and not line.startswith('-'):
            case_id = line.split(':')[0].strip()
            error = ':'.join(line.split(':')[1:]).strip()
            failures.append((case_id, error))

    if failures:
        print(f"\n{'='*70}")
        print(f"Recommendations:")
        print(f"{'='*70}")
        print(f"1. Run with --reprocess to retry failed cases with increased timeout")
        print(f"2. Check the error breakdown above to identify common issues")
        print(f"3. Failed cases: {len(failures)}")
        print(f"   Success rate: {len(processed_files)/(len(processed_files)+len(failures))*100:.1f}%")


def reprocess_dataset(config_path: Path, force_all: bool = False):
    """Reprocess the dataset"""
    print(f"\n{'='*70}")
    print(f"APE Dataset Reprocessing")
    print(f"{'='*70}\n")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update config for reprocessing
    if force_all:
        print(f"⚠️  FORCE MODE: Reprocessing ALL cases (this will take hours!)")
        config['data']['force_reprocess'] = True
    else:
        print(f"Reprocessing failed cases only...")
        config['data']['force_reprocess'] = False

    # Create dataloader (this triggers preprocessing)
    print(f"\nInitializing dataset...")
    print(f"Cache directory: {config['data']['cache_dir']}")
    print(f"Timeout: 15 minutes per case")
    print(f"\nThis may take a while...\n")

    try:
        dataloader = get_dataloader(config['data'], split='train')
        print(f"\n✓ Dataset ready!")
        print(f"  Total samples: {len(dataloader.dataset)}")

        # Check failure report
        cache_dir = Path(config['data']['cache_dir'])
        failure_report_path = cache_dir / 'preprocessing_failures.txt'

        if failure_report_path.exists():
            print(f"\n  See failure report: {failure_report_path}")

    except Exception as e:
        logger.error(f"Error during reprocessing: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Reprocess APE Dataset and Analyze Failures'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/cloud_train_config_a100.yaml',
        help='Path to training config file'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze failures only (no reprocessing)'
    )
    parser.add_argument(
        '--reprocess',
        action='store_true',
        help='Reprocess failed cases'
    )
    parser.add_argument(
        '--force-all',
        action='store_true',
        help='Force reprocess ALL cases (use with caution!)'
    )

    args = parser.parse_args()

    # Load config to get cache directory
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cache_dir = Path(config['data']['cache_dir'])

    # Execute based on arguments
    if args.analyze:
        analyze_failures(cache_dir)
    elif args.reprocess or args.force_all:
        reprocess_dataset(config_path, force_all=args.force_all)
    else:
        # Default: analyze
        print(f"No action specified. Use --analyze or --reprocess")
        print(f"\nQuick analysis:")
        analyze_failures(cache_dir)


if __name__ == "__main__":
    main()
