#!/usr/bin/env python3
"""
Download APE Dataset from HuggingFace

This script downloads the complete APE dataset (.zip files) from HuggingFace
and saves them to the local storage, preserving the category structure.

Usage:
    python scripts/download_ape_dataset.py --output_dir /workspace/storage_a100/dataset

The script will create:
    /workspace/storage_a100/dataset/APE/*.zip
    /workspace/storage_a100/dataset/non-APE/*.zip
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import logging

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub library not installed. Please run: pip install huggingface_hub")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_ape_dataset(output_dir: str = '/workspace/storage_a100/dataset',
                         dataset_name: str = 't2ance/APE-data',
                         repo_type: str = 'dataset'):
    """
    Download APE dataset .zip files from HuggingFace repository

    Args:
        output_dir: Output directory for .zip files
        dataset_name: HuggingFace dataset name
        repo_type: Type of repository ('dataset' or 'model')
    """

    output_dir = Path(output_dir)
    ape_dir = output_dir / 'APE'
    non_ape_dir = output_dir / 'non-APE'

    # Create directories
    ape_dir.mkdir(parents=True, exist_ok=True)
    non_ape_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dataset: {dataset_name} (type: {repo_type})")
    logger.info("")

    # Get HF token from environment (for gated datasets)
    hf_token = os.environ.get('HF_TOKEN', None)
    if hf_token:
        logger.info("Using HuggingFace authentication token")
    else:
        logger.warning("No HF_TOKEN found - this may fail for gated datasets")

    # Initialize HuggingFace API
    logger.info("Connecting to HuggingFace Hub...")
    api = HfApi()

    try:
        # List all files in the repository
        logger.info("Fetching file list from repository...")
        files = api.list_repo_files(
            repo_id=dataset_name,
            repo_type=repo_type,
            token=hf_token
        )

        # Filter for .zip files
        zip_files = [f for f in files if f.endswith('.zip')]

        logger.info(f"Found {len(zip_files)} .zip files in repository")
        logger.info("")

    except Exception as e:
        logger.error(f"Failed to list repository files: {e}")
        logger.error("Please check your internet connection and dataset name")
        return

    # Download statistics
    stats = {
        'total': 0,
        'ape': 0,
        'non_ape': 0,
        'skipped': 0,
        'failed': 0
    }

    # Download files
    logger.info("Starting download...")
    logger.info("=" * 70)

    for file_path in tqdm(zip_files, desc="Downloading"):
        stats['total'] += 1

        # Extract filename
        filename = Path(file_path).name

        # Determine category from path
        if 'APE/' in file_path and 'non' not in file_path.lower():
            save_dir = ape_dir
            category_label = 'APE'
            stats['ape'] += 1
        elif 'non APE/' in file_path or 'non-APE/' in file_path:
            save_dir = non_ape_dir
            category_label = 'non-APE'
            stats['non_ape'] += 1
        else:
            logger.warning(f"  [{stats['total']}] Skipping {file_path} (unknown category)")
            stats['skipped'] += 1
            continue

        output_path = save_dir / filename

        # Check if already downloaded
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 1:  # Valid zip should be > 1 MB
                stats['skipped'] += 1
                logger.debug(f"  [{stats['total']}] Skipping {filename} (already exists, {file_size_mb:.1f} MB)")
                continue
            else:
                # File exists but too small, re-download
                logger.warning(f"  [{stats['total']}] Re-downloading {filename} (file too small: {file_size_mb:.1f} MB)")
                output_path.unlink()

        # Download file
        try:
            logger.info(f"  [{stats['total']}/{len(zip_files)}] Downloading {filename} → {category_label}/")

            # Download using HuggingFace Hub (uses cache by default)
            downloaded_path = hf_hub_download(
                repo_id=dataset_name,
                repo_type=repo_type,
                filename=file_path,
                token=hf_token
            )

            # Copy to target location (don't move, as HF cache should be preserved)
            import shutil
            shutil.copy2(downloaded_path, output_path)

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"      ✓ {filename} ({file_size_mb:.1f} MB)")

        except Exception as e:
            logger.error(f"  [{stats['total']}] Failed to download {filename}: {e}")
            stats['failed'] += 1
            if output_path.exists():
                output_path.unlink()  # Clean up partial file

    # Final statistics
    logger.info("=" * 70)
    logger.info("")
    logger.info("✓ Download complete!")
    logger.info("")
    logger.info("Statistics:")
    logger.info(f"  Total processed: {stats['total']}")
    logger.info(f"  APE cases: {stats['ape']} downloaded")
    logger.info(f"  non-APE cases: {stats['non_ape']} downloaded")
    logger.info(f"  Skipped (already exist): {stats['skipped']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info("")

    # Verify files on disk
    ape_files = list(ape_dir.glob('*.zip'))
    non_ape_files = list(non_ape_dir.glob('*.zip'))

    logger.info("Files on disk:")
    logger.info(f"  {ape_dir}: {len(ape_files)} files")
    logger.info(f"  {non_ape_dir}: {len(non_ape_files)} files")
    logger.info(f"  Total: {len(ape_files) + len(non_ape_files)} files")

    # Calculate total size
    total_size_bytes = sum(f.stat().st_size for f in ape_files + non_ape_files)
    total_size_gb = total_size_bytes / (1024 ** 3)
    logger.info(f"  Total size: {total_size_gb:.2f} GB")
    logger.info("")

    if stats['failed'] > 0:
        logger.warning(f"WARNING: {stats['failed']} cases failed to download")
        logger.warning("You may need to re-run this script to retry failed downloads")


def main():
    parser = argparse.ArgumentParser(
        description='Download APE dataset from HuggingFace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location
  python scripts/download_ape_dataset.py

  # Download to custom location
  python scripts/download_ape_dataset.py --output_dir /path/to/dataset

  # Verbose logging
  python scripts/download_ape_dataset.py --verbose
        """
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='/workspace/storage_a100/dataset',
        help='Output directory for .zip files (default: /workspace/storage_a100/dataset)'
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        default='t2ance/APE-data',
        help='HuggingFace dataset name (default: t2ance/APE-data)'
    )

    parser.add_argument(
        '--repo_type',
        type=str,
        default='dataset',
        help='Repository type (default: dataset)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Ignored arguments for compatibility
    parser.add_argument('--split', type=str, default='train', help=argparse.SUPPRESS)
    parser.add_argument('--streaming', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run download
    download_ape_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        repo_type=args.repo_type
    )


if __name__ == '__main__':
    main()
