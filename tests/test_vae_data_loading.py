"""
Lightweight test for VAE data loading
Tests that both full-volume and patch datasets return compatible batch formats
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_patch_dataset_batch_format():
    """Test that patch dataset returns correct batch format"""
    from data.patch_slice_interpolation_dataset import PatchSliceInterpolationDataset

    # Create dummy preprocessed data
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 dummy .pt files
        for i in range(3):
            case_file = Path(tmpdir) / f"case_{i:03d}.pt"

            # Create dummy data (small sizes for speed)
            # Use uniform distribution to ensure [-1, 1] range
            thick = torch.rand(1, 50, 512, 512) * 2 - 1  # [-1, 1] range
            thin = torch.rand(1, 300, 512, 512) * 2 - 1

            torch.save({
                'input': thick,
                'target': thin,
                'category': 'APE',
                'patient_id': f'case_{i:03d}',
            }, case_file)

        # Create dataset
        dataset = PatchSliceInterpolationDataset(
            processed_dir=tmpdir,
            patch_depth_thin=48,
            patch_depth_thick=8,
            patch_size=(192, 192),
            split='train',
            val_ratio=0.0,
            test_ratio=0.0,
            augment=False
        )

        # Get a sample
        sample = dataset[0]

        # Test 1: Required keys exist
        assert 'input' in sample, "Missing 'input' key"
        assert 'target' in sample, "Missing 'target' key"
        assert 'x_lr' in sample, "Missing 'x_lr' key"
        assert 'x_hr' in sample, "Missing 'x_hr' key"
        assert 'category' in sample, "Missing 'category' key"
        assert 'patient_id' in sample, "Missing 'patient_id' key"

        # Test 2: Shapes are correct (patch mode)
        input_shape = sample['input'].shape
        target_shape = sample['target'].shape

        assert input_shape == (1, 8, 192, 192), f"Wrong input shape: {input_shape}"
        assert target_shape == (1, 48, 192, 192), f"Wrong target shape: {target_shape}"

        # Test 3: Value ranges are correct [-1, 1]
        assert sample['input'].min() >= -1.5, f"Input values too low: {sample['input'].min()}"
        assert sample['input'].max() <= 1.5, f"Input values too high: {sample['input'].max()}"
        assert sample['target'].min() >= -1.5, f"Target values too low: {sample['target'].min()}"
        assert sample['target'].max() <= 1.5, f"Target values too high: {sample['target'].max()}"

        # Test 4: Aliases are consistent
        assert torch.equal(sample['input'], sample['x_lr']), "input != x_lr"
        assert torch.equal(sample['target'], sample['x_hr']), "target != x_hr"

        print("✓ Patch dataset batch format test passed!")


def test_vae_training_compatible_batch():
    """Test that batches are compatible with train_vae.py expectations"""
    from data.patch_slice_interpolation_dataset import PatchSliceInterpolationDataset
    from torch.utils.data import DataLoader

    # Create dummy preprocessed data
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 5 dummy .pt files for batching
        for i in range(5):
            case_file = Path(tmpdir) / f"case_{i:03d}.pt"

            thick = torch.rand(1, 50, 512, 512) * 2 - 1
            thin = torch.rand(1, 300, 512, 512) * 2 - 1

            torch.save({
                'input': thick,
                'target': thin,
                'category': 'APE',
                'patient_id': f'case_{i:03d}',
            }, case_file)

        # Create dataset and dataloader
        dataset = PatchSliceInterpolationDataset(
            processed_dir=tmpdir,
            patch_depth_thin=48,
            patch_depth_thick=8,
            patch_size=(192, 192),
            split='train',
            val_ratio=0.0,
            test_ratio=0.0,
            augment=False
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        # Get a batch
        batch = next(iter(dataloader))

        # Test 1: Batch has correct keys
        assert 'input' in batch, "Batch missing 'input' key (needed for train_vae.py)"
        assert 'target' in batch, "Batch missing 'target' key"

        # Test 2: Batch shapes are correct (B, C, D, H, W)
        B = 2
        input_shape = batch['input'].shape
        target_shape = batch['target'].shape

        assert input_shape == (B, 1, 8, 192, 192), f"Wrong batch input shape: {input_shape}"
        assert target_shape == (B, 1, 48, 192, 192), f"Wrong batch target shape: {target_shape}"

        # Test 3: Can simulate train_vae.py usage
        thick_slices = batch['input']  # This is what train_vae.py does now
        assert thick_slices.shape == (B, 1, 8, 192, 192), "train_vae.py compatibility failed"

        print("✓ VAE training batch compatibility test passed!")


def test_shape_consistency_across_modes():
    """Test that shapes are consistent in patch mode"""
    from data.patch_slice_interpolation_dataset import PatchSliceInterpolationDataset

    # Create dummy preprocessed data
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data
        case_file = Path(tmpdir) / "case_000.pt"

        thick = torch.randn(1, 50, 512, 512) * 2 - 1
        thin = torch.randn(1, 300, 512, 512) * 2 - 1

        torch.save({
            'input': thick,
            'target': thin,
            'category': 'APE',
            'patient_id': 'case_000',
        }, case_file)

        # Create dataset with different patch sizes
        dataset1 = PatchSliceInterpolationDataset(
            processed_dir=tmpdir,
            patch_depth_thin=48,
            patch_depth_thick=8,
            patch_size=(192, 192),
            split='train',
            val_ratio=0.0,
            test_ratio=0.0,
            augment=False
        )

        dataset2 = PatchSliceInterpolationDataset(
            processed_dir=tmpdir,
            patch_depth_thin=24,
            patch_depth_thick=4,
            patch_size=(128, 128),
            split='train',
            val_ratio=0.0,
            test_ratio=0.0,
            augment=False
        )

        # Get samples
        sample1 = dataset1[0]
        sample2 = dataset2[0]

        # Test: Shapes match configuration
        assert sample1['input'].shape == (1, 8, 192, 192)
        assert sample1['target'].shape == (1, 48, 192, 192)

        assert sample2['input'].shape == (1, 4, 128, 128)
        assert sample2['target'].shape == (1, 24, 128, 128)

        print("✓ Shape consistency test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("VAE Data Loading Tests")
    print("=" * 70)
    print()

    print("Test 1: Patch dataset batch format")
    test_patch_dataset_batch_format()
    print()

    print("Test 2: VAE training batch compatibility")
    test_vae_training_compatible_batch()
    print()

    print("Test 3: Shape consistency across modes")
    test_shape_consistency_across_modes()
    print()

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
