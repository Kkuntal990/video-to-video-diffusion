"""
Test Trainer Integration with Slice Interpolation Dataset

This test verifies that the dataloader returns batches with the correct keys
that the trainer expects: 'input' and 'target'

This test would have caught the KeyError: 'input' bug!
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml

def test_batch_keys():
    """Test that batch has 'input' and 'target' keys (required by trainer)"""
    print("="*80)
    print("TEST: Batch Keys for Trainer Integration")
    print("="*80)

    # Load config
    config_path = project_root / 'config' / 'slice_interpolation_full_medium.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nConfig: {config_path}")
    print(f"Dataset: {config['data']['data_source']}")

    # Create dataloader
    from data.get_dataloader import get_dataloader

    print("\nCreating train dataloader...")
    train_loader = get_dataloader(config['data'], split='train')
    print(f"✓ Created dataloader with {len(train_loader)} batches")

    # Get first batch
    print("\nLoading first batch...")
    batch = next(iter(train_loader))

    # Check required keys
    print("\n" + "-"*80)
    print("Checking batch keys...")
    print("-"*80)

    required_keys = ['input', 'target']
    optional_keys = ['thick', 'thin', 'category', 'patient_id', 'num_thick_slices', 'num_thin_slices']

    all_keys = list(batch.keys())
    print(f"\nAll keys in batch: {all_keys}")

    # Verify required keys
    missing = []
    for key in required_keys:
        if key in batch:
            print(f"  ✓ '{key}' - PRESENT (required by trainer)")
        else:
            print(f"  ✗ '{key}' - MISSING (REQUIRED BY TRAINER!)")
            missing.append(key)

    # Show optional keys
    for key in optional_keys:
        if key in batch:
            print(f"  ✓ '{key}' - present (optional)")

    if missing:
        print(f"\n❌ FAILED: Missing required keys: {missing}")
        print("The trainer expects batch['input'] and batch['target']")
        print("Check the collate function in slice_interpolation_dataset.py")
        return False

    # Verify shapes
    print("\n" + "-"*80)
    print("Verifying shapes...")
    print("-"*80)

    v_in = batch['input']
    v_gt = batch['target']

    print(f"\n  batch['input']:  {v_in.shape}")
    print(f"  batch['target']: {v_gt.shape}")

    # Check they are tensors (NOT lists!)
    if isinstance(v_in, list):
        print(f"  ✗ 'input' is a list, not a tensor!")
        print(f"     This will cause AttributeError: 'list' object has no attribute 'to'")
        return False
    if isinstance(v_gt, list):
        print(f"  ✗ 'target' is a list, not a tensor!")
        print(f"     This will cause AttributeError: 'list' object has no attribute 'to'")
        return False

    assert isinstance(v_in, torch.Tensor), "input must be a tensor"
    assert isinstance(v_gt, torch.Tensor), "target must be a tensor"
    print(f"\n  ✓ Both are torch.Tensor objects (not lists!)")

    # Check they can be moved to device (simulating trainer behavior)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Testing device transfer to '{device}'...")
    try:
        v_in_dev = v_in.to(device)
        v_gt_dev = v_gt.to(device)
        print(f"  ✓ Successfully moved to {device}")
    except Exception as e:
        print(f"  ✗ Failed to move to device: {e}")
        return False

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nSummary:")
    print("  ✓ Batch has 'input' key (trainer compatible)")
    print("  ✓ Batch has 'target' key (trainer compatible)")
    print("  ✓ Values can be moved to device")
    print("  ✓ Ready for trainer.train_epoch()")
    print("\n")

    return True


def test_trainer_forward_pass():
    """Test actual trainer forward pass simulation"""
    print("="*80)
    print("TEST: Trainer Forward Pass Simulation")
    print("="*80)

    # Load config
    config_path = project_root / 'config' / 'slice_interpolation_full_medium.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create dataloader
    from data.get_dataloader import get_dataloader
    train_loader = get_dataloader(config['data'], split='train')

    # Get batch
    batch = next(iter(train_loader))

    # Simulate trainer code (lines 168-169 of trainer.py)
    print("\nSimulating trainer.train_epoch() code:")
    print("  Line 168: v_in = batch['input'].to(self.device)")
    print("  Line 169: v_gt = batch['target'].to(self.device)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # This is EXACTLY what the trainer does
        v_in = batch['input'].to(device)
        v_gt = batch['target'].to(device)

        print(f"\n✓ Trainer simulation successful!")
        print(f"  v_in: {v_in.shape} on {device}")
        print(f"  v_gt: {v_gt.shape} on {device}")

        return True

    except KeyError as e:
        print(f"\n✗ KeyError (SAME AS PRODUCTION BUG): {e}")
        print("This is the error you would see in training!")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  TRAINER INTEGRATION TEST")
    print("  Tests dataloader compatibility with training/trainer.py")
    print("="*80 + "\n")

    results = {}

    # Test 1: Batch keys
    results['batch_keys'] = test_batch_keys()

    # Test 2: Trainer simulation
    results['trainer_simulation'] = test_trainer_forward_pass()

    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)

    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:25s}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("  🎉 ALL TESTS PASSED!")
        print("  The dataloader is compatible with the trainer.")
    else:
        print("  ⚠️  SOME TESTS FAILED!")
        print("  The dataloader may not work with the trainer.")
    print("="*80 + "\n")

    sys.exit(0 if all_passed else 1)
