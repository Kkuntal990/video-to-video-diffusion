"""
Test script to verify APE-data can be loaded correctly

This script will:
1. Test APE dataset loading from local files
2. Verify data shapes and ranges
3. Test the complete training pipeline with minimal config
"""

import torch
import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.ape_dataset import APEDataset, get_ape_dataloader


def test_ape_dataset_local():
    """Test loading APE data from local directory"""
    print("=" * 60)
    print("TEST 1: APE Dataset Loading (Local Files)")
    print("=" * 60)

    data_dir = "/Users/kuntalkokate/Desktop/LLM_agents_projects/dataset"

    if not Path(data_dir).exists():
        print(f"✗ Data directory not found: {data_dir}")
        return False

    try:
        print("\nCreating APE dataset...")
        dataset = APEDataset(
            data_dir=data_dir,
            num_frames=8,
            resolution=(128, 128),
            categories=['APE'],
            cache_extracted=False
        )

        print(f"✓ Dataset created successfully")
        print(f"  Total patients: {len(dataset)}")

        if len(dataset) == 0:
            print("✗ No data found in dataset")
            return False

        print("\n" + "-" * 60)
        print("Loading first sample...")
        sample = dataset[0]

        print(f"\n✓ Sample loaded successfully")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Input shape: {sample['input'].shape}")
        print(f"  Target shape: {sample['target'].shape}")
        print(f"  Category: {sample['category']}")
        print(f"  Patient ID: {sample['patient_id']}")
        print(f"  Input value range: [{sample['input'].min():.3f}, {sample['input'].max():.3f}]")
        print(f"  Target value range: [{sample['target'].min():.3f}, {sample['target'].max():.3f}]")

        # Verify shapes
        expected_shape = (3, 8, 128, 128)  # (C, T, H, W)
        if sample['input'].shape == expected_shape and sample['target'].shape == expected_shape:
            print(f"\n✓ Shapes are correct: {expected_shape}")
        else:
            print(f"\n✗ Shape mismatch!")
            print(f"  Expected: {expected_shape}")
            print(f"  Got input: {sample['input'].shape}, target: {sample['target'].shape}")
            return False

        # Verify value ranges (should be normalized to [-1, 1])
        if -1.0 <= sample['input'].min() <= 1.0 and -1.0 <= sample['input'].max() <= 1.0:
            print(f"✓ Values are normalized to [-1, 1] range")
        else:
            print(f"⚠ Warning: Values may not be properly normalized")

        print("\n" + "=" * 60)
        print("✓ TEST 1 PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ape_dataloader():
    """Test APE dataloader with batching"""
    print("\n" * 2)
    print("=" * 60)
    print("TEST 2: APE DataLoader (Batching)")
    print("=" * 60)

    data_dir = "/Users/kuntalkokate/Desktop/LLM_agents_projects/dataset"

    try:
        config = {
            'num_frames': 8,
            'resolution': [128, 128],
            'categories': ['APE'],
            'batch_size': 1,
            'num_workers': 0,
            'cache_extracted': False
        }

        print("\nCreating dataloader...")
        dataloader = get_ape_dataloader(data_dir, config, split='train')

        print(f"✓ Dataloader created")
        print(f"  Dataset size: {len(dataloader.dataset)}")
        print(f"  Batch size: {dataloader.batch_size}")
        print(f"  Num workers: {dataloader.num_workers}")

        print("\n" + "-" * 60)
        print("Loading first batch...")

        batch = next(iter(dataloader))

        print(f"\n✓ Batch loaded successfully")
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Input batch shape: {batch['input'].shape}")
        print(f"  Target batch shape: {batch['target'].shape}")

        expected_batch_shape = (1, 3, 8, 128, 128)  # (B, C, T, H, W)
        if batch['input'].shape == expected_batch_shape:
            print(f"✓ Batch shape is correct: {expected_batch_shape}")
        else:
            print(f"✗ Batch shape mismatch!")
            print(f"  Expected: {expected_batch_shape}")
            print(f"  Got: {batch['input'].shape}")
            return False

        print("\n" + "=" * 60)
        print("✓ TEST 2 PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_with_ape_data():
    """Test that the model can process APE data"""
    print("\n" * 2)
    print("=" * 60)
    print("TEST 3: Model Forward Pass with APE Data")
    print("=" * 60)

    try:
        from models.model import VideoToVideoDiffusion

        # Create minimal test config (flat structure matching model.__init__)
        config = {
            'in_channels': 3,
            'latent_dim': 4,
            'vae_base_channels': 32,
            'unet_model_channels': 64,
            'unet_num_res_blocks': 1,
            'unet_attention_levels': [1],
            'unet_channel_mult': [1, 2],
            'unet_num_heads': 4,
            'unet_time_embed_dim': 256,
            'noise_schedule': 'cosine',
            'diffusion_timesteps': 100,
            'pretrained': {
                'use_pretrained': False
            }
        }

        print("\nCreating model...")
        model = VideoToVideoDiffusion(config, load_pretrained=False)
        model.eval()
        print("✓ Model created")

        # Create dummy batch matching APE data format
        print("\nCreating test batch...")
        batch = {
            'input': torch.randn(1, 3, 8, 128, 128),
            'target': torch.randn(1, 3, 8, 128, 128)
        }

        print("\n" + "-" * 60)
        print("Testing training forward pass...")
        with torch.no_grad():
            loss, metrics = model(batch['input'], batch['target'])

        print(f"\n✓ Training forward pass successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Metrics: {metrics}")

        print("\n" + "-" * 60)
        print("Testing inference generation...")
        with torch.no_grad():
            output = model.generate(
                batch['input'],
                sampler='ddim',
                num_inference_steps=5
            )

        print(f"\n✓ Inference generation successful")
        print(f"  Input shape: {batch['input'].shape}")
        print(f"  Output shape: {output.shape}")

        if output.shape == batch['input'].shape:
            print("✓ Output shape matches input shape")
        else:
            print(f"✗ Shape mismatch! Expected {batch['input'].shape}, got {output.shape}")
            return False

        print("\n" + "=" * 60)
        print("✓ TEST 3 PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "APE-Data Loading Test Suite" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    results = []

    # Test 1: Dataset loading
    results.append(("APE Dataset Loading", test_ape_dataset_local()))

    # Test 2: DataLoader
    results.append(("APE DataLoader", test_ape_dataloader()))

    # Test 3: Model integration
    results.append(("Model Integration", test_model_with_ape_data()))

    # Summary
    print("\n" * 2)
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {test_name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready for training.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Update config/train_config.yaml with APE data settings")
        print("3. Run training: python train.py --config config/train_config.yaml")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
