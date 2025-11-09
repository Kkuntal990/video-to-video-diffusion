"""
Comprehensive Test for Full-Volume CT Slice Interpolation Pipeline

Tests the complete implementation before deployment:
1. Data loading (full volumes from dataset)
2. Model architecture (Medium U-Net 599M + MAISI VAE)
3. Loss functions (Diffusion + Perceptual + MS-SSIM)
4. Memory usage (verify A100 80GB compatibility)
5. Training step (forward + backward pass)

Usage:
    # Local testing (with actual data)
    python scripts/test_full_volume_pipeline.py

    # Quick test (without data, synthetic only)
    python scripts/test_full_volume_pipeline.py --synthetic
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
import time
from typing import Dict, Optional


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("CPU mode (no GPU)")


def test_data_loading_real(config: Dict):
    """Test with real data from dataset"""
    print_section("TEST 1: Data Loading (Real Dataset)")

    try:
        from data.get_dataloader import get_dataloader

        print("Creating dataloader from real dataset...")
        print(f"Dataset path: {config['data']['dataset_path']}")

        train_loader = get_dataloader(config['data'], split='train')

        print(f"✓ Dataloader created: {len(train_loader)} batches")

        # Load one batch
        print("\nLoading first batch...")
        batch = next(iter(train_loader))

        thick = batch['thick']
        thin = batch['thin']

        print(f"\nBatch structure:")
        print(f"  Thick: {thick.shape}")
        print(f"  Thin:  {thin.shape}")
        print(f"  Patients: {batch['patient_id']}")
        print(f"  Categories: {batch['category']}")

        print(f"\nValue ranges:")
        print(f"  Thick: [{thick.min():.3f}, {thick.max():.3f}]")
        print(f"  Thin:  [{thin.min():.3f}, {thin.max():.3f}]")

        # Check interpolation factor
        D_thick = thick.shape[2]
        D_thin = thin.shape[2]
        print(f"\nInterpolation: {D_thick} → {D_thin} ({D_thin/D_thick:.1f}×)")

        print("\n✅ Real data loading test PASSED")
        return True, train_loader

    except Exception as e:
        print(f"\n❌ Real data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_data_loading_synthetic():
    """Test with synthetic data (quick test without dataset)"""
    print_section("TEST 1: Data Loading (Synthetic Data)")

    try:
        # Create synthetic batch mimicking real data
        B = 2
        thick = torch.randn(B, 1, 50, 512, 512)  # 50 thick slices
        thin = torch.randn(B, 1, 300, 512, 512)  # 300 thin slices

        print(f"Created synthetic data:")
        print(f"  Thick: {thick.shape}")
        print(f"  Thin:  {thin.shape}")
        print(f"  Interpolation: {thick.shape[2]} → {thin.shape[2]} ({thin.shape[2]/thick.shape[2]:.1f}×)")

        print("\n✅ Synthetic data test PASSED")
        return True, [(thick, thin)]

    except Exception as e:
        print(f"\n❌ Synthetic data test FAILED: {e}")
        return False, None


def test_model_creation(config: Dict, device: str):
    """Test model creation with medium U-Net"""
    print_section("TEST 2: Model Creation (Medium U-Net + MAISI VAE)")

    try:
        from models.model import VideoToVideoDiffusion

        print(f"Device: {device}")
        print(f"U-Net channels: {config['model']['unet_model_channels']}")
        print(f"Expected total params: ~729M (599M U-Net + 130M VAE)")

        # Create model config matching the structure
        model_config = {
            'in_channels': config['model']['in_channels'],
            'latent_dim': config['model']['latent_dim'],
            'vae_base_channels': config['model'].get('vae_base_channels', 64),
            'unet_model_channels': config['model']['unet_model_channels'],
            'unet_num_res_blocks': config['model']['unet_num_res_blocks'],
            'unet_attention_levels': config['model']['unet_attention_levels'],
            'unet_channel_mult': config['model']['unet_channel_mult'],
            'unet_num_heads': config['model']['unet_num_heads'],
            'unet_time_embed_dim': config['model'].get('unet_time_embed_dim', 1024),
            'noise_schedule': config['model']['noise_schedule'],
            'diffusion_timesteps': config['model']['diffusion_timesteps'],
            'pretrained': {
                'use_pretrained': False,  # Test architecture first without loading weights
                'vae': {
                    'enabled': False,
                    'use_custom_maisi': False,
                }
            }
        }

        # Create model
        model = VideoToVideoDiffusion(model_config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        unet_params = sum(p.numel() for p in model.unet.parameters())
        vae_params = sum(p.numel() for p in model.vae.parameters())

        print(f"\n✓ Model created:")
        print(f"  Total: {total_params / 1e6:.1f}M params")
        print(f"  U-Net: {unet_params / 1e6:.1f}M params")
        print(f"  VAE: {vae_params / 1e6:.1f}M params")

        # Move to device
        model = model.to(device)
        print(f"\n✓ Moved to {device}")

        print_memory_usage()

        print("\n✅ Model creation test PASSED")
        return True, model

    except Exception as e:
        print(f"\n❌ Model creation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_loss_functions(device: str):
    """Test multi-scale loss functions"""
    print_section("TEST 3: Loss Functions (Perceptual + MS-SSIM)")

    try:
        from models.losses import VGGPerceptualLoss, MS_SSIM_Loss, CombinedLoss

        print("Creating loss modules...")

        # Test individual losses
        perceptual = VGGPerceptualLoss(slice_sample_rate=0.2).to(device)
        ms_ssim = MS_SSIM_Loss().to(device)

        print("✓ VGGPerceptualLoss created")
        print("✓ MS_SSIM_Loss created")

        # Test combined loss
        combined = CombinedLoss(
            lambda_perceptual=0.1,
            lambda_ssim=0.1,
            perceptual_every_n_steps=10,
            ssim_every_n_steps=10,
        ).to(device)

        print("✓ CombinedLoss created")

        # Test with dummy data
        B, C, D, H, W = 1, 1, 50, 256, 256
        pred = torch.randn(B, C, D, H, W, device=device)
        target = torch.randn(B, C, D, H, W, device=device)
        diffusion_loss = torch.tensor(0.1, device=device)

        print(f"\nTesting with shape: ({B}, {C}, {D}, {H}, {W})")

        start = time.time()
        total_loss, loss_dict = combined(pred, target, diffusion_loss, compute_auxiliary=True)
        elapsed = time.time() - start

        print(f"\n✓ Loss computed in {elapsed:.3f}s")
        print(f"  Components: {loss_dict}")

        print("\n✅ Loss functions test PASSED")
        return True, combined

    except Exception as e:
        print(f"\n❌ Loss functions test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model, data, device: str):
    """Test forward pass with full volumes"""
    print_section("TEST 4: Forward Pass (Full Volumes)")

    try:
        model.eval()

        # Get data
        if isinstance(data, list):
            # Synthetic data
            thick, thin = data[0]
        else:
            # Real data loader
            batch = next(iter(data))
            thick = batch['thick']
            thin = batch['thin']

        thick = thick.to(device)
        thin = thin.to(device)

        print(f"Input shapes:")
        print(f"  Thick (v_in): {thick.shape}")
        print(f"  Thin (v_gt):  {thin.shape}")

        print("\nRunning forward pass...")
        print_memory_usage()

        start = time.time()

        with torch.no_grad():
            # Model forward returns (loss, metrics) tuple
            result = model(thick, thin)
            if isinstance(result, tuple):
                loss, metrics = result
            else:
                loss = result
                metrics = {}

        elapsed = time.time() - start

        print(f"\n✓ Forward pass completed in {elapsed:.3f}s")
        print(f"  Loss: {loss.item():.4f}")
        if metrics:
            print(f"  Metrics: {metrics}")

        print_memory_usage()

        print("\n✅ Forward pass test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Forward pass test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model, data, device: str):
    """Test complete training step"""
    print_section("TEST 5: Training Step (Forward + Backward)")

    try:
        model.train()

        # Get data
        if isinstance(data, list):
            thick, thin = data[0]
        else:
            batch = next(iter(data))
            thick = batch['thick']
            thin = batch['thin']

        thick = thick.to(device)
        thin = thin.to(device)

        print(f"Input shapes:")
        print(f"  Thick (v_in): {thick.shape}")
        print(f"  Thin (v_gt):  {thin.shape}")

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        print("\nRunning training step...")
        print("  [Forward]")
        print_memory_usage()

        start = time.time()

        # Forward
        result = model(thick, thin)
        if isinstance(result, tuple):
            loss, metrics = result
        else:
            loss = result
            metrics = {}

        print(f"    Loss: {loss.item():.4f}")
        if metrics:
            print(f"    Metrics: {metrics}")
        print_memory_usage()

        # Backward
        print("  [Backward]")
        optimizer.zero_grad()
        loss.backward()
        print_memory_usage()

        # Step
        print("  [Optimizer]")
        optimizer.step()

        elapsed = time.time() - start

        print(f"\n✓ Training step completed in {elapsed:.3f}s")
        print_memory_usage()

        print("\n✅ Training step test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Training step test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_check(device: str):
    """Check memory usage against limits"""
    print_section("TEST 6: Memory Usage (A100 80GB)")

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - skipping")
        return True

    try:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_alloc = torch.cuda.max_memory_allocated() / 1024**3

        print(f"Memory usage:")
        print(f"  Current: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Peak: {max_alloc:.2f} GB")

        LIMIT_GB = 80
        SAFE_THRESHOLD = 0.6

        safe_limit = LIMIT_GB * SAFE_THRESHOLD

        print(f"\nLimits:")
        print(f"  Total: {LIMIT_GB} GB")
        print(f"  Safe threshold: {safe_limit:.1f} GB ({SAFE_THRESHOLD*100:.0f}%)")

        if max_alloc < safe_limit:
            print(f"\n✅ Memory is SAFE ({max_alloc:.2f} GB < {safe_limit:.1f} GB)")
            return True
        else:
            print(f"\n⚠️  Memory is HIGH ({max_alloc:.2f} GB >= {safe_limit:.1f} GB)")
            return False

    except Exception as e:
        print(f"\n❌ Memory check FAILED: {e}")
        return False


def main():
    """Run all tests"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data (no real dataset needed)')
    parser.add_argument('--config', default='config/slice_interpolation_full_medium.yaml', help='Config file path')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("  FULL-VOLUME CT SLICE INTERPOLATION - INTEGRATION TEST")
    print("="*80)

    # Load config
    config_path = project_root / args.config
    print(f"\nConfig: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override for local testing
    if 'dataset_path' not in config['data']:
        config['data']['dataset_path'] = '/Users/kuntalkokate/Desktop/LLM_agents_projects/dataset'

    print("✓ Config loaded")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Run tests
    results = {}

    # Test 1: Data
    if args.synthetic:
        results['data'], data = test_data_loading_synthetic()
    else:
        results['data'], data = test_data_loading_real(config)

    if not results['data']:
        print("\n❌ Data loading failed. Stopping.")
        return

    # Test 2: Model
    results['model'], model = test_model_creation(config, device)
    if not results['model']:
        print("\n❌ Model creation failed. Stopping.")
        return

    # Test 3: Losses
    results['losses'], _ = test_loss_functions(device)

    # Test 4: Forward
    results['forward'] = test_forward_pass(model, data, device)

    # Test 5: Training
    results['training'] = test_training_step(model, data, device)

    # Test 6: Memory
    results['memory'] = test_memory_check(device)

    # Summary
    print_section("SUMMARY")

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:15s}: {status}")

    print("\n" + "="*80)
    if all_passed:
        print("  🎉 ALL TESTS PASSED! Ready for deployment.")
    else:
        print("  ⚠️  SOME TESTS FAILED. Review errors above.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
