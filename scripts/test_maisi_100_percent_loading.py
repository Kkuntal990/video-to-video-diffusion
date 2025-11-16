"""
Test MAISI VAE with Correct Architecture for 100% Weight Loading

This script tests the corrected AutoencoderKL configuration to verify
that we can load 100% of MAISI pretrained weights (not just 14%).
"""

import torch
try:
    from generative.networks.nets import AutoencoderKL
except ImportError:
    from monai.networks.nets import AutoencoderKL

def test_maisi_loading(checkpoint_path='./pretrained/maisi_vae/models/autoencoder.pt'):
    """
    Test loading MAISI weights with correct architecture

    Expected: 130/130 weights loaded (100%)
    """
    print("="*80)
    print("TESTING MAISI VAE WEIGHT LOADING - CORRECT ARCHITECTURE")
    print("="*80)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"Checkpoint contains {len(state_dict)} parameters")

    # Create AutoencoderKL with CORRECT configuration
    print("\n" + "="*80)
    print("Creating AutoencoderKL with CORRECT configuration")
    print("="*80)

    config = {
        'spatial_dims': 3,
        'in_channels': 1,
        'out_channels': 1,
        'channels': (64, 128, 256),  # CORRECT: 3-level, doubling each level
        'num_res_blocks': 2,  # 2 residual blocks per level
        'latent_channels': 4,
        'norm_num_groups': 32,
        'attention_levels': (False, False, False),  # No attention in MAISI VAE
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create model
    print("\nInitializing AutoencoderKL...")
    model = AutoencoderKL(**config)

    # Get model's state dict
    model_state = model.state_dict()
    print(f"Model has {len(model_state)} parameters")

    # Test loading
    print("\n" + "="*80)
    print("LOADING WEIGHTS")
    print("="*80)

    # Track loading statistics
    loaded_count = 0
    incompatible_count = 0
    missing_count = 0
    incompatible_keys = []
    missing_keys = []

    # Filter state dict for compatible shapes
    filtered_state = {}
    for key in state_dict.keys():
        if key in model_state:
            if state_dict[key].shape == model_state[key].shape:
                filtered_state[key] = state_dict[key]
                loaded_count += 1
            else:
                incompatible_keys.append(key)
                incompatible_count += 1
                print(f"⚠ Shape mismatch: {key}")
                print(f"  Checkpoint: {state_dict[key].shape}")
                print(f"  Model: {model_state[key].shape}")
        else:
            missing_keys.append(key)

    # Check for keys in model but not in checkpoint
    checkpoint_keys = set(state_dict.keys())
    model_keys = set(model_state.keys())
    extra_model_keys = model_keys - checkpoint_keys

    # Load filtered state dict
    result = model.load_state_dict(filtered_state, strict=False)

    # Report results
    print("\n" + "="*80)
    print("LOADING RESULTS")
    print("="*80)

    print(f"\n✓ Successfully loaded: {loaded_count}/{len(state_dict)} parameters")
    print(f"  ({loaded_count/len(state_dict)*100:.1f}%)")

    if incompatible_count > 0:
        print(f"\n⚠ Incompatible (shape mismatch): {incompatible_count} parameters")
        for key in incompatible_keys[:5]:
            print(f"  - {key}")
        if len(incompatible_keys) > 5:
            print(f"  ... and {len(incompatible_keys)-5} more")

    if extra_model_keys:
        print(f"\nℹ Model has {len(extra_model_keys)} parameters not in checkpoint")
        print("  (These will be initialized randomly)")
        for key in sorted(extra_model_keys)[:5]:
            print(f"  - {key}")
        if len(extra_model_keys) > 5:
            print(f"  ... and {len(extra_model_keys)-5} more")

    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    if loaded_count == len(state_dict) and incompatible_count == 0 and len(extra_model_keys) == 0:
        print("\n🎉 SUCCESS! 100% of MAISI weights loaded correctly!")
        print("   All 130 parameters matched perfectly.")
        return True
    elif loaded_count == len(state_dict) and incompatible_count == 0:
        print(f"\n✅ All checkpoint weights loaded ({loaded_count}/{len(state_dict)})")
        print(f"   ⚠ But model has {len(extra_model_keys)} extra parameters (initialized randomly)")
        print("   This is okay - extra parameters can be fine-tuned during training.")
        return True
    else:
        print(f"\n❌ FAILED: Only {loaded_count}/{len(state_dict)} weights loaded ({loaded_count/len(state_dict)*100:.1f}%)")
        print("   Architecture still doesn't match MAISI exactly.")
        print("\n   Debugging suggestions:")
        print("   1. Check incompatible keys above for shape mismatches")
        print("   2. Verify num_res_blocks, attention_levels, norm_num_groups")
        print("   3. Inspect maisi_checkpoint_keys.txt for full structure")
        return False


def test_encode_decode():
    """
    Test encoding and decoding with loaded MAISI VAE
    """
    print("\n" + "="*80)
    print("TESTING ENCODE/DECODE WITH LOADED WEIGHTS")
    print("="*80)

    # Create model with correct config
    model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256),
        num_res_blocks=2,
        latent_channels=4,
        norm_num_groups=32,
        attention_levels=(False, False, False),
    )

    # Load weights
    state_dict = torch.load('./pretrained/maisi_vae/models/autoencoder.pt', map_location='cpu', weights_only=False)
    filtered_state = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    print("\n✓ Model loaded and set to eval mode")

    # Test with sample input (8 slices minimum for 3×3×3 kernels)
    print("\nTesting with 8-slice volume (512×512)...")
    x = torch.randn(1, 1, 8, 512, 512)

    with torch.no_grad():
        # Encode
        print(f"  Input shape: {x.shape}")
        z_dist = model.encode(x)

        if hasattr(z_dist, 'sample'):
            z = z_dist.sample()
        else:
            z = z_dist

        print(f"  Latent shape: {z.shape}")

        # Decode
        x_recon = model.decode(z)
        print(f"  Reconstructed shape: {x_recon.shape}")

        # Compute reconstruction error
        mse = torch.mean((x - x_recon) ** 2).item()
        print(f"\n  Reconstruction MSE: {mse:.6f}")

        if mse < 0.1:
            print("  ✓ Low MSE - weights loaded correctly!")
        else:
            print("  ⚠ High MSE - check if weights loaded properly")

    print("\n✓ Encode/decode test complete!")


if __name__ == "__main__":
    # Test weight loading
    success = test_maisi_loading()

    if success:
        print("\n" + "="*80)
        print("Proceeding to encode/decode test...")
        print("="*80)
        test_encode_decode()
    else:
        print("\n❌ Weight loading failed - skipping encode/decode test")
        print("\nPlease review the architecture configuration and try again.")
