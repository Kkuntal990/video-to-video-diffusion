"""
Test Complete MAISI VAE Loading with MONAI GenerativeModels

This script loads the MAISI checkpoint directly using the proper
generative.networks.nets.AutoencoderKL class.
"""

import torch
from generative.networks.nets import AutoencoderKL

def test_complete_maisi_loading(checkpoint_path='./pretrained/maisi_vae/models/autoencoder.pt'):
    """
    Test loading complete MAISI checkpoint
    """
    print("="*80)
    print("TESTING COMPLETE MAISI VAE LOADING")
    print("="*80)

    # Load checkpoint to inspect structure
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"Checkpoint type: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())[:10]}...")
        print(f"Total parameters: {len(checkpoint)}")

    print("\n" + "="*80)
    print("APPROACH 1: Direct State Dict Loading")
    print("="*80)

    # Try to create AutoencoderKL and load directly
    try:
        # Create AutoencoderKL - we'll try to infer the config from checkpoint
        # Based on our analysis: channels=(64, 128, 256), latent_channels=4

        print("\nAttempt 1: Standard configuration (3 levels)")
        model = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 128, 256),  # Correct parameter name
            latent_channels=4,
            num_res_blocks=(2, 2, 2),  # 2 blocks per level
            norm_num_groups=32,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

        # Load state dict
        result = model.load_state_dict(checkpoint, strict=False)

        print(f"\nLoading results:")
        print(f"  Missing keys: {len(result.missing_keys)}")
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")

        if result.missing_keys:
            print(f"\n  First 5 missing keys:")
            for key in list(result.missing_keys)[:5]:
                print(f"    - {key}")

        if result.unexpected_keys:
            print(f"\n  First 5 unexpected keys:")
            for key in list(result.unexpected_keys)[:5]:
                print(f"    - {key}")

        # Count how many weights were actually loaded
        model_params = set(model.state_dict().keys())
        checkpoint_params = set(checkpoint.keys())

        loaded_params = model_params & checkpoint_params
        print(f"\n  Intersection: {len(loaded_params)}/{len(checkpoint_params)} checkpoint parameters")
        print(f"  Coverage: {len(loaded_params)/len(checkpoint_params)*100:.1f}%")

        if len(loaded_params) == len(checkpoint_params):
            print("\n🎉 SUCCESS! 100% of MAISI weights loaded!")
            return model, True
        else:
            print(f"\n⚠ Partial loading: {len(loaded_params)}/{len(checkpoint_params)} parameters")

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("APPROACH 2: Load as Complete Module")
    print("="*80)

    # Alternative: Try loading the checkpoint as a complete saved model
    try:
        print("\nAttempt 2: Load as torch model")

        # Some checkpoints save the entire model
        if hasattr(checkpoint, 'eval'):
            print("  Checkpoint is a complete model!")
            model = checkpoint
            model.eval()
            print("  ✓ Loaded complete model")
            return model, True
        else:
            print("  Checkpoint is state dict only")

    except Exception as e:
        print(f"  Failed: {e}")

    print("\n" + "="*80)
    print("APPROACH 3: Wrap State Dict Directly")
    print("="*80)

    # Last resort: Create a wrapper that uses the state dict directly
    try:
        print("\nAttempt 3: Create custom wrapper")

        # This approach creates a minimal wrapper around the checkpoint
        model = create_maisi_wrapper(checkpoint)
        print("  ✓ Created custom wrapper")
        return model, True

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()

    return None, False


def create_maisi_wrapper(state_dict):
    """
    Create a minimal wrapper around MAISI state dict

    This wrapper provides encode() and decode() methods
    while keeping the original checkpoint structure intact.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    class MAISIWrapper(nn.Module):
        """
        Minimal wrapper for MAISI VAE checkpoint

        Loads the checkpoint state dict directly and provides
        encode/decode interface compatible with our diffusion model.
        """
        def __init__(self, state_dict):
            super().__init__()

            # Store state dict
            self.state_dict_loaded = state_dict

            # Create parameter dict
            for key, value in state_dict.items():
                # Convert to Parameter
                self.register_buffer(key.replace('.', '_'), value)

            self.latent_channels = 4
            self.spatial_compression = 8

            print(f"    Wrapped {len(state_dict)} parameters")

        def encode(self, x):
            """
            Encode using MAISI VAE

            Note: This is a placeholder - actual encoding requires
            reconstructing the full forward pass from the state dict.
            """
            raise NotImplementedError(
                "Direct state dict encoding not implemented. "
                "Need to use proper AutoencoderKL with matched architecture."
            )

        def decode(self, z):
            """Decode placeholder"""
            raise NotImplementedError(
                "Direct state dict decoding not implemented. "
                "Need to use proper AutoencoderKL with matched architecture."
            )

    return MAISIWrapper(state_dict)


def test_encode_decode(model):
    """Test encode/decode with loaded model"""
    print("\n" + "="*80)
    print("TESTING ENCODE/DECODE")
    print("="*80)

    try:
        model.eval()

        # Test with 8-slice volume (minimum for 3×3×3 kernels)
        print("\nTesting with 8-slice volume (512×512)...")
        x = torch.randn(1, 1, 8, 512, 512)

        with torch.no_grad():
            # Encode
            print(f"  Input shape: {x.shape}")
            z = model.encode(x)

            if hasattr(z, 'sample'):
                z = z.sample()

            print(f"  Latent shape: {z.shape}")

            # Decode
            x_recon = model.decode(z)
            print(f"  Reconstructed shape: {x_recon.shape}")

            # Compute reconstruction error
            mse = torch.mean((x - x_recon) ** 2).item()
            print(f"\n  Reconstruction MSE: {mse:.6f}")

            if mse < 0.1:
                print("  ✓ Low MSE - pretrained weights working correctly!")
                return True
            else:
                print("  ⚠ High MSE - weights may not be fully loaded")
                return False

    except Exception as e:
        print(f"\n✗ Encode/decode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test loading
    model, success = test_complete_maisi_loading()

    if success and model is not None:
        # Test encode/decode
        encode_success = test_encode_decode(model)

        if encode_success:
            print("\n" + "="*80)
            print("✅ COMPLETE MAISI VAE LOADING SUCCESSFUL!")
            print("="*80)
            print("\nModel is ready for use in slice interpolation.")
        else:
            print("\n" + "="*80)
            print("⚠ Model loaded but encode/decode needs verification")
            print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ Failed to load complete MAISI VAE")
        print("="*80)
        print("\nRecommendation: Continue with partial loading approach (14%)")
        print("or investigate MAISI model architecture further.")
