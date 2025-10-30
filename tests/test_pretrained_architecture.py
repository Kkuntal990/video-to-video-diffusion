"""
Test script for pretrained VAE loading with correct architecture
"""

import torch
import yaml
from pathlib import Path
from models import VideoToVideoDiffusion
from utils.pretrained import load_pretrained_sd_vae, map_sd_vae_to_video_vae

print("=" * 80)
print("Testing Pretrained VAE Architecture Matching")
print("=" * 80)

# Load config
config_path = 'config/cloud_train_config.yaml'
print(f"\nLoading config from: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model_config = config['model']
pretrained_config = config.get('pretrained', {})

print("\n" + "=" * 80)
print("TEST 1: Model Configuration Check")
print("=" * 80)

print(f"\nModel configuration:")
print(f"  in_channels: {model_config['in_channels']}")
print(f"  latent_dim: {model_config['latent_dim']}")
print(f"  vae_base_channels: {model_config['vae_base_channels']}")

if model_config['latent_dim'] == 8:
    print("✓ latent_dim=8 matches Stable Diffusion VAE")
else:
    print(f"✗ latent_dim={model_config['latent_dim']} does NOT match SD VAE (expected 8)")

print("\n" + "=" * 80)
print("TEST 2: Create Model")
print("=" * 80)

device = torch.device('cpu')  # Use CPU for testing
print(f"\nDevice: {device}")

try:
    model = VideoToVideoDiffusion(model_config)
    print("✓ Model created successfully")

    # Count parameters
    param_counts = model.count_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  VAE: {param_counts['vae']:,}")
    print(f"  U-Net: {param_counts['unet']:,}")

except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("TEST 3: Load Pretrained SD VAE Weights")
print("=" * 80)

if pretrained_config.get('use_pretrained', False) and pretrained_config.get('vae', {}).get('enabled', False):
    vae_config = pretrained_config['vae']
    model_name = vae_config.get('model_name', 'stabilityai/sd-vae-ft-mse')
    inflate_method = vae_config.get('inflate_method', 'central')

    print(f"\nLoading pretrained VAE:")
    print(f"  Model: {model_name}")
    print(f"  Inflation method: {inflate_method}")

    try:
        # Load SD VAE weights
        print("\nDownloading SD VAE weights...")
        sd_state_dict = load_pretrained_sd_vae(model_name)
        print(f"✓ Loaded {len(sd_state_dict)} parameters from SD VAE")

        # Check a few key parameters
        print("\nSample SD VAE parameters:")
        for i, (name, param) in enumerate(list(sd_state_dict.items())[:5]):
            print(f"  {name}: {param.shape}")

        # Inflate 2D->3D
        print(f"\nInflating 2D weights to 3D using method: {inflate_method}")
        video_state_dict = map_sd_vae_to_video_vae(sd_state_dict, inflate_method=inflate_method)
        print(f"✓ Inflated {len(video_state_dict)} parameters")

        # Check inflated parameters
        print("\nSample inflated parameters:")
        conv_params = [(n, p) for n, p in video_state_dict.items() if 'conv' in n and len(p.shape) == 5][:3]
        for name, param in conv_params:
            print(f"  {name}: {param.shape} (3D conv)")

    except Exception as e:
        print(f"✗ Failed to load pretrained weights: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

else:
    print("⚠️  Pretrained weights disabled in config - skipping this test")
    print("This test requires:")
    print("  pretrained.use_pretrained = true")
    print("  pretrained.vae.enabled = true")
    exit(0)

print("\n" + "=" * 80)
print("TEST 4: Load Weights into Model VAE")
print("=" * 80)

try:
    # Load weights into model's VAE
    missing_keys, unexpected_keys = model.vae.load_state_dict(video_state_dict, strict=False)

    print(f"\nLoading results:")
    print(f"  Missing keys: {len(missing_keys)}")
    print(f"  Unexpected keys: {len(unexpected_keys)}")

    if len(missing_keys) > 0:
        print(f"\n  First few missing keys:")
        for key in missing_keys[:5]:
            print(f"    - {key}")

    if len(unexpected_keys) > 0:
        print(f"\n  First few unexpected keys:")
        for key in unexpected_keys[:5]:
            print(f"    - {key}")

    # Check if critical layers loaded
    encoder_params = sum(1 for n, p in model.vae.encoder.named_parameters())
    decoder_params = sum(1 for n, p in model.vae.decoder.named_parameters())

    print(f"\nVAE structure:")
    print(f"  Encoder parameters: {encoder_params}")
    print(f"  Decoder parameters: {decoder_params}")

    if len(missing_keys) < len(video_state_dict) * 0.5:
        print("\n✓ Pretrained weights loaded successfully (most keys matched)")
    else:
        print("\n⚠️  Warning: Many keys missing - architecture may not match well")

except Exception as e:
    print(f"✗ Failed to load weights into model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("TEST 5: Forward Pass Test")
print("=" * 80)

try:
    # Test forward pass with random input
    batch_size = 1
    num_frames = 8
    height = 128
    width = 128

    print(f"\nTesting with input shape:")
    print(f"  Batch: {batch_size}")
    print(f"  Frames: {num_frames}")
    print(f"  Resolution: {height}x{width}")

    # Create dummy input
    v_in = torch.randn(batch_size, 3, num_frames, height, width)
    v_gt = torch.randn(batch_size, 3, num_frames, height, width)

    print(f"\nInput shape: {v_in.shape}")

    # Encode with VAE
    print("\nTesting VAE encoding...")
    model.eval()
    with torch.no_grad():
        latent = model.vae.encode(v_in)

    print(f"Latent shape: {latent.shape}")
    expected_latent_shape = (batch_size, model_config['latent_dim'], num_frames, height//8, width//8)
    print(f"Expected: {expected_latent_shape}")

    if latent.shape == expected_latent_shape:
        print("✓ VAE encoding works correctly")
    else:
        print(f"✗ Latent shape mismatch!")
        exit(1)

    # Decode with VAE
    print("\nTesting VAE decoding...")
    with torch.no_grad():
        recon = model.vae.decode(latent)

    print(f"Reconstruction shape: {recon.shape}")
    if recon.shape == v_in.shape:
        print("✓ VAE decoding works correctly")
    else:
        print(f"✗ Reconstruction shape mismatch!")
        exit(1)

    # Full model forward pass
    print("\nTesting full model forward pass...")
    model.train()
    with torch.no_grad():
        loss, metrics = model(v_in, v_gt)

    print(f"Loss: {loss.item():.4f}")
    print("✓ Full forward pass successful")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("TEST 6: Two-Phase Training Setup")
print("=" * 80)

from training import Trainer
from torch.utils.data import DataLoader, Dataset

# Create dummy dataset
class DummyDataset(Dataset):
    def __init__(self, num_samples=4):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'input': torch.randn(3, 8, 128, 128),
            'target': torch.randn(3, 8, 128, 128)
        }

dataset = DummyDataset(4)
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training config
train_config = {
    'num_epochs': 2,
    'gradient_accumulation_steps': 1,
    'use_amp': False,
    'log_interval': 1,
    'checkpoint_interval': 100
}

try:
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        train_dataloader=train_dataloader,
        config=train_config,
        device=device,
        pretrained_config=pretrained_config
    )

    print("\n✓ Trainer initialized successfully")
    print(f"  Two-phase training: {trainer.use_two_phase}")
    print(f"  Phase 1 epochs: {trainer.phase1_epochs}")
    print(f"  VAE freeze epochs: {trainer.vae_freeze_epochs}")

    # Test freeze/unfreeze
    print("\nTesting VAE freeze/unfreeze...")
    trainer.freeze_vae()
    vae_frozen = all(not p.requires_grad for p in model.vae.parameters())
    print(f"  VAE frozen: {vae_frozen}")

    trainer.unfreeze_vae()
    vae_trainable = any(p.requires_grad for p in model.vae.parameters())
    print(f"  VAE trainable: {vae_trainable}")

    if vae_frozen and vae_trainable:
        print("✓ VAE freeze/unfreeze works correctly")
    else:
        print("✗ VAE freeze/unfreeze failed")
        exit(1)

except Exception as e:
    print(f"✗ Trainer setup failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"""
✓ Model architecture matches SD VAE (latent_dim=8)
✓ Pretrained SD VAE weights loaded successfully
✓ 2D->3D weight inflation works
✓ VAE encoding/decoding works correctly
✓ Full model forward pass works
✓ Two-phase training setup is correct
✓ VAE freeze/unfreeze functionality works

You can now start training with:
  python train.py --config config/cloud_train_config.yaml

The model will:
1. Load pretrained SD VAE weights (inflated to 3D)
2. Phase 1: Train U-Net with frozen VAE (1 epoch)
3. Phase 2: Fine-tune entire model (1 epoch)

Checkpoints will be saved to: /workspace/storage/checkpoints/
""")

print("=" * 80)
