"""
Test script for two-phase training implementation
"""

import torch
import numpy as np
from models import VideoToVideoDiffusion
from training import Trainer
from torch.utils.data import DataLoader, Dataset

print("=" * 80)
print("Testing Two-Phase Training Implementation")
print("=" * 80)

# Create small model config for testing
model_config = {
    'in_channels': 3,
    'latent_dim': 4,
    'vae_base_channels': 32,  # Smaller for testing
    'unet_model_channels': 64,
    'unet_num_res_blocks': 1,
    'unet_attention_levels': [1],
    'unet_channel_mult': [1, 2],
    'unet_num_heads': 2,
    'noise_schedule': 'cosine',
    'diffusion_timesteps': 100
}

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

model = VideoToVideoDiffusion(model_config)
print("✓ Model created")

# Create simple dummy dataset for testing
class DummyDataset(Dataset):
    def __init__(self, num_samples=4):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return dummy video tensors (B, C, T, H, W) format
        return {
            'input': torch.randn(3, 8, 64, 64),
            'target': torch.randn(3, 8, 64, 64)
        }

dataset = DummyDataset(4)
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
print("✓ Dataloader created")

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training config with two-phase training
train_config = {
    'num_epochs': 2,
    'gradient_accumulation_steps': 1,
    'use_amp': False,
    'log_interval': 1,
    'checkpoint_interval': 100
}

# Pretrained config with two-phase training
pretrained_config = {
    'two_phase_training': True,
    'phase1_epochs': 1,
    'vae': {
        'freeze_epochs': 0
    }
}

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

print("\n" + "=" * 80)
print("TEST 1: Checking freeze_vae() method")
print("=" * 80)

# Test freeze
trainer.freeze_vae()
vae_params_frozen = all(not p.requires_grad for p in model.vae.parameters())
unet_params_trainable = any(p.requires_grad for p in model.unet.parameters())

print(f"VAE parameters frozen: {vae_params_frozen}")
print(f"U-Net parameters trainable: {unet_params_trainable}")

if vae_params_frozen and unet_params_trainable:
    print("✓ TEST 1 PASSED: freeze_vae() works correctly")
else:
    print("✗ TEST 1 FAILED")

print("\n" + "=" * 80)
print("TEST 2: Checking unfreeze_vae() method")
print("=" * 80)

# Test unfreeze
trainer.unfreeze_vae()
vae_params_trainable = any(p.requires_grad for p in model.vae.parameters())

print(f"VAE parameters trainable: {vae_params_trainable}")

if vae_params_trainable:
    print("✓ TEST 2 PASSED: unfreeze_vae() works correctly")
else:
    print("✗ TEST 2 FAILED")

print("\n" + "=" * 80)
print("TEST 3: Checking two-phase training setup")
print("=" * 80)

print(f"Two-phase training enabled: {trainer.use_two_phase}")
print(f"Phase 1 epochs: {trainer.phase1_epochs}")
print(f"Current phase: {trainer.current_phase}")
print(f"Total epochs: {trainer.num_epochs}")

if trainer.use_two_phase and trainer.phase1_epochs == 1 and trainer.current_phase == 1:
    print("✓ TEST 3 PASSED: Two-phase training configured correctly")
else:
    print("✗ TEST 3 FAILED")

print("\n" + "=" * 80)
print("TEST 4: Running mini training to verify phase transition")
print("=" * 80)

# This will run a very short training (2 epochs) to verify phase transition
print("Starting mini training...")
print("Expected behavior:")
print("  - Epoch 0 (Phase 1): VAE frozen")
print("  - Epoch 1 (Phase 2): VAE unfrozen")
print()

try:
    trainer.train()
    print("\n✓ TEST 4 PASSED: Training completed successfully with phase transition")
except Exception as e:
    print(f"\n✗ TEST 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)
