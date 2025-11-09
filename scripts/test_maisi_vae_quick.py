"""Quick test of custom MAISI VAE"""

import sys
sys.path.append('.')

import torch
from models.maisi_vae import MAISIVAE

print("Creating MAISI VAE...")
model = MAISIVAE()

print("Model created successfully!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass (small input)
print("\nTesting forward pass...")
x = torch.randn(1, 1, 8, 64, 64)
print(f"Input: {x.shape}")

z = model.encode(x)
print(f"Encoded: {z.shape}")

x_recon = model.decode(z)
print(f"Decoded: {x_recon.shape}")

print("\n✓ Forward pass successful!")

# Load checkpoint and compare keys
print("\nLoading checkpoint to compare keys...")
checkpoint = torch.load('./pretrained/maisi_vae/models/autoencoder.pt', map_location='cpu', weights_only=False)

model_keys = set(model.state_dict().keys())
checkpoint_keys = set(checkpoint.keys())

matching_keys = model_keys & checkpoint_keys
print(f"\nKey matching:")
print(f"  Model keys: {len(model_keys)}")
print(f"  Checkpoint keys: {len(checkpoint_keys)}")
print(f"  Matching: {len(matching_keys)} ({len(matching_keys)/len(checkpoint_keys)*100:.1f}%)")

if len(matching_keys) < len(checkpoint_keys):
    print("\nMissing keys (first 10):")
    missing = checkpoint_keys - model_keys
    for i, key in enumerate(sorted(missing)[:10]):
        print(f"  {i+1}. {key}")

if len(model_keys) > len(checkpoint_keys):
    print("\nExtra keys in model (first 10):")
    extra = model_keys - checkpoint_keys
    for i, key in enumerate(sorted(extra)[:10]):
        print(f"  {i+1}. {key}")
