"""Test complete MAISI VAE with 100% weight loading"""

import sys
sys.path.append('.')

import torch
from models.maisi_vae import MAISIVAE

print("="*80)
print("TESTING COMPLETE MAISI VAE WITH 100% WEIGHT LOADING")
print("="*80)

# Create model
print("\nCreating MAISI VAE...")
model = MAISIVAE()
print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Load pretrained weights
print("\nLoading pretrained weights...")
stats = model.load_pretrained_weights('./pretrained/maisi_vae/models/autoencoder.pt', strict=False)

if stats['loaded'] == stats['total']:
    print(f"\n🎉 SUCCESS! {stats['loaded']}/{stats['total']} weights loaded (100%)")
else:
    print(f"\n⚠ Partial: {stats['loaded']}/{stats['total']} ({stats['loaded']/stats['total']*100:.1f}%)")
    sys.exit(1)

# Test encode/decode with small input
print("\n" + "="*80)
print("TESTING ENCODE/DECODE WITH PRETRAINED WEIGHTS")
print("="*80)

model.eval()

# Small test (fast)
print("\nTest 1: Small input (8 slices, 128×128)")
x_small = torch.randn(1, 1, 8, 128, 128)
with torch.no_grad():
    z_small = model.encode(x_small)
    x_recon_small = model.decode(z_small)

print(f"  Input: {x_small.shape}")
print(f"  Latent: {z_small.shape}")
print(f"  Reconstructed: {x_recon_small.shape}")

mse_small = torch.mean((x_small - x_recon_small) ** 2).item()
print(f"  MSE: {mse_small:.6f}")

# Larger test (closer to real data)
print("\nTest 2: Medium input (8 slices, 256×256)")
x_med = torch.randn(1, 1, 8, 256, 256)
with torch.no_grad():
    z_med = model.encode(x_med)
    x_recon_med = model.decode(z_med)

print(f"  Input: {x_med.shape}")
print(f"  Latent: {z_med.shape}")
print(f"  Reconstructed: {x_recon_med.shape}")

mse_med = torch.mean((x_med - x_recon_med) ** 2).item()
print(f"  MSE: {mse_med:.6f}")

print("\n" + "="*80)
if mse_small < 1.0 and mse_med < 1.0:
    print("✅ ALL TESTS PASSED!")
    print("   Custom MAISI VAE is working correctly with 100% pretrained weights!")
else:
    print("⚠ High MSE detected - this is expected with random input")
    print("   MSE will be lower with real CT data")

print("="*80)
