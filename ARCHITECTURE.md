# Video-to-Video Diffusion Model Architecture

Complete architecture specification with layer dimensions and data flow diagrams.

---

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [VAE Architecture](#vae-architecture)
4. [U-Net Architecture](#unet-architecture)
5. [Diffusion Process](#diffusion-process)
6. [Training Strategy](#training-strategy)
7. [Layer Dimensions](#layer-dimensions)

---

## Overview

**Model Type**: Video-to-Video Diffusion with 3D VAE
**Task**: Medical CT scan reconstruction and enhancement
**Input**: Source CT video (8 slices, 128×128)
**Output**: Enhanced CT video (8 slices, 128×128)

### High-Level Architecture

```
Input Video (B×3×8×128×128)
          ↓
    ┌─────────────┐
    │   VAE       │  Encode to latent space
    │   Encoder   │  (B×4×8×16×16)
    └─────────────┘
          ↓
    ┌─────────────┐
    │   Add       │  Forward diffusion
    │   Noise     │  (training only)
    └─────────────┘
          ↓
    ┌─────────────┐
    │   U-Net     │  Denoise latent
    │   Denoiser  │  (B×4×8×16×16)
    └─────────────┘
          ↓
    ┌─────────────┐
    │   VAE       │  Decode to video
    │   Decoder   │  (B×3×8×128×128)
    └─────────────┘
          ↓
Output Video (B×3×8×128×128)
```

### Model Statistics

| Component | Parameters | Input Shape | Output Shape |
|-----------|-----------|-------------|--------------|
| **VAE Encoder** | 86M | (B,3,8,128,128) | (B,4,8,16,16) |
| **VAE Decoder** | 86M | (B,4,8,16,16) | (B,3,8,128,128) |
| **U-Net (4-level)** | 270M | (B,4,8,16,16) | (B,4,8,16,16) |
| **Total** | **441M** | - | - |

**Compression Ratio**: 8× spatial (128→16), no temporal compression

---

## Model Architecture

### Complete Forward Pass

```
┌─────────────────────────────────────────────────────────────────┐
│                    Video-to-Video Diffusion                      │
│                                                                   │
│  Input (v_in)              Target (v_gt)                         │
│  [B,3,8,128,128]          [B,3,8,128,128]                       │
│        │                        │                                │
│        ├────────────────────────┤                                │
│        │    VAE Encoding        │                                │
│        ▼                        ▼                                │
│   z_in [B,4,8,16,16]      z_gt [B,4,8,16,16]                    │
│        │                        │                                │
│        │              ┌─────────┴─────────┐                      │
│        │              │  Add Noise (t)    │                      │
│        │              │  ε ~ N(0,1)       │                      │
│        │              └─────────┬─────────┘                      │
│        │                        │                                │
│        │                  z_noisy [B,4,8,16,16]                  │
│        │                        │                                │
│        │              ┌─────────┴─────────┐                      │
│        └──────────────►   U-Net Denoiser  │                      │
│         (condition)    │   (predict ε)     │                      │
│                        └─────────┬─────────┘                      │
│                                  │                                │
│                           ε_pred [B,4,8,16,16]                    │
│                                  │                                │
│                        ┌─────────┴─────────┐                      │
│                        │   MSE Loss        │                      │
│                        │   L = ||ε - ε_pred||²                   │
│                        └───────────────────┘                      │
│                                                                   │
│  Inference Only:                                                  │
│  z_pred → VAE Decoder → v_out [B,3,8,128,128]                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## VAE Architecture

### VAE Encoder

**Purpose**: Compress video from pixel space to latent space

```
Input: (B, 3, T, H, W) = (B, 3, 8, 128, 128)

┌────────────────────────────────────────────────────────────┐
│                     VAE Encoder                             │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Conv_in (3→128)                                            │
│  [B,3,8,128,128] ──→ [B,128,8,128,128]                     │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │  Down Block 1                         │                  │
│  │  • ResBlock3D (128→128)               │                  │
│  │  • ResBlock3D (128→128)               │                  │
│  │  • Downsample (128→256, /2 spatial)   │                  │
│  │  [B,128,8,128,128] → [B,256,8,64,64] │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │  Down Block 2                         │                  │
│  │  • ResBlock3D (256→256)               │                  │
│  │  • ResBlock3D (256→256)               │                  │
│  │  • Downsample (256→512, /2 spatial)   │                  │
│  │  [B,256,8,64,64] → [B,512,8,32,32]   │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │  Down Block 3                         │                  │
│  │  • ResBlock3D (512→512)               │                  │
│  │  • ResBlock3D (512→512)               │                  │
│  │  • Downsample (512→512, /2 spatial)   │                  │
│  │  [B,512,8,32,32] → [B,512,8,16,16]   │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │  Middle Blocks                        │                  │
│  │  • ResBlock3D (512→512)               │                  │
│  │  • ResBlock3D (512→512)               │                  │
│  │  [B,512,8,16,16] → [B,512,8,16,16]   │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  Conv_out (512→4, kernel=1)                                 │
│  [B,512,8,16,16] ──→ [B,4,8,16,16]                         │
│                                                             │
└────────────────────────────────────────────────────────────┘

Output: (B, 4, 8, 16, 16) - Latent representation
```

### VAE Decoder

**Purpose**: Reconstruct video from latent space to pixel space

```
Input: (B, 4, T, h, w) = (B, 4, 8, 16, 16)

┌────────────────────────────────────────────────────────────┐
│                     VAE Decoder                             │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Conv_in (4→512)                                            │
│  [B,4,8,16,16] ──→ [B,512,8,16,16]                         │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │  Middle Blocks                        │                  │
│  │  • ResBlock3D (512→512)               │                  │
│  │  • ResBlock3D (512→512)               │                  │
│  │  [B,512,8,16,16] → [B,512,8,16,16]   │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │  Up Block 1                           │                  │
│  │  • Upsample (512→512, ×2 spatial)     │                  │
│  │  • ResBlock3D (512→512)               │                  │
│  │  • ResBlock3D (512→512)               │                  │
│  │  [B,512,8,16,16] → [B,512,8,32,32]   │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │  Up Block 2                           │                  │
│  │  • Upsample (512→256, ×2 spatial)     │                  │
│  │  • ResBlock3D (256→256)               │                  │
│  │  • ResBlock3D (256→256)               │                  │
│  │  [B,512,8,32,32] → [B,256,8,64,64]   │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │  Up Block 3                           │                  │
│  │  • Upsample (256→128, ×2 spatial)     │                  │
│  │  • ResBlock3D (128→128)               │                  │
│  │  • ResBlock3D (128→128)               │                  │
│  │  [B,256,8,64,64] → [B,128,8,128,128] │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  Conv_out (128→3, kernel=3)                                 │
│  [B,128,8,128,128] ──→ [B,3,8,128,128]                     │
│                                                             │
└────────────────────────────────────────────────────────────┘

Output: (B, 3, 8, 128, 128) - Reconstructed video
```

### Building Blocks

#### ResBlock3D
```
Input (C channels)
      │
      ├────────────────┐ (skip connection)
      │                │
      ▼                │
  Conv3D (C→C)         │
  GroupNorm            │
  SiLU                 │
      │                │
      ▼                │
  Conv3D (C→C)         │
  GroupNorm            │
      │                │
      ▼                │
    Add ◄──────────────┘
      │
      ▼
    SiLU
      │
   Output
```

#### DownsampleBlock
```
Input (C_in channels, H×W spatial)
      │
      ▼
  Conv3D (C_in → C_out)
  kernel: (3,4,4)
  stride: (1,2,2)  ← Downsample spatial only
      │
      ▼
  GroupNorm
      │
      ▼
    SiLU
      │
   Output (C_out channels, H/2 × W/2 spatial)
```

#### UpsampleBlock
```
Input (C_in channels, H×W spatial)
      │
      ▼
  ConvTranspose3D (C_in → C_out)
  kernel: (3,4,4)
  stride: (1,2,2)  ← Upsample spatial only
      │
      ▼
  GroupNorm
      │
      ▼
    SiLU
      │
   Output (C_out channels, H×2 × W×2 spatial)
```

---

## U-Net Architecture

### 3D U-Net Denoiser

**Purpose**: Denoise latent representations conditioned on input

```
Inputs:
  - Noisy latent: (B, 4, 8, 16, 16)
  - Timestep: (B,)
  - Condition: (B, 4, 8, 16, 16)

┌──────────────────────────────────────────────────────────────┐
│                      3D U-Net Denoiser                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Time Embedding (Sinusoidal)                                  │
│  t [B] ──→ t_emb [B,1024]                                    │
│                                                               │
│  Input Processing                                             │
│  z_noisy [B,4,8,16,16] ──→ Conv_in ──→ [B,128,8,16,16]      │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │                 Encoder Path (4 levels)              │     │
│  ├─────────────────────────────────────────────────────┤     │
│  │                                                      │     │
│  │  Level 0: [B,128,8,16,16]                           │     │
│  │  • ResBlock (128, t_emb)                            │ ────┼──┐
│  │  • ResBlock (128, t_emb)                            │     │  │
│  │  • Downsample → [B,256,8,8,8]                       │     │  │
│  │                                                      │     │  │
│  │  Level 1: [B,256,8,8,8]                             │     │  │
│  │  • ResBlock (256, t_emb)                            │ ────┼──┼──┐
│  │  • ResBlock (256, t_emb)                            │     │  │  │
│  │  • Attention (heads=8)                               │     │  │  │
│  │  • Downsample → [B,512,8,4,4]                       │     │  │  │
│  │                                                      │     │  │  │
│  │  Level 2: [B,512,8,4,4]                             │     │  │  │
│  │  • ResBlock (512, t_emb)                            │ ────┼──┼──┼──┐
│  │  • ResBlock (512, t_emb)                            │     │  │  │  │
│  │  • Attention (heads=8)                               │     │  │  │  │
│  │  • Downsample → [B,512,8,2,2]                       │     │  │  │  │
│  │                                                      │     │  │  │  │
│  │  Level 3 (Bottleneck): [B,512,8,2,2]                │     │  │  │  │
│  │  • ResBlock (512, t_emb)                            │     │  │  │  │
│  │  • ResBlock (512, t_emb)                            │     │  │  │  │
│  └──────────────────────────────────────────────────────┘     │  │  │  │
│                                                               │  │  │  │
│  ┌─────────────────────────────────────────────────────┐     │  │  │  │
│  │            Middle Block (at bottleneck)              │     │  │  │  │
│  │  • ResBlock (512, t_emb)                            │     │  │  │  │
│  │  • TemporalAttention (heads=8)                       │     │  │  │  │
│  │  • ResBlock (512, t_emb)                            │     │  │  │  │
│  └──────────────────────────────────────────────────────┘     │  │  │  │
│                                                               │  │  │  │
│  ┌─────────────────────────────────────────────────────┐     │  │  │  │
│  │                 Decoder Path (4 levels)              │     │  │  │  │
│  ├─────────────────────────────────────────────────────┤     │  │  │  │
│  │                                                      │     │  │  │  │
│  │  Level 3 → 2: [B,512,8,2,2]                         │     │  │  │  │
│  │  • Concat skip ← [B,512,8,2,2] ──────────────────────────┼──┼──┼──┘
│  │  • ResBlock (1024→512, t_emb)                       │     │  │  │
│  │  • ResBlock (512, t_emb)                            │     │  │  │
│  │  • Upsample → [B,512,8,4,4]                         │     │  │  │
│  │                                                      │     │  │  │
│  │  Level 2 → 1: [B,512,8,4,4]                         │     │  │  │
│  │  • Concat skip ← [B,512,8,4,4] ──────────────────────────┼──┼──┘
│  │  • ResBlock (1024→512, t_emb)                       │     │  │
│  │  • ResBlock (512, t_emb)                            │     │  │
│  │  • Attention (heads=8)                               │     │  │
│  │  • Upsample → [B,512,8,8,8]                         │     │  │
│  │                                                      │     │  │
│  │  Level 1 → 0: [B,512,8,8,8]                         │     │  │
│  │  • Concat skip ← [B,256,8,8,8] ──────────────────────────┼──┼──┘
│  │  • ResBlock (768→256, t_emb)                        │     │  │
│  │  • ResBlock (256, t_emb)                            │     │  │
│  │  • Attention (heads=8)                               │     │  │
│  │  • Upsample → [B,256,8,16,16]                       │     │  │
│  │                                                      │     │  │
│  │  Level 0 (final): [B,256,8,16,16]                   │     │  │
│  │  • Concat skip ← [B,128,8,16,16] ────────────────────────┼──┘
│  │  • ResBlock (384→128, t_emb)                        │     │
│  │  • ResBlock (128, t_emb)                            │     │
│  │  • ResBlock (128, t_emb)                            │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                               │
│  Output Projection                                            │
│  [B,128,8,16,16] ──→ Conv_out ──→ [B,4,8,16,16]             │
│                                                               │
└──────────────────────────────────────────────────────────────┘

Output: ε_pred [B,4,8,16,16] - Predicted noise
```

### U-Net Building Blocks

#### ResBlockWithTimeEmbed
```
Input: z [B,C,T,H,W], t_emb [B,1024]

      z                  t_emb
      │                    │
      ├────────────┐       │
      │            │       ▼
      ▼            │   Linear (1024→C)
  GroupNorm        │       │
      │            │       ▼
      ▼            │    SiLU
    SiLU           │       │
      │            │       └──────┐
      ▼            │              │
  Conv3D (C→C)     │              │
      │            │              │
      ▼            │              │
  GroupNorm        │              │
      │            │              │
   Add ◄───────────┼──────────────┘  (add time emb)
      │            │
      ▼            │
    SiLU           │
      │            │
      ▼            │
  Conv3D (C→C)     │
      │            │
      ▼            │
    Add ◄──────────┘  (skip connection)
      │
   Output
```

#### SpatialTransformer3D (Attention)
```
Input: [B,C,T,H,W]

      │
      ├───────────┐ (skip)
      │           │
      ▼           │
  Reshape to      │
  [B,T×H×W,C]     │
      │           │
      ▼           │
  LayerNorm       │
      │           │
      ▼           │
  Multi-Head      │
  Self-Attention  │
  (heads=8)       │
      │           │
      ▼           │
  Feed-Forward    │
  Network         │
      │           │
      ▼           │
  Reshape to      │
  [B,C,T,H,W]     │
      │           │
      ▼           │
    Add ◄─────────┘
      │
   Output
```

---

## Diffusion Process

### Forward Diffusion (Training)

```
Ground Truth Latent: z_gt [B,4,8,16,16]
                     │
                     ▼
        Sample timestep t ~ Uniform(0, 1000)
                     │
                     ▼
        Sample noise ε ~ N(0, I)
                     │
                     ▼
        z_noisy = √(ᾱ_t) · z_gt + √(1-ᾱ_t) · ε
                     │
        [B,4,8,16,16]
```

**Noise Schedule**: Cosine schedule
- `t=0`: Clean latent (no noise)
- `t=1000`: Pure noise

### Reverse Diffusion (Inference)

```
Pure Noise: z_T ~ N(0, I) [B,4,8,16,16]
              │
              ▼
   ┌──────────────────┐
   │  For t = T to 1: │
   │                  │
   │  1. Predict ε:   │
   │     ε = UNet(z_t, t, condition)
   │                  │
   │  2. Denoise:     │
   │     z_{t-1} = (z_t - √(1-ᾱ_t)·ε) / √(ᾱ_t)
   │                  │
   │  3. Add noise:   │
   │     z_{t-1} += σ_t · ε'  (if t > 1)
   │                  │
   └──────────────────┘
              │
              ▼
    Clean Latent: z_0 [B,4,8,16,16]
              │
              ▼
        VAE Decoder
              │
              ▼
    Output Video: v_out [B,3,8,128,128]
```

**Sampling**: DDIM (50 steps for fast inference)

---

## Training Strategy

### Two-Phase Training

```
┌─────────────────────────────────────────────────────────┐
│                    Phase 1 (Epoch 0)                     │
│                                                          │
│  VAE: FROZEN ❄️  (requires_grad=False)                  │
│  U-Net: TRAINING 🔥 (requires_grad=True)                │
│                                                          │
│  Learning Rates:                                         │
│  • VAE Encoder: 0 (frozen)                              │
│  • VAE Decoder: 0 (frozen)                              │
│  • U-Net: 1e-4                                          │
│                                                          │
│  Goal: Learn denoising without changing VAE             │
└─────────────────────────────────────────────────────────┘
               │
               ▼  Automatic transition
┌─────────────────────────────────────────────────────────┐
│                    Phase 2 (Epoch 1+)                    │
│                                                          │
│  VAE: TRAINING 🔥 (requires_grad=True)                  │
│  U-Net: TRAINING 🔥 (requires_grad=True)                │
│                                                          │
│  Learning Rates:                                         │
│  • VAE Encoder: 1e-5 (10× lower than U-Net)            │
│  • VAE Decoder: 1e-5 (10× lower than U-Net)            │
│  • U-Net: 1e-4                                          │
│                                                          │
│  Goal: Fine-tune entire model end-to-end                │
└─────────────────────────────────────────────────────────┘
```

**Benefits:**
1. Faster convergence (U-Net learns first)
2. Stable VAE features initially
3. Fine-tuning improves reconstruction quality
4. Better final performance

### Layer-Wise Learning Rates

| Component | Learning Rate | Multiplier | Parameters |
|-----------|--------------|------------|------------|
| U-Net (4-level) | 1e-4 | 1.0× | 270M |
| VAE Encoder | 1e-5 | 0.1× | 86M |
| VAE Decoder | 1e-5 | 0.1× | 86M |

**Rationale**: VAE learns slower to preserve stable latent representations and maintain training stability.

---

## Layer Dimensions

### Complete Dimension Table

#### VAE Encoder Dimensions

| Layer | Input Shape | Output Shape | Channels | Spatial |
|-------|-------------|--------------|----------|---------|
| Input | - | (B,3,8,128,128) | 3 | 128×128 |
| Conv_in | (B,3,8,128,128) | (B,128,8,128,128) | 128 | 128×128 |
| Down1.ResBlock | (B,128,8,128,128) | (B,128,8,128,128) | 128 | 128×128 |
| Down1.Downsample | (B,128,8,128,128) | (B,256,8,64,64) | 256 | 64×64 |
| Down2.ResBlock | (B,256,8,64,64) | (B,256,8,64,64) | 256 | 64×64 |
| Down2.Downsample | (B,256,8,64,64) | (B,512,8,32,32) | 512 | 32×32 |
| Down3.ResBlock | (B,512,8,32,32) | (B,512,8,32,32) | 512 | 32×32 |
| Down3.Downsample | (B,512,8,32,32) | (B,512,8,16,16) | 512 | 16×16 |
| Mid.ResBlock | (B,512,8,16,16) | (B,512,8,16,16) | 512 | 16×16 |
| Conv_out | (B,512,8,16,16) | (B,4,8,16,16) | 4 | 16×16 |
| **Output** | - | **(B,4,8,16,16)** | **4** | **16×16** |

**Compression**: 128×128 → 16×16 = **8× spatial reduction**

#### VAE Decoder Dimensions

| Layer | Input Shape | Output Shape | Channels | Spatial |
|-------|-------------|--------------|----------|---------|
| Input | - | (B,4,8,16,16) | 4 | 16×16 |
| Conv_in | (B,4,8,16,16) | (B,512,8,16,16) | 512 | 16×16 |
| Mid.ResBlock | (B,512,8,16,16) | (B,512,8,16,16) | 512 | 16×16 |
| Up1.Upsample | (B,512,8,16,16) | (B,512,8,32,32) | 512 | 32×32 |
| Up1.ResBlock | (B,512,8,32,32) | (B,512,8,32,32) | 512 | 32×32 |
| Up2.Upsample | (B,512,8,32,32) | (B,256,8,64,64) | 256 | 64×64 |
| Up2.ResBlock | (B,256,8,64,64) | (B,256,8,64,64) | 256 | 64×64 |
| Up3.Upsample | (B,256,8,64,64) | (B,128,8,128,128) | 128 | 128×128 |
| Up3.ResBlock | (B,128,8,128,128) | (B,128,8,128,128) | 128 | 128×128 |
| Conv_out | (B,128,8,128,128) | (B,3,8,128,128) | 3 | 128×128 |
| **Output** | - | **(B,3,8,128,128)** | **3** | **128×128** |

**Expansion**: 16×16 → 128×128 = **8× spatial expansion**

#### U-Net Dimensions (4-Level Architecture)

| Level | Stage | Input Shape | Output Shape | Channels | Spatial |
|-------|-------|-------------|--------------|----------|---------|
| - | Input | - | (B,4,8,16,16) | 4 | 16×16 |
| - | Conv_in | (B,4,8,16,16) | (B,128,8,16,16) | 128 | 16×16 |
| 0 | Encoder | (B,128,8,16,16) | (B,128,8,16,16) | 128 | 16×16 |
| 0 | Down | (B,128,8,16,16) | (B,256,8,8,8) | 256 | 8×8 |
| 1 | Encoder | (B,256,8,8,8) | (B,256,8,8,8) | 256 | 8×8 |
| 1 | Down | (B,256,8,8,8) | (B,512,8,4,4) | 512 | 4×4 |
| 2 | Encoder | (B,512,8,4,4) | (B,512,8,4,4) | 512 | 4×4 |
| 2 | Down | (B,512,8,4,4) | (B,512,8,2,2) | 512 | 2×2 |
| 3 | Bottleneck | (B,512,8,2,2) | (B,512,8,2,2) | 512 | 2×2 |
| 3 | Middle | (B,512,8,2,2) | (B,512,8,2,2) | 512 | 2×2 |
| 2 | Decoder+Skip | (B,1024,8,2,2) | (B,512,8,2,2) | 512 | 2×2 |
| 2 | Up | (B,512,8,2,2) | (B,512,8,4,4) | 512 | 4×4 |
| 1 | Decoder+Skip | (B,1024,8,4,4) | (B,512,8,4,4) | 512 | 4×4 |
| 1 | Up | (B,512,8,4,4) | (B,512,8,8,8) | 512 | 8×8 |
| 0 | Decoder+Skip | (B,768,8,8,8) | (B,256,8,8,8) | 256 | 8×8 |
| 0 | Up | (B,256,8,8,8) | (B,256,8,16,16) | 256 | 16×16 |
| - | Final+Skip | (B,384,8,16,16) | (B,128,8,16,16) | 128 | 16×16 |
| - | Conv_out | (B,128,8,16,16) | (B,4,8,16,16) | 4 | 16×16 |
| - | **Output** | - | **(B,4,8,16,16)** | **4** | **16×16** |

**Note**: Decoder channels increase due to skip connection concatenation. The 4th level doubles the model depth compared to 3-level architecture.

---

## Memory Requirements

### Training (Batch Size = 1, 128×128, 8 frames)

| Component | Activation Memory | Weight Memory | Total |
|-----------|------------------|---------------|-------|
| VAE Encoder | ~500 MB | ~344 MB | ~844 MB |
| U-Net (4-level) | ~3.5 GB | ~1.08 GB | ~4.58 GB |
| VAE Decoder | ~500 MB | ~344 MB | ~844 MB |
| Optimizer States | - | ~3.5 GB | ~3.5 GB |
| **Total (Estimated)** | **~4.5 GB** | **~5.3 GB** | **~9.8 GB** |

**Recommended**: 16GB+ GPU for 128×128 training

### Training (Batch Size = 1, 256×256, 16 frames) - Production Config

| Component | Memory (without grad checkpoint) | Memory (with grad checkpoint) |
|-----------|----------------------------------|-------------------------------|
| Model Weights (FP16) | ~2 GB | ~2 GB |
| Activations | ~16 GB | ~8 GB |
| Gradients | ~2 GB | ~2 GB |
| Optimizer States | ~3.5 GB | ~3.5 GB |
| Overhead | ~1 GB | ~1 GB |
| **Total (Estimated)** | **~24.5 GB** | **~16.5 GB** |

**Critical**: Requires gradient checkpointing enabled for V100 32GB!

### Inference (Batch Size = 1)

| Component | Memory |
|-----------|--------|
| Model Weights | ~1.76 GB |
| Activations | ~1.5 GB |
| **Total (Estimated)** | **~3.3 GB** |

**Recommended**: 8GB+ GPU for inference

---

## Configuration Summary

### Current Training Config

```yaml
model:
  in_channels: 3
  latent_dim: 4
  vae_base_channels: 128        # Base channels = 128
  unet_model_channels: 128      # U-Net channels = 128
  unet_num_res_blocks: 2        # 2 ResBlocks per level
  unet_attention_levels: [1,2]  # Attention at levels 1 and 2
  unet_channel_mult: [1,2,4]    # Channel multipliers: 128,256,512

training:
  num_epochs: 2
  batch_size: 1
  learning_rate: 1e-4
  resolution: [128, 128]
  num_frames: 8

pretrained:
  two_phase_training: true
  phase1_epochs: 1
  layer_lr_multipliers:
    vae_encoder: 0.1
    vae_decoder: 0.1
    unet: 1.0
```

---

## Performance Characteristics

### Throughput Estimates

| Hardware | Batch Size | Samples/sec | Hours/Epoch (206 patients) |
|----------|-----------|-------------|---------------------------|
| Tesla V100 32GB | 1 | ~0.5 | ~2 hours |
| Tesla V100 32GB | 2 | ~0.8 | ~1.3 hours |
| A100 40GB | 2 | ~1.2 | ~0.9 hours |
| A100 40GB | 4 | ~2.0 | ~0.5 hours |

**Optimization Tips:**
- Enable mixed precision (`fp16`): 2× speedup
- Gradient accumulation: Simulate larger batch sizes
- Reduce resolution for testing: 64×64 → 4× faster

---

## Architecture Design Choices

### Why 3D Convolutions?

✅ **Temporal coherence**: Maintains consistency across CT slices
✅ **Efficiency**: Single model for all slices
✅ **Context**: Each slice sees neighboring slices

vs 2D: Would process each slice independently

### Why Latent Diffusion?

✅ **Efficiency**: 8× compression → 64× fewer pixels to denoise
✅ **Speed**: Faster training and inference
✅ **Quality**: Focus computation on semantic features

vs Pixel Diffusion: Would denoise in full resolution (very slow)

### Why Two-Phase Training?

✅ **Convergence**: U-Net learns quickly with fixed VAE
✅ **Stability**: Prevents VAE collapse early in training
✅ **Quality**: Fine-tuning adapts VAE to task

vs Joint Training: Can be unstable, slower convergence

---

## Summary

**Model**: Video-to-Video Latent Diffusion
**Parameters**: 335M total (172M VAE + 163M U-Net)
**Compression**: 8× spatial (128→16)
**Strategy**: Two-phase training with layer-wise LR
**Memory**: ~7.5GB training, ~2.5GB inference
**Speed**: ~2 hours/epoch on V100

**Input**: Source CT video (8 slices, 128×128, RGB)
**Output**: Enhanced CT video (8 slices, 128×128, RGB)
**Latent**: Compressed representation (8 frames, 16×16, 4 channels)

This architecture balances efficiency and quality for medical video-to-video translation tasks.
