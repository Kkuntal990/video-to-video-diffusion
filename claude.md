# Claude Context: CT Slice Interpolation Project

**Last Updated**: 2025-01-13
**Model**: Latent Diffusion + MAISI VAE for Medical Imaging
**Task**: Anisotropic super-resolution (50 thick slices @ 5.0mm → 300 thin slices @ 1.0mm)

---

## 📋 Project Overview

### Task Description
- **Domain**: Medical imaging (CT scans for APE detection)
- **Input**: Thick CT slices (50 @ 5.0mm spacing)
- **Output**: Thin CT slices (300 @ 1.0mm spacing)
- **Goal**: 6× depth interpolation for improved diagnostic quality

### Dataset
- **Name**: APE (Acute Pulmonary Embolism) Dataset
- **Total patients**: 356 (after preprocessing)
- **Split**: 75% train (267), 15% val (53), 10% test (36)
- **Categories**: APE, non-APE
- **Format**: DICOM ZIP files → Preprocessed .pt cache

---

## 🏗️ Architecture

### Model Components

#### 1. VAE (Frozen, Pretrained)
- **Type**: Custom MAISI VAE (NVIDIA Medical Imaging)
- **Parameters**: 130M (100% pretrained weights loaded)
- **Compression**:
  - Spatial: 512×512 → 128×128 (4× downsampling)
  - Depth: D → D/4 (4× downsampling)
  - Channels: 1 (grayscale) → 4 (latent)
- **Scaling factor**: 0.18215
- **Status**: Frozen from epoch 0

#### 2. U-Net (Trainable)
- **Type**: 3D U-Net for noise prediction
- **Parameters**: 599M (Medium model, 128 channels)
- **Architecture**:
  - Model channels: 128
  - Res blocks: 2 per level
  - Attention levels: [1, 2]
  - Channel mult: [1, 2, 4, 4]
  - Num heads: 8
  - Time embed dim: 1024

#### 3. Diffusion Process
- **Schedule**: Cosine noise schedule
- **Timesteps**: 1000
- **Training sampler**: DDPM
- **Inference sampler**: DDIM (20-50 steps)
- **Loss**: Masked MSE with SNR weighting

### Total Model
- **Total parameters**: 729M
- **Trainable**: 599M (U-Net only)
- **Frozen**: 130M (MAISI VAE)

---

## 🔄 Data Pipeline

### Step 1: DICOM Loading (Preprocessing Phase)
```python
# Raw DICOM files
thick_volume: (50, 512, 512) HU values [-1024, +3071]
thin_volume: (300, 512, 512) HU values [-1024, +3071]
```

### Step 2: CT Windowing
```python
# Window settings (soft tissue + vessels)
window_center = 40 HU
window_width = 400 HU
# Clip to [-160, +240] HU, normalize to [0, 1]
thick_windowed: (50, 512, 512) range [0.0, 1.0]
thin_windowed: (300, 512, 512) range [0.0, 1.0]
```

**Why windowing?**
- Selects relevant tissues (soft tissue + blood vessels)
- Enhances contrast where it matters clinically
- Full HU range would waste 95% of dynamic range on irrelevant tissues

### Step 3: Final Normalization
```python
# Normalize [0, 1] → [-1, 1] (standard for diffusion models)
thick_tensor = thick_tensor * 2.0 - 1.0
thin_tensor = thin_tensor * 2.0 - 1.0

# Cached to .pt files
torch.save({
    'input': (1, 50, 512, 512),   # range [-1, 1]
    'target': (1, 300, 512, 512),  # range [-1, 1]
    'category': 'APE',
    'patient_id': 'case_XXX',
    'num_thick_slices': 50,
    'num_thin_slices': 300
}, 'case_XXX.pt')
```

### Step 4: Batch Collation with Padding
```python
# Variable depths padded to max in batch
batch = {
    'input': (B, 1, max_thick, 512, 512),   # Padded thick slices
    'target': (B, 1, max_thin, 512, 512),   # Padded thin slices
    'thick_mask': (B, 1, max_thick),        # 1=real, 0=padding
    'thin_mask': (B, 1, max_thin),          # 1=real, 0=padding
}
# Padding value: -1.0 (air/background in CT)
```

---

## 🚂 Training Pipeline

### Forward Pass (Training)

#### 1. VAE Encoding
```python
# Encode input (thick slices)
v_in: (B, 1, 50, 512, 512) range [-1, 1]
z_in: (B, 4, 12, 128, 128) range ≈[-3, +3]

# Encode target (thin slices)
v_gt: (B, 1, 300, 512, 512) range [-1, 1]
z_gt: (B, 4, 75, 128, 128) range ≈[-3, +3]
```

#### 2. Conditioning Upsampling
```python
# Upsample z_in to match z_gt depth
z_in_upsampled = F.interpolate(z_in, size=(75, 128, 128), mode='trilinear')
# z_in_upsampled: (B, 4, 75, 128, 128)
```

#### 3. Mask Downsampling
```python
# Downsample padding mask to latent space
mask: (B, 1, 300) → z_mask: (B, 1, 75)
# Use nearest-neighbor to preserve binary values
```

#### 4. Forward Diffusion
```python
# Sample random timesteps
t = torch.randint(0, 1000, (B,))

# Add noise: z_t = √α_t · z_0 + √(1-α_t) · ε
noise = torch.randn_like(z_gt)
z_t = sqrt_alpha_cumprod * z_gt + sqrt_one_minus_alpha_cumprod * noise
```

#### 5. U-Net Noise Prediction
```python
# Predict noise
noise_pred = unet(z_t, t, z_in_upsampled)
# noise_pred: (B, 4, 75, 128, 128)
```

#### 6. Loss Computation
```python
# Masked MSE loss with SNR weighting
mse_per_element = (noise_pred - noise) ** 2
masked_mse = mse_per_element * mask_expanded

# Per-sample normalization
for i in range(B):
    num_valid = mask[i].sum()
    snr_weight = clamp(snr, max=5.0) / (snr + eps)  # Min-SNR-5
    sample_loss = (masked_mse[i].sum() / num_valid) * snr_weight

loss = mean(sample_losses)
```

**Why SNR weighting?**
- Diffusion loss naturally varies 100× across timesteps
- High timesteps (high noise) are easier, low timesteps are harder
- Min-SNR-5 weighting balances this variance

---

## 🎨 Inference Pipeline

### Generation Process

#### 1. VAE Encode Input
```python
v_in: (1, 1, 50, 512, 512) range [-1, 1]
z_in: (1, 4, 12, 128, 128) range ≈[-3, +3]
```

#### 2. Determine Target Latent Shape
```python
target_depth = 300  # Desired output slices
latent_depth_target = 300 // 4 = 75

# Upsample conditioning
z_in_upsampled: (1, 4, 75, 128, 128)
```

#### 3. Initialize with Noise
```python
z_t = torch.randn((1, 4, 75, 128, 128))
```

#### 4. DDIM Denoising (20 steps)
```python
timesteps = [999, 950, 900, ..., 50, 0]  # 20 uniformly spaced

for t_idx in timesteps:
    # Predict noise
    noise_pred = unet(z_t, t, z_in_upsampled)

    # Get alpha values
    alpha_t = alphas_cumprod[t_idx]
    alpha_t_prev = alphas_cumprod[t_idx - step]

    # Predict clean latent
    sqrt_alpha_t = sqrt(alpha_t)
    z_0_pred = (z_t - sqrt(1-alpha_t) * noise_pred) / sqrt_alpha_t

    # CRITICAL: Do NOT clamp latents!
    # VAE latent space is NOT bounded to [-1, 1]
    # Clamping destroys latent structure

    # Update for next step
    z_t = sqrt(alpha_t_prev) * z_0_pred + sqrt(1-alpha_t_prev) * noise_pred
```

#### 5. VAE Decode
```python
z_0: (1, 4, 75, 128, 128)
v_out: (1, 1, 300, 512, 512) range [-1, 1]
```

---

## ⚠️ Current Issues

### Issue 1: NaN Values in Validation Predictions

**Status**: CRITICAL - Under Investigation
**Symptoms**:
- All validation samples contain NaN values (60-80M pixels)
- PSNR = 0.00 dB, SSIM = 0.0000 (due to NaN detection)
- Training loss is stable (~0.02-0.07)

**Root Cause Identified**:
```python
# In DDIM sampler at high timesteps (t≈999):
sqrt_alpha_t = 0.070  # Very small value!
z_0_pred = (z_t - sqrt_one_minus_alpha_t * noise_pred) / 0.070
# Division by near-zero with FP16 → overflow → NaN
```

**Contributing Factors**:
1. **Numerical instability**: Division by small alpha values (~0.01-0.07)
2. **Mixed precision (BF16)**: Limited dynamic range near zero
3. **No NaN detection**: NaN propagates through all 20 sampling steps

**Planned Fixes** (Not Yet Implemented):
1. Add epsilon to denominator: `z_0_pred = ... / (sqrt_alpha_t + 1e-8)`
2. Add NaN detection after each sampling step
3. Clamp intermediate values to prevent overflow
4. Consider wrapping sampling in `autocast(enabled=False)` for FP32

### Issue 2: Only 24 Slices in Visualization

**Status**: UNDER INVESTIGATION
**Expected**: 300 thin slices in output
**Actual**: User reports seeing only 24 slices

**Preprocessed Cache Verified**:
- Input (thick): (1, 50, 512, 512) ✓ Correct
- Target (thin): (1, 300, 512, 512) ✓ Correct
- Cache files are correct!

**Possible Causes**:
1. VAE decode chunk_size limiting output
2. Latent shape calculation error (300/12.5 = 24)
3. Visualization script issue
4. Different downsampling factor than expected

**Need from User**: Clarification on where exactly "24 slices" is seen

---

## ✅ Recent Fixes

### Fix 1: Loss Normalization (Epoch 34+)
**Problem**: Training loss varying wildly (988× variance)
**Solution**: Per-sample normalization + SNR weighting
**Impact**: Loss variance reduced to 15×, stable training

### Fix 2: Padding Value
**Problem**: Padding with 0.0 (mid-range intensity in [-1, 1])
**Solution**: Pad with -1.0 (air/background in CT)
**Impact**: Model no longer learns on artificial padding intensities

### Fix 3: PSNR Calculation Stability
**Problem**: PSNR = inf when MSE ≈ 0
**Solution**: Clamp MSE to [1e-8, ∞], clamp PSNR to [0, 100]
**Impact**: Robust metrics, no inf/NaN propagation

### Fix 4: Mask Downsampling
**Problem**: Trilinear interpolation on binary masks → fractional values
**Solution**: Use nearest-neighbor interpolation
**Impact**: Clean binary masks at all scales

### Fix 5: Latent Clamping Removed
**Problem**: Clamping latents to [-1, 1] in DDIM sampler
**Solution**: Removed clamping (VAE latents can be [-5, +5])
**Impact**: Better latent structure preservation

---

## 🎯 Training Status

### Current State (Epoch 48)
- **Training loss**: 0.0234-0.0728 (stable)
- **Validation PSNR**: 0.00 dB (due to NaN)
- **Validation SSIM**: 0.0000 (due to NaN)
- **Checkpoint**: `/workspace/storage_a100/checkpoints/slice_interp_full_medium/`

### Training Configuration
```yaml
# Optimization
batch_size: 1 (effective: 2×8 = 16 with gradient accumulation)
learning_rate: 0.0001
optimizer: AdamW
weight_decay: 0.01
max_grad_norm: 1.0

# Mixed precision
precision: bf16  # BF16 on A100
gradient_checkpointing: true

# Scheduling
scheduler: cosine
warmup_steps: 1000
min_lr: 0.000001

# Validation
val_interval: 1000 steps
num_validation_samples: 20
```

### Expected Final Performance
- **PSNR**: 42-46 dB (after NaN fix)
- **SSIM**: 0.92-0.97
- **Training time**: 8-10 hours (100 epochs)

---

## 📁 Key Files

### Core Model Files
- `models/model.py` - Main VideoToVideoDiffusion class
- `models/maisi_vae.py` - Custom MAISI VAE (100% pretrained)
- `models/vae.py` - VAE wrapper with chunking
- `models/unet3d.py` - 3D U-Net for denoising
- `models/diffusion.py` - Gaussian diffusion process

### Data Pipeline
- `data/slice_interpolation_dataset.py` - CT data loader with preprocessing
- `data/transforms.py` - Video/CT transforms
- `data/__init__.py` - Unified dataloader interface

### Training & Inference
- `training/trainer.py` - Training loop with validation
- `inference/sampler.py` - DDPM/DDIM samplers
- `scripts/visualize_samples.py` - Generate visualization grids

### Utilities
- `utils/metrics.py` - PSNR/SSIM with NaN handling
- `utils/checkpoint.py` - Checkpoint saving/loading

### Configuration
- `config/slice_interpolation_full_medium.yaml` - Main training config

### Kubernetes
- `kub_files/train-job-a100.yaml` - Training job
- `kub_files/visualization-job-a100.yaml` - Visualization job
- `kub_files/preprocessing-job-256.yaml` - Data preprocessing job

---

## 🔍 Value Ranges Reference

| Stage | Tensor | Shape | Range | Notes |
|-------|--------|-------|-------|-------|
| **Raw DICOM** | HU values | (D, 512, 512) | [-1024, +3071] | Hounsfield Units |
| **Windowed** | CT display | (D, 512, 512) | [0.0, 1.0] | After soft tissue windowing |
| **Normalized** | Model input | (1, D, 512, 512) | [-1.0, +1.0] | Standard for diffusion |
| **VAE Latent** | Compressed | (4, D/4, 128, 128) | ≈[-3, +3] | VAE latent space |
| **Noisy Latent** | z_t | (4, D/4, 128, 128) | ≈[-5, +5] | With added noise |
| **Decoded** | Output | (1, D, 512, 512) | [-1.0, +1.0] | After VAE decode |

---

## 🚀 Next Steps

### Immediate (Fix NaN Issue)
1. **Add numerical stability to DDIM sampler**:
   - Add epsilon to denominator
   - Clamp intermediate values
   - Add NaN detection per step

2. **Test fixes**:
   - Run visualization with fixed sampler
   - Verify no NaN in outputs
   - Measure actual PSNR/SSIM

3. **Investigate "24 slices" issue**:
   - Get exact location where 24 is seen
   - Check VAE decode output shape
   - Verify target_depth propagation

### Short-term (Improve Quality)
1. Resume training with fixed sampler
2. Monitor validation metrics (should be 30-40 dB)
3. Add perceptual loss if needed
4. Tune inference steps (20 vs 50)

### Long-term (Production)
1. Final model selection (best checkpoint)
2. Test set evaluation
3. Clinical validation with radiologists
4. Deployment pipeline

---

## 💡 Key Insights

### Why CT Windowing?
- **Problem**: Raw CT HU range [-1024, +3071] is 95% irrelevant tissues
- **Solution**: Window to [-160, +240] HU for soft tissue + vessels
- **Impact**: Soft tissue uses 50% of dynamic range vs 2% without windowing

### Why SNR Weighting?
- **Problem**: Diffusion loss varies 100× across timesteps
- **Solution**: Min-SNR-5 weighting down-weights easy timesteps
- **Impact**: Training loss variance reduced from 988× to 15×

### Why Per-Sample Normalization?
- **Problem**: Variable depths create unequal loss magnitudes
- **Solution**: Normalize each sample by its valid elements
- **Impact**: Fair loss across all batch samples

### Why Padding with -1.0?
- **Problem**: Padding with 0.0 is mid-range intensity in [-1, 1]
- **Solution**: Pad with -1.0 (air/background in CT)
- **Impact**: Model doesn't learn artificial padding patterns

### Why No Latent Clamping?
- **Problem**: VAE latent space uses scaling_factor=0.18215
- **Fact**: Latents naturally range [-5, +5], not [-1, +1]
- **Solution**: Never clamp latents in sampler
- **Impact**: Preserves latent structure, better quality

---

## 📊 Storage & Compute

### Storage Usage
```
/workspace/storage_a100/
├── dataset/                    # DICOM ZIPs: ~50 GB
├── .cache/
│   ├── processed/              # Preprocessed .pt: ~15-20 GB
│   └── slice_interpolation_cache/  # Temp extraction: deleted after preprocessing
├── checkpoints/                # Model checkpoints: ~5-10 GB
├── visualizations/             # Visualization outputs: ~1-2 GB
├── logs/                       # Training logs: ~100 MB
└── pretrained/
    └── maisi_vae/              # MAISI checkpoint: ~500 MB
```

### Compute Requirements
- **GPU**: NVIDIA A100 80GB
- **Memory**: ~28-33 GB / 80 GB during training
- **Training time**: ~5-7 minutes/epoch
- **Inference time**: ~15 seconds/sample (DDIM 20 steps)

---

## 🐛 Debugging Commands

### Check Preprocessed Cache
```bash
# Download a sample
kubectl cp copy-pod:/workspace/storage_a100/.cache/processed/case_111.pt /tmp/case_111.pt

# Inspect with Python
~/miniconda3/envs/ct-superres-mps/bin/python3 << 'EOF'
import torch
sample = torch.load('/tmp/case_111.pt', weights_only=False)
print(f"Input: {sample['input'].shape}, range {sample['input'].min():.3f} to {sample['input'].max():.3f}")
print(f"Target: {sample['target'].shape}, range {sample['target'].min():.3f} to {sample['target'].max():.3f}")
EOF
```

### Check Training Logs
```bash
kubectl exec copy-pod -- tail -100 /workspace/storage_a100/logs/train_*.log
```

### Check Checkpoint Status
```bash
kubectl exec copy-pod -- ls -lh /workspace/storage_a100/checkpoints/slice_interp_full_medium/
```

### Run Visualization
```bash
kubectl apply -f kub_files/visualization-job-a100.yaml
kubectl logs -f v2v-diffusion-visualization-job-a100-xxxxx
```

---

## 📚 References

### Papers
- DDPM: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- DDIM: Denoising Diffusion Implicit Models (Song et al., 2020)
- Latent Diffusion: High-Resolution Image Synthesis (Rombach et al., 2022)
- MAISI: Medical AI for Synthetic Imaging (NVIDIA, 2024)
- Min-SNR Weighting: Imagen (Saharia et al., 2022)

### Repositories
- MONAI: https://github.com/Project-MONAI/MONAI
- Diffusers: https://github.com/huggingface/diffusers
- MAISI: https://github.com/Project-MONAI/GenerativeModels

---

## 📝 Notes

- **Critical files modified** in recent fixes:
  - `models/diffusion.py` (SNR weighting)
  - `data/slice_interpolation_dataset.py` (padding value)
  - `utils/metrics.py` (PSNR stability)
  - `models/model.py` (mask downsampling)
  - `inference/sampler.py` (latent clamping removed)

- **Visualization updated**:
  - `scripts/visualize_samples.py` (NaN detection, tensor range fix)
  - `kub_files/visualization-job-a100.yaml` (correct GPU, config, paths)

- **Git branch**: `maisi-pretrained-vae`

- **Last checkpoint**: `checkpoint_epoch_48.pt` (2025-01-13)

---

**End of Context Document**
