# Claude Context: CT Slice Interpolation Project

**Last Updated**: 2025-01-15
**Model**: Latent Diffusion for Medical CT Slice Interpolation
**Task**: Anisotropic super-resolution (50 thick slices @ 5.0mm → 300 thin slices @ 1.0mm)

---

## 📋 Project Overview

### Task Description
- **Domain**: Medical imaging (CT scans for APE detection)
- **Task**: **CT Slice Interpolation** (NOT video-to-video diffusion)
- **Input**: Thick CT slices (50 @ 5.0mm spacing)
- **Output**: Thin CT slices (300 @ 1.0mm spacing)
- **Goal**: 6× depth interpolation for improved diagnostic quality
- **Method**: Latent diffusion in compressed latent space

### Dataset
- **Name**: APE (Acute Pulmonary Embolism) Dataset
- **Total patients**: 323 (successfully preprocessed)
- **Split**: Train=243, Val=48, Test=32
- **Categories**: APE, non-APE
- **Format**: DICOM ZIP files → Preprocessed .pt cache
- **Cache location**: `/workspace/storage_a100/.cache/processed/`

---

## Environment
- **Python**: python3 using conda environment `ct-superres-mps`
- **GPU**: V100 16GB (training), A100 80GB (inference)
- **Docker**: `ghcr.io/kkuntal990/v2v-diffusion:latest`
- **Storage**: `/workspace/storage_a100/`

## 🏗️ Architecture

### Model Components

#### 1. VAE (Training from Scratch - NEW)
- **Status**: 🔄 **TRAINING IN PROGRESS** (abandoned pretrained MAISI)
- **Type**: Custom VideoVAE (deterministic autoencoder)
- **Parameters**: 43M (training from scratch)
- **Architecture**:
  - Encoder: 3-level downsampling (64→128→256 channels)
  - Decoder: 3-level upsampling (256→128→64 channels)
  - Latent channels: 4
  - Base channels: 64
- **Compression**:
  - Spatial: 512×512 → 64×64 (8× downsampling)
  - Depth: D → D (NO temporal compression)
  - Channels: 1 (grayscale) → 4 (latent)
- **Scaling factor**: 0.18215
- **Training**: 20 epochs, target PSNR >35 dB
- **Loss**: MSE + Perceptual (LPIPS) + MS-SSIM

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

### Total Model (After VAE Training)
- **Total parameters**: 642M
- **Trainable**: 599M (U-Net only, during diffusion training)
- **Frozen**: 43M (Custom VAE, after pretraining)

---

## 🔄 Data Pipeline (CT Slice Interpolation)

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

## 🚂 Training Pipeline (Diffusion Model - After VAE Training)

### Forward Pass (Training)

#### 1. VAE Encoding (NEW: Custom VAE Architecture)
```python
# Encode input (thick slices)
v_in: (B, 1, 50, 512, 512) range [-1, 1]
z_in: (B, 4, 50, 64, 64) range ≈[-3, +3]  # 8× spatial, NO depth compression

# Encode target (thin slices)
v_gt: (B, 1, 300, 512, 512) range [-1, 1]
z_gt: (B, 4, 300, 64, 64) range ≈[-3, +3]  # 8× spatial, NO depth compression
```

**Key Change**: Custom VAE has NO depth compression (D → D), only 8× spatial compression

#### 2. Conditioning Upsampling
```python
# Upsample z_in to match z_gt depth
z_in_upsampled = F.interpolate(z_in, size=(300, 64, 64), mode='trilinear')
# z_in_upsampled: (B, 4, 300, 64, 64)
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

#### 5. VAE Decode (NEW: Custom VAE)
```python
z_0: (1, 4, 300, 64, 64)
v_out: (1, 1, 300, 512, 512) range [-1, 1]
```

---

## 📊 Current Status (2025-01-15)

### ✅ Completed
1. ✓ **MAISI VAE Investigation** - Abandoned pretrained MAISI (incompatible checkpoint)
2. ✓ **Custom VAE Architecture** - Designed 43M parameter autoencoder
3. ✓ **VAE Training Setup** - Created `train_vae.py` with MSE+Perceptual+SSIM losses
4. ✓ **Model Integrity Tests** - Comprehensive test suite (3 test files, 50+ tests)
5. ✓ **Code Structure Validation** - All tests passed locally
6. ✓ **Input Shape Bug Fixed** - Dataset returns `(B, C, D, H, W)` correctly

### 🔄 In Progress
1. **Custom VAE Training** - Training from scratch on V100 GPU
   - Target: PSNR >35 dB reconstruction quality
   - Duration: 1-2 hours (20 epochs)
   - Status: Job deployed, waiting for completion

### 📝 Pending
1. **VAE Validation** - Test trained VAE reconstruction quality
2. **Diffusion Training** - Train U-Net with frozen custom VAE
3. **End-to-End Evaluation** - Test full slice interpolation pipeline

---

## ⚠️ Known Issues (Historical - RESOLVED by Custom VAE)

### Issue 1: MAISI VAE Incompatibility (RESOLVED)

**Status**: ✅ **RESOLVED** - Switched to custom VAE training
**Root Cause**: Pretrained MAISI checkpoint produced wrong output range
- Expected: [0, 1] normalized output
- Actual: [-0.64, 1.84] (corrupted/incompatible weights)
- PSNR: 7-13 dB (far below >35 dB target)

**Resolution**: Train custom VideoVAE from scratch (43M params)

### Issue 2: Input Shape Mismatch (RESOLVED)

**Status**: ✅ **RESOLVED** - Fixed in commit `58d2266`
**Root Cause**: Extra `unsqueeze(1)` added channel dimension twice
- Dataset output: `(B, C, D, H, W)` - already has channel dim
- Bug: Added extra unsqueeze → `(B, C, C, D, H, W)` (6D tensor)
- Error: VAE expected 5D input

**Resolution**: Removed extra `unsqueeze(1)` in `train_vae.py`

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

### Phase 1: VAE Training (Current)

**Status**: 🔄 In Progress
**Location**: `/workspace/storage_a100/checkpoints/vae_training/`

```yaml
# VAE Training Configuration
model:
  architecture: VideoVAE (3-level encoder/decoder)
  parameters: 43M
  in_channels: 1 (grayscale CT)
  latent_dim: 4
  base_channels: 64
  spatial_compression: 8× (512→64)
  depth_compression: 1× (D→D, no temporal compression)

training:
  num_epochs: 20
  batch_size: 2
  gradient_accumulation: 4 (effective batch = 8)
  learning_rate: 0.0001
  optimizer: AdamW
  weight_decay: 0.01
  scheduler: cosine
  mixed_precision: bf16

losses:
  reconstruction: MSE (λ=1.0)
  perceptual: LPIPS VGG (λ=0.1)
  ssim: MS-SSIM (λ=0.1)

gpu: V100 16GB
training_time: ~1-2 hours
```

**Expected Performance**:
- Epoch 1: PSNR ~25-28 dB
- Epoch 10: PSNR ~35-38 dB (target reached)
- Epoch 20: PSNR ~38-42 dB (final)

### Phase 2: Diffusion Training (Pending)

**Status**: ⏳ Waiting for VAE training completion

```yaml
# Diffusion Training Configuration (After VAE)
model:
  vae: Custom VideoVAE (frozen, 43M)
  unet: 3D U-Net (trainable, 599M)
  diffusion: Cosine schedule, 1000 timesteps

training:
  num_epochs: 100
  batch_size: 1
  gradient_accumulation: 8 (effective batch = 8)
  learning_rate: 0.0001
  optimizer: AdamW
  mixed_precision: bf16

gpu: A100 80GB
checkpoint: /workspace/storage_a100/checkpoints/slice_interp_full_medium/
training_time: ~8-10 hours
```

**Expected Final Performance**:
- PSNR: 42-46 dB
- SSIM: 0.92-0.97

---

## 📁 Key Files

### Core Model Files
- `models/model.py` - Main slice interpolation diffusion model
- `models/vae.py` - Custom VideoVAE (8× spatial compression, no temporal)
- `models/unet3d.py` - 3D U-Net for noise prediction
- `models/diffusion.py` - Gaussian diffusion process with SNR weighting
- `models/maisi_vae.py` - Legacy MAISI VAE (deprecated)

### Data Pipeline
- `data/slice_interpolation_dataset.py` - CT slice dataset with preprocessing
- `data/transforms.py` - CT-specific transforms
- `data/__init__.py` - Unified dataloader interface

### Training Scripts
- `train_vae.py` - **NEW** Custom VAE training from scratch
- `training/trainer.py` - Diffusion model training loop
- `train.py` - Main training entry point

### Inference & Validation
- `inference/sampler.py` - DDPM/DDIM samplers
- `scripts/visualize_samples.py` - Generate visualization grids
- `tests/test_vae_reconstruction.py` - VAE quality validation

### Testing
- `tests/test_model_integrity.py` - Comprehensive pytest suite (45+ tests)
- `tests/test_vae_compatibility.py` - VAE integration tests
- `tests/test_code_structure.py` - Code structure validation
- `tests/test_vae_shapes.py` - Simple shape validation

### Utilities
- `utils/metrics.py` - PSNR/SSIM with NaN handling
- `utils/checkpoint.py` - Checkpoint saving/loading

### Configuration
- `config/vae_training.yaml` - **NEW** VAE training config
- `config/slice_interpolation_full_medium.yaml` - Diffusion training config

### Kubernetes
- `kub_files/vae-training-job-a100.yaml` - **NEW** VAE training job (V100)
- `kub_files/train-job-a100.yaml` - Diffusion training job (A100)
- `kub_files/vae-test-job.yaml` - VAE validation job
- `kub_files/visualization-job-a100.yaml` - Visualization job
- `kub_files/preprocessing-job-256.yaml` - Data preprocessing job

---

## 🔍 Value Ranges Reference (Custom VAE Architecture)

| Stage | Tensor | Shape | Range | Notes |
|-------|--------|-------|-------|-------|
| **Raw DICOM** | HU values | (D, 512, 512) | [-1024, +3071] | Hounsfield Units |
| **Windowed** | CT display | (D, 512, 512) | [0.0, 1.0] | After soft tissue windowing |
| **Normalized** | Model input | (1, D, 512, 512) | [-1.0, +1.0] | Standard for diffusion |
| **VAE Latent** | Compressed | (4, D, 64, 64) | ≈[-3, +3] | **NEW**: 8× spatial, NO depth compression |
| **Noisy Latent** | z_t | (4, D, 64, 64) | ≈[-5, +5] | With added noise |
| **Decoded** | Output | (1, D, 512, 512) | [-1.0, +1.0] | After VAE decode |

**Key Change**: Custom VAE maintains depth dimension (D → D), only compresses spatially (512 → 64)

---

## 🚀 Next Steps

### Immediate (VAE Training)
1. **Monitor VAE training progress** 🔄
   - Watch for PSNR >35 dB milestone (epoch ~10)
   - Check for NaN/Inf in outputs
   - Validate reconstruction quality

2. **VAE validation**:
   - Run `test_vae_reconstruction.py` on trained VAE
   - Verify PSNR >35 dB, SSIM >0.95
   - Test on multiple patients

3. **Integrate trained VAE**:
   - Update `slice_interpolation_full_medium.yaml`
   - Point to best VAE checkpoint
   - Freeze VAE weights for diffusion training

### Short-term (Diffusion Training)
1. Train diffusion model with frozen custom VAE
2. Monitor validation metrics (target: 42-46 dB)
3. Test slice interpolation quality (50 → 300 slices)
4. Tune DDIM inference steps (20 vs 50)

### Long-term (Production)
1. Final model selection (best checkpoint)
2. Test set evaluation on 32 held-out patients
3. Clinical validation with radiologists
4. Deployment pipeline for inference

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
