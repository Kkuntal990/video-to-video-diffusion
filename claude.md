# Claude Context: CT Slice Interpolation Project

**Last Updated**: 2025-01-16
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
- **Docker**: `ghcr.io/kkuntal990/v2v-diffusion:latest`
- **Storage**: `/workspace/storage_a100/`

## 🏗️ Architecture

### Model Components

#### 1. VAE (Training from Scratch - NEW)
- **Status**: 🔄 **TRAINING IN PROGRESS** (abandoned pretrained MAISI)
- **Type**: Custom VideoVAE (deterministic autoencoder)
- **Parameters**: ~25M (training from scratch)
- **Architecture**:
  - Encoder: 3-level downsampling (64→128→256 channels)
  - Decoder: 3-level upsampling (256→128→64 channels)
  - Latent channels: 4
  - Base channels: 64
- **Compression**:
  - Spatial: 512×512 → 64×64 (8× downsampling)
  - Depth: D → D (NO temporal/depth compression)
  - Channels: 1 (grayscale) → 4 (latent)
- **Scaling factor**: 0.18215
- **Training**: 50 epochs, target PSNR >35 dB
- **Loss**: MSE + MS-SSIM (Perceptual disabled for memory)
- **Dimension Flow** (thick slices example):
  - Input: (B, 1, 50, 512, 512) range [-1, 1]
  - After conv_in: (B, 64, 50, 512, 512)
  - After down1: (B, 128, 50, 256, 256)
  - After down2: (B, 256, 50, 128, 128)
  - After down3: (B, 256, 50, 64, 64)
  - Latent: (B, 4, 50, 64, 64) range ≈[-3, +3]
  - Decoder reverses this process
  - Output: (B, 1, 50, 512, 512) range [-1, 1]

#### 2. U-Net (Trainable)
- **Type**: 3D U-Net for noise prediction
- **Parameters**: 270M (Medium model, 128 channels)
- **Architecture**:
  - Model channels: 128
  - Res blocks: 2 per level
  - Attention levels: [1, 2] (temporal attention)
  - Channel mult: [1, 2, 4, 4] → [128, 256, 512, 512]
  - Num heads: 8
  - Time embed dim: 1024
  - Gradient checkpointing: Enabled
- **Dimension Flow** (50→300 slices):
  - Input: Noisy latent (B, 4, 75, 64, 64) + Conditioning (B, 4, 75, 64, 64)
  - Concatenated: (B, 8, 75, 64, 64)
  - Encoder levels:
    - Level 0: (B, 128, 75, 64, 64)
    - Level 1: (B, 256, 75, 32, 32) + temporal attention
    - Level 2: (B, 512, 75, 16, 16) + temporal attention
    - Level 3: (B, 512, 75, 8, 8)
  - Middle: (B, 512, 75, 8, 8) + attention
  - Decoder: Mirrors encoder with skip connections
  - Output: Predicted noise (B, 4, 75, 64, 64)
- **Note**: Spatial downsampling only (depth preserved for temporal consistency)

#### 3. Diffusion Process
- **Schedule**: Cosine noise schedule
- **Timesteps**: 1000
- **Training sampler**: DDPM
- **Inference sampler**: DDIM (20-50 steps)
- **Loss**: Masked MSE with SNR weighting

### Total Model (After VAE Training)
- **Total parameters**: ~295M
- **Trainable**: 270M (U-Net only, during diffusion training)
- **Frozen**: ~25M (Custom VAE, after pretraining)

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


## Instructions
- Do not create any documents except when explicitly asked. 
- Do not commit and push unless explicitly stated.

**End of Context Document**
