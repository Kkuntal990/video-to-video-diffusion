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
1. Data pre-processing pipeline
2. Custom trained VAE. Froze while training the diffusion model. 
3. UNet diffusion model

Look for ARCHITECTURE.md file if need more info for each componenet. 


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

## Instructions

- Do not create any documentation except when explicitly asked. When asked to create documentation, always create in ./docs folder
- Do not commit and push unless explicitly stated.
- Never deploy on kubernetes, always give commands to deploy.
- Do not run large model tests on local, it crashes the system

## End of Context Document
