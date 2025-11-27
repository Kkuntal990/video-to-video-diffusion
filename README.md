# CT Slice Interpolation with Latent Diffusion

A PyTorch implementation of 3D latent diffusion for CT slice interpolation (anisotropic super-resolution), featuring custom VAE training, patch-based training, and Kubernetes deployment.

## 🎯 Task

**Anisotropic Super-Resolution for Medical CT Scans**

- **Input**: 8 thick CT slices @ 5.0mm spacing (low resolution in depth)
- **Output**: 48 thin CT slices @ 1.0mm spacing (6× interpolation)
- **Goal**: Improve diagnostic quality by generating missing intermediate slices

## 🚀 Quick Start

### Local Testing
```bash
# Test VAE reconstruction
python tests/test_vae_reconstruction.py

# Test model integrity
pytest tests/test_model_integrity.py
```

### Kubernetes Training (A100 GPU)
```bash
# 1. Deploy VAE training
kubectl apply -f kub_files/vae-training-job-a100.yaml

# 2. Monitor training
kubectl logs -f job/vae-training-job-a100

# 3. After VAE completes, train diffusion model
kubectl apply -f kub_files/train-job-a100.yaml
```

📖 **See [CLAUDE.md](CLAUDE.md) for complete project context and architecture details**

---

## 📋 What's New

### Recent Updates (2025-01)

✅ **VAE Architecture Refactored for Latent Diffusion**
- Removed encoder→decoder skip connections (incompatible with diffusion)
- Encoder and decoder now work independently
- Trained from scratch for CT slice interpolation task
- Target: PSNR ≥35 dB on encode→decode reconstruction

✅ **VAE-UNet Integration Fixed**
- Added `torch.no_grad()` around VAE encoding (saves 2-3 GB GPU memory)
- Set VAE to `.eval()` mode during diffusion training
- Fixed: VAE properly frozen, no gradient leakage

✅ **Metric Calculation Standardized**
- All PSNR/SSIM now use [0,1] normalization with `max_val=1.0`
- Metrics directly comparable between VAE and diffusion training
- Updated in training, evaluation, and test scripts

✅ **Data Pipeline Cleanup**
- Removed 5 legacy dataset files (2,833 lines, 60.5% reduction)
- Deleted wrong task implementations (temporal video pairs)
- Simplified to CT slice interpolation only
- Cleaner, focused codebase

✅ **Batch Size Increased**
- VAE training: batch_size=4 (up from 1)
- Skip connections removed → less memory usage
- Faster training convergence

---

## 📊 Architecture

### Model Pipeline

```
Thick Slices (8 @ 5.0mm, 512×512)
          ↓
    [ VAE Encoder ]  →  Latent (8 @ 64×64)  [8× spatial compression]
          ↓
    [ Add Noise ]  →  Noisy Latent (training)
          ↓
    [ 3D U-Net ]  →  Denoised Latent
          ↓
    [ VAE Decoder ]  →  Thin Slices (48 @ 1.0mm, 512×512)
```

### Model Components

| Component | Parameters | Compression | Training Status |
|-----------|-----------|-------------|-----------------|
| **VAE Encoder** | 86M | Spatial 8× (512→64) | ✅ Custom trained |
| **VAE Decoder** | 86M | Spatial 8× (64→512) | ✅ Custom trained |
| **3D U-Net** | 163M | None (latent→latent) | 🔄 In progress |
| **Total** | **335M** | - | - |

**Key Features:**
- No skip connections between VAE encoder/decoder (latent diffusion compatible)
- Depth preserved through entire pipeline (8 thick → 48 thin)
- BF16 mixed precision training on A100
- Patch-based training for memory efficiency

### Training Approach

**Phase 1: VAE Training (Complete)**
```yaml
Task: Learn to encode/decode CT patches
Input: 192×192 patches (8 thick OR 48 thin slices)
Objective: Reconstruction quality (PSNR ≥35 dB)
Status: ✅ Complete (best checkpoint available)
```

**Phase 2: Diffusion Training (Current)**
```yaml
Task: Learn to interpolate thick → thin in latent space
Input: 192×192 patches (8 thick + 48 thin slices)
Objective: High-quality slice interpolation
Status: 🔄 In progress (VAE frozen, U-Net training)
```

📖 **See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture diagrams** (if exists)

---

## 🏗️ Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for A100 GPU)
- 80GB GPU memory (A100) for batch_size=4 training
- 32GB GPU memory (V100) for batch_size=1 training

### Local Setup

```bash
# Clone repository
git clone <repository-url>
cd LLM_agent_v2v

# Create conda environment
conda create -n ct-superres-mps python=3.10
conda activate ct-superres-mps

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Docker (Kubernetes)

Pre-built image available:
```bash
docker pull ghcr.io/kkuntal990/v2v-diffusion:latest
```

---

## 📚 Dataset

### APE Dataset (Acute Pulmonary Embolism)

- **Source**: Medical CT scans for pulmonary embolism detection
- **Total Cases**: 323 successfully preprocessed
- **Split**: Train=243, Val=48, Test=32
- **Categories**: APE (positive), non-APE (negative)
- **Format**: DICOM ZIP files → Preprocessed .pt cache
- **Storage**: Raw ~50GB → Cached ~15-20GB

### Data Pipeline

#### Current Capabilities ✅

**1. Full Preprocessing Pipeline (Local ZIPs)**
```
Raw ZIPs (/workspace/storage_a100/dataset/)
    ↓
Extract DICOMs (temp directory)
    ↓
Load & Window CT scans (HU → [-1,1])
    ↓
Resize to 512×512
    ↓
Cache as .pt tensors (/workspace/storage_a100/.cache/processed/)
    ↓
Delete DICOMs (save 30-35GB storage)
```

**Implementation**: `data/slice_interpolation_dataset.py`
- Handles ZIP extraction, DICOM loading, preprocessing, caching
- Auto-resume: skips already-processed cases
- Configurable storage paths via YAML config
- Works with different storage locations (just update paths)

**2. Patch-Based Training**
```
Preprocessed cache (.pt files)
    ↓
Extract 3D patches (8 thick → 48 thin @ 192×192)
    ↓
Random sampling for training
    ↓
Data augmentation (flips, rotations)
```

**Implementation**: `data/patch_slice_interpolation_dataset.py`
- Loads from preprocessed cache only (no raw processing)
- Fixed-size patches for efficient training
- Supports large batch sizes (batch_size=4+)

#### Current Limitations ⚠️

**What's NOT Supported:**
- ❌ HuggingFace Hub downloading (deleted with legacy files)
- ❌ Timeout handling for slow DICOM files
- ❌ Metadata JSON tracking

**Workarounds:**
- **HF downloading**: Use `huggingface-cli download t2ance/APE-data` manually
- **Timeout issues**: Monitor preprocessing logs for stuck cases
- **Metadata**: Categories derived from folder structure (APE/ and non-APE/)

**Note**: If you have local ZIP files, current pipeline is FULLY FUNCTIONAL and optimized for the slice interpolation task.

### Configuration Example

```yaml
# config/vae_training.yaml
data:
  data_source: 'slice_interpolation'
  use_patches: true

  # Configurable storage paths
  dataset_path: '/workspace/storage_a100/dataset'           # Raw ZIPs
  extract_dir: '/workspace/storage_a100/.cache/temp'        # Temp extraction
  processed_dir: '/workspace/storage_a100/.cache/processed' # .pt cache

  # Patch configuration
  patch_depth_thick: 8
  patch_depth_thin: 48
  patch_size: [192, 192]

  # Common settings
  categories: ['APE', 'non-APE']
  resolution: [512, 512]
  window_center: 40
  window_width: 400
  batch_size: 4
  num_workers: 4
```

---

## 🎓 Training

### 1. VAE Training (First Step)

Train custom VAE from scratch on CT patches:

```bash
# Kubernetes (A100)
kubectl apply -f kub_files/vae-training-job-a100.yaml

# Monitor
kubectl logs -f job/vae-training-job-a100
```

**Configuration**: `config/vae_training.yaml`
```yaml
model:
  latent_dim: 8
  base_channels: 128
  scaling_factor: 1.0
  use_skip_connections: false  # CRITICAL: Disabled for latent diffusion

training:
  num_epochs: 100
  learning_rate: 0.0001
  batch_size: 4  # Increased (skip connections removed)
  mixed_precision: true
  precision: 'bf16'

  # Training ratio
  thick_slice_ratio: 0.2  # 20% thick, 80% thin
```

**Expected Results:**
- Target: PSNR ≥35 dB on encode→decode
- Training time: ~2-4 hours (60-80 epochs on A100)
- Best checkpoint: `/workspace/storage_a100/checkpoints/vae_training_custom_vae_no_skips/vae_best.pt`

### 2. Diffusion Training (Second Step)

Train U-Net denoiser with frozen VAE:

```bash
# Kubernetes (A100)
kubectl apply -f kub_files/train-job-a100.yaml

# Monitor
kubectl logs -f job/v2v-diffusion-training-a100
```

**Configuration**: `config/slice_interpolation_full_medium.yaml`
```yaml
model:
  latent_dim: 8
  vae_base_channels: 128
  unet_model_channels: 192

  # VAE checkpoint (frozen during training)
  checkpoint_path: '/workspace/storage_a100/checkpoints/vae_training_custom_vae_no_skips/vae_best.pt'

training:
  num_epochs: 100
  learning_rate: 0.0001
  batch_size: 8
  mixed_precision: true
  precision: 'bf16'

  # VAE is FROZEN (requires_grad=False)
  freeze_vae: true
```

**Expected Results:**
- Target: PSNR 35-42 dB, SSIM 0.92-0.98 on thin slice generation
- Training time: ~5-7 minutes/epoch on A100
- Best checkpoint: `/workspace/storage_a100/checkpoints/slice_interpolation_full_medium/best.pt`

### Resume Training

```bash
# Update config
resume_from_checkpoint: '/workspace/storage_a100/checkpoints/<job_name>/checkpoint_epoch_X.pt'

# Redeploy
kubectl delete job <job-name>
kubectl apply -f kub_files/<job-file>.yaml
```

### Monitoring

```bash
# Training logs
kubectl logs -f job/<job-name>

# GPU utilization
kubectl exec <pod-name> -- nvidia-smi

# Storage usage
kubectl exec <pod-name> -- df -h /workspace/storage_a100

# Checkpoint list
kubectl exec <pod-name> -- ls -lh /workspace/storage_a100/checkpoints/
```

---

## 🔮 Inference & Evaluation

### Evaluate VAE Reconstruction

```bash
# Test VAE quality on validation patches
python scripts/evaluate_vae_reconstruction.py \
  --checkpoint /workspace/storage_a100/checkpoints/vae_training_custom_vae_no_skips/vae_best.pt \
  --config config/vae_training.yaml \
  --split val \
  --num_samples 10 \
  --save_visualizations
```

### Evaluate Diffusion Model

```bash
# Generate and evaluate thin slices from thick slices
python scripts/evaluate_and_visualize_patches.py \
  --checkpoint /workspace/storage_a100/checkpoints/slice_interpolation_full_medium/best.pt \
  --config config/slice_interpolation_full_medium.yaml \
  --split val \
  --num_samples 5 \
  --sampler ddim \
  --steps 20
```

### Visualization Output

Generated visualizations saved to:
```
/workspace/storage_a100/visualizations/<timestamp>/
├── sample_0.png   # Input | Target | Prediction comparison
├── sample_1.png
└── metrics.json   # PSNR, SSIM for each sample
```

---

## 📁 Project Structure

```
LLM_agent_v2v/
├── config/
│   ├── vae_training.yaml                      # VAE training config
│   └── slice_interpolation_full_medium.yaml   # Diffusion training config
│
├── models/
│   ├── vae.py                    # Custom VideoVAE (NO skip connections)
│   ├── unet3d.py                 # 3D U-Net denoiser
│   ├── diffusion.py              # Gaussian diffusion process
│   └── model.py                  # Complete latent diffusion model
│
├── data/
│   ├── slice_interpolation_dataset.py         # Full-volume CT dataset
│   ├── patch_slice_interpolation_dataset.py   # Patch-based dataset
│   ├── get_dataloader.py                      # Unified dataloader interface
│   └── transforms.py                          # Video transforms
│
├── training/
│   ├── trainer.py                # Training loop with validation
│   └── scheduler.py              # Learning rate schedulers
│
├── inference/
│   └── sampler.py                # DDPM/DDIM samplers
│
├── utils/
│   ├── metrics.py                # PSNR, SSIM (standardized [0,1] range)
│   ├── checkpoint.py             # Checkpoint saving/loading
│   └── logger.py                 # Logging utilities
│
├── scripts/
│   ├── evaluate_vae_reconstruction.py         # VAE quality testing
│   ├── evaluate_and_visualize_patches.py      # Diffusion evaluation
│   └── visualize_samples.py                   # Visualization utilities
│
├── tests/
│   ├── test_model_integrity.py               # Comprehensive pytest suite (45+ tests)
│   ├── test_vae_reconstruction.py            # VAE validation
│   └── test_vae_compatibility.py             # VAE integration tests
│
├── kub_files/                                  # Kubernetes deployment
│   ├── vae-training-job-a100.yaml            # VAE training (V100)
│   ├── train-job-a100.yaml                   # Diffusion training (A100)
│   ├── vae-evaluation-job.yaml               # VAE evaluation
│   └── visualization-job-a100.yaml           # Visualization generation
│
├── train_vae.py                  # VAE training script
├── train.py                      # Diffusion training script
├── CLAUDE.md                     # Complete project context
└── README.md                     # This file
```

**Note**: Legacy files removed (dataset.py, ape_dataset.py, ape_hf_dataset.py, ape_cached_dataset.py, dicom_utils.py) - 2,833 lines cleaned up for focused CT slice interpolation pipeline.

---

## ⚡ Performance

### Training Speed (A100 80GB)

| Task | Batch Size | Time/Epoch | GPU Memory | Throughput |
|------|-----------|-----------|------------|-----------|
| VAE Training | 4 | ~8-10 min | 28-33 GB | ~0.4 samples/sec |
| Diffusion Training | 8 | ~5-7 min | 40-50 GB | ~0.5 samples/sec |

**Optimizations:**
- ✅ BF16 mixed precision (better than FP16 for A100)
- ✅ Batch size increased (skip connections removed)
- ✅ Preprocessed .pt cache (100-200× faster than DICOM loading)
- ✅ Patch-based training (fixed size, no padding)

### Inference Speed

| Sampler | Steps | Time/Sample | Quality |
|---------|-------|------------|---------|
| DDIM | 20 | ~15 sec | Good |
| DDIM | 50 | ~30 sec | Better |
| DDPM | 1000 | ~10 min | Best |

---

## 🧪 Testing

### Run All Tests

```bash
# Comprehensive model integrity tests (45+ tests)
pytest tests/test_model_integrity.py -v

# VAE reconstruction quality
python tests/test_vae_reconstruction.py

# VAE-UNet compatibility
pytest tests/test_vae_compatibility.py -v

# Code structure validation
pytest tests/test_code_structure.py -v
```

### Test Coverage

✅ **Model Integrity** (test_model_integrity.py)
- Forward pass shapes
- VAE encoding/decoding
- U-Net denoising
- Diffusion process
- Gradient flow
- Memory management

✅ **VAE Reconstruction** (test_vae_reconstruction.py)
- Encode→decode quality
- Patch processing
- Full volume handling
- NaN detection

✅ **Integration** (test_vae_compatibility.py)
- VAE-UNet integration
- Checkpoint loading
- Config parsing

---

## 🔧 Troubleshooting

### Out of Memory

**Problem**: CUDA out of memory during training

**Solutions**:
```yaml
# Reduce batch size
batch_size: 2  # or 1

# Reduce patch size
patch_size: [128, 128]  # from [192, 192]

# Enable gradient accumulation
gradient_accumulation_steps: 4

# Reduce workers
num_workers: 2
```

### VAE Reconstruction Poor Quality

**Problem**: VAE PSNR < 35 dB

**Check**:
1. Skip connections disabled: `use_skip_connections: false` in config
2. Using forward() method, not encode()/decode() separately
3. Metrics use [0,1] normalization: `max_val=1.0`
4. Training long enough (60-80 epochs minimum)

### Diffusion Results Poor

**Problem**: Diffusion PSNR ~6-7 dB instead of 35-42 dB

**Check**:
1. VAE properly frozen: `freeze_vae: true` in config
2. VAE checkpoint loaded correctly
3. VAE uses NO skip connections (custom_vae_no_skips checkpoint)
4. Metrics standardized to [0,1] range across all scripts

### Kubernetes Pod Pending

**Problem**: Job stuck in pending state

**Check**:
```bash
# GPU node availability
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB

# Resource limits
kubectl describe job <job-name>

# PVC binding
kubectl get pvc
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Complete project context, architecture, and technical details |
| [README.md](README.md) | This file - project overview and quick start |

---

## 🎯 Roadmap

### Completed ✅
- [x] Custom VAE training from scratch
- [x] Removed skip connections for latent diffusion compatibility
- [x] Patch-based training pipeline
- [x] Data preprocessing with caching
- [x] BF16 mixed precision training
- [x] Metric standardization ([0,1] range)
- [x] VAE-UNet integration fixes
- [x] Data pipeline cleanup
- [x] Comprehensive test suite (45+ tests)
- [x] Kubernetes deployment (A100 GPU)

### In Progress 🔄
- [ ] Diffusion model training (VAE frozen)
- [ ] Hyperparameter tuning
- [ ] Validation metrics tracking

### Planned 📋
- [ ] Full-volume inference with stitching
- [ ] TensorBoard logging
- [ ] Multi-GPU distributed training
- [ ] Inference optimization (compile, TensorRT)
- [ ] Clinical validation

---

## 📝 Citation

This implementation is based on:

```bibtex
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={NeurIPS},
  year={2020}
}

@article{song2020denoising,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  journal={ICLR},
  year={2021}
}

@article{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  journal={CVPR},
  year={2022}
}
```

**Dataset**: APE-data (Acute Pulmonary Embolism CT scans)

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- Latent diffusion concept from [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- DDIM sampling from [Song et al. 2021](https://arxiv.org/abs/2010.02502)
- Medical imaging techniques from MONAI framework
- Dataset: APE-data for pulmonary embolism detection

---

## 🚀 Get Started Now!

```bash
# 1. Test VAE reconstruction
python tests/test_vae_reconstruction.py

# 2. Deploy VAE training to Kubernetes
kubectl apply -f kub_files/vae-training-job-a100.yaml

# 3. Monitor training
kubectl logs -f job/vae-training-job-a100

# 4. After VAE completes, train diffusion model
kubectl apply -f kub_files/train-job-a100.yaml
```

**Questions?** Check [CLAUDE.md](CLAUDE.md) for complete project context and architecture details.

Happy Training! 🎉
