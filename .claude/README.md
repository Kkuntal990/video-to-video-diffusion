# Claude Context: CT Slice Interpolation Training Project

**Project**: Video-to-Video Diffusion for CT Slice Interpolation
**Owner**: Kuntal Kokate (kkokate@ucsd.edu)
**Last Updated**: November 9, 2025

---

## Project Overview

This project implements a **Video-to-Video Diffusion model** for CT slice interpolation, converting thick CT slices (50 @ 5.0mm) to thin slices (300 @ 1.0mm) at 512×512 resolution. The model uses a pretrained MAISI VAE (frozen) with a trainable 3D U-Net.

**Key Components**:
- **VAE**: MAISI pretrained (130M params, frozen)
- **U-Net**: 3D denoising network (264M params, trainable)
- **Diffusion**: Cosine noise schedule, 1000 timesteps
- **Dataset**: APE CT scans (323 patients)

---

## Critical Information for Claude

### Current Status (November 9, 2025)
✅ **All critical bugs fixed** - Ready for production training

**Recent Fixes Applied**:
1. Freeze MAISI VAE weights (models/vae.py:252-255)
2. Chunked encoding for memory optimization (32-slice chunks)
3. Filter optimizer to trainable params only (train.py:152-210)
4. Fix validation interval config (val_interval: 500)
5. Add end-of-training validation (trainer.py:404-409)
6. Validate files before train-test split (dataset.py:138-412)

**See**: `CURRENT_STATUS.md` for complete details

---

## Key Configuration

**Model Config**: `config/slice_interpolation_full_medium.yaml`

**Critical Settings**:
- `use_custom_maisi: true` - Use pretrained MAISI VAE
- `vae_encoder: 0.0`, `vae_decoder: 0.0` - VAE frozen
- `gradient_checkpointing: true` - Memory optimization
- `val_interval: 500` - Validation every 500 steps
- `resolution: [512, 512]` - High resolution for medical imaging
- `use_amp: true` - Mixed precision training

**Memory Optimization**:
- Chunked encoding: 32 slices per chunk
- Chunked decoding: 8 slices per chunk
- Expected peak: 30-40 GB (fits A100 80GB)

---

## Important File Locations

### Code Structure
```
models/
├── vae.py              # MAISI VAE with chunked encoding
├── model.py            # Main VideoToVideoDiffusion model
├── unet3d.py           # 3D U-Net architecture
├── diffusion.py        # Gaussian diffusion process
└── losses.py           # Loss functions

data/
├── slice_interpolation_dataset.py  # Main dataset loader
├── ape_cached_dataset.py           # Legacy dataset
└── get_dataloader.py               # Dataloader factory

training/
├── trainer.py          # Main training loop
└── scheduler.py        # Learning rate schedulers

config/
└── slice_interpolation_full_medium.yaml  # Active config

kub_files/
├── training-job-a100.yaml          # Kubernetes training job
└── download-dataset-job.yaml       # Dataset download job
```

### Remote Paths (Kubernetes)
- **Dataset**: `/workspace/storage_a100/dataset`
- **Cache**: `/workspace/storage_a100/.cache/processed`
- **Logs**: `/workspace/storage_a100/logs/slice_interp_full_medium`
- **Checkpoints**: `/workspace/storage_a100/checkpoints/slice_interp_full_medium`
- **MAISI checkpoint**: `/workspace/storage_a100/pretrained/maisi_vae/models/autoencoder.pt`

---

## Kubernetes Context

**Cluster**: Nautilus HyperCluster (Pacific Research Platform)
**Namespace**: `ecepxie`
**Authentication**: OIDC via authentik.nrp-nautilus.io
**User**: kkokate@ucsd.edu (ID: 402254)

**Key Commands**:
```bash
# Set namespace
kubectl config set-context --current --namespace=ecepxie

# Get pod name
POD_NAME=$(kubectl get pods -l job-name=v2v-diffusion-training-job-a100 -o jsonpath='{.items[0].metadata.name}')

# View logs
kubectl logs -f $POD_NAME

# Monitor GPU
kubectl exec $POD_NAME -- nvidia-smi

# Delete/restart job
kubectl delete -f kub_files/training-job-a100.yaml
kubectl apply -f kub_files/training-job-a100.yaml
```

---

## Docker Image

**Registry**: GitHub Container Registry (ghcr.io)
**Image**: `ghcr.io/kkuntal990/v2v-diffusion:latest`

**Build Command**:
```bash
docker buildx build --no-cache --platform linux/amd64 \
  -t ghcr.io/kkuntal990/v2v-diffusion:latest \
  --push .
```

**Base**: Miniconda with PyTorch 2.1.0, CUDA 12.1

---

## Common Issues and Solutions

### Issue: OOM Errors
**Solutions**:
1. Reduce chunk sizes (models/vae.py): 32→16 (encode), 8→4 (decode)
2. Reduce resolution (config): 512→384
3. Disable mixed precision: `use_amp: false`
4. Check VAE is frozen: Look for "0 trainable" in logs

### Issue: GradScaler Assertion Errors
**Root Cause**: Optimizer contains frozen parameters
**Solution**: Filter optimizer to trainable params only (already implemented in train.py)

### Issue: Validation Not Running
**Check**: Config has `val_interval: 500` (not `validate_every`)
**Location**: Line 118 in config file

### Issue: Dataset Preprocessing Failures
**Check**: `/workspace/storage_a100/.cache/preprocessing_failures.txt`
**Solution**: Add case IDs to `corrupted_cases` list in dataset.py

---

## Expected Training Behavior

### Healthy Training Logs
```
✓ MAISI VAE weights frozen (requires_grad=False)
Total parameters: 285,561,605
  Trainable: 264,616,708
  VAE: 20,944,897 (0 trainable)
  U-Net: 264,616,708

Optimizer param groups: 1
  Group 0 (unet): 264,616,708 params, LR: 1.00e-04

Train set: 243 patients
Val set: 48 patients

Using chunked encoding: 50 slices → 2 chunks of max 32 slices
Memory savings: ~36% peak reduction vs full volume

Epoch 1:  41%|████| 100/243 [20:00<29:00, 12.5s/it, loss=0.95]
Step 500: Validation - Loss: 0.92, PSNR: 28.5 dB, SSIM: 0.82
```

### Red Flags
- ❌ "VAE: 20,944,897 (20,944,897 trainable)" - VAE not frozen!
- ❌ "Optimizer param groups: 3" - Frozen params in optimizer!
- ❌ Memory usage > 70 GB - Chunking not working
- ❌ No chunking message - Check vae.py implementation
- ❌ Training fails at batch 7 - GradScaler issue

---

## Code Modification Guidelines

### When Modifying VAE Code
- **Always check**: VAE freezing (models/vae.py:252-255)
- **Always check**: Chunked encoding enabled (models/vae.py:409-515)
- **Test with**: Small volume first (50 slices)

### When Modifying Training Code
- **Always check**: Optimizer filters trainable params (train.py:152-196)
- **Always check**: Validation interval config name (val_interval)
- **Test with**: 1 epoch first

### When Modifying Dataset Code
- **Always check**: Train-test split happens after validation
- **Always check**: Failed cases excluded from splits
- **Cache location**: `/workspace/storage_a100/.cache/processed`

---

## Performance Expectations

**Training Speed**:
- Batch time: 12-15 seconds
- Epoch time: ~50 minutes (243 batches)
- Total training: ~85 hours (100 epochs)

**Memory**:
- Peak: 30-40 GB (with all optimizations)
- Without VAE freezing: 78+ GB (OOM)
- Without chunking: 100-189 GB (OOM)

**Metrics** (by end of training):
- PSNR: 30-35 dB (higher is better)
- SSIM: 0.85-0.92 (higher is better)
- Loss: ~0.7 (from initial ~1.0)

---

## Documentation Structure

**Essential Docs** (keep these):
1. `README.md` - Project overview and quick start
2. `ARCHITECTURE.md` - System architecture and design
3. `CURRENT_STATUS.md` - Latest fixes and deployment status ⭐
4. `DEPLOYMENT_GUIDE.md` - Kubernetes deployment procedures
5. `MAISI_VAE_GUIDE.md` - MAISI VAE integration details
6. `SLICE_INTERPOLATION_GUIDE.md` - Task-specific guide

**This File** (`.claude/README.md`):
- Persistent context for Claude across sessions
- Quick reference for critical information
- Troubleshooting guide

---

## Working with Claude

### When Starting a New Session
1. Claude automatically reads this file
2. Refer to `CURRENT_STATUS.md` for latest state
3. Check recent git commits for changes

### When Debugging Issues
1. Check logs first: `kubectl logs -f $POD_NAME`
2. Check GPU memory: `kubectl exec $POD_NAME -- nvidia-smi`
3. Compare with "Expected Behavior" sections above
4. Check cache for failures: `.cache/preprocessing_failures.txt`

### When Making Changes
1. Read relevant guide first (MAISI_VAE_GUIDE, etc.)
2. Make changes locally
3. Test build: `docker build -t test .`
4. Deploy to Kubernetes
5. Monitor first epoch carefully
6. Update `CURRENT_STATUS.md` if adding new features/fixes

---

## Quick Reference

**Build & Deploy**:
```bash
# 1. Build
docker buildx build --no-cache --platform linux/amd64 \
  -t ghcr.io/kkuntal990/v2v-diffusion:latest --push .

# 2. Deploy
kubectl delete -f kub_files/training-job-a100.yaml
kubectl apply -f kub_files/training-job-a100.yaml

# 3. Monitor
POD_NAME=$(kubectl get pods -l job-name=v2v-diffusion-training-job-a100 -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f $POD_NAME
```

**Check Training Health**:
```bash
# Watch for "frozen", "trainable", "chunks", "Optimizer param groups"
kubectl logs $POD_NAME | grep -E "(frozen|trainable|chunks|Optimizer param groups)"

# Monitor GPU memory
watch -n 2 "kubectl exec $POD_NAME -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
```

## Important CLAUDE CODE instructions 

1. Do not create readme files while or after solving a task unless explicity asked to. 
2. We have python environment ~/miniconda3/envs/ct-superres-mps/bin/python3
3. Do not commit or push unless explicity instructed.

**Last Updated**: November 9, 2025 by Claude
**Status**: Production ready ✅
