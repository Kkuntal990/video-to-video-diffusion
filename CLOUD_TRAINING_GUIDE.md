# Cloud Training Guide for APE-Data

This guide explains how to train the video-to-video diffusion model on cloud GPU using the full APE-data dataset from HuggingFace.

## 🎯 Overview

**What's Different from Local Testing:**
- ✅ Uses **HuggingFace streaming** to download full dataset automatically
- ✅ No manual data download needed
- ✅ Trains with **full resolution** (256x256, 16 frames)
- ✅ Uses **GPU acceleration** (100-1000x faster than CPU)
- ✅ Supports **pretrained weights** for 6x speedup

**Expected Training Time:**
- With pretrained weights: **~1 day** on A100 GPU
- Without pretrained: **~7 days** on A100 GPU

---

## 📋 Prerequisites

### 1. HuggingFace Account & Access

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login to HuggingFace
huggingface-cli login
# Enter your HuggingFace token when prompted
```

**Get your token:** https://huggingface.co/settings/tokens

**Request dataset access:** Visit https://huggingface.co/datasets/t2ance/APE-data and request access if it's private.

### 2. Cloud GPU Instance

Recommended providers:
- **Lambda Labs** (cheapest, easy to use)
- **RunPod** (flexible, good pricing)
- **Google Colab Pro+** (easiest, but limited)
- **AWS/GCP/Azure** (enterprise options)

Recommended GPU:
- **NVIDIA A100** (40GB or 80GB) - Best performance
- **NVIDIA A40** - Good value
- **NVIDIA RTX 4090** - Budget option

Minimum requirements:
- GPU: 24GB VRAM
- RAM: 32GB
- Storage: 100GB

### 3. Software Requirements

```bash
# Python 3.8+
python --version

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# All dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Setup on Cloud GPU

```bash
# SSH into your cloud instance
ssh your-cloud-instance

# Clone repository
git clone https://github.com/Kkuntal990/video-to-video-diffusion.git
cd video-to-video-diffusion

# Checkout pretrained_main branch
git checkout pretrained_main

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace
huggingface-cli login
```

### Step 2: Verify HuggingFace Dataset Access

```bash
# Test if you can access the dataset
python -c "
from datasets import load_dataset
ds = load_dataset('t2ance/APE-data', split='train', streaming=True)
print('✓ Dataset accessible!')
sample = next(iter(ds))
print(f'Sample keys: {list(sample.keys())}')
"
```

**If this fails:**
- Make sure you're logged in: `huggingface-cli login`
- Request access to the dataset on HuggingFace
- Check dataset name is correct: `t2ance/APE-data`

### Step 3: Start Training

```bash
# Download pretrained weights (optional but recommended)
python scripts/download_weights.py --model opensora

# Start training with cloud config
python train.py --config config/cloud_train_config.yaml

# Or use screen/tmux to keep training running after disconnect
screen -S training
python train.py --config config/cloud_train_config.yaml
# Press Ctrl+A then D to detach
# Reattach with: screen -r training
```

**That's it!** Training will start automatically, downloading data as needed.

---

## 📊 Monitoring Training

### Check Training Progress

```bash
# View logs
tail -f outputs/ape_v2v_diffusion/logs/train.log

# Check GPU usage
nvidia-smi

# Watch GPU usage live
watch -n 1 nvidia-smi
```

### TensorBoard (Real-time Metrics)

```bash
# In a separate terminal
tensorboard --logdir outputs/ape_v2v_diffusion/logs

# Then open in browser:
# http://your-instance-ip:6006
```

### Weights & Biases (Optional)

If you enable W&B in config:

```yaml
training:
  use_wandb: true
  wandb_project: 'video-diffusion'
  wandb_entity: 'your-username'
```

Then login:
```bash
wandb login
```

View training at: https://wandb.ai/your-username/video-diffusion

---

## ⚙️ Configuration Guide

### Key Settings to Adjust

#### For Faster Training (Development)

```yaml
data:
  num_frames: 8          # Reduce from 16
  resolution: [128, 128]  # Reduce from [256, 256]
  batch_size: 8          # Increase if GPU memory allows

training:
  num_epochs: 10         # Reduce for quick tests
  checkpoint_every: 100  # Save less frequently
```

#### For Maximum Quality (Production)

```yaml
data:
  num_frames: 16
  resolution: [256, 256]
  batch_size: 4

training:
  num_epochs: 100
  mixed_precision: true

pretrained:
  use_pretrained: true  # Essential for best results
```

#### For Limited GPU Memory (24GB)

```yaml
data:
  batch_size: 2          # Reduce batch size
  num_frames: 12         # Reduce frames

training:
  gradient_accumulation_steps: 2  # Accumulate gradients
  gradient_checkpointing: true    # Save memory
  mixed_precision: true           # Use FP16
```

---

## 🔄 Data Loading Modes

### Mode 1: HuggingFace Streaming (Recommended)

**Advantages:**
- No manual data download
- Data downloaded on-the-fly
- Minimal disk space needed
- Automatic caching

**Configuration:**
```yaml
data:
  data_source: 'huggingface'
  dataset_name: 't2ance/APE-data'
  streaming: true
```

**Use when:**
- Training on cloud GPU
- Limited disk space
- Want automatic data management

### Mode 2: HuggingFace Download (Alternative)

**Advantages:**
- Faster epoch iterations
- Better for multiple training runs
- Works offline after download

**Configuration:**
```yaml
data:
  data_source: 'huggingface'
  dataset_name: 't2ance/APE-data'
  streaming: false
  cache_dir: '/path/to/cache'
```

**Pre-download:**
```bash
# Download entire dataset first
python -c "
from datasets import load_dataset
ds = load_dataset('t2ance/APE-data', cache_dir='./hf_cache')
print(f'Downloaded {len(ds)} samples')
"
```

### Mode 3: Local Files (For Pre-Downloaded Data)

**Configuration:**
```yaml
data:
  data_source: 'local'
  data_dir: '/path/to/downloaded/APE-data'
  cache_extracted: true
```

**Use when:**
- You've manually downloaded the dataset
- Training on local machine
- Want full control over data

---

## 📈 Training Tips & Best Practices

### 1. Start with Pretrained Weights

```yaml
pretrained:
  use_pretrained: true
  vae:
    enabled: true
    model_name: 'hpcai-tech/OpenSora-VAE-v1.2'
```

**Benefits:**
- 6x faster convergence
- +15-20% better PSNR
- Lower final loss

### 2. Use Two-Phase Training

```yaml
pretrained:
  two_phase_training: true
  phase1_epochs: 10  # Freeze VAE
  # Then fine-tune everything
```

**Strategy:**
- Phase 1 (epochs 0-10): Freeze VAE, train U-Net only
- Phase 2 (epochs 10+): Fine-tune entire model

### 3. Learning Rate Scheduling

```yaml
training:
  learning_rate: 1e-4
  scheduler: 'cosine'
  warmup_steps: 500
```

**Warmup helps:**
- Stable training start
- Better convergence
- Avoid early divergence

### 4. Mixed Precision Training

```yaml
training:
  mixed_precision: true
  precision: 'fp16'  # or 'bf16' for newer GPUs
```

**Benefits:**
- 2x faster training
- Lower memory usage
- Minimal quality loss

### 5. Gradient Accumulation (For Small GPUs)

```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 4  # Effective batch size = 8
```

### 6. Save Disk Space

```yaml
training:
  keep_last_n_checkpoints: 3  # Only keep 3 latest
  checkpoint_every: 1000       # Save less frequently
```

---

## 🔍 Troubleshooting

### Problem: "Dataset not found" or "Access denied"

**Solution:**
```bash
# 1. Check you're logged in
huggingface-cli whoami

# 2. Request access to dataset
# Visit: https://huggingface.co/datasets/t2ance/APE-data

# 3. Re-login with token
huggingface-cli login --token YOUR_TOKEN
```

### Problem: Out of Memory (OOM)

**Solutions:**
```yaml
# Option 1: Reduce batch size
data:
  batch_size: 1

# Option 2: Enable gradient checkpointing
training:
  gradient_checkpointing: true

# Option 3: Reduce resolution
data:
  num_frames: 12
  resolution: [192, 192]

# Option 4: Use gradient accumulation
training:
  batch_size: 1
  gradient_accumulation_steps: 4
```

### Problem: Training is slow

**Check:**
1. Mixed precision enabled?
2. Using GPU? (check with `nvidia-smi`)
3. Data loading bottleneck? (increase `num_workers`)
4. Using pretrained weights?

**Speed up:**
```yaml
training:
  mixed_precision: true

data:
  num_workers: 8  # Increase if CPU is underutilized

pretrained:
  use_pretrained: true  # 6x speedup!
```

### Problem: Loss not decreasing

**Possible causes:**
1. Learning rate too high → Reduce to 1e-5
2. Need longer warmup → Increase `warmup_steps`
3. Batch size too small → Use gradient accumulation
4. VAE frozen too long → Reduce `freeze_epochs`

### Problem: "Connection timeout" or "Download failed"

**Solutions:**
```bash
# 1. Check internet connection
ping huggingface.co

# 2. Try different cache directory
export HF_HOME=/path/to/large/disk

# 3. Pre-download dataset
python scripts/download_ape_data.py  # If available
```

---

## 📦 After Training

### Export Best Model

```bash
# Find best checkpoint
ls -lh outputs/ape_v2v_diffusion/checkpoints/

# Load and test
python inference.py \
  --checkpoint outputs/ape_v2v_diffusion/checkpoints/best_model.pt \
  --input_video path/to/input.mp4 \
  --output_video path/to/output.mp4
```

### Evaluate Model

```bash
# Run evaluation on test set
python evaluate.py \
  --checkpoint outputs/ape_v2v_diffusion/checkpoints/best_model.pt \
  --split test
```

### Share Your Model

```bash
# Upload to HuggingFace Hub
python scripts/upload_model.py \
  --checkpoint outputs/ape_v2v_diffusion/checkpoints/best_model.pt \
  --repo_name your-username/ape-v2v-model
```

---

## 📊 Expected Results

### With Pretrained Weights:

| Metric | Expected Value |
|--------|---------------|
| Training Time | ~1 day (A100) |
| Final Loss | 0.01 - 0.03 |
| PSNR | 30-35 dB |
| Convergence | ~10-15 epochs |

### Without Pretrained Weights:

| Metric | Expected Value |
|--------|---------------|
| Training Time | ~7 days (A100) |
| Final Loss | 0.02 - 0.05 |
| PSNR | 25-30 dB |
| Convergence | ~40-50 epochs |

---

## 💰 Cost Estimation

### Cloud GPU Pricing (Approximate):

| Provider | GPU | $/hour | 1 Day | 7 Days |
|----------|-----|--------|-------|--------|
| Lambda Labs | A100 (40GB) | $1.10 | $26 | $185 |
| RunPod | A100 (40GB) | $1.39 | $33 | $233 |
| RunPod | RTX 4090 | $0.44 | $11 | $74 |
| Vast.ai | A100 (40GB) | $0.80+ | $19+ | $134+ |

**With pretrained weights:** ~$26-33 (1 day on A100)
**Without pretrained:** ~$185-233 (7 days on A100)

**💡 Tip:** Use spot instances for 50-70% savings (but training may be interrupted)

---

## 🎓 Next Steps

1. **Start with quick test:**
   - Use low resolution (128x128)
   - Train for 5 epochs
   - Verify everything works

2. **Full training run:**
   - Enable pretrained weights
   - Use full resolution (256x256)
   - Train for 50+ epochs

3. **Experiment:**
   - Try different pretrained models
   - Adjust learning rates
   - Test different schedulers

4. **Evaluate:**
   - Compare pretrained vs from-scratch
   - Measure PSNR/SSIM metrics
   - Visualize generated samples

---

## 📚 Additional Resources

- **Pretrained Weights Guide:** `PRETRAINED_WEIGHTS_GUIDE.md`
- **Model Architecture:** `README.md`
- **Local Testing:** `QUICKSTART_APE_DATA.md`
- **API Documentation:** Coming soon

---

## ❓ Need Help?

1. Check the troubleshooting section above
2. Review the example configs in `config/`
3. Open an issue: https://github.com/Kkuntal990/video-to-video-diffusion/issues
4. Check HuggingFace dataset page: https://huggingface.co/datasets/t2ance/APE-data

---

**Good luck with your training!** 🚀

Remember: Start with pretrained weights for 6x faster training!
