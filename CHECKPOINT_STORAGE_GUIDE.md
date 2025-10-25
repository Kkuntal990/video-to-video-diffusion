# Checkpoint Storage Guide for Kubernetes

This guide explains how to save checkpoints to persistent storage in Kubernetes and how to use them for resuming training or inference.

## Table of Contents
1. [Understanding the Storage Setup](#understanding-the-storage-setup)
2. [Saving Checkpoints to Persistent Storage](#saving-checkpoints-to-persistent-storage)
3. [Verifying Checkpoints are Saved](#verifying-checkpoints-are-saved)
4. [Using Saved Checkpoints](#using-saved-checkpoints)
5. [Troubleshooting](#troubleshooting)

---

## Understanding the Storage Setup

### Current Kubernetes Configuration

Your setup includes:

**PersistentVolumeClaim (PVC):**
- Name: `v2v-diffuser-kuntal`
- Size: 20Gi
- Storage Class: `rook-ceph-block`
- Access Mode: ReadWriteOnce

**Volume Mount in Pods:**
- Persistent storage is mounted at: `/workspace/storage`
- Everything saved to `/workspace/storage/*` persists across pod restarts
- Everything saved to `/workspace/*` (outside storage) is **ephemeral** and lost when pod dies

### Storage Paths

```
/workspace/                          # Ephemeral (lost on pod restart)
  ├── code/                          # Application code
  ├── train.py
  └── storage/                       # PERSISTENT VOLUME (survives pod restarts)
      ├── checkpoints/               # ✓ Checkpoints saved here persist
      ├── logs/                      # ✓ Logs saved here persist
      ├── outputs/                   # ✓ Outputs saved here persist
      └── .cache/                    # ✓ Model cache (HuggingFace, Torch)
          ├── huggingface/
          └── torch/
```

---

## Saving Checkpoints to Persistent Storage

### Step 1: Update Training Configuration

The configuration has already been updated in `config/cloud_train_config.yaml`:

```yaml
training:
  # Logging
  output_dir: '/workspace/storage/outputs'           # Persistent
  log_dir: '/workspace/storage/logs'                 # Persistent
  checkpoint_dir: '/workspace/storage/checkpoints'   # Persistent
  experiment_name: 'ape_v2v_diffusion'
```

**✓ This ensures all checkpoints, logs, and outputs are saved to persistent storage.**

### Step 2: Verify PVC is Created

```bash
# Check if PVC exists
kubectl get pvc v2v-diffuser-kuntal

# If not, create it
kubectl apply -f kub_files/persistent_storage.yaml
```

### Step 3: Deploy Training Pod/Job

```bash
# Option 1: Training Job (recommended for production)
kubectl apply -f kub_files/training-job.yaml

# Option 2: Training Pod (for testing)
kubectl apply -f kub_files/training-pod.yaml

# Option 3: Interactive Pod (for development)
kubectl apply -f kub_files/interactive-pod.yaml
```

---

## Verifying Checkpoints are Saved

### Method 1: Use the Verification Script

```bash
./verify_persistent_storage.sh
```

This script checks:
- PVC exists and is bound
- Pod is running
- Storage is mounted correctly
- Checkpoint/log directories exist

### Method 2: Manual Verification

```bash
# Set your pod name
POD_NAME="v2v-diffusion-training-pod"  # or v2v-diffusion-training-xxxxx for jobs

# Check storage mount
kubectl exec $POD_NAME -- df -h /workspace/storage

# List checkpoints
kubectl exec $POD_NAME -- ls -lh /workspace/storage/checkpoints/

# List logs
kubectl exec $POD_NAME -- ls -lh /workspace/storage/logs/

# Check storage usage
kubectl exec $POD_NAME -- du -sh /workspace/storage/*
```

### Method 3: Monitor Checkpoints During Training

```bash
# Watch checkpoints being created (updates every 5 seconds)
kubectl exec $POD_NAME -- watch -n 5 ls -lh /workspace/storage/checkpoints/

# Or use kubectl logs to monitor training progress
kubectl logs -f $POD_NAME
```

### Expected Checkpoint Structure

```
/workspace/storage/checkpoints/ape_v2v_diffusion/
├── checkpoint_epoch_0.pt          # Checkpoint after epoch 0
├── checkpoint_epoch_1.pt          # Checkpoint after epoch 1
├── checkpoint_step_500.pt         # Checkpoint after 500 steps
├── checkpoint_step_1000.pt        # Checkpoint after 1000 steps
└── checkpoint_final.pt            # Final checkpoint
```

Each checkpoint contains:
- Model state dict (VAE + U-Net weights)
- Optimizer state dict
- Scheduler state dict (if used)
- Training metadata (epoch, step, loss)

---

## Using Saved Checkpoints

### Option 1: Resume Training from Checkpoint

#### Step 1: Find Available Checkpoints

```bash
# List all checkpoints
kubectl exec $POD_NAME -- ls -lh /workspace/storage/checkpoints/ape_v2v_diffusion/

# Example output:
# checkpoint_epoch_0.pt   (128 MB)
# checkpoint_epoch_1.pt   (169 MB)
# checkpoint_final.pt     (169 MB)
```

#### Step 2: Update Config to Resume

Edit `config/cloud_train_config.yaml`:

```yaml
resume:
  checkpoint_path: '/workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_epoch_1.pt'
  resume_optimizer: true   # Resume optimizer state
  resume_scheduler: true   # Resume scheduler state
```

#### Step 3: Run Training with Resume

```bash
# Using the training script
python train.py --config config/cloud_train_config.yaml \
  --resume /workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_epoch_1.pt
```

Or in Kubernetes:

```bash
# Update the pod/job args to include --resume flag
kubectl exec $POD_NAME -- python train.py \
  --config config/cloud_train_config.yaml \
  --resume /workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_epoch_1.pt
```

### Option 2: Load Checkpoint for Inference

Create an inference script:

```python
# inference.py
import torch
from models import VideoToVideoDiffusion

# Load model config
model_config = {
    'in_channels': 3,
    'latent_dim': 4,
    'vae_base_channels': 128,
    'unet_model_channels': 128,
    # ... other config
}

# Create model
model = VideoToVideoDiffusion(model_config)

# Load checkpoint from persistent storage
checkpoint_path = '/workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_final.pt'
checkpoint = torch.load(checkpoint_path, map_location='cuda')

# Load model weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")

# Now use model for inference
# ...
```

### Option 3: Download Checkpoints to Local Machine

```bash
# Copy entire checkpoint directory
kubectl cp v2v-diffusion-training-pod:/workspace/storage/checkpoints ./local_checkpoints

# Copy specific checkpoint
kubectl cp v2v-diffusion-training-pod:/workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_final.pt ./checkpoint_final.pt

# Copy with compression (for large files)
kubectl exec v2v-diffusion-training-pod -- tar czf - /workspace/storage/checkpoints | tar xzf - -C ./
```

### Option 4: Use Checkpoints in New Pod

When creating a new pod, the same PVC will be automatically mounted:

```yaml
# Any new pod using the same PVC
volumes:
- name: storage
  persistentVolumeClaim:
    claimName: v2v-diffuser-kuntal  # Same PVC = same checkpoints available
```

```bash
# Deploy interactive pod to explore checkpoints
kubectl apply -f kub_files/interactive-pod.yaml

# Exec into pod
kubectl exec -it v2v-diffusion-interactive -- /bin/bash

# Inside pod, checkpoints are available
cd /workspace/storage/checkpoints
ls -lh
```

---

## Advanced Usage

### 1. Checkpoint Management Script

Create `manage_checkpoints.py`:

```python
#!/usr/bin/env python3
"""
Checkpoint management utility
"""
import os
from pathlib import Path
import torch

CHECKPOINT_DIR = Path('/workspace/storage/checkpoints/ape_v2v_diffusion')

def list_checkpoints():
    """List all available checkpoints"""
    if not CHECKPOINT_DIR.exists():
        print("No checkpoints found")
        return

    checkpoints = sorted(CHECKPOINT_DIR.glob('*.pt'))
    print(f"\nAvailable checkpoints in {CHECKPOINT_DIR}:\n")

    for ckpt in checkpoints:
        size_mb = ckpt.stat().st_size / (1024 * 1024)

        # Load checkpoint metadata
        try:
            checkpoint = torch.load(ckpt, map_location='cpu')
            epoch = checkpoint.get('epoch', 'N/A')
            step = checkpoint.get('global_step', 'N/A')
            print(f"  {ckpt.name:40s} | {size_mb:6.1f} MB | Epoch: {epoch:3} | Step: {step:6}")
        except Exception as e:
            print(f"  {ckpt.name:40s} | {size_mb:6.1f} MB | Error: {e}")

def get_best_checkpoint():
    """Get the latest/best checkpoint"""
    checkpoints = list(CHECKPOINT_DIR.glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        return None

    # Sort by epoch number
    latest = sorted(checkpoints)[-1]
    return latest

def clean_old_checkpoints(keep_last_n=3):
    """Keep only the last N checkpoints to save space"""
    checkpoints = sorted(CHECKPOINT_DIR.glob('checkpoint_epoch_*.pt'))

    if len(checkpoints) <= keep_last_n:
        print(f"Only {len(checkpoints)} checkpoints, nothing to clean")
        return

    to_delete = checkpoints[:-keep_last_n]

    for ckpt in to_delete:
        print(f"Deleting {ckpt.name}...")
        ckpt.unlink()

    print(f"✓ Cleaned {len(to_delete)} old checkpoints")

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_checkpoints.py list")
        print("  python manage_checkpoints.py best")
        print("  python manage_checkpoints.py clean [keep_n]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'list':
        list_checkpoints()
    elif cmd == 'best':
        best = get_best_checkpoint()
        if best:
            print(f"Best checkpoint: {best}")
        else:
            print("No checkpoints found")
    elif cmd == 'clean':
        keep_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        clean_old_checkpoints(keep_n)
    else:
        print(f"Unknown command: {cmd}")
```

Usage:
```bash
# Inside pod
python manage_checkpoints.py list
python manage_checkpoints.py best
python manage_checkpoints.py clean 3  # Keep last 3 checkpoints
```

### 2. Automatic Checkpoint Cleanup

Add to `config/cloud_train_config.yaml`:

```yaml
training:
  checkpoint_every: 500           # Save every 500 steps
  keep_last_n_checkpoints: 3      # Only keep last 3 checkpoints
```

### 3. Copy Checkpoints to External Storage

```bash
# Copy to S3 bucket
kubectl exec $POD_NAME -- aws s3 cp \
  /workspace/storage/checkpoints/ape_v2v_diffusion/ \
  s3://my-bucket/v2v-checkpoints/ \
  --recursive

# Copy to Google Cloud Storage
kubectl exec $POD_NAME -- gsutil -m cp -r \
  /workspace/storage/checkpoints/ape_v2v_diffusion/ \
  gs://my-bucket/v2v-checkpoints/
```

---

## Troubleshooting

### Problem 1: Checkpoints Not Persisting

**Symptom:** Checkpoints disappear after pod restart

**Solution:**
```bash
# 1. Verify checkpoint_dir points to persistent storage
grep checkpoint_dir config/cloud_train_config.yaml
# Should show: /workspace/storage/checkpoints

# 2. Check if storage is mounted
kubectl exec $POD_NAME -- mount | grep storage

# 3. Verify PVC is bound
kubectl get pvc v2v-diffuser-kuntal
# Status should be "Bound"
```

### Problem 2: Out of Disk Space

**Symptom:** "No space left on device" error

**Solution:**
```bash
# Check storage usage
kubectl exec $POD_NAME -- df -h /workspace/storage

# Clean old checkpoints
kubectl exec $POD_NAME -- python manage_checkpoints.py clean 2

# Or manually delete
kubectl exec $POD_NAME -- rm /workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_epoch_*.pt

# If still out of space, resize PVC
kubectl edit pvc v2v-diffuser-kuntal
# Change: storage: 20Gi -> storage: 50Gi
```

### Problem 3: Checkpoint Loading Fails

**Symptom:** "Error loading checkpoint" or "KeyError" when resuming

**Solution:**
```bash
# Check checkpoint integrity
kubectl exec $POD_NAME -- python -c "
import torch
ckpt = torch.load('/workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_final.pt')
print('Keys:', ckpt.keys())
print('Epoch:', ckpt.get('epoch'))
print('Step:', ckpt.get('global_step'))
"

# If checkpoint is corrupted, use previous checkpoint
kubectl exec $POD_NAME -- ls -lht /workspace/storage/checkpoints/ape_v2v_diffusion/
```

### Problem 4: PVC Not Mounting

**Symptom:** Pod stuck in "ContainerCreating" state

**Solution:**
```bash
# Check pod events
kubectl describe pod $POD_NAME | grep -A 10 Events

# Check PVC status
kubectl describe pvc v2v-diffuser-kuntal

# If PVC is "Pending", check storage class
kubectl get storageclass rook-ceph-block

# Recreate PVC if needed
kubectl delete pvc v2v-diffuser-kuntal
kubectl apply -f kub_files/persistent_storage.yaml
```

### Problem 5: Permission Denied

**Symptom:** "Permission denied" when writing checkpoints

**Solution:**
```bash
# Check directory permissions
kubectl exec $POD_NAME -- ls -ld /workspace/storage

# Fix permissions
kubectl exec $POD_NAME -- chmod -R 777 /workspace/storage

# Or create directories with correct ownership
kubectl exec $POD_NAME -- mkdir -p /workspace/storage/checkpoints
kubectl exec $POD_NAME -- chmod 777 /workspace/storage/checkpoints
```

---

## Best Practices

### 1. Checkpoint Naming Convention

Use descriptive names:
```python
# In training config
checkpoint_name = f"checkpoint_epoch_{epoch}_loss_{loss:.4f}.pt"
```

### 2. Save Checkpoint Frequency

Balance between safety and storage:
```yaml
training:
  checkpoint_every: 500      # Save every 500 steps (frequent)
  keep_last_n_checkpoints: 5  # Keep only last 5
```

### 3. Monitor Storage Usage

Add monitoring to training script:
```python
import shutil

def check_disk_space(path='/workspace/storage'):
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)

    if free_gb < 1:
        logger.warning(f"⚠️  Low disk space: {free_gb:.2f} GB remaining")

    return free_gb
```

### 4. Backup Critical Checkpoints

```bash
# Backup final checkpoint
kubectl cp $POD_NAME:/workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_final.pt \
  ./backups/checkpoint_final_$(date +%Y%m%d).pt
```

### 5. Use Symbolic Links for Latest

```bash
# Inside pod, create symlink to latest checkpoint
ln -sf checkpoint_epoch_10.pt checkpoint_latest.pt

# Then always use checkpoint_latest.pt for resuming
```

---

## Quick Reference Commands

```bash
# Check PVC status
kubectl get pvc v2v-diffuser-kuntal

# List checkpoints
kubectl exec $POD_NAME -- ls -lh /workspace/storage/checkpoints/ape_v2v_diffusion/

# Monitor training logs
kubectl logs -f $POD_NAME

# Check disk usage
kubectl exec $POD_NAME -- df -h /workspace/storage

# Copy checkpoint to local
kubectl cp $POD_NAME:/workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_final.pt ./

# Resume training
kubectl exec $POD_NAME -- python train.py \
  --config config/cloud_train_config.yaml \
  --resume /workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_final.pt

# Clean old checkpoints
kubectl exec $POD_NAME -- find /workspace/storage/checkpoints -name "checkpoint_step_*.pt" -delete
```

---

## Summary

✅ **Checkpoints are now configured to save to persistent storage**
- Path: `/workspace/storage/checkpoints/`
- PVC: `v2v-diffuser-kuntal` (20Gi)
- Survives pod restarts and failures

✅ **Checkpoints can be used for:**
- Resuming training after interruption
- Inference and evaluation
- Transfer learning
- Sharing models across pods

✅ **Run verification:**
```bash
./verify_persistent_storage.sh
```

✅ **Start training:**
```bash
kubectl apply -f kub_files/training-job.yaml
kubectl logs -f v2v-diffusion-training-xxxxx
```

For questions or issues, check the troubleshooting section above.
