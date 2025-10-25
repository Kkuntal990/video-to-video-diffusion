# How to Run Training in Kubernetes

This guide provides step-by-step instructions to run your Video-to-Video Diffusion training on Kubernetes.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Step-by-Step Instructions](#step-by-step-instructions)
4. [Monitoring Training](#monitoring-training)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

✓ Docker image built and pushed: `kkokate990/v2v-diffusion:latest` or `ghcr.io/kkuntal990/v2v-diffusion:latest`
✓ Kubernetes cluster access with GPU nodes
✓ kubectl configured and working
✓ PersistentVolumeClaim created (20Gi storage)

---

## Quick Start

### Option 1: Training Job (Recommended for Production)

```bash
# 1. Create PersistentVolumeClaim (one-time setup)
kubectl apply -f kub_files/persistent_storage.yaml

# 2. Submit training job
kubectl apply -f kub_files/training-job.yaml

# 3. Monitor logs
kubectl logs -f job/v2v-diffusion-training
```

### Option 2: Interactive Development

```bash
# 1. Create PersistentVolumeClaim (one-time setup)
kubectl apply -f kub_files/persistent_storage.yaml

# 2. Start interactive pod
kubectl apply -f kub_files/interactive-pod.yaml

# 3. Wait for pod to be ready
kubectl wait --for=condition=ready pod/v2v-diffusion-interactive --timeout=5m

# 4. Exec into pod
kubectl exec -it v2v-diffusion-interactive -- /bin/bash

# 5. Inside pod, run training
python train.py --config config/cloud_train_config.yaml
```

---

## Step-by-Step Instructions

### Step 1: Verify Kubernetes Access

```bash
# Check cluster connection
kubectl cluster-info

# Check available GPU nodes
kubectl get nodes -l nvidia.com/gpu.product=Tesla-V100-SXM2-32GB

# Expected output:
# NAME           STATUS   ROLES    AGE   VERSION
# gpu-node-1     Ready    <none>   10d   v1.27.0
```

### Step 2: Create Persistent Storage

```bash
# Create PersistentVolumeClaim (20Gi)
kubectl apply -f kub_files/persistent_storage.yaml

# Verify PVC is created and bound
kubectl get pvc v2v-diffuser-kuntal

# Expected output:
# NAME                   STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS        AGE
# v2v-diffuser-kuntal    Bound    pvc-xxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx      20Gi       RWO            rook-ceph-block     1m

# If status is "Pending", check storage class:
kubectl get storageclass rook-ceph-block
```

### Step 3: Choose Your Deployment Method

#### Method A: Training Job (Batch Processing)

**When to use:**
- Production training runs
- Automated training pipelines
- Don't need interactive access

**Commands:**
```bash
# Submit training job
kubectl apply -f kub_files/training-job.yaml

# Check job status
kubectl get jobs v2v-diffusion-training

# Get pod name created by job
POD_NAME=$(kubectl get pods -l job-name=v2v-diffusion-training -o jsonpath='{.items[0].metadata.name}')
echo "Pod name: $POD_NAME"

# Monitor training logs (real-time)
kubectl logs -f $POD_NAME

# Or use job logs directly
kubectl logs -f job/v2v-diffusion-training
```

#### Method B: Training Pod (Simple)

**When to use:**
- Quick training runs
- Testing configurations
- Simpler than jobs

**Commands:**
```bash
# Start training pod
kubectl apply -f kub_files/training-pod.yaml

# Wait for pod to start
kubectl wait --for=condition=ready pod/v2v-diffusion-training-pod --timeout=5m

# Monitor logs
kubectl logs -f v2v-diffusion-training-pod
```

#### Method C: Interactive Pod (Development)

**When to use:**
- Development and debugging
- Testing code changes
- Exploring checkpoints
- Manual training control

**Commands:**
```bash
# Start interactive pod
kubectl apply -f kub_files/interactive-pod.yaml

# Wait for pod to be ready
kubectl wait --for=condition=ready pod/v2v-diffusion-interactive --timeout=5m

# Exec into pod
kubectl exec -it v2v-diffusion-interactive -- /bin/bash

# Inside the pod:
cd /workspace
ls -la  # Check files

# Run training
python train.py --config config/cloud_train_config.yaml

# Or with specific options
python train.py --config config/cloud_train_config.yaml --resume /workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_epoch_1.pt
```

---

## Monitoring Training

### Check Pod Status

```bash
# List all pods
kubectl get pods | grep v2v

# Detailed pod info
kubectl describe pod <POD_NAME>

# Check resource usage
kubectl top pod <POD_NAME>
```

### View Training Logs

```bash
# Real-time logs
kubectl logs -f <POD_NAME>

# Last 100 lines
kubectl logs <POD_NAME> --tail=100

# Save logs to file
kubectl logs <POD_NAME> > training.log

# Filter for specific info
kubectl logs <POD_NAME> | grep "Epoch"
kubectl logs <POD_NAME> | grep "loss"
```

### Check GPU Usage

```bash
# Exec into pod and check GPU
kubectl exec <POD_NAME> -- nvidia-smi

# Watch GPU usage (updates every 2 seconds)
kubectl exec <POD_NAME> -- watch -n 2 nvidia-smi

# Check GPU memory usage
kubectl exec <POD_NAME> -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Monitor Checkpoints

```bash
# List checkpoints
kubectl exec <POD_NAME> -- ls -lh /workspace/storage/checkpoints/ape_v2v_diffusion/

# Watch checkpoints being created
kubectl exec <POD_NAME> -- watch -n 5 'ls -lh /workspace/storage/checkpoints/ape_v2v_diffusion/'

# Check storage usage
kubectl exec <POD_NAME> -- df -h /workspace/storage
kubectl exec <POD_NAME> -- du -sh /workspace/storage/*
```

### Monitor Training Progress

```bash
# Follow loss values
kubectl logs -f <POD_NAME> | grep "loss"

# Check TensorBoard logs (if you have them mounted locally)
kubectl exec <POD_NAME> -- ls -lh /workspace/storage/logs/ape_v2v_diffusion/

# Copy logs locally for TensorBoard
kubectl cp <POD_NAME>:/workspace/storage/logs ./local_logs
tensorboard --logdir=./local_logs
```

---

## Training Configuration

The training uses: `/workspace/config/cloud_train_config.yaml`

**Key settings:**
```yaml
training:
  num_epochs: 2                    # Total training epochs
  batch_size: 1                    # Batch size per GPU
  learning_rate: 0.0001            # Base learning rate
  checkpoint_every: 500            # Save checkpoint every N steps
  checkpoint_dir: '/workspace/storage/checkpoints'  # Persistent storage

data:
  data_source: 'huggingface'       # Stream from HuggingFace
  dataset_name: 't2ance/APE-data'  # Dataset name
  streaming: true                  # No need to download entire dataset
  max_samples: 10                  # Limit samples for testing
  num_frames: 8                    # CT slices per sample
  resolution: [128, 128]           # Image resolution

pretrained:
  use_pretrained: false            # Training from scratch
  two_phase_training: true         # Use two-phase strategy
  phase1_epochs: 1                 # VAE frozen for first epoch
```

**To modify config in pod:**
```bash
kubectl exec -it <POD_NAME> -- /bin/bash
cd /workspace
vi config/cloud_train_config.yaml  # or nano
```

---

## Stopping and Restarting Training

### Stop Training

```bash
# For Job
kubectl delete job v2v-diffusion-training

# For Pod
kubectl delete pod v2v-diffusion-training-pod

# For Interactive Pod
kubectl delete pod v2v-diffusion-interactive
```

**Note:** Checkpoints in `/workspace/storage/` are preserved!

### Resume Training

```bash
# Option 1: Update config to resume from checkpoint
# Edit config/cloud_train_config.yaml:
# resume:
#   checkpoint_path: '/workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_epoch_1.pt'

# Then restart training
kubectl apply -f kub_files/training-job.yaml

# Option 2: Use --resume flag directly
kubectl exec <POD_NAME> -- python train.py \
  --config config/cloud_train_config.yaml \
  --resume /workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_epoch_1.pt
```

---

## Retrieving Results

### Copy Checkpoints to Local Machine

```bash
# Copy entire checkpoint directory
kubectl cp <POD_NAME>:/workspace/storage/checkpoints ./local_checkpoints

# Copy specific checkpoint
kubectl cp <POD_NAME>:/workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_final.pt ./checkpoint_final.pt

# Copy with compression (for large files)
kubectl exec <POD_NAME> -- tar czf /tmp/checkpoints.tar.gz /workspace/storage/checkpoints
kubectl cp <POD_NAME>:/tmp/checkpoints.tar.gz ./checkpoints.tar.gz
tar xzf checkpoints.tar.gz
```

### Copy Logs

```bash
# Copy TensorBoard logs
kubectl cp <POD_NAME>:/workspace/storage/logs ./local_logs

# Copy training log file
kubectl cp <POD_NAME>:/workspace/storage/logs/ape_v2v_diffusion/train.log ./train.log
```

---

## Troubleshooting

### Problem: Pod Stuck in "Pending"

**Check:**
```bash
kubectl describe pod <POD_NAME>

# Common causes:
# 1. No GPU nodes available
kubectl get nodes -l nvidia.com/gpu.product=Tesla-V100-SXM2-32GB

# 2. PVC not bound
kubectl get pvc v2v-diffuser-kuntal

# 3. Insufficient resources
kubectl describe nodes | grep -A 5 "Allocated resources"
```

**Fix:**
```bash
# If PVC issue:
kubectl delete pvc v2v-diffuser-kuntal
kubectl apply -f kub_files/persistent_storage.yaml

# If no GPU nodes, adjust node selector in YAML or wait for GPU node
```

### Problem: Pod Crashing / CrashLoopBackOff

**Check logs:**
```bash
kubectl logs <POD_NAME>
kubectl logs <POD_NAME> --previous  # Logs from previous crash
```

**Common causes:**
- Out of memory: Reduce batch_size in config
- CUDA error: Check GPU allocation
- Import error: Docker image issue

### Problem: Training Very Slow

**Check:**
```bash
# 1. Verify GPU is being used
kubectl exec <POD_NAME> -- nvidia-smi

# 2. Check GPU utilization
kubectl exec <POD_NAME> -- nvidia-smi dmon -c 10

# 3. Check data loading
kubectl logs <POD_NAME> | grep "batch"
```

**Fix:**
- Increase `num_workers` in config
- Enable mixed precision: `mixed_precision: true`
- Increase batch size if GPU memory allows

### Problem: Out of Disk Space

**Check storage:**
```bash
kubectl exec <POD_NAME> -- df -h /workspace/storage
```

**Fix:**
```bash
# Delete old checkpoints
kubectl exec <POD_NAME> -- rm /workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_step_*.pt

# Or keep only last N checkpoints (set in config):
# keep_last_n_checkpoints: 3

# Resize PVC if needed:
kubectl edit pvc v2v-diffuser-kuntal
# Change: storage: 20Gi -> storage: 50Gi
```

### Problem: Training Hanging

**Check:**
```bash
# 1. Check if process is running
kubectl exec <POD_NAME> -- ps aux | grep python

# 2. Check GPU isn't deadlocked
kubectl exec <POD_NAME> -- nvidia-smi

# 3. Check logs
kubectl logs <POD_NAME> --tail=50
```

**Fix:**
```bash
# Restart pod
kubectl delete pod <POD_NAME>
kubectl apply -f kub_files/<YOUR_POD_CONFIG>.yaml
```

---

## Advanced Usage

### Run with Different Config

```bash
# Copy custom config to pod
kubectl cp ./my_config.yaml <POD_NAME>:/workspace/config/

# Run with custom config
kubectl exec <POD_NAME> -- python train.py --config config/my_config.yaml
```

### Debug Inside Pod

```bash
# Start shell
kubectl exec -it <POD_NAME> -- /bin/bash

# Inside pod, run Python interactively
python
>>> from models import VideoToVideoDiffusion
>>> import torch
>>> # Test code here
```

### Port Forward for TensorBoard

```bash
# If running TensorBoard in pod:
kubectl exec <POD_NAME> -- tensorboard --logdir=/workspace/storage/logs --host=0.0.0.0 --port=6006 &

# Forward port to local machine
kubectl port-forward <POD_NAME> 6006:6006

# Open browser: http://localhost:6006
```

---

## Summary of Key Commands

```bash
# Setup (one-time)
kubectl apply -f kub_files/persistent_storage.yaml

# Start training (choose one)
kubectl apply -f kub_files/training-job.yaml          # Batch job
kubectl apply -f kub_files/training-pod.yaml          # Simple pod
kubectl apply -f kub_files/interactive-pod.yaml       # Interactive

# Monitor
kubectl get pods                                       # List pods
kubectl logs -f <POD_NAME>                            # Watch logs
kubectl exec <POD_NAME> -- nvidia-smi                 # Check GPU

# Check checkpoints
kubectl exec <POD_NAME> -- ls -lh /workspace/storage/checkpoints/ape_v2v_diffusion/

# Copy results
kubectl cp <POD_NAME>:/workspace/storage/checkpoints ./local_checkpoints

# Clean up
kubectl delete job v2v-diffusion-training             # Delete job
kubectl delete pod <POD_NAME>                          # Delete pod
```

---

## Training Timeline Estimates

**With Current Config (2 epochs, 10 samples, 128x128, batch_size=1):**
- V100 GPU: ~10-30 minutes (testing)

**Full Dataset Training:**
- Dataset: 206 patients
- Epochs: 50-100
- V100 GPU: 2-5 days
- A100 GPU: 1-3 days

**To speed up:**
1. Increase batch_size (if GPU memory allows)
2. Enable mixed precision: `mixed_precision: true`
3. Use gradient accumulation: `gradient_accumulation_steps: 8`
4. Use multiple GPUs (set `distributed: true`)

---

## Next Steps

1. **Verify setup works:**
   ```bash
   kubectl apply -f kub_files/interactive-pod.yaml
   kubectl exec -it v2v-diffusion-interactive -- python train.py --config config/cloud_train_config.yaml
   ```

2. **Monitor first few batches** to ensure:
   - Data loads correctly
   - GPU is being utilized
   - Loss decreases
   - Checkpoints are saved

3. **Scale up** once verified:
   - Increase `num_epochs`
   - Remove `max_samples` limit
   - Use training job for automation

4. **Monitor and iterate:**
   - Check TensorBoard
   - Adjust learning rate if needed
   - Resume from checkpoints

---

For issues or questions, check:
- **Checkpoint Storage Guide**: `CHECKPOINT_STORAGE_GUIDE.md`
- **Logs**: `kubectl logs -f <POD_NAME>`
- **Pod events**: `kubectl describe pod <POD_NAME>`

Happy Training! 🚀
