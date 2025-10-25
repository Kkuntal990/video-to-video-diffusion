# Production Training Guide - 256×256 Resolution on V100 32GB

This guide covers running the **maximum quality** configuration (256×256, 16 frames) on Tesla V100 32GB with critical monitoring procedures.

---

## ⚠️ Configuration Summary

### Current Setup (Maximum Quality)

```yaml
Resolution: 256×256 (4× more detail than 128×128)
Frames: 16 (full temporal context)
Batch size: 1 (required for memory)
Effective batch: 16 (via gradient accumulation)
Gradient checkpointing: ENABLED (critical!)
Training time: ~6-7 days
Memory usage: ~21-24 GB (peak ~28GB)
```

### Why This is Challenging

❗ **Memory Tight**: Using ~70-85% of V100 32GB capacity
❗ **Gradient Checkpointing Required**: Without it, would need 37GB
❗ **First 24 Hours Critical**: Memory patterns need monitoring
❗ **Phase Transition Risk**: Epoch 5→6 when VAE unfreezes

---

## 🚀 Starting Training

### Step 1: Deploy to Kubernetes

```bash
# 1. Ensure PVC exists
kubectl get pvc v2v-diffuser-kuntal
# Status should be "Bound"

# 2. Deploy interactive pod (recommended for monitoring)
kubectl apply -f kub_files/interactive-pod.yaml

# 3. Wait for pod to be ready
kubectl wait --for=condition=ready pod/v2v-diffusion-interactive --timeout=5m

# 4. Verify GPU is available
kubectl exec v2v-diffusion-interactive -- nvidia-smi

# Expected output:
# Tesla V100-SXM2-32GB
# Memory: 0 MiB / 32510 MiB
```

### Step 2: Start Training

```bash
# Exec into pod
kubectl exec -it v2v-diffusion-interactive -- /bin/bash

# Inside pod:
cd /workspace

# Verify config
cat config/cloud_train_config.yaml | grep -E "resolution|num_frames|batch_size|gradient_checkpointing"

# Expected:
# resolution: [256, 256]
# num_frames: 16
# batch_size: 1
# gradient_checkpointing: true

# Start training
python train.py --config config/cloud_train_config.yaml
```

---

## 📊 Critical Monitoring (First 24 Hours)

### Terminal Setup (Use 3 terminals)

**Terminal 1: GPU Memory Monitor**
```bash
# Real-time memory monitoring
kubectl exec v2v-diffusion-interactive -- nvidia-smi dmon -s mu -c 1000

# Output columns:
# gpu   pwr  temp    sm   mem   enc   dec  mclk  pclk
#   0    250   65    100    85     0     0  877  1530
#         ↑            ↑     ↑
#      Watts        GPU%  Memory%
```

**Terminal 2: Training Logs**
```bash
# Follow training logs
kubectl logs -f v2v-diffusion-interactive

# Watch for:
# "✓ Successfully loaded pretrained VAE weights" (should NOT appear - training from scratch)
# "🔒 VAE frozen" (at epoch 0)
# "Epoch 0 - Average loss: X.XXXX"
# "🔓 VAE unfrozen" (at epoch 5)
```

**Terminal 3: Memory Tracking**
```bash
# Track peak memory every 10 seconds
kubectl exec v2v-diffusion-interactive -- bash -c 'while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 10; done'

# Save to file for analysis:
kubectl exec v2v-diffusion-interactive -- bash -c 'while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 10; done' > memory_log.txt
```

---

## 🎯 What to Watch For

### ✅ GOOD SIGNS (Training is Stable)

**Step 1-10:**
```
Memory usage: 8-12 GB (model loading)
GPU utilization: 0-20% (data loading)
```

**Step 10-100:**
```
Memory usage: 20-24 GB (stabilized)
GPU utilization: 90-100% (good!)
Loss: Decreasing from ~1.5 to ~1.0
Step time: ~6-8 seconds per step
```

**After 100 steps:**
```
Memory: Stable at 21-24 GB
Loss: Continuing to decrease
No OOM errors in logs
```

### ⚠️ WARNING SIGNS (Action Needed)

**Memory > 28GB consistently:**
```bash
# Memory too high - prepare to stop and adjust config
echo "WARNING: Memory at 28+ GB, close to limit"

# If it persists for >10 steps, STOP training
kubectl exec v2v-diffusion-interactive -- pkill -f train.py
```

**GPU utilization < 70%:**
```bash
# Data loading bottleneck
# Increase num_workers in config (currently 4, try 8)
```

**Loss not decreasing after 50 steps:**
```bash
# Potential training issue
# Check logs for NaN values
# Verify learning rate is reasonable
```

### 🚨 CRITICAL ISSUES (Stop Immediately)

**"CUDA out of memory" in logs:**
```bash
# OOM occurred - training will crash
# Apply fallback configuration (see below)
```

**Memory > 31GB:**
```bash
# About to OOM - stop training NOW
kubectl exec v2v-diffusion-interactive -- pkill -f train.py
```

**Loss = NaN:**
```bash
# Training diverged - numerical instability
# Reduce learning rate or check for data issues
```

---

## 🔧 Fallback Procedures

### If OOM Occurs

**Fallback Level 1: Reduce Frames (Recommended)**

```bash
# Stop training
kubectl exec v2v-diffusion-interactive -- pkill -f train.py

# Edit config
kubectl exec v2v-diffusion-interactive -- bash -c "sed -i 's/num_frames: 16/num_frames: 12/' /workspace/config/cloud_train_config.yaml"

# Verify change
kubectl exec v2v-diffusion-interactive -- grep "num_frames" /workspace/config/cloud_train_config.yaml

# Restart training
kubectl exec v2v-diffusion-interactive -- bash -c "cd /workspace && python train.py --config config/cloud_train_config.yaml"
```

**Memory savings**: ~25% (reduces from ~24GB to ~18GB)

**Fallback Level 2: Reduce Resolution**

```bash
# Stop training
kubectl exec v2v-diffusion-interactive -- pkill -f train.py

# Edit config (keep 16 frames, reduce resolution)
kubectl exec v2v-diffusion-interactive -- bash -c "sed -i 's/resolution: \[256, 256\]/resolution: [224, 224]/' /workspace/config/cloud_train_config.yaml"

# Restart training
kubectl exec v2v-diffusion-interactive -- bash -c "cd /workspace && python train.py --config config/cloud_train_config.yaml"
```

**Memory savings**: ~30% (reduces from ~24GB to ~17GB)

**Fallback Level 3: Both Reductions**

```bash
# Stop training
kubectl exec v2v-diffusion-interactive -- pkill -f train.py

# Edit config (reduce both)
kubectl exec v2v-diffusion-interactive -- bash -c "sed -i 's/num_frames: 16/num_frames: 12/' /workspace/config/cloud_train_config.yaml"
kubectl exec v2v-diffusion-interactive -- bash -c "sed -i 's/resolution: \[256, 256\]/resolution: [224, 224]/' /workspace/config/cloud_train_config.yaml"

# Restart
kubectl exec v2v-diffusion-interactive -- bash -c "cd /workspace && python train.py --config config/cloud_train_config.yaml"
```

**Memory savings**: ~50% (reduces from ~24GB to ~12GB - very safe!)

---

## 📈 Phase Transition Monitoring

### Epoch 5 → Epoch 6 (Critical Moment)

**What happens:**
- VAE parameters unfreeze
- Optimizer states for VAE are initialized
- Brief memory spike possible

**How to monitor:**

```bash
# Before epoch 5 completes, prepare monitoring
# Watch memory closely during transition

# In logs, you'll see:
# "Epoch 4 - Average loss: X.XXXX"
# "Checkpoint saved to checkpoints/checkpoint_epoch_4.pt"
# "🔓 VAE unfrozen"
# "============================================================"
# "PHASE 2: Fine-tuning entire model (VAE unfrozen)"
# "============================================================"
# "Epoch 5:   0%|          | 0/206 [00:00<?, ?it/s]"

# Memory might spike +2-3GB briefly, then stabilize
```

**If memory spikes > 30GB during transition:**
- This is temporary, wait 5-10 steps
- If it doesn't drop back to ~24GB, stop and apply fallback

---

## 📊 Training Progress Metrics

### Expected Loss Trajectory

**Phase 1 (Epochs 0-5, VAE Frozen):**
```
Epoch 0: Loss ~1.5 (random init, U-Net learning)
Epoch 1: Loss ~1.2
Epoch 2: Loss ~0.9
Epoch 3: Loss ~0.7
Epoch 4: Loss ~0.6
Epoch 5: Loss ~0.5
```

**Phase 2 (Epochs 6-50, Full Fine-tuning):**
```
Epoch 6:  Loss ~0.48 (slight increase when VAE unfreezes)
Epoch 10: Loss ~0.35
Epoch 20: Loss ~0.25
Epoch 30: Loss ~0.18
Epoch 40: Loss ~0.15
Epoch 50: Loss ~0.12 (convergence)
```

**Red flags:**
- Loss not decreasing after 10 epochs
- Loss suddenly jumps > 2×
- Loss becomes NaN

### Checkpoint Verification

```bash
# Check checkpoints are being saved
kubectl exec v2v-diffusion-interactive -- ls -lh /workspace/storage/checkpoints/ape_v2v_diffusion/

# Should see (after a few hours):
# checkpoint_epoch_0.pt   (~670 MB)
# checkpoint_epoch_1.pt   (~670 MB)
# ...

# Verify checkpoint size is reasonable
# Each checkpoint should be ~650-700 MB
# If significantly different, might indicate corruption
```

---

## 🕐 Timeline Expectations

### Detailed Timeline

| Time | Event | Memory | Status |
|------|-------|--------|--------|
| 0:00 | Training starts | 8-12 GB | Model loading |
| 0:05 | First batch | 20-24 GB | Stabilizing |
| 0:30 | 50 steps | 21-24 GB | Should be stable |
| 1:00 | 100 steps | 21-24 GB | ✅ Safe to leave unattended |
| 6:00 | Epoch 1 complete | 21-24 GB | Save checkpoint |
| 30:00 | Epoch 5 complete (Phase 1 done) | 21-24 GB | VAE unfreezes |
| 30:10 | Epoch 6 starts (Phase 2) | 23-26 GB | Brief spike possible |
| 36:00 | Epoch 6 complete | 21-24 GB | Should stabilize again |
| 150:00 | ~Day 6-7, Epoch 50 | 21-24 GB | Training complete! |

### Milestones

✅ **First Hour**: If no OOM, configuration is working
✅ **First Epoch**: Validates data loading and training loop
✅ **Epoch 5→6 Transition**: Critical phase change survived
✅ **Epoch 10**: Can extrapolate final loss and quality
✅ **Epoch 50**: Training complete!

---

## 🎉 After Training Completes

### Step 1: Verify Final Checkpoint

```bash
# Check final checkpoint exists
kubectl exec v2v-diffusion-interactive -- ls -lh /workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_final.pt

# Should be ~670 MB
```

### Step 2: Copy Checkpoints Locally

```bash
# Copy all checkpoints
kubectl cp v2v-diffusion-interactive:/workspace/storage/checkpoints ./local_checkpoints

# Or just final checkpoint
kubectl cp v2v-diffusion-interactive:/workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_final.pt ./checkpoint_final_256x256.pt
```

### Step 3: Run Inference Test

```bash
# Inside pod, test inference
kubectl exec -it v2v-diffusion-interactive -- bash

cd /workspace

# Generate a test video
python inference.py \
  --checkpoint /workspace/storage/checkpoints/ape_v2v_diffusion/checkpoint_final.pt \
  --sampler ddim \
  --steps 50 \
  --num-frames 16 \
  --resolution 256 256
```

### Step 4: Quality Assessment

**Visual inspection:**
- Check for fine anatomical details
- Verify no artifacts or blurring
- Compare to input CT scans

**Quantitative metrics:**
- PSNR: Should be > 25 dB (good quality)
- SSIM: Should be > 0.85 (good structure preservation)

---

## 📞 Troubleshooting Common Issues

### Issue 1: Training Starts Then Immediately OOMs

**Symptoms:**
- Training begins, gets to step 1-5
- "CUDA out of memory" error
- Pod crashes or training stops

**Solution:**
```bash
# Gradient checkpointing might not be working
# Verify it's enabled in code

# Quick fix: Reduce to 12 frames
kubectl exec v2v-diffusion-interactive -- bash -c "sed -i 's/num_frames: 16/num_frames: 12/' /workspace/config/cloud_train_config.yaml"
```

### Issue 2: Very Slow Training (< 1 step/minute)

**Symptoms:**
- Each step takes > 60 seconds
- GPU utilization < 50%

**Solution:**
```bash
# Likely data loading bottleneck
# Increase num_workers

kubectl exec v2v-diffusion-interactive -- bash -c "sed -i 's/num_workers: 4/num_workers: 8/' /workspace/config/cloud_train_config.yaml"

# Restart training
```

### Issue 3: Loss Not Decreasing

**Symptoms:**
- Loss stays at ~1.5 for > 20 epochs
- Or loss increases over time

**Solution:**
```bash
# Check learning rate
# Might be too high or too low

# Reduce learning rate:
kubectl exec v2v-diffusion-interactive -- bash -c "sed -i 's/learning_rate: 0.0001/learning_rate: 0.00005/' /workspace/config/cloud_train_config.yaml"
```

### Issue 4: Checkpoints Not Saving

**Symptoms:**
- No checkpoint files in /workspace/storage/checkpoints/

**Solution:**
```bash
# Check PVC is mounted
kubectl exec v2v-diffusion-interactive -- df -h /workspace/storage

# Check permissions
kubectl exec v2v-diffusion-interactive -- ls -la /workspace/storage/

# Check disk space
kubectl exec v2v-diffusion-interactive -- df -h /workspace/storage
# Should have > 5GB free
```

---

## 📋 Monitoring Checklist

### First Hour
- [ ] Memory stable at 21-24 GB
- [ ] GPU utilization > 90%
- [ ] Loss decreasing
- [ ] No OOM errors
- [ ] Step time ~6-8 seconds

### First 24 Hours
- [ ] Completed at least 3 epochs
- [ ] Loss following expected trajectory
- [ ] Checkpoints saving correctly
- [ ] No memory spikes > 28GB
- [ ] Data loading keeping up with GPU

### Phase Transition (Epoch 5→6)
- [ ] "🔓 VAE unfrozen" message appears
- [ ] Memory spike handled (if any)
- [ ] Training continues smoothly
- [ ] Loss doesn't jump dramatically

### Mid-Training (Epoch 25)
- [ ] Loss continuing to decrease
- [ ] ~2GB disk space for remaining checkpoints
- [ ] No anomalies in logs
- [ ] Still have 3+ days of training left

### Final Days (Epoch 45+)
- [ ] Loss converging (< 0.15)
- [ ] Checkpoints all present
- [ ] Preparing for inference testing

---

## 🎯 Success Criteria

After 50 epochs, you should have:

✅ **Training metrics:**
- Final loss: < 0.15
- Loss curve: Smooth decrease
- No NaN or infinity values
- All 50 epochs completed

✅ **Checkpoints:**
- checkpoint_final.pt (670 MB)
- Last 3 epoch checkpoints
- All accessible in /workspace/storage/

✅ **Quality:**
- Generated videos look sharp
- Fine anatomical details visible
- No obvious artifacts
- Better than 128×128 baseline

---

## 📚 Additional Resources

- **Main README**: [README.md](README.md)
- **Architecture Details**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Checkpoint Management**: [CHECKPOINT_STORAGE_GUIDE.md](CHECKPOINT_STORAGE_GUIDE.md)
- **General Training**: [RUN_TRAINING_GUIDE.md](RUN_TRAINING_GUIDE.md)

---

## 🆘 Emergency Contacts

If training fails catastrophically:

1. **Save logs**: `kubectl logs v2v-diffusion-interactive > emergency_log.txt`
2. **Check last checkpoint**: May be able to resume from epoch N-1
3. **Review this guide**: Most issues covered above
4. **Check pod events**: `kubectl describe pod v2v-diffusion-interactive`

---

**Good luck with training!** 🚀

With proper monitoring in the first 24 hours, this configuration should successfully complete in ~6-7 days, producing high-quality 256×256 CT reconstruction results.
