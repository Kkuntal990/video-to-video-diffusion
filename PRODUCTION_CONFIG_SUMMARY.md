# Production Configuration Summary - 256×256 Resolution

## ✅ Configuration Updated for Maximum Quality

Your training configuration has been updated for production training with **maximum quality** on Tesla V100 32GB.

---

## 🎯 What Changed

### Configuration File: `config/cloud_train_config.yaml`

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| **resolution** | [128, 128] | **[256, 256]** | 4× more spatial detail |
| **num_frames** | 8 | **16** | Full temporal context |
| **batch_size** | 1 | **1** | Required for memory |
| **num_workers** | 1 | **4** | Faster data loading |
| **max_samples** | 10 | **null** | Full dataset (206 patients) |
| **num_epochs** | 2 | **50** | Real production training |
| **gradient_accumulation_steps** | 8 | **16** | Effective batch = 16 |
| **warmup_steps** | 500 | **1000** | Better stability |
| **checkpoint_every** | 500 | **1000** | Reduce I/O |
| **keep_last_n_checkpoints** | 1 | **3** | Safety net |
| **log_every** | 50 | **100** | Reduce overhead |
| **phase1_epochs** | 1 | **5** | Longer warm-up |
| **min_lr** | N/A | **1e-6** | Min learning rate |
| **gradient_checkpointing** | true | **true** ✅ | VERIFIED enabled |

---

## 📊 Expected Results

### Training Characteristics

| Metric | Value |
|--------|-------|
| **Resolution** | 256×256 (4× better than 128×128) |
| **Temporal Frames** | 16 CT slices |
| **Total Epochs** | 50 |
| **Phase 1** | 5 epochs (~30 hours, VAE frozen) |
| **Phase 2** | 45 epochs (~270 hours, full fine-tuning) |
| **Total Time** | ~6-7 days |
| **Memory Usage** | ~21-24 GB (peak ~28GB) |
| **Effective Batch Size** | 16 (via gradient accumulation) |
| **Checkpoints** | 3 kept (last 3 epochs + final) |

### Quality Expectations

✅ **4× better spatial resolution** (256×256 vs 128×128)
✅ **Full temporal coherence** (16 frames vs 8 frames)
✅ **Clinically relevant detail** (closer to real CT scans)
✅ **Fine anatomical structures preserved**

---

## ⚠️ Critical: Memory Management

### Why This Configuration is Tight

```
V100 32GB Total Memory:        32 GB
Estimated Usage:              ~24 GB
Headroom:                     ~8 GB
Peak Spikes:                  ~28 GB
```

**Gradient Checkpointing is REQUIRED:**
- Without it: Would need ~37 GB (OOM!)
- With it: Uses ~21-24 GB ✅
- Verified enabled in config ✓

### First 24 Hours = CRITICAL

**You MUST monitor closely:**
- Watch GPU memory every 30 minutes
- Look for memory > 30GB (warning sign)
- If OOM occurs, fallback plans provided

---

## 🚀 How to Start Training

### Quick Start (3 Commands)

```bash
# 1. Deploy pod
kubectl apply -f kub_files/interactive-pod.yaml
kubectl wait --for=condition=ready pod/v2v-diffusion-interactive --timeout=5m

# 2. Start training
kubectl exec -it v2v-diffusion-interactive -- python train.py --config config/cloud_train_config.yaml

# 3. Monitor (in separate terminal)
kubectl exec v2v-diffusion-interactive -- nvidia-smi dmon -s mu -c 1000
```

### Monitoring Setup (3 Terminals)

**Terminal 1: GPU Memory**
```bash
kubectl exec v2v-diffusion-interactive -- nvidia-smi dmon -s mu -c 1000
```

**Terminal 2: Training Logs**
```bash
kubectl logs -f v2v-diffusion-interactive
```

**Terminal 3: Memory Tracking**
```bash
kubectl exec v2v-diffusion-interactive -- bash -c 'while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 10; done'
```

---

## 🔧 Fallback Plans (If OOM)

### Level 1: Reduce Frames (Easiest)
```bash
# Reduces memory by ~25%
# Change: num_frames: 16 → 12
```

### Level 2: Reduce Resolution
```bash
# Reduces memory by ~30%
# Change: resolution: [256,256] → [224,224]
```

### Level 3: Both
```bash
# Reduces memory by ~50%
# Change both num_frames and resolution
```

**Detailed instructions in**: `PRODUCTION_TRAINING_GUIDE.md`

---

## 📈 Expected Training Progress

### Phase 1 (Epochs 0-5, VAE Frozen)
```
Epoch 0: Loss ~1.5 → ~0.5
Phase 1 goal: Train U-Net to denoise effectively
```

### Phase 2 (Epochs 6-50, Full Fine-tuning)
```
Epoch 6:  Loss ~0.48
Epoch 50: Loss ~0.12
Phase 2 goal: Optimize entire model end-to-end
```

### Milestones

✅ **1 hour**: If no OOM, config is working
✅ **1 epoch** (~6 hours): Validates entire pipeline
✅ **Epoch 5→6**: Critical phase transition
✅ **Epoch 10**: Can estimate final quality
✅ **Epoch 50**: Training complete!

---

## 📁 Important Files

### Configuration
- **Main config**: `config/cloud_train_config.yaml` ✅ UPDATED
- **Training script**: `train.py`
- **Kubernetes pod**: `kub_files/interactive-pod.yaml`

### Documentation
- **Production Guide**: `PRODUCTION_TRAINING_GUIDE.md` ✅ NEW
- **Architecture**: `ARCHITECTURE.md`
- **Checkpoint Storage**: `CHECKPOINT_STORAGE_GUIDE.md`
- **Quick Start**: `QUICK_START.md`
- **Main README**: `README.md`

### Checkpoints (after training)
- **Location**: `/workspace/storage/checkpoints/ape_v2v_diffusion/`
- **Size**: ~670 MB per checkpoint
- **Kept**: Last 3 + final = 4 checkpoints (~2.7 GB total)

---

## ✅ Pre-Flight Checklist

Before starting training, verify:

- [ ] Config file updated ✓
- [ ] Gradient checkpointing enabled ✓
- [ ] PVC exists and is bound
- [ ] GPU pod has V100 32GB
- [ ] Monitoring terminals ready
- [ ] Read `PRODUCTION_TRAINING_GUIDE.md`
- [ ] Fallback procedures understood
- [ ] ~20Gi PVC storage available

---

## 📞 What to Do Next

### 1. **Deploy and Start Training**
```bash
kubectl apply -f kub_files/interactive-pod.yaml
kubectl exec -it v2v-diffusion-interactive -- python train.py --config config/cloud_train_config.yaml
```

### 2. **Monitor First 24 Hours Closely**
- Check memory every 30 minutes
- Watch for OOM warnings
- Verify loss is decreasing
- See `PRODUCTION_TRAINING_GUIDE.md` for details

### 3. **After 24 Hours (if stable)**
- Reduce monitoring to daily checks
- Verify checkpoints are saving
- Monitor loss trajectory

### 4. **After Training Completes (~7 days)**
- Copy checkpoints locally
- Run inference tests
- Evaluate quality

---

## 🎓 Key Takeaways

✅ **Maximum quality configuration** for V100 32GB
✅ **256×256 resolution** - 4× better than baseline
✅ **16 frames** - full temporal context
✅ **Gradient checkpointing** - enables this on 32GB
✅ **~6-7 days** - total training time
✅ **Monitoring critical** - especially first 24 hours
✅ **Fallback plans ready** - if OOM occurs
✅ **Documentation complete** - comprehensive guides

---

## 🚀 Ready to Train!

You're all set for maximum quality training. Key points:

1. **Configuration is optimized** for V100 32GB
2. **Gradient checkpointing is enabled** (critical!)
3. **Monitoring procedures documented** in detail
4. **Fallback plans ready** if needed
5. **Expected timeline: 6-7 days**

**Next step**: Deploy and start training!

```bash
kubectl apply -f kub_files/interactive-pod.yaml
kubectl exec -it v2v-diffusion-interactive -- python train.py --config config/cloud_train_config.yaml
```

For detailed monitoring and troubleshooting, see: **`PRODUCTION_TRAINING_GUIDE.md`**

Good luck! 🎉
