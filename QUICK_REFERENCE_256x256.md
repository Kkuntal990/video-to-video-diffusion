# Quick Reference: 256×256 Production Training

## 🚀 Start Training (Copy-Paste Ready)

```bash
# Deploy
kubectl apply -f kub_files/interactive-pod.yaml && kubectl wait --for=condition=ready pod/v2v-diffusion-interactive --timeout=5m

# Start Training
kubectl exec -it v2v-diffusion-interactive -- python train.py --config config/cloud_train_config.yaml
```

## 📊 Monitor (3 Terminals)

**Terminal 1: GPU Memory**
```bash
kubectl exec v2v-diffusion-interactive -- nvidia-smi dmon -s mu -c 1000
```

**Terminal 2: Logs**
```bash
kubectl logs -f v2v-diffusion-interactive
```

**Terminal 3: Memory Values**
```bash
kubectl exec v2v-diffusion-interactive -- bash -c 'while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 10; done'
```

## ⚠️ Watch For

✅ **GOOD**: Memory 21-24 GB, GPU 90-100%, Loss decreasing
⚠️ **WARNING**: Memory > 28GB, GPU < 70%, Loss flat
🚨 **CRITICAL**: Memory > 31GB, OOM error, Loss = NaN

## 🔧 Quick Fixes

**If OOM:**
```bash
# Reduce frames 16→12
kubectl exec v2v-diffusion-interactive -- sed -i 's/num_frames: 16/num_frames: 12/' /workspace/config/cloud_train_config.yaml
```

**If slow:**
```bash
# Increase workers 4→8
kubectl exec v2v-diffusion-interactive -- sed -i 's/num_workers: 4/num_workers: 8/' /workspace/config/cloud_train_config.yaml
```

## 📋 Config Summary

- Resolution: **256×256** (4× detail)
- Frames: **16** (temporal context)
- Epochs: **50** (~6-7 days)
- Memory: **~24GB** (V100 32GB)
- Batch: **1** (effective=16 via grad accum)
- Checkpointing: **ON** (required!)

## 📁 Files

- Config: `config/cloud_train_config.yaml`
- Full Guide: `PRODUCTION_TRAINING_GUIDE.md`
- Summary: `PRODUCTION_CONFIG_SUMMARY.md`

## 🎯 Timeline

- **1 hour**: First stability check
- **6 hours**: First epoch complete
- **30 hours**: Phase 1 done (VAE unfreezes)
- **6-7 days**: Training complete!

## 📞 Checkpoints

```bash
# List checkpoints
kubectl exec v2v-diffusion-interactive -- ls -lh /workspace/storage/checkpoints/ape_v2v_diffusion/

# Copy to local
kubectl cp v2v-diffusion-interactive:/workspace/storage/checkpoints ./local_checkpoints
```

## ✅ Success Criteria

After 50 epochs:
- Loss < 0.15
- Memory stable 21-24GB
- All checkpoints saved
- No OOM errors

---

**For details**: See `PRODUCTION_TRAINING_GUIDE.md`
