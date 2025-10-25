# Quick Start - Video-to-Video Diffusion Training

## 🚀 Fast Track (3 Commands)

```bash
# 1. Create storage
kubectl apply -f kub_files/persistent_storage.yaml

# 2. Start interactive pod
kubectl apply -f kub_files/interactive-pod.yaml && \
kubectl wait --for=condition=ready pod/v2v-diffusion-interactive --timeout=5m

# 3. Run training
kubectl exec -it v2v-diffusion-interactive -- python train.py --config config/cloud_train_config.yaml
```

## 📊 Monitor Training

```bash
# Get pod name
POD_NAME=$(kubectl get pods -l app=v2v-diffusion -o jsonpath='{.items[0].metadata.name}')

# Watch logs
kubectl logs -f $POD_NAME

# Check GPU
kubectl exec $POD_NAME -- nvidia-smi

# Check checkpoints
kubectl exec $POD_NAME -- ls -lh /workspace/storage/checkpoints/ape_v2v_diffusion/
```

## 💾 Get Results

```bash
# Copy checkpoints
kubectl cp $POD_NAME:/workspace/storage/checkpoints ./local_checkpoints

# Copy logs
kubectl cp $POD_NAME:/workspace/storage/logs ./local_logs
```

## 🔧 Configuration

Training config: `config/cloud_train_config.yaml`

**Key settings:**
- Epochs: 2 (testing)
- Batch size: 1
- Resolution: 128x128
- Samples: 10 (limited for testing)
- Checkpoints: `/workspace/storage/checkpoints/`

## ⚙️ Two-Phase Training Strategy

**Phase 1** (Epoch 0): Train U-Net only, VAE frozen
**Phase 2** (Epoch 1): Fine-tune entire model

This speeds up convergence and improves quality.

## 📝 Test Results

✅ **test_two_phase.py**: All 4 tests passed
- VAE freeze/unfreeze works
- Phase transition works correctly
- Training completes successfully

✅ **test_ape_data_loading.py**: Model integration passed
- Forward pass works
- Inference generation works
- Output shapes correct

## ⚠️ Current Setup

- **Pretrained VAE**: Disabled (architecture mismatch)
- **Training**: From scratch
- **Storage**: Persistent (survives pod restarts)
- **GPU**: Tesla V100 32GB required

## 🐛 Common Issues

**Pod pending?**
```bash
kubectl describe pod $POD_NAME | grep Events
```

**Out of memory?**
- Reduce batch_size in config
- Enable gradient accumulation

**Training slow?**
- Check GPU usage: `kubectl exec $POD_NAME -- nvidia-smi`
- Enable mixed precision in config

## 📚 Full Documentation

- **Complete Guide**: `RUN_TRAINING_GUIDE.md`
- **Checkpoint Storage**: `CHECKPOINT_STORAGE_GUIDE.md`

## 🎯 Next Steps

1. Verify tests pass locally: `python test_two_phase.py`
2. Deploy to Kubernetes (commands above)
3. Monitor first few epochs
4. Scale up:
   - Increase `num_epochs` to 50-100
   - Remove `max_samples` limit
   - Full dataset training

---

**Ready to train?** Run the 3 commands at the top! 🚀
