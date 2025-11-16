# Deployment Guide - CT Slice Interpolation

**Status**: ✅ Ready to Deploy
**Date**: 2025-11-07

---

## Quick Start

```bash
# 1. Deploy dataset download job
kubectl apply -f kub_files/download-dataset-job.yaml

# 2. Monitor download (1-2 hours)
kubectl logs -f $(kubectl get pods -l job-name=ape-dataset-download-job -o jsonpath='{.items[0].metadata.name}')

# 3. Verify download
kubectl exec copy-pod -- sh -c "ls /workspace/storage_a100/dataset/APE/ | wc -l"
kubectl exec copy-pod -- sh -c "ls /workspace/storage_a100/dataset/'non-APE'/ | wc -l"

# 4. Deploy training job
kubectl apply -f kub_files/training-job-a100.yaml

# 5. Monitor training
kubectl logs -f $(kubectl get pods -l job-name=v2v-diffusion-training-job-a100 -o jsonpath='{.items[0].metadata.name}')
```

**Total time**: ~10-12 hours (download + training)

---

## Prerequisites

### Required

- ✅ PVC `v2v-diffuser-kuntal-a100` with 300GB storage
- ✅ Copy-pod running for verification
- ✅ Docker image `kuntalkokate/llm_agent_v2v_train:latest`
- ✅ Training config [config/slice_interpolation_full_medium.yaml](config/slice_interpolation_full_medium.yaml)

### Optional (Better Performance)

- ⚠️ MAISI VAE checkpoint at `/workspace/storage_a100/pretrained/maisi_vae/models/autoencoder.pt`
  - If missing: VAE trains from scratch (still works, just slower convergence)

---

## Step-by-Step Deployment

### Step 1: Deploy Dataset Download Job

Download the APE dataset from HuggingFace and save .zip files to PVC.

```bash
cd /Users/kuntalkokate/Desktop/LLM_agents_projects/LLM_agent_v2v

# Deploy download job
kubectl apply -f kub_files/download-dataset-job.yaml
```

**Expected output**:
```
job.batch/ape-dataset-download-job created
configmap/dataset-download-info created
```

**What it does**:
- Downloads 475 cases from HuggingFace dataset `t2ance/APE-data`
- Filters out 25 failed cases (no valid data)
- Saves ~356 .zip files organized by category (APE/non-APE)
- Total size: ~50-55 GB

---

### Step 2: Monitor Download Progress

```bash
# Get pod name
POD_NAME=$(kubectl get pods -l job-name=ape-dataset-download-job -o jsonpath='{.items[0].metadata.name}')

# Monitor logs in real-time
kubectl logs -f $POD_NAME
```

**Expected logs**:
```
2025-11-07 09:00:00 - INFO - Output directory: /workspace/storage_a100/dataset
2025-11-07 09:00:00 - INFO - Dataset: t2ance/APE-data (split: train)
2025-11-07 09:00:15 - INFO - Total cases in dataset: 475
2025-11-07 09:00:15 - INFO - Starting download...
======================================================================
Downloading: 0%|          | 0/475 [00:00<?, ?it/s]
  [1] case_001 → APE/ (118.6 MB)
  [2] case_002 → APE/ (117.7 MB)
  ...
```

**Duration**: ~1-2 hours (network dependent)

**Progress tracking**:
- Download progress saved automatically
- Can safely stop and resume
- Failed downloads logged for retry
- Job auto-retries up to 3 times

---

### Step 3: Verify Dataset

After download completes, verify the dataset:

```bash
# Check job status
kubectl get job ape-dataset-download-job
# Should show: COMPLETIONS=1/1

# Count files by category
echo "APE cases:"
kubectl exec copy-pod -- sh -c "ls /workspace/storage_a100/dataset/APE/ | wc -l"

echo "non-APE cases:"
kubectl exec copy-pod -- sh -c "ls /workspace/storage_a100/dataset/'non-APE'/ | wc -l"

# Check total storage
kubectl exec copy-pod -- du -sh /workspace/storage_a100/dataset/
```

**Expected results**:
```
APE cases: 189
non-APE cases: 167
Total: 356 files
Storage: ~50-55 GB
```

**Note**: HuggingFace has 475 cases, but ~25 fail preprocessing (no valid data), leaving 356 usable cases.

---

### Step 4: Verify MAISI Checkpoint (Optional)

Check if MAISI VAE pretrained weights are available:

```bash
# Check checkpoint existence
kubectl exec copy-pod -- ls -lh /workspace/storage_a100/pretrained/maisi_vae/models/autoencoder.pt
```

**If checkpoint missing**:

**Option A**: Upload from local machine (if available):
```bash
kubectl cp /path/to/autoencoder.pt \
  copy-pod:/workspace/storage_a100/pretrained/maisi_vae/models/autoencoder.pt
```

**Option B**: Skip pretrained VAE (still works):
```yaml
# Edit config/slice_interpolation_full_medium.yaml:
pretrained:
  use_pretrained: false  # ← Change to false
```

**Recommendation**: Training works without MAISI (VAE trains from scratch). Performance is slightly lower initially but still converges well.

---

### Step 5: Deploy Training Job

Deploy the training job with the new dataset:

```bash
# Verify config points to correct dataset
cat config/slice_interpolation_full_medium.yaml | grep dataset_path
# Should show: dataset_path: '/workspace/storage_a100/dataset'

# Delete any previous failed jobs
kubectl delete job v2v-diffusion-training-job-a100 2>/dev/null || true

# Deploy training job
kubectl apply -f kub_files/training-job-a100.yaml
```

**Expected output**:
```
job.batch/v2v-diffusion-training-job-a100 created
```

---

### Step 6: Monitor Training

```bash
# Get training pod name
TRAIN_POD=$(kubectl get pods -l job-name=v2v-diffusion-training-job-a100 -o jsonpath='{.items[0].metadata.name}')

# Monitor logs
kubectl logs -f $TRAIN_POD
```

**Expected logs** (successful start):
```
2025-11-07 10:00:00 - training - INFO - Starting training with config: slice_interpolation_full_medium
2025-11-07 10:00:00 - training - INFO - ========================================

Using data source: slice_interpolation
Loading CT Slice Interpolation data from: /workspace/storage_a100/dataset
Task: Thick slices (50 @ 5.0mm) → Thin slices (300 @ 1.0mm)
Mode: Full volumes (NO patches, NO downsampling)

Scanning dataset...
  Found 189 patients in APE
  Found 167 patients in non-APE

✓ Train set: 267 patients  # ← Should NOT be 0!
  Categories: ['APE', 'non-APE']
  Resolution: (512, 512)
  Slice limits: thick=50, thin=300

✓ Val set: 53 patients
✓ Test set: 36 patients

Creating model...
  U-Net: 598.9M parameters
  VAE: 130.0M parameters (MAISI pretrained)  # OR 43.1M (trained from scratch)
  Total: 729.0M parameters

Starting training...
Epoch 1/100, Step 0: Loss=0.8234
Epoch 1/100, Step 50: Loss=0.6123
...
```

---

## Expected Timeline

### Download Phase
- **Start**: T+0
- **Duration**: 1-2 hours
- **End**: Dataset ready (~356 cases, 50 GB)

### Training Phase
- **Start**: T+2 hours
- **Epoch duration**: ~5-7 minutes (356 patients, batch_size=2)
- **Total epochs**: 100 (early stopping around 50-70)
- **Total training time**: ~8-10 hours
- **End**: Model converged (PSNR ~44-46 dB)

### Total
- **~10-12 hours** from start to trained model

---

## Monitoring Training

### Check Metrics

```bash
# View recent logs
kubectl logs --tail=100 $TRAIN_POD

# Check GPU utilization
kubectl exec $TRAIN_POD -- nvidia-smi

# Check current epoch
kubectl logs $TRAIN_POD | grep "Epoch" | tail -5
```

### View Validation Samples

```bash
# List samples
kubectl exec copy-pod -- ls -lh /workspace/storage_a100/outputs/slice_interp_full_medium/samples/

# Copy sample to local
kubectl cp copy-pod:/workspace/storage_a100/outputs/slice_interp_full_medium/samples/epoch_010_sample_0.png ./validation_sample.png
```

### Check Checkpoints

```bash
# List checkpoints
kubectl exec copy-pod -- ls -lh /workspace/storage_a100/checkpoints/slice_interp_full_medium/

# Latest checkpoint
kubectl exec copy-pod -- ls -lt /workspace/storage_a100/checkpoints/slice_interp_full_medium/ | head -3
```

---

## Troubleshooting

### Issue: Dataset Not Found

**Symptoms**:
```
Warning: Category directory not found: /workspace/storage_a100/dataset/APE
✓ Train set: 0 patients
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

**Solution 1**: Check dataset location
```bash
# Run helper script
bash scripts/check_pvc_dataset.sh

# If dataset exists at different path, update config
nano config/slice_interpolation_full_medium.yaml
# Change dataset_path to actual location
```

**Solution 2**: Re-run download job
```bash
# Delete failed job
kubectl delete job ape-dataset-download-job

# Re-deploy
kubectl apply -f kub_files/download-dataset-job.yaml
```

---

### Issue: Download Job Fails

**Check status**:
```bash
# Job status
kubectl describe job ape-dataset-download-job

# Pod status
kubectl describe pod $POD_NAME
```

**Common causes**:

1. **Network timeout** → Job auto-retries (wait)
2. **Disk full** → Check PVC storage:
   ```bash
   kubectl describe pvc v2v-diffuser-kuntal-a100
   ```
3. **Image pull error** → Verify Docker image exists

---

### Issue: Training Job Fails

**Check logs for errors**:
```bash
kubectl logs $TRAIN_POD | grep -A 10 "ERROR"
```

**Common causes**:

1. **Dataset not found** → See "Dataset Not Found" above
2. **MAISI checkpoint missing** → See Step 4
3. **OOM error** → Reduce batch_size in config:
   ```yaml
   training:
     batch_size: 1  # Reduce from 2
   ```
4. **CUDA error** → Check GPU availability:
   ```bash
   kubectl exec $TRAIN_POD -- nvidia-smi
   ```

---

### Issue: Slow Download

If download < 1 MB/s:

**Solutions**:
- Run overnight (network bottleneck)
- Wait for HuggingFace rate limit to reset
- Use streaming mode (no pre-download):
  ```yaml
  data:
    streaming: true  # Download on-the-fly during training
  ```

---

### Issue: Old Cached Data

**Problem**: Previous cached data incompatible with slice interpolation

**Why**:
- ❌ Downsampled to 24 frames (need 50→300 slices)
- ❌ Downsampled to 256×256 (need 512×512)
- ❌ No thick/thin distinction
- ❌ Cannot recover original structure

**Solution**: Re-download (as instructed above)

**Storage cleanup** (optional, after successful training):
```bash
# Free up 62 GB of old cache
kubectl exec copy-pod -- rm -rf /workspace/storage_a100/ape_cache/processed/
kubectl exec copy-pod -- rm -rf /workspace/storage_a100/ape_cache/raw/

# Keep metadata for reference
kubectl exec copy-pod -- cp /workspace/storage_a100/ape_cache/metadata.json \
  /workspace/storage_a100/dataset/
```

---

## Verification Checklist

### Before Training

- [ ] Download job completed successfully
- [ ] Dataset verified: ~356 .zip files in APE/ and non-APE/
- [ ] MAISI checkpoint verified (or config updated to skip)
- [ ] Training config reviewed
- [ ] PVC has >100 GB free space for checkpoints

### After Training Starts

- [ ] Training job deployed
- [ ] Logs show correct dataset loading (267 train, 53 val, 36 test)
- [ ] First epoch completes without errors
- [ ] GPU memory stable (~28-33 GB / 80 GB)

---

## Cleanup

After successful training and checkpoint verification:

```bash
# Delete download job
kubectl delete job ape-dataset-download-job

# Delete old cache (frees 62 GB)
kubectl exec copy-pod -- rm -rf /workspace/storage_a100/ape_cache/
```

---

## Next Steps

After training completes:

1. **Evaluate model** on test set:
   ```bash
   python scripts/evaluate_test_set.py \
     --checkpoint /workspace/storage_a100/checkpoints/best.pth
   ```

2. **Visualize predictions** vs ground truth

3. **Compute metrics**: PSNR, SSIM, MSE

4. **Tune hyperparameters** if needed (learning rate, model size)

5. **Deploy for inference** on new CT scans

---

## Key Files

### Deployment
- [kub_files/download-dataset-job.yaml](kub_files/download-dataset-job.yaml) - Dataset download job
- [kub_files/training-job-a100.yaml](kub_files/training-job-a100.yaml) - Training job
- [scripts/download_ape_dataset.py](scripts/download_ape_dataset.py) - Download script
- [scripts/check_pvc_dataset.sh](scripts/check_pvc_dataset.sh) - Verification helper

### Configuration
- [config/slice_interpolation_full_medium.yaml](config/slice_interpolation_full_medium.yaml) - Training config

### Documentation
- [SLICE_INTERPOLATION_GUIDE.md](SLICE_INTERPOLATION_GUIDE.md) - Complete implementation guide
- [MAISI_VAE_GUIDE.md](MAISI_VAE_GUIDE.md) - MAISI VAE details

---

## Summary

**Status**: ✅ All components ready for deployment

**Deployment Steps**:
1. Deploy download job → 2 hours
2. Verify dataset → 5 minutes
3. Deploy training job → 8-10 hours
4. Total: ~10-12 hours

**Expected Results**:
- PSNR: 44-46 dB
- SSIM: 0.92-0.97
- Model: 729M parameters (Medium U-Net + MAISI VAE)

**Ready to deploy!** 🚀

Follow the steps above sequentially for smooth deployment.
