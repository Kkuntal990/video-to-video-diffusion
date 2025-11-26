# Patch Evaluation & Visualization Guide

This guide explains how to evaluate and visualize your trained CT slice interpolation model on patch-based validation data.

## 📝 Quick Start

### Option 1: Run on Kubernetes (Recommended)

```bash
# Deploy evaluation job
kubectl apply -f kub_files/patch-evaluation-job.yaml

# Monitor progress
kubectl logs -f $(kubectl get pods -l app=v2v-patch-evaluation -o jsonpath='{.items[0].metadata.name}')

# Check results
kubectl exec copy-pod -- ls -lh /workspace/storage_a100/results/patch_evaluation/

# Download results to local machine
kubectl cp copy-pod:/workspace/storage_a100/results/patch_evaluation ./local_results
```

### Option 2: Run Locally

```bash
python scripts/evaluate_and_visualize_patches.py \
  --checkpoint /path/to/checkpoint_best_epoch_24_slice_interp_full3.pt \
  --config config/slice_interpolation_full_medium.yaml \
  --num-samples 50 \
  --output-dir results/patch_eval/
```

---

## 🎯 What It Does

The script will:

1. **Load your trained model** from checkpoint (with automatic `model_suffix` support)
2. **Load validation patches** (8 thick slices → 48 thin slices @ 192×192)
3. **Generate predictions** using DDIM sampling
4. **Calculate metrics** (PSNR & SSIM) for each patch
5. **Create visualizations** showing side-by-side comparisons
6. **Save detailed results** in JSON and CSV format
7. **Show statistics** (best/worst patches, per-category metrics)

---

## 📊 Output Structure

After running, you'll get:

```
results/patch_eval/
├── visualizations/          # PNG images for each patch
│   ├── patch_0000_case_123.png
│   ├── patch_0001_case_124.png
│   └── ...
├── metrics/                 # Detailed metrics
│   ├── patch_metrics.json   # Complete metrics with config
│   └── patch_metrics.csv    # Simple CSV for analysis
└── predictions/             # Optional: saved prediction tensors
    ├── patch_0000_prediction.pt
    └── ...
```

### Visualization Format

Each PNG shows:
- **Row 1**: Input thick slices (8 slices @ 5mm spacing)
- **Row 2**: Ground truth thin slices (48 slices @ 1mm spacing)
- **Row 3**: Model predictions (48 slices @ 1mm spacing)
- **Columns**: 3 representative slices (start, middle, end)
- **Title**: Patient ID, category, PSNR, SSIM

---

## 🔧 Command-Line Options

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--checkpoint` | Path to trained model checkpoint | `checkpoint_best_epoch_24_slice_interp_full3.pt` |
| `--config` | Path to config YAML | `config/slice_interpolation_full_medium.yaml` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-samples` | 20 | Number of patches to evaluate |
| `--split` | val | Dataset split (train/val/test) |
| `--batch-size` | 1 | Batch size for evaluation |
| `--num-inference-steps` | 20 | DDIM steps (fewer = faster, lower quality) |
| `--sampler` | ddim | Sampling method (ddim/ddpm) |
| `--output-dir` | results/patch_eval | Where to save results |
| `--save-predictions` | False | Save .pt files of predictions |
| `--visualize-slices` | 0,24,47 | Which slices to show (thin slice indices) |
| `--device` | cuda | Device to use |
| `--seed` | 42 | Random seed |

---

## 📈 Example Output

```
================================================================================
CT Slice Interpolation - Patch Evaluation & Visualization
================================================================================

Output directory: results/patch_eval

Loading config from config/slice_interpolation_full_medium.yaml...

Patch Configuration:
  Input:  8 slices @ 192×192 (thick)
  Output: 48 slices @ 192×192 (thin)
  Ratio:  6.0× depth interpolation

Loading model...
  Model created: VideoToVideoDiffusion

Loading checkpoint from checkpoint_best_epoch_24_slice_interp_full3.pt...
  Checkpoint epoch: 24
  Best loss: 0.0412

Creating val dataloader...
  Dataset size: 64 patients
  Batch size: 1
  Evaluating 50 patches

================================================================================
Starting Evaluation...
================================================================================

Evaluating patches: 100%|████████████| 50/50 [02:15<00:00, 2.71s/it, PSNR=38.42, SSIM=0.9654]

================================================================================
Evaluation Results
================================================================================

Overall Statistics (50 patches):
  PSNR: 38.24 ± 2.15 dB
    Min: 33.12, Max: 42.67
  SSIM: 0.9642 ± 0.0123
    Min: 0.9312, Max: 0.9801

Per-Category Statistics:
  APE (25 patches):
    PSNR: 37.89 ± 2.34 dB
    SSIM: 0.9621 ± 0.0145
  non-APE (25 patches):
    PSNR: 38.59 ± 1.92 dB
    SSIM: 0.9663 ± 0.0095

Best Patches (by PSNR):
  1. Patch 0042 (case_287): PSNR=42.67, SSIM=0.9801
  2. Patch 0023 (case_156): PSNR=41.85, SSIM=0.9756
  3. Patch 0031 (case_203): PSNR=40.92, SSIM=0.9723
  4. Patch 0015 (case_098): PSNR=40.51, SSIM=0.9701
  5. Patch 0007 (case_045): PSNR=39.88, SSIM=0.9687

Worst Patches (by PSNR):
  1. Patch 0018 (case_112): PSNR=33.12, SSIM=0.9312
  2. Patch 0034 (case_215): PSNR=34.67, SSIM=0.9401
  3. Patch 0029 (case_189): PSNR=35.23, SSIM=0.9445
  4. Patch 0011 (case_071): PSNR=35.89, SSIM=0.9478
  5. Patch 0003 (case_021): PSNR=36.12, SSIM=0.9502

Metrics saved to: results/patch_eval/metrics/patch_metrics.json
CSV saved to: results/patch_eval/metrics/patch_metrics.csv

================================================================================
Evaluation Complete!
================================================================================

Results saved to: results/patch_eval
  Visualizations: 50 images
  Metrics: JSON + CSV
  Predictions: 50 .pt files
```

---

## 📊 Analyzing Results

### Load Metrics in Python

```python
import json
import pandas as pd

# Load JSON
with open('results/patch_eval/metrics/patch_metrics.json', 'r') as f:
    metrics = json.load(f)

# Overall statistics
print(f"Mean PSNR: {metrics['overall']['psnr_mean']:.2f} dB")
print(f"Mean SSIM: {metrics['overall']['ssim_mean']:.4f}")

# Load as DataFrame
df = pd.DataFrame(metrics['per_patch'])

# Group by category
print(df.groupby('category')[['psnr', 'ssim']].mean())

# Find best/worst
print(df.nlargest(5, 'psnr'))
print(df.nsmallest(5, 'psnr'))
```

### Load CSV

```python
import pandas as pd

df = pd.read_csv('results/patch_eval/metrics/patch_metrics.csv')

# Histogram of PSNR
df['psnr'].hist(bins=20)

# PSNR vs SSIM scatter
df.plot.scatter(x='ssim', y='psnr')
```

---

## 🎯 Tips

### For Quick Testing (Fast)
```bash
python scripts/evaluate_and_visualize_patches.py \
  --num-samples 10 \
  --num-inference-steps 10  # Faster but lower quality
```

### For Comprehensive Evaluation (Slow)
```bash
python scripts/evaluate_and_visualize_patches.py \
  --num-samples 100 \
  --num-inference-steps 50 \
  --save-predictions  # Save all predictions for later analysis
```

### For Test Set Evaluation
```bash
python scripts/evaluate_and_visualize_patches.py \
  --split test \
  --num-samples -1  # Evaluate ALL test patches
```

---

## 🔍 Troubleshooting

### Issue: "No checkpoint found"
```bash
# Check checkpoint directory
kubectl exec copy-pod -- ls -lh /workspace/storage_a100/checkpoints/slice_interp_full_medium/

# Manually specify checkpoint
python scripts/evaluate_and_visualize_patches.py \
  --checkpoint /full/path/to/checkpoint.pt
```

### Issue: "CUDA out of memory"
```bash
# Reduce batch size (already 1 by default)
# Or reduce number of inference steps
python scripts/evaluate_and_visualize_patches.py \
  --num-inference-steps 10
```

### Issue: "Config file not found"
```bash
# Make sure you're running from project root
cd /workspace
python scripts/evaluate_and_visualize_patches.py --config config/slice_interpolation_full_medium.yaml
```

---

## 📚 Next Steps

1. **Run evaluation** on your latest checkpoint
2. **Inspect visualizations** to see which patches work well
3. **Analyze metrics** to identify patterns (APE vs non-APE, quality distribution)
4. **Investigate worst patches** to understand model weaknesses
5. **Compare checkpoints** from different epochs

For full-volume reconstruction (stitching patches together), see the separate guide (coming soon).
