# Multi-Model Training Guide

This guide explains how to train and manage multiple model variants simultaneously.

## Overview

The `model_suffix` parameter allows you to train multiple models (e.g., with different architectures, pretrained weights, or hyperparameters) and keep their checkpoints organized.

## Configuration

Add `model_suffix` to your training config:

```yaml
training:
  model_suffix: 'pretrained_sd_vae'  # Your model identifier
  # ... other training params
```

## Checkpoint Naming

With `model_suffix` set, checkpoints will be saved as:

- **Best checkpoint**: `checkpoint_best_epoch_10_pretrained_sd_vae.pt`
- **Latest checkpoint**: `checkpoint_latest_pretrained_sd_vae.pt`
- **Final checkpoint**: `checkpoint_final_pretrained_sd_vae.pt`
- **Periodic checkpoints**: `checkpoint_step_5000_pretrained_sd_vae.pt`

Without `model_suffix` (or empty string):

- **Best checkpoint**: `checkpoint_best_epoch_10.pt`
- **Latest checkpoint**: `checkpoint_latest.pt`
- etc.

## Training Multiple Models

### Example 1: Pretrained vs From Scratch

**Config 1** (`config/pretrained_model.yaml`):
```yaml
model_suffix: 'pretrained_sd_vae'
pretrained:
  use_pretrained: true
  vae:
    enabled: true
    model_name: 'stabilityai/sd-vae-ft-mse'
```

**Config 2** (`config/scratch_model.yaml`):
```yaml
model_suffix: 'from_scratch'
pretrained:
  use_pretrained: false
```

Run both:
```bash
# Terminal 1 - Pretrained model
python train.py --config config/pretrained_model.yaml

# Terminal 2 - From scratch model (different pod/GPU)
python train.py --config config/scratch_model.yaml
```

### Example 2: Different Architectures

**Config 1** (4-level U-Net):
```yaml
model_suffix: 'unet_4level'
model:
  unet_channel_mult: [1, 2, 4, 4]
```

**Config 2** (3-level U-Net):
```yaml
model_suffix: 'unet_3level'
model:
  unet_channel_mult: [1, 2, 4]
```

### Example 3: Hyperparameter Search

```yaml
# config_lr_1e4.yaml
model_suffix: 'lr_1e4'
training:
  learning_rate: 0.0001

# config_lr_5e5.yaml
model_suffix: 'lr_5e5'
training:
  learning_rate: 0.00005
```

## Checkpoint Management

### Auto-Resume

The training script automatically finds and resumes from the best checkpoint **with the matching suffix**:

```bash
# Will resume from checkpoint_best_epoch_X_pretrained_sd_vae.pt if exists
python train.py --config config/pretrained_model.yaml
```

### Manual Resume

Specify exact checkpoint:
```bash
python train.py --config config/pretrained_model.yaml \
  --resume checkpoints/ape_v2v_diffusion/checkpoint_best_epoch_15_pretrained_sd_vae.pt
```

### List Checkpoints

```bash
# List all checkpoints for a specific model
ls checkpoints/ape_v2v_diffusion/*_pretrained_sd_vae.pt

# List best checkpoints for all models
ls checkpoints/ape_v2v_diffusion/checkpoint_best_*.pt
```

## Evaluation

When evaluating on the test set, specify the model:

```bash
python scripts/evaluate_test_set.py \
  --checkpoint checkpoints/ape_v2v_diffusion/checkpoint_best_epoch_25_pretrained_sd_vae.pt \
  --config config/cloud_train_config_a100.yaml \
  --output results/test_evaluation_pretrained_sd_vae.json
```

Compare models:
```bash
# Evaluate pretrained model
python scripts/evaluate_test_set.py \
  --checkpoint checkpoints/ape_v2v_diffusion/checkpoint_best_epoch_25_pretrained_sd_vae.pt \
  --output results/test_pretrained.json

# Evaluate scratch model
python scripts/evaluate_test_set.py \
  --checkpoint checkpoints/ape_v2v_diffusion/checkpoint_best_epoch_30_from_scratch.pt \
  --output results/test_scratch.json

# Compare results
python scripts/compare_models.py \
  results/test_pretrained.json \
  results/test_scratch.json
```

## Storage Considerations

Each model maintains:
- **1 best checkpoint** (automatically deleted when new best is found)
- **1 latest checkpoint** (overwritten each epoch)
- **Periodic checkpoints** (if `checkpoint_every` is set)

Example with 3 models after 50 epochs:
```
checkpoints/ape_v2v_diffusion/
├── checkpoint_best_epoch_48_pretrained_sd_vae.pt   (~2GB)
├── checkpoint_latest_pretrained_sd_vae.pt          (~2GB)
├── checkpoint_best_epoch_45_from_scratch.pt        (~2GB)
├── checkpoint_latest_from_scratch.pt               (~2GB)
├── checkpoint_best_epoch_42_opensora_vae.pt        (~2GB)
└── checkpoint_latest_opensora_vae.pt               (~2GB)
Total: ~12GB for 3 models
```

## Best Practices

1. **Use descriptive suffixes**: `pretrained_sd_vae` is better than `model1`

2. **Keep suffix short**: Long suffixes make filenames unwieldy

3. **Document your experiments**: Keep a log of what each suffix represents
   ```
   pretrained_sd_vae: Uses SD VAE, trained with 2-phase approach
   from_scratch: No pretrained weights, standard training
   opensora_vae: Uses OpenSora VAE instead of SD VAE
   ```

4. **Clean up old checkpoints**: Remove periodic checkpoints if not needed
   ```bash
   # Keep only best and latest
   rm checkpoints/ape_v2v_diffusion/checkpoint_step_*.pt
   ```

5. **Set `keep_last_n_checkpoints: 1`** in config to save disk space

## Tips

- Use the same `experiment_name` for all model variants to group them in one directory
- Set different `model_suffix` values to distinguish them
- The suffix appears in logs: `"Model suffix: pretrained_sd_vae"`
- Checkpoints from different models won't conflict or overwrite each other
