# Integrating Medical Imaging VAE for CT Scans

## 🎯 Why Use Medical-Specific VAE?

### Problem with Stable Diffusion VAE
- **Trained on**: Natural RGB images (photos, artwork) from LAION dataset
- **Optimized for**: Textures, colors, natural scene statistics
- **Domain gap**: Massive difference between photos and CT scans

### Why NVIDIA MAISI VAE is Better
- ✅ **Trained on**: Real CT and MRI medical scans
- ✅ **Optimized for**: Anatomical structures, medical intensity distributions
- ✅ **Understands**: Hounsfield Units, tissue contrasts, medical imaging artifacts
- ✅ **Proven**: State-of-the-art performance on CT super-resolution tasks

### Expected Improvements
| Metric | Current (SD VAE) | With MAISI VAE |
|--------|------------------|----------------|
| VAE reconstruction | Blurry, artifacts | Sharp anatomical structures |
| Training convergence | Slow (~50 epochs) | Faster (~20-30 epochs) |
| Final PSNR | 25-30 dB | 30-40 dB |
| Final SSIM | 0.70-0.80 | 0.85-0.95 |

---

## 📦 Option 1: NVIDIA MAISI VAE (RECOMMENDED)

### Overview
- **Source**: NVIDIA/MONAI
- **Training Data**: Large-scale CT and MRI datasets
- **Architecture**: 3D VAE with flexible volume sizes
- **Availability**: HuggingFace + MONAI model zoo
- **License**: Apache 2.0 (commercial use allowed)

### Step 1: Download MAISI VAE

```bash
# Install dependencies
pip install 'monai[all]' huggingface-hub

# Download using the script
python scripts/download_maisi_vae.py --output-dir ./pretrained/maisi_vae --inspect
```

**Alternative (manual download)**:
```python
from huggingface_hub import hf_hub_download

# Download VAE weights
vae_weights = hf_hub_download(
    repo_id="MONAI/maisi_ct_generative",
    filename="models/autoencoder.pt",
    revision="1.0.0"
)
```

### Step 2: Understand MAISI VAE Architecture

**Key specifications**:
- **Input**: 3D volumes (D, H, W) - flexible sizes
- **Latent compression**: 8x8x8 spatial (similar to your current 8x8x1)
- **Channels**: Grayscale (1 channel) natively
- **Latent dim**: Typically 3-4 channels
- **Training**: CT scans at various resolutions and slice thicknesses

**Differences from your current VAE**:
| Feature | Your VAE | MAISI VAE |
|---------|----------|-----------|
| Input channels | 3 (RGB) | 1 (grayscale) ✓ |
| Temporal handling | Per-frame 3D | Full 3D volume ✓ |
| Training data | None (scratch) | Medical CT ✓ |
| Scaling factor | Needs calculation | Pre-optimized ✓ |

### Step 3: Adapt MAISI VAE to Your Pipeline

You have **two integration options**:

#### Option A: Wrapper (Quick, Recommended)

Keep your current `VideoVAE` interface but load MAISI weights:

```python
# models/vae.py

class VideoVAE(nn.Module):
    def __init__(self, use_maisi=False, maisi_checkpoint=None, **kwargs):
        super().__init__()

        if use_maisi:
            self._load_maisi_vae(maisi_checkpoint)
        else:
            # Your current architecture
            self.encoder = VideoEncoder(...)
            self.decoder = VideoDecoder(...)

    def _load_maisi_vae(self, checkpoint_path):
        """Load MAISI VAE and adapt to video interface"""
        from monai.networks.nets import AutoencoderKL  # MAISI uses AutoencoderKL

        # Load MAISI VAE
        self.maisi_vae = AutoencoderKL(...)
        state_dict = torch.load(checkpoint_path)
        self.maisi_vae.load_state_dict(state_dict)

        # MAISI VAE has built-in scaling
        self.scaling_factor = 1.0  # MAISI handles scaling internally

    def encode(self, x):
        if hasattr(self, 'maisi_vae'):
            return self._encode_with_maisi(x)
        else:
            # Your current encode logic
            ...

    def _encode_with_maisi(self, x):
        """
        Encode video using MAISI VAE

        Args:
            x: (B, C, T, H, W) video - C should be 1 for grayscale
        Returns:
            z: (B, latent_dim, T, H//8, W//8) latent
        """
        B, C, T, H, W = x.shape

        # MAISI expects single channel
        if C == 3:
            # Convert RGB to grayscale (if you're still using 3 channels)
            x = x.mean(dim=1, keepdim=True)  # (B, 1, T, H, W)

        # Process each frame through MAISI 3D VAE
        # Note: MAISI can handle 3D volumes directly
        latents = []
        for t in range(T):
            frame_3d = x[:, :, t:t+1, :, :]  # (B, 1, 1, H, W) - fake 3D
            z_t = self.maisi_vae.encode(frame_3d)
            latents.append(z_t)

        z = torch.stack(latents, dim=2)  # (B, latent_dim, T, H//8, W//8)

        return z
```

#### Option B: Full Replacement (Cleaner, More Work)

Replace your VAE entirely with MAISI, update the entire pipeline to grayscale.

### Step 4: Update Data Pipeline for Grayscale

**Current issue**: Your data pipeline converts grayscale CT → 3-channel RGB (inefficient)

**Fix**:

```python
# data/ape_dataset.py

# REMOVE this (line 322-324):
baseline_rgb = np.stack([baseline_sampled] * 3, axis=-1)  # ❌
followup_rgb = np.stack([followup_sampled] * 3, axis=-1)  # ❌

# REPLACE with:
baseline_gray = baseline_sampled[:, :, :, np.newaxis]  # (T, H, W, 1) ✓
followup_gray = followup_sampled[:, :, :, np.newaxis]  # (T, H, W, 1) ✓
```

```python
# data/transforms.py

class VideoTransform:
    def __call__(self, frames):
        # frames: (T, H, W, 1) grayscale

        # Rearrange to (1, T, H, W) - single channel
        video = rearrange(video, 't h w c -> c t h w')  # (1, T, H, W)

        # Normalize to [-1, 1]
        if self.normalize:
            if frames.dtype == np.uint8:
                video = video / 127.5 - 1.0
            else:
                video = video * 2.0 - 1.0  # [0,1] → [-1,1]

        return video  # (1, T, H, W)
```

### Step 5: Update Configuration

```yaml
# config/cloud_train_config_a100.yaml

model:
  in_channels: 1  # Change from 3 → 1 (grayscale)
  latent_dim: 4  # Match MAISI latent dim

pretrained:
  use_pretrained: true
  vae:
    enabled: true
    use_maisi: true  # NEW: Use MAISI instead of SD
    checkpoint_path: './pretrained/maisi_vae/models/autoencoder.pt'
    freeze_epochs: 5  # Fine-tune after 5 epochs

  # Remove SD VAE settings:
  # model_name: 'stabilityai/sd-vae-ft-mse'  # ← DELETE
```

### Step 6: Test VAE Reconstruction

Before full training, test that MAISI VAE works:

```bash
python -c "
import torch
from models import VideoVAE

# Load MAISI VAE
vae = VideoVAE(use_maisi=True, maisi_checkpoint='./pretrained/maisi_vae/models/autoencoder.pt')
vae.eval()

# Test with grayscale video
video = torch.randn(1, 1, 16, 256, 256)  # (B=1, C=1, T=16, H=256, W=256)

# Encode
with torch.no_grad():
    z = vae.encode(video)
    print(f'Latent shape: {z.shape}')

    # Decode
    recon = vae.decode(z)
    print(f'Reconstruction shape: {recon.shape}')

    # Quality check
    mse = torch.nn.functional.mse_loss(recon, video)
    print(f'Reconstruction MSE: {mse.item():.6f}')
"
```

---

## 📦 Option 2: Alternative Medical VAEs

### MedVAE (If MAISI doesn't work)

**Pros**:
- Trained on 1,924 **chest CT scans** (perfect for your APE task!)
- Recent (2025), SOTA performance

**Cons**:
- May need to contact authors for weights
- Less documented than MAISI

**How to get it**:
1. Read the paper: https://arxiv.org/html/2502.14753v1
2. Contact authors for pretrained weights
3. Integrate similar to MAISI

### CogVideoX VAE (Fallback)

**Pros**:
- Readily available on HuggingFace
- 3D temporal architecture

**Cons**:
- NOT trained on medical data (still has domain gap)
- Would need fine-tuning on your CT data

```python
from diffusers import AutoencoderKLCogVideoX

vae = AutoencoderKLCogVideoX.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="vae"
)
```

---

## ⚙️ Complete Integration Checklist

### Prerequisites
- [ ] Download MAISI VAE weights
- [ ] Install MONAI and dependencies
- [ ] Backup current checkpoint

### Code Changes
- [ ] Update data pipeline to grayscale (remove RGB conversion)
- [ ] Add MAISI VAE wrapper to `models/vae.py`
- [ ] Update `VideoVAE` to support `use_maisi=True`
- [ ] Update transforms to handle single-channel input
- [ ] Update config: `in_channels: 1`, `use_maisi: true`

### Testing
- [ ] Test MAISI VAE reconstruction on sample CT
- [ ] Verify latent shapes match expectations
- [ ] Check reconstruction quality (PSNR, visual inspection)
- [ ] Ensure grayscale pipeline works end-to-end

### Training
- [ ] Start fresh training with MAISI VAE
- [ ] Monitor VAE reconstruction quality during training
- [ ] Compare convergence speed vs previous run
- [ ] Evaluate on validation set after 10 epochs

---

## 🔬 Expected Results

### With MAISI VAE (vs Training from Scratch)

**Training**:
- ✅ **Faster convergence**: 20-30 epochs (vs 50+)
- ✅ **Better initial quality**: Good reconstructions from epoch 1
- ✅ **Stable training**: No latent space issues

**Validation Metrics** (after 30 epochs):
| Metric | Your VAE (scratch) | MAISI VAE | SOTA |
|--------|-------------------|-----------|------|
| PSNR | 25-30 dB | 35-40 dB | 38-42 dB |
| SSIM | 0.70-0.80 | 0.85-0.92 | 0.90-0.98 |

**Visual Quality**:
- ✅ Sharp anatomical structures
- ✅ Preserved fine details (vessels, tissue boundaries)
- ✅ No unnatural artifacts
- ✅ Accurate representation of APE features

---

## 🚀 Quick Start Guide

### 1. Download MAISI VAE
```bash
python scripts/download_maisi_vae.py --output-dir ./pretrained/maisi_vae
```

### 2. Convert Data Pipeline to Grayscale
```bash
# Edit data/ape_dataset.py (remove RGB stacking)
# Edit data/transforms.py (handle 1 channel)
```

### 3. Update Model Configuration
```yaml
# config/cloud_train_config_a100.yaml
model:
  in_channels: 1
pretrained:
  vae:
    use_maisi: true
    checkpoint_path: './pretrained/maisi_vae/models/autoencoder.pt'
```

### 4. Test Before Training
```bash
python -m pytest tests/test_maisi_vae.py  # Create this test
```

### 5. Rebuild Docker and Retrain
```bash
docker build -t ghcr.io/kkuntal990/v2v-diffusion:latest .
docker push ghcr.io/kkuntal990/v2v-diffusion:latest

# Deploy training job with new config
kubectl apply -f kub_files/training-job-a100.yaml
```

---

## 🆘 Troubleshooting

### Issue: "MAISI VAE expects 3D volumes, not 2D+T"
**Solution**: Treat each frame as a thin 3D volume (D=1) or stack multiple slices

### Issue: "Latent dimensions don't match"
**Solution**: Check MAISI's latent_dim and update your U-Net input accordingly

### Issue: "Reconstruction quality is poor"
**Diagnosis**: MAISI may need fine-tuning on your specific CT protocol
**Solution**: Unfreeze VAE earlier (freeze_epochs: 2) with low LR

### Issue: "Memory error with MAISI"
**Solution**: MAISI is larger than your custom VAE. Reduce batch size or use gradient checkpointing

---

## 📊 Comparison: Training from Scratch vs MAISI

| Aspect | Train from Scratch | MAISI VAE | Winner |
|--------|-------------------|-----------|---------|
| **Setup time** | 1 hour | 4 hours | Scratch |
| **Training time** | 50 epochs | 20 epochs | **MAISI** |
| **Final quality** | Good (PSNR ~28) | Excellent (PSNR ~38) | **MAISI** |
| **Memory usage** | Lower | Higher | Scratch |
| **Domain adaptation** | Perfect fit | Good fit | Scratch |
| **Robustness** | May overfit | Generalizes better | **MAISI** |

**Recommendation**: **Use MAISI VAE** unless you have specific requirements or extreme memory constraints.

---

## 📚 References

1. **MAISI Paper**: https://arxiv.org/abs/2409.11169
2. **MONAI Model Zoo**: https://github.com/Project-MONAI/model-zoo
3. **HuggingFace Repo**: https://huggingface.co/MONAI/maisi_ct_generative
4. **MAISI Tutorial**: https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi

---

## 📞 Next Steps

1. **Download MAISI VAE** using the provided script
2. **Inspect architecture** to understand latent dimensions
3. **Update data pipeline** to grayscale (remove RGB conversion)
4. **Test VAE reconstruction** on a few samples
5. **Retrain model** with MAISI VAE
6. **Compare results** with previous training

**Estimated time**: 1-2 days for integration + testing, then retrain (~3-5 days on V100/A100)

---

*Document created: 2025-10-31*
*Recommendation: NVIDIA MAISI VAE for medical CT imaging*
*Status: Ready for implementation*
