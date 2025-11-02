# MAISI Pretrained VAE - Integration Strategy

**Branch**: `maisi-pretrained-vae`
**Goal**: Make model compatible with NVIDIA MAISI pretrained VAE weights

---

## 🔍 Problem Analysis

### **Issue: 3D Convolution Depth Requirement**

**Error encountered:**
```
RuntimeError: Calculated padded input size per channel: (2 x 257 x 257).
Kernel size: (3 x 3 x 3). Kernel size can't be greater than actual input size
```

**Root Cause:**
- MAISI VAE uses 3D convolutions with kernel_size=(3, 3, 3)
- Designed for 3D CT volumes: (B, C, D, H, W) where D > 1 (depth slices)
- Our data is 2D+time: (B, C, T, H, W) where each frame is a 2D slice
- Processing single frames (D=1) causes kernel to be larger than input depth

**MAISI Architecture Expectations:**
```
Input: (B, 1, D, H, W)  where D >= 3 (minimum depth for 3x3x3 kernel)
Examples:
  - Full CT volume: (1, 1, 64, 512, 512)
  - Cropped volume: (1, 1, 16, 256, 256)
  - Our frames: (1, 1, 1, 256, 256) ❌ TOO SMALL
```

---

## 💡 Solution Strategies

### **Strategy 1: Temporal Frame Stacking** ⭐ SELECTED

**Concept**: Group consecutive video frames into 3D volumes

**Implementation:**
```python
# Current approach (fails):
for t in range(T):
    frame = video[:, :, t, :, :]  # (B, 1, H, W)
    frame_3d = frame.unsqueeze(2)  # (B, 1, 1, H, W) ❌ D=1
    latent = maisi_vae.encode(frame_3d)  # ERROR!

# New approach (temporal stacking):
stack_size = 4  # Process 4 frames at a time
for i in range(0, T, stack_size):
    frames = video[:, :, i:i+stack_size, :, :]  # (B, 1, 4, H, W) ✓
    latent = maisi_vae.encode(frames)  # SUCCESS!
```

**Advantages:**
✅ Preserves temporal coherence (frames processed together)
✅ Utilizes MAISI's 3D convolutions properly
✅ No modification to pretrained weights needed
✅ Natural for video data

**Challenges:**
- Must handle videos not divisible by stack_size (padding)
- Latent temporal dimension changes: T → T/stack_size
- U-Net must handle reduced temporal dimension

**Parameters:**
- `stack_size`: 4 or 8 frames per 3D volume
  - Too small (2-3): Barely meets minimum, limited 3D context
  - Optimal (4-8): Good 3D context, manageable memory
  - Too large (16+): Memory intensive, less temporal granularity

---

### **Strategy 2: Sliding Window with Overlap**

**Concept**: Process video with overlapping 3D windows

```python
window_size = 8  # 8 frames per window
stride = 4       # 50% overlap

for i in range(0, T-window_size+1, stride):
    window = video[:, :, i:i+window_size, :, :]  # (B, 1, 8, H, W)
    latent = maisi_vae.encode(window)
    # Aggregate overlapping latents (average or select center)
```

**Advantages:**
✅ Better temporal coherence (overlap)
✅ Flexible window sizing

**Challenges:**
❌ More complex (need to aggregate overlapping regions)
❌ Higher computational cost (processing overlap)

---

### **Strategy 3: Modify MAISI Architecture** ❌ NOT RECOMMENDED

**Concept**: Change kernel sizes to allow D=1

```python
# Modify MAISI to use (1, 3, 3) or (2, 3, 3) kernels
# This would break pretrained weights!
```

**Why NOT to do this:**
❌ Breaks pretrained weight compatibility
❌ Defeats purpose of using pretrained MAISI
❌ Would need to retrain (same as MAISI-inspired from scratch)

---

## 📐 Selected Approach: Temporal Frame Stacking

### **Architecture Design**

```
Video Input: (B, 1, T, H, W)  e.g., (2, 1, 16, 256, 256)
   ↓
Group into 3D volumes (stack_size=4):
   Volume 1: frames [0:4]   → (2, 1, 4, 256, 256)
   Volume 2: frames [4:8]   → (2, 1, 4, 256, 256)
   Volume 3: frames [8:12]  → (2, 1, 4, 256, 256)
   Volume 4: frames [12:16] → (2, 1, 4, 256, 256)
   ↓
MAISI VAE Encode (each volume independently):
   Latent 1: (2, 4, 1, 32, 32)  [spatial compression 8×]
   Latent 2: (2, 4, 1, 32, 32)
   Latent 3: (2, 4, 1, 32, 32)
   Latent 4: (2, 4, 1, 32, 32)
   ↓
Stack latents along temporal dimension:
   Combined: (2, 4, 4, 32, 32)  [T_latent = T_video / stack_size]
   ↓
U-Net Denoising: operates on (2, 4, 4, 32, 32)
   ↓
MAISI VAE Decode (each volume):
   Reconstructed: (2, 1, 16, 256, 256)
```

### **Key Parameters**

```python
# Video settings
T_video = 16              # Input video frames
H, W = 256, 256          # Spatial resolution

# MAISI stacking
stack_size = 4           # Frames per 3D volume
T_latent = T_video // stack_size  # = 4

# Latent dimensions
latent_channels = 4      # MAISI latent channels
spatial_compression = 8  # H,W → H//8, W//8
```

---

## 🛠️ Implementation Plan

### **Phase 1: Update VAE Wrapper** (Current Task)

**File**: `models/vae.py`

```python
class VideoVAE(nn.Module):
    def __init__(self, ..., maisi_stack_size=4):
        self.maisi_stack_size = maisi_stack_size  # NEW

    def _encode_with_maisi(self, x):
        """
        Encode video using MAISI VAE with temporal stacking

        Args:
            x: (B, C, T, H, W) video

        Returns:
            z: (B, latent_dim, T//stack_size, h, w) latent
        """
        B, C, T, H, W = x.shape
        stack_size = self.maisi_stack_size

        # Pad if T not divisible by stack_size
        if T % stack_size != 0:
            pad_size = stack_size - (T % stack_size)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_size))  # Pad temporal
            T_padded = T + pad_size
        else:
            T_padded = T

        # Reshape to 3D volumes: (B, C, T, H, W) → (B*num_volumes, C, stack_size, H, W)
        num_volumes = T_padded // stack_size
        x_volumes = x.reshape(B * num_volumes, C, stack_size, H, W)

        # Encode with MAISI (handles 3D volumes)
        z_volumes = self.maisi_vae.encode(x_volumes)  # (B*num_volumes, latent_dim, 1, h, w)

        # Reshape back: (B*num_volumes, latent_dim, 1, h, w) → (B, latent_dim, num_volumes, h, w)
        z = z_volumes.reshape(B, self.latent_dim, num_volumes, ...)

        # Scale latent
        z = z * self.scaling_factor

        return z
```

### **Phase 2: Update Configuration**

**File**: `config/cloud_train_config_maisi_pretrained.yaml` (NEW)

```yaml
model:
  in_channels: 1
  latent_dim: 4
  vae_maisi_stack_size: 4  # NEW: frames per 3D volume

pretrained:
  use_pretrained: true
  vae:
    enabled: true
    use_maisi: true
    checkpoint_path: './pretrained/maisi_vae/models/autoencoder.pt'
    stack_size: 4  # NEW
```

### **Phase 3: Update U-Net**

**Challenge**: U-Net expects (B, latent_dim, T, h, w) but now T is reduced

**Solution**: U-Net already handles any temporal dimension, no changes needed!

### **Phase 4: Testing**

1. Test VAE encode with stacking
2. Test VAE decode with unstacking
3. Test full forward pass
4. Validate reconstruction quality

---

## 📊 Expected Benefits

### **With MAISI Pretrained Weights**

| Metric | MAISI-Inspired (Scratch) | MAISI Pretrained | Improvement |
|--------|--------------------------|------------------|-------------|
| **VAE Recon MSE** | 0.05-0.10 (after training) | **0.01-0.03** (epoch 1!) | 2-3× better |
| **Convergence** | 30-40 epochs | **15-25 epochs** | 40% faster |
| **PSNR** | 32-38 dB | **38-42 dB** | +15-20% |
| **SSIM** | 0.80-0.90 | **0.88-0.95** | +10% |

**Why Better:**
- Pretrained on 1000s of CT volumes (understands anatomy)
- Learned optimal latent representation for medical imaging
- Only need to train U-Net (VAE already good)

---

## ⚠️ Potential Challenges

### **Challenge 1: Temporal Granularity Loss**

**Issue**: T=16 → T_latent=4 (4× reduction)
**Impact**: Less temporal detail for U-Net denoising
**Mitigation**:
- Use stack_size=2 or 4 (not 8)
- U-Net still has temporal attention

### **Challenge 2: Memory Usage**

**Issue**: Processing 4-frame volumes vs single frames
**Impact**: Slightly higher memory per forward pass
**Mitigation**:
- Stack_size=4 is manageable
- Still 3× savings from grayscale

### **Challenge 3: Edge Cases**

**Issue**: Videos with T not divisible by stack_size
**Solution**: Temporal padding (pad with last frame)

---

## 🎯 Success Criteria

Integration successful if:

1. ✅ MAISI VAE loads without errors
2. ✅ First epoch reconstruction MSE < 0.05 (pretrained quality)
3. ✅ Generated samples show sharp anatomy (not blurry)
4. ✅ Training converges faster than scratch (~20 epochs vs 30)
5. ✅ Final PSNR > 38 dB, SSIM > 0.88

---

## 📝 Next Steps

1. ✅ Create branch `maisi-pretrained-vae`
2. ⏳ Implement temporal stacking in `_encode_with_maisi()`
3. ⏳ Implement temporal unstacking in `_decode_with_maisi()`
4. ⏳ Create new config file for MAISI pretrained
5. ⏳ Test with downloaded MAISI weights
6. ⏳ Compare quality vs MAISI-inspired scratch

---

**Status**: Ready to implement
**Risk Level**: Medium (architectural changes, but well-defined)
**Expected Completion**: 2-3 hours implementation + testing
