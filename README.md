# Video-to-Video Diffusion Model

A PyTorch implementation of a simple video-to-video diffusion model for video transformation tasks such as denoising, restoration, or low-light enhancement.

## Overview

This implementation provides a complete pipeline for training and inference with a video-to-video diffusion model:

- **3D Video VAE**: Compresses videos into latent space
- **3D U-Net Denoiser**: Predicts noise conditioned on input video
- **Diffusion Process**: Forward and reverse diffusion with cosine/linear noise schedules
- **DDIM/DDPM Sampling**: Fast deterministic (DDIM) or stochastic (DDPM) sampling
- **Training Infrastructure**: Mixed precision training, gradient accumulation, checkpointing
- **HuggingFace Integration**: Easy data loading from HuggingFace datasets

## Architecture

### Model Components

1. **Video Encoder-Decoder (VAE)**
   - Input: `(B, 3, T, H, W)` video clips
   - Latent: `(B, 4, T, H/8, W/8)` compressed representation
   - 3D convolutions with residual blocks
   - 8x spatial compression

2. **3D U-Net Denoiser**
   - Predicts noise: `ε_θ(z_t, t, c)`
   - Conditioning via channel-wise concatenation
   - Temporal attention blocks
   - Time embedding injection

3. **Diffusion Process**
   - Forward: `z_t = α_t·z + σ_t·ε`
   - Training loss: `L = E[||ε - ε_θ(z_t, t, c)||²]`
   - Reverse: DDIM or DDPM sampling

## Installation

```bash
# Clone repository
git clone <repository-url>
cd LLM_agent_v2v

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU training)
- 24GB+ GPU memory recommended for full-scale training

## Dataset

The model is designed to work with the [APE-data](https://huggingface.co/datasets/t2ance/APE-data) dataset from HuggingFace.

### Data Format

The dataset should provide video pairs with:
- `input`: Input video (e.g., noisy, low-light)
- `target`: Ground truth video (clean, enhanced)

Alternatively, you can use a directory structure:
```
data/
  input/
    video1.mp4
    video2.mp4
  target/
    video1.mp4
    video2.mp4
```

## Training

### Quick Start

```bash
# Train with default config
python train.py --config config/train_config.yaml
```

### Resume Training

```bash
# Resume from checkpoint
python train.py --config config/train_config.yaml --resume checkpoints/checkpoint_epoch_10.pt
```

### Configuration

Edit `config/train_config.yaml` to customize:

```yaml
# Model settings
model:
  latent_dim: 4
  vae_base_channels: 64
  unet_model_channels: 128
  diffusion_timesteps: 1000

# Data settings
data:
  data_source: 't2ance/APE-data'
  num_frames: 16
  resolution: [256, 256]
  batch_size: 2

# Training settings
training:
  num_epochs: 100
  learning_rate: 0.0001
  use_amp: true
```

### Training Details

- **Epochs**: 100 (default)
- **Batch size**: 2 (requires ~24GB GPU)
- **Resolution**: 256×256
- **Frames per clip**: 16
- **Learning rate**: 1e-4 with cosine decay
- **Optimizer**: Adam
- **Mixed precision**: Enabled by default

### Monitoring

Training logs and metrics are saved to:
- TensorBoard logs: `logs/<experiment_name>/`
- Checkpoints: `checkpoints/<experiment_name>/`

View training progress:
```bash
tensorboard --logdir logs/
```

## Inference

### Generate Video

```bash
python inference.py \
  --checkpoint checkpoints/checkpoint_final.pt \
  --input input_video.mp4 \
  --output output_video.mp4 \
  --sampler ddim \
  --steps 20
```

### Arguments

- `--checkpoint`: Path to trained model checkpoint
- `--input`: Input video file
- `--output`: Output video file
- `--sampler`: `ddim` (faster) or `ddpm` (more diverse)
- `--steps`: Number of denoising steps (default: 20)
- `--num-frames`: Number of frames to process (default: 16)
- `--resolution`: Output resolution (default: 256 256)

### Example

```bash
# Fast inference with DDIM (20 steps)
python inference.py \
  --checkpoint checkpoints/video_diffusion_v1/checkpoint_epoch_50.pt \
  --input data/test_video.mp4 \
  --output results/enhanced_video.mp4 \
  --sampler ddim \
  --steps 20

# Higher quality with more steps
python inference.py \
  --checkpoint checkpoints/video_diffusion_v1/checkpoint_final.pt \
  --input data/test_video.mp4 \
  --output results/enhanced_video_hq.mp4 \
  --sampler ddim \
  --steps 50
```

## Project Structure

```
LLM_agent_v2v/
├── config/
│   └── train_config.yaml       # Training configuration
├── models/
│   ├── vae.py                  # 3D Video VAE
│   ├── unet3d.py              # 3D U-Net denoiser
│   ├── diffusion.py           # Diffusion process
│   └── model.py               # Complete V2V model
├── data/
│   ├── dataset.py             # Video dataset loader
│   └── transforms.py          # Video preprocessing
├── training/
│   ├── trainer.py             # Training loop
│   └── scheduler.py           # LR schedulers
├── inference/
│   ├── sampler.py             # DDIM/DDPM samplers
│   └── generate.py            # Video generation
├── utils/
│   ├── logger.py              # Logging utilities
│   └── metrics.py             # PSNR, SSIM metrics
├── train.py                   # Main training script
├── inference.py               # Main inference script
└── requirements.txt           # Dependencies
```

## Model Specifications

### Default Configuration

- **VAE latent dim**: 4 channels
- **VAE base channels**: 64
- **U-Net channels**: 128
- **U-Net depth**: 4 levels (1, 2, 4, 4 channel multipliers)
- **Attention levels**: [1, 2]
- **Temporal attention heads**: 4
- **Diffusion timesteps**: 1000 (training), 20 (inference)
- **Noise schedule**: Cosine

### Memory Requirements

- **Training**: ~24GB GPU for batch size 2
- **Inference**: ~12GB GPU for 256×256 resolution
- Reduce resolution or batch size for smaller GPUs

## Performance

### Metrics

The model tracks:
- Training loss (MSE)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

### Speed

- **Training**: ~2 sec/step (batch size 2, A100 GPU)
- **Inference**: ~30 sec/video (DDIM 20 steps, A100 GPU)

## Advanced Usage

### Custom Dataset

```python
from data import VideoDataset

# Use custom video pairs
video_pairs = [
    {'input': 'path/to/input1.mp4', 'target': 'path/to/target1.mp4'},
    {'input': 'path/to/input2.mp4', 'target': 'path/to/target2.mp4'},
]

dataset = VideoDataset(
    data_source=video_pairs,
    source_type='list',
    num_frames=16,
    resolution=(256, 256)
)
```

### Programmatic Training

```python
from models import VideoToVideoDiffusion
from training import Trainer

# Create model
config = {...}  # Your config dict
model = VideoToVideoDiffusion(config)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_loader,
    config=train_config
)

# Train
trainer.train()
```

### Programmatic Inference

```python
from models import VideoToVideoDiffusion
from inference.generate import generate_video

# Load model
model, _ = VideoToVideoDiffusion.load_checkpoint('checkpoint.pt')

# Generate
output = generate_video(
    model=model,
    input_video_path='input.mp4',
    output_path='output.mp4',
    sampler_type='ddim',
    num_inference_steps=20
)
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in config
- Reduce `resolution` (e.g., 128×128)
- Reduce `num_frames` (e.g., 8)
- Disable `use_amp` (mixed precision)

### Training Instability

- Reduce learning rate
- Increase warmup epochs
- Enable gradient clipping (`max_grad_norm`)
- Check data normalization

### Poor Quality

- Train for more epochs
- Increase model capacity (`unet_model_channels`)
- Use more training data
- Increase inference steps (50+)

## Citation

This is a simplified implementation based on:

```bibtex
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={NeurIPS},
  year={2020}
}

@article{song2020denoising,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  journal={ICLR},
  year={2021}
}
```

## License

MIT License

## Acknowledgments

- Video diffusion model architecture inspired by [lucidrains/video-diffusion-pytorch](https://github.com/lucidrains/video-diffusion-pytorch)
- DDIM sampling implementation based on [Song et al. 2021](https://arxiv.org/abs/2010.02502)
- Dataset: [APE-data](https://huggingface.co/datasets/t2ance/APE-data)
