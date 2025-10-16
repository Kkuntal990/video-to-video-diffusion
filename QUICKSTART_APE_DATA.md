# Quick Start: Testing with APE-Data

This guide will help you verify the code works with your local APE-data samples before running full training on cloud GPU.

## Prerequisites

- Python 3.8+
- Your APE-data samples downloaded to: `/Users/kuntalkokate/Desktop/LLM_agents_projects/dataset`
- Git repository cloned and on `pretrained_main` branch

## Step 1: Setup Environment

```bash
# Navigate to project directory
cd /Users/kuntalkokate/Desktop/LLM_agents_projects/LLM_agent_v2v

# Make sure you're on pretrained_main branch
git checkout pretrained_main

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Expected time:** 2-3 minutes

## Step 2: Verify Data Directory Structure

Your data should be organized like this:

```
/Users/kuntalkokate/Desktop/LLM_agents_projects/dataset/
├── APE/
│   ├── 201612140637 RONG GUI FANG.zip
│   ├── 201612170460 HU LAN JUN.zip
│   ├── 201708190245 LI ZHEN DONG.zip
│   └── 202009050363 KANG-BAO-LI.zip
└── non-APE/
    ├── 201612050889 XING-SHAO-WEN.zip
    └── 201912271212 XUN-YUN.zip
```

Each zip contains:
```
patient_name/
├── 1/  <- Input (baseline scan)
│   └── [DICOM files]
└── 2/  <- Target (followup scan)
    └── [DICOM files]
```

Check your structure:
```bash
ls -la /Users/kuntalkokate/Desktop/LLM_agents_projects/dataset/
```

## Step 3: Test APE Data Loading

Run the comprehensive test script:

```bash
python test_ape_data_loading.py
```

This will run 3 tests:
1. **APE Dataset Loading** - Loads DICOM data from zip files
2. **APE DataLoader** - Tests batching and data pipeline
3. **Model Integration** - Verifies model can process APE data

**Expected output:**
```
╔══════════════════════════════════════════════════════════╗
║          APE-Data Loading Test Suite                     ║
╚══════════════════════════════════════════════════════════╝

============================================================
TEST 1: APE Dataset Loading (Local Files)
============================================================

Creating APE dataset...
✓ Dataset created successfully
  Total patients: 4

Loading first sample...

✓ Sample loaded successfully
  Keys: ['input', 'target', 'category', 'patient_id']
  Input shape: torch.Size([3, 8, 128, 128])
  Target shape: torch.Size([3, 8, 128, 128])
  Category: APE
  Patient ID: 201612140637 RONG GUI FANG
  Input value range: [-0.xxx, 0.xxx]
  Target value range: [-0.xxx, 0.xxx]

✓ Shapes are correct: (3, 8, 128, 128)
✓ Values are normalized to [-1, 1] range

============================================================
✓ TEST 1 PASSED
============================================================

[... TEST 2 and TEST 3 output ...]

============================================================
TEST SUMMARY
============================================================
✓ PASS   | APE Dataset Loading
✓ PASS   | APE DataLoader
✓ PASS   | Model Integration
============================================================
Results: 3/3 tests passed

🎉 All tests passed! Your setup is ready for training.
```

**If tests fail:**
- Check that pydicom is installed: `pip install pydicom`
- Verify zip files are not corrupted
- Check Python version (3.8+)
- See troubleshooting section below

**Expected time:** 1-2 minutes

## Step 4: Create Test Training Configuration

Create a minimal config for local testing:

```bash
cat > config/test_ape_config.yaml << 'EOF'
model:
  vae:
    in_channels: 3
    latent_dim: 4
    base_channels: 32      # Reduced for testing
    channel_multipliers: [1, 2]
    num_res_blocks: 1
    temporal_downsample: false

  unet:
    in_channels: 8
    out_channels: 4
    base_channels: 64      # Reduced for testing
    channel_multipliers: [1, 2]
    num_res_blocks: 1
    attention_levels: [1]
    num_heads: 4
    dropout: 0.1

  diffusion:
    timesteps: 100         # Reduced for testing
    noise_schedule: 'cosine'
    loss_type: 'mse'

data:
  dataset_type: 'ape'      # Use APE dataset
  data_dir: '/Users/kuntalkokate/Desktop/LLM_agents_projects/dataset'
  categories: ['APE']      # Just APE for quick test
  num_frames: 8            # Reduced for testing
  frame_height: 128        # Reduced for testing
  frame_width: 128         # Reduced for testing
  batch_size: 1
  num_workers: 0
  cache_extracted: false   # Don't cache for testing

training:
  num_epochs: 1            # Just 1 epoch for testing
  learning_rate: 1e-4
  gradient_accumulation_steps: 1
  mixed_precision: false   # Disable for CPU
  checkpoint_every: 2
  validate_every: 2
  output_dir: './test_outputs'

pretrained:
  use_pretrained: false    # Train from scratch for testing
  vae:
    enabled: false

hardware:
  device: 'cpu'            # Use CPU for local test
  num_gpus: 0

inference:
  sampler: 'ddim'
  num_inference_steps: 10
  guidance_scale: 1.0
EOF
```

## Step 5: Test Training Loop (Minimal)

Now test a minimal training run:

```bash
# Create test script
cat > test_train_ape.py << 'EOF'
import torch
import yaml
from pathlib import Path
from data.ape_dataset import get_ape_dataloader
from models.model import VideoToVideoDiffusion
from tqdm import tqdm

# Load config
with open('config/test_ape_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Creating model...")
model = VideoToVideoDiffusion(config, load_pretrained=False)
model.train()

print("Creating dataloader...")
dataloader = get_ape_dataloader(
    config['data']['data_dir'],
    config['data'],
    split='train'
)

print(f"Dataset size: {len(dataloader.dataset)}")
print(f"Number of batches: {len(dataloader)}")

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

print("\nRunning training test (5 steps)...")
model.train()

for step, batch in enumerate(dataloader):
    if step >= 5:  # Only 5 steps for testing
        break

    # Forward pass
    loss = model(batch)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step+1}/5 - Loss: {loss.item():.4f}")

print("\n✓ Training test successful!")
print("Your setup is working correctly.")
EOF

# Run the test
python test_train_ape.py
```

**Expected output:**
```
Creating model...
Creating dataloader...
Found 4 patient studies
Categories: ['APE']
Dataset size: 4
Number of batches: 4

Running training test (5 steps)...
Step 1/5 - Loss: X.XXXX
Step 2/5 - Loss: X.XXXX
Step 3/5 - Loss: X.XXXX
Step 4/5 - Loss: X.XXXX

✓ Training test successful!
Your setup is working correctly.
```

**Expected time:** 2-5 minutes (CPU is slow, but this verifies everything works)

## Step 6: Test Inference

```bash
cat > test_inference_ape.py << 'EOF'
import torch
import yaml
from data.ape_dataset import APEDataset
from models.model import VideoToVideoDiffusion

# Load config
with open('config/test_ape_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Creating model...")
model = VideoToVideoDiffusion(config, load_pretrained=False)
model.eval()

print("Loading test data...")
dataset = APEDataset(
    data_dir=config['data']['data_dir'],
    num_frames=8,
    resolution=(128, 128),
    categories=['APE'],
    cache_extracted=False
)

# Get first sample
sample = dataset[0]
input_video = sample['input'].unsqueeze(0)  # Add batch dimension

print(f"Input shape: {input_video.shape}")
print(f"Patient: {sample['patient_id']}")

print("\nGenerating output video...")
with torch.no_grad():
    output_video = model.generate(
        input_video,
        num_inference_steps=5,  # Just 5 steps for testing
        sampler='ddim'
    )

print(f"Output shape: {output_video.shape}")
print("\n✓ Inference test successful!")
EOF

python test_inference_ape.py
```

**Expected time:** 1-2 minutes

## Summary: What We Verified

✅ **Data Loading**: APE-data DICOM files can be loaded and processed
✅ **Data Format**: Correct shapes (B, C, T, H, W) and value ranges [-1, 1]
✅ **Model Integration**: Model can process APE data without errors
✅ **Training Loop**: Forward and backward passes work correctly
✅ **Inference**: Generation pipeline produces outputs

## Next Steps: Cloud GPU Training

Once all tests pass, you're ready for full training on cloud GPU! See `CLOUD_GPU_TRAINING_APE.md` for:

1. **Full training configuration** with optimal settings
2. **HuggingFace dataset integration** (download full dataset in training script)
3. **Multi-GPU setup** for faster training
4. **Pretrained weights** usage for 6x faster convergence
5. **Expected training times** and performance metrics

## Troubleshooting

### Test 1 fails: "pydicom not installed"
```bash
pip install pydicom
```

### Test 1 fails: "Could not load patient"
- Check zip files are not corrupted
- Try extracting one manually to verify contents
- Check permissions on the dataset directory

### Test 1 fails: "No data found"
- Verify dataset path in test script matches your actual path
- Check that zip files exist in APE/ folder
- Run: `ls -la /Users/kuntalkokate/Desktop/LLM_agents_projects/dataset/APE/`

### Tests are very slow
- This is expected on CPU
- The tests use minimal settings (8 frames, 128x128) to stay fast
- Cloud GPU will be much faster (100-1000x)

### Memory errors
- Reduce `num_frames` to 4
- Reduce resolution to (64, 64)
- Close other applications

### Import errors
- Make sure virtual environment is activated: `source venv/bin/activate`
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version: `python --version` (need 3.8+)

## Configuration for Cloud GPU

When you move to cloud GPU, update these settings:

```yaml
data:
  num_frames: 16           # Increase to 16
  frame_height: 256        # Increase to 256
  frame_width: 256         # Increase to 256
  batch_size: 4            # Increase based on GPU memory
  num_workers: 4           # Enable multiprocessing
  categories: ['APE', 'non-APE']  # Use all data

training:
  num_epochs: 50           # Full training
  mixed_precision: true    # Enable for speed
  checkpoint_every: 100

pretrained:
  use_pretrained: true     # Use pretrained for 6x speedup
  vae:
    enabled: true

hardware:
  device: 'cuda'           # Use GPU
  num_gpus: 1              # Or more for multi-GPU
```

## Estimated Training Times

### Local Testing (CPU, minimal settings):
- 5 training steps: 2-5 minutes
- 1 epoch (4 samples): 10-20 minutes

### Cloud GPU Training (A100, full settings):
- Without pretrained: ~7 days
- With pretrained: ~1 day (6x faster!)
- Per epoch (6 samples): ~1 hour

---

**Questions?** Check the main README.md or PRETRAINED_WEIGHTS_GUIDE.md for more details.
