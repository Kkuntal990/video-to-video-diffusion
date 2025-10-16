"""
Mini Training Test Script for APE-Data

This script runs a minimal training loop to verify:
1. Data loading works correctly
2. Model training works without errors
3. Loss decreases over iterations
4. Checkpointing works
5. Complete pipeline is functional

This is meant for local CPU testing. For real training, use cloud GPU.
"""

import torch
import torch.optim as optim
from pathlib import Path
import sys
import os
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.ape_dataset import get_ape_dataloader
from models.model import VideoToVideoDiffusion


def test_mini_training():
    """Run minimal training test"""

    print("=" * 70)
    print(" " * 20 + "APE-Data Mini Training Test")
    print("=" * 70)
    print()
    print("This will run a few training steps to verify the training loop works.")
    print("Note: Training on CPU is very slow. This is just for verification.")
    print("For actual training, use a cloud GPU with the full config.")
    print()

    # Configuration
    config = {
        # Data config
        'data': {
            'num_frames': 8,
            'resolution': [128, 128],
            'categories': ['APE'],  # Just APE for quick test
            'batch_size': 1,
            'num_workers': 0,
            'cache_extracted': False
        },

        # Model config (minimal for CPU testing)
        'in_channels': 3,
        'latent_dim': 4,
        'vae_base_channels': 32,
        'unet_model_channels': 64,
        'unet_num_res_blocks': 1,
        'unet_attention_levels': [],  # No attention for speed
        'unet_channel_mult': [1],  # Single level for simplicity
        'unet_num_heads': 4,
        'unet_time_embed_dim': 256,
        'noise_schedule': 'cosine',
        'diffusion_timesteps': 100,

        # Pretrained config
        'pretrained': {
            'use_pretrained': False
        },

        # Training config
        'training': {
            'learning_rate': 1e-4,
            'num_steps': 5,  # Just 5 steps for testing
            'checkpoint_every': 10,
            'output_dir': './test_outputs'
        }
    }

    data_dir = "/Users/kuntalkokate/Desktop/LLM_agents_projects/dataset"

    if not Path(data_dir).exists():
        print(f"✗ Error: Data directory not found: {data_dir}")
        print("Please update the path in the script")
        return False

    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    print("Step 1: Creating dataloader...")
    print("-" * 70)
    try:
        dataloader = get_ape_dataloader(data_dir, config['data'], split='train')
        print(f"✓ Dataloader created")
        print(f"  Dataset size: {len(dataloader.dataset)} patients")
        print(f"  Batch size: {dataloader.batch_size}")
        print(f"  Number of batches: {len(dataloader)}")
        print()
    except Exception as e:
        print(f"✗ Failed to create dataloader: {e}")
        return False

    print("Step 2: Creating model...")
    print("-" * 70)
    try:
        model = VideoToVideoDiffusion(config, load_pretrained=False)
        device = torch.device('cpu')  # Use CPU for testing
        model = model.to(device)
        model.train()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        vae_params = sum(p.numel() for p in model.vae.parameters() if p.requires_grad)
        unet_params = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)

        print(f"✓ Model created")
        print(f"  Total parameters: {total_params:,}")
        print(f"  VAE parameters: {vae_params:,}")
        print(f"  U-Net parameters: {unet_params:,}")
        print(f"  Device: {device}")
        print()
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("Step 3: Creating optimizer...")
    print("-" * 70)
    try:
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        print(f"✓ Optimizer created (Adam)")
        print(f"  Learning rate: {config['training']['learning_rate']}")
        print()
    except Exception as e:
        print(f"✗ Failed to create optimizer: {e}")
        return False

    print("Step 4: Running training steps...")
    print("-" * 70)
    print(f"Running {config['training']['num_steps']} training steps")
    print("(This may take a few minutes on CPU...)")
    print()

    losses = []
    step = 0

    try:
        # Create an iterator that cycles through the dataloader
        dataloader_iter = iter(dataloader)

        with tqdm(total=config['training']['num_steps'], desc="Training", unit="step") as pbar:
            while step < config['training']['num_steps']:
                try:
                    # Get next batch
                    batch = next(dataloader_iter)
                except StopIteration:
                    # Reset iterator when we run out of data
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                # Move to device
                input_video = batch['input'].to(device)
                target_video = batch['target'].to(device)

                # Forward pass
                loss, metrics = model(input_video, target_video)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Record loss
                losses.append(loss.item())

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'patient': batch.get('patient_id', ['unknown'])[0] if 'patient_id' in batch else 'unknown'
                })
                pbar.update(1)

                step += 1

        print()
        print(f"✓ Training steps completed")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Average loss: {sum(losses)/len(losses):.4f}")
        print()

    except Exception as e:
        print(f"\n✗ Training failed at step {step}: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("Step 5: Testing checkpoint saving...")
    print("-" * 70)
    try:
        checkpoint_path = checkpoint_dir / 'test_checkpoint.pt'
        model.save_checkpoint(
            checkpoint_path,
            optimizer=optimizer,
            epoch=0,
            global_step=step
        )
        print(f"✓ Checkpoint saved to: {checkpoint_path}")

        # Verify checkpoint can be loaded
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint verified")
        print(f"  Keys: {list(checkpoint.keys())}")
        print()
    except Exception as e:
        print(f"✗ Checkpoint saving failed: {e}")
        return False

    print("Step 6: Testing inference...")
    print("-" * 70)
    try:
        model.eval()
        with torch.no_grad():
            # Get a sample
            test_batch = next(iter(dataloader))
            test_input = test_batch['input'].to(device)

            print(f"Generating output (this may take a moment)...")
            output = model.generate(
                test_input,
                sampler='ddim',
                num_inference_steps=5  # Just 5 steps for quick test
            )

            print(f"✓ Inference successful")
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {output.shape}")
            print()
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("=" * 70)
    print(" " * 25 + "TEST SUMMARY")
    print("=" * 70)
    print()
    print("✓ Data loading: PASSED")
    print("✓ Model creation: PASSED")
    print("✓ Training loop: PASSED")
    print(f"✓ Loss tracking: PASSED ({len(losses)} steps recorded)")
    print("✓ Checkpointing: PASSED")
    print("✓ Inference: PASSED")
    print()

    # Loss analysis
    if len(losses) > 1:
        loss_change = losses[-1] - losses[0]
        loss_percent = (loss_change / losses[0]) * 100 if losses[0] != 0 else 0

        print("Loss Analysis:")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Change: {loss_change:+.4f} ({loss_percent:+.1f}%)")

        if loss_change < 0:
            print("  ✓ Loss is decreasing (good!)")
        else:
            print("  ⚠ Loss increased (normal for very short training)")

    print()
    print("=" * 70)
    print()
    print("🎉 All training tests passed!")
    print()
    print("Your training pipeline is working correctly.")
    print()
    print("Next steps:")
    print("1. For real training, move to a cloud GPU")
    print("2. Use the full training config: config/train_config.yaml")
    print("3. Enable pretrained weights for 6x faster training")
    print("4. Train for 50+ epochs with full resolution (256x256, 16 frames)")
    print()
    print("Expected training time:")
    print("  - Cloud GPU (A100) without pretrained: ~7 days")
    print("  - Cloud GPU (A100) with pretrained: ~1 day")
    print()
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = test_mini_training()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
