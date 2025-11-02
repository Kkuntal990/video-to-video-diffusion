#!/usr/bin/env python3
"""
Download NVIDIA MAISI VAE for Medical CT Imaging

This script downloads the pretrained MAISI VAE from MONAI model zoo
and adapts it for the video-to-video diffusion pipeline.

MAISI VAE is specifically trained on CT and MRI scans, making it
ideal for medical imaging tasks.

Usage:
    python scripts/download_maisi_vae.py --output-dir ./pretrained/maisi_vae
"""

import argparse
import logging
from pathlib import Path
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_maisi_bundle(output_dir: str):
    """
    Download MAISI CT generative bundle from MONAI model zoo

    Args:
        output_dir: Directory to save the bundle
    """
    try:
        from monai.bundle import download
        logger.info("Downloading MAISI bundle from MONAI model zoo...")

        # Download the bundle
        download(
            name="maisi_ct_generative",
            version="1.0.0",
            bundle_dir=output_dir,
            source="github"  # Download from model-zoo GitHub
        )

        logger.info(f"✓ Successfully downloaded MAISI bundle to {output_dir}")
        return True

    except ImportError:
        logger.error("MONAI not installed. Install with: pip install 'monai[all]'")
        return False
    except Exception as e:
        logger.error(f"Error downloading bundle: {e}")
        logger.info("\\nTrying alternative method (HuggingFace)...")
        return download_from_huggingface(output_dir)


def download_from_huggingface(output_dir: str):
    """
    Download MAISI VAE from HuggingFace as fallback

    Args:
        output_dir: Directory to save weights
    """
    try:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading from HuggingFace: MONAI/maisi_ct_generative...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Download VAE checkpoint
        vae_file = hf_hub_download(
            repo_id="MONAI/maisi_ct_generative",
            filename="models/autoencoder.pt",  # VAE weights
            revision="1.0.0",
            cache_dir=str(output_path)
        )

        logger.info(f"✓ Downloaded VAE weights to {vae_file}")

        # Download config
        config_file = hf_hub_download(
            repo_id="MONAI/maisi_ct_generative",
            filename="configs/config_maisi_vae_train.json",
            revision="1.0.0",
            cache_dir=str(output_path)
        )

        logger.info(f"✓ Downloaded config to {config_file}")

        return True

    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Error downloading from HuggingFace: {e}")
        return False


def inspect_maisi_vae(checkpoint_path: str):
    """
    Inspect MAISI VAE architecture

    Args:
        checkpoint_path: Path to VAE checkpoint
    """
    logger.info(f"\\nInspecting MAISI VAE checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            logger.info("\\nCheckpoint keys:")
            for key in checkpoint.keys():
                logger.info(f"  - {key}")

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            logger.info("\\nModel architecture (first 10 layers):")
            for i, (name, param) in enumerate(state_dict.items()):
                if i >= 10:
                    logger.info(f"  ... ({len(state_dict)} total layers)")
                    break
                logger.info(f"  {name}: {param.shape}")

            # Try to infer architecture details
            logger.info("\\nArchitecture details:")
            encoder_layers = [k for k in state_dict.keys() if 'encoder' in k.lower()]
            decoder_layers = [k for k in state_dict.keys() if 'decoder' in k.lower()]

            logger.info(f"  Encoder layers: {len(encoder_layers)}")
            logger.info(f"  Decoder layers: {len(decoder_layers)}")

            # Check for 3D convolutions
            conv3d_layers = [k for k in state_dict.keys() if 'conv3d' in k.lower() or '.conv' in k.lower()]
            logger.info(f"  Conv layers: {len(conv3d_layers)}")

        else:
            logger.warning("Checkpoint is not a dictionary, may be raw state_dict")

    except Exception as e:
        logger.error(f"Error inspecting checkpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description='Download NVIDIA MAISI VAE for CT imaging')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./pretrained/maisi_vae',
        help='Directory to save the pretrained VAE'
    )
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Inspect the downloaded VAE architecture'
    )

    args = parser.parse_args()

    logger.info(f"{'='*70}")
    logger.info(f"NVIDIA MAISI VAE Download")
    logger.info(f"{'='*70}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download the bundle
    success = download_maisi_bundle(args.output_dir)

    if not success:
        logger.error("Failed to download MAISI VAE")
        return

    # Inspect if requested
    if args.inspect:
        # Try to find the VAE checkpoint
        possible_paths = [
            output_path / "models" / "autoencoder.pt",
            output_path / "maisi_ct_generative" / "models" / "autoencoder.pt",
        ]

        for path in possible_paths:
            if path.exists():
                inspect_maisi_vae(str(path))
                break
        else:
            logger.warning("Could not find VAE checkpoint for inspection")

    logger.info(f"\\n{'='*70}")
    logger.info(f"Download Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"VAE saved to: {output_path}")
    logger.info(f"\\nNext steps:")
    logger.info(f"1. Inspect the VAE architecture")
    logger.info(f"2. Adapt it to your VideoVAE interface")
    logger.info(f"3. Update config to use pretrained MAISI VAE")


if __name__ == "__main__":
    main()
