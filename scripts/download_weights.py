"""
Download Pretrained Weights

Automated script to download pretrained model weights from HuggingFace.
Supports:
- Open-Sora VAE v1.2
- HunyuanVideo VAE
- CogVideoX VAE
- Stable Diffusion VAE (various versions)
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.pretrained import (
    load_pretrained_opensora_vae,
    load_pretrained_cogvideox_vae,
    load_pretrained_sd_vae,
    save_state_dict_to_file
)


AVAILABLE_MODELS = {
    'opensora': {
        'name': 'hpcai-tech/OpenSora-VAE-v1.2',
        'description': 'Open-Sora VAE v1.2 (3D, well-documented)',
        'loader': load_pretrained_opensora_vae
    },
    'cogvideox-2b': {
        'name': 'THUDM/CogVideoX-2b',
        'description': 'CogVideoX 2B VAE',
        'loader': load_pretrained_cogvideox_vae
    },
    'cogvideox-5b': {
        'name': 'THUDM/CogVideoX-5b',
        'description': 'CogVideoX 5B VAE',
        'loader': load_pretrained_cogvideox_vae
    },
    'sd-vae': {
        'name': 'stabilityai/sd-vae-ft-mse',
        'description': 'Stable Diffusion VAE (fine-tuned MSE, 2D)',
        'loader': load_pretrained_sd_vae
    },
    'sdxl-vae': {
        'name': 'stabilityai/sdxl-vae',
        'description': 'Stable Diffusion XL VAE (2D)',
        'loader': load_pretrained_sd_vae
    }
}


def download_model(model_key, output_dir='pretrained_weights', cache_dir=None):
    """
    Download pretrained model weights

    Args:
        model_key: Model key from AVAILABLE_MODELS
        output_dir: Directory to save weights
        cache_dir: HuggingFace cache directory
    """
    if model_key not in AVAILABLE_MODELS:
        print(f"Error: Unknown model '{model_key}'")
        print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        return False

    model_info = AVAILABLE_MODELS[model_key]
    model_name = model_info['name']
    loader_func = model_info['loader']

    print(f"\n{'='*80}")
    print(f"Downloading: {model_info['description']}")
    print(f"Model: {model_name}")
    print(f"{'='*80}\n")

    try:
        # Load model weights
        if model_key.startswith('cogvideox'):
            state_dict = loader_func(model_name, cache_dir=cache_dir)
        elif model_key.startswith('sd'):
            state_dict = loader_func(model_name, cache_dir=cache_dir)
        else:
            state_dict = loader_func(model_name, cache_dir=cache_dir)

        # Save to local directory
        output_path = Path(output_dir) / f"{model_key}_vae.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_state_dict_to_file(state_dict, output_path, format='pt')

        print(f"\n✓ Successfully downloaded and saved to: {output_path}")
        print(f"  Total parameters: {sum(p.numel() for p in state_dict.values()):,}")

        return True

    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        return False


def list_available_models():
    """Print available models"""
    print(f"\n{'='*80}")
    print("Available Pretrained Models")
    print(f"{'='*80}\n")

    for key, info in AVAILABLE_MODELS.items():
        print(f"  {key:20s} - {info['description']}")
        print(f"  {'':20s}   {info['name']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Download pretrained weights for Video Diffusion Model'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=list(AVAILABLE_MODELS.keys()) + ['all'],
        help='Model to download (or "all" for all models)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='pretrained_weights',
        help='Output directory for weights (default: pretrained_weights)'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='HuggingFace cache directory'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models'
    )

    args = parser.parse_args()

    # List models
    if args.list or args.model is None:
        list_available_models()
        return

    # Download model(s)
    if args.model == 'all':
        print("Downloading all available models...")
        for model_key in AVAILABLE_MODELS.keys():
            success = download_model(model_key, args.output_dir, args.cache_dir)
            if not success:
                print(f"Failed to download {model_key}, continuing...")
    else:
        download_model(args.model, args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
