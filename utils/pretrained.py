"""
Pretrained Weight Loading and Conversion Utilities

Provides utilities for:
- Loading pretrained VAE weights (Open-Sora, HunyuanVideo, Stable Diffusion)
- Converting 2D Stable Diffusion weights to 3D video weights
- Weight format conversion (safetensors <-> PyTorch)
- Layer mapping and inspection
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


def inflate_conv2d_to_conv3d(conv2d_weight, method='central'):
    """
    Inflate 2D conv weights to 3D conv weights

    Args:
        conv2d_weight: (out_channels, in_channels, kernel_h, kernel_w)
        method: 'central', 'replicate', or 'average'

    Returns:
        conv3d_weight: (out_channels, in_channels, kernel_t, kernel_h, kernel_w)
    """
    if len(conv2d_weight.shape) != 4:
        raise ValueError(f"Expected 4D tensor, got {conv2d_weight.shape}")

    out_c, in_c, k_h, k_w = conv2d_weight.shape

    if method == 'central':
        # Central initialization: put all weight in center temporal position
        # Common approach for video models (AnimateDiff, etc.)
        k_t = 3  # Standard temporal kernel size
        conv3d_weight = torch.zeros(out_c, in_c, k_t, k_h, k_w)
        center_t = k_t // 2
        conv3d_weight[:, :, center_t, :, :] = conv2d_weight

    elif method == 'replicate':
        # Replicate weights across temporal dimension
        k_t = 3
        conv3d_weight = conv2d_weight.unsqueeze(2).repeat(1, 1, k_t, 1, 1) / k_t

    elif method == 'average':
        # Average initialization
        k_t = 3
        conv3d_weight = conv2d_weight.unsqueeze(2).repeat(1, 1, k_t, 1, 1) / k_t

    else:
        raise ValueError(f"Unknown inflation method: {method}")

    return conv3d_weight


def load_state_dict_from_file(path: Union[str, Path]) -> Dict:
    """
    Load state dict from file (supports .pt, .pth, .safetensors)

    Args:
        path: Path to checkpoint file

    Returns:
        state_dict: Model state dictionary
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix == '.safetensors':
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(path))
        except ImportError:
            raise ImportError("safetensors library required. Install with: pip install safetensors")
    elif path.suffix in ['.pt', '.pth', '.ckpt']:
        state_dict = torch.load(str(path), map_location='cpu')
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return state_dict


def save_state_dict_to_file(state_dict: Dict, path: Union[str, Path], format='pt'):
    """
    Save state dict to file

    Args:
        state_dict: Model state dictionary
        path: Save path
        format: 'pt', 'pth', or 'safetensors'
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'safetensors':
        try:
            from safetensors.torch import save_file
            save_file(state_dict, str(path))
        except ImportError:
            raise ImportError("safetensors library required. Install with: pip install safetensors")
    elif format in ['pt', 'pth']:
        torch.save(state_dict, str(path))
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved state dict to {path}")


def map_sd_vae_to_video_vae(sd_state_dict: Dict, inflate_method='central') -> Dict:
    """
    Map Stable Diffusion 2D VAE weights to 3D Video VAE weights

    Inflates 2D convolutions to 3D and maps parameter names.

    Args:
        sd_state_dict: Stable Diffusion VAE state dict
        inflate_method: Method for inflating 2D->3D convs

    Returns:
        video_vae_state_dict: Mapped 3D VAE state dict
    """
    video_state_dict = {}

    for name, param in sd_state_dict.items():
        # Check if this is a conv layer that needs inflation
        if 'conv' in name.lower() and len(param.shape) == 4:
            # This is a 2D conv weight, inflate to 3D
            inflated = inflate_conv2d_to_conv3d(param, method=inflate_method)
            video_state_dict[name] = inflated
            logger.debug(f"Inflated {name}: {param.shape} -> {inflated.shape}")
        else:
            # Keep as is (biases, norms, etc.)
            video_state_dict[name] = param

    return video_state_dict


def load_pretrained_sd_vae(model_name='stabilityai/sd-vae-ft-mse',
                           cache_dir=None,
                           subfolder='') -> Dict:
    """
    Load pretrained Stable Diffusion VAE from HuggingFace

    Args:
        model_name: HF model name
        cache_dir: Cache directory
        subfolder: Subfolder within model repo

    Returns:
        state_dict: VAE state dictionary
    """
    try:
        from diffusers import AutoencoderKL
    except ImportError:
        raise ImportError("diffusers library required. Install with: pip install diffusers")

    logger.info(f"Loading Stable Diffusion VAE from {model_name}...")

    # Load using diffusers
    vae = AutoencoderKL.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        subfolder=subfolder
    )

    return vae.state_dict()


def load_pretrained_opensora_vae(model_name='hpcai-tech/OpenSora-VAE-v1.2',
                                 cache_dir=None) -> Dict:
    """
    Load pretrained Open-Sora VAE from HuggingFace

    Args:
        model_name: HF model name
        cache_dir: Cache directory

    Returns:
        state_dict: VAE state dictionary
    """
    try:
        from diffusers import AutoencoderKL
    except ImportError:
        raise ImportError("diffusers library required. Install with: pip install diffusers")

    logger.info(f"Loading Open-Sora VAE from {model_name}...")

    # Load using diffusers
    vae = AutoencoderKL.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )

    return vae.state_dict()


def load_pretrained_cogvideox_vae(model_name='THUDM/CogVideoX-5b',
                                  cache_dir=None) -> Dict:
    """
    Load pretrained CogVideoX VAE from HuggingFace

    Args:
        model_name: HF model name
        cache_dir: Cache directory

    Returns:
        state_dict: VAE state dictionary
    """
    try:
        from diffusers import AutoencoderKLCogVideoX
    except ImportError:
        raise ImportError("diffusers library required. Install with: pip install diffusers")

    logger.info(f"Loading CogVideoX VAE from {model_name}...")

    # Load using diffusers
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_name,
        subfolder="vae",
        cache_dir=cache_dir,
        torch_dtype=torch.float32
    )

    return vae.state_dict()


def inspect_checkpoint(path: Union[str, Path]):
    """
    Inspect checkpoint file and print information

    Args:
        path: Path to checkpoint file
    """
    state_dict = load_state_dict_from_file(path)

    print(f"\n{'='*80}")
    print(f"Checkpoint: {path}")
    print(f"{'='*80}")
    print(f"Total parameters: {len(state_dict)}")
    print(f"\nParameter shapes:")
    print(f"{'-'*80}")

    total_params = 0
    for name, param in sorted(state_dict.items()):
        param_count = param.numel()
        total_params += param_count
        print(f"{name:60s} {str(param.shape):20s} {param_count:>15,}")

    print(f"{'-'*80}")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"{'='*80}\n")


def filter_state_dict_by_prefix(state_dict: Dict, prefix: str, remove_prefix=True) -> Dict:
    """
    Filter state dict by parameter name prefix

    Args:
        state_dict: Full state dict
        prefix: Prefix to filter by (e.g., 'encoder.', 'decoder.')
        remove_prefix: Whether to remove the prefix from keys

    Returns:
        filtered_dict: Filtered state dictionary
    """
    filtered = {}

    for name, param in state_dict.items():
        if name.startswith(prefix):
            new_name = name[len(prefix):] if remove_prefix else name
            filtered[new_name] = param

    return filtered


if __name__ == "__main__":
    # Test utilities
    logging.basicConfig(level=logging.INFO)

    print("Testing pretrained weight utilities...")

    # Test 2D->3D inflation
    conv2d_weight = torch.randn(64, 32, 3, 3)
    print(f"\n2D Conv weight shape: {conv2d_weight.shape}")

    conv3d_weight = inflate_conv2d_to_conv3d(conv2d_weight, method='central')
    print(f"3D Conv weight shape (central): {conv3d_weight.shape}")

    conv3d_weight = inflate_conv2d_to_conv3d(conv2d_weight, method='replicate')
    print(f"3D Conv weight shape (replicate): {conv3d_weight.shape}")

    print("\nPretrained weight utilities test complete!")
