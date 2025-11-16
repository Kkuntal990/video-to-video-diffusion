"""
Inspect MAISI Checkpoint to Extract Exact Architecture Configuration

This script loads the MAISI VAE checkpoint and extracts the exact architecture
configuration needed for 100% weight loading compatibility.
"""

import torch
import json
from collections import defaultdict

def analyze_checkpoint_structure(checkpoint_path):
    """
    Analyze checkpoint structure to understand architecture

    Args:
        checkpoint_path: Path to MAISI autoencoder.pt file

    Returns:
        dict with architecture analysis
    """
    print("="*80)
    print("MAISI CHECKPOINT ARCHITECTURE ANALYSIS")
    print("="*80)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Check checkpoint structure
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

    # Get state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    print(f"\nTotal parameters in state_dict: {len(state_dict)}")

    # Analyze layer structure
    print("\n" + "="*80)
    print("LAYER STRUCTURE ANALYSIS")
    print("="*80)

    # Group by module
    encoder_layers = []
    decoder_layers = []
    quant_layers = []
    other_layers = []

    for key in state_dict.keys():
        if 'encoder' in key:
            encoder_layers.append(key)
        elif 'decoder' in key:
            decoder_layers.append(key)
        elif 'quant' in key or 'post_quant' in key:
            quant_layers.append(key)
        else:
            other_layers.append(key)

    print(f"\nEncoder layers: {len(encoder_layers)}")
    print(f"Decoder layers: {len(decoder_layers)}")
    print(f"Quantization layers: {len(quant_layers)}")
    print(f"Other layers: {len(other_layers)}")

    # Analyze encoder structure
    print("\n" + "="*80)
    print("ENCODER STRUCTURE")
    print("="*80)

    # Group encoder layers by block
    encoder_blocks = defaultdict(list)
    for key in encoder_layers:
        # Extract block structure
        parts = key.split('.')
        if len(parts) >= 2:
            block_name = '.'.join(parts[:2])
            encoder_blocks[block_name].append(key)

    print(f"\nEncoder has {len(encoder_blocks)} top-level blocks:")
    for block_name in sorted(encoder_blocks.keys()):
        print(f"  {block_name}: {len(encoder_blocks[block_name])} parameters")

    # Print first few layers of each block
    print("\nFirst layers in each encoder block:")
    for block_name in sorted(encoder_blocks.keys())[:5]:
        layers = sorted(encoder_blocks[block_name])[:3]
        for layer in layers:
            shape = state_dict[layer].shape
            print(f"  {layer}: {shape}")

    # Analyze decoder structure
    print("\n" + "="*80)
    print("DECODER STRUCTURE")
    print("="*80)

    decoder_blocks = defaultdict(list)
    for key in decoder_layers:
        parts = key.split('.')
        if len(parts) >= 2:
            block_name = '.'.join(parts[:2])
            decoder_blocks[block_name].append(key)

    print(f"\nDecoder has {len(decoder_blocks)} top-level blocks:")
    for block_name in sorted(decoder_blocks.keys()):
        print(f"  {block_name}: {len(decoder_blocks[block_name])} parameters")

    # Print first few layers of each block
    print("\nFirst layers in each decoder block:")
    for block_name in sorted(decoder_blocks.keys())[:5]:
        layers = sorted(decoder_blocks[block_name])[:3]
        for layer in layers:
            shape = state_dict[layer].shape
            print(f"  {layer}: {shape}")

    # Detect channel progression
    print("\n" + "="*80)
    print("CHANNEL PROGRESSION DETECTION")
    print("="*80)

    # Look for conv layers to detect channel progression
    conv_layers = [(k, v.shape) for k, v in state_dict.items() if 'conv' in k and len(v.shape) == 5]

    print(f"\nFound {len(conv_layers)} 3D conv layers")

    # Group by encoder/decoder and extract channels
    encoder_convs = [(k, s) for k, s in conv_layers if 'encoder' in k]
    decoder_convs = [(k, s) for k, s in conv_layers if 'decoder' in k]

    print(f"\nEncoder convolutions: {len(encoder_convs)}")
    if encoder_convs:
        print("First 10 encoder conv layers:")
        for key, shape in encoder_convs[:10]:
            print(f"  {key}: {shape} (in={shape[1]}, out={shape[0]})")

    print(f"\nDecoder convolutions: {len(decoder_convs)}")
    if decoder_convs:
        print("First 10 decoder conv layers:")
        for key, shape in decoder_convs[:10]:
            print(f"  {key}: {shape} (in={shape[1]}, out={shape[0]})")

    # Extract unique channel values
    encoder_channels = set()
    for key, shape in encoder_convs:
        encoder_channels.add(shape[0])  # out_channels
        encoder_channels.add(shape[1])  # in_channels

    decoder_channels = set()
    for key, shape in decoder_convs:
        decoder_channels.add(shape[0])
        decoder_channels.add(shape[1])

    print(f"\n Unique encoder channels: {sorted(encoder_channels)}")
    print(f"Unique decoder channels: {sorted(decoder_channels)}")

    # Detect latent channels
    print("\n" + "="*80)
    print("LATENT SPACE CONFIGURATION")
    print("="*80)

    quant_conv_layers = [(k, v.shape) for k, v in state_dict.items() if 'quant_conv' in k]
    post_quant_conv_layers = [(k, v.shape) for k, v in state_dict.items() if 'post_quant_conv' in k]

    print(f"\nQuantization conv layers:")
    for key, shape in quant_conv_layers:
        print(f"  {key}: {shape}")

    print(f"\nPost-quantization conv layers:")
    for key, shape in post_quant_conv_layers:
        print(f"  {key}: {shape}")

    # Extract latent_dim from quant_conv
    if quant_conv_layers:
        latent_dim = quant_conv_layers[0][1][0]  # output channels
        print(f"\n✓ Detected latent_dim: {latent_dim}")

    # Detect number of resolution levels
    print("\n" + "="*80)
    print("RESOLUTION LEVELS")
    print("="*80)

    # Count downsample/upsample blocks
    downsample_blocks = [k for k in encoder_blocks.keys() if 'down' in k.lower()]
    upsample_blocks = [k for k in decoder_blocks.keys() if 'up' in k.lower()]

    print(f"\nDownsample blocks: {len(downsample_blocks)}")
    for block in sorted(downsample_blocks):
        print(f"  {block}")

    print(f"\nUpsample blocks: {len(upsample_blocks)}")
    for block in sorted(upsample_blocks):
        print(f"  {block}")

    # Recommended configuration
    print("\n" + "="*80)
    print("RECOMMENDED AUTOENCODER_KL CONFIGURATION")
    print("="*80)

    # Infer configuration from structure
    all_channels = sorted(encoder_channels | decoder_channels)

    print("\nBased on analysis:")
    print(f"  spatial_dims: 3  # 3D volumetric")
    print(f"  in_channels: 1   # Grayscale CT")

    if quant_conv_layers:
        latent_channels = quant_conv_layers[0][1][0]
        print(f"  latent_channels: {latent_channels}")

    if all_channels:
        # Estimate channel multipliers
        base_ch = min(ch for ch in all_channels if ch > 1)
        channel_mult = tuple(sorted(set(ch // base_ch for ch in all_channels if ch >= base_ch)))
        print(f"  channels: ({', '.join(map(str, sorted(set(ch for ch in all_channels if ch >= base_ch))))})")
        print(f"  channel_mult: {channel_mult}")

    # Count residual blocks per level
    # Look for patterns like encoder.down.0.block.0, encoder.down.0.block.1, etc.
    res_block_pattern = defaultdict(set)
    for key in encoder_layers:
        if '.block.' in key:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'block' and i+1 < len(parts):
                    block_idx = parts[i+1]
                    level = parts[i-1] if i > 0 else '0'
                    res_block_pattern[level].add(block_idx)

    if res_block_pattern:
        num_res_blocks = max(len(blocks) for blocks in res_block_pattern.values())
        print(f"  num_res_blocks: {num_res_blocks}")

    # Save full key list for debugging
    print("\n" + "="*80)
    print("Saving full parameter list to maisi_checkpoint_keys.txt")
    print("="*80)

    with open('maisi_checkpoint_keys.txt', 'w') as f:
        f.write("MAISI Checkpoint Parameter Keys\n")
        f.write("="*80 + "\n\n")
        for key in sorted(state_dict.keys()):
            shape = state_dict[key].shape
            f.write(f"{key}: {shape}\n")

    print("✓ Saved to maisi_checkpoint_keys.txt")

    return {
        'total_params': len(state_dict),
        'encoder_layers': len(encoder_layers),
        'decoder_layers': len(decoder_layers),
        'encoder_channels': sorted(encoder_channels),
        'decoder_channels': sorted(decoder_channels),
        'encoder_blocks': list(encoder_blocks.keys()),
        'decoder_blocks': list(decoder_blocks.keys()),
    }


if __name__ == "__main__":
    checkpoint_path = './pretrained/maisi_vae/models/autoencoder.pt'

    try:
        analysis = analyze_checkpoint_structure(checkpoint_path)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nTotal parameters: {analysis['total_params']}")
        print(f"Encoder layers: {analysis['encoder_layers']}")
        print(f"Decoder layers: {analysis['decoder_layers']}")

    except Exception as e:
        print(f"\nError analyzing checkpoint: {e}")
        import traceback
        traceback.print_exc()
