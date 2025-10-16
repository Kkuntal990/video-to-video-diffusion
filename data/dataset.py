"""
Video Dataset for Video-to-Video Diffusion

Loads video pairs (input, ground_truth) from various sources:
- HuggingFace datasets
- Local video files
- Video directories
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from pathlib import Path
from PIL import Image
import av
from typing import Optional, List, Tuple
from .transforms import VideoTransform


class VideoDataset(Dataset):
    """
    Video dataset for video-to-video tasks

    Expects data in one of these formats:
    1. HuggingFace dataset with 'input' and 'target' video columns
    2. Directory structure:
        data_dir/
            input/
                video1.mp4
                video2.mp4
            target/
                video1.mp4
                video2.mp4
    3. Paired video lists
    """

    def __init__(
        self,
        data_source,
        source_type='huggingface',  # 'huggingface', 'directory', or 'list'
        num_frames=16,
        resolution=(256, 256),
        stride=8,
        transform=None,
        split='train'
    ):
        """
        Args:
            data_source: HF dataset name, directory path, or list of video pairs
            source_type: type of data source
            num_frames: number of frames per clip
            resolution: (height, width) to resize videos
            stride: frame stride for sampling clips
            transform: additional transforms
            split: 'train', 'val', or 'test'
        """
        super().__init__()

        self.source_type = source_type
        self.num_frames = num_frames
        self.resolution = resolution
        self.stride = stride
        self.split = split

        # Default transform
        if transform is None:
            self.transform = VideoTransform(resolution=resolution, num_frames=num_frames)
        else:
            self.transform = transform

        # Load data based on source type
        if source_type == 'huggingface':
            self.video_pairs = self._load_huggingface(data_source, split)
        elif source_type == 'directory':
            self.video_pairs = self._load_directory(data_source)
        elif source_type == 'list':
            self.video_pairs = data_source
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        print(f"Loaded {len(self.video_pairs)} video pairs for {split}")

    def _load_huggingface(self, dataset_name, split):
        """Load video pairs from HuggingFace dataset"""
        from datasets import load_dataset

        try:
            dataset = load_dataset(dataset_name, split=split)
            video_pairs = []

            for item in dataset:
                # Assuming dataset has 'input' and 'target' columns
                # Adjust based on actual dataset structure
                if 'input' in item and 'target' in item:
                    video_pairs.append({
                        'input': item['input'],
                        'target': item['target']
                    })
                # Alternative: 'video_in' and 'video_out'
                elif 'video_in' in item and 'video_out' in item:
                    video_pairs.append({
                        'input': item['video_in'],
                        'target': item['video_out']
                    })

            return video_pairs

        except Exception as e:
            print(f"Error loading HuggingFace dataset: {e}")
            print("Falling back to empty dataset. Please provide data in the correct format.")
            return []

    def _load_directory(self, data_dir):
        """Load video pairs from directory structure"""
        data_dir = Path(data_dir)
        input_dir = data_dir / 'input'
        target_dir = data_dir / 'target'

        if not input_dir.exists() or not target_dir.exists():
            raise ValueError(f"Input/target directories not found in {data_dir}")

        # Get all video files
        input_videos = sorted(list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi')))
        target_videos = sorted(list(target_dir.glob('*.mp4')) + list(target_dir.glob('*.avi')))

        if len(input_videos) != len(target_videos):
            print(f"Warning: Mismatched number of videos: {len(input_videos)} inputs, {len(target_videos)} targets")

        video_pairs = []
        for input_path, target_path in zip(input_videos, target_videos):
            video_pairs.append({
                'input': str(input_path),
                'target': str(target_path)
            })

        return video_pairs

    def _load_video_frames(self, video_path, start_frame=0):
        """
        Load frames from video file

        Args:
            video_path: path to video or video data
            start_frame: starting frame index

        Returns:
            frames: numpy array of shape (T, H, W, 3)
        """
        # If video_path is already a numpy array or tensor, return it
        if isinstance(video_path, (np.ndarray, torch.Tensor)):
            return video_path

        # Load from file
        if isinstance(video_path, (str, Path)):
            video_path = str(video_path)

            # Use PyAV for video loading (more reliable than OpenCV)
            try:
                container = av.open(video_path)
                frames = []

                for i, frame in enumerate(container.decode(video=0)):
                    if i < start_frame:
                        continue
                    if len(frames) >= self.num_frames * self.stride:
                        break

                    if (i - start_frame) % self.stride == 0:
                        img = frame.to_ndarray(format='rgb24')
                        frames.append(img)

                container.close()

                if len(frames) < self.num_frames:
                    # Pad with last frame if not enough frames
                    while len(frames) < self.num_frames:
                        frames.append(frames[-1])

                return np.stack(frames[:self.num_frames])

            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                # Return dummy frames
                H, W = self.resolution
                return np.zeros((self.num_frames, H, W, 3), dtype=np.uint8)

        # Fallback: return dummy frames
        H, W = self.resolution
        return np.zeros((self.num_frames, H, W, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.video_pairs)

    def __getitem__(self, idx):
        """
        Get a video pair

        Returns:
            dict with:
                'input': input video tensor (3, T, H, W)
                'target': target video tensor (3, T, H, W)
        """
        pair = self.video_pairs[idx]

        # Load frames
        input_frames = self._load_video_frames(pair['input'])
        target_frames = self._load_video_frames(pair['target'])

        # Apply transforms
        input_video = self.transform(input_frames)
        target_video = self.transform(target_frames)

        return {
            'input': input_video,
            'target': target_video
        }


def collate_fn(batch):
    """Custom collate function for video batches"""
    inputs = torch.stack([item['input'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])

    return {
        'input': inputs,
        'target': targets
    }


def get_dataloader(config, split='train'):
    """
    Create dataloader from config

    Args:
        config: dataset config dict
        split: 'train', 'val', or 'test'

    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = VideoDataset(
        data_source=config['data_source'],
        source_type=config.get('source_type', 'huggingface'),
        num_frames=config.get('num_frames', 16),
        resolution=tuple(config.get('resolution', [256, 256])),
        stride=config.get('stride', 8),
        split=split
    )

    batch_size = config.get('batch_size', 2)
    num_workers = config.get('num_workers', 4)
    shuffle = (split == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


if __name__ == "__main__":
    # Test the dataset
    print("Testing video dataset...")

    # Example: Create dummy video directory for testing
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure
        input_dir = Path(tmpdir) / 'input'
        target_dir = Path(tmpdir) / 'target'
        input_dir.mkdir()
        target_dir.mkdir()

        # Create dummy videos (using images for simplicity)
        print(f"Created test directories in {tmpdir}")

        # Test with list of video pairs (using dummy data)
        video_pairs = [
            {'input': np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8),
             'target': np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8)}
            for _ in range(4)
        ]

        dataset = VideoDataset(
            data_source=video_pairs,
            source_type='list',
            num_frames=16,
            resolution=(256, 256)
        )

        print(f"Dataset size: {len(dataset)}")

        # Test __getitem__
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input shape: {sample['input'].shape}")
        print(f"Target shape: {sample['target'].shape}")
        print(f"Input range: [{sample['input'].min():.2f}, {sample['input'].max():.2f}]")

        # Test dataloader
        config = {
            'data_source': video_pairs,
            'source_type': 'list',
            'batch_size': 2,
            'num_workers': 0,
            'num_frames': 16,
            'resolution': [256, 256]
        }

        dataloader = get_dataloader(config, split='train')
        batch = next(iter(dataloader))

        print(f"\nBatch keys: {batch.keys()}")
        print(f"Batch input shape: {batch['input'].shape}")
        print(f"Batch target shape: {batch['target'].shape}")

    print("\nDataset test successful!")
