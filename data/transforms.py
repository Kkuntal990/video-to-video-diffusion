"""
Video transforms for preprocessing

Handles:
- Resizing
- Normalization
- Temporal sampling
- Data augmentation
"""

import torch
import numpy as np
import cv2
from torchvision import transforms as T
from einops import rearrange


class VideoTransform:
    """
    Transform for video data

    Input: numpy array of shape (T, H, W, 3) in [0, 255]
    Output: torch tensor of shape (3, T, H, W) in [-1, 1]
    """

    def __init__(self, resolution=(256, 256), num_frames=16, normalize=True):
        """
        Args:
            resolution: (height, width) to resize to
            num_frames: number of frames to sample
            normalize: whether to normalize to [-1, 1]
        """
        self.resolution = resolution
        self.num_frames = num_frames
        self.normalize = normalize

    def __call__(self, frames):
        """
        Apply transforms to video frames

        Args:
            frames: numpy array (T, H, W, 3) in [0, 255]

        Returns:
            video: torch tensor (3, T, H, W) in [-1, 1]
        """
        # Ensure we have the right number of frames
        if len(frames) > self.num_frames:
            # Uniformly sample frames
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = frames[indices]
        elif len(frames) < self.num_frames:
            # Pad with last frame
            pad_length = self.num_frames - len(frames)
            last_frame = frames[-1:].repeat(pad_length, axis=0)
            frames = np.concatenate([frames, last_frame], axis=0)

        # Resize frames
        resized_frames = []
        for frame in frames:
            # OpenCV resize (faster than PIL)
            resized = cv2.resize(frame, self.resolution[::-1], interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized)

        frames = np.stack(resized_frames)  # (T, H, W, 3)

        # Convert to torch tensor
        video = torch.from_numpy(frames).float()  # (T, H, W, 3)

        # Rearrange to (C, T, H, W)
        video = rearrange(video, 't h w c -> c t h w')

        # Normalize to [-1, 1]
        if self.normalize:
            video = video / 127.5 - 1.0

        return video


class VideoAugmentation:
    """
    Data augmentation for videos

    Applies:
    - Random horizontal flip
    - Random brightness/contrast
    - Random temporal crop
    """

    def __init__(self, flip_prob=0.5, brightness_range=0.2, contrast_range=0.2):
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, video):
        """
        Apply augmentation to video

        Args:
            video: torch tensor (C, T, H, W)

        Returns:
            video: augmented torch tensor (C, T, H, W)
        """
        # Random horizontal flip
        if torch.rand(1) < self.flip_prob:
            video = torch.flip(video, dims=[-1])

        # Random brightness
        if self.brightness_range > 0:
            brightness_factor = 1.0 + torch.rand(1).item() * self.brightness_range * 2 - self.brightness_range
            video = video * brightness_factor
            video = torch.clamp(video, -1, 1)

        # Random contrast
        if self.contrast_range > 0:
            contrast_factor = 1.0 + torch.rand(1).item() * self.contrast_range * 2 - self.contrast_range
            mean = video.mean(dim=[2, 3], keepdim=True)
            video = (video - mean) * contrast_factor + mean
            video = torch.clamp(video, -1, 1)

        return video


class DenormalizeVideo:
    """
    Denormalize video from [-1, 1] to [0, 255]

    Used for visualization
    """

    def __call__(self, video):
        """
        Args:
            video: torch tensor (C, T, H, W) or (B, C, T, H, W) in [-1, 1]

        Returns:
            video: torch tensor in [0, 255]
        """
        video = (video + 1.0) * 127.5
        video = torch.clamp(video, 0, 255)
        return video.to(torch.uint8)


def video_to_numpy(video):
    """
    Convert video tensor to numpy array for saving/visualization

    Args:
        video: torch tensor (C, T, H, W) or (B, C, T, H, W)

    Returns:
        numpy array (T, H, W, C) or (B, T, H, W, C)
    """
    # Handle batch dimension
    if video.dim() == 5:
        video = rearrange(video, 'b c t h w -> b t h w c')
    else:
        video = rearrange(video, 'c t h w -> t h w c')

    # Convert to numpy
    video = video.cpu().numpy()

    return video


def save_video(video, path, fps=8):
    """
    Save video tensor to file

    Args:
        video: torch tensor (C, T, H, W) in [-1, 1] or [0, 255]
        path: output path (e.g., 'output.mp4')
        fps: frames per second
    """
    import imageio

    # Denormalize if needed
    if video.dtype != torch.uint8:
        if video.min() < 0:  # Likely in [-1, 1]
            denorm = DenormalizeVideo()
            video = denorm(video)

    # Convert to numpy
    frames = video_to_numpy(video)  # (T, H, W, C)

    # Save using imageio
    imageio.mimsave(path, frames, fps=fps)
    print(f"Video saved to {path}")


def load_video(path, num_frames=None):
    """
    Load video from file

    Args:
        path: video file path
        num_frames: number of frames to load (None = all)

    Returns:
        frames: numpy array (T, H, W, 3)
    """
    import av

    container = av.open(path)
    frames = []

    for i, frame in enumerate(container.decode(video=0)):
        if num_frames is not None and i >= num_frames:
            break
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)

    container.close()

    return np.stack(frames)


if __name__ == "__main__":
    # Test transforms
    print("Testing video transforms...")

    # Create dummy video (T, H, W, C) in [0, 255]
    frames = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
    print(f"Input frames shape: {frames.shape}")
    print(f"Input range: [{frames.min()}, {frames.max()}]")

    # Test basic transform
    transform = VideoTransform(resolution=(256, 256), num_frames=16)
    video = transform(frames)
    print(f"\nTransformed video shape: {video.shape}")
    print(f"Transformed range: [{video.min():.2f}, {video.max():.2f}]")

    # Test augmentation
    aug = VideoAugmentation(flip_prob=1.0, brightness_range=0.2)
    video_aug = aug(video)
    print(f"\nAugmented video shape: {video_aug.shape}")
    print(f"Augmented range: [{video_aug.min():.2f}, {video_aug.max():.2f}]")

    # Test denormalization
    denorm = DenormalizeVideo()
    video_denorm = denorm(video)
    print(f"\nDenormalized video shape: {video_denorm.shape}")
    print(f"Denormalized range: [{video_denorm.min()}, {video_denorm.max()}]")
    print(f"Denormalized dtype: {video_denorm.dtype}")

    # Test video_to_numpy
    video_np = video_to_numpy(video_denorm)
    print(f"\nNumpy video shape: {video_np.shape}")

    # Test with batch
    video_batch = torch.stack([video, video])  # (B, C, T, H, W)
    video_batch_np = video_to_numpy(video_batch)
    print(f"Batch numpy shape: {video_batch_np.shape}")

    print("\nTransform test successful!")
