"""Utility functions for visual generation pipelines."""

import torch


@torch.compile
def postprocess_video_tensor(video: torch.Tensor) -> torch.Tensor:
    """Post-process video tensor from VAE decoder output to final format.

    This is a more efficient implementation than using VideoProcessor for single-batch cases,
    as it avoids loop overhead and processes the entire batch with vectorized operations.

    Args:
        video: Video tensor in (B, C, T, H, W) format from VAE decoder

    Returns:
        Post-processed video tensor in (B, T, H, W, C) uint8 format.

    Note:
        Assumes video values are in [-1, 1] range (standard VAE decoder output).
    """
    # Convert to (B, T, H, W, C) format
    video = video.permute(0, 2, 3, 4, 1)  # (B, C, T, H, W) -> (B, T, H, W, C)

    # Normalize to [0, 1] range
    video = (video / 2 + 0.5).clamp(0, 1)

    # Convert to uint8
    video = (video * 255).round().to(torch.uint8)

    return video


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)
