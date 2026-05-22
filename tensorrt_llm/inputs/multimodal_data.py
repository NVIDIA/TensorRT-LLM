# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image


class ContentHasher(Protocol):
    """Hash object that accepts bytes."""

    def update(self, data: bytes) -> None:
        """Update the hash with raw bytes."""


def serialize_item(obj: object) -> bytes:
    """Serialize a supported multimodal hash leaf to bytes."""
    if isinstance(obj, str):
        return obj.encode("utf-8")
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, (int, float)):
        return np.array(obj).tobytes()

    if isinstance(obj, Image.Image):
        return np.array(obj.convert("RGBA")).tobytes()
    if isinstance(obj, torch.Tensor):
        return obj.numpy().tobytes()
    if isinstance(obj, np.ndarray):
        return obj.tobytes()
    if isinstance(obj, (tuple, list)):
        container_tag = b"T" if isinstance(obj, tuple) else b"L"
        parts = [container_tag, len(obj).to_bytes(8, "big", signed=False)]
        for item in obj:
            payload = serialize_item(item)
            parts.append(len(payload).to_bytes(8, "big", signed=False))
            parts.append(payload)
        return b"".join(parts)

    raise ValueError(f"Unsupported object type: {type(obj)}")


class BaseModalityData:
    """Base class for modality-specific data."""

    def update_hash(self, hasher: ContentHasher) -> None:
        """Update a content hash with this modality payload."""
        raise NotImplementedError(f"{type(self).__name__} must implement update_hash()")


@dataclass
class AudioData(BaseModalityData):
    """Structured audio payload."""

    samples: np.ndarray | torch.Tensor
    sample_rate: int

    def __post_init__(self) -> None:
        if not isinstance(self.samples, (np.ndarray, torch.Tensor)):
            raise TypeError("samples must be a NumPy array or PyTorch tensor")
        if not isinstance(self.sample_rate, int):
            self.sample_rate = int(self.sample_rate)

    def update_hash(self, hasher: ContentHasher) -> None:
        samples = self.samples
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().contiguous()
        hasher.update(b"<audio>")
        hasher.update(serialize_item((samples, self.sample_rate)))


@dataclass
class VideoData(BaseModalityData):
    """Data class for video loading results.

    Attributes:
        frames: List of video frames, either as PIL Images or PyTorch tensors.
        metadata: Dictionary containing video metadata including:
            - total_num_frames: Total number of frames in the video
            - fps: Original frames per second of the video
            - duration: Duration of the video in seconds
            - frames_indices: List of indices of the sampled frames
        audio: Structured audio payload from the video, when extracted.
    """

    frames: list[Image.Image] | list[torch.Tensor]
    metadata: dict[str, Any]
    audio: AudioData | None = None

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("frames list cannot be empty")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

    def update_hash(self, hasher: ContentHasher) -> None:
        for frame in self.frames:
            hasher.update(b"<frame>")
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().contiguous()
            hasher.update(serialize_item(frame))
        # Extend this to include metadata if fields such as sampled frame
        # indices become part of the model-visible cache identity.
        if self.audio is not None:
            self.audio.update_hash(hasher)
