# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import struct
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image

# Video metadata fields that participate in the cache-key hash. These describe
# how frames were sampled and therefore change the model-visible content.
_VIDEO_HASH_METADATA_FIELDS = (
    "frames_indices",
    "fps",
    "duration",
    "total_num_frames",
)


class ContentHasher(Protocol):
    """Hash object that accepts bytes."""

    def update(self, data: bytes) -> None:
        """Update the hash with raw bytes."""


def _u8(value: int) -> bytes:
    """Encode an unsigned 8-bit integer."""
    return value.to_bytes(1, "big", signed=False)


def _u32(value: int) -> bytes:
    """Encode an unsigned 32-bit big-endian integer."""
    return value.to_bytes(4, "big", signed=False)


def _u64(value: int) -> bytes:
    """Encode an unsigned 64-bit big-endian integer."""
    return value.to_bytes(8, "big", signed=False)


def _len_prefixed(payload: bytes) -> bytes:
    """Encode a byte payload prefixed with its u64 length."""
    return _u64(len(payload)) + payload


def serialize_item(obj: object) -> bytes:
    """Serialize a supported multimodal hash leaf to bytes.

    The encoding is canonical and self-describing: every value is
    `[1-byte type tag][typed metadata][length-prefixed payload]` with all
    multi-byte integers big-endian. This prevents cache-key hash collisions
    between distinct values that happen to share a raw byte payload (for
    example transposed image dimensions or reshaped arrays).
    """
    if isinstance(obj, str):
        return _u8(0x01) + _len_prefixed(obj.encode("utf-8"))
    if isinstance(obj, bytes):
        return _u8(0x02) + _len_prefixed(obj)
    # bool must be checked before int: bool is a subclass of int.
    if isinstance(obj, bool):
        return _u8(0x05) + _u8(1 if obj else 0)
    if isinstance(obj, int):
        nbytes = (obj.bit_length() + 8) // 8  # +1 sign bit, then ceil-divide.
        return _u8(0x03) + _u8(nbytes) + obj.to_bytes(nbytes, "big", signed=True)
    if isinstance(obj, float):
        return _u8(0x04) + struct.pack(">d", obj)

    if isinstance(obj, Image.Image):
        width, height = obj.size
        payload = np.array(obj.convert("RGBA")).tobytes()
        return (
            _u8(0x10)
            + _len_prefixed(obj.mode.encode("utf-8"))
            + _u32(width)
            + _u32(height)
            + _len_prefixed(payload)
        )
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        # The container (torch.Tensor vs np.ndarray) is not part of the content
        # identity -- only dtype, shape, and raw bytes are. Normalize both to a
        # contiguous NumPy array so identical content hashes identically.
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().contiguous().numpy()
        array = np.ascontiguousarray(obj)
        parts = [
            _u8(0x11),
            _len_prefixed(array.dtype.str.encode("utf-8")),
            _u8(array.ndim),
        ]
        parts.extend(_u64(dim) for dim in array.shape)
        parts.append(_len_prefixed(array.tobytes()))
        return b"".join(parts)
    if isinstance(obj, (tuple, list)):
        # Ordered sequence; the container (tuple vs list) is not part of the
        # content identity.
        parts = [_u8(0x20), _u64(len(obj))]
        parts.extend(serialize_item(item) for item in obj)
        return b"".join(parts)
    if isinstance(obj, dict):
        parts = [_u8(0x22), _u64(len(obj))]
        for key in sorted(obj):
            parts.append(serialize_item(key))
            parts.append(serialize_item(obj[key]))
        return b"".join(parts)

    if isinstance(obj, np.generic):
        # numpy scalar (e.g. np.int64 / np.float32 / np.bool_): normalize to the
        # equivalent Python scalar and recurse, so numpy-typed values hash
        # identically to their Python counterparts. In numpy 2.x these are not
        # subclasses of Python int/float/bool, so they bypass the checks above.
        return serialize_item(obj.item())

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
        hasher.update(b"<audio>")
        hasher.update(serialize_item((self.samples, self.sample_rate)))


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
        hasher.update(b"<video>")
        # Sampling metadata is part of the model-visible cache identity.
        meta = {k: self.metadata[k] for k in _VIDEO_HASH_METADATA_FIELDS if k in self.metadata}
        hasher.update(serialize_item(meta))
        for frame in self.frames:
            hasher.update(b"<frame>")
            hasher.update(serialize_item(frame))
        if self.audio is not None:
            self.audio.update_hash(hasher)
