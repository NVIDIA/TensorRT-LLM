# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import struct
from dataclasses import dataclass
from typing import Any, Callable, Protocol

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
    parts: list[Any] = []
    _update_serialized_item(parts.append, obj)
    return b"".join(parts)


def _update_serialized_item(update: Callable[[Any], None], obj: object) -> None:
    """Stream ``serialize_item(obj)`` into ``hasher`` without a payload copy.

    Large image and tensor payloads are passed through the buffer protocol.
    The emitted byte sequence is identical to :func:`serialize_item`, so this
    changes only hashing cost and temporary memory, not cache identity.
    """
    if isinstance(obj, str):
        payload = obj.encode("utf-8")
        update(_u8(0x01))
        update(_u64(len(payload)))
        update(payload)
        return

    if isinstance(obj, bytes):
        update(_u8(0x02))
        update(_u64(len(obj)))
        update(obj)
        return

    # bool must be checked before int: bool is a subclass of int.
    if isinstance(obj, bool):
        update(_u8(0x05))
        update(_u8(1 if obj else 0))
        return

    if isinstance(obj, int):
        nbytes = (obj.bit_length() + 8) // 8  # +1 sign bit, then ceil-divide.
        update(_u8(0x03))
        update(_u8(nbytes))
        update(obj.to_bytes(nbytes, "big", signed=True))
        return

    if isinstance(obj, float):
        update(_u8(0x04))
        update(struct.pack(">d", obj))
        return

    if isinstance(obj, Image.Image):
        width, height = obj.size
        rgba = np.asarray(obj.convert("RGBA"))
        update(_u8(0x10))
        update(_len_prefixed(obj.mode.encode("utf-8")))
        update(_u32(width))
        update(_u32(height))
        update(_u64(rgba.nbytes))
        update(memoryview(rgba).cast("B"))
        return

    if isinstance(obj, (torch.Tensor, np.ndarray)):
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().contiguous().numpy()
        array = np.ascontiguousarray(obj)
        update(_u8(0x11))
        update(_len_prefixed(array.dtype.str.encode("utf-8")))
        update(_u8(array.ndim))
        for dim in array.shape:
            update(_u64(dim))
        update(_u64(array.nbytes))
        update(memoryview(array).cast("B"))
        return

    if isinstance(obj, (tuple, list)):
        update(_u8(0x20))
        update(_u64(len(obj)))
        for item in obj:
            _update_serialized_item(update, item)
        return

    if isinstance(obj, dict):
        update(_u8(0x22))
        update(_u64(len(obj)))
        for key in sorted(obj):
            _update_serialized_item(update, key)
            _update_serialized_item(update, obj[key])
        return

    if isinstance(obj, np.generic):
        _update_serialized_item(update, obj.item())
        return

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
        frames: Video frames as a list of PIL Images, a list of PyTorch
            tensors, or a single 4D numpy array of shape (N, H, W, 3).
        metadata: Dictionary containing video metadata including:
            - total_num_frames: Total number of frames in the video
            - fps: Original frames per second of the video
            - duration: Duration of the video in seconds
            - frames_indices: List of indices of the sampled frames
        audio: Structured audio payload from the video, when extracted.
        raw_bytes_hash: BLAKE3 hex digest of the source video bytes when
            available. Populated by media loaders that hold the source
            (e.g. `VideoMediaIO`); `None` when the `VideoData` is
            constructed from a frame list directly. When set, it acts as
            the source anchor in `update_hash` — combined with the
            sampling metadata, it uniquely identifies the decoded frame
            content without walking pixels.
    """

    frames: list[Image.Image] | list[torch.Tensor] | np.ndarray
    metadata: dict[str, Any]
    audio: AudioData | None = None
    raw_bytes_hash: str | None = None

    def __post_init__(self) -> None:
        if len(self.frames) == 0:
            raise ValueError("frames cannot be empty")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

    def update_hash(self, hasher: ContentHasher) -> None:
        hasher.update(b"<video>")
        # Sampling metadata is part of the model-visible cache identity.
        # `frames_indices` is a deterministic function of source bytes plus
        # media IO `num_frames`/`fps` kwargs, so per-request kwarg
        # overrides land here implicitly.
        meta = {k: self.metadata[k] for k in _VIDEO_HASH_METADATA_FIELDS if k in self.metadata}
        hasher.update(serialize_item(meta))
        if self.raw_bytes_hash is not None:
            # Source anchor: metadata + source digest is equivalent to
            # hashing decoded frames, since cv2 decoding is deterministic
            # given source bytes and sampling is captured in `meta`.
            hasher.update(b"<raw_bytes>")
            hasher.update(self.raw_bytes_hash.encode("utf-8"))
        else:
            # No source anchor available (frame list constructed directly).
            for frame in self.frames:
                hasher.update(b"<frame>")
                hasher.update(serialize_item(frame))
        if self.audio is not None:
            self.audio.update_hash(hasher)
