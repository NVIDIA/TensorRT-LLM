# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precomputed multimodal embeddings passed through ``multi_modal_embeddings``.

Each entry of ``multi_modal_embeddings[modality]`` is either a bare tensor of
shape ``[rows, width]`` or a dict holding the tensor under ``"mm_embeddings"``
plus the geometry and timing metadata that the raw-media path would have
derived from the image or video (grid, timestamps, ...). This module splits
the container, checks the tensor contract, the item keys and the types of
their values; models check the values against their own geometry and timing.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from .multimodal import split_multimodal_embedding_items

EXTERNAL_EMBEDDING_MODALITIES = ("image", "video")

_EMBEDDING_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

_ITEM_KEYS = frozenset(
    {
        "image_grid_thw",
        "video_grid_thw",
        "timestamps",
        "fps",
        "frames_indices",
        "second_per_grid_ts",
    }
)


@dataclass(frozen=True)
class ExternalEmbeddingItem:
    """One precomputed image or video embedding with its source metadata."""

    modality: str
    embedding: torch.Tensor
    """Embedding rows in prompt order, shape ``[rows, width]``."""
    grid_thw: Optional[tuple[int, int, int]] = None
    """Pre-merge patch grid ``(t, h, w)`` as the vision processor reports it."""
    timestamps: Optional[tuple[float, ...]] = None
    """Timestamp in seconds of each temporal patch, for models that print frame
    timestamps. Compute them in float64 as the HF processor does."""
    fps: Optional[float] = None
    """Source video frames per second, read with ``frames_indices`` (Qwen3-VL)."""
    frames_indices: Optional[tuple[int, ...]] = None
    """Source frame index of each sampled frame (Qwen3-VL)."""
    second_per_grid_ts: Optional[float] = None
    """Seconds covered by one temporal patch, for models that scale MRoPE time."""

    @property
    def num_rows(self) -> int:
        """Number of embedding rows, which is the number of prompt placeholders."""
        return int(self.embedding.shape[0])


def _as_int_tuple(value: Any, name: str, max_len: Optional[int] = None) -> tuple[int, ...]:
    """Convert a list, tuple or tensor of integers to a tuple of Python ints."""
    if isinstance(value, torch.Tensor):
        if value.dtype.is_floating_point or value.dtype.is_complex:
            raise ValueError(f"{name} must hold integers, got {value.dtype}")
        if max_len is not None and value.numel() > max_len:
            raise ValueError(f"{name} has {value.numel()} entries, at most {max_len}")
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list, tuple or tensor of integers")
    result = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ValueError(f"{name} must hold integers, got {type(entry).__name__}")
        result.append(int(entry))
    return tuple(result)


def _as_float_tuple(value: Any, name: str) -> tuple[float, ...]:
    """Convert a list, tuple or tensor of numbers to a tuple of finite floats.

    Floating tensors must be float64: a value computed in lower precision can
    print to different digits than the float64 value HF computes.
    """
    if isinstance(value, torch.Tensor):
        if value.dtype.is_complex or (
            value.dtype.is_floating_point and value.dtype != torch.float64
        ):
            raise ValueError(
                f"{name} must be float64 or integer, got {value.dtype}; compute the "
                "values in float64 (.double() or .tolist() of a lower-precision tensor "
                "does not restore them)"
            )
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list, tuple or tensor of numbers")
    result = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise ValueError(f"{name} must hold numbers, got {type(entry).__name__}")
        entry = float(entry)
        if not math.isfinite(entry):
            raise ValueError(f"{name} must hold finite numbers")
        result.append(entry)
    return tuple(result)


def _as_positive_float(value: Any, name: str) -> float:
    """Convert a scalar to a finite positive float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


def _check_embedding_tensor(embedding: Any, name: str) -> torch.Tensor:
    """Check the dense floating rank-2 tensor contract of an attached embedding."""
    if not isinstance(embedding, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(embedding).__name__}")
    if embedding.layout != torch.strided:
        raise ValueError(f"{name} must be a dense tensor, got layout {embedding.layout}")
    if embedding.device.type not in ("cpu", "cuda"):
        raise ValueError(f"{name} must be on cpu or cuda, got {embedding.device.type}")
    # A broadcast view (stride 0) claims more rows than its storage holds.
    if embedding.untyped_storage().nbytes() < embedding.numel() * embedding.element_size():
        raise ValueError(f"{name} must not be a self-overlapping view; pass a contiguous copy")
    # Batching concatenates items before casting to the model dtype, and
    # float8 does not promote with other dtypes.
    if embedding.dtype not in _EMBEDDING_DTYPES:
        raise ValueError(
            f"{name} must be a float16, bfloat16, float32 or float64 tensor, got {embedding.dtype}"
        )
    if embedding.dim() != 2:
        raise ValueError(f"{name} must be rank 2, got shape {tuple(embedding.shape)}")
    if embedding.shape[0] <= 0 or embedding.shape[1] <= 0:
        raise ValueError(f"{name} must be non-empty, got shape {tuple(embedding.shape)}")
    return embedding


def _parse_item(
    modality: str, index: int, embedding: Any, metadata: dict[str, Any], temporal_patch_size: int
) -> ExternalEmbeddingItem:
    """Validate one item, as split_multimodal_embedding_items splits it."""
    name = f"{modality} embedding {index}"
    unknown = set(metadata) - _ITEM_KEYS
    if unknown:
        raise ValueError(f"{name} has unknown keys {sorted(unknown)}")
    # The cache key hashes the metadata as passed, and None has no hash.
    absent = sorted(key for key, value in metadata.items() if value is None)
    if absent:
        raise ValueError(f"{name}: omit {absent} instead of passing None")
    grid_key = f"{modality}_grid_thw"
    other_grid_key = "video_grid_thw" if modality == "image" else "image_grid_thw"
    if other_grid_key in metadata:
        raise ValueError(f"{name} must use '{grid_key}', not '{other_grid_key}'")
    if modality == "image":
        for key in ("timestamps", "fps", "frames_indices", "second_per_grid_ts"):
            if key in metadata:
                raise ValueError(f"{name}: '{key}' only applies to video")

    grid = None
    if grid_key in metadata:
        grid = _as_int_tuple(metadata[grid_key], grid_key, max_len=3)
        if len(grid) != 3:
            raise ValueError(f"{name}: {grid_key} must have 3 entries (t, h, w), got {grid}")
        if any(dim <= 0 for dim in grid):
            raise ValueError(f"{name}: {grid_key} entries must be positive, got {grid}")

    # Bound per-frame lists by the grid before copying them: one timestamp per
    # temporal patch, at most temporal_patch_size sampled frames per patch.
    for key, per_patch in (("timestamps", 1), ("frames_indices", temporal_patch_size)):
        value = metadata.get(key)
        if isinstance(value, (torch.Tensor, list, tuple)):
            if grid is None:
                raise ValueError(f"{name}: '{key}' requires '{grid_key}'")
            length = value.numel() if isinstance(value, torch.Tensor) else len(value)
            if length > grid[0] * per_patch:
                raise ValueError(
                    f"{name}: {key} has {length} entries for {grid[0]} temporal patches"
                )

    def optional(field: str, convert: Callable[[Any, str], Any]) -> Any:
        """Convert ``metadata[field]`` if present."""
        return convert(metadata[field], field) if field in metadata else None

    return ExternalEmbeddingItem(
        modality,
        _check_embedding_tensor(embedding, name),
        grid_thw=grid,
        timestamps=optional("timestamps", _as_float_tuple),
        fps=optional("fps", _as_positive_float),
        frames_indices=optional("frames_indices", _as_int_tuple),
        second_per_grid_ts=optional("second_per_grid_ts", _as_positive_float),
    )


def parse_external_embeddings(
    multimodal_embedding: Any,
    temporal_patch_size: int = 1,
) -> dict[str, list[ExternalEmbeddingItem]]:
    """Normalize ``multi_modal_embeddings`` to per-modality item lists.

    Args:
        multimodal_embedding: Mapping of modality to a tensor, a dict entry, or
            a list of them, in the order of that modality's prompt placeholders.
            Each embedding is a dense ``[rows, width]`` float16, bfloat16,
            float32 or float64 tensor; the model checks ``rows`` against
            ``<modality>_grid_thw`` and ``width`` against its own row width
            (the text hidden size, times 1 + deepstack levels for Qwen3-VL).
        temporal_patch_size: Video frames per temporal patch, bounding ``frames_indices``.

    Returns:
        Modality to items, in the order given.
    """
    if not isinstance(multimodal_embedding, dict):
        raise ValueError("multimodal_embedding must be a dictionary")
    unsupported = set(multimodal_embedding) - set(EXTERNAL_EMBEDDING_MODALITIES)
    if unsupported:
        raise ValueError(
            f"Unsupported modalities {sorted(unsupported)} for external multimodal "
            f"embeddings; supported: {list(EXTERNAL_EMBEDDING_MODALITIES)}"
        )
    items: dict[str, list[ExternalEmbeddingItem]] = {}
    for modality, entries in multimodal_embedding.items():
        if not isinstance(entries, (torch.Tensor, dict, list, tuple)):
            raise TypeError(
                f"multi_modal_embeddings['{modality}'] must be a tensor, dict, or a list of them"
            )
        embeddings, metadata = split_multimodal_embedding_items(entries)
        items[modality] = [
            _parse_item(modality, i, embedding, item_metadata, temporal_patch_size)
            for i, (embedding, item_metadata) in enumerate(zip(embeddings, metadata))
        ]
    if not any(items.values()):
        raise ValueError("At least one multimodal embedding is required")
    return items
