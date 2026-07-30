# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for native Wan VAE temporal decode batching."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan.parallel_vae import (
    TRTLLM_WAN_VAE_DECODE_CHUNK_SIZE,
    _native_decode_chunk_size,
)
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import _decode_chunk_slices


@pytest.mark.parametrize(
    ("num_frames", "chunk_size", "expected"),
    [
        (0, 3, []),
        (1, 4, [(0, 1)]),
        (5, 1, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        (5, 2, [(0, 1), (1, 3), (3, 5)]),
        (6, 4, [(0, 1), (1, 5), (5, 6)]),
    ],
)
def test_decode_chunk_slices_preserve_first_frame(
    num_frames: int,
    chunk_size: int,
    expected: list[tuple[int, int]],
) -> None:
    actual = [(chunk.start, chunk.stop) for chunk in _decode_chunk_slices(num_frames, chunk_size)]
    assert actual == expected


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_decode_chunk_slices_reject_invalid_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        _decode_chunk_slices(num_frames=5, chunk_size=chunk_size)


@pytest.mark.parametrize(
    ("parallel_size", "dtype", "expected"),
    [
        (1, torch.bfloat16, 1),
        (1, torch.float32, 1),
        (2, torch.bfloat16, 2),
        (4, torch.bfloat16, 4),
        (4, torch.float32, 2),
        (8, torch.bfloat16, 2),
    ],
)
def test_native_decode_chunk_size_uses_tuned_or_conservative_value(
    monkeypatch,
    parallel_size: int,
    dtype: torch.dtype,
    expected: int,
) -> None:
    monkeypatch.delenv(TRTLLM_WAN_VAE_DECODE_CHUNK_SIZE, raising=False)
    assert _native_decode_chunk_size(parallel_size, dtype) == expected


def test_native_decode_chunk_size_honors_env_override(monkeypatch):
    monkeypatch.setenv(TRTLLM_WAN_VAE_DECODE_CHUNK_SIZE, "5")
    assert _native_decode_chunk_size(1, torch.bfloat16) == 5


@pytest.mark.parametrize("override", ["0", "-1", "invalid"])
def test_native_decode_chunk_size_rejects_invalid_env_override(
    monkeypatch,
    override: str,
) -> None:
    monkeypatch.setenv(TRTLLM_WAN_VAE_DECODE_CHUNK_SIZE, override)
    with pytest.raises(ValueError, match="must be a positive integer"):
        _native_decode_chunk_size(4, torch.bfloat16)
