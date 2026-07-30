# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for native Wan VAE temporal decode batching."""

import pytest

from tensorrt_llm._torch.visual_gen.models.wan.parallel_vae import _native_decode_chunk_size
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
    ("parallel_size", "expected"),
    [
        (1, 1),
        (2, 1),
        (4, 4),
        (8, 1),
    ],
)
def test_native_decode_chunk_size_uses_validated_parallel_case(
    parallel_size: int,
    expected: int,
) -> None:
    assert _native_decode_chunk_size(parallel_size) == expected
