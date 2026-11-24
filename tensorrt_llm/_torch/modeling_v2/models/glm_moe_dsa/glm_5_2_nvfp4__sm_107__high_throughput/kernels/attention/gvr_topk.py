# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rubin decode policy for balanced and static hint-free GVR kernels."""

from functools import lru_cache

import torch

from tensorrt_llm.logger import logger

_SPLITS = {
    (40, 8): 4,
    (48, 8): 4,
    (48, 6): 4,
}
_BALANCED_SHAPES = {(rows, 6) for rows in range(54, 108, 6)} | {(56, 8)}
_ENVELOPE = 1048576
_WORKSPACE_ELEMENTS = (2048 + 160 * 16384 * 8) // 4
_workspaces: dict[int, torch.Tensor] = {}
_launchers: dict[tuple[int, int, int, int], tuple] = {}


@lru_cache(maxsize=None)
def _supported_device(device: int) -> bool:
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor, props.multi_processor_count) == (10, 7, 212)


def supports(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
    next_n: int,
    max_seq_len: int | None,
) -> bool:
    return (
        scores.is_cuda
        and _supported_device(scores.get_device())
        and scores.ndim == 2
        and ((scores.shape[0], next_n) in _SPLITS or (scores.shape[0], next_n) in _BALANCED_SHAPES)
        and scores.dtype == torch.float32
        and max_seq_len is not None
        and _ENVELOPE <= max_seq_len <= _ENVELOPE + 256
        and scores.stride() == (((max_seq_len + 255) // 256) * 256, 1)
        and scores.shape[1] <= max_seq_len
        and scores.data_ptr() % 16 == 0
        and lengths.device == scores.device
        and lengths.dtype == torch.int32
        and lengths.shape == (scores.shape[0] // next_n,)
        and lengths.is_contiguous()
        and indices.device == scores.device
        and indices.dtype == torch.int32
        and indices.shape == (scores.shape[0], 2048)
        and indices.is_contiguous()
        and indices.data_ptr() % 16 == 0
        and scores.untyped_storage().nbytes()
        >= (scores.storage_offset() + scores.shape[0] * scores.stride(0)) * 4
    )


def workspace(scores: torch.Tensor) -> torch.Tensor:
    """Keep the eager warmup slab alive across capture streams and graph pools."""
    key = scores.get_device()
    buffer = _workspaces.get(key)
    if buffer is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm Rubin GVR workspace before CUDA graph capture")
        buffer = torch.zeros(_WORKSPACE_ELEMENTS, dtype=torch.int32, device=scores.device)
        _workspaces[key] = buffer
    return buffer


def run(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
    buffer: torch.Tensor,
    next_n: int,
) -> None:
    stride = scores.stride(0)
    key = (scores.get_device(), scores.shape[0], next_n, stride)
    launcher = _launchers.get(key)
    if launcher is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm Rubin GVR TopK before CUDA graph capture")
        rows = scores.shape[0]
        if (rows, next_n) in _BALANCED_SHAPES:
            from .gvr_balanced import get_compiled

            fn = get_compiled(
                (1024, 8, 1, 256, 2, True, False, next_n, 0, 3, (rows + 31) // 32),
                hf=True,
            )
            pre = (0, stride, 2048, 8192, 4096, 212, 0, 0, 0, 0, 0)
            tail = (4096, 48, 7168, 0, 0)
            policy = "balanced, CTAs=212"
        else:
            from tensorrt_llm._torch.kernels.blackwell.top_k.gvr_topk_decode_self_sampling import (
                get_compiled,
            )

            split = _SPLITS[(rows, next_n)]
            assert rows * split <= 212 and split * 2048 <= 16384
            fn = get_compiled(
                (1024, 8, 1, 256, 2, True, False, next_n, 0, split),
                hint_free=True,
                fused_k=2048,
            )
            # Match the shared varlen launcher's tail parameters for this grid.
            big = rows * split <= 148
            pre = (0, stride, 2048, 8192, 4096, split, 0, 0, 0, 0, 0)
            tail = (
                4096 if big else 2816,
                32 if split == 2 else 48,
                6144 if split == 2 else 7168,
                0 if big else 1,
                0,
            )
            policy = f"R={split}, CTAs={rows * split}"
        launcher = (fn, pre, tail)
        _launchers[key] = launcher
        logger.info(f"Rubin GVR TopK: rows={rows}, next_n={next_n}, {policy}, stride={stride}")
    fn, pre, tail = launcher
    logits = scores.as_strided((scores.shape[0], stride), (stride, 1))
    # The hint-free engine does not read the legacy pre_idx ABI slot.
    fn(logits, indices, indices, buffer, *pre, lengths, *tail)
