# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-to-device uploads of the input preparation.

:func:`stage_or_copy` stages a host-to-device copy in the engine's open batched input commit of the
step when there is one (``pyexecutor/input_commit.py``, ``TRTLLM_BATCHED_INPUT_COMMIT=1``) and copies
directly otherwise; :func:`stage_scatter` and :func:`flush_input_commit` are its scatter and flush.
"""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch.pyexecutor.input_commit import active_input_commit

# Destination dtypes the engine's input commit kernel stages.
_STAGEABLE = (torch.int32, torch.int64, torch.bool, torch.float32, torch.uint8, torch.int8)


def stage_or_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst.copy_(src, non_blocking=True)`` of a host ``src``: staged in the open input commit of
    the step when there is one (written by its next commit kernel), else copied directly."""
    commit = active_input_commit()
    if (
        commit is not None
        and commit.is_open
        and src.device.type == "cpu"
        and dst.is_cuda
        and dst.is_contiguous()
        and dst.numel() == src.numel()
        and dst.dtype in _STAGEABLE
    ):
        commit.stage(dst, src)
        return
    dst.copy_(src, non_blocking=True)


def stage_scatter(target: torch.Tensor, slots: Sequence[int], values: Sequence[int]) -> bool:
    """``target[slots[i]] = values[i]`` for a 1-D CUDA ``target`` and distinct ``slots``, staged in
    the open input commit with one entry per slot; False (nothing staged) without an open commit."""
    commit = active_input_commit()
    if (
        commit is None
        or not commit.is_open
        or not target.is_cuda
        or target.dim() != 1
        or not target.is_contiguous()
        or target.dtype not in _STAGEABLE
    ):
        return False
    for slot, value in zip(slots, values):
        commit.stage(target[slot : slot + 1], [value])
    return True


def flush_input_commit() -> None:
    """Write every copy staged so far in this step (one kernel) and keep staging: for a device
    operation of the input preparation that reads a staged destination."""
    commit = active_input_commit()
    if commit is not None and commit.is_open:
        commit.commit()
        commit.begin()
