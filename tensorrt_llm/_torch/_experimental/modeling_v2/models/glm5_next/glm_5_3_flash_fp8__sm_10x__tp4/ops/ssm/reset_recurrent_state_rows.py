# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero the recurrent-state rows of requests that start fresh.

One program per (request, 256-element block of the row), masked by the request's flag and slot. Modeling
V1's ``reset_recurrent_state_rows`` (``tensorrt_llm/_torch/modules/mamba/recurrent_state_cache.py``)
launches the same kernel, which can also zero the conv-state rows; the target's fused prefill convolution
clears fresh conv rows itself, so that path is not part of this entry. This entry does not import V1.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Elements of a row one program zeroes.
_BLOCK_SIZE = 256


@triton.jit
def _reset_recurrent_state_rows_kernel(
    recurrent_states,
    state_indices,
    has_initial_states,
    recurrent_state_stride,
    NUM_CACHE_LINES: tl.constexpr,
    RECURRENT_STATE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    request_idx = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    state_idx = tl.load(state_indices + request_idx).to(tl.int64)
    needs_reset = ~tl.load(has_initial_states + request_idx).to(tl.int1)
    valid_state = (state_idx >= 0) & (state_idx < NUM_CACHE_LINES)
    recurrent_row_offset = state_idx * recurrent_state_stride.to(tl.int64)
    tl.store(
        recurrent_states + recurrent_row_offset + offsets,
        0.0,
        mask=needs_reset & valid_state & (offsets < RECURRENT_STATE_SIZE),
    )


class ResetRecurrentStateRows(OpWrapper):
    """Zero slot ``state_indices[i]``'s row of ``recurrent_states`` for every request i with
    ``has_initial_states[i]`` false; one launch, returns None.

    Args:
        recurrent_states: ``[slots, ...]`` pool (KDA: ``[slots, H, V, K]`` bf16 or fp32), each slot's row
            contiguous; the slot stride is free (the cache manager's per-slot records are wider than one row).
        state_indices: int32 or int64 ``[N]`` slot of each request. Indices outside ``[0, slots)`` are skipped.
        has_initial_states: bool ``[N]``; a request with True keeps its row.
    """

    ARCHS = frozenset({Arch.SM_100})
    # GLM-5.3-Flash's KDA at TP4 (16 of 64 heads, K = V = 128), on pools whose slot stride is a whole per-slot
    # record, as the hybrid cache manager lays them out. The shapes the shipped target resets are in this entry's
    # test.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="fp32 pool: out-of-range and negative slots skipped",
            spec=dict(
                slots=10,
                indices=(3, -1, 9, 10, 0),
                has_init=(False, False, True, False, False),
                state_dtype=torch.float32,
            ),
        ),
        Cell(
            why="int64 slot indices, every request fresh, rows larger than one 256-element program",
            spec=dict(
                slots=6,
                indices=(5, 1, 2),
                has_init=(False, False, False),
                state_dtype=torch.bfloat16,
                index_dtype=torch.int64,
            ),
        ),
    )
    note = """
    The flag is read per request: True keeps the slot's row, False zeroes it. A
    slot index outside [0, slots) (the -1 of a padding request, or a slot past the
    pool) is skipped rather than rejected. Zeroing is a masked store of 0.0, so any
    pool dtype works; only the elements of the slot's row are written, never the
    padding between rows.

    KDA prefill calls this before the recurrence: the prefill kernel reads every
    selected slot as its initial state, so a fresh request's slot must be zeroed
    first.

    The launch is ordered on the current stream and takes no host sync, so it is
    CUDA-graph capturable once compiled; the row size and slot count are
    compile-time constants, so the first call per pool geometry compiles.
    """

    def __call__(
        self,
        recurrent_states: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
    ) -> None:
        num_requests = state_indices.shape[0]
        if num_requests == 0:
            return
        recurrent_state_size = recurrent_states.numel() // recurrent_states.shape[0]
        grid = (num_requests, triton.cdiv(recurrent_state_size, _BLOCK_SIZE))
        _reset_recurrent_state_rows_kernel[grid](
            recurrent_states,
            state_indices,
            has_initial_states,
            recurrent_states.stride(0),
            recurrent_states.shape[0],
            recurrent_state_size,
            _BLOCK_SIZE,
        )

    def reference(
        self,
        recurrent_states: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
    ) -> torch.Tensor:
        """``recurrent_states`` as the call leaves it; mutates nothing. Exact: the op writes zeros or nothing."""
        recurrent = recurrent_states.clone()
        for slot, keep in zip(state_indices.tolist(), has_initial_states.tolist()):
            if keep or not 0 <= slot < recurrent.shape[0]:
                continue
            recurrent[slot] = 0
        return recurrent

    def is_valid(
        self,
        recurrent_states: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
    ) -> None:
        # The kernel zeroes numel / slots elements from slot * stride(0) and checks nothing.
        assert recurrent_states.shape[0] > 0, "the recurrent pool must have at least one slot"
        assert recurrent_states[0].is_contiguous(), (
            "each recurrent-state row must be contiguous: the kernel zeroes one dense run per slot"
        )
        assert recurrent_states.stride(0) >= recurrent_states[0].numel(), (
            "recurrent-state rows must not overlap: zeroing one would clear part of the next"
        )
        assert state_indices.ndim == 1, "state_indices must be one slot per request"
        assert has_initial_states.shape == state_indices.shape, (
            "has_initial_states must hold one flag per request: the kernel reads it by request index"
        )


reset_recurrent_state_rows = ResetRecurrentStateRows()
