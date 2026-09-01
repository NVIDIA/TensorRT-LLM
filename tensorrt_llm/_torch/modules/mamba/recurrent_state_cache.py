# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared indexed state-cache operations for recurrent attention modules."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _reset_recurrent_state_rows_kernel(
    recurrent_states,
    conv_states,
    state_indices,
    has_initial_states,
    recurrent_state_stride,
    conv_state_stride,
    NUM_CACHE_LINES: tl.constexpr,
    RECURRENT_STATE_SIZE: tl.constexpr,
    CONV_STATE_SIZE: tl.constexpr,
    RESET_CONV: tl.constexpr,
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
    if RESET_CONV:
        conv_row_offset = state_idx * conv_state_stride.to(tl.int64)
        tl.store(
            conv_states + conv_row_offset + offsets,
            0.0,
            mask=needs_reset & valid_state & (offsets < CONV_STATE_SIZE),
        )


def reset_recurrent_state_rows(
    recurrent_states: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_states: torch.Tensor,
    conv_states: Optional[torch.Tensor] = None,
) -> None:
    """Zero fresh-request rows in recurrent and optional convolution pools.

    Invalid or negative slot indices are ignored, matching the cache-manager
    reset contract used by GDN and KDA. The launch is ordered on PyTorch's
    current stream, so a following indexed recurrence observes the zeroed rows.
    """
    num_requests = state_indices.shape[0]
    if num_requests == 0:
        return
    if recurrent_states.shape[0] == 0:
        raise ValueError("recurrent state pool must have at least one cache line")
    if conv_states is not None and conv_states.shape[0] != recurrent_states.shape[0]:
        raise ValueError("recurrent and convolution pools must have the same slot count")
    if has_initial_states.shape != state_indices.shape:
        raise ValueError("has_initial_states and state_indices must have matching shapes")
    recurrent_state_size = recurrent_states.numel() // recurrent_states.shape[0]
    conv_state_size = conv_states.numel() // conv_states.shape[0] if conv_states is not None else 0
    if not recurrent_states[0].is_contiguous():
        raise ValueError("recurrent state rows must be contiguous")
    if recurrent_states.stride(0) < recurrent_state_size:
        raise ValueError("recurrent state rows must not overlap")
    if conv_states is not None and not conv_states[0].is_contiguous():
        raise ValueError("convolution state rows must be contiguous")
    if conv_states is not None and conv_states.stride(0) < conv_state_size:
        raise ValueError("convolution state rows must not overlap")

    block_size = 256
    grid = (
        num_requests,
        triton.cdiv(max(recurrent_state_size, conv_state_size), block_size),
    )
    _reset_recurrent_state_rows_kernel[grid](
        recurrent_states,
        conv_states if conv_states is not None else recurrent_states,
        state_indices,
        has_initial_states,
        recurrent_states.stride(0),
        conv_states.stride(0) if conv_states is not None else 0,
        recurrent_states.shape[0],
        recurrent_state_size,
        conv_state_size,
        conv_states is not None,
        block_size,
    )
