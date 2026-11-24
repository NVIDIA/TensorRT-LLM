# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of Inkling 2's width-4 causal short convolutions, with a per-request window state.

One step holds the new tokens of several requests, request-major (`query_start`). Token `t` of a
request reads its own input and the 3 inputs before it: inputs of the same step from `x`, earlier
ones from the request's window state. A request that starts at position 0 has no earlier inputs
(`has_state` 0), which reads as zeros.

`conv_out` computes, in fp32, per channel `c`:

    out[t, c] = (sum over lag 3..0 of tap(t, c, lag) * x[t - lag, c] + x[t, c]) * out_scale

with `tap = taps[lag, c]` (static), or `taps[lag, c] + dynamic_taps[t, c // channels_per_head * 4
+ lag]` (dynamic), and rounds once to the output dtype. `update_state` then stores the last 3
inputs of each request, oldest first, in its slot of `state` `[slots, 3, C]`. The state is read by
the first kernel and written by the second, so the two must run in this order on one stream.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Kernel width of every Inkling 2 short convolution.
KERNEL_SIZE = 4


@triton.jit
def _conv_out_kernel(
    X,
    TAPS,
    DYN,
    STATE,
    SLOTS,
    QSTART,
    HAS_STATE,
    TOKEN_REQ,
    OUT,
    num_tokens,
    channels,
    stride_x,
    stride_dyn,
    stride_state_slot,
    stride_state_win,
    stride_out,
    out_scale,
    CHANNELS_PER_HEAD: tl.constexpr,
    DYNAMIC: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_t = offs_t < num_tokens
    mask_c = offs_c < channels
    mask = mask_t[:, None] & mask_c[None, :]

    req = tl.load(TOKEN_REQ + offs_t, mask=mask_t, other=0)
    local = offs_t - tl.load(QSTART + req, mask=mask_t, other=0)
    has_state = tl.load(HAS_STATE + req, mask=mask_t, other=0) != 0
    slot = tl.load(SLOTS + req, mask=mask_t, other=0).to(tl.int64)
    rows = offs_t.to(tl.int64)

    x0 = tl.load(X + rows[:, None] * stride_x + offs_c[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    acc = tl.zeros([BLOCK_T, BLOCK_C], dtype=tl.float32)
    # Oldest input first, the order of a convolution over the window.
    for lag in tl.static_range(3, -1, -1):
        if lag == 0:
            xv = x0
        else:
            src = local - lag
            in_step = src >= 0
            xv_step = tl.load(
                X + (rows - lag)[:, None] * stride_x + offs_c[None, :],
                mask=mask & in_step[:, None],
                other=0.0,
            )
            # Window state index of an input `src` (< 0) steps before the request's first new token.
            win = 3 + src
            from_state = (~in_step) & has_state & (win >= 0)
            xv_state = tl.load(
                STATE
                + slot[:, None] * stride_state_slot
                + win[:, None] * stride_state_win
                + offs_c[None, :],
                mask=mask & from_state[:, None],
                other=0.0,
            )
            xv = tl.where(in_step[:, None], xv_step.to(tl.float32), xv_state.to(tl.float32))
        tap = tl.load(TAPS + lag * channels + offs_c, mask=mask_c, other=0.0)[None, :]
        if DYNAMIC:
            head = offs_c // CHANNELS_PER_HEAD
            dyn = tl.load(
                DYN + rows[:, None] * stride_dyn + head[None, :] * 4 + lag, mask=mask, other=0.0
            ).to(tl.float32)
            tap = tap + dyn
        acc += tap * xv
    out = (acc + x0) * out_scale
    tl.store(
        OUT + rows[:, None] * stride_out + offs_c[None, :], out.to(OUT.dtype.element_ty), mask=mask
    )


@triton.jit
def _update_state_kernel(
    X,
    STATE,
    SLOTS,
    QSTART,
    HAS_STATE,
    channels,
    stride_x,
    stride_state_slot,
    stride_state_win,
    BLOCK_C: tl.constexpr,
):
    req = tl.program_id(0)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < channels
    start = tl.load(QSTART + req)
    n = tl.load(QSTART + req + 1) - start
    has_state = tl.load(HAS_STATE + req) != 0
    base = STATE + tl.load(SLOTS + req).to(tl.int64) * stride_state_slot

    # New window entry j is the input at local index n - 3 + j: from this step, or from the old
    # window (index n + j) when the request appended fewer than 3 tokens. All three are read
    # before any is written.
    v0 = _window_entry(X, base, start, n, 0, has_state, offs_c, mask_c, stride_x, stride_state_win)
    v1 = _window_entry(X, base, start, n, 1, has_state, offs_c, mask_c, stride_x, stride_state_win)
    v2 = _window_entry(X, base, start, n, 2, has_state, offs_c, mask_c, stride_x, stride_state_win)
    tl.store(base + offs_c, v0, mask=mask_c)
    tl.store(base + stride_state_win + offs_c, v1, mask=mask_c)
    tl.store(base + 2 * stride_state_win + offs_c, v2, mask=mask_c)


@triton.jit
def _window_entry(
    X, base, start, n, j: tl.constexpr, has_state, offs_c, mask_c, stride_x, stride_state_win
):
    src = n - 3 + j
    from_step = src >= 0
    v_step = tl.load(
        X + (start + src).to(tl.int64) * stride_x + offs_c, mask=mask_c & from_step, other=0.0
    )
    v_state = tl.load(
        base + (n + j) * stride_state_win + offs_c,
        mask=mask_c & (~from_step) & has_state,
        other=0.0,
    )
    return tl.where(from_step, v_step.to(base.dtype.element_ty), v_state)


def short_conv(
    x: torch.Tensor,
    taps: torch.Tensor,
    dynamic_taps: torch.Tensor | None,
    state: torch.Tensor,
    slots: torch.Tensor,
    query_start: torch.Tensor,
    has_state: torch.Tensor,
    token_request: torch.Tensor,
    out_scale: float,
    channels_per_head: int = 1,
) -> torch.Tensor:
    """Launch both kernels; see the module docstring. Returns `out` `[T, C]` in `x`'s dtype."""
    num_tokens, channels = x.shape
    out = torch.empty_like(x)
    if num_tokens == 0:
        return out
    block_t, block_c = 16, 128
    dynamic = dynamic_taps is not None
    _conv_out_kernel[(triton.cdiv(num_tokens, block_t), triton.cdiv(channels, block_c))](
        x,
        taps,
        dynamic_taps if dynamic else x,
        state,
        slots,
        query_start,
        has_state,
        token_request,
        out,
        num_tokens,
        channels,
        x.stride(0),
        dynamic_taps.stride(0) if dynamic else 0,
        state.stride(0),
        state.stride(1),
        out.stride(0),
        float(out_scale),
        CHANNELS_PER_HEAD=channels_per_head,
        DYNAMIC=dynamic,
        BLOCK_T=block_t,
        BLOCK_C=block_c,
    )
    num_requests = slots.shape[0]
    _update_state_kernel[(num_requests, triton.cdiv(channels, block_c))](
        x,
        state,
        slots,
        query_start,
        has_state,
        channels,
        x.stride(0),
        state.stride(0),
        state.stride(1),
        BLOCK_C=block_c,
    )
    return out
