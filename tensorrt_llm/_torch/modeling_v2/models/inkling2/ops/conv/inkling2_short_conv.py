# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling 2's width-4 causal short convolutions over a step of several requests.

Every decoder layer runs two static convolutions on its full-width hidden states: one after the
attention output projection and one after the MoE. A sliding-window layer also convolves its key
and value projections (static); a global layer does so with dynamic taps, the static taps plus
per-token taps projected from the layer's key or value input, shared by the channels of one KV
head. Both forms add the input back and scale the sum::

    static:  out = (sum over lags of taps[lag] * x[t - lag] + x[t]) / sqrt(5)
    dynamic: out = (sum over lags of (taps[lag] + dyn[t, head, lag]) * x[t - lag] + x[t]) / 3

in fp32, rounded once. `taps` is lag-major, `[4, C]` fp32: `taps[0]` weights the current token.
A checkpoint's `nn.Conv1d` weight `[C, 1, 4]` holds the same taps oldest first
(`taps[lag] = weight[:, 0, 3 - lag]`); a dynamic convolution's `static_w` `[4, C]` is already
lag-major.

The 3 inputs before a request's first new token come from its window state `[slots, 3, C]`
(oldest first) when `has_state` is set, and are zeros otherwise. The call stores each request's
last 3 inputs back into its slot.
"""

from __future__ import annotations

import math

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.short_conv import KERNEL_SIZE, short_conv

#: The static convolution's output scale, 1 / sqrt(kernel size + 1).
STATIC_SCALE = 1.0 / math.sqrt(KERNEL_SIZE + 1)
#: The dynamic convolution's output scale, 1 / sqrt(2 * kernel size + 1).
DYNAMIC_SCALE = 1.0 / math.sqrt(2 * KERNEL_SIZE + 1)

#: Request mixes the cells run. Each entry is (new tokens, cached tokens before the step).
_FRESH_PREFILL = ((17, 0),)
_DECODE = ((1, 40), (1, 3), (1, 1), (1, 1000))
_MIXED = ((5, 0), (2, 9), (1, 1), (1, 2), (33, 64))


def _reference(
    x, taps, dynamic_taps, state, slots, query_start, has_state, out_scale, channels_per_head
):
    num_channels = x.shape[1]
    out = torch.empty_like(x)
    bounds = query_start.tolist()
    updates = []
    for r, slot in enumerate(slots.tolist()):
        start, end = bounds[r], bounds[r + 1]
        n = end - start
        if int(has_state[r]):
            history = state[slot].float()
        else:
            history = torch.zeros(KERNEL_SIZE - 1, num_channels, device=x.device)
        seq = torch.cat([history, x[start:end].float()])
        acc = torch.zeros(n, num_channels, device=x.device)
        for lag in range(KERNEL_SIZE - 1, -1, -1):
            tap = taps[lag].float().expand(n, num_channels)
            if dynamic_taps is not None:
                per_head = dynamic_taps[start:end].float().view(n, -1, KERNEL_SIZE)[:, :, lag]
                tap = tap + per_head.repeat_interleave(channels_per_head, dim=1)
            first = KERNEL_SIZE - 1 - lag
            acc = acc + tap * seq[first : first + n]
        out[start:end] = ((acc + seq[KERNEL_SIZE - 1 :]) * out_scale).to(x.dtype)
        updates.append((slot, seq[-(KERNEL_SIZE - 1) :].to(state.dtype)))
    for slot, window in updates:
        state[slot] = window
    return out


def _check(
    x, taps, dynamic_taps, state, slots, query_start, has_state, token_request, channels_per_head
):
    num_tokens, num_channels = x.shape
    assert x.dim() == 2 and x.stride(1) == 1, "x must be [T, C] with contiguous channels"
    assert taps.shape == (KERNEL_SIZE, num_channels) and taps.dtype == torch.float32, (
        f"taps must be lag-major fp32 [{KERNEL_SIZE}, {num_channels}]"
    )
    assert taps.is_contiguous(), "taps must be contiguous"
    assert state.dim() == 3 and state.shape[1:] == (KERNEL_SIZE - 1, num_channels), (
        f"state must be [slots, {KERNEL_SIZE - 1}, {num_channels}]"
    )
    assert state.stride(2) == 1 and state.dtype == x.dtype, (
        "state must hold contiguous channels in x's dtype"
    )
    num_requests = slots.shape[0]
    metadata = (
        ("slots", slots),
        ("query_start", query_start),
        ("has_state", has_state),
        ("token_request", token_request),
    )
    for name, t in metadata:
        assert t.dtype == torch.int32 and t.dim() == 1, f"{name} must be a 1-D int32 tensor"
        # The kernels index request metadata at unit stride.
        assert t.is_contiguous(), f"{name} must be contiguous"
    assert query_start.shape[0] == num_requests + 1 and has_state.shape[0] == num_requests, (
        "query_start must be [R + 1] and has_state [R] for R = len(slots)"
    )
    bounds = query_start.tolist()
    assert bounds[0] == 0 and bounds[-1] == num_tokens, "query_start must run from 0 to T"
    assert all(b > a for a, b in zip(bounds, bounds[1:])), "every request must append tokens"
    slot_list = slots.tolist()
    # Two requests in one slot race: each reads the other's window as its own.
    assert len(set(slot_list)) == len(slot_list), "each request needs its own slot"
    assert all(0 <= s < state.shape[0] for s in slot_list), "slots must index state"
    expected = torch.repeat_interleave(
        torch.arange(num_requests, dtype=torch.int32), torch.tensor(bounds).diff()
    )
    assert token_request.dtype == torch.int32 and torch.equal(token_request.cpu(), expected), (
        "token_request must map each token to its request index"
    )
    if dynamic_taps is not None:
        assert num_channels % channels_per_head == 0, "channels must split into whole heads"
        heads = num_channels // channels_per_head
        assert dynamic_taps.shape == (num_tokens, heads * KERNEL_SIZE), (
            f"dynamic_taps must be [T, {heads} heads * {KERNEL_SIZE}]"
        )
        assert dynamic_taps.stride(1) == 1, "dynamic_taps must be row-contiguous"


class Inkling2ShortConv(OpWrapper):
    """Static short convolution: `x` `[T, C]` with lag-major fp32 `taps` `[4, C]`; returns `[T, C]`.

    `state` `[slots, 3, C]` (x's dtype) holds each request's last 3 inputs and is updated in
    place. `slots` `[R]`, `query_start` `[R + 1]`, `has_state` `[R]` and `token_request` `[T]`
    are int32 on the device; `token_request` maps each token to its request index.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="sliding-layer k/v at TP4 (256 channels), one fresh prefill",
            spec=dict(channels=256, requests=_FRESH_PREFILL),
        ),
        Cell(
            why="sliding-layer k/v at TP4, four decodes, one at position 1",
            spec=dict(channels=256, requests=_DECODE),
        ),
        Cell(
            why="hidden-width convolution (4608), a mixed step with 1- and 2-token appends",
            spec=dict(channels=4608, requests=_MIXED),
        ),
    )

    note = """
    The residual and the scale are part of the convolution, and the window holds the
    convolution's input, not its output. A request at position 1 or 2 reads a window that is
    partly the zeros before its first token, which only a decode right after a short prefill
    exercises; the decode cell has one.
    """

    def __call__(
        self,
        x: torch.Tensor,
        taps: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        query_start: torch.Tensor,
        has_state: torch.Tensor,
        token_request: torch.Tensor,
    ) -> torch.Tensor:
        return short_conv(
            x, taps, None, state, slots, query_start, has_state, token_request, STATIC_SCALE
        )

    def reference(
        self,
        x: torch.Tensor,
        taps: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        query_start: torch.Tensor,
        has_state: torch.Tensor,
        token_request: torch.Tensor,
    ) -> torch.Tensor:
        """Per request, the convolution over `[window or zeros, new inputs]` in fp32; updates
        `state` in place as the op does. `token_request` is not needed."""
        return _reference(x, taps, None, state, slots, query_start, has_state, STATIC_SCALE, 1)

    def is_valid(
        self,
        x: torch.Tensor,
        taps: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        query_start: torch.Tensor,
        has_state: torch.Tensor,
        token_request: torch.Tensor,
    ) -> None:
        _check(x, taps, None, state, slots, query_start, has_state, token_request, 1)


class Inkling2DynamicShortConv(OpWrapper):
    """Dynamic short convolution of a global layer's keys or values.

    As `Inkling2ShortConv`, with per-token taps `dynamic_taps` `[T, heads * 4]` (head-major, lag
    minor; the layer's `weight_proj` output) added to `taps` for the `channels_per_head` channels
    of each KV head, and the output scale 1/3.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="global-layer k/v at TP4 (2 heads of 128), one fresh prefill",
            spec=dict(channels=256, heads=2, requests=_FRESH_PREFILL),
        ),
        Cell(
            why="global-layer k/v at TP4, four decodes",
            spec=dict(channels=256, heads=2, requests=_DECODE),
        ),
        Cell(
            why="global-layer k/v at TP1 (8 heads), a mixed step",
            spec=dict(channels=1024, heads=8, requests=_MIXED),
        ),
    )

    note = """
    The per-token taps belong to the token being computed, not to the inputs it reads, and
    one head's taps cover exactly that head's channels. Taps shared by every head, or applied
    per input token, still give a plausible convolution; the cells use per-head random taps so
    either mistake shows.
    """

    def __call__(
        self,
        x: torch.Tensor,
        taps: torch.Tensor,
        dynamic_taps: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        query_start: torch.Tensor,
        has_state: torch.Tensor,
        token_request: torch.Tensor,
        channels_per_head: int,
    ) -> torch.Tensor:
        return short_conv(
            x,
            taps,
            dynamic_taps,
            state,
            slots,
            query_start,
            has_state,
            token_request,
            DYNAMIC_SCALE,
            channels_per_head,
        )

    def reference(
        self,
        x: torch.Tensor,
        taps: torch.Tensor,
        dynamic_taps: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        query_start: torch.Tensor,
        has_state: torch.Tensor,
        token_request: torch.Tensor,
        channels_per_head: int,
    ) -> torch.Tensor:
        """As `Inkling2ShortConv.reference`, with each token's head taps added to `taps`."""
        return _reference(
            x,
            taps,
            dynamic_taps,
            state,
            slots,
            query_start,
            has_state,
            DYNAMIC_SCALE,
            channels_per_head,
        )

    def is_valid(
        self,
        x: torch.Tensor,
        taps: torch.Tensor,
        dynamic_taps: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        query_start: torch.Tensor,
        has_state: torch.Tensor,
        token_request: torch.Tensor,
        channels_per_head: int,
    ) -> None:
        _check(
            x,
            taps,
            dynamic_taps,
            state,
            slots,
            query_start,
            has_state,
            token_request,
            channels_per_head,
        )


inkling2_short_conv = Inkling2ShortConv()
inkling2_dynamic_short_conv = Inkling2DynamicShortConv()
