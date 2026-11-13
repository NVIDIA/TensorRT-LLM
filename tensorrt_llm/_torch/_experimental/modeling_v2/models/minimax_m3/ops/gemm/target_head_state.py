# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common head host implementation, bound to per-target state and callbacks."""

from __future__ import annotations

from typing import Callable, Optional, Protocol

import torch


class HeadState(Protocol):
    weight: torch.Tensor
    coded: bool
    w: Optional[torch.Tensor]
    base: int
    exc_off: Optional[torch.Tensor]
    exc_pos: Optional[torch.Tensor]
    exc_w: Optional[torch.Tensor]
    part: Optional[torch.Tensor]
    counter: Optional[torch.Tensor]
    grid: int


class HeadTarget(Protocol):
    _NUM_SMS: Optional[int]
    _num_sms: Callable[[], int]
    TargetHead: Callable[..., HeadState]
    CTA_WARPS: int
    MAX_ROWS: int
    encode_chunked: Callable[..., tuple]
    round_trip_ok: Callable[..., bool]
    _target_head_op: Callable[..., torch.Tensor]


def _num_sms(state: HeadTarget, /) -> int:
    if state._NUM_SMS is None:
        state._NUM_SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    return state._NUM_SMS


def head_supported(state: HeadTarget, /, lm_head: object) -> bool:
    """A bias-free 2-D BF16 head of whole 16-row tiles whose K the coding covers (a multiple of 64)."""
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return False
    return (
        getattr(lm_head, "bias", None) is None
        and weight.shape[0] % 16 == 0
        and (weight.shape[1] % 64 == 0)
    )


def prepare_head(
    state: HeadTarget, /, lm_head: torch.nn.Module, grid: Optional[int] = None
) -> HeadState:
    """Code ``lm_head``'s local shard and check it decodes to the BF16 weight bit for bit (fails closed: a shard that
    does not is returned with ``coded=False`` and nothing kept). Built once, eagerly, before the executor's warmup
    (``M3Eagle3Worker.prebuild_target_head``)."""
    weight = lm_head.weight
    with torch.no_grad():
        w = weight.detach()
        packed, base, exc_off, exc_pos, exc_w = state.encode_chunked(w)
        if not state.round_trip_ok(packed, base, exc_off, exc_pos, exc_w, w):
            return state.TargetHead(weight=weight, coded=False)
    if grid is None:
        grid = state._num_sms()
    n = weight.shape[0]
    return state.TargetHead(
        weight=weight,
        coded=True,
        w=packed,
        base=base,
        exc_off=exc_off,
        exc_pos=exc_pos,
        exc_w=exc_w.view(-1),
        part=torch.zeros(
            grid * state.CTA_WARPS * 2 * 4 * 32, dtype=torch.float32, device=weight.device
        ),
        counter=torch.zeros((n + 15) // 16, dtype=torch.int32, device=weight.device),
        grid=grid,
    )


def logits(
    state: HeadTarget, /, head: HeadState, x: torch.Tensor, use_pdl: bool = True
) -> torch.Tensor:
    """This rank's BF16 logits shard [rows, n] of the rows x [rows <= MAX_ROWS, K] (BF16) from the coded shard."""
    assert head.coded and 1 <= x.shape[0] <= state.MAX_ROWS, (head.coded, x.shape)
    if x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
        x = x.contiguous()
    n = head.weight.shape[0]
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    return state._target_head_op(
        head.w,
        x,
        out,
        head.part,
        head.counter,
        head.exc_off,
        head.exc_pos,
        head.exc_w,
        head.base,
        n,
        head.grid,
        use_pdl,
    )
