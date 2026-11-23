# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared unchanged sampling references and test-only input validation."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

import torch


class SamplingTarget(Protocol):
    TEMP_EPSILON: float
    nucleus_probs: Callable[..., torch.Tensor]
    strict_accept: Callable[..., dict[str, torch.Tensor]]


def nucleus_probs(
    state: SamplingTarget, /, logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor
) -> torch.Tensor:
    """Per-row sampling distribution [rows, Vs] (fp32): w = exp(x / (T + 1e-6) - max), kept iff the mass strictly
    above its weight is at most top_p * sum(w) (ties included), renormalized over the kept tokens."""
    inv = 1.0 / (temperatures.float() + state.TEMP_EPSILON)
    x = logits.float() * inv[:, None]
    w = torch.exp(x - x.max(dim=1, keepdim=True).values).double()
    z = w.sum(dim=1, keepdim=True)
    order = torch.sort(w, dim=1, descending=True)
    sw = order.values
    csum = torch.cumsum(sw, dim=1) - sw
    first = torch.ones_like(sw, dtype=torch.bool)
    first[:, 1:] = sw[:, 1:] != sw[:, :-1]
    group = torch.cumsum(first.long(), dim=1) - 1
    starts = torch.zeros_like(csum)
    starts.scatter_reduce_(1, group, csum, reduce="amin", include_self=False)
    above = torch.gather(starts, 1, group)
    keep_sorted = above <= top_ps.double()[:, None] * z
    keep = torch.zeros_like(keep_sorted)
    keep.scatter_(1, order.indices, keep_sorted)
    p = torch.where(keep, w, torch.zeros_like(w))
    return (p / p.sum(dim=1, keepdim=True)).float()


def sampled_reference(
    self: Any,
    state: SamplingTarget,
    /,
    logits: torch.Tensor,
    vocab_offset: int,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    seeds: torch.Tensor,
    offsets: torch.Tensor,
    ws_uc: torch.Tensor,
    ws_mc_ptr: int,
    counters: torch.Tensor,
    tokens: torch.Tensor,
    rank: int,
    world: int,
    draft: torch.Tensor,
    batch_idx: torch.Tensor,
    acc_tokens: torch.Tensor,
    num_acc: torch.Tensor,
    gather_ids: torch.Tensor,
    force_total: int,
    next_new: Optional[torch.Tensor],
    force_vec: Optional[torch.Tensor],
    extra_attempts: Optional[torch.Tensor],
    use_pdl: bool = True,
    top_ks: Optional[torch.Tensor] = None,
    min_ps: Optional[torch.Tensor] = None,
) -> dict:
    """{"probs": the per-row nucleus distribution over this shard [rows, Vs] (global id = vocab_offset +
    column), "accept": tokens -> the strict acceptance outputs}. The draw itself is random; ``seeds`` /
    ``offsets`` select it and do not change the distribution. World 1: this shard is the whole vocabulary."""
    rows = logits.shape[0]
    probs = state.nucleus_probs(logits, temperatures[:rows], top_ps[:rows])

    def accept(drawn: torch.Tensor) -> dict[str, torch.Tensor]:
        return state.strict_accept(drawn, draft, batch_idx, force_total, next_new, force_vec)

    return {"probs": probs, "accept": accept}


def validate_sampling(
    self: Any,
    state: SamplingTarget,
    /,
    logits: torch.Tensor,
    vocab_offset: int,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    seeds: torch.Tensor,
    offsets: torch.Tensor,
    ws_uc: torch.Tensor,
    ws_mc_ptr: int,
    counters: torch.Tensor,
    tokens: torch.Tensor,
    rank: int,
    world: int,
    draft: torch.Tensor,
    batch_idx: torch.Tensor,
    acc_tokens: torch.Tensor,
    num_acc: torch.Tensor,
    gather_ids: torch.Tensor,
    force_total: int,
    next_new: Optional[torch.Tensor],
    force_vec: Optional[torch.Tensor],
    extra_attempts: Optional[torch.Tensor],
    use_pdl: bool = True,
    top_ks: Optional[torch.Tensor] = None,
    min_ps: Optional[torch.Tensor] = None,
) -> None:
    rows = logits.shape[0]
    if top_ks is not None:
        k = top_ks[:rows]
        assert bool(((k <= 0) | (k >= logits.shape[1] * world)).all()), (
            "top-k rows must take the generic path"
        )
    if min_ps is not None:
        assert not bool((min_ps[:rows] > 0).any()), "min-p rows must take the generic path"
    assert rows == draft.shape[0] * (draft.shape[1] + 1), (
        "logits rows must be nreq * (draft_len + 1)"
    )
    for t in (temperatures, top_ps, seeds, offsets):
        assert t.numel() >= rows and t.is_contiguous(), "per-row parameters must cover every row"
    assert temperatures.dtype == torch.float32 and top_ps.dtype == torch.float32, (
        "fp32 sampling parameters"
    )
    assert seeds.dtype == torch.int64 and offsets.dtype == torch.int64, (
        "int64 Philox seeds / offsets"
    )
    assert int(counters[0].item()) == 0, "counters[0] (finished rows) must be zero between calls"
