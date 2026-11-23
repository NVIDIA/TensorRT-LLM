# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Torch mirrors of the SM107 shard-pick and strict-acceptance epilogues."""

from __future__ import annotations

from typing import Optional

import torch

_LOW32 = 0xFFFFFFFF


def ordered_bits(v: torch.Tensor) -> torch.Tensor:
    """The kernel's order-preserving uint32 image of fp32 values, as int64 (-0.0 maps to +0.0's image: they tie)."""
    b = v.float().contiguous().view(torch.int32).to(torch.int64) & _LOW32
    b = torch.where(b == 0x80000000, torch.zeros_like(b), b)
    return torch.where(b >= 0x80000000, _LOW32 - b, b | 0x80000000)


def make_keys(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Exchange keys of (fp32 value, global token index) pairs: larger value wins, ties take the lower index."""
    return (ordered_bits(values) << 32) | (_LOW32 - index.to(torch.int64))


def unsigned_max(keys: torch.Tensor, dim: int) -> torch.Tensor:
    """Max of int64-stored uint64 keys (the kernel compares them unsigned)."""
    sign = torch.iinfo(torch.int64).min
    return (keys ^ sign).max(dim=dim).values ^ sign


def key_token(key: torch.Tensor) -> torch.Tensor:
    return (_LOW32 - (key & _LOW32)).to(torch.int32)


def local_argmax(logits: torch.Tensor, vocab_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(max value, lowest global index reaching it) per row of a [M, Vs] fp32 shard."""
    best = logits.max(dim=1).values
    idx = (logits == best[:, None]).float().argmax(dim=1)  # first (lowest) index at the max
    return best, idx.to(torch.int64) + vocab_offset


def exchange_tokens(
    own_keys: torch.Tensor,
    ws_uc: torch.Tensor,
    counters: torch.Tensor,
    rank: int,
    world: int,
    slot_rows: int,
) -> torch.Tensor:
    """Global tokens of the cross-rank pick: this rank's keys against the other ranks' keys already in its slots.

    Slot (half * slot_rows + t) * world + r of ``ws_uc`` holds rank r's key of row t; half = call counter & 1. A
    single-device run stands in for the other ranks by writing their keys there before the call.
    """
    m = own_keys.shape[0]
    half = int(counters[1].item()) & 1
    slots = (
        ws_uc.view(torch.int64)[half * slot_rows * world : (half * slot_rows + m) * world]
        .view(m, world)
        .clone()
    )
    slots[:, rank] = own_keys
    return key_token(unsigned_max(slots, dim=1))


def strict_accept(
    tokens: torch.Tensor,
    draft: torch.Tensor,
    batch_idx: torch.Tensor,
    force_total: int,
    next_new: Optional[torch.Tensor],
    force_vec: Optional[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The strict acceptance epilogue from the verify-row tokens [nreq * (D + 1)] (request-major).

    accepted[b] = tokens of request b's rows; num_accepted[b] = 1 + the matching prefix of draft[b] (overridden by
    force_vec[b], else by force_total > 0); gather_ids[b] = batch_idx[b] * (D + 1) + num_accepted[b] - 1, and
    next_new[b, 0] (when given) = the token at that row. Returns the outputs plus ``next_new`` as written.
    """
    nreq, d = draft.shape
    k1 = d + 1
    tokens = tokens.to(torch.int32)
    accepted = tokens.view(nreq, k1).clone()
    match = (draft.to(torch.int32) == accepted[:, :d]).to(torch.int32)
    num = 1 + match.cumprod(dim=1).sum(dim=1)
    if force_vec is not None:
        num = force_vec[:nreq].to(num.dtype)
    elif force_total > 0:
        num = torch.full_like(num, force_total)
    gather = batch_idx.to(torch.int64) * k1 + num.to(torch.int64) - 1
    out = {"accepted": accepted, "num_accepted": num.to(torch.int32), "gather_ids": gather}
    if next_new is not None:
        nn = next_new.clone()
        nn[:, 0] = tokens[gather]
        out["next_new"] = nn
    return out
