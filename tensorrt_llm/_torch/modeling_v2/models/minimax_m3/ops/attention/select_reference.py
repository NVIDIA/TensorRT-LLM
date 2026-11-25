# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unchanged integer-ranking reference, using each target's forcing constants."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class SelectReferenceTarget(Protocol):
    TOPK: int
    INIT_SCORE: float
    LOCAL_SCORE: float


def select_reference(
    self: Any,
    state: SelectReferenceTarget,
    /,
    scores: torch.Tensor,
    n_valid_blocks: torch.Tensor,
    out: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    use_pdl: bool = False,
    early_trigger: bool = False,
    trigger_phase: int = 0,
) -> torch.Tensor:
    """The [total_q, num_kv_heads, 16] int32 table the op is supposed to write.

    Exact integer ranking, no floating point beyond the forcing: each
    score's bit pattern maps to an unsigned key that is smaller for a larger
    score (NaN above +inf), packed with the block id so one sort ranks by
    (score descending, block ascending). ``out``, ``use_pdl``, ``early_trigger`` and
    ``trigger_phase`` only say where and how the op writes; they do not enter the result.
    """
    num_kv_heads, num_blocks, total_q = scores.shape
    device = scores.device
    blocks = torch.arange(num_blocks, device=device, dtype=torch.int64)
    raw = n_valid_blocks[:total_q].to(torch.int64)
    valid = raw.clamp(0, num_blocks)
    local_start = (raw - local_blocks).clamp_min(0)
    result = torch.full((total_q, num_kv_heads, state.TOPK), -1, dtype=torch.int32, device=device)
    for head in range(num_kv_heads):
        forced = scores[head].float().t().clone()
        forced = torch.where(blocks[None, :] < init_blocks, state.INIT_SCORE, forced)
        forced = torch.where(blocks[None, :] >= local_start[:, None], state.LOCAL_SCORE, forced)
        bits = forced.view(torch.int32).to(torch.int64) & 4294967295
        key = torch.where(bits >= 2147483648, bits, ~bits & 2147483647)
        in_row = blocks[None, :] < valid[:, None]
        packed = torch.where(in_row, key << 16 | blocks[None, :], torch.iinfo(torch.int64).max)
        k = min(state.TOPK, num_blocks)
        picked = torch.topk(packed, k, dim=1, largest=False).values
        picked_key = picked >> 16
        keep = (picked != torch.iinfo(torch.int64).max) & (picked_key != 4286578688)
        ids = torch.where(keep, picked & 65535, torch.iinfo(torch.int64).max)
        ids = ids.sort(dim=1).values
        ids = torch.where(ids == torch.iinfo(torch.int64).max, -1, ids)
        result[:, head, :k] = ids.to(torch.int32)
    return result
