# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python versions of the adaptive EPLB route and statistics ops (``adaptiveEplbKernels.h``): the
reference of the kernel tests, and the ops of the CPU tests."""

from collections import Counter

import numpy as np
import torch

from tensorrt_llm._torch.moe.live_rows import live_row_count

SPLIT_BLOCK = 1024
BLOCK_MULTIPLIER = 0x9E3779B9
EXPERT_MULTIPLIER = 0x632BE5AB
RANK_MULTIPLIER = 0x85EBCA6B
UINT32_MAX = (1 << 32) - 1


def route(ids, counts, starts, slots, cumulative_shares, ep_rank) -> list:
    """The slot of every flat expert id; ``cumulative_shares`` as unsigned 32-bit values."""
    num_experts, num_slots = len(counts), len(slots)
    out = []
    for base in range(0, len(ids), SPLIT_BLOCK):
        block = base // SPLIT_BLOCK
        experts = [
            e if 0 <= e < num_experts else num_experts for e in ids[base : base + SPLIT_BLOCK]
        ]
        spread = [e if e < num_experts and counts[e] > 1 else num_experts for e in experts]
        total = Counter(spread)
        seen = Counter()
        for expert, spread_id in zip(experts, spread):
            out.append(num_slots)
            if expert == num_experts:
                continue
            copies, start = counts[expert], starts[expert]
            if copies <= 0 or start < 0 or start + copies > num_slots:
                continue
            k = 0
            if spread_id < num_experts:
                n = total[spread_id]
                step = UINT32_MAX if n <= 1 else (1 << 32) // n
                point = expert * EXPERT_MULTIPLIER + block * BLOCK_MULTIPLIER
                point = (point + ep_rank * RANK_MULTIPLIER + seen[spread_id] * step) & UINT32_MAX
                seen[spread_id] += 1
                while k < copies - 1 and point >= cumulative_shares[start + k]:
                    k += 1
            out[-1] = slots[start + k]
    return out


def _live(ids, live_row_word) -> int:
    """The rows of ``ids`` (``[rows, top_k]``) that ``live_row_word`` keeps; all without a word."""
    rows = ids.shape[0] if ids.dim() > 1 else ids.numel()
    return rows if live_row_word is None else live_row_count(int(live_row_word[0]), rows)


def record(ids, rows, forwards, enabled, first=True, live_row_word=None) -> None:
    """Add the ids of the live rows to ``rows`` and, on a forward's first call, one to
    ``forwards``, in place."""
    if int(enabled[0]) == 0:
        return
    ids = ids[: _live(ids, live_row_word)]
    flat = ids.reshape(-1).to(torch.int64).cpu().numpy()
    valid = flat[(flat >= 0) & (flat < rows.numel())]
    counts = np.bincount(valid, minlength=rows.numel()).astype(np.float32)
    if first:
        forwards += 1
    rows.copy_(torch.from_numpy(rows.cpu().numpy().astype(np.float32) + counts))


def route_into(
    ids,
    counts,
    starts,
    slots,
    cumulative_shares,
    ep_rank,
    out,
    weights=None,
    weights_out=None,
    rows=None,
    forwards=None,
    enabled=None,
    first=True,
    live_row_word=None,
):
    """``torch.ops.trtllm.adaptive_eplb_route_into`` on CPU tensors."""
    tables = [t.tolist() for t in (counts, starts, slots)]
    shares = [value & UINT32_MAX for value in cumulative_shares.tolist()]
    routed = route(ids.reshape(-1).tolist(), *tables, shares, ep_rank)
    flat, top_k, live = out.view(-1), ids.shape[-1], _live(ids, live_row_word)
    flat[: len(routed)] = torch.tensor(routed, dtype=out.dtype)
    flat[live * top_k :] = -1
    if weights is not None:
        weights_out[:live, :top_k] = weights[:live]
        weights_out[live : flat.numel() // top_k, :top_k] = 0.0
    if rows is not None:
        record(ids, rows, forwards, enabled, first, live_row_word)
