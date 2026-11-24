# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the Kimi K3.1 spec entries' tests."""

from __future__ import annotations

import zlib

import pytest
import torch

on_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="the Kimi K3.1 family entries are certified on sm_103 (sm_107 driven separately)",
)


def seed_of(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def run_tp_group(call, shards):
    """Every rank's result of ``call(local, all_reduce, tp_size, tp_rank)`` in one process.

    A first pass records each rank's published pair buffer; the second hands every rank their
    rank-order sum, which is what the TP group's float32 all-reduce returns.
    """
    tp = len(shards)
    if tp == 1:
        return [call(shards[0], None, 1, 0)]
    published = []

    def record(buf: torch.Tensor) -> torch.Tensor:
        published.append(buf.clone())
        return buf

    for rank, shard in enumerate(shards):
        call(shard, record, tp, rank)
    assert len(published) == tp
    total = published[0].clone()
    for buf in published[1:]:
        total += buf
    return [call(shard, lambda buf: total.clone(), tp, rank) for rank, shard in enumerate(shards)]


#: The engine's disable sentinels and greedy temperature (SpecMetadata's one-model normalization).
DISABLE_TOP_K = torch.iinfo(torch.int32).max
GREEDY_SENTINEL_T = 1e-5


def sampling_inputs(spec: dict, seed: int, *, fused: bool):
    """``(logits, temperatures, top_ks, top_ps, min_ps)`` of a sampler cell.

    ``fused``: every filter is a per-row tensor, disabled rows at their sentinel. Otherwise
    (the flashinfer chain) a filter the cell leaves unset is None, as the advanced sampling mode
    resolves it, and min-p is None.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, vocab = spec["rows"], spec["vocab"]
    logits = (torch.randn(rows, vocab, generator=gen, device="cuda") * 2.0).to(spec["dtype"])
    temperatures = torch.full((rows,), spec["temperature"], dtype=torch.float32, device="cuda")
    if spec["greedy_rows"]:
        greedy = torch.randperm(rows, generator=gen, device="cuda")[: spec["greedy_rows"]]
        temperatures[greedy] = GREEDY_SENTINEL_T
    enabled = (
        torch.rand(rows, generator=gen, device="cuda") < 0.5
        if spec["mixed"]
        else torch.ones(rows, dtype=torch.bool, device="cuda")
    )

    def column(value, off, dtype):
        if value is None:
            return torch.full((rows,), off, dtype=dtype, device="cuda") if fused else None
        on = torch.full((rows,), value, dtype=dtype, device="cuda")
        return torch.where(enabled, on, torch.full_like(on, off))

    top_ks = column(spec["top_k"], DISABLE_TOP_K, torch.int32)
    top_ps = column(spec["top_p"], 1.0, torch.float32)
    min_ps = column(spec["min_p"], 0.0, torch.float32) if fused else None
    return logits, temperatures, top_ks, top_ps, min_ps


def philox_state(rows: int, seed: int, base: int = 0):
    """Per-row int64 ``(seed, offset)``: one seed, distinct offsets."""
    seeds = torch.full((rows,), seed, dtype=torch.int64, device="cuda")
    offsets = base + torch.arange(rows, dtype=torch.int64, device="cuda")
    return seeds, offsets


def row_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).abs().sum(dim=-1)


def hot_rows(rows: int, vocab: int, seed: int) -> torch.Tensor:
    """``rows`` copies of one float32 logits row whose mass sits on eight hot tokens."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    row = torch.full((vocab,), -30.0, device="cuda")
    hot = torch.randperm(vocab, generator=gen, device="cuda")[:8]
    row[hot] = torch.tensor([0.0, -0.2, -0.5, -0.8, -1.2, -1.6, -2.2, -3.0], device="cuda")
    return row.expand(rows, vocab).contiguous()
