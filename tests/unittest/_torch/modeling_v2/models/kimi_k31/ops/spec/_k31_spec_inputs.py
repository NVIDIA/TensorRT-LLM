# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the Kimi K3.1 spec entries' tests: the sampling and DFlash2 selector entries'
inputs and the in-process TP group."""

from __future__ import annotations

import zlib

import torch

__extra_import_path__ = ["../.."]
from _k31_utils import on_sm

on_sm103 = on_sm(103, 107)


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


def row_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).abs().sum(dim=-1)


# --- the DFlash2 candidate selector's inputs ---

#: K3.1's DFlash2 selector: rank 256 codebooks over the 163,840-token vocabulary, a 7168 hidden size.
VOCAB, RANK, HIDDEN = 163840, 256, 7168
_CODEBOOKS: dict = {}


def selector_codebooks(dtype=torch.bfloat16) -> tuple:
    """One pair of full-size codebooks (predecessor, successor) per dtype for the whole process
    (80 MB each in BF16)."""
    if dtype not in _CODEBOOKS:
        g = torch.Generator(device="cuda").manual_seed(7)
        pred = (0.05 * torch.randn(VOCAB, RANK, generator=g, device="cuda")).to(dtype)
        succ = (0.05 * torch.randn(VOCAB, RANK, generator=g, device="cuda")).to(dtype)
        _CODEBOOKS[dtype] = (pred, succ)
    return _CODEBOOKS[dtype]


def selector_walk_inputs(spec: dict, seed: int, *, block_tensor: bool) -> list:
    """The greedy selector entries' arguments for a cell: the BF16 codebooks, the hidden
    projection, each slot's 32 distinct candidates and their descending unary scores (``ties``: all
    zero, with a zero projection; ``special``: row 0 all -inf, a NaN in row 1, one -inf in row 2),
    the hidden rows, the anchors, and the block: a random ``[n, slots, VOCAB]`` tensor of dtype
    ``spec["block"]`` with ``block_tensor``, else that dtype."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, slots = spec["n"], spec["slots"]
    pred_cb, succ_cb = selector_codebooks()
    weight = (0.02 * torch.randn(RANK, HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    cand = torch.stack(
        [torch.randperm(VOCAB, generator=g, device="cuda")[:32] for _ in range(n * slots)]
    ).view(n, slots, 32)
    unary = (
        (10.0 + 2.0 * torch.randn(n, slots, 32, generator=g, device="cuda"))
        .sort(dim=-1, descending=True)
        .values
    )
    if spec["scores"] == "ties":
        unary = torch.zeros_like(unary)
        weight = torch.zeros_like(weight)
    if spec["scores"] == "special":
        unary[0] = float("-inf")
        unary[1, :, 5] = float("nan")
        unary[2, :, 0] = float("-inf")
    hidden = torch.randn(n, slots, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    anchor = torch.randint(0, VOCAB, (n,), generator=g, device="cuda", dtype=torch.int32)
    block = spec["block"]
    if block_tensor:
        block = torch.randn(n, slots, VOCAB, generator=g, device="cuda").to(spec["block"])
    return [pred_cb, succ_cb, weight, cand, unary.to(spec["unary"]), hidden, anchor, block]
