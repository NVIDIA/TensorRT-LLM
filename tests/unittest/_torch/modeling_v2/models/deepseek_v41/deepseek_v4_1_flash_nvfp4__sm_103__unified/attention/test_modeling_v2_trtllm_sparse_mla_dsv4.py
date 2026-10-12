# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the trtllm_sparse_mla_dsv4 entry.

The entry owns what is certified: `CELLS` is the small-pool list (decode, DSpark verify, varlen
prefill, the window-only placeholder and the non-causal draft block, at every per-rank head count
H 8, 16, 32 and 64), and `reference` and `compare` are the gate. This file turns a cell's spec into
tensors and keeps every check a cell cannot carry: the engine-size pools, the engine envelope,
the native-storage compact reads, CUDA-graph capture, the guard-page trials, the workspace high-water mark
and the wrong-reading controls.

The reference (`op.reference`) gathers the pool rows each index row names, takes fp32 scores scaled
by softmax_scale, fp32 max and sum, casts the probabilities to bf16 for the PV product (the precision
of DeepSeek-V4.1's reference sparse_attn kernel), and adds the per-head sink to the denominator only,
in natural-log units. `_reference` below reads it with explicit per-row window counts; its
deliberately wrong readings (`_wrong_reading`) are the controls.

Pools are sized as a cache manager would size them -- independent of the batch -- and every request's
rows are reached through a randomly permuted page table: SWA pool 96 pages x 128 rows (12288 slots),
compressed pool 160 pages x 64 rows (10240 slots). Those widths were chosen here, plainly larger than
any batch in this file; they are not taken from a runtime-value dump.

The `test_real_pools_*` cases use the pool geometry the core DeepSeek-V4.1 target's cache manager
allocates: SWA 188160 x 128 rows, compressed 453120 x 64 (ratio 2) and 151040 x 128 (ratio 1), and
attend rows in the last pages of both pools -- past the 2^31 / 2^32-byte and 2^31-element offsets of
a 1024-byte row.

The `test_kvq_*` cases hold the native-storage target's compact read (its two persistent bf16 read buffers,
with the index renumbered onto them and -1 window columns backed by zero rows) bitwise to the
container read on the same values.

The engine-pool, engine-envelope (`test_e_*`), workspace, pool-size and guard-page cases run at every
per-rank head count of the target's members, H 8, 16, 32 and 64 (MEMBER_HEADS / ALL_HEADS).
"""

import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.trtllm_sparse_mla_dsv4 import (  # noqa: E501
    trtllm_sparse_mla_dsv4 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "trtllm_sparse_mla_dsv4 requires a CUDA device"

# The guard-page and pool-size trials run this file as a script in a child process, where the
# repo's `__extra_import_path__` hook is not installed: the child reaches `_validating` through
# PYTHONPATH instead (set only in the child's environment).
_CHILD_ENV = dict(
    os.environ,
    PYTHONPATH=os.pathsep.join(
        p for p in (str(Path(__file__).resolve().parents[4]), os.environ.get("PYTHONPATH")) if p
    ),
)

HEAD_DIM = 512
WINDOW = 128
SCALE = HEAD_DIM**-0.5
LOG2E = 1.4426950408889634
SWA_PAGES, SWA_PAGE = 96, 128
COMP_PAGES, COMP_PAGE = 160, 64
RATIO = 2
INDEX_TOPK = 512
DEV = torch.device("cuda")
# Rows of mapped memory every pool the call may load a -1 index from must span from its first row
# (Precondition 8; measured in the guard-page tests below).
POOL_MIN_ROWS = 128

# The engine envelope (per rank). E1: the CUDA-graph capture set -- every G 1..32 plus TRT-LLM's
# padded default list for cuda_graph_config {enable_padding: true, max_batch_size: 256} (llm_args.py,
# _generate_cuda_graph_batch_sizes: 1, 2, 4, 8, 16..128 step 8, then 192, 256); the padding-off list
# at 256 (1..32, 64, 128, 256) is a subset. 46 values. The decode (next_n 1) and verify (next_n 6)
# graphs of the target's generation call and the DSpark drafter's graph (N = 5G rows) are captured at
# every one of them. E3: a step holds up to 256 requests, and an eager step any G 1..256.
GSET = sorted(set(range(1, 33)) | set(range(40, 129, 8)) | {192, 256})
assert len(GSET) == 46
E_B = 256
# Per-rank query heads of the target's trunk call, 64 / attention TP: 16 at tep4 / tp4, 64 at dep4 /
# dep8 (attention DP), 8 at tep8 (attention TP 8 over two NVL72 trays), 32 at pp2 x {tep2, tp2}. The
# DSpark drafter's call runs all 64 heads on every member. The envelope cases below run at all four.
MEMBER_HEADS = (8, 32)
ALL_HEADS = (8, 16, 32, 64)

# The entry's gate (`op.compare`, derivation in its docstring): rtol 1.6e-2 plus atol
# 2^-8 * max|V|. RTOL, `_atol` and `_excess` restate it to report an output as a multiple of the gate;
# `_atol` takes max|V| over whole pools, a bound on the attended rows' max.
RTOL = 1.6e-2


def _atol(*pools: torch.Tensor) -> float:
    return 2.0**-8 * max(p.float().abs().max().item() for p in pools)


def _workspace() -> torch.Tensor:
    return torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)


def _pools(gen: torch.Generator):
    swa = torch.randn(SWA_PAGES, 1, SWA_PAGE, HEAD_DIM, generator=gen, device=DEV).to(
        torch.bfloat16
    )
    comp = torch.randn(COMP_PAGES, 1, COMP_PAGE, HEAD_DIM, generator=gen, device=DEV).to(
        torch.bfloat16
    )
    return swa, comp


def _sinks(num_heads: int) -> torch.Tensor:
    # Spans "negligible" to "dominant" against sum(exp(s - max)) over up to 640 columns.
    return torch.linspace(-2.0, 9.0, num_heads, device=DEV, dtype=torch.float32)


def _page_tables(gen: torch.Generator):
    swa_bt = torch.randperm(SWA_PAGES, generator=gen, device=DEV)
    comp_bt = torch.randperm(COMP_PAGES, generator=gen, device=DEV)
    return swa_bt, comp_bt


def _build_rows(
    requests,
    gen: torch.Generator,
    topk: int,
    swa_pages_per_req: int,
    comp_pages_per_req: int,
    *,
    swa_bt: torch.Tensor | None = None,
    comp_bt: torch.Tensor | None = None,
    swa_page: int = SWA_PAGE,
    comp_page: int = COMP_PAGE,
    ratio: int = RATIO,
):
    """Index rows the way a DeepSeek-V4.1 layer builds them.

    requests: list of (cached_len, q_len). Query j of a request sits at pos = cached_len + j and sees
    window positions max(0, pos-127)..pos (all valid, in position order) and min(512, (pos+1)//2)
    compressed positions (a random subset, sorted, as the indexer top-k is), compacted at the front;
    sparse_topk_lens = 128 + that count. topk=128 builds window-only rows.
    Each request owns swa_pages_per_req SWA pages used as a windowed ring (position p lives in page
    (p // 128) % swa_pages_per_req) and comp_pages_per_req compressed pages used the same way, all
    drawn from a random permutation of the pool's pages -- or, when swa_bt / comp_bt are given,
    taken in order from those page lists (request b gets entries [b * per_req, (b + 1) * per_req)).
    Returns sparse_indices [N, topk] int32 (pool token rows, -1 padding), sparse_topk_lens [N],
    seq_lens [B] (= cached_len + q_len), cum_seq_lens_q [B+1].
    """
    if swa_bt is None:
        swa_bt, comp_bt = _page_tables(gen)
    assert len(requests) * swa_pages_per_req <= swa_bt.numel()
    assert comp_pages_per_req == 0 or len(requests) * comp_pages_per_req <= comp_bt.numel()
    rows_idx, lens = [], []
    for b, (cached, q_len) in enumerate(requests):
        swa_pages = swa_bt[b * swa_pages_per_req : (b + 1) * swa_pages_per_req]
        if comp_pages_per_req:
            comp_pages = comp_bt[b * comp_pages_per_req : (b + 1) * comp_pages_per_req]
        for j in range(q_len):
            pos = cached + j
            row = torch.full((topk,), -1, dtype=torch.int32, device=DEV)
            win = torch.arange(max(0, pos - WINDOW + 1), pos + 1, device=DEV)
            page = swa_pages[(win // swa_page) % swa_pages_per_req]
            row[: win.numel()] = (page * swa_page + win % swa_page).int()
            k = 0
            if topk > WINDOW:
                visible = (pos + 1) // ratio
                k = min(topk - WINDOW, visible)
                sel = torch.randperm(visible, generator=gen, device=DEV)[:k].sort().values
                page = comp_pages[(sel // comp_page) % comp_pages_per_req]
                row[WINDOW : WINDOW + k] = (page * comp_page + sel % comp_page).int()
            rows_idx.append(row)
            lens.append(WINDOW + k)
    idx = torch.stack(rows_idx).contiguous()
    active = torch.tensor(lens, dtype=torch.int32, device=DEV)
    seq_lens = torch.tensor([c + q for c, q in requests], dtype=torch.int32, device=DEV)
    cu = torch.tensor([0] + [q for _, q in requests], dtype=torch.int32, device=DEV).cumsum(0).int()
    return idx, active, seq_lens, cu


def _window_counts(seq_lens: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    """Per query row: min(128, seq_lens[b] - q_len_b + j + 1)."""
    cu = cu.long()
    q_lens = cu[1:] - cu[:-1]
    req = torch.repeat_interleave(torch.arange(seq_lens.numel(), device=DEV), q_lens)
    j = torch.arange(req.numel(), device=DEV) - cu[:-1][req]
    return (seq_lens.long()[req] - q_lens[req] + j + 1).clamp(max=WINDOW)


def _reference(
    q: torch.Tensor,
    swa: torch.Tensor,
    comp: torch.Tensor,
    idx: torch.Tensor,
    active: torch.Tensor,
    window_count: torch.Tensor,
    sinks: torch.Tensor,
    *,
    sink: str = "denominator",
    negative_index: str = "zero_row",
    dedup: bool = False,
) -> torch.Tensor:
    """q [N,H,512] bf16 -> [N,H,512] fp32, with an explicit window count per query row.

    The correct reading is the entry's `op.reference`, given each row as its own q_len-1 request
    whose virtual `seq_lens` is that row's window count -- the device the contract names for a
    per-row count (min(128, seq_lens - 1 + 0 + 1) is the count itself). Setting `sink`,
    `negative_index` or `dedup` away from the correct reading selects a control (`_wrong_reading`).
    """
    if (sink, negative_index, dedup) != ("denominator", "zero_row", False):
        return _wrong_reading(
            q,
            swa,
            comp,
            idx,
            active,
            window_count,
            sinks,
            sink=sink,
            negative_index=negative_index,
            dedup=dedup,
        )
    q3 = q.reshape(-1, q.shape[-2], HEAD_DIM)
    n = q3.shape[0]
    expected, _ = op.reference(
        query=q3,
        swa_kv_cache=swa,
        compressed_kv_cache=comp,
        workspace_buffer=None,
        sparse_indices=idx,
        sparse_topk_lens=active,
        seq_lens=window_count.to(torch.int32),
        sinks=sinks,
        softmax_scale=SCALE,
        cum_seq_lens_q=torch.arange(n + 1, dtype=torch.int32, device=q.device),
        max_q_len=1,
    )
    return expected


def _wrong_reading(
    q: torch.Tensor,
    swa: torch.Tensor,
    comp: torch.Tensor,
    idx: torch.Tensor,
    active: torch.Tensor,
    window_count: torch.Tensor,
    sinks: torch.Tensor,
    *,
    sink: str = "denominator",
    negative_index: str = "zero_row",
    dedup: bool = False,
) -> torch.Tensor:
    """The controls' deliberately wrong readings; q [N,H,512] bf16 -> [N,H,512] fp32.

    Column c of row r is attended iff c < active[r] and (c >= 128 or c < window_count[r]).
    negative_index "masked" drops an attended -1 instead of reading it as an all-zero row; sink
    "numerator" / "log2e" / "scaled" / "ignored" misplace the sink; dedup attends a repeated row
    once. Only reached through `_reference` with one of those set: the correct reading is
    `op.reference`.
    """
    n, h, _ = q.shape
    t = idx.shape[1]
    swa_f = swa.reshape(-1, HEAD_DIM)
    comp_f = comp.reshape(-1, HEAD_DIM)
    col = torch.arange(t, device=DEV)
    out = torch.empty(n, h, HEAD_DIM, dtype=torch.float32, device=DEV)
    for s0 in range(0, n, 256):
        s1 = min(n, s0 + 256)
        ib = idx[s0:s1].long()
        kv = torch.cat(
            [swa_f[ib[:, :WINDOW].clamp_min(0)], comp_f[ib[:, WINDOW:].clamp_min(0)]], 1
        ).float()
        attended = (col[None] < active[s0:s1, None]) & (
            (col[None] >= WINDOW) | (col[None] < window_count[s0:s1, None])
        )
        if negative_index == "zero_row":
            kv = kv * (ib >= 0)[..., None]
        else:
            attended = attended & (ib >= 0)
        if dedup:
            same_pool = swa.data_ptr() == comp.data_ptr()
            key = ib if same_pool else torch.where(col[None] < WINDOW, ib, ib + (1 << 40))
            for r in range(ib.shape[0]):
                seen = set()
                for c in attended[r].nonzero().flatten().tolist():
                    if int(key[r, c]) in seen:
                        attended[r, c] = False
                    seen.add(int(key[r, c]))
        s = torch.einsum("nhd,ntd->nht", q[s0:s1].float(), kv) * SCALE
        s = s.masked_fill(~attended[:, None, :], float("-inf"))
        m = s.amax(-1)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = torch.exp(s - m[..., None])
        denom = p.sum(-1)
        num = torch.einsum("nht,ntd->nhd", p.to(torch.bfloat16).float(), kv)
        sk = sinks[None, :].float()
        if sink == "denominator":
            denom = denom + torch.exp(sk - m)
        elif sink == "numerator":
            e = torch.exp(sk - m)
            denom = denom + e
            num = num + e[..., None]
        elif sink == "log2e":
            denom = denom + torch.exp(sk * LOG2E - m)
        elif sink == "scaled":
            denom = denom + torch.exp(sk * SCALE - m)
        elif sink != "ignored":
            raise ValueError(sink)
        o = num / denom[..., None]
        out[s0:s1] = torch.where(attended.any(-1)[:, None, None], o, torch.zeros_like(o))
    return out


def _run(q, swa, comp, idx, active, seq_lens, sinks, cu=None, max_q_len=None, out=None):
    return op(
        query=q,
        swa_kv_cache=swa,
        compressed_kv_cache=comp,
        workspace_buffer=_workspace(),
        sparse_indices=idx,
        sparse_topk_lens=active,
        seq_lens=seq_lens,
        sinks=sinks,
        softmax_scale=SCALE,
        cum_seq_lens_q=cu,
        max_q_len=max_q_len,
        out=out,
    )


def _check(o, ref, swa, comp):
    """The entry's gate, with max|V| taken over both pools."""
    op.compare(o.reshape(ref.shape), (ref, _atol(swa, comp) * 2.0**8))


# ---------------------------------------------------------------------------------------------
# the entry's cells: small-pool decode, DSpark verify, varlen prefill, window-only, draft block
# ---------------------------------------------------------------------------------------------
# Compressed pages each request's ring holds (`_build_rows` wraps positions onto them), per kind.
_CELL_COMP_PAGES = {"decode": 17, "verify": 30, "prefill": 50, "window_only": 0}


def _cell_inputs(spec) -> dict:
    """The op's keyword arguments for one cell, on the file's small pools.

    Seeds and draws are per kind, so a cell's data does not depend on which other cells run.
    """
    kind, h = spec["kind"], spec["H"]
    sinks = _sinks(h)
    if kind == "draft_block":
        # Draft block of 5 queries that all see the same ring rows plus all 5 block rows
        # (non-causal). Each draft query is passed as its own q_len-1 request whose seq_lens is
        # the number of valid ring rows, so the window count equals exactly that number; the 5
        # block rows sit in the compressed segment (padded to 8 columns with -1 beyond
        # sparse_topk_lens = 128 + 5).
        gen = torch.Generator(device=DEV).manual_seed(5)
        swa, _ = _pools(gen)
        block_pool = torch.randn(4, 1, COMP_PAGE, HEAD_DIM, generator=gen, device=DEV).to(
            torch.bfloat16
        )
        blk = spec["block"]
        rows, seq = [], []
        for req, n_ring in enumerate(spec["ring_rows"]):
            ring = torch.randperm(SWA_PAGES * SWA_PAGE, generator=gen, device=DEV)[:n_ring].int()
            block_rows = (req * COMP_PAGE + torch.arange(blk, device=DEV)).int()
            for _ in range(blk):
                r = torch.full((spec["width"],), -1, dtype=torch.int32, device=DEV)
                r[:n_ring] = ring
                r[WINDOW : WINDOW + blk] = block_rows
                rows.append(r)
                seq.append(n_ring)
        idx = torch.stack(rows).contiguous()
        n = idx.shape[0]
        q = torch.randn(n, 1, h, HEAD_DIM, generator=gen, device=DEV).mul_(2).to(torch.bfloat16)
        return dict(
            query=q,
            swa_kv_cache=swa,
            compressed_kv_cache=block_pool,
            workspace_buffer=_workspace(),
            sparse_indices=idx,
            sparse_topk_lens=torch.full((n,), WINDOW + blk, dtype=torch.int32, device=DEV),
            seq_lens=torch.tensor(seq, dtype=torch.int32, device=DEV),
            sinks=sinks,
            softmax_scale=SCALE,
        )

    requests, topk = [tuple(r) for r in spec["requests"]], spec["topk"]
    seed = {
        "decode": 1000 + h + topk,
        "verify": 11 + h,
        "prefill": sum(q_len for _, q_len in requests) + h + topk,
        "window_only": 7,
    }[kind]
    gen = torch.Generator(device=DEV).manual_seed(seed)
    swa, comp = _pools(gen)
    if kind == "window_only":
        # Ratio-0 layers: a dummy one-page compressed pool full of NaN, never read at width 128.
        comp = torch.full(
            (1, 1, COMP_PAGE, HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=DEV
        )
    idx, active, seq_lens, cu = _build_rows(
        requests, gen, topk, swa_pages_per_req=2, comp_pages_per_req=_CELL_COMP_PAGES[kind]
    )
    b, n = len(requests), idx.shape[0]
    args = dict(
        swa_kv_cache=swa,
        compressed_kv_cache=comp,
        workspace_buffer=_workspace(),
        sparse_indices=idx,
        sparse_topk_lens=active,
        seq_lens=seq_lens,
        sinks=sinks,
        softmax_scale=SCALE,
    )
    if kind == "prefill":
        q = torch.randn(n, h, HEAD_DIM, generator=gen, device=DEV).mul_(2).to(torch.bfloat16)
        args.update(cum_seq_lens_q=cu, max_q_len=max(q_len for _, q_len in requests))
    else:
        q_len = requests[0][1]
        q = torch.randn(b, q_len, h, HEAD_DIM, generator=gen, device=DEV).mul_(2)
        q = q.to(torch.bfloat16)
    args["query"] = q
    return args


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _cell_inputs(cell.spec)
    o = op(**args)
    assert o.shape == args["query"].shape, (o.shape, args["query"].shape)
    if cell.spec["kind"] == "window_only":
        assert torch.isfinite(o).all(), "the NaN placeholder pool was read"
    op.compare(o, op.reference(**args))


# ---------------------------------------------------------------------------------------------
# DSpark drafter draft block at the drafter's own surface: 64 heads, index width 136
# ---------------------------------------------------------------------------------------------
# DeepSeek-V4.1's DSpark drafter runs one attention call per stage over G requests x 5 draft
# queries, each query passed as its own q_len-1 request (dense [G*5, 1, 64, 512]). swa_kv_cache is
# the speculative worker's ring [ring_slots, 3, 128, 512] bf16 viewed as
# [ring_slots * 3, 1, 128, 512]: request g's stage-s rows are page slot_g * 3 + s, compact from row
# 0. compressed_kv_cache is the block buffer holding the G*5 transient block K/V rows, row
# g * 5 + j. Index row [base + 0..127 | g*5 + 0..4, -1 x3] (width 136), sparse_topk_lens 133,
# seq_lens = the ring's valid-row count repeated 5x (window_count = min(128, seq_lens)).
# ring_slots is the worker's allocation, max_bs + 1 (one scratch row for CUDA-graph dummies): 33
# at the engine's max batch size 32, and 257 (max_bs 256) chosen here so the ring is plainly
# larger than any batch. Requests take random slots, the scratch row among them. Ring rows past a
# request's valid count hold a finite decoy (30.0, a score that would dominate the softmax), as a
# reused slot holds a previous request's rows.
DS_HEADS, DS_BLOCK, DS_STAGES = 64, 5, 3
DS_WIDTH = WINDOW + 8  # 136: window 128 | block 5 | -1 x3 (the width must be a multiple of 4)
DS_TOPK = WINDOW + DS_BLOCK  # 133
DS_RING_SLOTS = (33, 257)
DS_BATCHES = tuple(range(1, 33))
# The worker's valid-row count per request: 1 at the first draft (the bonus row alone), short and
# edge counts, and counts past the window (the count is not clamped by the worker; the kernel
# saturates it at 128), up to the E4 extreme 1,048,595 (Lm + 5 at max_seq_len 1,048,576).
DS_VALID = (1, 2, 3, 5, 6, 64, 126, 127, 128, 129, 4000, 16398, 262163, 1048595)
DS_DECOY = 30.0


def _ds_ring(ring_slots: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(ring_slots, DS_STAGES, WINDOW, HEAD_DIM, generator=gen, device=DEV).to(
        torch.bfloat16
    )


def _ds_inputs(G, ring, gen, shift=0, block_form="page", valid=None):
    """Write fresh ring rows for G requests into `ring` and build one stage's call.

    Returns (q [G*5, 1, 64, 512], block buffer, idx [G*5, 136], active [G*5], seq_lens [G*5]).
    block_form "page" passes the block buffer as [1, 1, G*5, 512] (one page), "row" as
    [G*5, 1, 1, 512] (one row per page).
    """
    slots = torch.randperm(ring.shape[0], generator=gen, device=DEV)[:G]
    stage = (G + shift) % DS_STAGES
    if valid is None:
        valid = [DS_VALID[(g + shift) % len(DS_VALID)] for g in range(G)]
    valid = torch.tensor(valid, dtype=torch.int32, device=DEV)
    ring[slots] = torch.randn(G, DS_STAGES, WINDOW, HEAD_DIM, generator=gen, device=DEV).to(
        torch.bfloat16
    )
    for g in range(G):
        ring[slots[g], stage, min(WINDOW, int(valid[g])) :] = DS_DECOY
    rows = G * DS_BLOCK
    # The block rows sit at the head of an allocation of at least POOL_MIN_ROWS rows (Precondition 8).
    backing = torch.zeros(max(rows, POOL_MIN_ROWS), HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    backing[:rows] = torch.randn(rows, HEAD_DIM, generator=gen, device=DEV).to(torch.bfloat16)
    block = backing[:rows]
    block = (
        block.view(1, 1, rows, HEAD_DIM)
        if block_form == "page"
        else block.view(rows, 1, 1, HEAD_DIM)
    )
    base = (slots.long() * DS_STAGES + stage) * WINDOW
    idx = torch.full((rows, DS_WIDTH), -1, dtype=torch.int32, device=DEV)
    idx[:, :WINDOW] = (
        base.repeat_interleave(DS_BLOCK)[:, None] + torch.arange(WINDOW, device=DEV)
    ).int()
    idx[:, WINDOW:DS_TOPK] = (
        torch.arange(rows, device=DEV).view(G, DS_BLOCK).repeat_interleave(DS_BLOCK, 0).int()
    )
    active = torch.full((rows,), DS_TOPK, dtype=torch.int32, device=DEV)
    seq_lens = valid.repeat_interleave(DS_BLOCK)
    q = (
        torch.randn(rows, 1, DS_HEADS, HEAD_DIM, generator=gen, device=DEV)
        .mul_(2)
        .to(torch.bfloat16)
    )
    return q, block, idx, active, seq_lens


def _ds_counts(seq_lens: torch.Tensor) -> torch.Tensor:
    return seq_lens.long().clamp(max=WINDOW)


def _ds_atol(ring_view, block, idx, counts) -> float:
    """The file's gate, 2^-8 * max|V|, over the rows the call attends (the decoys are not attended)."""
    col = torch.arange(WINDOW, device=DEV)
    w = idx[:, :WINDOW][col[None] < counts[:, None]]
    m = ring_view.view(-1, HEAD_DIM)[torch.unique(w).long()].float().abs().max()
    m = torch.maximum(m, block.float().abs().max())
    return 2.0**-8 * m.item()


def _ds_reference(q, ring_view, block, idx, active, counts, sinks, **kw):
    return _reference(
        q.view(-1, DS_HEADS, HEAD_DIM), ring_view, block, idx, active, counts, sinks, **kw
    )


def test_dspark_draft_block_h64_every_batch():
    """Eager draft block at 64 heads, both block-buffer forms: every G 1..32 on the 33-slot ring
    (max batch 32) and every G 1..256 on the 257-slot ring (max batch 256; N = 5G up to 1280)."""
    gen = torch.Generator(device=DEV).manual_seed(4100)
    sinks = _sinks(DS_HEADS)
    for ring_slots in DS_RING_SLOTS:
        ring = _ds_ring(ring_slots, gen)
        view = ring.view(ring_slots * DS_STAGES, 1, WINDOW, HEAD_DIM)
        for G in range(1, ring_slots):
            form = "page" if G % 2 else "row"
            q, block, idx, active, seq_lens = _ds_inputs(
                G, ring, gen, shift=ring_slots + G, block_form=form
            )
            o = _run(q, view, block, idx, active, seq_lens, sinks)
            assert o.shape == (G * DS_BLOCK, 1, DS_HEADS, HEAD_DIM)
            counts = _ds_counts(seq_lens)
            ref = _ds_reference(q, view, block, idx, active, counts, sinks)
            _check_atol(o, ref, _ds_atol(view, block, idx, counts))


@pytest.mark.parametrize("G", GSET)
def test_dspark_draft_block_h64_cuda_graph(G):
    """The draft block captured at every drafter batch G of the capture set GSET (46 values, 1..256;
    N = 5G = 5..1280 query rows), on the 33-slot ring up to G 32 and the 257-slot ring above.

    As the engine does it: one eager call at this G (its warm-up), then capture with a persistent
    workspace and `out`. The first replay equals the eager call bitwise; two more replays on new q,
    new ring rows and slots (new index rows), new valid counts and a new block buffer, all written
    into the captured buffers, equal a fresh eager call on the same inputs bitwise and match the
    reference.
    """
    gen = torch.Generator(device=DEV).manual_seed(4200 + G)
    sinks = _sinks(DS_HEADS)
    ring = _ds_ring(DS_RING_SLOTS[0] if G < DS_RING_SLOTS[0] else DS_RING_SLOTS[1], gen)
    view = ring.view(-1, 1, WINDOW, HEAD_DIM)
    form = "page" if G % 2 else "row"
    q, block, idx, active, seq_lens = _ds_inputs(G, ring, gen, block_form=form)
    ws = _workspace()
    out = torch.empty_like(q)

    def call():
        return op(
            query=q,
            swa_kv_cache=view,
            compressed_kv_cache=block,
            workspace_buffer=ws,
            sparse_indices=idx,
            sparse_topk_lens=active,
            seq_lens=seq_lens,
            sinks=sinks,
            softmax_scale=SCALE,
            out=out,
        )

    call()
    eager = out.clone()
    counts = _ds_counts(seq_lens)
    _check_atol(
        eager,
        _ds_reference(q, view, block, idx, active, counts, sinks),
        _ds_atol(view, block, idx, counts),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    for shift in (1, 2):
        nq, nblock, nidx, nactive, nseq = _ds_inputs(G, ring, gen, shift=shift, block_form=form)
        q.copy_(nq)
        block.copy_(nblock)
        idx.copy_(nidx)
        active.copy_(nactive)
        seq_lens.copy_(nseq)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        call()  # a fresh eager call on the same inputs
        torch.cuda.synchronize()
        assert torch.equal(replayed, out), f"replay != eager at shift {shift}"
        ncounts = _ds_counts(nseq)
        ref = _ds_reference(nq, view, nblock, nidx, nactive, ncounts, sinks)
        _check_atol(out, ref, _ds_atol(view, nblock, nidx, ncounts))
    del graph


def test_dspark_draft_block_h64_wrong_variants_fail_the_gate():
    """At the drafter surface (G 32, 64 heads) the gate sees every plausible wrong reading.

    Wrong: a causal block (query j attends block rows 0..j), the whole ring page attended (seq_lens
    ignored: the stale decoy rows join), the ring count one short, the block rows dropped
    (sparse_topk_lens 128), the sink ignored.
    """
    gen = torch.Generator(device=DEV).manual_seed(4300)
    sinks = _sinks(DS_HEADS)
    ring = _ds_ring(DS_RING_SLOTS[0], gen)
    view = ring.view(-1, 1, WINDOW, HEAD_DIM)
    G = 32
    q, block, idx, active, seq_lens = _ds_inputs(G, ring, gen)
    o = _run(q, view, block, idx, active, seq_lens, sinks)
    counts = _ds_counts(seq_lens)
    atol = _ds_atol(view, block, idx, counts)

    def excess(ref):
        return ((o.float().reshape(ref.shape) - ref).abs() / (atol + RTOL * ref.abs())).max().item()

    correct = excess(_ds_reference(q, view, block, idx, active, counts, sinks))
    causal = (WINDOW + 1 + torch.arange(DS_BLOCK, device=DEV).repeat(G)).int()
    wrong = {
        "causal block": excess(_ds_reference(q, view, block, idx, causal, counts, sinks)),
        "whole ring page": excess(
            _ds_reference(q, view, block, idx, active, torch.full_like(counts, WINDOW), sinks)
        ),
        "ring count one short": excess(
            _ds_reference(q, view, block, idx, active, counts - 1, sinks)
        ),
        "block rows dropped": excess(
            _ds_reference(q, view, block, idx, torch.full_like(active, WINDOW), counts, sinks)
        ),
        "sink ignored": excess(
            _ds_reference(q, view, block, idx, active, counts, sinks, sink="ignored")
        ),
    }
    print(
        f"draft block h64: gate-normalized error correct {correct:.3f}, "
        f"wrong { {k: round(v, 1) for k, v in wrong.items()} }"
    )
    assert correct <= 1.0, correct
    for name, e in wrong.items():
        assert e > 10.0 * max(correct, 0.1), (
            f"wrong variant {name!r} not discriminated: {e:.2f} vs {correct:.2f}"
        )


def test_dspark_draft_block_zero_seq_lens_reads_a_full_window():
    """seq_lens 0 is not an empty window: every one of the 128 window columns is attended.

    Measured on sm_103 (probe, then pinned here): the output matches the window_count-128 reading
    and fails the window_count-0 reading. A caller must not pass 0 to mean "no ring rows"; rows
    it wants unattended are excluded only by a count >= 1.
    """
    gen = torch.Generator(device=DEV).manual_seed(4400)
    sinks = _sinks(DS_HEADS)
    ring = _ds_ring(DS_RING_SLOTS[0], gen)
    view = ring.view(-1, 1, WINDOW, HEAD_DIM)
    G = 4
    q, block, idx, active, _ = _ds_inputs(G, ring, gen, valid=[WINDOW] * G)
    zero = torch.zeros(G * DS_BLOCK, dtype=torch.int32, device=DEV)
    o = _run(q, view, block, idx, active, zero, sinks)
    full = torch.full((G * DS_BLOCK,), WINDOW, dtype=torch.int64, device=DEV)
    atol = _ds_atol(view, block, idx, full)
    _check_atol(o, _ds_reference(q, view, block, idx, active, full, sinks), atol)
    with pytest.raises(AssertionError):
        _check_atol(
            o, _ds_reference(q, view, block, idx, active, torch.zeros_like(full), sinks), atol
        )


# ---------------------------------------------------------------------------------------------
# pool extent: a loaded -1 index reads one row in 96..127 of its pool (guard-page tests)
# ---------------------------------------------------------------------------------------------
# The kernel loads index columns in groups of 4. A loaded column holding -1 -- the masked tail of a
# compressed segment whose sparse_topk_lens is not a multiple of 4 (the DSpark draft block: lens
# 133, columns 133..135 are -1), or a -1 anywhere in the 128 window columns (all of them are
# loaded) -- makes the call read one row in 96..127 of that pool, and discard it. On a pool with
# fewer than 128 rows that read lands past the pool's end: harmless while the next bytes happen to
# be mapped, an illegal memory access when the allocation ends a mapped range.
#
# The guard-page tests place a pool with the CUDA VMM API so that exactly M rows are mapped from
# its first row and the VA after them is reserved but unmapped: any access at row >= M faults, on
# every run. A fault poisons the CUDA context, so every guard trial runs in a child process (this
# file run as a script) and the parent asserts on the child's output and exit status.


def _vmm_place(nrows: int, mapped_rows: int):
    """A [nrows, 512] bf16 tensor whose first row starts `mapped_rows` rows before the end of a
    mapping, followed by >= 64 MiB of reserved, unmapped VA. Returns (tensor, release)."""
    from cuda.bindings import driver as cu

    def ck(res):
        err = res[0] if isinstance(res, tuple) else res
        assert int(getattr(err, "value", err)) == 0, f"CUDA driver error {err}"
        return res[1] if isinstance(res, tuple) and len(res) >= 2 else None

    need = mapped_rows * HEAD_DIM * 2
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = torch.cuda.current_device()
    gran = ck(
        cu.cuMemGetAllocationGranularity(
            prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
    )
    size = (need + gran - 1) // gran * gran
    rsize = size + 34 * gran
    va = ck(cu.cuMemAddressReserve(rsize, 0, 0, 0))
    handle = ck(cu.cuMemCreate(size, prop, 0))
    ck(cu.cuMemMap(int(va), size, 0, handle, 0))
    acc = cu.CUmemAccessDesc()
    acc.location = prop.location
    acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    ck(cu.cuMemSetAccess(int(va), size, [acc], 1))
    base = int(va) + size - need

    class _Cai:
        def __init__(self):
            self.__cuda_array_interface__ = {
                "shape": (need,),
                "typestr": "|u1",
                "data": (base, False),
                "version": 3,
            }

    raw = torch.as_tensor(_Cai(), device=DEV)
    raw.zero_()
    t = raw.view(torch.bfloat16).view(mapped_rows, HEAD_DIM)[:nrows]

    def release():
        torch.cuda.synchronize()
        ck(cu.cuMemUnmap(int(va), size))
        ck(cu.cuMemRelease(handle))
        ck(cu.cuMemAddressFree(va, rsize))

    return t, release


def _guard_case(case):
    """One trial's inputs: (inputs, which pool is under test, that pool's rows [n, 512])."""
    gen = torch.Generator(device=DEV).manual_seed(case["seed"])
    sinks = _sinks(DS_HEADS)
    ring = _ds_ring(DS_RING_SLOTS[0], gen)
    swa = ring.view(-1, 1, WINDOW, HEAD_DIM)
    kind = case["kind"]
    if kind == "draft":
        # The DSpark draft-block call; the block buffer is the pool under test.
        q, block, idx, active, seq_lens = _ds_inputs(case["G"], ring, gen, block_form=case["form"])
        if case.get("tail") == "repeat":  # masked columns 133..135 name column 132's row, not -1
            idx[:, DS_TOPK:] = idx[:, DS_TOPK - 1 : DS_TOPK]
        inp = dict(
            q=q, swa=swa, idx=idx, active=active, seq_lens=seq_lens, sinks=sinks, form=case["form"]
        )
        return inp, "comp", block.reshape(-1, HEAD_DIM)
    if kind == "engine":
        # An engine-size pool (runtime-value dump figures) whose last row ends the mapping; the call attends its
        # last rows with a -1 loaded from the same pool. Only the attended rows are nonzero.
        rows_n, page = case["rows"], case["page"]
        pool = torch.zeros(rows_n, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        b, h, lens = 4, case["heads"], case["lens"]
        q = torch.randn(b, 1, h, HEAD_DIM, generator=gen, device=DEV).mul_(2).to(torch.bfloat16)
        idx = torch.full((b, 640), -1, dtype=torch.int32, device=DEV)
        small = torch.randn(5 * WINDOW, HEAD_DIM, generator=gen, device=DEV).to(
            torch.bfloat16
        )  # >= 512 rows
        if case["pool"] == "comp":
            idx[:, :WINDOW] = torch.arange(WINDOW, device=DEV, dtype=torch.int32)
            idx[:, WINDOW:lens] = (rows_n - 1 - torch.arange(lens - WINDOW, device=DEV)).int()
            seq_lens = torch.full((b,), WINDOW, dtype=torch.int32, device=DEV)
            inp = dict(swa=small.view(-1, 1, WINDOW, HEAD_DIM), form=page)
        else:
            k = case["window"]  # valid window rows in the pool's last page, then -1
            idx[:, :k] = (rows_n - WINDOW + torch.arange(k, device=DEV)).int()
            idx[:, WINDOW:lens] = torch.arange(lens - WINDOW, device=DEV, dtype=torch.int32)
            seq_lens = torch.full((b,), k, dtype=torch.int32, device=DEV)
            inp = dict(comp=small.view(-1, 1, 1, HEAD_DIM), swa_page=page)
        used = idx[:, :WINDOW] if case["pool"] == "swa" else idx[:, WINDOW:]
        used = torch.unique(used[used >= 0]).long()
        pool[used] = torch.randn(used.numel(), HEAD_DIM, generator=gen, device=DEV).to(
            torch.bfloat16
        )
        active = torch.full((b,), lens, dtype=torch.int32, device=DEV)
        inp.update(q=q, idx=idx, active=active, seq_lens=seq_lens, sinks=_sinks(h))
        return inp, case["pool"], pool
    # The worst / dummy / swa trials run at 64 heads unless the trial names another head count.
    hq = case.get("heads", DS_HEADS)
    sinks = _sinks(hq)
    q = torch.randn(5, 1, hq, HEAD_DIM, generator=gen, device=DEV).mul_(2).to(torch.bfloat16)
    idx = torch.full((5, DS_WIDTH), -1, dtype=torch.int32, device=DEV)
    if kind == "worst":
        # Every query's partial compressed group is [v, .., -1]: v mod 64 in 60..63 reads row 127.
        v, lens = case["v"], case["lens"]
        comp = torch.randn(v + 1, HEAD_DIM, generator=gen, device=DEV).to(torch.bfloat16)
        idx[:, :WINDOW] = torch.arange(WINDOW, device=DEV, dtype=torch.int32)
        idx[:, WINDOW:lens] = v
        active = torch.full((5,), lens, dtype=torch.int32, device=DEV)
        seq_lens = torch.full((5,), WINDOW, dtype=torch.int32, device=DEV)
        inp = dict(q=q, swa=swa, idx=idx, active=active, seq_lens=seq_lens, sinks=sinks, form="row")
        return inp, "comp", comp
    if kind == "dummy":
        # The core's window-only call: index width 128, lens 128, a [1, 1, rows, 512] placeholder as
        # the compressed pool, window rows [0..k-1, -1, ..] with count k (a prefill window's tail).
        k = case["window"]
        dummy = torch.full((case["rows"], HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=DEV)
        idx = torch.full((5, WINDOW), -1, dtype=torch.int32, device=DEV)
        idx[:, :k] = torch.arange(k, device=DEV, dtype=torch.int32)
        active = torch.full((5,), WINDOW, dtype=torch.int32, device=DEV)
        seq_lens = torch.full((5,), k, dtype=torch.int32, device=DEV)
        inp = dict(
            q=q, swa=swa, idx=idx, active=active, seq_lens=seq_lens, sinks=sinks, form="page"
        )
        return inp, "comp", dummy
    assert kind == "swa"
    # A 5-row SWA pool, window row [0..4, -1 x 123] with count 5; the compressed pool is compliant.
    rows = torch.randn(5, HEAD_DIM, generator=gen, device=DEV).to(torch.bfloat16)
    comp = torch.randn(POOL_MIN_ROWS, HEAD_DIM, generator=gen, device=DEV).to(torch.bfloat16)
    idx[:, :5] = torch.arange(5, device=DEV, dtype=torch.int32)
    idx[:, WINDOW:DS_TOPK] = torch.arange(5, device=DEV, dtype=torch.int32)
    active = torch.full((5,), DS_TOPK, dtype=torch.int32, device=DEV)
    seq_lens = torch.full((5,), 5, dtype=torch.int32, device=DEV)
    inp = dict(
        q=q,
        comp=comp.view(-1, 1, 1, HEAD_DIM),
        idx=idx,
        active=active,
        seq_lens=seq_lens,
        sinks=sinks,
    )
    return inp, "swa", rows


def _guard_call(inp, which, rows):
    n = rows.shape[0]
    if which == "comp":
        page = {"row": 1, "page": n}.get(inp["form"], inp["form"])
        swa, comp = inp["swa"], rows.view(n // page, 1, page, HEAD_DIM)
    else:
        page = inp.get("swa_page", n)
        swa, comp = rows.view(n // page, 1, page, HEAD_DIM), inp["comp"]
    return _run(inp["q"], swa, comp, inp["idx"], inp["active"], inp["seq_lens"], inp["sinks"])


def _guard_child(trials):
    """Run guard trials in order in this process; print TRY before and OK after each call."""
    for case in trials:
        inp, which, rows = _guard_case(case)
        n = rows.shape[0]
        if n < POOL_MIN_ROWS:
            safe = torch.zeros(POOL_MIN_ROWS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
            safe[:n] = rows
            rows = safe[:n]
        want = _guard_call(inp, which, rows).clone()
        torch.cuda.synchronize()
        placed, release = _vmm_place(n, case["mapped"])
        placed.copy_(rows)
        print(f"TRY {case['name']}", flush=True)
        got = _guard_call(inp, which, placed)
        torch.cuda.synchronize()
        same = torch.equal(got.view(torch.int16), want.view(torch.int16))
        if case["kind"] == "dummy":  # the NaN placeholder is never read at all
            assert torch.isfinite(got).all(), case["name"]
        release()
        print(f"OK {case['name']} bitwise={int(same)}", flush=True)


def _guard_run(trials):
    p = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--guard-child", json.dumps(trials)],
        env=_CHILD_ENV,
        capture_output=True,
        text=True,
        timeout=900,
    )
    ok = {
        m.group(1): m.group(2) == "1"
        for m in re.finditer(r"^OK (\S+) bitwise=(\d)$", p.stdout, re.M)
    }
    tried = re.findall(r"^TRY (\S+)$", p.stdout, re.M)
    return p, ok, tried


def _draft_trial(G, form, mapped, seed, tail="-1"):
    return dict(
        name=f"draft-G{G}-{form}-tail{tail}-M{mapped}",
        kind="draft",
        G=G,
        form=form,
        tail=tail,
        mapped=mapped,
        seed=seed,
    )


def _worst_trial(v, lens, mapped, heads=None):
    if heads is None:
        return dict(
            name=f"worst-v{v}-lens{lens}-M{mapped}",
            kind="worst",
            v=v,
            lens=lens,
            mapped=mapped,
            seed=5100 + v + lens,
        )
    return dict(
        name=f"worst-v{v}-lens{lens}-M{mapped}-H{heads}",
        kind="worst",
        v=v,
        lens=lens,
        mapped=mapped,
        seed=5100 + v + lens,
        heads=heads,
    )


def _guard_compliant():
    """Trials that must all complete: every pool spans >= 128 rows from its first row (the block
    buffer padded to exactly 128 where 5G < 128, or 5G >= 128 rows of its own), or no -1 is loaded
    from it."""
    trials = [
        _draft_trial(G, form, POOL_MIN_ROWS, 5000 + G)
        for G in range(1, 26)
        for form in ("row", "page")
    ]
    trials += [
        _draft_trial(G, form, 5 * G, 5200 + G) for G in range(26, 33) for form in ("row", "page")
    ]
    trials += [_worst_trial(v, lens, POOL_MIN_ROWS) for v in (60, 62, 63) for lens in (129, 131)]
    trials += [_draft_trial(G, "row", 5 * G, 5300 + G, tail="repeat") for G in (1, 8, 25)]
    trials += [dict(name="swa-5rows-M128", kind="swa", mapped=POOL_MIN_ROWS, seed=5400)]
    trials += [
        dict(
            name=f"window-only-dummy{rows}-window{k}",
            kind="dummy",
            rows=rows,
            window=k,
            mapped=rows,
            seed=5450 + k,
        )
        for rows in (64, 1)
        for k in (1, 5, 128)
    ]
    # The members' head counts (other cubins): the worst index values at 128 mapped rows, and the
    # window-only call's NaN placeholder ending a mapping.
    trials += [
        _worst_trial(v, lens, POOL_MIN_ROWS, heads=h)
        for h in MEMBER_HEADS
        for v in (60, 62, 63)
        for lens in (129, 131)
    ]
    trials += [
        dict(
            name=f"window-only-dummy{rows}-window{k}-H{h}",
            kind="dummy",
            rows=rows,
            window=k,
            mapped=rows,
            seed=5460 + k,
            heads=h,
        )
        for h in MEMBER_HEADS
        for rows in (64, 1)
        for k in (1, 5, 128)
    ]
    engine = [
        ("r2-comp", "comp", REAL_COMP[2], 129, 16, 0),
        ("r2-comp", "comp", REAL_COMP[2], 639, 64, 0),
        ("r1-comp", "comp", REAL_COMP[1], 131, 16, 0),
        ("r1-comp", "comp", REAL_COMP[1], 133, 64, 0),
        ("swa-window5", "swa", REAL_SWA, 133, 16, 5),
        ("swa-window128", "swa", REAL_SWA, 640, 64, 128),
    ]
    engine += [
        (tag, pool, geom, lens, h, window)
        for h in MEMBER_HEADS
        for tag, pool, geom, lens, _, window in engine
    ]
    for tag, pool, (pages, page), lens, h, window in engine:
        trials.append(
            dict(
                name=f"engine-{tag}-lens{lens}-H{h}",
                kind="engine",
                pool=pool,
                rows=pages * page,
                page=page,
                lens=lens,
                heads=h,
                window=window,
                mapped=pages * page,
                seed=5800 + lens + h,
            )
        )
    return trials


# Each control runs one compliant call, then the same call with the pool ending a mapping short
# of 128 rows; the second must fault.
GUARD_FAULTS = {
    "draft_block_G1_row_unpadded": [
        _draft_trial(1, "row", POOL_MIN_ROWS, 5500),
        _draft_trial(1, "row", 5, 5500),
    ],
    "draft_block_G8_page_unpadded": [
        _draft_trial(8, "page", POOL_MIN_ROWS, 5508),
        _draft_trial(8, "page", 40, 5508),
    ],
    "127_rows_are_not_enough": [_worst_trial(62, 129, POOL_MIN_ROWS), _worst_trial(62, 129, 127)],
    "127_rows_are_not_enough_H8": [
        _worst_trial(62, 129, POOL_MIN_ROWS, heads=8),
        _worst_trial(62, 129, 127, heads=8),
    ],
    "127_rows_are_not_enough_H32": [
        _worst_trial(62, 129, POOL_MIN_ROWS, heads=32),
        _worst_trial(62, 129, 127, heads=32),
    ],
    "swa_pool_minus_one_window": [
        dict(name="swa-5rows-M128", kind="swa", mapped=POOL_MIN_ROWS, seed=5400),
        dict(name="swa-5rows-M5", kind="swa", mapped=5, seed=5400),
    ],
}


def test_pool_extent_128_rows_never_fault():
    """With >= 128 mapped rows from each pool's first row nothing faults, and the output is bitwise
    the output on a normal allocation: the DSpark block buffer at every G 1..25 padded to exactly
    128 rows and at every G 26..32 unpadded (130..160 rows of its own), both forms; the worst index
    values (read row 127) at 3 and 1 masked columns; the -1 tail replaced by column 132's row,
    unpadded (no -1 is loaded, so nothing is read past the buffer); a 5-row SWA pool with a -1
    window tail, padded; the window-only call's 64-row (and a 1-row) NaN placeholder ending a
    mapping, with -1 window tails (never read); and the three engine-size pools (runtime-value
    dump figures), each ending a mapping at its last row while the call attends that row and loads
    a -1 from the same pool: nothing is read past a pool's end."""
    trials = _guard_compliant()
    p, ok, tried = _guard_run(trials)
    names = [c["name"] for c in trials]
    assert p.returncode == 0, f"child failed after {tried[-1:]}: {p.stderr[-2000:]}"
    assert list(ok) == names, (list(ok), names)
    assert all(ok.values()), [k for k, v in ok.items() if not v]
    print(
        f"guard page: {len(names)} compliant trials completed, each bitwise = the normal allocation"
    )


@pytest.mark.parametrize("control", list(GUARD_FAULTS))
def test_pool_extent_guard_page_controls_fault(control):
    """The same calls with the pool ending a mapping short of 128 rows fault: the unpadded DSpark
    block buffer (G 1 [5,1,1,512], G 8 [1,1,40,512]), the worst index value with 127 mapped rows
    (row 127 is read, so 127 rows are not enough), and a 5-row SWA pool whose window row carries -1.
    The harness sees the defect: the compliant call before it completed in the same process."""
    first, second = GUARD_FAULTS[control]
    p, ok, tried = _guard_run([first, second])
    out = p.stdout + p.stderr
    assert ok.get(first["name"]) is True, out[-2000:]
    assert tried == [first["name"], second["name"]] and second["name"] not in ok, out[-2000:]
    assert p.returncode != 0
    assert "illegal memory access" in out, out[-2000:]
    print(f"{control}: compliant call ok, short pool faulted (rc {p.returncode})")


def test_overread_rows_never_change_the_output():
    """The row a loaded -1 reads is discarded: with the pool's rows at the head of a 256-row
    allocation, filling every row past them with 0, NaN, Inf or 30.0 (a score that would dominate
    the softmax) leaves the output bitwise unchanged, and it matches the reference."""
    cases = [
        _draft_trial(G, form, 0, 5600 + G) for G, form in ((1, "row"), (8, "page"), (25, "row"))
    ]
    cases += [_worst_trial(62, lens, 0) for lens in (129, 130, 131)]
    for case in cases:
        inp, which, rows = _guard_case(case)
        n = rows.shape[0]
        backing = torch.zeros(2 * POOL_MIN_ROWS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        backing[:n] = rows
        outs = []
        for fill in (0.0, float("nan"), float("inf"), 30.0):
            backing[n:] = fill
            outs.append(_guard_call(inp, which, backing[:n]))
        for fill, o in zip(("nan", "inf", "30.0"), outs[1:]):
            assert torch.equal(o.view(torch.int16), outs[0].view(torch.int16)), (
                f"{case['name']}: {fill} fill changed the output"
            )
        counts = _ds_counts(inp["seq_lens"])
        ref = _ds_reference(
            inp["q"], inp["swa"], rows, inp["idx"], inp["active"], counts, inp["sinks"]
        )
        _check_atol(outs[0], ref, _ds_atol(inp["swa"], rows, inp["idx"], counts))


def test_repeated_tail_equals_minus_one_tail():
    """Filling the draft block's masked columns 133..135 with column 132's row instead of -1 (which
    loads no -1, so reads nothing past the buffer) gives the bitwise-identical output."""
    for G in (1, 8, 25):
        a, which, rows = _guard_case(_draft_trial(G, "row", 0, 5700 + G))
        b, _, rows_b = _guard_case(_draft_trial(G, "row", 0, 5700 + G, tail="repeat"))
        assert torch.equal(rows, rows_b) and not torch.equal(a["idx"], b["idx"])
        backing = torch.zeros(POOL_MIN_ROWS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        backing[: rows.shape[0]] = rows
        oa = _guard_call(a, which, backing[: rows.shape[0]])
        ob = _guard_call(b, which, backing[: rows.shape[0]])
        assert torch.equal(oa.view(torch.int16), ob.view(torch.int16)), G


# ---------------------------------------------------------------------------------------------
# index semantics the contract states
# ---------------------------------------------------------------------------------------------
def test_lengths_mask_the_tails():
    """Window columns >= min(128, seq_lens - q_len + j + 1) and all columns >= sparse_topk_lens are
    masked: here they name decoy rows (every element 30.0, a score that would dominate the softmax)
    and the output still matches the count-bounded reference."""
    gen = torch.Generator(device=DEV).manual_seed(21)
    swa, comp = _pools(gen)
    decoy_swa = 5 * SWA_PAGE + 7
    decoy_comp = 9 * COMP_PAGE + 3
    swa.view(-1, HEAD_DIM)[decoy_swa] = 30.0
    comp.view(-1, HEAD_DIM)[decoy_comp] = 30.0
    h, b = 16, 6
    seq = torch.tensor([1, 10, 100, 127, 128, 4000], dtype=torch.int32, device=DEV)
    idx = torch.empty(b, 640, dtype=torch.int32, device=DEV)
    idx[:, :WINDOW] = decoy_swa
    idx[:, WINDOW:] = decoy_comp
    active = torch.tensor([128, 200, 129, 640, 300, 131], dtype=torch.int32, device=DEV)
    counts = seq.long().clamp(max=WINDOW)
    good = torch.randperm(SWA_PAGES * SWA_PAGE, generator=gen, device=DEV)
    good = good[good != decoy_swa]
    goodc = torch.randperm(COMP_PAGES * COMP_PAGE, generator=gen, device=DEV)
    goodc = goodc[goodc != decoy_comp]
    for r in range(b):
        idx[r, : counts[r]] = good[r * WINDOW : r * WINDOW + int(counts[r])].int()
        nc = int(active[r]) - WINDOW
        idx[r, WINDOW : WINDOW + nc] = goodc[r * 512 : r * 512 + nc].int()
    q = torch.randn(b, 1, h, HEAD_DIM, generator=gen, device=DEV).mul_(2).to(torch.bfloat16)
    sinks = _sinks(h)
    o = _run(q, swa, comp, idx, active, seq, sinks)
    ref = _reference(q.view(b, h, HEAD_DIM), swa, comp, idx, active, counts, sinks)
    swa_real = swa.clone()
    swa_real.view(-1, HEAD_DIM)[decoy_swa] = 0
    comp_real = comp.clone()
    comp_real.view(-1, HEAD_DIM)[decoy_comp] = 0
    _check(o, ref, swa_real, comp_real)


def test_masked_window_columns_are_still_loaded():
    """A masked window column is loaded all the same: a NaN row it names poisons the output
    (0 * NaN in the PV product). Masked columns must hold -1 or name finite rows."""
    gen = torch.Generator(device=DEV).manual_seed(22)
    swa, comp = _pools(gen)
    nan_row = 3 * SWA_PAGE + 1
    swa.view(-1, HEAD_DIM)[nan_row] = float("nan")
    h = 16
    idx = torch.full((1, 640), -1, dtype=torch.int32, device=DEV)
    idx[0, :10] = torch.arange(10, device=DEV).int()
    idx[0, 100] = nan_row  # beyond window_count = 10
    idx[0, WINDOW : WINDOW + 200] = torch.arange(200, device=DEV).int()
    q = torch.randn(1, 1, h, HEAD_DIM, generator=gen, device=DEV).to(torch.bfloat16)
    seq = torch.tensor([10], dtype=torch.int32, device=DEV)
    active = torch.tensor([WINDOW + 200], dtype=torch.int32, device=DEV)
    o = _run(q, swa, comp, idx, active, seq, _sinks(h))
    assert not torch.isfinite(o).all()
    idx[0, 100] = -1
    o = _run(q, swa, comp, idx, active, seq, _sinks(h))
    assert torch.isfinite(o).all()


def test_minus_one_inside_the_active_range_reads_as_a_zero_row():
    """An attended -1 is NOT a mask: it behaves as an all-zero K/V row (score 0, so exp(0 - max)
    joins the denominator). Callers must compact valid entries and bound them with the lengths."""
    gen = torch.Generator(device=DEV).manual_seed(31)
    swa, comp = _pools(gen)
    h, b = 16, 4
    seq = torch.full((b,), 4000, dtype=torch.int32, device=DEV)
    idx = torch.empty(b, 640, dtype=torch.int32, device=DEV)
    for r in range(b):
        idx[r, :WINDOW] = torch.randperm(SWA_PAGES * SWA_PAGE, generator=gen, device=DEV)[
            :WINDOW
        ].int()
        idx[r, WINDOW:] = torch.randperm(COMP_PAGES * COMP_PAGE, generator=gen, device=DEV)[
            :512
        ].int()
    idx[0, 0:128:2] = -1  # holes inside the window
    idx[1, 300:640] = -1  # compressed tail left inside sparse_topk_lens
    idx[2, 5] = -1
    idx[3, :] = -1  # nothing but holes -> exact zeros
    active = torch.full((b,), 640, dtype=torch.int32, device=DEV)
    counts = torch.full((b,), WINDOW, dtype=torch.int64, device=DEV)
    q = torch.randn(b, 1, h, HEAD_DIM, generator=gen, device=DEV).mul_(2).to(torch.bfloat16)
    sinks = _sinks(h)
    o = _run(q, swa, comp, idx, active, seq, sinks)
    qf = q.view(b, h, HEAD_DIM)
    _check(
        o,
        _reference(qf, swa, comp, idx, active, counts, sinks, negative_index="zero_row"),
        swa,
        comp,
    )
    assert torch.equal(o[3], torch.zeros_like(o[3]))
    masked = _reference(qf, swa, comp, idx, active, counts, sinks, negative_index="masked")
    with pytest.raises(AssertionError):
        _check(o[:2], masked[:2], swa, comp)


def test_overlap_is_not_deduplicated():
    """A row named twice (the window and the compressed segment over one aliased pool, or twice in a
    segment) is attended twice; the kernel does not de-duplicate."""
    gen = torch.Generator(device=DEV).manual_seed(41)
    pool, _ = _pools(gen)
    h, b = 16, 3
    idx = torch.empty(b, 256, dtype=torch.int32, device=DEV)
    for r in range(b):
        w = torch.randperm(SWA_PAGES * SWA_PAGE, generator=gen, device=DEV)[:WINDOW].int()
        c = torch.randperm(SWA_PAGES * SWA_PAGE, generator=gen, device=DEV)[:WINDOW].int()
        c[:64] = w[:64]
        c[64:96] = c[96:128]
        idx[r] = torch.cat([w, c])
    active = torch.full((b,), 256, dtype=torch.int32, device=DEV)
    seq = torch.full((b,), 4000, dtype=torch.int32, device=DEV)
    counts = torch.full((b,), WINDOW, dtype=torch.int64, device=DEV)
    q = torch.randn(b, 1, h, HEAD_DIM, generator=gen, device=DEV).mul_(2).to(torch.bfloat16)
    sinks = _sinks(h)
    o = _run(q, pool, pool, idx, active, seq, sinks)
    qf = q.view(b, h, HEAD_DIM)
    _check(o, _reference(qf, pool, pool, idx, active, counts, sinks), pool, pool)
    with pytest.raises(AssertionError):
        _check(o, _reference(qf, pool, pool, idx, active, counts, sinks, dedup=True), pool, pool)


def test_wrong_variants_fail_the_gate():
    """Discrimination of the tolerance: the correct reference passes with margin, and every
    deliberately wrong reading of the sink or of the index set fails it by a wide margin."""
    gen = torch.Generator(device=DEV).manual_seed(51)
    swa, comp = _pools(gen)
    atol = _atol(swa, comp)

    def excess(
        o, ref
    ):  # max over elements of |o - ref| / (atol + rtol * |ref|); <= 1 passes the gate
        return ((o.float().reshape(ref.shape) - ref).abs() / (atol + RTOL * ref.abs())).max().item()

    requests = [(p, 1) for p in (130, 700, 3000, 9000)] + [(0, 200)]
    idx, active, seq_lens, cu = _build_rows(
        requests, gen, 640, swa_pages_per_req=2, comp_pages_per_req=30
    )
    n, h = idx.shape[0], 16
    q = torch.randn(n, h, HEAD_DIM, generator=gen, device=DEV).mul_(2).to(torch.bfloat16)
    sinks = _sinks(h)
    o = _run(q, swa, comp, idx, active, seq_lens, sinks, cu=cu, max_q_len=200)
    counts = _window_counts(seq_lens, cu)
    correct = [excess(o, _reference(q, swa, comp, idx, active, counts, sinks))]
    wrong = {
        name: excess(o, _reference(q, swa, comp, idx, active, counts, sinks, sink=name))
        for name in ("ignored", "numerator", "log2e", "scaled")
    }
    # de-duplicated overlap: 96 window rows repeated in the compressed segment of one aliased pool
    dreq = [(p, 1) for p in (130, 700, 3000, 9000)]
    didx, dactive, dseq, dcu = _build_rows(
        dreq, gen, 640, swa_pages_per_req=2, comp_pages_per_req=30
    )
    didx[:, WINDOW : WINDOW + 96] = didx[:, :96]
    dq = (
        torch.randn(len(dreq), 1, h, HEAD_DIM, generator=gen, device=DEV).mul_(2).to(torch.bfloat16)
    )
    do = _run(dq, swa, swa, didx, dactive, dseq, sinks)
    dqf = dq.view(len(dreq), h, HEAD_DIM)
    dcounts = _window_counts(dseq, dcu)
    correct.append(excess(do, _reference(dqf, swa, swa, didx, dactive, dcounts, sinks)))
    wrong["dedup"] = excess(
        do, _reference(dqf, swa, swa, didx, dactive, dcounts, sinks, dedup=True)
    )
    worst = max(correct)
    print(
        f"gate-normalized error: correct {[round(c, 3) for c in correct]}, wrong "
        f"{ {k: round(v, 1) for k, v in wrong.items()} }"
    )
    assert worst <= 1.0, f"correct reference outside the gate: {correct}"
    for name, e in wrong.items():
        assert e > 10.0 * max(worst, 0.1), (
            f"wrong variant {name!r} not discriminated: {e:.2f} vs {worst:.2f}"
        )


# ---------------------------------------------------------------------------------------------
# call mechanics
# ---------------------------------------------------------------------------------------------
def test_out_buffer_and_cuda_graph_replay():
    gen = torch.Generator(device=DEV).manual_seed(61)
    swa, comp = _pools(gen)
    requests = [(p, 1) for p in (5, 400, 3000, 9000)]
    idx, active, seq_lens, cu = _build_rows(
        requests, gen, 640, swa_pages_per_req=2, comp_pages_per_req=40
    )
    b, h = len(requests), 16
    q = torch.randn(b, 1, h, HEAD_DIM, generator=gen, device=DEV).to(torch.bfloat16)
    sinks = _sinks(h)
    ws = _workspace()
    out = torch.empty(b, 1, h, HEAD_DIM, dtype=torch.bfloat16, device=DEV)

    def call():
        return op(
            query=q,
            swa_kv_cache=swa,
            compressed_kv_cache=comp,
            workspace_buffer=ws,
            sparse_indices=idx,
            sparse_topk_lens=active,
            seq_lens=seq_lens,
            sinks=sinks,
            softmax_scale=SCALE,
            out=out,
        )

    ret = call()
    assert ret.data_ptr() == out.data_ptr()
    eager = out.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    # new inputs in the same buffers replay correctly
    q.copy_(torch.randn(q.shape, generator=gen, device=DEV).to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    ref = _reference(
        q.view(b, h, HEAD_DIM), swa, comp, idx, active, _window_counts(seq_lens, cu), sinks
    )
    _check(out, ref, swa, comp)
    assert math.isfinite(out.float().abs().max().item())


# ---------------------------------------------------------------------------------------------
# The context (varlen prefill) call over the engine's configuration ENVELOPE: up to
# 8192 tokens per step (max_num_tokens), up to 32 requests per step (max_batch_size), cached prefix
# + q_len <= 16384 per request (max_seq_len), H 16 / 64, index width 128 (ratio-0, window only)
# and 640 (window + 512 compressed, ratio 2 or 1). Calls are made exactly as the core target makes
# them in a step: the context call on the first tc rows (q, index rows, lengths and out are views
# of step-sized tensors, seq_lens = kv_lens[:nc], cum_seq_lens_q = cu[:nc + 1]) and, in a mixed
# step, the decode call on the views that start at row tc (seq_lens = kv_lens[nc:B]).
# ---------------------------------------------------------------------------------------------
ENV_T = 8192
ENV_B = 32
ENV_SEQ = 16384
ENV_LAYERS = {"r0": (0, 64), "r2": (2, 64), "r1": (1, 128)}  # ratio, compressed page rows
ENV_SWA_PAGES = 2 * ENV_B + 32  # a ring of 2 SWA pages per request: a 128-row window spans <= 2
ENV_COMP_PAGES = {2: ENV_B * (ENV_SEQ // 2 // 64) + 64, 1: ENV_B * (ENV_SEQ // 128) + 64}

# Every context call the runtime-value dumps showed: (tc, nc, max_q_len) per head count, from the
# boot gates' and the accuracy-gate previews' logs, parsed line by line with an arity check; each
# cell ran at width 128 and 640 in the engine. Three tuples that cannot be a step (tc > nc *
# max_q_len, or max_q_len > tc) are interleaving garbles and are left out.
DUMPED_CTX_CELLS = {
    16: (
        "1/1/1 2/1/2 48/1/48 493/1/493 494/1/494 503/1/503 529/1/529 549/1/549 555/1/555 556/1/556 "
        "562/1/562 569/1/569 571/1/571 573/1/573 578/1/578 579/1/579 581/1/581 586/1/586 590/1/590 "
        "593/1/593 594/1/594 595/1/595 598/1/598 599/1/599 601/1/601 603/1/603 606/1/606 609/1/609 "
        "612/1/612 614/1/614 616/1/616 617/1/617 619/1/619 625/1/625 626/1/626 627/1/627 629/1/629 "
        "631/1/631 632/1/632 634/1/634 635/1/635 637/1/637 640/1/640 641/1/641 642/1/642 643/1/643 "
        "644/1/644 646/1/646 650/1/650 651/1/651 652/1/652 653/1/653 654/1/654 656/1/656 660/1/660 "
        "661/1/661 662/1/662 663/1/663 665/1/665 666/1/666 667/1/667 668/1/668 670/1/670 674/1/674 "
        "675/1/675 676/1/676 679/1/679 680/1/680 681/1/681 682/1/682 683/1/683 684/1/684 687/1/687 "
        "688/1/688 689/1/689 690/1/690 691/1/691 692/1/692 693/1/693 694/1/694 696/1/696 697/1/697 "
        "698/1/698 699/1/699 700/1/700 701/1/701 702/1/702 703/1/703 704/1/704 706/1/706 708/1/708 "
        "710/1/710 711/1/711 712/1/712 713/1/713 714/1/714 718/1/718 719/1/719 720/1/720 721/1/721 "
        "722/1/722 723/1/723 724/1/724 725/1/725 726/1/726 727/1/727 728/1/728 729/1/729 730/1/730 "
        "732/1/732 733/1/733 734/1/734 735/1/735 736/1/736 737/1/737 738/1/738 739/1/739 740/1/740 "
        "741/1/741 742/1/742 743/1/743 745/1/745 746/1/746 747/1/747 748/1/748 749/1/749 751/1/751 "
        "752/1/752 753/1/753 755/1/755 756/1/756 757/1/757 758/1/758 759/1/759 760/1/760 761/1/761 "
        "762/1/762 763/1/763 764/1/764 765/1/765 766/1/766 767/1/767 768/1/768 769/1/769 770/1/770 "
        "771/1/771 772/1/772 773/1/773 774/1/774 775/1/775 776/1/776 777/1/777 778/1/778 779/1/779 "
        "780/1/780 781/1/781 782/1/782 783/1/783 784/1/784 785/1/785 786/1/786 787/1/787 788/1/788 "
        "790/1/790 791/1/791 792/1/792 794/1/794 795/1/795 796/1/796 797/1/797 798/1/798 799/1/799 "
        "800/1/800 801/1/801 802/1/802 803/1/803 804/1/804 805/1/805 806/1/806 807/1/807 808/1/808 "
        "809/1/809 810/1/810 811/1/811 812/1/812 813/1/813 814/1/814 815/1/815 816/1/816 817/1/817 "
        "818/1/818 819/1/819 820/1/820 822/1/822 823/1/823 824/1/824 825/1/825 826/1/826 827/1/827 "
        "828/1/828 829/1/829 830/1/830 831/1/831 832/1/832 833/1/833 834/1/834 835/1/835 836/1/836 "
        "837/1/837 838/1/838 839/1/839 840/1/840 841/1/841 843/1/843 844/1/844 845/1/845 846/1/846 "
        "847/1/847 848/1/848 849/1/849 850/1/850 851/1/851 852/1/852 853/1/853 854/1/854 855/1/855 "
        "856/1/856 857/1/857 858/1/858 859/1/859 860/1/860 861/1/861 862/1/862 863/1/863 864/1/864 "
        "866/1/866 867/1/867 868/1/868 869/1/869 870/1/870 871/1/871 872/1/872 873/1/873 874/1/874 "
        "875/1/875 876/1/876 877/1/877 878/1/878 879/1/879 880/1/880 881/1/881 882/1/882 883/1/883 "
        "884/1/884 885/1/885 886/1/886 887/1/887 888/1/888 889/1/889 890/1/890 891/1/891 892/1/892 "
        "893/1/893 894/1/894 895/1/895 896/1/896 897/1/897 898/1/898 899/1/899 900/1/900 901/1/901 "
        "902/1/902 903/1/903 904/1/904 906/1/906 907/1/907 908/1/908 909/1/909 910/1/910 911/1/911 "
        "912/1/912 913/1/913 914/1/914 915/1/915 916/1/916 917/1/917 918/1/918 919/1/919 920/1/920 "
        "921/1/921 922/1/922 923/1/923 924/1/924 925/1/925 926/1/926 927/1/927 928/1/928 929/1/929 "
        "930/1/930 931/1/931 932/1/932 933/1/933 935/1/935 936/1/936 937/1/937 938/1/938 939/1/939 "
        "940/1/940 941/1/941 942/1/942 943/1/943 944/1/944 945/1/945 946/1/946 947/1/947 949/1/949 "
        "950/1/950 951/1/951 952/1/952 953/1/953 954/1/954 955/1/955 956/1/956 957/1/957 958/1/958 "
        "959/1/959 960/1/960 961/1/961 962/1/962 963/1/963 964/1/964 965/1/965 966/1/966 967/1/967 "
        "968/1/968 969/1/969 970/1/970 972/1/972 973/1/973 974/1/974 976/1/976 977/1/977 979/1/979 "
        "980/1/980 982/1/982 983/1/983 984/1/984 986/1/986 987/1/987 988/1/988 989/1/989 990/1/990 "
        "991/1/991 992/1/992 993/1/993 994/1/994 996/1/996 997/1/997 998/1/998 999/1/999 1000/1/1000 "
        "1002/1/1002 1003/1/1003 1004/1/1004 1005/1/1005 1007/1/1007 1008/1/1008 1011/1/1011 1012/1/1012 "
        "1013/1/1013 1014/1/1014 1015/1/1015 1016/1/1016 1018/1/1018 1019/1/1019 1020/1/1020 1021/1/1021 "
        "1022/1/1022 1023/1/1023 1025/1/1025 1026/1/1026 1028/1/1028 1031/1/1031 1032/1/1032 1034/1/1034 "
        "1036/1/1036 1038/1/1038 1039/1/1039 1040/1/1040 1042/1/1042 1043/1/1043 1045/1/1045 1046/1/1046 "
        "1048/1/1048 1050/1/1050 1055/1/1055 1057/1/1057 1058/1/1058 1062/1/1062 1065/1/1065 1066/1/1066 "
        "1067/1/1067 1068/1/1068 1070/1/1070 1072/1/1072 1076/1/1076 1079/1/1079 1080/1/1080 1081/1/1081 "
        "1082/1/1082 1084/1/1084 1085/1/1085 1086/1/1086 1087/1/1087 1090/1/1090 1091/1/1091 1092/1/1092 "
        "1093/1/1093 1094/1/1094 1095/1/1095 1098/1/1098 1099/1/1099 1100/1/1100 1101/1/1101 1104/1/1104 "
        "1106/1/1106 1109/1/1109 1111/1/1111 1115/1/1115 1118/1/1118 1119/1/1119 1120/1/1120 1122/1/1122 "
        "1125/1/1125 1126/1/1126 1128/1/1128 1129/1/1129 1137/1/1137 1139/1/1139 1140/1/1140 1141/1/1141 "
        "1143/1/1143 1145/1/1145 1146/1/1146 1149/1/1149 1150/1/1150 1157/1/1157 1170/1/1170 1171/1/1171 "
        "1172/1/1172 1178/1/1178 1182/1/1182 1185/1/1185 1187/1/1187 1198/1/1198 1200/1/1200 1204/1/1204 "
        "1211/1/1211 1217/1/1217 1226/1/1226 1236/1/1236 1241/2/629 1261/1/1261 1273/1/1273 1275/2/697 "
        "1283/2/780 1298/1/1298 1301/1/1301 1333/2/670 1345/2/746 1355/2/698 1374/2/768 1388/2/757 "
        "1391/2/730 1395/2/738 1397/2/811 1407/2/858 1408/2/738 1410/2/855 1419/2/723 1423/2/734 "
        "1432/2/938 1433/2/787 1448/2/756 1462/2/774 1464/2/782 1466/2/825 1467/2/877 1473/2/777 "
        "1477/2/792 1481/2/777 1482/2/788 1484/2/835 1494/2/895 1496/2/854 1497/2/818 1498/2/779 "
        "1501/1/1501 1505/2/882 1510/2/820 1513/2/788 1514/2/759 1514/2/844 1518/2/776 1519/2/793 "
        "1520/2/786 1523/2/832 1524/2/825 1525/2/831 1525/2/952 1533/2/780 1537/2/839 1539/2/859 "
        "1542/2/799 1543/2/854 1546/2/863 1546/2/906 1547/2/784 1547/2/917 1550/2/815 1551/2/809 "
        "1551/2/895 1554/2/799 1554/2/850 1561/2/801 1562/2/828 1566/2/784 1567/2/788 1567/2/811 "
        "1567/2/911 1568/2/815 1569/2/888 1570/2/851 1571/2/831 1574/2/811 1574/2/834 1575/2/848 "
        "1579/2/796 1583/2/1034 1584/2/799 1585/2/807 1585/2/815 1591/2/804 1593/2/862 1594/2/801 "
        "1594/2/875 1594/2/918 1595/2/848 1595/2/870 1601/2/849 1601/2/909 1605/2/824 1606/2/847 "
        "1606/2/914 1608/2/873 1610/2/853 1614/2/820 1616/2/941 1620/2/878 1620/2/880 1621/2/967 "
        "1624/2/866 1625/2/869 1626/2/920 1627/2/834 1627/2/931 1630/2/889 1636/2/883 1637/2/822 "
        "1637/2/888 1639/2/954 1640/2/914 1641/2/868 1642/2/1005 1644/2/869 1646/2/873 1647/2/887 "
        "1649/2/871 1654/2/924 1655/2/964 1656/2/845 1661/2/942 1661/2/1064 1662/2/833 1663/2/866 "
        "1663/2/870 1664/2/903 1664/2/962 1665/2/871 1666/2/943 1667/2/910 1668/2/1036 1669/2/896 "
        "1670/2/900 1672/2/841 1672/2/854 1672/2/888 1675/2/856 1675/2/871 1675/2/926 1676/2/880 "
        "1680/2/939 1682/2/930 1685/2/911 1685/2/916 1685/2/919 1686/2/925 1687/2/907 1688/2/851 "
        "1691/2/907 1693/2/996 1695/2/876 1695/2/1109 1696/2/1008 1699/2/1019 1700/2/864 1705/2/876 "
        "1706/2/959 1707/2/856 1709/2/987 1710/2/934 1710/2/1008 1712/2/877 1713/2/906 1714/2/858 "
        "1716/2/877 1719/2/1007 1720/2/917 1720/2/921 1725/2/925 1725/2/973 1726/2/918 1728/2/871 "
        "1728/2/895 1731/2/877 1731/2/903 1737/2/930 1738/2/954 1740/2/1003 1741/2/907 1742/2/950 "
        "1747/2/966 1747/2/990 1748/2/909 1749/2/942 1750/2/918 1752/2/1100 1753/2/993 1754/2/884 "
        "1754/2/915 1755/2/888 1756/2/898 1756/2/951 1757/2/986 1758/2/923 1758/2/1098 1760/2/988 "
        "1764/2/952 1769/2/893 1769/2/944 1770/2/960 1771/2/1091 1772/2/992 1772/2/1048 1773/2/1005 "
        "1775/2/926 1775/2/958 1775/2/1119 1776/2/953 1781/2/893 1781/2/976 1781/2/1019 1783/2/1137 "
        "1785/2/924 1790/2/966 1791/2/930 1797/2/1198 1799/2/910 1801/2/912 1802/2/934 1802/2/980 "
        "1803/2/996 1804/2/914 1805/2/951 1808/2/1079 1812/2/988 1814/2/953 1814/2/983 1817/2/910 "
        "1817/2/912 1817/2/941 1817/2/1089 1818/2/1043 1819/2/977 1820/2/1003 1821/2/1048 1823/2/912 "
        "1824/2/963 1825/2/929 1826/2/1067 1832/2/1143 1838/2/1021 1844/2/1045 1850/2/961 1850/2/1149 "
        "1852/2/928 1855/2/975 1856/2/951 1856/2/977 1857/2/998 1859/2/1000 1863/2/945 1863/2/975 "
        "1864/2/1074 1868/2/947 1869/2/974 1871/2/1026 1873/2/1125 1873/2/1176 1875/2/1122 1876/2/1201 "
        "1878/2/957 1887/2/945 1891/2/1136 1892/2/966 1892/2/992 1895/2/1003 1901/2/1050 1902/2/1087 "
        "1910/2/1011 1910/2/1028 1912/2/1068 1913/2/1086 1920/2/1081 1921/2/1201 1924/2/1004 1925/2/1154 "
        "1926/2/977 1926/2/1031 1926/2/1099 1931/2/1043 1934/2/1920 1935/2/1058 1939/2/1100 1964/2/1058 "
        "1968/2/994 1968/2/1091 1972/2/1076 1976/2/988 1976/2/1008 1977/2/1119 1979/2/1053 1980/2/1015 "
        "1980/2/1036 1982/2/1038 1982/2/1053 1987/2/1045 1988/2/1011 1993/2/1024 1997/2/1032 1998/2/999 "
        "2004/2/1049 2007/2/1176 2009/2/1130 2013/2/1097 2016/2/1154 2017/2/1038 2017/2/1104 2019/2/1094 "
        "2026/2/1118 2026/3/811 2034/2/1040 2045/2/1080 2045/2/1092 2048/1/2048 2051/2/1057 2054/2/1099 "
        "2072/2/1170 2073/2/1080 2077/2/1066 2085/3/730 2092/2/1198 2094/2/1167 2117/2/1109 2121/2/1082 "
        "2135/2/1501 2140/2/1128 2164/2/1185 2165/2/1273 2179/2/1136 2193/2/1177 2213/3/900 2228/2/1139 "
        "2232/3/835 2260/2/1145 2313/3/910 2349/2/1237 2367/3/1018 2369/3/891 2401/3/930 2411/3/925 "
        "2413/2/1298 2419/3/846 2433/3/936 2457/3/896 2465/3/923 2466/3/896 2480/3/936 2489/3/883 "
        "2498/3/884 2506/3/1023 2541/3/936 2556/3/872 2626/3/1065 2643/3/1095 2654/3/963 2687/3/999 "
        "2693/3/945 2713/3/999 2715/3/942 2721/3/1140 2728/3/1045 2749/3/1177 2769/3/1080 2813/3/1170 "
        "2819/3/989 2851/3/1081 2866/3/1085 2910/3/1049 2968/3/1261 3206/4/1114 3250/4/911 3289/4/867 "
        "3336/3/1237 8102/10/1085 8141/9/1163 8153/10/1318 8192/1/8192 "
    ),
    64: (
        "1/1/1 2/1/2 14/1/14 16/1/16 48/1/48 549/1/549 550/1/550 562/1/562 571/1/571 586/1/586 590/1/590 "
        "597/1/597 599/1/599 601/1/601 606/1/606 609/1/609 616/1/616 619/1/619 623/1/623 626/1/626 "
        "627/1/627 630/1/630 631/1/631 632/1/632 634/1/634 635/1/635 637/1/637 643/1/643 644/1/644 "
        "646/1/646 649/1/649 650/1/650 651/1/651 652/1/652 653/1/653 654/1/654 656/1/656 661/1/661 "
        "662/1/662 663/1/663 666/1/666 667/1/667 668/1/668 670/1/670 672/1/672 674/1/674 675/1/675 "
        "676/1/676 680/1/680 681/1/681 682/1/682 683/1/683 684/1/684 685/1/685 688/1/688 689/1/689 "
        "690/1/690 691/1/691 694/1/694 696/1/696 697/1/697 698/1/698 699/1/699 701/1/701 702/1/702 "
        "704/1/704 706/1/706 711/1/711 712/1/712 713/1/713 715/1/715 719/1/719 720/1/720 721/1/721 "
        "722/1/722 723/1/723 725/1/725 726/1/726 727/1/727 728/1/728 729/1/729 730/1/730 732/1/732 "
        "733/1/733 734/1/734 735/1/735 737/1/737 738/1/738 739/1/739 740/1/740 741/1/741 742/1/742 "
        "743/1/743 745/1/745 746/1/746 747/1/747 748/1/748 749/1/749 751/1/751 752/1/752 753/1/753 "
        "754/1/754 755/1/755 756/1/756 757/1/757 758/1/758 759/1/759 760/1/760 761/1/761 762/1/762 "
        "763/1/763 764/1/764 765/1/765 766/1/766 767/1/767 768/1/768 769/1/769 770/1/770 771/1/771 "
        "773/1/773 774/1/774 775/1/775 776/1/776 777/1/777 778/1/778 779/1/779 780/1/780 781/1/781 "
        "782/1/782 783/1/783 784/1/784 785/1/785 786/1/786 787/1/787 788/1/788 790/1/790 791/1/791 "
        "792/1/792 793/1/793 794/1/794 795/1/795 796/1/796 797/1/797 798/1/798 799/1/799 800/1/800 "
        "801/1/801 802/1/802 803/1/803 804/1/804 805/1/805 806/1/806 807/1/807 808/1/808 809/1/809 "
        "810/1/810 811/1/811 812/1/812 813/1/813 814/1/814 815/1/815 816/1/816 817/1/817 818/1/818 "
        "819/1/819 820/1/820 822/1/822 823/1/823 824/1/824 825/1/825 826/1/826 827/1/827 828/1/828 "
        "829/1/829 830/1/830 831/1/831 832/1/832 833/1/833 834/1/834 835/1/835 836/1/836 837/1/837 "
        "838/1/838 839/1/839 840/1/840 841/1/841 843/1/843 844/1/844 845/1/845 846/1/846 847/1/847 "
        "848/1/848 849/1/849 850/1/850 851/1/851 852/1/852 853/1/853 854/1/854 855/1/855 856/1/856 "
        "857/1/857 858/1/858 859/1/859 860/1/860 861/1/861 862/1/862 863/1/863 864/1/864 866/1/866 "
        "867/1/867 868/1/868 869/1/869 870/1/870 871/1/871 872/1/872 873/1/873 874/1/874 875/1/875 "
        "876/1/876 877/1/877 878/1/878 879/1/879 880/1/880 881/1/881 882/1/882 883/1/883 884/1/884 "
        "885/1/885 886/1/886 887/1/887 888/1/888 889/1/889 890/1/890 891/1/891 892/1/892 893/1/893 "
        "894/1/894 895/1/895 896/1/896 897/1/897 898/1/898 899/1/899 900/1/900 901/1/901 902/1/902 "
        "903/1/903 904/1/904 906/1/906 907/1/907 908/1/908 909/1/909 910/1/910 911/1/911 912/1/912 "
        "913/1/913 914/1/914 915/1/915 916/1/916 917/1/917 918/1/918 919/1/919 920/1/920 921/1/921 "
        "922/1/922 923/1/923 924/1/924 925/1/925 926/1/926 927/1/927 928/1/928 929/1/929 930/1/930 "
        "931/1/931 932/1/932 933/1/933 934/1/934 936/1/936 937/1/937 939/1/939 940/1/940 941/1/941 "
        "942/1/942 943/1/943 944/1/944 945/1/945 946/1/946 947/1/947 949/1/949 950/1/950 951/1/951 "
        "952/1/952 953/1/953 954/1/954 955/1/955 957/1/957 958/1/958 959/1/959 961/1/961 962/1/962 "
        "963/1/963 964/1/964 965/1/965 966/1/966 967/1/967 968/1/968 970/1/970 972/1/972 973/1/973 "
        "974/1/974 975/1/975 976/1/976 977/1/977 979/1/979 980/1/980 983/1/983 984/1/984 986/1/986 "
        "987/1/987 988/1/988 989/1/989 990/1/990 991/1/991 992/1/992 993/1/993 994/1/994 996/1/996 "
        "997/1/997 998/1/998 999/1/999 1000/1/1000 1002/1/1002 1003/1/1003 1004/1/1004 1005/1/1005 "
        "1007/1/1007 1008/1/1008 1011/1/1011 1012/1/1012 1013/1/1013 1014/1/1014 1015/1/1015 1016/1/1016 "
        "1018/1/1018 1019/1/1019 1020/1/1020 1021/1/1021 1022/1/1022 1023/1/1023 1025/1/1025 1026/1/1026 "
        "1028/1/1028 1031/1/1031 1032/1/1032 1034/1/1034 1036/1/1036 1038/1/1038 1039/1/1039 1040/1/1040 "
        "1043/1/1043 1046/1/1046 1048/1/1048 1050/1/1050 1055/1/1055 1057/1/1057 1058/1/1058 1062/1/1062 "
        "1064/1/1064 1065/1/1065 1067/1/1067 1068/1/1068 1070/1/1070 1079/1/1079 1080/1/1080 1081/1/1081 "
        "1082/1/1082 1084/1/1084 1085/1/1085 1089/1/1089 1090/1/1090 1092/1/1092 1093/1/1093 1094/1/1094 "
        "1095/1/1095 1099/1/1099 1100/1/1100 1101/1/1101 1104/1/1104 1106/1/1106 1111/1/1111 1115/1/1115 "
        "1119/1/1119 1120/1/1120 1122/1/1122 1126/1/1126 1130/1/1130 1135/1/1135 1136/1/1136 1137/1/1137 "
        "1139/1/1139 1141/1/1141 1142/1/1142 1143/1/1143 1145/1/1145 1149/1/1149 1150/1/1150 1154/1/1154 "
        "1157/1/1157 1160/1/1160 1170/1/1170 1171/1/1171 1172/1/1172 1177/1/1177 1178/1/1178 1182/1/1182 "
        "1185/1/1185 1187/1/1187 1198/1/1198 1200/1/1200 1201/1/1201 1204/1/1204 1211/1/1211 1217/1/1217 "
        "1226/1/1226 1237/1/1237 1261/1/1261 1273/1/1273 1301/1/1301 1313/2/764 1344/2/698 1403/2/740 "
        "1424/2/730 1439/2/729 1462/2/770 1464/2/753 1472/2/742 1507/2/839 1508/2/847 1511/2/780 "
        "1512/2/784 1528/2/771 1535/2/848 1538/2/775 1562/2/841 1566/2/854 1574/2/797 1574/2/834 "
        "1575/2/848 1580/2/854 1584/2/826 1596/2/860 1601/2/909 1605/2/842 1606/2/836 1615/2/858 "
        "1615/2/859 1626/2/920 1627/2/834 1629/2/825 1637/2/913 1647/2/887 1656/2/845 1661/2/942 "
        "1662/2/943 1663/2/858 1663/2/908 1668/2/849 1678/2/904 1680/2/908 1683/2/924 1684/2/916 "
        "1685/2/916 1687/2/923 1689/2/965 1694/2/848 1696/2/1008 1701/2/883 1702/2/1003 1703/2/919 "
        "1704/2/879 1705/2/856 1707/2/856 1709/2/857 1709/2/987 1710/2/1129 1712/2/910 1716/2/912 "
        "1720/2/917 1723/2/990 1724/2/900 1734/2/929 1734/2/950 1735/2/998 1742/2/893 1742/2/950 "
        "1743/2/917 1743/2/964 1744/2/1004 1747/2/891 1747/2/1091 1750/2/921 1750/2/926 1752/2/1087 "
        "1757/2/986 1758/2/923 1758/2/1098 1759/2/1023 1764/2/952 1766/2/935 1780/2/929 1782/2/916 "
        "1791/2/956 1793/2/915 1794/2/1076 1799/2/1119 1802/2/1146 1808/2/969 1810/2/1067 1812/2/988 "
        "1820/2/952 1820/2/984 1820/2/1003 1825/2/929 1825/2/930 1828/2/962 1828/2/1060 1836/2/1122 "
        "1837/2/974 1843/2/947 1856/2/951 1856/2/1122 1861/2/955 1864/2/1074 1869/2/941 1873/2/965 "
        "1873/2/1109 1873/2/1176 1882/2/1170 1885/2/966 1911/2/969 1913/2/1086 1914/2/1025 1921/2/1045 "
        "1921/2/1070 1937/2/982 1939/2/1100 1947/2/1150 1952/2/1008 1953/2/1085 1954/2/1066 1956/2/999 "
        "1969/2/1024 1969/2/1031 1972/2/1076 1973/2/1012 1973/2/1118 1975/2/1055 1984/2/1094 2008/2/1080 "
        "2013/2/1097 2019/2/1125 2020/2/1319 2048/1/2048 2054/4/688 2077/2/1066 2082/2/1112 2105/2/1128 "
        "2107/2/1058 2117/2/1109 2117/2/1140 2135/2/1501 2146/2/1236 2155/3/914 2160/2/1139 2163/2/1091 "
        "2333/3/836 2413/2/1298 2511/3/919 2523/3/977 2605/3/938 2629/3/877 2644/3/1000 2698/3/1015 "
        "2791/3/977 2851/3/1081 2864/3/1045 3114/5/702 3162/5/701 3278/5/697 3601/4/1057 8071/12/833 "
        "8095/10/927 8117/9/1097 8119/10/942 8135/10/950 8136/10/944 8150/8/1320 8156/11/835 8160/11/842 "
        "8161/11/823 8178/8/1141 8178/8/1165 8192/1/8192 "
    ),
}

# The same for the parallel members' head counts: every context call (tc, nc, max_q_len) in mn's
# per-rank runtime-value dumps at H 8 (tep8 boot and GSM8K) and H 32 (pp2 x tp2 GSM8K), the union
# parsed tuple by tuple with an arity check (the members' runtime-value dumps). Every parsed cell is a
# possible step (max_q_len <= tc <= nc * max_q_len); each ran at width 128 and 640 in the engine.
MEMBER_DUMPED_CTX_CELLS = {
    8: (
        "1/1/1 2/1/2 16/1/16 48/1/48 128/1/128 256/2/128 384/3/128 512/4/128 549/1/549 581/1/581 590/1/590 "
        "597/1/597 599/1/599 609/1/609 616/1/616 619/1/619 621/1/621 622/1/622 623/1/623 626/1/626 "
        "630/1/630 634/1/634 635/1/635 637/1/637 640/1/640 640/5/128 643/1/643 644/1/644 646/1/646 "
        "649/1/649 650/1/650 653/1/653 656/1/656 661/1/661 663/1/663 665/1/665 666/1/666 667/1/667 "
        "668/1/668 670/1/670 674/1/674 675/1/675 679/1/679 680/1/680 681/1/681 683/1/683 684/1/684 "
        "685/1/685 687/1/687 688/1/688 689/1/689 690/1/690 691/1/691 692/1/692 694/1/694 696/1/696 "
        "697/1/697 698/1/698 699/1/699 701/1/701 702/1/702 704/1/704 706/1/706 707/1/707 710/1/710 "
        "711/1/711 712/1/712 714/1/714 718/1/718 719/1/719 721/1/721 722/1/722 723/1/723 724/1/724 "
        "725/1/725 726/1/726 727/1/727 728/1/728 729/1/729 730/1/730 731/1/731 733/1/733 734/1/734 "
        "735/1/735 736/1/736 737/1/737 738/1/738 740/1/740 741/1/741 742/1/742 743/1/743 745/1/745 "
        "747/1/747 749/1/749 752/1/752 753/1/753 755/1/755 756/1/756 757/1/757 758/1/758 759/1/759 "
        "760/1/760 762/1/762 763/1/763 764/1/764 765/1/765 766/1/766 767/1/767 768/1/768 768/6/128 "
        "769/1/769 770/1/770 771/1/771 773/1/773 774/1/774 775/1/775 776/1/776 777/1/777 778/1/778 "
        "779/1/779 780/1/780 781/1/781 782/1/782 783/1/783 784/1/784 785/1/785 786/1/786 787/1/787 "
        "788/1/788 789/1/789 790/1/790 791/1/791 792/1/792 793/1/793 795/1/795 796/1/796 797/1/797 "
        "799/1/799 800/1/800 801/1/801 802/1/802 803/1/803 806/1/806 807/1/807 808/1/808 809/1/809 "
        "810/1/810 811/1/811 812/1/812 814/1/814 815/1/815 816/1/816 817/1/817 818/1/818 819/1/819 "
        "820/1/820 822/1/822 823/1/823 824/1/824 825/1/825 826/1/826 827/1/827 828/1/828 829/1/829 "
        "830/1/830 831/1/831 832/1/832 833/1/833 834/1/834 835/1/835 836/1/836 837/1/837 838/1/838 "
        "839/1/839 841/1/841 842/1/842 844/1/844 845/1/845 846/1/846 847/1/847 848/1/848 849/1/849 "
        "850/1/850 851/1/851 852/1/852 853/1/853 854/1/854 855/1/855 856/1/856 857/1/857 858/1/858 "
        "859/1/859 860/1/860 861/1/861 862/1/862 863/1/863 864/1/864 866/1/866 867/1/867 868/1/868 "
        "869/1/869 870/1/870 871/1/871 872/1/872 873/1/873 874/1/874 875/1/875 876/1/876 877/1/877 "
        "878/1/878 879/1/879 880/1/880 881/1/881 882/1/882 883/1/883 884/1/884 887/1/887 888/1/888 "
        "889/1/889 890/1/890 892/1/892 893/1/893 894/1/894 895/1/895 896/1/896 897/1/897 899/1/899 "
        "900/1/900 901/1/901 903/1/903 904/1/904 905/1/905 906/1/906 907/1/907 908/1/908 909/1/909 "
        "910/1/910 911/1/911 912/1/912 913/1/913 914/1/914 915/1/915 916/1/916 917/1/917 918/1/918 "
        "919/1/919 920/1/920 921/1/921 922/1/922 923/1/923 924/1/924 925/1/925 926/1/926 927/1/927 "
        "928/1/928 929/1/929 930/1/930 931/1/931 933/1/933 934/1/934 936/1/936 938/1/938 939/1/939 "
        "941/1/941 942/1/942 943/1/943 944/1/944 945/1/945 946/1/946 947/1/947 949/1/949 950/1/950 "
        "951/1/951 952/1/952 953/1/953 954/1/954 956/1/956 957/1/957 958/1/958 959/1/959 960/1/960 "
        "961/1/961 963/1/963 964/1/964 965/1/965 966/1/966 967/1/967 968/1/968 969/1/969 970/1/970 "
        "975/1/975 977/1/977 979/1/979 980/1/980 981/1/981 982/1/982 983/1/983 984/1/984 986/1/986 "
        "987/1/987 988/1/988 989/1/989 991/1/991 992/1/992 993/1/993 994/1/994 996/1/996 998/1/998 "
        "999/1/999 1000/1/1000 1003/1/1003 1004/1/1004 1005/1/1005 1007/1/1007 1008/1/1008 1011/1/1011 "
        "1012/1/1012 1013/1/1013 1014/1/1014 1015/1/1015 1016/1/1016 1018/1/1018 1019/1/1019 1021/1/1021 "
        "1024/1/1024 1025/1/1025 1026/1/1026 1031/1/1031 1036/1/1036 1038/1/1038 1039/1/1039 1040/1/1040 "
        "1042/1/1042 1043/1/1043 1046/1/1046 1048/1/1048 1049/1/1049 1050/1/1050 1055/1/1055 1057/1/1057 "
        "1058/1/1058 1062/1/1062 1064/1/1064 1066/1/1066 1067/1/1067 1068/1/1068 1070/1/1070 1074/1/1074 "
        "1076/1/1076 1081/1/1081 1082/1/1082 1084/1/1084 1085/1/1085 1087/1/1087 1089/1/1089 1091/1/1091 "
        "1092/1/1092 1093/1/1093 1094/1/1094 1098/1/1098 1099/1/1099 1100/1/1100 1101/1/1101 1102/1/1102 "
        "1104/1/1104 1106/1/1106 1111/1/1111 1112/1/1112 1115/1/1115 1119/1/1119 1120/1/1120 1122/1/1122 "
        "1128/1/1128 1130/1/1130 1136/1/1136 1137/1/1137 1139/1/1139 1140/1/1140 1141/1/1141 1143/1/1143 "
        "1145/1/1145 1146/1/1146 1149/1/1149 1152/9/128 1170/1/1170 1171/1/1171 1172/1/1172 1178/1/1178 "
        "1187/1/1187 1198/1/1198 1200/1/1200 1201/1/1201 1204/1/1204 1211/1/1211 1236/1/1236 1237/1/1237 "
        "1241/2/629 1261/1/1261 1273/1/1273 1275/2/697 1285/2/688 1298/1/1298 1313/2/764 1326/2/727 "
        "1336/2/676 1350/2/719 1374/2/768 1376/2/757 1380/2/745 1388/2/757 1397/2/811 1412/2/780 "
        "1419/2/723 1423/2/734 1423/2/773 1433/2/787 1439/2/729 1442/2/723 1447/2/751 1450/2/796 "
        "1452/2/730 1458/2/833 1460/2/780 1464/2/797 1466/2/805 1472/2/742 1473/2/740 1476/2/747 "
        "1477/2/742 1477/2/758 1477/2/792 1492/2/876 1495/2/794 1497/2/807 1497/2/836 1499/2/829 "
        "1506/2/944 1507/2/784 1507/2/839 1511/2/780 1512/2/854 1513/2/788 1523/2/785 1525/2/847 "
        "1526/2/807 1526/2/826 1526/2/850 1527/2/836 1528/2/771 1530/2/898 1531/2/799 1531/2/831 "
        "1533/2/776 1535/2/801 1536/2/858 1536/2/874 1538/2/793 1539/2/859 1539/2/877 1540/2/909 "
        "1544/2/901 1546/2/863 1546/2/895 1547/2/805 1547/2/834 1547/2/847 1547/2/893 1548/2/791 "
        "1548/2/815 1549/2/807 1564/2/898 1564/2/938 1566/2/784 1567/2/788 1568/2/815 1569/2/873 "
        "1570/2/828 1571/2/874 1572/2/874 1573/2/846 1574/2/793 1574/2/811 1575/2/789 1577/2/824 "
        "1581/2/823 1581/2/832 1581/2/950 1582/2/824 1584/2/825 1585/2/804 1585/2/807 1587/2/797 "
        "1589/2/990 1590/2/824 1590/2/889 1592/2/881 1594/2/825 1594/2/875 1594/2/918 1595/2/832 "
        "1595/2/874 1596/2/863 1597/2/1019 1599/2/859 1603/2/803 1608/2/862 1608/2/868 1610/2/853 "
        "1610/2/926 1613/2/910 1614/2/858 1615/2/893 1616/2/888 1618/2/920 1619/2/943 1620/2/880 "
        "1620/2/926 1622/2/858 1623/2/922 1624/2/866 1625/2/869 1626/2/920 1627/2/931 1628/2/841 "
        "1628/2/850 1630/2/939 1631/2/867 1632/2/854 1634/2/881 1636/2/925 1637/2/822 1638/2/847 "
        "1638/2/879 1640/2/994 1641/2/868 1641/2/919 1642/2/1005 1642/2/1011 1644/2/869 1645/2/830 "
        "1645/2/869 1646/2/873 1647/2/887 1647/2/998 1649/2/864 1650/2/923 1652/2/874 1652/2/877 "
        "1653/2/900 1655/2/878 1655/2/964 1656/2/919 1659/2/831 1660/2/872 1660/2/875 1660/2/912 "
        "1661/2/871 1662/2/833 1664/2/897 1665/2/871 1666/2/848 1667/2/914 1670/2/900 1672/2/841 "
        "1675/2/871 1675/2/963 1680/2/953 1682/2/930 1684/2/847 1684/2/924 1685/2/910 1685/2/911 "
        "1685/2/919 1686/2/944 1687/2/849 1687/2/923 1688/2/877 1688/2/928 1689/2/883 1689/2/965 "
        "1690/2/941 1692/2/895 1692/2/917 1692/2/929 1693/2/852 1696/2/970 1697/2/896 1697/2/921 "
        "1698/2/886 1700/2/867 1702/2/856 1703/2/883 1704/2/879 1705/2/928 1705/2/991 1706/2/953 "
        "1706/2/959 1709/2/880 1709/2/987 1710/2/1008 1712/2/908 1712/2/910 1713/2/943 1714/2/909 "
        "1716/2/877 1716/2/912 1717/2/898 1717/2/965 1719/2/921 1722/2/915 1724/2/900 1724/2/954 "
        "1725/2/973 1726/2/876 1726/2/918 1727/2/965 1728/2/866 1730/2/887 1730/2/929 1731/2/877 "
        "1731/2/911 1731/2/962 1734/2/929 1734/2/950 1737/2/930 1738/2/954 1739/2/891 1742/2/925 "
        "1742/2/961 1744/2/1004 1745/2/937 1745/2/941 1747/2/999 1751/2/945 1752/2/1100 1753/2/993 "
        "1754/2/884 1756/2/951 1758/2/923 1758/2/1098 1760/2/911 1762/2/882 1763/2/901 1764/2/884 "
        "1765/2/924 1765/2/926 1766/2/935 1769/2/942 1770/2/940 1770/2/960 1771/2/894 1772/2/977 "
        "1775/2/926 1775/2/958 1775/2/969 1776/2/953 1778/2/920 1780/2/891 1781/2/976 1782/2/896 "
        "1782/2/1081 1784/2/1007 1786/2/910 1786/2/1067 1786/2/1130 1789/2/931 1789/2/994 1790/2/943 "
        "1791/2/911 1791/2/930 1791/2/1031 1792/2/939 1792/2/1065 1794/2/994 1797/2/910 1798/2/994 "
        "1799/2/904 1800/2/949 1800/2/992 1801/2/907 1801/2/983 1802/2/920 1802/2/980 1803/2/911 "
        "1804/2/1008 1805/2/951 1806/2/1040 1806/2/1085 1807/2/911 1808/2/969 1808/2/1079 1809/2/935 "
        "1810/2/1067 1812/2/988 1812/2/1020 1813/2/982 1814/2/926 1814/2/947 1814/2/953 1814/2/983 "
        "1814/2/1034 1816/2/944 1816/2/1025 1817/2/909 1817/2/910 1818/2/1043 1819/2/912 1819/2/966 "
        "1820/2/920 1821/2/1048 1824/2/963 1824/2/1026 1824/2/1140 1825/2/930 1827/2/1058 1828/2/915 "
        "1829/2/940 1832/2/997 1833/2/918 1837/2/945 1837/2/1002 1839/2/927 1840/2/989 1841/2/1157 "
        "1844/2/973 1845/2/988 1846/2/1099 1847/2/979 1851/2/967 1855/2/939 1858/2/991 1859/2/1000 "
        "1860/2/1094 1861/2/932 1861/2/955 1862/2/1177 1863/2/975 1865/2/972 1867/2/1081 1868/2/1172 "
        "1869/2/1003 1871/2/1003 1872/2/968 1873/2/1109 1873/2/1125 1874/2/1143 1875/2/958 1877/2/958 "
        "1877/2/1070 1879/2/983 1881/2/983 1882/2/1008 1883/2/1019 1889/2/1101 1891/2/1068 1895/2/955 "
        "1895/2/1003 1897/2/1094 1901/2/1057 1902/2/1067 1908/2/1062 1909/2/956 1909/2/965 1909/2/982 "
        "1910/2/1028 1911/2/1057 1913/2/984 1913/2/1024 1915/2/1045 1918/2/1020 1921/2/1201 1924/2/1004 "
        "1926/2/1031 1928/2/974 1934/2/1046 1935/2/1058 1936/2/1920 1939/2/1011 1940/2/1092 1941/2/1093 "
        "1943/2/980 1945/2/988 1950/2/984 1950/2/1004 1951/2/1122 1952/2/1008 1956/2/986 1957/2/1050 "
        "1960/2/1066 1966/2/1036 1968/2/994 1969/2/1031 1975/2/1055 1976/2/988 1979/2/1053 1980/2/1036 "
        "1982/2/1053 1983/2/1149 1986/2/1072 1987/2/1045 1993/2/1014 1995/2/1139 2000/2/1122 2007/2/1176 "
        "2008/2/1080 2010/2/1064 2011/2/1021 2013/2/1097 2015/2/1211 2018/2/1137 2026/2/1118 2029/2/1085 "
        "2039/2/1021 2042/2/1080 2048/1/2048 2050/2/1115 2059/2/1141 2067/2/1143 2072/2/1170 2072/3/811 "
        "2077/2/1066 2078/3/742 2079/2/1109 2093/2/1085 2093/3/721 2100/2/1102 2102/2/1141 2104/2/1204 "
        "2118/2/1141 2121/2/1182 2132/2/1100 2139/3/720 2142/2/1200 2156/2/1120 2168/2/1143 2174/2/1182 "
        "2191/2/1143 2193/2/1177 2199/3/873 2202/2/1126 2203/3/777 2206/3/787 2210/2/1119 2214/3/794 "
        "2230/2/1185 2247/2/1128 2250/3/826 2252/2/1501 2286/3/832 2289/3/881 2306/3/842 2307/2/1217 "
        "2316/3/895 2318/3/811 2327/3/859 2332/3/880 2335/3/990 2342/3/877 2346/3/848 2353/3/895 "
        "2354/3/944 2355/3/815 2364/3/900 2366/3/858 2367/3/881 2369/3/855 2369/3/891 2370/3/881 "
        "2371/3/875 2375/3/874 2379/3/828 2379/3/909 2386/3/926 2388/3/898 2404/3/914 2405/3/858 "
        "2407/3/984 2410/3/831 2412/3/913 2421/3/900 2423/3/998 2424/2/1226 2424/3/857 2427/3/913 "
        "2428/3/834 2429/3/875 2433/3/848 2441/3/924 2442/3/980 2446/3/852 2447/3/959 2449/3/964 "
        "2452/3/844 2455/3/950 2467/3/872 2468/3/952 2470/3/909 2472/3/963 2480/3/1000 2485/3/893 "
        "2485/3/1038 2496/3/912 2502/3/910 2514/3/981 2515/3/914 2517/3/914 2521/3/923 2523/3/977 "
        "2524/3/856 2524/3/1019 2525/3/885 2535/3/958 2541/3/936 2542/3/925 2546/3/876 2552/3/911 "
        "2553/3/856 2560/3/887 2562/3/929 2562/3/945 2563/3/910 2563/3/915 2564/3/930 2569/3/910 "
        "2571/3/951 2575/3/952 2577/3/950 2581/3/996 2597/3/933 2600/3/888 2610/3/941 2611/3/903 "
        "2615/3/977 2616/3/947 2622/3/1048 2623/3/1031 2625/3/884 2625/3/1154 2626/3/921 2626/3/954 "
        "2629/3/933 2629/3/1032 2631/3/927 2639/3/994 2640/3/1015 2643/3/973 2652/3/996 2666/3/908 "
        "2672/3/998 2673/3/937 2673/3/973 2677/3/924 2678/3/1008 2681/3/947 2684/3/953 2689/3/1028 "
        "2689/3/1095 2691/3/1099 2697/3/1146 2698/3/951 2698/3/980 2701/3/1176 2706/3/1067 2713/3/928 "
        "2717/3/999 2719/3/1086 2720/3/1034 2726/3/997 2739/3/1084 2743/3/1019 2744/3/1046 2763/3/941 "
        "2764/3/1036 2774/3/999 2777/3/1099 2780/3/1118 2789/3/1126 2792/3/957 2793/3/1109 2800/3/1000 "
        "2813/3/957 2814/3/1015 2843/3/1048 2850/3/1139 2858/3/1044 2866/3/1119 2867/3/1048 2869/3/1109 "
        "2870/3/1273 2874/3/1082 2886/3/1295 2886/3/1501 2890/3/1021 2910/3/1049 2918/3/1154 2928/3/1067 "
        "2942/3/999 2946/3/1099 2956/3/1129 2987/3/1150 3007/3/1080 3027/4/831 3031/3/1295 3047/3/1100 "
        "3053/4/850 3068/3/1145 3083/3/1115 3174/4/828 3178/3/1139 3185/4/855 3234/3/1185 3281/4/925 "
        "3283/4/1021 3290/4/876 3323/4/1034 3325/3/1298 3336/3/1237 3342/4/1116 3392/4/1032 3401/4/908 "
        "3423/4/991 3448/4/1040 3470/4/1129 3480/4/1065 3488/4/938 3503/4/1074 3536/4/1038 3538/4/1070 "
        "3549/4/969 3551/4/1081 3555/4/1023 3555/4/1200 3557/4/988 3568/4/1003 3658/4/1301 3672/4/1042 "
        "3690/4/1170 3694/4/986 3779/4/1236 3815/4/1070 3822/4/1136 3849/4/1066 3895/5/929 3931/5/891 "
        "4001/4/1198 4060/5/848 4077/5/910 4093/4/1150 4137/5/993 4224/5/957 4338/5/1100 4358/4/1217 "
        "4419/5/1044 4425/5/1087 4427/5/962 4509/5/1141 4663/5/1226 5222/6/964 8044/10/1320 8121/10/1087 "
        "8159/9/1165 8192/1/8192 "
    ),
    32: (
        "1/1/1 2/1/2 128/1/128 256/2/128 384/3/128 512/4/128 549/1/549 586/1/586 590/1/590 597/1/597 "
        "599/1/599 609/1/609 612/1/612 619/1/619 623/1/623 626/1/626 631/1/631 632/1/632 635/1/635 "
        "637/1/637 640/5/128 656/1/656 661/1/661 662/1/662 663/1/663 667/1/667 668/1/668 670/1/670 "
        "676/1/676 680/1/680 683/1/683 684/1/684 685/1/685 688/1/688 691/1/691 696/1/696 697/1/697 "
        "698/1/698 699/1/699 700/1/700 701/1/701 702/1/702 704/1/704 707/1/707 711/1/711 712/1/712 "
        "713/1/713 714/1/714 719/1/719 720/1/720 721/1/721 722/1/722 723/1/723 724/1/724 725/1/725 "
        "726/1/726 727/1/727 728/1/728 729/1/729 730/1/730 731/1/731 733/1/733 735/1/735 736/1/736 "
        "737/1/737 739/1/739 740/1/740 741/1/741 742/1/742 743/1/743 745/1/745 746/1/746 747/1/747 "
        "748/1/748 749/1/749 751/1/751 752/1/752 753/1/753 757/1/757 758/1/758 759/1/759 761/1/761 "
        "762/1/762 763/1/763 764/1/764 768/1/768 768/6/128 769/1/769 770/1/770 771/1/771 773/1/773 "
        "774/1/774 775/1/775 776/1/776 777/1/777 778/1/778 780/1/780 781/1/781 782/1/782 784/1/784 "
        "785/1/785 787/1/787 790/1/790 792/1/792 794/1/794 795/1/795 796/1/796 797/1/797 798/1/798 "
        "799/1/799 801/1/801 802/1/802 804/1/804 805/1/805 806/1/806 807/1/807 811/1/811 812/1/812 "
        "815/1/815 817/1/817 818/1/818 819/1/819 820/1/820 822/1/822 823/1/823 824/1/824 825/1/825 "
        "826/1/826 827/1/827 828/1/828 829/1/829 830/1/830 831/1/831 832/1/832 833/1/833 834/1/834 "
        "835/1/835 836/1/836 837/1/837 838/1/838 839/1/839 842/1/842 844/1/844 845/1/845 846/1/846 "
        "847/1/847 848/1/848 849/1/849 850/1/850 851/1/851 852/1/852 853/1/853 854/1/854 855/1/855 "
        "856/1/856 857/1/857 858/1/858 859/1/859 860/1/860 861/1/861 862/1/862 864/1/864 868/1/868 "
        "869/1/869 870/1/870 871/1/871 872/1/872 873/1/873 874/1/874 875/1/875 876/1/876 877/1/877 "
        "878/1/878 879/1/879 880/1/880 881/1/881 882/1/882 883/1/883 884/1/884 887/1/887 888/1/888 "
        "889/1/889 890/1/890 891/1/891 892/1/892 893/1/893 894/1/894 895/1/895 896/1/896 898/1/898 "
        "900/1/900 901/1/901 903/1/903 906/1/906 907/1/907 908/1/908 909/1/909 910/1/910 911/1/911 "
        "912/1/912 913/1/913 914/1/914 915/1/915 916/1/916 918/1/918 919/1/919 920/1/920 921/1/921 "
        "922/1/922 923/1/923 924/1/924 926/1/926 927/1/927 928/1/928 929/1/929 930/1/930 931/1/931 "
        "933/1/933 934/1/934 935/1/935 936/1/936 938/1/938 939/1/939 940/1/940 941/1/941 942/1/942 "
        "944/1/944 945/1/945 946/1/946 947/1/947 949/1/949 950/1/950 951/1/951 952/1/952 953/1/953 "
        "955/1/955 957/1/957 958/1/958 959/1/959 963/1/963 965/1/965 968/1/968 969/1/969 970/1/970 "
        "973/1/973 977/1/977 979/1/979 980/1/980 981/1/981 982/1/982 983/1/983 984/1/984 987/1/987 "
        "988/1/988 990/1/990 991/1/991 992/1/992 994/1/994 997/1/997 998/1/998 999/1/999 1000/1/1000 "
        "1003/1/1003 1004/1/1004 1005/1/1005 1008/1/1008 1011/1/1011 1015/1/1015 1016/1/1016 1018/1/1018 "
        "1019/1/1019 1020/1/1020 1021/1/1021 1024/1/1024 1025/1/1025 1031/1/1031 1034/1/1034 1036/1/1036 "
        "1038/1/1038 1039/1/1039 1040/1/1040 1043/1/1043 1045/1/1045 1048/1/1048 1050/1/1050 1055/1/1055 "
        "1057/1/1057 1058/1/1058 1064/1/1064 1065/1/1065 1067/1/1067 1068/1/1068 1070/1/1070 1074/1/1074 "
        "1076/1/1076 1079/1/1079 1080/1/1080 1081/1/1081 1082/1/1082 1084/1/1084 1085/1/1085 1089/1/1089 "
        "1091/1/1091 1093/1/1093 1095/1/1095 1097/1/1097 1099/1/1099 1100/1/1100 1102/1/1102 1106/1/1106 "
        "1109/1/1109 1112/1/1112 1120/1/1120 1122/1/1122 1129/1/1129 1130/1/1130 1139/1/1139 1140/1/1140 "
        "1141/1/1141 1143/1/1143 1150/1/1150 1152/9/128 1154/1/1154 1170/1/1170 1171/1/1171 1172/1/1172 "
        "1177/1/1177 1182/1/1182 1187/1/1187 1198/1/1198 1200/1/1200 1204/1/1204 1211/1/1211 1217/1/1217 "
        "1236/1/1236 1237/1/1237 1280/10/128 1295/1/1295 1374/2/706 1374/2/721 1374/2/768 1388/2/757 "
        "1421/2/765 1423/2/734 1447/2/751 1448/2/756 1460/2/780 1464/2/763 1472/2/742 1477/2/763 "
        "1481/2/900 1483/2/827 1485/2/835 1487/2/798 1491/2/817 1493/2/850 1506/2/944 1508/2/817 "
        "1513/2/788 1513/2/861 1515/2/836 1517/2/841 1524/2/825 1526/2/807 1526/2/826 1531/2/799 "
        "1535/2/848 1542/2/778 1543/2/854 1543/2/914 1544/2/893 1546/2/863 1549/2/793 1560/2/938 "
        "1562/2/841 1564/2/898 1567/2/788 1569/2/888 1570/2/786 1571/2/831 1572/2/874 1577/2/809 "
        "1579/2/796 1581/2/950 1582/2/824 1587/2/797 1588/2/910 1590/2/889 1594/2/875 1594/2/967 "
        "1595/2/848 1596/2/805 1596/2/863 1597/2/831 1599/2/859 1601/2/848 1603/2/803 1603/2/877 "
        "1606/2/836 1606/2/914 1614/2/858 1615/2/837 1620/2/840 1620/2/926 1622/2/820 1627/2/834 "
        "1627/2/931 1630/2/875 1631/2/867 1633/2/827 1634/2/826 1636/2/896 1640/2/855 1646/2/944 "
        "1647/2/859 1648/2/913 1651/2/917 1651/2/1007 1656/2/993 1661/2/942 1663/2/866 1664/2/897 "
        "1666/2/943 1667/2/914 1668/2/1036 1670/2/836 1670/2/900 1671/2/855 1671/2/965 1672/2/841 "
        "1680/2/889 1680/2/953 1685/2/911 1688/2/848 1689/2/965 1690/2/986 1696/2/970 1697/2/896 "
        "1697/2/921 1698/2/886 1699/2/929 1704/2/879 1704/2/895 1705/2/856 1710/2/1008 1716/2/912 "
        "1717/2/898 1719/2/872 1725/2/925 1727/2/876 1727/2/965 1731/2/903 1734/2/924 1734/2/929 "
        "1735/2/992 1736/2/920 1738/2/954 1742/2/893 1742/2/961 1743/2/964 1745/2/921 1745/2/937 "
        "1749/2/901 1752/2/1087 1757/2/986 1758/2/1098 1760/2/996 1765/2/926 1768/2/900 1772/2/947 "
        "1772/2/957 1773/2/993 1774/2/894 1775/2/1119 1776/2/893 1778/2/920 1781/2/976 1782/2/1040 "
        "1782/2/1081 1789/2/994 1790/2/943 1791/2/1031 1791/2/1048 1792/2/922 1792/2/939 1792/2/1021 "
        "1794/2/1076 1795/2/1022 1799/2/965 1801/2/912 1807/2/1032 1813/2/982 1814/2/926 1816/2/950 "
        "1817/2/912 1817/2/999 1819/2/912 1822/2/942 1824/2/963 1827/2/967 1828/2/962 1829/2/940 "
        "1830/2/916 1845/2/988 1847/2/955 1855/2/939 1856/2/977 1858/2/991 1863/2/936 1865/2/972 "
        "1865/2/983 1869/2/1003 1871/2/1026 1874/2/1143 1875/2/958 1876/2/1201 1881/2/1126 1882/2/1008 "
        "1884/2/1104 1889/2/1101 1890/2/1038 1902/2/1067 1910/2/1011 1911/2/1066 1920/2/1081 1921/2/1003 "
        "1922/2/1044 1926/2/1099 1928/2/974 1934/2/1046 1938/2/1120 1950/2/984 1951/2/1122 1952/2/1008 "
        "1959/2/1034 1966/2/1000 1979/2/1053 1983/2/1149 1984/2/1094 1988/2/1011 1993/2/1014 1997/2/1032 "
        "1998/2/999 2004/2/1049 2007/2/1176 2009/2/1170 2013/2/1043 2015/2/1100 2023/2/1200 2037/2/1226 "
        "2039/2/1139 2042/2/1080 2051/2/1057 2060/3/729 2062/2/1146 2067/2/1143 2102/2/1141 2155/2/1100 "
        "2169/2/1261 2176/2/1090 2179/2/1136 2179/2/1273 2221/3/785 2236/3/846 2260/2/1145 2270/3/854 "
        "2284/3/876 2289/3/858 2321/3/810 2324/3/994 2333/3/906 2339/3/801 2369/3/877 2392/3/824 "
        "2392/3/919 2394/3/904 2410/3/908 2415/3/848 2449/3/883 2452/3/844 2479/3/887 2482/3/998 "
        "2491/3/992 2499/3/908 2499/3/918 2502/3/882 2505/3/958 2517/3/1019 2527/3/877 2530/3/930 "
        "2550/3/857 2556/3/966 2560/3/887 2584/3/974 2584/3/1028 2595/3/891 2621/3/1007 2629/3/933 "
        "2629/3/964 2632/3/1031 2634/3/1000 2644/3/969 2650/3/968 2664/3/1137 2673/3/1023 2677/3/924 "
        "2677/3/980 2687/3/1122 2692/3/994 2698/3/980 2710/3/1067 2757/3/932 2769/3/1013 2780/3/987 "
        "2780/3/1118 2803/3/1002 2816/3/1062 2893/3/1092 2912/3/1111 2932/4/975 2943/3/1125 2968/3/1115 "
        "3008/3/1099 3008/3/1157 3040/3/1091 3103/4/1018 3110/4/801 3158/3/1198 3209/3/1185 3210/3/1143 "
        "3241/4/862 3254/4/1019 3326/4/910 3334/4/881 3334/4/885 3369/4/910 3374/4/1046 3407/3/1298 "
        "3496/4/1094 3515/4/996 3538/4/963 3646/4/1178 3652/4/1050 3659/4/989 3666/4/953 4016/5/954 "
        "4117/4/1128 4136/5/906 4210/4/1301 5615/6/1501 8040/10/993 8044/10/1320 8069/11/1141 8119/10/1116 "
        "8121/10/1087 8159/9/1165 8192/1/8192 "
    ),
}

# Kernel map of the context call on sm_103 (152 SMs; the pinned FlashInfer's trtllm-gen cubins), from a
# profiler probe of 94,776 calls: every max_q_len 1..8192 at one request, every (max_q_len, B) with
# max_q_len * B <= 2048 for B 2..32 (at two totals T each), and 3,000 random partitions per (H,
# width). The kernel(s) of a call are a function of (H, width, B, max_q_len) alone, never of T.
# VARLEN_BANDS[(H, width)][B] = ((lowest max_q_len, highest max_q_len, VARLEN_KERNELS index), ...);
# the last band of every B runs to the envelope's top, 8192 - (B - 1). Names drop the common prefix
# fmhaSm100fKernel_QkvBfloat16OBfloat16H512.
VARLEN_KERNELS = (
    (
        "HVPerCta128PagedKvDenseDynamicTokenSparseP1MultiCtasKvGmemSepVarSeqQ64Kv128StaticKeepsAbForGen",
        "fmhaReductionKernel<64, 128, false, __nv_bfloat16, __nv_bfloat16>",
    ),
    ("HVPerCta128PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",),
    ("HVPerCta128PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",),
    ("HVPerCta128PagedKvDenseDynamicTokenSparseP1VarSeqQ16Kv128PersistentSwapsAbForGen",),
    ("HVPerCta128PagedKvDenseDynamicTokenSparseP1VarSeqQ64Kv128PersistentKeepsAbForGen",),
    ("HVPerCta128PagedKvDenseDynamicTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen",),
    (
        "HVPerCta256PagedKvDenseDynamicTokenSparseP1MultiCtasKvGmemSepVarSeqQ64Kv128StaticKeepsAbForGen",
        "fmhaReductionKernel<64, 256, false, __nv_bfloat16, __nv_bfloat16>",
    ),
    ("HVPerCta256PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",),
    ("HVPerCta256PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",),
    ("HVPerCta256PagedKvDenseDynamicTokenSparseP1VarSeqQ16Kv128PersistentSwapsAbForGen",),
    ("HVPerCta256PagedKvDenseDynamicTokenSparseP1VarSeqQ64Kv128PersistentKeepsAbForGen",),
    ("HVPerCta256PagedKvDenseDynamicTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen",),
    (
        "PagedKvDenseDynamicTokenSparseP1MultiCtasKvGmemSepVarSeqQ64Kv128StaticKeepsAbForGen",
        "fmhaReductionKernel<64, 512, false, __nv_bfloat16, __nv_bfloat16>",
    ),
    ("PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",),
    ("PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",),
    ("PagedKvDenseDynamicTokenSparseP1VarSeqQ16Kv128PersistentSwapsAbForGen",),
    ("PagedKvDenseDynamicTokenSparseP1VarSeqQ64Kv128PersistentKeepsAbForGen",),
    ("PagedKvDenseDynamicTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen",),
)
VARLEN_BANDS = {
    (16, 128): {
        1: ((1, 19, 5), (20, 38, 11), (39, 8192, 17)),
        2: ((1, 9, 5), (10, 19, 11), (20, 8191, 17)),
        3: ((1, 6, 5), (7, 12, 11), (13, 8190, 17)),
        4: ((1, 4, 5), (5, 9, 11), (10, 8189, 17)),
        5: ((1, 3, 5), (4, 7, 11), (8, 8188, 17)),
        6: ((1, 3, 5), (4, 6, 11), (7, 8187, 17)),
        7: ((1, 2, 5), (3, 5, 11), (6, 8186, 17)),
        8: ((1, 2, 5), (3, 4, 11), (5, 8185, 17)),
        9: ((1, 2, 5), (3, 4, 11), (5, 8184, 17)),
        10: ((1, 1, 5), (2, 3, 11), (4, 8183, 17)),
        11: ((1, 1, 5), (2, 3, 11), (4, 8182, 17)),
        12: ((1, 1, 5), (2, 3, 11), (4, 8181, 17)),
        13: ((1, 1, 5), (2, 2, 11), (3, 8180, 17)),
        14: ((1, 1, 5), (2, 2, 11), (3, 8179, 17)),
        15: ((1, 1, 5), (2, 2, 11), (3, 8178, 17)),
        16: ((1, 1, 5), (2, 2, 11), (3, 8177, 17)),
        17: ((1, 1, 5), (2, 2, 11), (3, 8176, 17)),
        18: ((1, 1, 5), (2, 2, 11), (3, 8175, 17)),
        19: ((1, 1, 5), (2, 2, 11), (3, 8174, 17)),
        20: ((1, 1, 3), (2, 3, 9), (4, 8173, 15)),
        21: ((1, 1, 3), (2, 3, 9), (4, 8172, 15)),
        22: ((1, 1, 3), (2, 3, 9), (4, 8171, 15)),
        23: ((1, 1, 3), (2, 3, 9), (4, 8170, 15)),
        24: ((1, 1, 3), (2, 3, 9), (4, 8169, 15)),
        25: ((1, 1, 3), (2, 3, 9), (4, 8168, 15)),
        26: ((1, 1, 3), (2, 2, 9), (3, 8167, 15)),
        27: ((1, 1, 3), (2, 2, 9), (3, 8166, 15)),
        28: ((1, 1, 3), (2, 2, 9), (3, 8165, 15)),
        29: ((1, 1, 3), (2, 2, 9), (3, 8164, 15)),
        30: ((1, 1, 3), (2, 2, 9), (3, 8163, 15)),
        31: ((1, 1, 3), (2, 2, 9), (3, 8162, 15)),
        32: ((1, 1, 3), (2, 2, 9), (3, 8161, 15)),
    },
    (16, 640): {
        1: ((1, 6, 2), (7, 12, 8), (13, 38, 14), (39, 8192, 17)),
        2: ((1, 3, 2), (4, 6, 8), (7, 19, 14), (20, 8191, 17)),
        3: ((1, 2, 2), (3, 4, 8), (5, 12, 14), (13, 8190, 17)),
        4: ((1, 3, 1), (4, 6, 7), (7, 19, 13), (20, 8189, 15)),
        5: ((1, 2, 1), (3, 5, 7), (6, 15, 13), (16, 8188, 15)),
        6: ((1, 2, 1), (3, 4, 7), (5, 12, 13), (13, 8187, 15)),
        7: ((1, 1, 1), (2, 3, 7), (4, 10, 13), (11, 8186, 15)),
        8: ((1, 1, 1), (2, 3, 7), (4, 9, 13), (10, 8185, 15)),
        9: ((1, 1, 1), (2, 2, 7), (3, 8, 13), (9, 8184, 15)),
        10: ((1, 1, 1), (2, 2, 7), (3, 7, 13), (8, 8183, 15)),
        11: ((1, 1, 1), (2, 2, 7), (3, 6, 13), (7, 8182, 15)),
        12: ((1, 1, 1), (2, 2, 7), (3, 6, 13), (7, 8181, 15)),
        13: ((1, 1, 7), (2, 5, 13), (6, 8180, 15)),
        14: ((1, 1, 7), (2, 5, 13), (6, 8179, 15)),
        15: ((1, 1, 7), (2, 5, 13), (6, 8178, 15)),
        16: ((1, 1, 7), (2, 4, 13), (5, 8177, 15)),
        17: ((1, 1, 7), (2, 4, 13), (5, 8176, 15)),
        18: ((1, 1, 7), (2, 4, 13), (5, 8175, 15)),
        19: ((1, 1, 7), (2, 4, 13), (5, 8174, 15)),
        20: ((1, 1, 7), (2, 3, 13), (4, 8173, 15)),
        21: ((1, 1, 7), (2, 3, 13), (4, 8172, 15)),
        22: ((1, 1, 7), (2, 3, 13), (4, 8171, 15)),
        23: ((1, 1, 7), (2, 3, 13), (4, 8170, 15)),
        24: ((1, 1, 7), (2, 3, 13), (4, 8169, 15)),
        25: ((1, 1, 7), (2, 3, 13), (4, 8168, 15)),
        26: ((1, 2, 13), (3, 8167, 15)),
        27: ((1, 2, 13), (3, 8166, 15)),
        28: ((1, 2, 13), (3, 8165, 15)),
        29: ((1, 2, 13), (3, 8164, 15)),
        30: ((1, 2, 13), (3, 8163, 15)),
        31: ((1, 2, 13), (3, 8162, 15)),
        32: ((1, 2, 13), (3, 8161, 15)),
    },
    (64, 128): {
        1: ((1, 38, 4), (39, 76, 10), (77, 8192, 16)),
        2: ((1, 19, 4), (20, 38, 10), (39, 8191, 16)),
        3: ((1, 12, 4), (13, 25, 10), (26, 8190, 16)),
        4: ((1, 9, 4), (10, 19, 10), (20, 8189, 16)),
        5: ((1, 7, 4), (8, 15, 10), (16, 8188, 16)),
        6: ((1, 6, 4), (7, 12, 10), (13, 8187, 16)),
        7: ((1, 5, 4), (6, 10, 10), (11, 8186, 16)),
        8: ((1, 4, 4), (5, 9, 10), (10, 8185, 16)),
        9: ((1, 4, 4), (5, 8, 10), (9, 8184, 16)),
        10: ((1, 3, 4), (4, 7, 10), (8, 8183, 16)),
        11: ((1, 3, 4), (4, 6, 10), (7, 8182, 16)),
        12: ((1, 3, 4), (4, 6, 10), (7, 8181, 16)),
        13: ((1, 2, 4), (3, 5, 10), (6, 8180, 16)),
        14: ((1, 2, 4), (3, 5, 10), (6, 8179, 16)),
        15: ((1, 2, 4), (3, 5, 10), (6, 8178, 16)),
        16: ((1, 2, 4), (3, 4, 10), (5, 8177, 16)),
        17: ((1, 2, 4), (3, 4, 10), (5, 8176, 16)),
        18: ((1, 2, 4), (3, 4, 10), (5, 8175, 16)),
        19: ((1, 2, 4), (3, 4, 10), (5, 8174, 16)),
        20: ((1, 1, 4), (2, 3, 10), (4, 8173, 16)),
        21: ((1, 1, 4), (2, 3, 10), (4, 8172, 16)),
        22: ((1, 1, 4), (2, 3, 10), (4, 8171, 16)),
        23: ((1, 1, 4), (2, 3, 10), (4, 8170, 16)),
        24: ((1, 1, 4), (2, 3, 10), (4, 8169, 16)),
        25: ((1, 1, 4), (2, 3, 10), (4, 8168, 16)),
        26: ((1, 1, 4), (2, 2, 10), (3, 8167, 16)),
        27: ((1, 1, 4), (2, 2, 10), (3, 8166, 16)),
        28: ((1, 1, 4), (2, 2, 10), (3, 8165, 16)),
        29: ((1, 1, 4), (2, 2, 10), (3, 8164, 16)),
        30: ((1, 1, 4), (2, 2, 10), (3, 8163, 16)),
        31: ((1, 1, 4), (2, 2, 10), (3, 8162, 16)),
        32: ((1, 1, 4), (2, 2, 10), (3, 8161, 16)),
    },
    (64, 640): {
        1: ((1, 12, 0), (13, 25, 6), (26, 76, 12), (77, 8192, 16)),
        2: ((1, 6, 0), (7, 12, 6), (13, 38, 12), (39, 8191, 16)),
        3: ((1, 4, 0), (5, 8, 6), (9, 25, 12), (26, 8190, 16)),
        4: ((1, 3, 0), (4, 6, 6), (7, 19, 12), (20, 8189, 16)),
        5: ((1, 2, 0), (3, 5, 6), (6, 15, 12), (16, 8188, 16)),
        6: ((1, 2, 0), (3, 4, 6), (5, 12, 12), (13, 8187, 16)),
        7: ((1, 1, 0), (2, 3, 6), (4, 10, 12), (11, 8186, 16)),
        8: ((1, 1, 0), (2, 3, 6), (4, 9, 12), (10, 8185, 16)),
        9: ((1, 1, 0), (2, 2, 6), (3, 8, 12), (9, 8184, 16)),
        10: ((1, 1, 0), (2, 2, 6), (3, 7, 12), (8, 8183, 16)),
        11: ((1, 1, 0), (2, 2, 6), (3, 6, 12), (7, 8182, 16)),
        12: ((1, 1, 0), (2, 2, 6), (3, 6, 12), (7, 8181, 16)),
        13: ((1, 1, 6), (2, 5, 12), (6, 8180, 16)),
        14: ((1, 1, 6), (2, 5, 12), (6, 8179, 16)),
        15: ((1, 1, 6), (2, 5, 12), (6, 8178, 16)),
        16: ((1, 1, 6), (2, 4, 12), (5, 8177, 16)),
        17: ((1, 1, 6), (2, 4, 12), (5, 8176, 16)),
        18: ((1, 1, 6), (2, 4, 12), (5, 8175, 16)),
        19: ((1, 1, 6), (2, 4, 12), (5, 8174, 16)),
        20: ((1, 1, 6), (2, 3, 12), (4, 8173, 16)),
        21: ((1, 1, 6), (2, 3, 12), (4, 8172, 16)),
        22: ((1, 1, 6), (2, 3, 12), (4, 8171, 16)),
        23: ((1, 1, 6), (2, 3, 12), (4, 8170, 16)),
        24: ((1, 1, 6), (2, 3, 12), (4, 8169, 16)),
        25: ((1, 1, 6), (2, 3, 12), (4, 8168, 16)),
        26: ((1, 2, 12), (3, 8167, 16)),
        27: ((1, 2, 12), (3, 8166, 16)),
        28: ((1, 2, 12), (3, 8165, 16)),
        29: ((1, 2, 12), (3, 8164, 16)),
        30: ((1, 2, 12), (3, 8163, 16)),
        31: ((1, 2, 12), (3, 8162, 16)),
        32: ((1, 2, 12), (3, 8161, 16)),
    },
}


def _variant_of(names) -> tuple:
    """The trtllm-gen kernels of one call, as recorded in VARLEN_KERNELS."""
    out = []
    for n in names:
        if n.startswith("fmhaSm100fKernel_QkvBfloat16OBfloat16H512"):
            out.append(n[len("fmhaSm100fKernel_QkvBfloat16OBfloat16H512") :])
        elif "fmhaReductionKernel<" in n:
            out.append(re.search(r"fmhaReductionKernel<[^>]*>", n).group(0))
    return tuple(out)


@pytest.fixture(scope="module")
def env_pools():
    """Random pools large enough for 32 requests of 16384 positions without any aliasing of a
    request's compressed rows; every request reaches its rows through a random page table."""
    gen = torch.Generator(device=DEV).manual_seed(4242)
    swa = torch.randn(ENV_SWA_PAGES, 1, SWA_PAGE, HEAD_DIM, generator=gen, device=DEV).to(
        torch.bfloat16
    )
    pools = {"swa": swa, "comp": {}, "atol": {}}
    for layer, (ratio, page) in ENV_LAYERS.items():
        if ratio:
            comp = torch.randn(
                ENV_COMP_PAGES[ratio], 1, page, HEAD_DIM, generator=gen, device=DEV
            ).to(torch.bfloat16)
            pools["atol"][layer] = _atol(swa, comp)
        else:
            # The core target's ratio-0 placeholder shape; NaN, so that any read of it would show.
            comp = torch.full((1, 1, 64, HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=DEV)
            pools["atol"][layer] = _atol(swa)
        pools["comp"][layer] = comp
    yield pools
    del pools
    torch.cuda.empty_cache()


def _env_tables(requests, gen, ratio, page, swa_pages, comp_pages, cap_pages=None):
    """Random page tables: a 2-page SWA ring per request, and enough compressed pages per request
    for every compressed position it can see (distinct rows). With cap_pages, a request that can see
    more compressed positions than cap_pages pages hold gets cap_pages pages used as a ring
    (position e on page (e // page) % cap_pages, so far positions alias nearer rows)."""
    b = len(requests)
    swa_bt = torch.randperm(swa_pages, generator=gen, device=DEV)[: 2 * b].view(b, 2)
    comp_bts = None
    if ratio:
        need = [max(1, -(-((c + q) // ratio) // page)) for c, q in requests]
        if cap_pages is not None:
            need = [min(n, cap_pages) for n in need]
        perm = torch.randperm(comp_pages, generator=gen, device=DEV)
        assert sum(need) <= comp_pages
        starts = [0]
        for n in need:
            starts.append(starts[-1] + n)
        comp_bts = [perm[starts[i] : starts[i + 1]] for i in range(b)]
    return swa_bt, comp_bts


def _env_rows(requests, gen, width, ratio, comp_page, swa_bt, comp_bts, swa_page=SWA_PAGE):
    """Vectorized _build_rows (same layout rules): query j of request (cached, q_len) sits at
    pos = cached + j, sees window positions max(0, pos-127)..pos in order (-1 after them), and,
    for width 640, min(512, (pos+1)//ratio) compressed positions (all of them while <= 512, else a
    random sorted subset), compacted at the front. Returns idx [N, width] int32, active [N] int32,
    seq_lens [B] int32, cu [B+1] int32 and the window counts [N] (min(128, pos + 1)).
    A request's compressed position e lives on page (e // comp_page) % (its page count): distinct
    rows while its pages hold every position it can see, a ring over them otherwise (_env_tables'
    cap_pages). Past 65536 visible positions the subset is drawn one position per stratum
    [s * v // 512, (s + 1) * v // 512) (distinct and sorted), not by a [rows, v] score matrix."""
    ar = torch.arange(WINDOW, device=DEV)
    idx_l, act_l, cnt_l = [], [], []
    for b, (cached, q_len) in enumerate(requests):
        pos = torch.arange(cached, cached + q_len, device=DEV)
        wpos = (pos - (WINDOW - 1)).clamp_min(0)[:, None] + ar[None]
        ring = swa_bt[b]
        widx = ring[(wpos // swa_page) % ring.numel()] * swa_page + wpos % swa_page
        widx = torch.where(wpos <= pos[:, None], widx, -1).int()
        cnt_l.append((pos + 1).clamp(max=WINDOW))
        if width == WINDOW:
            idx_l.append(widx)
            act_l.append(torch.full((q_len,), WINDOW, dtype=torch.int32, device=DEV))
            continue
        k_max = width - WINDOW
        vis = (pos + 1) // ratio
        vmax = (cached + q_len) // ratio
        pages = comp_bts[b]
        sels, valids = [], []
        for s0 in range(0, q_len, 1024):
            v = vis[s0 : s0 + 1024]
            if vmax <= k_max:
                sel = torch.arange(k_max, device=DEV)[None].expand(v.numel(), k_max)
                valid = sel < v[:, None]
            elif vmax > 65536:
                strata = torch.arange(k_max + 1, device=DEV)
                lo = (strata[None, :-1] * v[:, None]) // k_max
                hi = (strata[None, 1:] * v[:, None]) // k_max
                u = torch.rand(v.numel(), k_max, generator=gen, device=DEV)
                strat = lo + (u * (hi - lo)).long()
                sel = torch.where(v[:, None] > k_max, strat, torch.arange(k_max, device=DEV)[None])
                valid = sel < v[:, None]
            else:
                score = torch.rand(v.numel(), vmax, generator=gen, device=DEV)
                score.masked_fill_(torch.arange(vmax, device=DEV)[None] >= v[:, None], -1.0)
                val, sel = score.topk(k_max, dim=1)
                sel = torch.where(val >= 0, sel, vmax).sort(dim=1).values
                valid = sel < vmax
            sels.append(sel)
            valids.append(valid)
        sel, valid = torch.cat(sels), torch.cat(valids)
        sel = torch.where(valid, sel, 0)
        cpage = pages[(sel // comp_page) % pages.numel()]
        cidx = torch.where(valid, cpage * comp_page + sel % comp_page, -1).int()
        idx_l.append(torch.cat([widx, cidx], 1))
        act_l.append((WINDOW + valid.sum(1)).int())
    idx = torch.cat(idx_l).contiguous()
    active = torch.cat(act_l)
    seq_lens = torch.tensor([c + q for c, q in requests], dtype=torch.int32, device=DEV)
    cu = torch.tensor([0] + [q for _, q in requests], dtype=torch.int32, device=DEV).cumsum(0).int()
    return idx, active, seq_lens, cu, torch.cat(cnt_l)


class _Step:
    """One engine step: nc context requests (their rows first), then G generation requests of
    next_n rows each (next_n 1: decode; 6: the DSpark target verify).

    The per-row and per-request tensors are prefixes of buffers sized for the envelope, as the
    runtime's metadata buffers are; the two calls get the target's views of them."""

    def __init__(
        self,
        ctx,
        dec_positions,
        num_heads,
        layer,
        pools,
        gen,
        swa_bt=None,
        comp_bts=None,
        next_n=1,
        cap_pages=None,
    ):
        ratio, page = ENV_LAYERS[layer]
        self.layer = layer
        self.width = WINDOW + (INDEX_TOPK if ratio else 0)
        self.next_n = next_n
        self.requests = list(ctx) + [(p, next_n) for p in dec_positions]
        self.nc, self.tc = len(ctx), sum(q for _, q in ctx)
        self.B, self.G = len(self.requests), len(dec_positions)
        self.max_ctx_q = max((q for _, q in ctx), default=0)
        self.swa, self.comp = pools["swa"], pools["comp"][layer]
        if swa_bt is None:
            swa_bt, comp_bts = _env_tables(
                self.requests, gen, ratio, page, self.swa.shape[0], self.comp.shape[0], cap_pages
            )
        idx, active, seq_lens, cu, self.counts = _env_rows(
            self.requests, gen, self.width, ratio, page, swa_bt, comp_bts
        )
        self.T = idx.shape[0]
        self.idx = idx
        self.active = torch.full((max(self.T, ENV_T),), -7, dtype=torch.int32, device=DEV)[: self.T]
        self.active.copy_(active)
        self.kv_lens = torch.full((max(self.B, ENV_B) + 1,), -7, dtype=torch.int32, device=DEV)
        self.kv_lens[: self.B] = seq_lens
        self.cu = torch.full((max(self.B, ENV_B) + 2,), -7, dtype=torch.int32, device=DEV)
        self.cu[: self.B + 1] = cu
        self.q3 = (
            torch.randn(self.T, num_heads, HEAD_DIM, generator=gen, device=DEV)
            .mul_(2)
            .to(torch.bfloat16)
        )
        self.sinks = _sinks(num_heads)

    def run(self, ws, out=None, swa=None, comp=None, idx=None):
        """The target's two calls; swa / comp / idx replace the container pools and index (a native
        read form of the same step)."""
        tc, gen_rows, h, nn = self.tc, self.T - self.tc, self.q3.shape[1], self.next_n
        swa = self.swa if swa is None else swa
        comp = self.comp if comp is None else comp
        idx = self.idx if idx is None else idx
        o = (
            torch.empty(self.T, h, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
            if out is None
            else out
        )
        if tc:
            op(
                torch.split(self.q3, [tc, gen_rows], 0)[0],
                swa,
                comp,
                ws,
                torch.split(idx, [tc, gen_rows], 0)[0],
                torch.split(self.active, [tc, gen_rows], 0)[0],
                self.kv_lens[: self.nc],
                self.sinks,
                SCALE,
                1.0,
                cum_seq_lens_q=self.cu[: self.nc + 1],
                max_q_len=self.max_ctx_q,
                out=torch.split(o, [tc, gen_rows], 0)[0],
            )
        if gen_rows:
            g = self.G
            op(
                torch.split(self.q3, [tc, gen_rows], 0)[1].reshape(g, nn, h, HEAD_DIM),
                swa,
                comp,
                ws,
                torch.split(idx, [tc, gen_rows], 0)[1],
                torch.split(self.active, [tc, gen_rows], 0)[1],
                self.kv_lens[self.nc : self.B],
                self.sinks,
                SCALE,
                1.0,
                out=torch.split(o, [tc, gen_rows], 0)[1].reshape(g, nn, h, HEAD_DIM),
            )
        return o

    def reference(self, idx=None):
        return _reference(
            self.q3,
            self.swa,
            self.comp,
            self.idx if idx is None else idx,
            self.active,
            self.counts,
            self.sinks,
        )


def _excess(o, ref, atol) -> float:
    """max |o - ref| / (atol + RTOL * |ref|): <= 1 is inside the file's gate."""
    return ((o.float().reshape(ref.shape) - ref).abs_() / (atol + RTOL * ref.abs())).amax().item()


def _fresh_or_cached(rng, q_len):
    return 0 if rng.random() < 0.4 else rng.randint(0, ENV_SEQ - q_len)


@pytest.mark.parametrize("cached", [0, 8192])
@pytest.mark.parametrize("layer", ["r0", "r2"])
@pytest.mark.parametrize("num_heads", [16, 64])
def test_prefill_every_max_q_len_one_request(env_pools, num_heads, layer, cached):
    """One context request of every q_len 1..8192 (max_q_len = T = q_len) against the fp32
    reference. Query row j's result does not depend on q_len (its window count is
    min(128, cached + j + 1) and its index row is its own), so each call is held to the first
    q_len rows of ONE reference computed for the 8192-row request."""
    gen = torch.Generator(device=DEV).manual_seed(31000 + num_heads + cached + ENV_LAYERS[layer][0])
    st = _Step([(cached, ENV_T)], [], num_heads, layer, env_pools, gen)
    ref = st.reference()
    den = env_pools["atol"][layer] + RTOL * ref.abs()
    ws = _workspace()
    out = torch.empty_like(st.q3)
    worst = torch.zeros(ENV_T, device=DEV)
    seq = torch.zeros(1, dtype=torch.int32, device=DEV)
    cu = torch.zeros(2, dtype=torch.int32, device=DEV)
    diff = torch.empty_like(ref)  # one fp32 scratch buffer, so the sweep allocates nothing per call
    for n in range(1, ENV_T + 1):
        seq.fill_(cached + n)
        cu[1].fill_(n)
        op(
            st.q3[:n],
            st.swa,
            st.comp,
            ws,
            st.idx[:n],
            st.active[:n],
            seq,
            st.sinks,
            SCALE,
            1.0,
            cum_seq_lens_q=cu,
            max_q_len=n,
            out=out[:n],
        )
        d = diff[:n]
        d.copy_(out[:n]).sub_(ref[:n]).abs_().div_(den[:n])
        torch.amax(d, out=worst[n - 1])
    torch.cuda.synchronize()
    bad = (worst > 1.0).nonzero().flatten()
    print(f"every-q_len H{num_heads} {layer} cached {cached}: max excess {worst.max().item():.3f}")
    assert bad.numel() == 0, (
        f"q_len {(bad + 1).tolist()[:20]} exceed the gate (max {worst.max().item():.2f})"
    )
    assert torch.isfinite(out).all()


def _band_cases(num_heads, width, rng):
    """Every kernel band of VARLEN_BANDS at every request count 1..32: its lowest, one interior and
    its highest max_q_len. The other B-1 requests get random q_len in [1, max_q_len] within the
    8192-token step budget."""
    cases = []
    for b, bands in VARLEN_BANDS[(num_heads, width)].items():
        for lo, hi, vid in bands:
            for mq in sorted({lo, (lo + hi) // 2, hi}):
                budget = ENV_T - mq
                others = [rng.randint(1, mq) for _ in range(b - 1)]
                while sum(others) > budget:
                    i = max(range(b - 1), key=lambda k: others[k])
                    others[i] = max(1, others[i] - max(1, (sum(others) - budget)))
                q_lens = [mq] + others
                rng.shuffle(q_lens)
                cases.append(([(_fresh_or_cached(rng, q), q) for q in q_lens], vid))
    return cases


@pytest.mark.parametrize("width", [128, 640])
@pytest.mark.parametrize("num_heads", [16, 64])
def test_prefill_kernel_variant_bands(env_pools, num_heads, width):
    """Kernel bands of the context call. The kernel is picked on the host from (H, width, number of
    requests, max_q_len) (probe: 94,776 profiled calls, never two variants for one such cell, total
    T irrelevant); VARLEN_BANDS is that map. Every band is run at its bottom, an interior point and
    its top at every request count 1..32, with the kernels read back from the profiler, and every
    call is held to the fp32 reference."""
    rng = random.Random(num_heads * 1000 + width)
    gen = torch.Generator(device=DEV).manual_seed(num_heads * 1000 + width)
    ws = _workspace()
    cases = _band_cases(num_heads, width, rng)
    seen, worst, wrong = [], 0.0, []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for i, (ctx, vid) in enumerate(cases):
            layer = "r0" if width == WINDOW else ("r2", "r1")[i % 2]
            st = _Step(ctx, [], num_heads, layer, env_pools, gen)
            torch.cuda.synchronize()
            torch.cuda._sleep(10)
            o = st.run(ws)
            torch.cuda._sleep(10)
            e = _excess(o, st.reference(), env_pools["atol"][layer])
            worst = max(worst, e)
            if e > 1.0:
                wrong.append((ctx, e))
        torch.cuda.synchronize()
    ev = sorted(
        (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda e: e.time_range.start,
    )
    open_call, cur = False, []
    for e in ev:
        if "spin_kernel" in e.name or "sleep" in e.name:
            if open_call:
                seen.append(_variant_of(cur))
            open_call, cur = not open_call, []
        elif open_call:
            cur.append(e.name)
    assert len(seen) == len(cases), (len(seen), len(cases))
    mism = [(c, VARLEN_KERNELS[v], s) for (c, v), s in zip(cases, seen) if VARLEN_KERNELS[v] != s]
    used = {v for _, v in cases}
    print(
        f"bands H{num_heads} W{width}: {len(cases)} calls, {len(used)} variants, max excess {worst:.3f}"
    )
    assert not mism, mism[:3]
    assert not wrong, wrong[:3]


def _partition_case(seed):
    """A random step: k in 1..32 context requests over T in 1..8192 tokens (log-uniform or uniform),
    random q_len split and order, fresh or cached prefix (cached + q_len <= 16384); for half the
    seeds, G random decode rows after the context rows (k + G <= 32, T + G <= 8192)."""
    rng = random.Random(900000 + seed)
    num_heads = (16, 64)[seed % 2]
    layer = ("r0", "r2", "r1")[(seed // 2) % 3]
    k = rng.randint(1, ENV_B)
    if rng.random() < 0.5:
        t = int(round(math.exp(rng.uniform(0.0, math.log(ENV_T)))))
    else:
        t = rng.randint(1, ENV_T)
    t = max(t, k)
    cuts = sorted(rng.sample(range(1, t), k - 1)) if k > 1 else []
    q_lens = [b - a for a, b in zip([0] + cuts, cuts + [t])]
    ctx = [(_fresh_or_cached(rng, q), q) for q in q_lens]
    dec = []
    if seed % 4 >= 2 and k < ENV_B and t < ENV_T:
        g = rng.randint(1, min(ENV_B - k, ENV_T - t))
        dec = [rng.randint(0, ENV_SEQ - 1) for _ in range(g)]
    return num_heads, layer, ctx, dec


PARTITION_SEEDS = list(range(240))


@pytest.mark.parametrize("seed", PARTITION_SEEDS)
def test_prefill_random_partition(env_pools, seed):
    """240 seeded random partitions of the context envelope, half of them mixed steps whose
    decode call reads the views starting at row tc, all against the fp32 reference."""
    num_heads, layer, ctx, dec = _partition_case(seed)
    gen = torch.Generator(device=DEV).manual_seed(500000 + seed)
    st = _Step(ctx, dec, num_heads, layer, env_pools, gen)
    o = st.run(_workspace())
    e = _excess(o, st.reference(), env_pools["atol"][layer])
    print(
        f"partition {seed}: H{num_heads} {layer} T {st.T} tc {st.tc} nc {st.nc} G {st.G} "
        f"max_q_len {st.max_ctx_q} excess {e:.3f}"
    )
    assert e <= 1.0, (seed, e)


def test_prefill_random_partitions_reach_the_envelope():
    """The 240 partitions span the envelope: k 1..32, T up to 8192, cached prefixes past 8192, and
    mixed steps whose decode views start at odd rows / odd requests."""
    cases = [_partition_case(s) for s in PARTITION_SEEDS]
    ks = {len(c[2]) for c in cases}
    ts = [sum(q for _, q in c[2]) for c in cases]
    mixed = [c for c in cases if c[3]]
    odd_tc = sum(sum(q for _, q in c[2]) % 2 for c in mixed)
    odd_nc = sum(len(c[2]) % 2 for c in mixed)
    assert min(ks) == 1 and max(ks) >= 30, sorted(ks)
    assert min(ts) <= 4 and max(ts) >= 8000, (min(ts), max(ts))
    assert max(c + q for case in cases for c, q in case[2]) >= 16000
    assert len(mixed) >= 100 and odd_tc >= 20 and odd_nc >= 20, (len(mixed), odd_tc, odd_nc)
    assert {(c[0], c[1]) for c in cases} == {(h, layer) for h in (16, 64) for layer in ENV_LAYERS}


def test_prefill_partition_wrong_readings_fail_the_gate(env_pools):
    """The partition harness can see a step read wrongly. On the first six mixed partitions that
    have a fresh context request among several, the kernel's output sits inside the gate while the
    reference built on each wrong reading lands far outside it: every context query given its
    request's LAST window count (queries not causal within a request), each request's window count
    taken from its neighbour's seq_lens (requests paired with the wrong kv_lens entry), and the decode
    rows' q read from row 0 of the step instead of row tc (the row-offset view lost)."""
    picked = []
    for seed in PARTITION_SEEDS:
        num_heads, layer, ctx, dec = _partition_case(seed)
        if dec and len(ctx) >= 2 and any(c == 0 and q >= 2 for c, q in ctx):
            picked.append(seed)
        if len(picked) == 6:
            break
    assert len(picked) == 6
    for seed in picked:
        num_heads, layer, ctx, dec = _partition_case(seed)
        gen = torch.Generator(device=DEV).manual_seed(500000 + seed)
        st = _Step(ctx, dec, num_heads, layer, env_pools, gen)
        o = st.run(_workspace())
        atol = env_pools["atol"][layer]
        correct = _excess(o, st.reference(), atol)
        q_lens = [q for _, q in st.requests]
        seq = st.kv_lens[: st.B].long()
        last = torch.repeat_interleave(seq.clamp(max=WINDOW), torch.tensor(q_lens, device=DEV))
        rolled = seq.roll(1)
        pos_rolled = torch.cat(
            [torch.arange(q, device=DEV) + int(s) - q for s, q in zip(rolled.tolist(), q_lens)]
        )
        wrong = {
            "non_causal_window": _reference(
                st.q3, st.swa, st.comp, st.idx, st.active, last, st.sinks
            ),
            "neighbour_seq_lens": _reference(
                st.q3,
                st.swa,
                st.comp,
                st.idx,
                st.active,
                (pos_rolled + 1).clamp(1, WINDOW),
                st.sinks,
            ),
        }
        q_lost = st.q3.clone()
        q_lost[st.tc :] = st.q3[: st.G]
        wrong["decode_view_from_row_0"] = _reference(
            q_lost, st.swa, st.comp, st.idx, st.active, st.counts, st.sinks
        )
        excess = {k: _excess(o, r, atol) for k, r in wrong.items()}
        print(
            f"partition controls seed {seed} (nc {st.nc}, G {st.G}): correct {correct:.3f}, wrong "
            f"{ {k: round(v, 1) for k, v in excess.items()} }"
        )
        assert correct <= 1.0, correct
        for name, e in excess.items():
            assert e > 10.0 * max(correct, 0.1), (
                f"seed {seed}: {name} not discriminated ({e:.2f} vs {correct:.2f})"
            )


def _dumped_cells(num_heads):
    cells = {**DUMPED_CTX_CELLS, **MEMBER_DUMPED_CTX_CELLS}[num_heads]
    return [tuple(int(x) for x in tok.split("/")) for tok in "".join(cells).split()]


def _dumped_partition(tc, nc, mq, rng):
    """A step with this (tc, nc, max_q_len): one request of max_q_len, the other nc - 1 splitting
    the rest with every q_len in [1, max_q_len], random order and cached prefixes."""
    parts, extra = [1] * (nc - 1), tc - mq - (nc - 1)
    while extra > 0:
        i = rng.randrange(nc - 1)
        add = min(extra, mq - parts[i], rng.randint(1, extra))
        parts[i] += add
        extra -= add
    q_lens = [mq] + parts
    rng.shuffle(q_lens)
    return [(_fresh_or_cached(rng, q), q) for q in q_lens]


@pytest.mark.parametrize("width", [128, 640])
@pytest.mark.parametrize("num_heads", ALL_HEADS)
def test_prefill_every_dumped_cell(env_pools, num_heads, width):
    """Every context call (tc, H, nc, max_q_len) the boot gates and the accuracy-gate previews
    dumped (DUMPED_CTX_CELLS; H 8 / 32 from the parallel members' dumps, MEMBER_DUMPED_CTX_CELLS),
    at width 128 (ratio 0) and 640 (ratio 2 and ratio 1 on alternate cells), as a step with that
    many requests, against the fp32 reference."""
    rng = random.Random(77000 + num_heads + width)
    gen = torch.Generator(device=DEV).manual_seed(77000 + num_heads + width)
    ws = _workspace()
    cells = _dumped_cells(num_heads)
    worst, bad = 0.0, []
    for i, (tc, nc, mq) in enumerate(cells):
        layer = "r0" if width == WINDOW else ("r2", "r1")[i % 2]
        ctx = _dumped_partition(tc, nc, mq, rng)
        assert (sum(q for _, q in ctx), len(ctx), max(q for _, q in ctx)) == (tc, nc, mq)
        st = _Step(ctx, [], num_heads, layer, env_pools, gen)
        e = _excess(st.run(ws), st.reference(), env_pools["atol"][layer])
        worst = max(worst, e)
        if e > 1.0:
            bad.append(((tc, nc, mq), layer, e))
    print(f"dumped cells H{num_heads} W{width}: {len(cells)} calls, max excess {worst:.3f}")
    assert not bad, bad[:5]


# ---------------------------------------------------------------------------------------------
# Pool-size invariance. The kernel receives each pool's base pointer and row
# stride, and the compressed pool's row count (mNumPagesInMemPool, which sizes a TMA descriptor);
# these tests run the same logical rows at the same page indices at B = the smallest pool holding
# them, the engine's size and the largest pool the GPU can allocate, in a child process that owns
# the whole device.
# ---------------------------------------------------------------------------------------------
# Page rows and the engine's pool size (runtime-value dump, as REAL_SWA / REAL_COMP below) per role.
POOL_ROLES = {"swa": (SWA_PAGE, 188160), "r2": (64, 453120), "r1": (128, 151040)}
POOL_RESERVE = 6 << 30  # bytes left free next to the largest pool, for q, out, workspace, reference
# The largest pool is sized from the free memory of the run; it must reach at least this many rows
# (262.5 GiB of 1024-byte rows, what the contract certifies as its upper bound).
POOL_MIN_MAX_ROWS = 2150400 * 128
POOL_BOUNDARIES = tuple(1 << k for k in range(21, 31))


POOL_CTX = [(0, 300), (4999, 240), (131, 1), (16000, 64)]
POOL_DEC = [16383, 77, 4000]


def _pool_tables(role, pages, gen, other):
    """Page tables of POOL_CTX + POOL_DEC whose target-role rows live only in `pages`, in order."""
    page = POOL_ROLES[role][0]
    reqs = POOL_CTX + [(p, 1) for p in POOL_DEC]
    if role == "swa":
        _, comp_bts = _env_tables(reqs, gen, 2, 64, 64, other.shape[0])
        return pages[: 2 * len(reqs)].view(len(reqs), 2), comp_bts
    ratio = ENV_LAYERS[role][0]
    swa_bt, _ = _env_tables(reqs, gen, 0, page, other.shape[0], 1)
    need = [max(1, -(-((c + q) // ratio) // page)) for c, q in reqs]
    assert sum(need) == pages.numel()
    starts = [0]
    for n in need:
        starts.append(starts[-1] + n)
    return swa_bt, [pages[starts[i] : starts[i + 1]] for i in range(len(reqs))]


def _pool_role(role, results):
    """Pool-size invariance for one pool role, in a process that owns the device."""
    page, engine_pages = POOL_ROLES[role]
    layer = "r2" if role == "swa" else role
    gen = torch.Generator(device=DEV).manual_seed({"swa": 1, "r2": 2, "r1": 3}[role])
    other_page = 64 if role == "swa" else SWA_PAGE
    other = torch.randn(
        4096 if role == "swa" else 256, 1, other_page, HEAD_DIM, generator=gen, device=DEV
    )
    other = other.to(torch.bfloat16)
    reqs = POOL_CTX + [(p, 1) for p in POOL_DEC]
    ratio = ENV_LAYERS[layer][0]
    n_pages = (
        2 * len(reqs)
        if role == "swa"
        else sum(max(1, -(-((c + q) // ratio) // page)) for c, q in reqs)
    )
    # Distinct logical pages scattered over [0, 4 * n_pages); the highest one ends the smallest pool.
    logical = torch.cat(
        [
            torch.randperm(4 * n_pages - 1, generator=gen, device=DEV)[: n_pages - 1],
            torch.tensor([4 * n_pages - 1], device=DEV),
        ]
    )
    logical = logical[torch.randperm(n_pages, generator=gen, device=DEV)]
    b_small = 4 * n_pages
    data = torch.randn(n_pages, 1, page, HEAD_DIM, generator=gen, device=DEV).to(torch.bfloat16)
    atol = 2.0**-8 * max(data.float().abs().max().item(), other.float().abs().max().item())
    tables = _pool_tables(role, logical, gen, other)
    qs = {
        h: torch.randn(
            sum(q for _, q in POOL_CTX) + len(POOL_DEC), h, HEAD_DIM, generator=gen, device=DEV
        ).to(torch.bfloat16)
        for h in (16, 64)
    }
    # The members' head counts draw their q from a generator of their own (the H 16 / 64 q stay as
    # they were); the far-end placement below renames the pages every head count's step attends.
    gen_m = torch.Generator(device=DEV).manual_seed(7 + {"swa": 1, "r2": 2, "r1": 3}[role])
    qs.update(
        {
            h: torch.randn(
                sum(q for _, q in POOL_CTX) + len(POOL_DEC),
                h,
                HEAD_DIM,
                generator=gen_m,
                device=DEV,
            ).to(torch.bfloat16)
            for h in MEMBER_HEADS
        }
    )
    ws = _workspace()

    def step(pool, h, tables):
        pools = {
            "swa": pool if role == "swa" else other,
            "comp": {layer: other if role == "swa" else pool},
        }
        st = _Step(
            POOL_CTX, POOL_DEC, h, layer, pools, torch.Generator(device=DEV).manual_seed(h), *tables
        )
        st.q3.copy_(qs[h])
        return st

    def run_all(pool, tables=tables):
        return {h: step(pool, h, tables).run(ws).clone() for h in ALL_HEADS}

    def make(b):
        pool = torch.zeros(b, 1, page, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        pool[logical] = data
        return pool

    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    b_max = (free - POOL_RESERVE) // (page * HEAD_DIM * 2)
    b_mid = engine_pages
    assert b_small < b_mid < b_max and b_max * page >= POOL_MIN_MAX_ROWS, (b_small, b_mid, b_max)
    entry = {
        "B_small": b_small,
        "B_mid": b_mid,
        "B_max": b_max,
        "page_rows": page,
        "device_free_bytes": free,
        "device_total_bytes": total,
    }
    pool = make(b_small)
    ref_outs = run_all(pool)
    # Rows of the role pool each head count's step attends (for the far-end placement below).
    attended = None
    for h in ALL_HEADS:
        st = step(pool, h, tables)
        entry.setdefault("ref_excess", 0.0)
        entry["ref_excess"] = max(entry["ref_excess"], _excess(ref_outs[h], st.reference(), atol))
        col = torch.arange(st.idx.shape[1], device=DEV)
        att = (col[None] < st.active[:, None].long()) & (
            (col[None] >= WINDOW) | (col[None] < st.counts[:, None])
        )
        sl = slice(0, WINDOW) if role == "swa" else slice(WINDOW, None)
        rows = st.idx[:, sl][att[:, sl] & (st.idx[:, sl] >= 0)]
        pages_h = set((rows // page).unique().tolist())
        attended = pages_h if attended is None else attended & pages_h
    same = {"small": True}
    del pool
    torch.cuda.empty_cache()
    pool = make(b_mid)
    same["mid"] = all(torch.equal(o, ref_outs[h]) for h, o in run_all(pool).items())
    del pool
    torch.cuda.empty_cache()
    pool = make(b_max)
    same["max"] = all(torch.equal(o, ref_outs[h]) for h, o in run_all(pool).items())
    for name, b in (("max_view_small", b_small), ("max_view_mid", b_mid)):
        same[name] = all(torch.equal(o, ref_outs[h]) for h, o in run_all(pool[:b]).items())
    # The same logical step with its pages renamed onto the far end of the largest pool: the last
    # page, and pages straddling rows 2^21 .. 2^30, go to logical pages the step attends.
    total_rows = b_max * page
    must = [b_max - 1] + [r // page for r in POOL_BOUNDARIES if r < total_rows]
    must += [r // page - 1 for r in POOL_BOUNDARIES if r < total_rows]
    att_logical = [p for p in logical.tolist() if p in attended]
    rest_logical = [p for p in logical.tolist() if p not in attended]
    lut = torch.full((b_small,), -1, dtype=torch.int64, device=DEV)
    order = att_logical + rest_logical
    far_pool = list(dict.fromkeys(must[: len(att_logical)]))
    filler = (
        b_max - 1 - 16 * n_pages + torch.randperm(16 * n_pages, generator=gen, device=DEV)
    ).tolist()
    far_pool += [p for p in filler if p not in set(far_pool)]
    far = torch.tensor(far_pool[:n_pages], dtype=torch.int64, device=DEV)
    lut[torch.tensor(order, device=DEV)] = far
    pool[lut[logical]] = data
    far_tables = (
        (lut[tables[0]], tables[1]) if role == "swa" else (tables[0], [lut[c] for c in tables[1]])
    )
    far_named = set(far.tolist())
    alias = sorted({(p * page % WRAP_ROWS) // page for p in far_named} - far_named)
    a = torch.tensor(alias, dtype=torch.int64, device=DEV)
    pool[a] = torch.randn(a.numel(), 1, page, HEAD_DIM, generator=gen, device=DEV).to(
        torch.bfloat16
    )
    far_ex, wrap_ex, max_row = 0.0, float("inf"), -1
    far_same = True
    for h in ALL_HEADS:
        st = step(pool, h, far_tables)
        o = st.run(ws)
        far_same &= torch.equal(o, ref_outs[h])
        far_ex = max(far_ex, _excess(o, st.reference(), atol))
        sl = slice(0, WINDOW) if role == "swa" else slice(WINDOW, None)
        col = torch.arange(st.idx.shape[1], device=DEV)
        att = (col[None] < st.active[:, None].long()) & (
            (col[None] >= WINDOW) | (col[None] < st.counts[:, None])
        )
        max_row = max(max_row, int(torch.where(att[:, sl], st.idx[:, sl], -1).max()))
        wrapped = st.idx.clone()
        seg = wrapped[:, sl]
        wrapped[:, sl] = torch.where(seg >= 0, seg % WRAP_ROWS, seg)
        wrap_ex = min(wrap_ex, _excess(o, st.reference(wrapped), atol))
    same["max_far_pages_renamed"] = bool(far_same)
    entry.update(
        bitwise=same,
        far_excess=far_ex,
        wrap_excess=wrap_ex,
        max_attended_row=max_row,
        last_row=total_rows - 1,
        must=must,
        must_attended=sorted(set(must) & set(lut[torch.tensor(att_logical, device=DEV)].tolist())),
        pool_shape=list(pool.shape),
    )
    results[role] = entry
    del pool
    torch.cuda.empty_cache()


def _pool_child():
    results = {}
    for role in POOL_ROLES:
        _pool_role(role, results)
    print("POOL_CHILD_RESULT " + json.dumps(results), flush=True)


@pytest.fixture(scope="module")
def pool_invariance():
    torch.cuda.empty_cache()
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--pool-child"],
        env=_CHILD_ENV,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("POOL_CHILD_RESULT ")]
    assert proc.returncode == 0 and line, (
        proc.returncode,
        proc.stdout[-3000:],
        proc.stderr[-3000:],
    )
    return json.loads(line[-1][len("POOL_CHILD_RESULT ") :])


@pytest.mark.parametrize("role", list(POOL_ROLES))
def test_pool_size_invariance(pool_invariance, role):
    """A mixed step (context + decode, H 8, 16, 32 and 64) over the same logical rows at the same page
    indices is bitwise identical at the smallest pool that holds them, the engine's pool size and
    the largest pool the GPU can allocate (separate allocations, and views of the largest one); at
    that largest pool, rows in its last page and in pages straddling rows 2^21..2^30 match the
    reference, and reading them at (row mod 2^22) instead fails the gate by a wide margin."""
    r = pool_invariance[role]
    print(f"pool invariance {role}: {r}")
    assert all(r["bitwise"].values()), r["bitwise"]
    assert r["ref_excess"] <= 1.0 and r["far_excess"] <= 1.0, r
    assert r["max_attended_row"] >= r["last_row"] - r["page_rows"] + 1, r
    assert r["B_max"] - 1 in r["must_attended"], r
    assert r["wrap_excess"] > 10.0, r


# ---------------------------------------------------------------------------------------------
# the engine's real pool sizes and decode batch sizes (runtime-value dump)
# ---------------------------------------------------------------------------------------------
# Pool geometry the core DeepSeek-V4.1 target's cache manager allocated per rank in its boot and
# eval runs: one SWA pool shared by all 40 layers ([188160, 128, 512], 24.7 GB bf16) and one
# compressed pool per KV-source owner ([453120, 64, 512] at ratio 2, 29.7 GB; [151040, 128, 512] at
# ratio 1, 19.8 GB). The target hands them over as [pages, 1, page, 512] views. A 1024-byte row puts
# row 2^21 at byte 2^31, row 2^22 at byte 2^32 and at element 2^31, so a 32-bit offset anywhere in
# the address path would wrap from those rows on. Every case below attends rows in the LAST page of
# each pool it uses, plus pages straddling rows 2^21 / 2^22 / 2^23 / 2^24, plus the first page.
REAL_SWA = (188160, 128)
REAL_COMP = {2: (453120, 64), 1: (151040, 128)}
WRAP_ROWS = 1 << 22  # rows per 2^32 bytes (and per 2^31 bf16 elements) at 1024 B per row
ROW_BOUNDARIES = (1 << 21, 1 << 22, 1 << 23, 1 << 24)
# Decode batch sizes the engine ran: eager mixed steps saw G in 1..10, 16..19, 24, 27..32. The decode
# CUDA graphs are captured per rank at every batch 1..8 (the boot gate, padding off) and at 1, 2, 4,
# 8, 16, 24, 32 (the accuracy gate, padding on): the capture set. The native-storage compact-read
# capture tests below still use this set (their buffers are sized for max_batch_size 32).
GRAPH_BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32]
assert set(GRAPH_BATCHES) <= set(GSET)
# DSpark target verify: 1 + 5 draft tokens per request, dense [G, 6, H, 512] with next_n 6.
VERIFY_Q = 6
# Decode positions of requests 1.. (request 0 is always a long one, so that its first page -- the
# pool's last page -- holds window rows, the SWA pool's last row among them, and compressed rows).
REAL_POSITIONS = [0, 3, 60, 126, 127, 128, 500, 1500, 5000, 9000, 65000]
# Envelope E4: the kv lengths the engine can hand this call at max_seq_len L in {16384, 32768, 65536,
# 131072, 262144, 1048576}: L itself (DSpark off), the cache manager's length Lm = L + 14 and Lm + 5
# (DSpark on: positions reach about Lm + 5). Decode / verify requests cycle through REAL_POSITIONS
# interleaved with these, placed so that the request's seq_lens IS the listed kv length.
LONG_KV = sorted(
    {L + d for L in (16384, 32768, 65536, 131072, 262144, 1048576) for d in (0, 14, 19)}
)


def _cached_positions(q_len: int):
    """The LONG_KV lengths (as cached prefixes for a q_len request) interleaved with REAL_POSITIONS,
    a long one first."""
    long = [kv - q_len for kv in LONG_KV]
    out = []
    for i in range(max(len(REAL_POSITIONS), len(long))):
        out += long[i : i + 1] + REAL_POSITIONS[i : i + 1]
    return out


def _paged_rows(requests, gen, width, ratio, swa_bt, comp_bt, swa_page, comp_page):
    """_build_rows, vectorized over every query row: request b's pages are the rows swa_bt[b] and
    comp_bt[b], used as rings (position p on page (p // page) % pages per request). Query j of
    request (cached, q_len) sits at pos = cached + j and sees window positions max(0, pos-127)..pos
    in order (-1 after them) and min(512, (pos + 1) // ratio) compressed positions: all of them
    while there are at most 512, else one random position per stratum [s * v // 512,
    (s + 1) * v // 512) (distinct and sorted, as the indexer top-k is), -1 after them.
    Returns (idx [N, width] int32, active [N] int32, seq_lens [B] int32, cu [B + 1] int32)."""
    q_lens = torch.tensor([q for _, q in requests], dtype=torch.int64, device=DEV)
    cached = torch.tensor([c for c, _ in requests], dtype=torch.int64, device=DEV)
    cu = torch.cat([torch.zeros(1, dtype=torch.int64, device=DEV), q_lens.cumsum(0)])
    n = int(cu[-1])
    req = torch.repeat_interleave(torch.arange(len(requests), device=DEV), q_lens)
    pos = cached[req] + torch.arange(n, device=DEV) - cu[req]
    wpos = (pos - (WINDOW - 1)).clamp_min(0)[:, None] + torch.arange(WINDOW, device=DEV)[None]
    ring = swa_bt[req]
    widx = ring.gather(1, (wpos // swa_page) % ring.shape[1]) * swa_page + wpos % swa_page
    widx = torch.where(wpos <= pos[:, None], widx, -1)
    if width == WINDOW:
        idx, active = widx, torch.full((n,), WINDOW, dtype=torch.int64, device=DEV)
    else:
        k = width - WINDOW
        vis = (pos + 1) // ratio
        strata = torch.arange(k + 1, device=DEV)
        lo = (strata[None, :-1] * vis[:, None]) // k
        hi = (strata[None, 1:] * vis[:, None]) // k
        u = torch.rand(n, k, generator=gen, device=DEV)
        sel = torch.where(vis[:, None] > k, lo + (u * (hi - lo)).long(), strata[None, :-1])
        valid = sel < vis[:, None]
        pages = comp_bt[req]
        cpage = pages.gather(1, (sel // comp_page) % pages.shape[1])
        cidx = torch.where(valid, cpage * comp_page + sel % comp_page, -1)
        idx, active = torch.cat([widx, cidx], 1), WINDOW + valid.sum(1)
    seq_lens = cached + q_lens
    return idx.int().contiguous(), active.int(), seq_lens.int(), cu.int()


class _RealPools:
    """Zero-initialized pools of the engine's size; each case fills only the pages it names."""

    def __init__(self, layer: str):
        self.layer = layer
        self.ratio = {"r2": 2, "r1": 1, "r0": 0}[layer]
        self.swa_pages, self.swa_page = REAL_SWA
        self.swa = torch.zeros(
            self.swa_pages, 1, self.swa_page, HEAD_DIM, dtype=torch.bfloat16, device=DEV
        )
        if self.ratio:
            self.comp_pages, self.comp_page = REAL_COMP[self.ratio]
            self.comp = torch.zeros(
                self.comp_pages, 1, self.comp_page, HEAD_DIM, dtype=torch.bfloat16, device=DEV
            )
            self.topk = WINDOW + INDEX_TOPK
        else:
            # Window-only layers pass a one-page placeholder; NaN, so that a read of it would show.
            self.comp_pages, self.comp_page = 1, 64
            self.comp = torch.full(
                (1, 1, 64, HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=DEV
            )
            self.topk = WINDOW

    @staticmethod
    def _pick(gen: torch.Generator, total: int, page: int, n: int) -> torch.Tensor:
        """n distinct pages: the pool's last page first, then pages straddling the row boundaries, the
        first page, and random pages from the pool's last stretch -- shuffled after the first."""
        must = [total - 1]
        for r in ROW_BOUNDARIES:
            if r < total * page:
                must += [r // page - 1, r // page]
        must.append(0)
        far = (total - 1 - 8 * n + torch.randperm(8 * n, generator=gen, device=DEV)).tolist()
        pages = list(dict.fromkeys(must + far))[:n]
        rest = torch.tensor(pages[1:], dtype=torch.int64, device=DEV)
        rest = rest[torch.randperm(rest.numel(), generator=gen, device=DEV)]
        return torch.cat([torch.tensor([pages[0]], dtype=torch.int64, device=DEV), rest])

    @staticmethod
    def _fill(gen: torch.Generator, pool: torch.Tensor, pages: torch.Tensor) -> None:
        """Random rows in the named pages, and independent random rows in the pages a 32-bit wrap of
        their row offsets would land on (row mod 2^22), so a wrapped read is a wrong value, not 0."""
        page = pool.shape[2]
        pool[pages] = torch.randn(pages.numel(), 1, page, HEAD_DIM, generator=gen, device=DEV).to(
            torch.bfloat16
        )
        named = set(pages.tolist())
        alias = sorted({(p * page % WRAP_ROWS) // page for p in named} - named)
        if alias:
            a = torch.tensor(alias, dtype=torch.int64, device=DEV)
            pool[a] = torch.randn(a.numel(), 1, page, HEAD_DIM, generator=gen, device=DEV).to(
                torch.bfloat16
            )

    def rows(self, requests, gen: torch.Generator, swa_per_req: int = 2, comp_per_req: int = 4):
        """Index rows for `requests`: request b's pages are entries [b * per_req, (b + 1) * per_req) of
        the picked page lists (request 0 gets the pool's last page first)."""
        b = len(requests)
        swa_bt = self._pick(gen, self.swa_pages, self.swa_page, swa_per_req * b)
        self._fill(gen, self.swa, swa_bt)
        comp_bt = None
        if self.ratio:
            comp_bt = self._pick(gen, self.comp_pages, self.comp_page, comp_per_req * b)
            self._fill(gen, self.comp, comp_bt)
        return _paged_rows(
            requests,
            gen,
            self.topk,
            max(self.ratio, 1),
            swa_bt.view(b, swa_per_req),
            comp_bt.view(b, comp_per_req) if comp_bt is not None else None,
            self.swa_page,
            self.comp_page,
        )

    def atol(self, idx: torch.Tensor) -> float:
        """The file's gate, 2^-8 * max|V|, over the rows this call names (the rest of the pool is 0)."""
        w = idx[:, :WINDOW]
        m = self.swa.view(-1, HEAD_DIM)[torch.unique(w[w >= 0]).long()].float().abs().max()
        c = idx[:, WINDOW:]
        if (c >= 0).any():
            m = torch.maximum(
                m, self.comp.view(-1, HEAD_DIM)[torch.unique(c[c >= 0]).long()].float().abs().max()
            )
        return 2.0**-8 * m.item()

    def assert_far_end_attended(self, idx, active, counts) -> None:
        """The call really attends rows in the last page of each pool, past row 2^22."""
        col = torch.arange(idx.shape[1], device=DEV)
        att = (col[None] < active[:, None].long()) & (
            (col[None] >= WINDOW) | (col[None] < counts[:, None])
        )
        w = torch.where(att[:, :WINDOW], idx[:, :WINDOW], -1)
        swa_last = (self.swa_pages - 1) * self.swa_page
        assert int(w.max()) >= swa_last >= WRAP_ROWS, (int(w.max()), swa_last)
        if self.ratio:
            c = torch.where(att[:, WINDOW:], idx[:, WINDOW:], -1)
            comp_last = (self.comp_pages - 1) * self.comp_page
            assert int(c.max()) >= comp_last >= WRAP_ROWS, (int(c.max()), comp_last)

    def release(self) -> None:
        del self.swa, self.comp
        torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=["r2", "r1", "r0"])
def real_pools(request):
    pools = _RealPools(request.param)
    yield pools
    pools.release()


def _decode_requests(batch: int, shift: int = 0):
    # Position 20095 + 256k: offset 127 of ring page 0, i.e. the SWA pool's very last row.
    first = (20095 + 256 * shift, 1)
    pos = _cached_positions(1)
    return [first] + [(pos[(i + shift) % len(pos)], 1) for i in range(batch - 1)]


def _check_atol(o, ref, atol):
    """The entry's gate at an explicit atol (2^-8 * max|V| over the rows the caller names)."""
    op.compare(o.reshape(ref.shape), (ref, atol * 2.0**8))


@pytest.mark.parametrize("num_heads", ALL_HEADS)
def test_real_pools_decode_every_batch_size(real_pools, num_heads):
    """Eager decode at every batch size 1..256 over the engine's pools, far-end rows attended, kv
    lengths up to the E4 extreme (LONG_KV) among the requests."""
    p = real_pools
    gen = torch.Generator(device=DEV).manual_seed(7000 + num_heads + p.ratio)
    sinks = _sinks(num_heads)
    for batch in range(1, E_B + 1):
        requests = _decode_requests(batch, shift=batch)
        idx, active, seq_lens, cu = p.rows(requests, gen)
        counts = _window_counts(seq_lens, cu)
        p.assert_far_end_attended(idx, active, counts)
        q = (
            torch.randn(batch, 1, num_heads, HEAD_DIM, generator=gen, device=DEV)
            .mul_(2)
            .to(torch.bfloat16)
        )
        o = _run(q, p.swa, p.comp, idx, active, seq_lens, sinks)
        assert o.shape == (batch, 1, num_heads, HEAD_DIM)
        assert torch.isfinite(o).all(), f"B={batch}"
        ref = _reference(
            q.view(batch, num_heads, HEAD_DIM), p.swa, p.comp, idx, active, counts, sinks
        )
        _check_atol(o, ref, p.atol(idx))


@pytest.mark.parametrize("num_heads", ALL_HEADS)
def test_real_pools_prefill_varlen(real_pools, num_heads):
    """Varlen prefill (fresh 2048-token prompt + chunked continuation + 1-token row) over the
    engine's pools, far-end rows attended."""
    p = real_pools
    gen = torch.Generator(device=DEV).manual_seed(8000 + num_heads + p.ratio)
    requests = [(0, 2048), (6000, 300), (90, 1)]
    idx, active, seq_lens, cu = p.rows(requests, gen, swa_per_req=2, comp_per_req=8)
    counts = _window_counts(seq_lens, cu)
    p.assert_far_end_attended(idx, active, counts)
    n = idx.shape[0]
    q = torch.randn(n, num_heads, HEAD_DIM, generator=gen, device=DEV).mul_(2).to(torch.bfloat16)
    sinks = _sinks(num_heads)
    o = _run(q, p.swa, p.comp, idx, active, seq_lens, sinks, cu=cu, max_q_len=2048)
    ref = _reference(q, p.swa, p.comp, idx, active, counts, sinks)
    _check_atol(o, ref, p.atol(idx))


@pytest.mark.parametrize("batch", GSET)
@pytest.mark.parametrize("num_heads", ALL_HEADS)
def test_real_pools_decode_cuda_graph(real_pools, num_heads, batch):
    """Decode captured into a CUDA graph at every batch of the envelope's capture set GSET (46
    values, 1..256); replay equals eager bitwise, then two replays on new q / index rows / lengths
    (new far-end pages) equal a fresh eager call on the same inputs bitwise and match the
    reference."""
    p = real_pools
    gen = torch.Generator(device=DEV).manual_seed(9000 + 100 * batch + num_heads + p.ratio)
    sinks = _sinks(num_heads)

    def make(shift):
        requests = _decode_requests(batch, shift)
        idx, active, seq_lens, cu = p.rows(requests, gen)
        p.assert_far_end_attended(idx, active, _window_counts(seq_lens, cu))
        q = (
            torch.randn(batch, 1, num_heads, HEAD_DIM, generator=gen, device=DEV)
            .mul_(2)
            .to(torch.bfloat16)
        )
        return q, idx, active, seq_lens, cu

    q, idx, active, seq_lens, cu = make(0)
    ws = _workspace()
    out = torch.empty(batch, 1, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=DEV)

    def call():
        return op(
            query=q,
            swa_kv_cache=p.swa,
            compressed_kv_cache=p.comp,
            workspace_buffer=ws,
            sparse_indices=idx,
            sparse_topk_lens=active,
            seq_lens=seq_lens,
            sinks=sinks,
            softmax_scale=SCALE,
            out=out,
        )

    call()
    eager = out.clone()
    ref = _reference(
        q.view(batch, num_heads, HEAD_DIM),
        p.swa,
        p.comp,
        idx,
        active,
        _window_counts(seq_lens, cu),
        sinks,
    )
    _check_atol(eager, ref, p.atol(idx))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    for shift in (1, 2):
        nq, nidx, nactive, nseq, ncu = make(shift)
        q.copy_(nq)
        idx.copy_(nidx)
        active.copy_(nactive)
        seq_lens.copy_(nseq)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        call()  # a fresh eager call on the same inputs
        torch.cuda.synchronize()
        assert torch.equal(replayed, out), f"replay != eager at shift {shift}"
        ref = _reference(
            nq.view(batch, num_heads, HEAD_DIM),
            p.swa,
            p.comp,
            nidx,
            nactive,
            _window_counts(nseq, ncu),
            sinks,
        )
        _check_atol(out, ref, p.atol(nidx))
    del graph


def _verify_requests(batch: int, shift: int = 0):
    # Request 0's last verify query sits at 20095 + 256k: offset 127 of ring page 0, the SWA pool's
    # very last row.
    first = (20090 + 256 * shift, VERIFY_Q)
    pos = _cached_positions(VERIFY_Q)
    return [first] + [(pos[(i + shift) % len(pos)], VERIFY_Q) for i in range(batch - 1)]


@pytest.mark.parametrize("num_heads", ALL_HEADS)
def test_real_pools_verify_every_batch_size(real_pools, num_heads):
    """Eager DSpark target verify (q_len 6, dense) at every batch size 1..256 over the engine's pools,
    kv lengths up to the E4 extreme (LONG_KV) among the requests."""
    p = real_pools
    gen = torch.Generator(device=DEV).manual_seed(7500 + num_heads + p.ratio)
    sinks = _sinks(num_heads)
    for batch in range(1, E_B + 1):
        idx, active, seq_lens, cu = p.rows(_verify_requests(batch, shift=batch), gen)
        counts = _window_counts(seq_lens, cu)
        p.assert_far_end_attended(idx, active, counts)
        q = (
            torch.randn(batch, VERIFY_Q, num_heads, HEAD_DIM, generator=gen, device=DEV)
            .mul_(2)
            .to(torch.bfloat16)
        )
        o = _run(q, p.swa, p.comp, idx, active, seq_lens, sinks)
        assert o.shape == (batch, VERIFY_Q, num_heads, HEAD_DIM)
        ref = _reference(q.view(-1, num_heads, HEAD_DIM), p.swa, p.comp, idx, active, counts, sinks)
        _check_atol(o, ref, p.atol(idx))


@pytest.mark.parametrize("batch", GSET)
@pytest.mark.parametrize("num_heads", ALL_HEADS)
def test_real_pools_verify_q_len_6_cuda_graph(real_pools, num_heads, batch):
    """DSpark verify (q_len 6) captured at every batch of GSET (1..256) on the engine's pools. The
    first replay equals the eager call bitwise; three further replays on new q, index rows, per-row
    lengths and seq_lens (new far-end pages) equal a fresh eager call on the same inputs bitwise and
    match the reference. Two wrong readings of the 6-query block must fail the gate."""
    p = real_pools
    gen = torch.Generator(device=DEV).manual_seed(9500 + 100 * batch + num_heads + p.ratio)
    sinks = _sinks(num_heads)
    rows = VERIFY_Q * batch

    def make(shift):
        idx, active, seq_lens, cu = p.rows(_verify_requests(batch, shift), gen)
        counts = _window_counts(seq_lens, cu)
        p.assert_far_end_attended(idx, active, counts)
        q = (
            torch.randn(batch, VERIFY_Q, num_heads, HEAD_DIM, generator=gen, device=DEV)
            .mul_(2)
            .to(torch.bfloat16)
        )
        return q, idx, active, seq_lens, counts

    q, idx, active0, seq0, counts = make(0)
    # Row / request prefixes of step buffers sized for the largest graph batch, as the target passes them.
    lens_buf = torch.full((VERIFY_Q * GSET[-1],), WINDOW, dtype=torch.int32, device=DEV)
    kv_lens_buf = torch.full((GSET[-1],), VERIFY_Q, dtype=torch.int32, device=DEV)
    active, seq_lens = lens_buf[:rows], kv_lens_buf[:batch]
    active.copy_(active0)
    seq_lens.copy_(seq0)
    ws = _workspace()

    def call(out):
        return op(
            query=q,
            swa_kv_cache=p.swa,
            compressed_kv_cache=p.comp,
            workspace_buffer=ws,
            sparse_indices=idx,
            sparse_topk_lens=active,
            seq_lens=seq_lens,
            sinks=sinks,
            softmax_scale=SCALE,
            out=out,
        )

    shape = (batch, VERIFY_Q, num_heads, HEAD_DIM)
    eager = call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))  # the engine's eager warm-up
    ref = _reference(q.view(rows, num_heads, HEAD_DIM), p.swa, p.comp, idx, active, counts, sinks)
    _check_atol(eager, ref, p.atol(idx))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Allocated inside the captured forward, as the target allocates its [T, H, 512] output.
        o = torch.empty(rows, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        out = call(o.view(shape))
    assert out.data_ptr() == o.data_ptr()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    for shift in (1, 2, 3):
        nq, nidx, nactive, nseq, ncounts = make(shift)
        q.copy_(nq)
        idx.copy_(nidx)
        active.copy_(nactive)
        seq_lens.copy_(nseq)
        out.fill_(float("nan"))
        graph.replay()
        fresh = call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
        torch.cuda.synchronize()
        assert torch.equal(out, fresh), f"replay != eager at shift {shift}"
        qf = nq.view(rows, num_heads, HEAD_DIM)
        ref = _reference(qf, p.swa, p.comp, nidx, nactive, ncounts, sinks)
        atol = p.atol(nidx)
        _check_atol(out, ref, atol)

        def excess(r):
            return ((out.float().reshape(r.shape) - r).abs() / (atol + RTOL * r.abs())).max().item()

        correct = excess(ref)
        if shift != 1 or batch < 2:
            print(
                f"verify-replay {p.layer} H{num_heads} G{batch} shift {shift}: correct {correct:.3f}"
            )
            continue
        # q read as [6, G] instead of [G, 6]: index row b*6+j paired with query row j*G+b.
        swapped = (
            qf.reshape(VERIFY_Q, batch, num_heads, HEAD_DIM)
            .transpose(0, 1)
            .reshape(rows, num_heads, HEAD_DIM)
        )
        wrong = {
            "q_block_transposed": excess(
                _reference(swapped, p.swa, p.comp, nidx, nactive, ncounts, sinks)
            )
        }
        # Non-causal block: every query of a request gets the window count of its last query.
        last = torch.repeat_interleave(nseq.long().clamp(max=WINDOW), VERIFY_Q)
        if not torch.equal(last, ncounts):
            wrong["block_not_causal"] = excess(
                _reference(qf, p.swa, p.comp, nidx, nactive, last, sinks)
            )
        print(
            f"verify-controls {p.layer} H{num_heads} G{batch}: correct {correct:.3f}, wrong "
            f"{ {k: round(v, 1) for k, v in wrong.items()} }"
        )
        assert correct <= 1.0, correct
        for name, e in wrong.items():
            assert e > 10.0 * max(correct, 0.1), (
                f"{name} not discriminated: {e:.2f} vs {correct:.2f}"
            )
    del graph


def test_real_pools_wrap_control(real_pools):
    """The harness can see a 32-bit address wrap: reading every attended row at (row mod 2^22) -- the
    row a wrapped 32-bit byte offset (or 31-bit element offset) of a 1024-byte row lands on, filled
    here with independent data -- fails the same gate by a wide margin, while the true rows pass."""
    p = real_pools
    gen = torch.Generator(device=DEV).manual_seed(9900 + p.ratio)
    batch, h = 32, 16
    idx, active, seq_lens, cu = p.rows(_decode_requests(batch), gen)
    counts = _window_counts(seq_lens, cu)
    p.assert_far_end_attended(idx, active, counts)
    q = torch.randn(batch, 1, h, HEAD_DIM, generator=gen, device=DEV).mul_(2).to(torch.bfloat16)
    sinks = _sinks(h)
    o = _run(q, p.swa, p.comp, idx, active, seq_lens, sinks)
    qf = q.view(batch, h, HEAD_DIM)
    atol = p.atol(idx)

    def excess(ref):
        return ((o.float().reshape(ref.shape) - ref).abs() / (atol + RTOL * ref.abs())).max().item()

    correct = excess(_reference(qf, p.swa, p.comp, idx, active, counts, sinks))
    wrapped = torch.where(idx >= 0, idx % WRAP_ROWS, idx)
    assert (wrapped != idx).any()
    wrong = excess(_reference(qf, p.swa, p.comp, wrapped, active, counts, sinks))
    print(
        f"{p.layer}: pools swa {tuple(p.swa.shape)} comp {tuple(p.comp.shape)} "
        f"({torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated), max attended row "
        f"swa {int(idx[:, :WINDOW].max())} comp {int(idx[:, WINDOW:].max()) if idx.shape[1] > WINDOW else -1}; "
        f"gate-normalized error correct {correct:.3f}, wrapped-address reading {wrong:.1f}"
    )
    assert correct <= 1.0, correct
    assert wrong > 10.0 * max(correct, 0.1), (wrong, correct)


# ---------------------------------------------------------------------------------------------
# The native-storage compact read
# ---------------------------------------------------------------------------------------------
# Under native KV storage the target stores the KV quantized and, every step, dequantizes
# exactly the rows its attention reads into two persistent bf16 buffers, allocated once per cache
# manager (zero-filled, outside capture) and never cleared; it hands THOSE buffers to this call:
#
# - window: swa_c [max_num_tokens * 128, 512], passed as [max_num_tokens, 1, 128, 512]. Query t's
#   window column c is dequantized to row t * 128 + c, a -1 column as an all-zero (+0.0) row, and
#   the index table's window half becomes that numbering: arange(T * 128).view(T, 128), a prefix
#   view [:T] of one persistent [max_num_tokens, 128] arange buffer. The window half then holds no
#   -1; where the container path has a -1 the compact path names a zero row.
# - compressed: comp_c [max_batch_size * 132 * 128, 512]. Request b's entry e (every e < kv_len // r,
#   b the request's position in the step) at row b * L_r + e, L_r = 132 * 128 / r, passed per ratio
#   as [max_batch_size * 132, 1, 128 / r, 512] over its first max_batch_size * 132 * 128 / r rows.
#   The compressed half of the index is b * L_r + e for a valid top-k entry and -1 after them (the
#   index op maps the top-k through a synthetic page table; certified in that entry).
# - window-only (ratio-0) layers pass the renumbered window half itself ([T, 128], the arange view)
#   and a zeros [1, 1, 64, 512] placeholder as the compressed pool; ratio 1 / 2 layers pass the
#   concatenation [T, 640].
#
# Sizes: max_num_tokens 2048 or 8192 and max_batch_size 8 or 32 in every envelope configuration
# (the metadata's swa_local_indices / compress_block_tables, measured in the dequantizer entries'
# native-storage tests): swa_c 262,144 / 1,048,576 rows, comp_c 135,168 / 540,672 rows.
#
# The container path (bf16 KV storage) is the call this file certifies above: the same values read
# from the cache manager's pools through the block tables, with -1 in the index. Here the container
# pools are env_pools' random pools, the compact buffers are filled from them by native torch gathers
# (what the two dequantizers write; the dequantizers themselves are certified bitwise to the bf16
# container in their own entries), and every compact read is held BITWISE to the container read on
# the same values, and the container read to the fp32 reference. The buffers are pre-filled with
# finite random rows (a previous step's content), so a row a step fails to rewrite would show.
KVQ_COLS = 132
KVQ_L = {2: KVQ_COLS * 64, 1: KVQ_COLS * 128}
KVQ_CONFIGS = ((8192, 32), (8192, 8), (2048, 32), (2048, 8))  # (max_num_tokens, max_batch_size)
KVQ_TAIL = 16397  # DSpark: generation positions reach max_seq_len 16398 - 1; contexts stay < 16384


class _KvqBuffers:
    """The target's two read buffers for one (max_num_tokens, max_batch_size), with the views and the
    window numbering it hands the call."""

    def __init__(self, max_t, seqs, gen):
        self.max_t, self.seqs = max_t, seqs
        self.swa_c = torch.randn(
            max_t * WINDOW, HEAD_DIM, generator=gen, device=DEV, dtype=torch.bfloat16
        )
        self.comp_c = torch.randn(
            seqs * KVQ_COLS * 128, HEAD_DIM, generator=gen, device=DEV, dtype=torch.bfloat16
        )
        self.swa_c4 = self.swa_c.view(max_t, 1, WINDOW, HEAD_DIM)
        self.comp_c4 = {
            r: self.comp_c[: seqs * KVQ_COLS * (128 // r)].view(
                seqs * KVQ_COLS, 1, 128 // r, HEAD_DIM
            )
            for r in (1, 2)
        }
        self.win_idx = torch.arange(max_t * WINDOW, dtype=torch.int32, device=DEV).view(
            max_t, WINDOW
        )
        self.dummy = torch.zeros(1, 1, 64, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        for v in self.comp_c4.values():
            assert (
                v.data_ptr() == self.comp_c.data_ptr() and v.shape[0] * v.shape[2] >= POOL_MIN_ROWS
            )


@pytest.fixture(scope="module")
def kvq_buffers():
    gen = torch.Generator(device=DEV).manual_seed(1_400_000)
    bufs = {cfg: _KvqBuffers(*cfg, gen) for cfg in KVQ_CONFIGS}
    yield bufs
    del bufs
    torch.cuda.empty_cache()


def _kvq_rows(requests, floors, gen, width, ratio, comp_page, swa_bt, comp_bts):
    """_env_rows with the local compressed positions kept (a random subset in score order when more
    than 512 are visible, the newest visible entry always among them), and an optional window floor per request
    (the metadata's swa_floor, an encoder-replay step's replay start: the window of position p is
    max(floor, p - 127) .. p, then -1, while the call's window count stays min(128, p + 1), so the
    -1s past the floored window sit INSIDE the count and are attended). Returns idx [N, width]
    (container rows, -1), active [N], seq_lens [B], cu [B+1], counts [N], req [N] (the request's
    position in the step) and loc [N, 512] (the compressed positions e, -1) or None."""
    ar = torch.arange(WINDOW, device=DEV)
    idx_l, act_l, cnt_l, req_l, loc_l = [], [], [], [], []
    for b, ((cached, q_len), floor) in enumerate(zip(requests, floors)):
        pos = torch.arange(cached, cached + q_len, device=DEV)
        start = (pos - (WINDOW - 1)).clamp_min(floor).clamp_min(0)
        wpos = start[:, None] + ar[None]
        ring = swa_bt[b]
        widx = ring[(wpos // SWA_PAGE) % ring.numel()] * SWA_PAGE + wpos % SWA_PAGE
        widx = torch.where(wpos <= pos[:, None], widx, -1).int()
        cnt_l.append((pos + 1).clamp(max=WINDOW))
        req_l.append(torch.full((q_len,), b, dtype=torch.int64, device=DEV))
        if width == WINDOW:
            idx_l.append(widx)
            act_l.append(torch.full((q_len,), WINDOW, dtype=torch.int32, device=DEV))
            continue
        k_max = width - WINDOW
        vis = (pos + 1) // ratio
        vmax = (cached + q_len) // ratio
        pages = comp_bts[b]
        assert pages.numel() * comp_page >= vmax
        sels, valids = [], []
        for s0 in range(0, q_len, 1024):
            v = vis[s0 : s0 + 1024]
            if vmax <= k_max:
                sel = torch.arange(k_max, device=DEV)[None].expand(v.numel(), k_max)
                valid = sel < v[:, None]
            else:
                score = torch.rand(v.numel(), vmax, generator=gen, device=DEV)
                col = torch.arange(vmax, device=DEV)[None]
                score.masked_fill_(
                    col == (v - 1)[:, None], 2.0
                )  # the newest visible entry is always picked
                score.masked_fill_(col >= v[:, None], -1.0)
                val, sel = score.topk(k_max, dim=1)
                valid = val >= 0  # score order: valid entries first, then -1
            sels.append(sel)
            valids.append(valid)
        sel, valid = torch.cat(sels), torch.cat(valids)
        sel0 = torch.where(valid, sel, 0)
        cidx = torch.where(valid, pages[sel0 // comp_page] * comp_page + sel0 % comp_page, -1).int()
        idx_l.append(torch.cat([widx, cidx], 1))
        loc_l.append(torch.where(valid, sel, -1).int())
        act_l.append((WINDOW + valid.sum(1)).int())
    idx = torch.cat(idx_l).contiguous()
    seq_lens = torch.tensor([c + q for c, q in requests], dtype=torch.int32, device=DEV)
    cu = torch.tensor([0] + [q for _, q in requests], dtype=torch.int32, device=DEV).cumsum(0).int()
    loc = torch.cat(loc_l) if loc_l else None
    return idx, torch.cat(act_l), seq_lens, cu, torch.cat(cnt_l), torch.cat(req_l), loc


class _KvqStep:
    """One engine step (nc context requests, then G generation requests of next_n rows each) in both
    read forms: the container form (env_pools' pools, idx with -1) and the native-storage compact form
    (bufs' views, the renumbered index cidx). Per-row and per-request tensors are prefixes of
    envelope-sized buffers, as the runtime's metadata buffers are."""

    def __init__(self, ctx, gen_pos, next_n, num_heads, layer, pools, bufs, gen, floors=None):
        ratio, page = ENV_LAYERS[layer]
        self.ratio, self.layer, self.bufs = ratio, layer, bufs
        self.width = WINDOW + (INDEX_TOPK if ratio else 0)
        self.requests = list(ctx) + [(p, next_n) for p in gen_pos]
        self.floors = list(floors if floors is not None else [0] * len(ctx)) + [0] * len(gen_pos)
        self.nc, self.tc = len(ctx), sum(q for _, q in ctx)
        self.B, self.G, self.next_n = len(self.requests), len(gen_pos), next_n
        assert self.B <= bufs.seqs
        self.max_ctx_q = max((q for _, q in ctx), default=0)
        self.swa, self.comp = pools["swa"], pools["comp"][layer]
        self.swa_bt, self.comp_bts = _env_tables(
            self.requests, gen, ratio, page, self.swa.shape[0], self.comp.shape[0]
        )
        idx, active, seq_lens, cu, self.counts, self.req, self.loc = _kvq_rows(
            self.requests, self.floors, gen, self.width, ratio, page, self.swa_bt, self.comp_bts
        )
        self.T = idx.shape[0]
        assert self.T <= bufs.max_t
        self.idx = idx
        self.active = torch.full((max(self.T, ENV_T),), -7, dtype=torch.int32, device=DEV)[: self.T]
        self.active.copy_(active)
        self.kv_lens = torch.full((max(self.B, ENV_B) + 1,), -7, dtype=torch.int32, device=DEV)
        self.kv_lens[: self.B] = seq_lens
        self.cu = torch.full((max(self.B, ENV_B) + 2,), -7, dtype=torch.int32, device=DEV)
        self.cu[: self.B + 1] = cu
        self.q3 = (
            torch.randn(self.T, num_heads, HEAD_DIM, generator=gen, device=DEV)
            .mul_(2)
            .to(torch.bfloat16)
        )
        self.sinks = _sinks(num_heads)
        win = bufs.win_idx[: self.T]
        if ratio:
            comp = torch.where(self.loc >= 0, self.req[:, None] * KVQ_L[ratio] + self.loc, -1).int()
            self.cidx = torch.cat([win, comp], 1)
            assert int(comp.max()) < bufs.comp_c4[ratio].shape[0] * bufs.comp_c4[ratio].shape[2]
        else:
            self.cidx = win

    def fill(self, minus_one=None, skip_request=None):
        """What the two dequantizers write this step (native torch gathers from the container):
        row t * 128 + c of swa_c for every query t < T and window column c -- a -1 column as +0.0,
        or as the constant `minus_one` for a control -- and row b * L_r + e of comp_c for every
        entry e < kv_len // r of every request b (except `skip_request`, for a control). Nothing
        else in the buffers changes."""
        bufs, T = self.bufs, self.T
        w = self.idx[:, :WINDOW].long()
        rows = self.swa.view(-1, HEAD_DIM)[w.clamp_min(0)]
        fillv = torch.zeros((), dtype=torch.bfloat16, device=DEV)
        if minus_one is not None:
            fillv = torch.tensor(minus_one, dtype=torch.bfloat16, device=DEV)
        bufs.swa_c[: T * WINDOW] = torch.where((w < 0)[..., None], fillv, rows).view(
            T * WINDOW, HEAD_DIM
        )
        if self.ratio:
            r, bk, L = self.ratio, SWA_PAGE // self.ratio, KVQ_L[self.ratio]
            dst, src = [], []
            for b, (c, q) in enumerate(self.requests):
                n = (c + q) // r
                if n and b != skip_request:
                    e = torch.arange(n, device=DEV)
                    src.append(self.comp_bts[b][e // bk] * bk + e % bk)
                    dst.append(b * L + e)
            if dst:
                bufs.comp_c[torch.cat(dst)] = self.comp.view(-1, HEAD_DIM)[torch.cat(src)]

    def run(self, ws, compact, idx=None, out=None):
        """The target's two calls: the context call on the first tc rows, then the generation call on
        the views from row tc as [G, next_n, H, 512] with seq_lens = kv_lens[nc:B]."""
        tc, gr, h = self.tc, self.T - self.tc, self.q3.shape[1]
        if compact:
            swa = self.bufs.swa_c4
            comp = self.bufs.comp_c4[self.ratio] if self.ratio else self.bufs.dummy
            idx = self.cidx if idx is None else idx
        else:
            swa, comp = self.swa, self.comp
            idx = self.idx if idx is None else idx
        o = (
            torch.empty(self.T, h, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
            if out is None
            else out
        )
        if tc:
            op(
                torch.split(self.q3, [tc, gr], 0)[0],
                swa,
                comp,
                ws,
                torch.split(idx, [tc, gr], 0)[0],
                torch.split(self.active, [tc, gr], 0)[0],
                self.kv_lens[: self.nc],
                self.sinks,
                SCALE,
                1.0,
                cum_seq_lens_q=self.cu[: self.nc + 1],
                max_q_len=self.max_ctx_q,
                out=torch.split(o, [tc, gr], 0)[0],
            )
        if gr:
            g, nn = self.G, self.next_n
            op(
                torch.split(self.q3, [tc, gr], 0)[1].reshape(g, nn, h, HEAD_DIM),
                swa,
                comp,
                ws,
                torch.split(idx, [tc, gr], 0)[1],
                torch.split(self.active, [tc, gr], 0)[1],
                self.kv_lens[self.nc : self.B],
                self.sinks,
                SCALE,
                1.0,
                out=torch.split(o, [tc, gr], 0)[1].reshape(g, nn, h, HEAD_DIM),
            )
        return o

    def reference(self, **kw):
        return _reference(
            self.q3, self.swa, self.comp, self.idx, self.active, self.counts, self.sinks, **kw
        )

    def attended_minus_one(self) -> int:
        """Window columns inside the window count whose container index is -1 (floored windows)."""
        col = torch.arange(WINDOW, device=DEV)
        return int(((col[None] < self.counts[:, None]) & (self.idx[:, :WINDOW] < 0)).sum())


def _bits(o):
    return o.view(torch.int16)


def _kvq_case(seed):
    """A random step of the engine envelope for one configuration: seeds 0..199 at (max_num_tokens,
    max_batch_size) = (8192, 32), 200..259 at the three smaller ones in turn. k in 1..max_batch_size
    requests, prefill-only (30 %), generation-only (20 %) or mixed; generation at next_n 1 or 6
    (DSpark verify), a quarter of generation requests in the DSpark tail up to position 16397;
    contexts split T at random (log-uniform or uniform), fresh or cached (cached + q_len <= 16384);
    a quarter of cached contexts carry a window floor."""
    rng = random.Random(1_500_000 + seed)
    num_heads = (16, 64)[seed % 2]
    layer = ("r0", "r2", "r1")[(seed // 2) % 3]
    cfg = KVQ_CONFIGS[0] if seed < 200 else KVQ_CONFIGS[1 + (seed - 200) % 3]
    max_t, seqs = cfg
    k = rng.randint(1, seqs)
    next_n = rng.choice((1, 6))
    kind = rng.random()
    n_gen = 0 if kind < 0.3 else (k if kind < 0.5 else rng.randint(1, k))
    n_ctx = k - n_gen
    ctx = []
    if n_ctx:
        budget = max_t - n_gen * next_n
        if rng.random() < 0.5:
            t = int(round(math.exp(rng.uniform(0.0, math.log(budget)))))
        else:
            t = rng.randint(n_ctx, budget)
        t = min(max(t, n_ctx), budget)
        cuts = sorted(rng.sample(range(1, t), n_ctx - 1)) if n_ctx > 1 else []
        q_lens = [b - a for a, b in zip([0] + cuts, cuts + [t])]
        ctx = [(_fresh_or_cached(rng, q), q) for q in q_lens]
    top = KVQ_TAIL + 1 - next_n
    gen = [
        rng.randint(ENV_SEQ - next_n, top) if rng.random() < 0.25 else rng.randint(0, top)
        for _ in range(n_gen)
    ]
    floors = [
        rng.randint(max(1, c - WINDOW + 1), c) if c and rng.random() < 0.25 else 0 for c, _ in ctx
    ]
    return num_heads, layer, cfg, ctx, gen, next_n, floors


KVQ_SEEDS = list(range(260))


@pytest.mark.parametrize("seed", KVQ_SEEDS)
def test_kvq_compact_read_random_partition(env_pools, kvq_buffers, seed):
    """Random steps for the compact read: 260 seeded random steps (200 on the (8192, 32) buffers, 20 on
    each smaller configuration's), each run in both forms: the compact read is bitwise the container
    read, and the container read is inside the gate against the fp32 reference."""
    num_heads, layer, cfg, ctx, gen_pos, next_n, floors = _kvq_case(seed)
    g = torch.Generator(device=DEV).manual_seed(1_600_000 + seed)
    st = _KvqStep(ctx, gen_pos, next_n, num_heads, layer, env_pools, kvq_buffers[cfg], g, floors)
    st.fill()
    ws = _workspace()
    o_pool = st.run(ws, compact=False)
    o_comp = st.run(ws, compact=True)
    e = _excess(o_pool, st.reference(), env_pools["atol"][layer])
    same = torch.equal(_bits(o_comp), _bits(o_pool))
    print(
        f"kvq partition {seed}: cfg {cfg} H{num_heads} {layer} T {st.T} tc {st.tc} nc {st.nc} G {st.G}x{next_n} "
        f"attended -1 {st.attended_minus_one()} excess {e:.3f} compact==container {same}"
    )
    assert same, (seed, int((_bits(o_comp) != _bits(o_pool)).sum()))
    assert e <= 1.0, (seed, e)


def test_kvq_compact_read_random_partitions_reach_the_envelope():
    """The 260 steps span the envelope: every (H, layer), every configuration, k from 1 to >= 30 and
    T to >= 8000 on (8192, 32), prefill-only, decode-only and verify-only, mixed at next_n 1 and 6,
    generation in the DSpark tail, and floored windows."""
    cases = [_kvq_case(s) for s in KVQ_SEEDS]
    big = [c for c in cases if c[2] == KVQ_CONFIGS[0]]
    ks = [len(c[3]) + len(c[4]) for c in big]
    ts = [sum(q for _, q in c[3]) + len(c[4]) * c[5] for c in big]
    kinds = {(bool(c[3]), bool(c[4]), c[5] if c[4] else 0) for c in big}
    assert len(big) == 200 and min(ks) == 1 and max(ks) >= 30, (len(big), min(ks), max(ks))
    assert min(ts) <= 4 and max(ts) >= 8000, (min(ts), max(ts))
    assert {
        (True, False, 0),
        (False, True, 1),
        (False, True, 6),
        (True, True, 1),
        (True, True, 6),
    } <= kinds, kinds
    assert sum(1 for c in cases for p in c[4] if p + c[5] - 1 >= ENV_SEQ) >= 50
    assert sum(1 for c in cases for f in c[6] if f) >= 50
    assert {c[2] for c in cases} == set(KVQ_CONFIGS)
    assert all(
        sum(q for _, q in c[3]) + len(c[4]) * c[5] <= c[2][0] and len(c[3]) + len(c[4]) <= c[2][1]
        for c in cases
    )
    assert {(c[0], c[1]) for c in cases} == {(h, layer) for h in (16, 64) for layer in ENV_LAYERS}


# ---------------------------------------------------------------------------------------------
# Floored windows passed with floor-adjusted seq_lens (seq_lens = kv_len - floor)
# ---------------------------------------------------------------------------------------------
def _floored_steps():
    """The random steps of _kvq_case that hold a cached context, re-floored: EVERY cached context
    (cached c > 0) gets a window floor, half of them at c itself (an encoder-replay chunk: its window
    starts at its own first row) and half at a random point of [max(1, c - 127), c - 1] (a decoder
    replay tail, whose first row can lie above its floor), fresh contexts none. The first 40 such
    seeds on the (8192, 32) buffers and the first 8 on the three smaller configurations."""
    out = []
    for lo, hi, n in ((0, 200, 40), (200, len(KVQ_SEEDS), 8)):
        k = 0
        for seed in KVQ_SEEDS[lo:hi]:
            num_heads, layer, cfg, ctx, gen, next_n, _ = _kvq_case(seed)
            if not any(c for c, _ in ctx):
                continue
            rng = random.Random(1_800_000 + seed)
            floors = [
                0
                if not c
                else (
                    c
                    if rng.random() < 0.5
                    else rng.randint(max(1, c - WINDOW + 1), c - 1)
                    if c > 1
                    else c
                )
                for c, _ in ctx
            ]
            out.append((seed, (num_heads, layer, cfg, ctx, gen, next_n, floors)))
            k += 1
            if k == n:
                break
    return out


KVQ_FLOOR_CASES = _floored_steps()


def _floor_adjusted(st):
    """Set the step's context seq_lens to kv_len - floor and return the per-row window counts that
    gives, min(128, p - floor + 1), with each row's position p and floor."""
    pos_l, flo_l = [], []
    for b, ((c, q), f) in enumerate(zip(st.requests, st.floors)):
        pos_l.append(torch.arange(c, c + q, device=DEV))
        flo_l.append(torch.full((q,), f, dtype=torch.int64, device=DEV))
        if b < st.nc:
            st.kv_lens[b] = c + q - f
    pos, flo = torch.cat(pos_l), torch.cat(flo_l)
    return (pos - flo + 1).clamp(max=WINDOW), pos, flo


@pytest.mark.parametrize("seed", [s for s, _ in KVQ_FLOOR_CASES])
def test_kvq_floored_windows_with_floor_adjusted_seq_lens(env_pools, kvq_buffers, seed):
    """A context request whose window is floored (the window of position p is max(floor, p - 127)
    .. p, then -1) passed with seq_lens = kv_len - floor. To the kernel that is a context with
    kv_len - floor - q_len cached rows (none for an encoder-replay chunk): its window count
    min(128, p - floor + 1) is exactly the floored window, so no -1 is attended and the result is
    the MASKED reading of the floored window (the zero-row reading of seq_lens = kv_len is the
    other one). In both read forms against the fp32 reference, compact == container bitwise; and
    every row whose window reaches no position below its floor -- fresh contexts, the generation
    rows, a floored request's rows at p - 127 >= floor -- is bitwise the same step called with
    seq_lens = kv_len."""
    _floored_step(env_pools, kvq_buffers, seed)


@pytest.mark.parametrize("seed", [s for s, _ in KVQ_FLOOR_CASES])
def test_kvq_floored_windows_at_the_members_head_counts(env_pools, kvq_buffers, seed):
    """The same 48 floored steps with the members' head counts in place of the case's (H 16 -> 8,
    H 64 -> 32), the same checks."""
    _floored_step(env_pools, kvq_buffers, seed, {16: 8, 64: 32})


def _floored_step(env_pools, kvq_buffers, seed, heads_map=None):
    num_heads, layer, cfg, ctx, gen_pos, next_n, floors = dict(KVQ_FLOOR_CASES)[seed]
    if heads_map is not None:
        num_heads = heads_map[num_heads]
    g = torch.Generator(device=DEV).manual_seed(1_900_000 + seed)
    st = _KvqStep(ctx, gen_pos, next_n, num_heads, layer, env_pools, kvq_buffers[cfg], g, floors)
    st.fill()
    ws = _workspace()
    o_kv = st.run(ws, compact=True)
    old_counts = st.counts
    counts, pos, flo = _floor_adjusted(st)
    col = torch.arange(WINDOW, device=DEV)
    attended_neg = int(((col[None] < counts[:, None]) & (st.idx[:, :WINDOW] < 0)).sum())
    valid = (st.idx[:, :WINDOW] >= 0).sum(1)
    assert attended_neg == 0 and torch.equal(counts, valid) and int(counts.min()) >= 1
    assert all(int(st.kv_lens[b]) >= q for b, (_, q) in enumerate(st.requests[: st.nc]))
    o_pool = st.run(ws, compact=False)
    o_comp = st.run(ws, compact=True)
    ref = _reference(st.q3, st.swa, st.comp, st.idx, st.active, counts, st.sinks)
    masked = _reference(
        st.q3, st.swa, st.comp, st.idx, st.active, old_counts, st.sinks, negative_index="masked"
    )
    assert torch.equal(ref, masked), "the floor-adjusted count is not the masked floored window"
    e = _excess(o_pool, ref, env_pools["atol"][layer])
    same = torch.equal(_bits(o_comp), _bits(o_pool))
    below = (pos - (WINDOW - 1) < flo) & (flo > 0)  # rows whose window the floor truncates
    keep = ~below
    unchanged = torch.equal(_bits(o_comp)[keep], _bits(o_kv)[keep])
    print(
        f"floored {seed}: cfg {cfg} H{num_heads} {layer} T {st.T} tc {st.tc} nc {st.nc} G {st.G}x{next_n} floors "
        f"{sum(1 for f in floors if f)} (at the first row {sum(1 for (c, _), f in zip(ctx, floors) if f and f == c)}) "
        f"truncated rows {int(below.sum())} excess {e:.3f} compact==container {same} other rows unchanged {unchanged}"
    )
    assert same, (seed, int((_bits(o_comp) != _bits(o_pool)).sum()))
    assert e <= 1.0, (seed, e)
    assert unchanged, (seed, int((_bits(o_comp)[keep] != _bits(o_kv)[keep]).sum()))


def test_kvq_floored_windows_reach_the_forms():
    """The 48 steps cover both floor forms (at the first row, and below it), H 16 / 64 x every layer,
    every buffer configuration, mixed steps at next_n 1 and 6, and a floored request whose first row
    sits >= 127 past its floor or at it with >= 127 cached rows below (the first row then counts 127
    columns fewer)."""
    cases = [c for _, c in KVQ_FLOOR_CASES]
    assert len(cases) == 48
    at_first = sum(1 for c in cases for (cc, _), f in zip(c[3], c[6]) if f and f == cc)
    below = sum(1 for c in cases for (cc, _), f in zip(c[3], c[6]) if f and f < cc)
    assert at_first >= 20 and below >= 20, (at_first, below)
    assert {(c[0], c[1]) for c in cases} == {(h, layer) for h in (16, 64) for layer in ENV_LAYERS}
    assert {c[5] for c in cases if c[4]} == {1, 6}
    assert {c[2] for c in cases} == set(KVQ_CONFIGS)
    assert (
        sum(
            1 for c in cases for (cc, _), f in zip(c[3], c[6]) if f and f == cc and cc >= WINDOW - 1
        )
        >= 10
    )


def test_kvq_floored_windows_kv_len_reading_fails_the_gate(env_pools, kvq_buffers):
    """The harness sees the difference the floor-adjusted seq_lens makes. On two encoder-replay chunks
    (floor = the first row, 127 and 3970, 129 rows each) beside a fresh context and generation rows,
    at H 16 / 64 and every layer: seq_lens = kv_len - floor is inside the gate against the masked
    reference, while seq_lens = kv_len (a chunk's first rows attend up to 127 below-floor -1 columns
    as zero rows) lands outside it on those rows; both calls agree bitwise on every other row. The
    chunk at 127 has few compressed columns (64 / 128 visible at ratio 2 / 1), so the zero rows'
    weight shows at every layer; at 3970 the 512 compressed columns dilute it (about the gate at
    ratio 2 / 1, printed)."""
    ws = _workspace()
    for num_heads in (16, 64):
        for layer in ENV_LAYERS:
            g = torch.Generator(device=DEV).manual_seed(2_100_000 + num_heads + 7 * len(layer))
            ctx, floors = [(127, 129), (3970, 129), (0, 300)], [127, 3970, 0]
            st = _KvqStep(
                ctx,
                [5000, 700],
                6,
                num_heads,
                layer,
                env_pools,
                kvq_buffers[KVQ_CONFIGS[0]],
                g,
                floors,
            )
            st.fill()
            atol = env_pools["atol"][layer]
            o_kv = st.run(ws, compact=True)
            counts, pos, flo = _floor_adjusted(st)
            o_fix = st.run(ws, compact=True)
            ref = _reference(st.q3, st.swa, st.comp, st.idx, st.active, counts, st.sinks)
            hit = (pos - (WINDOW - 1) < flo) & (flo > 0)
            wrong = {
                f: _excess(o_kv[hit & (flo == f)], ref[hit & (flo == f)], atol) for f in (127, 3970)
            }
            right = _excess(o_fix, ref, atol)
            print(
                f"floored control H{num_heads} {layer}: seq_lens = kv_len - floor {right:.3f} of the gate, "
                f"seq_lens = kv_len {wrong[127]:.1f} (floor 127) / {wrong[3970]:.1f} (floor 3970) on the "
                f"{int(hit.sum())} truncated rows"
            )
            assert right <= 1.0 and wrong[127] > 10.0 * max(right, 0.1), (
                num_heads,
                layer,
                right,
                wrong,
            )
            assert torch.equal(_bits(o_kv)[~hit], _bits(o_fix)[~hit])


@pytest.mark.parametrize("width", [128, 640])
@pytest.mark.parametrize("num_heads", [16, 64])
def test_kvq_compact_read_kernel_variant_bands(env_pools, kvq_buffers, num_heads, width):
    """Kernel bands for the compact read. The context call's kernel is picked from (H, width, number of
    requests, max_q_len), and the compact read changes none of those. Every band of VARLEN_BANDS at
    its bottom, one interior point and its top at every request count 1..32 (_band_cases) runs as a
    context step on the (8192, 32) compact buffers: the profiler reads back exactly the band's
    kernels for every compact call, and every compact call is bitwise the container read on the
    same values."""
    rng = random.Random(num_heads * 1000 + width + 17)
    gen = torch.Generator(device=DEV).manual_seed(num_heads * 1000 + width + 17)
    ws = _workspace()
    bufs = kvq_buffers[KVQ_CONFIGS[0]]
    cases = _band_cases(num_heads, width, rng)
    seen, differ = [], []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for i, (ctx, vid) in enumerate(cases):
            layer = "r0" if width == WINDOW else ("r2", "r1")[i % 2]
            st = _KvqStep(ctx, [], 1, num_heads, layer, env_pools, bufs, gen)
            st.fill()
            o_pool = st.run(ws, compact=False)
            torch.cuda.synchronize()
            torch.cuda._sleep(10)
            o = st.run(ws, compact=True)
            torch.cuda._sleep(10)
            if not torch.equal(_bits(o), _bits(o_pool)):
                differ.append((ctx, vid))
        torch.cuda.synchronize()
    ev = sorted(
        (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda e: e.time_range.start,
    )
    open_call, cur = False, []
    for e in ev:
        if "spin_kernel" in e.name or "sleep" in e.name:
            if open_call:
                seen.append(_variant_of(cur))
            open_call, cur = not open_call, []
        elif open_call:
            cur.append(e.name)
    assert len(seen) == len(cases), (len(seen), len(cases))
    mism = [(c, VARLEN_KERNELS[v], s) for (c, v), s in zip(cases, seen) if VARLEN_KERNELS[v] != s]
    used = {v for _, v in cases}
    print(
        f"kvq bands H{num_heads} W{width}: {len(cases)} compact calls, {len(used)} variants, "
        f"kernel mismatches {len(mism)}, not bitwise the container read {len(differ)}"
    )
    assert not mism, mism[:3]
    assert not differ, differ[:3]


@pytest.mark.parametrize("width", [128, 640])
@pytest.mark.parametrize("num_heads", [16, 64])
def test_kvq_compact_read_every_dumped_cell(env_pools, kvq_buffers, num_heads, width):
    """The dumped partitions, for the compact read: every context call (tc, H, nc, max_q_len)
    of DUMPED_CTX_CELLS as a step of nc requests (_dumped_partition) on the (8192, 32) compact
    buffers, at width 128 (ratio 0) and 640 (ratio 2 and ratio 1 on alternate cells): each compact
    read is bitwise the container read on the same values, which is inside the gate."""
    rng = random.Random(78000 + num_heads + width)
    gen = torch.Generator(device=DEV).manual_seed(78000 + num_heads + width)
    ws = _workspace()
    bufs = kvq_buffers[KVQ_CONFIGS[0]]
    cells = _dumped_cells(num_heads)
    worst, bad, differ = 0.0, [], []
    for i, (tc, nc, mq) in enumerate(cells):
        layer = "r0" if width == WINDOW else ("r2", "r1")[i % 2]
        ctx = _dumped_partition(tc, nc, mq, rng)
        st = _KvqStep(ctx, [], 1, num_heads, layer, env_pools, bufs, gen)
        st.fill()
        o_pool = st.run(ws, compact=False)
        o_comp = st.run(ws, compact=True)
        if not torch.equal(_bits(o_comp), _bits(o_pool)):
            differ.append((tc, nc, mq, layer))
        e = _excess(o_pool, st.reference(), env_pools["atol"][layer])
        worst = max(worst, e)
        if e > 1.0:
            bad.append(((tc, nc, mq), layer, e))
    print(
        f"kvq dumped cells H{num_heads} W{width}: {len(cells)} steps, not bitwise the container read "
        f"{len(differ)}, max excess {worst:.3f}"
    )
    assert not differ, differ[:5]
    assert not bad, bad[:5]


def _kvq_far_steps(max_t, seqs):
    """Steps that reach both buffers' far ends: T = max_num_tokens with max_batch_size requests, the
    last a DSpark verify ending at 16397 (the window half names swa_c's last row, the compressed half
    row (seqs - 1) * L_r + 16398 // r - 1), without and with window floors on every cached context;
    and a verify-only step of max_batch_size requests at the tail."""
    n_ctx = seqs - 1
    rest = max_t - 6 - 3 * (n_ctx - 1)
    ctx = [(ENV_SEQ - rest, rest)] + [(131 + 97 * i, 3) for i in range(n_ctx - 1)]
    tail = [KVQ_TAIL - 5]
    floors = [c - 40 for c, _ in ctx]
    verify = [KVQ_TAIL - 5 - (i % 3) for i in range(seqs)]
    return [(ctx, tail, 6, None), (ctx, tail, 6, floors), ([], verify, 6, None)]


@pytest.mark.parametrize("cfg", KVQ_CONFIGS, ids=[f"{t}x{s}" for t, s in KVQ_CONFIGS])
def test_kvq_compact_read_far_end_of_every_buffer_size(env_pools, kvq_buffers, cfg):
    """At each buffer size the target allocates, the steps of _kvq_far_steps at H 16 and 64 and
    every layer kind: the window half names the window buffer's last row, the compressed half the
    largest row the compressed buffer can hold for this configuration (asserted); the compact read is
    bitwise the container read (the floored steps with attended -1s among them), inside the gate."""
    bufs = kvq_buffers[cfg]
    max_t, seqs = cfg
    ws = _workspace()
    worst, n = 0.0, 0
    for i, (ctx, gen_pos, next_n, floors) in enumerate(_kvq_far_steps(max_t, seqs)):
        for num_heads in (16, 64):
            for layer in ENV_LAYERS:
                g = torch.Generator(device=DEV).manual_seed(
                    1_700_000 + 100 * i + num_heads + ENV_LAYERS[layer][0]
                )
                st = _KvqStep(ctx, gen_pos, next_n, num_heads, layer, env_pools, bufs, g, floors)
                if i < 2:
                    assert (
                        st.T == max_t
                        and st.B == seqs
                        and int(st.cidx[:, :WINDOW].max()) == max_t * WINDOW - 1
                    )
                    if st.ratio:
                        far = (seqs - 1) * KVQ_L[st.ratio] + (KVQ_TAIL + 1) // st.ratio - 1
                        assert int(st.cidx[:, WINDOW:].max()) == far
                if floors is not None:
                    assert st.attended_minus_one() > 0
                st.fill()
                o_pool = st.run(ws, compact=False)
                o_comp = st.run(ws, compact=True)
                assert torch.equal(_bits(o_comp), _bits(o_pool)), (cfg, i, num_heads, layer)
                e = _excess(o_pool, st.reference(), env_pools["atol"][layer])
                assert e <= 1.0, (cfg, i, num_heads, layer, e)
                worst, n = max(worst, e), n + 1
    print(f"kvq far end {cfg}: {n} steps, compact == container bitwise, max excess {worst:.3f}")


@pytest.mark.parametrize("layer", list(ENV_LAYERS))
@pytest.mark.parametrize("num_heads", [16, 64])
def test_kvq_compact_read_cuda_graph_capture_set(env_pools, kvq_buffers, num_heads, layer):
    """The capture set on the compact buffers: the target's generation call (dense [G, next_n, H, 512],
    seq_lens = kv_lens[0:G], index / lengths / seq_lens / q prefixes of step buffers, the output
    allocated inside the capture) at G in 1..8, 16, 24, 32 x next_n 1 and 6 on the (8192, 32)
    buffers and G 1..8 x next_n 1 and 6 on the (2048, 8) buffers, each captured after one eager
    call at that shape. The first replay equals the eager call bitwise; three more on new steps (new
    compact rows, q, compressed half, lengths, seq_lens; request 0 in the DSpark tail) each equal a
    fresh eager compact call AND the container read on the same values bitwise, and match the
    reference."""
    ratio = ENV_LAYERS[layer][0]
    ws = _workspace()
    sinks = _sinks(num_heads)
    n_graphs, worst = 0, 0.0
    for cfg, batches in ((KVQ_CONFIGS[0], GRAPH_BATCHES), (KVQ_CONFIGS[3], GRAPH_BATCHES[:8])):
        bufs = kvq_buffers[cfg]
        max_t, seqs = cfg
        width = WINDOW + (INDEX_TOPK if ratio else 0)
        idx_buf = torch.full((max_t, width), -1, dtype=torch.int32, device=DEV)
        lens_buf = torch.full((max_t,), WINDOW, dtype=torch.int32, device=DEV)
        kvl_buf = torch.full((seqs + 1,), 1, dtype=torch.int32, device=DEV)
        q_buf = torch.zeros(max_t, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        comp = bufs.comp_c4[ratio] if ratio else bufs.dummy
        for next_n in (1, 6):
            for G in batches:
                T = G * next_n
                gen = torch.Generator(device=DEV).manual_seed(
                    1_800_000 + 1000 * G + 100 * next_n + num_heads + ratio
                )
                rng = random.Random(1_900_000 + 1000 * G + 100 * next_n + num_heads + ratio)
                idx = bufs.win_idx[:T] if not ratio else idx_buf[:T]

                def make(step):
                    pos = [rng.randint(0, KVQ_TAIL + 1 - next_n) for _ in range(G)]
                    pos[0] = (KVQ_TAIL if step % 2 == 0 else ENV_SEQ - 1) + 1 - next_n
                    st = _KvqStep([], pos, next_n, num_heads, layer, env_pools, bufs, gen)
                    st.fill()
                    q_buf[:T].copy_(st.q3)
                    if ratio:
                        idx_buf[:T].copy_(st.cidx)
                    lens_buf[:T].copy_(st.active)
                    kvl_buf[:G].copy_(st.kv_lens[:G])
                    return st

                def call(out):
                    return op(
                        q_buf[:T].view(G, next_n, num_heads, HEAD_DIM),
                        bufs.swa_c4,
                        comp,
                        ws,
                        idx,
                        lens_buf[:T],
                        kvl_buf[:G],
                        sinks,
                        SCALE,
                        1.0,
                        out=out,
                    )

                shape = (G, next_n, num_heads, HEAD_DIM)
                st = make(0)
                eager = call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
                assert torch.equal(
                    _bits(eager.view(T, num_heads, HEAD_DIM)), _bits(st.run(ws, compact=False))
                )
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
                torch.cuda.current_stream().wait_stream(s)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    o = torch.empty(T, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
                    out = call(o.view(shape))
                out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(_bits(out), _bits(eager)), (cfg, G, next_n)
                prev = eager.clone()
                for it in range(1, 4):
                    st = make(it)
                    out.fill_(float("nan"))
                    graph.replay()
                    fresh = call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
                    pool = st.run(ws, compact=False)
                    torch.cuda.synchronize()
                    assert torch.equal(_bits(out), _bits(fresh)), (
                        cfg,
                        G,
                        next_n,
                        it,
                        "replay != eager",
                    )
                    assert torch.equal(_bits(out.view(T, num_heads, HEAD_DIM)), _bits(pool)), (
                        cfg,
                        G,
                        next_n,
                        it,
                    )
                    assert not torch.equal(_bits(out), _bits(prev)), (cfg, G, next_n, it)
                    e = _excess(pool, st.reference(), env_pools["atol"][layer])
                    assert e <= 1.0, (cfg, G, next_n, it, e)
                    worst = max(worst, e)
                    prev = out.clone()
                del graph
                n_graphs += 1
    print(
        f"kvq capture H{num_heads} {layer}: {n_graphs} graphs x 4 replays, replay = eager = container, "
        f"max excess {worst:.3f}"
    )


def _kvq_control_steps():
    """(H, layer, cfg, ctx, gen, next_n, floors): mixed steps whose cached contexts carry window
    floors (attended -1s), a fresh short context (masked -1 window tail), and DSpark-tail
    generation rows."""
    ctx = [(300, 200), (5000, 64), (0, 50), (16000, 300), (129, 1)]
    floors = [250, 4990, 0, 15950, 60]
    return [
        (16, "r2", KVQ_CONFIGS[0], ctx, [16392, 77, 4000], 6, floors),
        (64, "r1", KVQ_CONFIGS[0], ctx, [16397, 128, 9000], 1, floors),
        (16, "r0", KVQ_CONFIGS[3], ctx, [16392, 3], 6, floors),
        (64, "r2", KVQ_CONFIGS[2], ctx, [16397, 500, 12000, 7], 1, floors),
    ]


def test_kvq_compact_read_wrong_backing_and_numbering_fail_the_gate(env_pools, kvq_buffers):
    """The harness sees each way the compact read can go wrong. On mixed steps with floored windows
    (attended -1s) the correct compact read is bitwise the container read and inside the gate, while
    the reference lands far outside it for:
    - '-1 not zero-backed': the rows standing for -1 window columns hold 3.0 instead of +0.0;
    - the window half renumbered one row late (t * 128 + c + 1);
    - the compressed half renumbered one entry late (b * L_r + e + 1), or at the other ratio's row
      stride (b * L_{3-r} + e);
    - one request's compressed rows not rewritten this step (stale).
    On the same steps without floors (only masked -1s), the 3.0 backing leaves the output bitwise
    unchanged (masked columns carry no weight), while NaN backing poisons it (masked columns are
    loaded): the backing must be finite everywhere and zero where a -1 is attended."""
    ws = _workspace()
    for num_heads, layer, cfg, ctx, gen_pos, next_n, floors in _kvq_control_steps():
        bufs = kvq_buffers[cfg]
        ratio = ENV_LAYERS[layer][0]
        atol = env_pools["atol"][layer]
        for floored in (True, False):
            g = torch.Generator(device=DEV).manual_seed(2_000_000 + num_heads + 10 * ratio)
            st = _KvqStep(
                ctx,
                gen_pos,
                next_n,
                num_heads,
                layer,
                env_pools,
                bufs,
                g,
                floors if floored else None,
            )
            ref = st.reference()
            st.fill()
            o_pool = st.run(ws, compact=False)
            o_comp = st.run(ws, compact=True)
            assert torch.equal(_bits(o_comp), _bits(o_pool))
            correct = _excess(o_comp, ref, atol)
            assert correct <= 1.0, correct
            neg = st.attended_minus_one()
            st.fill(minus_one=3.0)
            backed = st.run(ws, compact=True)
            st.fill(minus_one=float("nan"))
            nan_backed = st.run(ws, compact=True)
            st.fill()
            if not floored:
                assert neg == 0
                assert torch.equal(_bits(backed), _bits(o_pool)), (
                    "a finite masked row changed the output"
                )
                assert not torch.isfinite(nan_backed).all(), (
                    "a NaN masked row did not poison the output"
                )
                print(
                    f"kvq controls H{num_heads} {layer} {cfg} unfloored: 3.0 backing bitwise unchanged, NaN backing "
                    f"poisons {int((~torch.isfinite(nan_backed)).any(-1).any(-1).sum())} of {st.T} rows"
                )
                continue
            assert neg > 0
            wrong = {"minus_one_not_zero_backed": _excess(backed, ref, atol)}
            w1 = st.cidx.clone()
            w1[:, :WINDOW] += 1
            wrong["window_renumbered_one_row_late"] = _excess(
                st.run(ws, compact=True, idx=w1), ref, atol
            )
            if ratio:
                c = st.cidx[:, WINDOW:]
                w2 = st.cidx.clone()
                w2[:, WINDOW:] = torch.where(c >= 0, c + 1, c)
                wrong["compressed_renumbered_one_entry_late"] = _excess(
                    st.run(ws, compact=True, idx=w2), ref, atol
                )
                w3 = st.cidx.clone()
                w3[:, WINDOW:] = torch.where(
                    st.loc >= 0, st.req[:, None] * KVQ_L[3 - ratio] + st.loc, -1
                ).int()
                assert int(w3.max()) < bufs.comp_c4[ratio].shape[0] * bufs.comp_c4[ratio].shape[2]
                wrong["other_ratio_row_stride"] = _excess(
                    st.run(ws, compact=True, idx=w3), ref, atol
                )
                L = KVQ_L[ratio]
                bufs.comp_c[L : 2 * L] = torch.randn(
                    L, HEAD_DIM, generator=g, device=DEV, dtype=torch.bfloat16
                )
                st.fill(
                    skip_request=1
                )  # request 1's rows keep the content above (an earlier step's)
                wrong["request_1_rows_stale"] = _excess(st.run(ws, compact=True), ref, atol)
                st.fill()
            print(
                f"kvq controls H{num_heads} {layer} {cfg}: attended -1 {neg}, correct {correct:.3f}, wrong "
                f"{ {k: round(v, 1) for k, v in wrong.items()} }"
            )
            for name, e in wrong.items():
                assert e > 10.0 * max(correct, 0.1), (
                    f"{name} not discriminated: {e:.2f} vs {correct:.2f}"
                )


def test_kvq_compact_buffers_ending_a_mapping_never_fault():
    """Precondition 8 at the compact buffers' exact sizes. Each view the target hands over is placed
    so that its last row ends a mapping (reserved, unmapped VA after it): the window buffer
    [2048 | 8192, 1, 128, 512] with its last 128 rows attended (the last query of a full step; the
    compact window half holds no -1); the compressed views [8 | 32 x 132, 1, 64 | 128, 512] with
    their last rows attended and a -1 loaded from them (lengths 129 / 131 / 133 / 639). Nothing
    faults, and every output is bitwise the output on a normal allocation. The smallest of these is
    67,584 rows: a loaded -1's over-read (a row in 96..127 from the first row) lands inside it."""
    trials = []
    for max_t, h in ((2048, 16), (8192, 64)):
        rows = max_t * WINDOW
        trials.append(
            dict(
                name=f"kvq-swa-{max_t}-H{h}",
                kind="engine",
                pool="swa",
                rows=rows,
                page=WINDOW,
                lens=WINDOW + INDEX_TOPK,
                heads=h,
                window=WINDOW,
                mapped=rows,
                seed=2_100_000 + max_t,
            )
        )
    for (seqs, r), lens, h in (
        ((8, 2), 129, 16),
        ((32, 2), 131, 64),
        ((8, 1), 133, 64),
        ((32, 1), 639, 16),
    ):
        page = 128 // r
        rows = seqs * KVQ_COLS * page
        assert rows >= 67_584
        trials.append(
            dict(
                name=f"kvq-comp-r{r}-seqs{seqs}-lens{lens}-H{h}",
                kind="engine",
                pool="comp",
                rows=rows,
                page=page,
                lens=lens,
                heads=h,
                window=0,
                mapped=rows,
                seed=2_100_000 + lens,
            )
        )
    p, ok, tried = _guard_run(trials)
    names = [c["name"] for c in trials]
    assert p.returncode == 0, f"child failed after {tried[-1:]}: {p.stderr[-2000:]}"
    assert list(ok) == names, (list(ok), names)
    assert all(ok.values()), [k for k, v in ok.items() if not v]
    print(
        f"kvq guard page: {len(names)} compact-buffer trials completed, each bitwise = the normal allocation"
    )


# ---------------------------------------------------------------------------------------------
# The native-storage compact read over max_batch_size
# ---------------------------------------------------------------------------------------------
# The compressed read buffer is sized by the engine's max_batch_size s, and the window buffer by its
# max_num_tokens only. Measured in the dequantizer entries' tests (every max_num_tokens 2048 / 8192 x
# s 1..32 x DSpark off / on, max_seq_len 16384 / 16398): the metadata's compress_block_tables[r] are
# [s, 132] and swa_local_indices [max_num_tokens, 128], and the CUDA-graph metadata copy keeps both.
# So the target hands this call
#   window     [max_num_tokens, 1, 128, 512]      (262,144 / 1,048,576 rows, whatever s is)
#   compressed [s * 132, 1, 128 / r, 512]         (16,896 * s / r rows; 8,448 / 16,896 at s = 1)
# with entry e of request b at row b * L_r + e, L_r = 132 * 128 / r. The largest compressed row a step
# can name is (s - 1) * L_r + 16398 // r - 1. The engine captures every decode batch G in 1..s
# (padding off; padding on captures a subset plus s). The tests below run the compact read at every
# s in 1..32, bitwise against the container read on the same values. The witness engine's
# configuration, max_num_tokens 2048 with max_batch_size 1, runs as its own configuration.
KVQ_MBS = tuple(range(1, 33))
KVQ_MBS_LISTED = (1, 2, 3, 4, 8, 16, 32)
KVQ_WITNESS = (2048, 1)
# (max_num_tokens, max_batch_size) -> random steps: 200 at every s (8192 tokens) and at the witness.
KVQ_MBS_STEPS = {(ENV_T, s): 200 for s in KVQ_MBS}
KVQ_MBS_STEPS[KVQ_WITNESS] = 200
KVQ_MBS_CAPTURE = [(ENV_T, s) for s in KVQ_MBS] + [KVQ_WITNESS]
KVQ_MBS_SMALLEST_ROWS = KVQ_COLS * 64  # the ratio-2 view at s = 1


class _KvqAxisBuffers:
    """_KvqBuffers for one (max_num_tokens, s) whose window buffer is shared by every s at that
    max_num_tokens (the window buffer does not depend on s). The compressed buffer is exactly
    s * 132 * 128 rows, its ratio views exactly [s * 132, 1, 128 / r, 512]."""

    def __init__(self, swa_c, seqs, gen):
        self.max_t, self.seqs = swa_c.shape[0] // WINDOW, seqs
        self.swa_c = swa_c
        self.comp_c = torch.randn(
            seqs * KVQ_COLS * 128, HEAD_DIM, generator=gen, device=DEV, dtype=torch.bfloat16
        )
        self.swa_c4 = self.swa_c.view(self.max_t, 1, WINDOW, HEAD_DIM)
        self.comp_c4 = {
            r: self.comp_c[: seqs * KVQ_COLS * (128 // r)].view(
                seqs * KVQ_COLS, 1, 128 // r, HEAD_DIM
            )
            for r in (1, 2)
        }
        self.win_idx = torch.arange(self.max_t * WINDOW, dtype=torch.int32, device=DEV).view(
            self.max_t, WINDOW
        )
        self.dummy = torch.zeros(1, 1, 64, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        for r, v in self.comp_c4.items():
            assert (
                v.data_ptr() == self.comp_c.data_ptr()
                and v.shape[0] * v.shape[2] == 16_896 * seqs // r
            )


@pytest.fixture(scope="module")
def kvq_axis_buffers():
    """Lazily built _KvqAxisBuffers per (max_num_tokens, s): about 1.3 GB of window buffers and 8.7 GB
    of compressed buffers once every s has been used."""
    gen = torch.Generator(device=DEV).manual_seed(3_400_000)
    swa = {}
    bufs = {}

    def get(cfg):
        if cfg not in bufs:
            max_t, seqs = cfg
            if max_t not in swa:
                swa[max_t] = torch.randn(
                    max_t * WINDOW, HEAD_DIM, generator=gen, device=DEV, dtype=torch.bfloat16
                )
            bufs[cfg] = _KvqAxisBuffers(swa[max_t], seqs, gen)
        return bufs[cfg]

    yield get
    bufs.clear()
    swa.clear()
    torch.cuda.empty_cache()


def _kvq_axis_case(cfg, i):
    """Step i of configuration cfg = (max_num_tokens, s): _kvq_case's distribution with k <= s. Step 0
    is a full batch (k = s; mixed when s >= 2), step 1 a single request, step 2 a verify-only full batch
    in the DSpark tail; the rest draw k from 1..s. H and the layer cycle with i."""
    max_t, seqs = cfg
    rng = random.Random(3_500_000 + 100_000 * (max_t // 2048) + 1000 * seqs + i)
    num_heads = (16, 64)[i % 2]
    layer = ("r0", "r2", "r1")[(i // 2) % 3]
    next_n = rng.choice((1, 6))
    if i == 0:
        k = seqs
        n_gen = rng.randint(1, k - 1) if k >= 2 else 0
    elif i == 1:
        k = 1
        n_gen = rng.randint(0, 1)
    elif i == 2:
        k, n_gen, next_n = seqs, seqs, 6
    else:
        k = rng.randint(1, seqs)
        kind = rng.random()
        n_gen = 0 if kind < 0.3 else (k if kind < 0.5 else rng.randint(1, k))
    n_ctx = k - n_gen
    ctx = []
    if n_ctx:
        budget = max_t - n_gen * next_n
        if i == 0 or rng.random() < 0.5:
            t = budget if i == 0 else int(round(math.exp(rng.uniform(0.0, math.log(budget)))))
        else:
            t = rng.randint(n_ctx, budget)
        t = min(max(t, n_ctx), budget)
        cuts = sorted(rng.sample(range(1, t), n_ctx - 1)) if n_ctx > 1 else []
        q_lens = [b - a for a, b in zip([0] + cuts, cuts + [t])]
        ctx = [(_fresh_or_cached(rng, q), q) for q in q_lens]
    top = KVQ_TAIL + 1 - next_n
    if i == 2:  # every request ends in 16384..16397
        gen = [rng.randint(ENV_SEQ - next_n + 1, top) for _ in range(n_gen)]
    else:
        gen = [
            rng.randint(ENV_SEQ - next_n, top) if rng.random() < 0.25 else rng.randint(0, top)
            for _ in range(n_gen)
        ]
    floors = [
        rng.randint(max(1, c - WINDOW + 1), c) if c and rng.random() < 0.25 else 0 for c, _ in ctx
    ]
    return num_heads, layer, ctx, gen, next_n, floors


def _kvq_axis_ids(cfgs):
    return [f"{t}x{s}" for t, s in cfgs]


@pytest.mark.parametrize("cfg", list(KVQ_MBS_STEPS), ids=_kvq_axis_ids(KVQ_MBS_STEPS))
def test_kvq_mbs_compact_read_random_partitions(env_pools, kvq_axis_buffers, cfg):
    """Random steps on the compact buffers of every max_batch_size s in 1..32 (max_num_tokens 8192) and of
    the witness configuration (2048, 1): 200 seeded steps at each (_kvq_axis_case: k <= s requests,
    T <= max_num_tokens, prefill, decode, verify and mixed, DSpark tail, floored windows, H 16 / 64 x
    every layer kind). Every compact
    read is bitwise the container read on the same values, and the container read is inside the
    gate."""
    bufs = kvq_axis_buffers(cfg)
    ws = _workspace()
    n = KVQ_MBS_STEPS[cfg]
    worst, rows, attended, ks, kinds = 0.0, 0, 0, set(), set()
    for i in range(n):
        num_heads, layer, ctx, gen_pos, next_n, floors = _kvq_axis_case(cfg, i)
        g = torch.Generator(device=DEV).manual_seed(
            3_600_000 + 1000 * cfg[1] + i + (cfg[0] == 2048) * 500_000
        )
        st = _KvqStep(ctx, gen_pos, next_n, num_heads, layer, env_pools, bufs, g, floors)
        assert st.B <= cfg[1] and st.T <= cfg[0]
        st.fill()
        o_pool = st.run(ws, compact=False)
        o_comp = st.run(ws, compact=True)
        same = torch.equal(_bits(o_comp), _bits(o_pool))
        assert same, (cfg, i, int((_bits(o_comp) != _bits(o_pool)).sum()))
        e = _excess(o_pool, st.reference(), env_pools["atol"][layer])
        assert e <= 1.0, (cfg, i, e)
        worst, rows, attended = max(worst, e), rows + st.T, attended + st.attended_minus_one()
        ks.add(st.B)
        kinds.add((bool(ctx), bool(gen_pos), next_n if gen_pos else 0))
    print(
        f"kvq mbs partitions {cfg}: {n} steps, {rows} query rows, k {min(ks)}..{max(ks)}, kinds {sorted(kinds)}, "
        f"attended -1 {attended}, compact == container bitwise, max excess {worst:.3f}"
    )
    assert min(ks) == 1 and max(ks) == cfg[1]
    if cfg[1] >= 2:
        assert any(c and g for c, g, _ in kinds), kinds


def test_kvq_mbs_random_partitions_reach_the_envelope():
    """The axis steps span what each configuration permits: every (H, layer) and a full batch of
    k = s requests at every s, a mixed step at every s >= 2, T = max_num_tokens everywhere, generation
    in the DSpark tail everywhere, and floored windows."""
    for cfg, n in KVQ_MBS_STEPS.items():
        cases = [_kvq_axis_case(cfg, i) for i in range(n)]
        ts = [sum(q for _, q in c[2]) + len(c[3]) * c[4] for c in cases]
        assert all(t <= cfg[0] and len(c[2]) + len(c[3]) <= cfg[1] for t, c in zip(ts, cases)), cfg
        assert {(c[0], c[1]) for c in cases} == {
            (h, layer) for h in (16, 64) for layer in ENV_LAYERS
        }, cfg
        assert max(ts) == cfg[0], (cfg, max(ts))
        assert max(len(c[2]) + len(c[3]) for c in cases) == cfg[1], cfg
        assert any(p + c[4] - 1 >= ENV_SEQ for c in cases for p in c[3]), cfg
        if cfg[1] >= 2:
            assert any(c[2] and c[3] for c in cases), cfg
    big = [_kvq_axis_case(cfg, i) for cfg, n in KVQ_MBS_STEPS.items() if n == 200 for i in range(n)]
    assert sum(1 for c in big for f in c[5] if f) >= 50


def _kvq_axis_far_steps(max_t, seqs):
    """_kvq_far_steps for every s. At s >= 2 a full step (T = max_num_tokens, s requests, the last a
    DSpark verify ending at 16397) names the window buffer's last row and compressed row
    (s - 1) * L_r + 16398 // r - 1; with floors, and a verify-only full batch in the tail. At s = 1 the
    window buffer's last row is named by one context of max_num_tokens tokens (also with a floor), and
    compressed row 16398 // r - 1 by one verify ending at 16397."""
    if seqs >= 2:
        return _kvq_far_steps(max_t, seqs)
    ctx = [(ENV_SEQ - max_t, max_t)]
    return [(ctx, [], 1, None), (ctx, [], 1, [ENV_SEQ - max_t - 40]), ([], [KVQ_TAIL - 5], 6, None)]


KVQ_MBS_FAR = [(t, s) for t in (2048, ENV_T) for s in KVQ_MBS]


@pytest.mark.parametrize("cfg", KVQ_MBS_FAR, ids=_kvq_axis_ids(KVQ_MBS_FAR))
def test_kvq_mbs_compact_read_far_end(env_pools, kvq_axis_buffers, cfg):
    """The far end of every buffer size the target allocates over max_batch_size 1..32 at both
    max_num_tokens: the steps of _kvq_axis_far_steps at H 16 and 64 and every layer kind name the
    window buffer's last row and the largest compressed row (s - 1) * L_r + 16398 // r - 1 of this s
    (asserted over the steps); the compact read is bitwise the container read (the floored steps with
    attended -1s among them), inside the gate."""
    bufs = kvq_axis_buffers(cfg)
    max_t, seqs = cfg
    ws = _workspace()
    worst, n = 0.0, 0
    top_win, top_comp, full = -1, {1: -1, 2: -1}, False
    for i, (ctx, gen_pos, next_n, floors) in enumerate(_kvq_axis_far_steps(max_t, seqs)):
        for num_heads in (16, 64):
            for layer in ENV_LAYERS:
                g = torch.Generator(device=DEV).manual_seed(
                    3_700_000 + 100 * i + num_heads + ENV_LAYERS[layer][0]
                )
                st = _KvqStep(ctx, gen_pos, next_n, num_heads, layer, env_pools, bufs, g, floors)
                full |= st.T == max_t and st.B == seqs
                top_win = max(top_win, int(st.cidx[:, :WINDOW].max()))
                if st.ratio:
                    top_comp[st.ratio] = max(top_comp[st.ratio], int(st.cidx[:, WINDOW:].max()))
                if floors is not None:
                    assert st.attended_minus_one() > 0
                st.fill()
                o_pool = st.run(ws, compact=False)
                o_comp = st.run(ws, compact=True)
                assert torch.equal(_bits(o_comp), _bits(o_pool)), (cfg, i, num_heads, layer)
                e = _excess(o_pool, st.reference(), env_pools["atol"][layer])
                assert e <= 1.0, (cfg, i, num_heads, layer, e)
                worst, n = max(worst, e), n + 1
    far = {r: (seqs - 1) * KVQ_L[r] + (KVQ_TAIL + 1) // r - 1 for r in (1, 2)}
    print(
        f"kvq mbs far end {cfg}: {n} steps, window row {top_win}, compressed rows {top_comp} (far {far}), "
        f"compact == container bitwise, max excess {worst:.3f}"
    )
    assert full and top_win == max_t * WINDOW - 1 and top_comp == far


def _kvq_capture(env_pools, bufs, num_heads, layer, batches, seed):
    """test_kvq_compact_read_cuda_graph_capture_set's procedure on one configuration's buffers:
    per next_n 1 / 6 and G in `batches`, one eager call, a capture, the first replay == the eager call,
    then three replays on new steps each == a fresh eager compact call == the container read, inside
    the gate. Returns (graphs, worst excess)."""
    ratio = ENV_LAYERS[layer][0]
    ws = _workspace()
    sinks = _sinks(num_heads)
    max_t, seqs = bufs.max_t, bufs.seqs
    width = WINDOW + (INDEX_TOPK if ratio else 0)
    idx_buf = torch.full((max_t, width), -1, dtype=torch.int32, device=DEV)
    lens_buf = torch.full((max_t,), WINDOW, dtype=torch.int32, device=DEV)
    kvl_buf = torch.full((seqs + 1,), 1, dtype=torch.int32, device=DEV)
    q_buf = torch.zeros(max_t, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    comp = bufs.comp_c4[ratio] if ratio else bufs.dummy
    n_graphs, worst = 0, 0.0
    for next_n in (1, 6):
        for G in batches:
            T = G * next_n
            gen = torch.Generator(device=DEV).manual_seed(
                seed + 1000 * G + 100 * next_n + num_heads + ratio
            )
            rng = random.Random(seed + 7 + 1000 * G + 100 * next_n + num_heads + ratio)
            idx = bufs.win_idx[:T] if not ratio else idx_buf[:T]

            def make(step):
                pos = [rng.randint(0, KVQ_TAIL + 1 - next_n) for _ in range(G)]
                pos[0] = (KVQ_TAIL if step % 2 == 0 else ENV_SEQ - 1) + 1 - next_n
                st = _KvqStep([], pos, next_n, num_heads, layer, env_pools, bufs, gen)
                st.fill()
                q_buf[:T].copy_(st.q3)
                if ratio:
                    idx_buf[:T].copy_(st.cidx)
                lens_buf[:T].copy_(st.active)
                kvl_buf[:G].copy_(st.kv_lens[:G])
                return st

            def call(out):
                return op(
                    q_buf[:T].view(G, next_n, num_heads, HEAD_DIM),
                    bufs.swa_c4,
                    comp,
                    ws,
                    idx,
                    lens_buf[:T],
                    kvl_buf[:G],
                    sinks,
                    SCALE,
                    1.0,
                    out=out,
                )

            shape = (G, next_n, num_heads, HEAD_DIM)
            st = make(0)
            eager = call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
            assert torch.equal(
                _bits(eager.view(T, num_heads, HEAD_DIM)), _bits(st.run(ws, compact=False))
            )
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                o = torch.empty(T, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
                out = call(o.view(shape))
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(_bits(out), _bits(eager)), (max_t, seqs, G, next_n)
            prev = eager.clone()
            for it in range(1, 4):
                st = make(it)
                out.fill_(float("nan"))
                graph.replay()
                fresh = call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
                pool = st.run(ws, compact=False)
                torch.cuda.synchronize()
                assert torch.equal(_bits(out), _bits(fresh)), (
                    max_t,
                    seqs,
                    G,
                    next_n,
                    it,
                    "replay != eager",
                )
                assert torch.equal(_bits(out.view(T, num_heads, HEAD_DIM)), _bits(pool)), (
                    max_t,
                    seqs,
                    G,
                    next_n,
                    it,
                )
                assert not torch.equal(_bits(out), _bits(prev)), (max_t, seqs, G, next_n, it)
                e = _excess(pool, st.reference(), env_pools["atol"][layer])
                assert e <= 1.0, (max_t, seqs, G, next_n, it, e)
                worst = max(worst, e)
                prev = out.clone()
            del graph
            n_graphs += 1
    return n_graphs, worst


@pytest.mark.parametrize("layer", list(ENV_LAYERS))
@pytest.mark.parametrize("num_heads", [16, 64])
@pytest.mark.parametrize("cfg", KVQ_MBS_CAPTURE, ids=_kvq_axis_ids(KVQ_MBS_CAPTURE))
def test_kvq_mbs_compact_read_cuda_graph_every_batch(
    env_pools, kvq_axis_buffers, cfg, num_heads, layer
):
    """Capture on the compact buffers of every max_batch_size s in 1..32 (max_num_tokens 8192) and of
    the witness configuration (2048, 1): the generation call at EVERY G in 1..s (the engine's
    capture set at s with padding off; padding on captures a subset of it) x next_n 1 and 6, at H 16 / 64
    and every layer kind. The first replay equals the eager call; three more replays on new steps
    (request 0 in the DSpark tail) each equal a fresh eager compact call and the container read, bitwise."""
    bufs = kvq_axis_buffers(cfg)
    n_graphs, worst = _kvq_capture(
        env_pools, bufs, num_heads, layer, range(1, cfg[1] + 1), 3_800_000 + cfg[1]
    )
    print(
        f"kvq mbs capture {cfg} H{num_heads} {layer}: {n_graphs} graphs x 4 replays, replay = eager = container, "
        f"max excess {worst:.3f}"
    )
    assert n_graphs == 2 * cfg[1]


KVQ_MBS_BAND_SEQS = (1, 2, 3, 4, 8, 16)


@pytest.mark.parametrize("width", [128, 640])
@pytest.mark.parametrize("num_heads", [16, 64])
def test_kvq_mbs_compact_read_kernel_variant_bands(env_pools, kvq_axis_buffers, num_heads, width):
    """Kernel bands on the small compact buffers: every band of VARLEN_BANDS with B <= s requests (bottom,
    interior, top; _band_cases) as a context step on the (8192, s) buffers for s in 1, 2, 3, 4, 8, 16 --
    at s = 1 every B = 1 band on the smallest buffers. The profiler reads back exactly the band's
    kernels for every compact call, and every compact call is bitwise the container read."""
    ws = _workspace()
    total, used = 0, set()
    for seqs in KVQ_MBS_BAND_SEQS:
        rng = random.Random(num_heads * 1000 + width + 37 + seqs)
        gen = torch.Generator(device=DEV).manual_seed(num_heads * 1000 + width + 37 + seqs)
        bufs = kvq_axis_buffers((ENV_T, seqs))
        cases = [c for c in _band_cases(num_heads, width, rng) if len(c[0]) <= seqs]
        seen, differ = [], []
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for i, (ctx, vid) in enumerate(cases):
                layer = "r0" if width == WINDOW else ("r2", "r1")[i % 2]
                st = _KvqStep(ctx, [], 1, num_heads, layer, env_pools, bufs, gen)
                st.fill()
                o_pool = st.run(ws, compact=False)
                torch.cuda.synchronize()
                torch.cuda._sleep(10)
                o = st.run(ws, compact=True)
                torch.cuda._sleep(10)
                if not torch.equal(_bits(o), _bits(o_pool)):
                    differ.append((ctx, vid))
            torch.cuda.synchronize()
        ev = sorted(
            (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
            key=lambda e: e.time_range.start,
        )
        open_call, cur = False, []
        for e in ev:
            if "spin_kernel" in e.name or "sleep" in e.name:
                if open_call:
                    seen.append(_variant_of(cur))
                open_call, cur = not open_call, []
            elif open_call:
                cur.append(e.name)
        assert len(seen) == len(cases), (seqs, len(seen), len(cases))
        mism = [
            (c, VARLEN_KERNELS[v], s) for (c, v), s in zip(cases, seen) if VARLEN_KERNELS[v] != s
        ]
        assert not mism, (seqs, mism[:3])
        assert not differ, (seqs, differ[:3])
        assert {len(ctx) for ctx, _ in cases} == set(range(1, seqs + 1))
        total += len(cases)
        used |= {v for _, v in cases}
        print(
            f"kvq mbs bands H{num_heads} W{width} s={seqs}: {len(cases)} compact calls (B 1..{seqs}), "
            f"{len({v for _, v in cases})} variants, profiler map = VARLEN_BANDS, compact == container bitwise"
        )
    print(f"kvq mbs bands H{num_heads} W{width}: {total} compact calls, {len(used)} variants")


def _kvq_axis_control_steps():
    """(H, layer, cfg, ctx, gen, next_n, floors) on the smallest buffers. s = 1: a floored cached
    context (attended -1s), a DSpark-tail verify, a fresh context (masked -1 window tails), on both
    s = 1 configurations. s = 2: a floored context plus a tail verify (request 1 has compressed rows),
    and a fresh context plus a decode row."""
    out = []
    for cfg in ((ENV_T, 1), KVQ_WITNESS):
        out += [
            (16, "r2", cfg, [(300, 200)], [], 1, [250]),
            (64, "r1", cfg, [(5000, 64)], [], 1, [4990]),
            (16, "r0", cfg, [(16000, 300)], [], 1, [15950]),
            (64, "r2", cfg, [], [KVQ_TAIL - 5], 6, None),
            (16, "r1", cfg, [(0, 50)], [], 1, None),
            (64, "r0", cfg, [(0, 50)], [], 1, None),
        ]
    cfg = (ENV_T, 2)
    out += [
        (16, "r2", cfg, [(300, 200)], [KVQ_TAIL - 5], 6, [250]),
        (64, "r1", cfg, [(5000, 64)], [KVQ_TAIL], 1, [4990]),
        (16, "r0", cfg, [(16000, 300)], [128], 1, [15950]),
        (64, "r2", cfg, [(0, 50)], [77], 1, None),
    ]
    return out


def test_kvq_mbs_compact_read_controls_on_the_smallest_buffers(env_pools, kvq_axis_buffers):
    """The harness sees a wrong compact read on the smallest buffers (s = 1 at both max_num_tokens, and
    s = 2). Each control applies only where it can differ, and the test says which did not apply:
    - '-1 not zero-backed' (the rows standing for -1 window columns hold 3.0): needs attended -1s
      (floored windows);
    - the window half renumbered one row late: every step;
    - the compressed half one entry late: ratio 1 / 2 layers;
    - the compressed half at the other ratio's row stride (b * L_{3-r} + e) and one request's rows
      left stale: need a request at position b >= 1 with compressed rows, so NOT applicable at s = 1
      (there b = 0 and the other stride names the same row);
    and on the steps whose only -1s are masked (fresh contexts), 3.0 backing leaves the output bitwise
    unchanged and NaN backing poisons it. The correct compact read is bitwise the container read and
    inside the gate; every applicable wrong reading lands > 10x the gate."""
    ws = _workspace()
    applied = {}
    for num_heads, layer, cfg, ctx, gen_pos, next_n, floors in _kvq_axis_control_steps():
        bufs = kvq_axis_buffers(cfg)
        ratio = ENV_LAYERS[layer][0]
        atol = env_pools["atol"][layer]
        g = torch.Generator(device=DEV).manual_seed(
            3_900_000 + num_heads + 10 * ratio + cfg[1] + len(gen_pos)
        )
        st = _KvqStep(ctx, gen_pos, next_n, num_heads, layer, env_pools, bufs, g, floors)
        ref = st.reference()
        st.fill()
        o_pool = st.run(ws, compact=False)
        o_comp = st.run(ws, compact=True)
        assert torch.equal(_bits(o_comp), _bits(o_pool))
        correct = _excess(o_comp, ref, atol)
        assert correct <= 1.0, correct
        neg = st.attended_minus_one()
        masked = int((st.idx[:, :WINDOW] < 0).sum()) - neg
        wrong, na = {}, []
        st.fill(minus_one=3.0)
        backed = st.run(ws, compact=True)
        st.fill(minus_one=float("nan"))
        nan_backed = st.run(ws, compact=True)
        st.fill()
        if neg:
            wrong["minus_one_not_zero_backed"] = _excess(backed, ref, atol)
        else:
            na.append("minus_one_not_zero_backed")
            assert torch.equal(_bits(backed), _bits(o_pool)), (
                "a finite masked row changed the output"
            )
            if masked:
                assert not torch.isfinite(nan_backed).all(), (
                    "a NaN masked row did not poison the output"
                )
        w1 = st.cidx.clone()
        w1[:, :WINDOW] += 1
        wrong["window_renumbered_one_row_late"] = _excess(
            st.run(ws, compact=True, idx=w1), ref, atol
        )
        if ratio:
            c = st.cidx[:, WINDOW:]
            w2 = st.cidx.clone()
            w2[:, WINDOW:] = torch.where(c >= 0, c + 1, c)
            wrong["compressed_renumbered_one_entry_late"] = _excess(
                st.run(ws, compact=True, idx=w2), ref, atol
            )
            if bool(((st.loc >= 0) & (st.req[:, None] >= 1)).any()):
                w3 = st.cidx.clone()
                w3[:, WINDOW:] = torch.where(
                    st.loc >= 0, st.req[:, None] * KVQ_L[3 - ratio] + st.loc, -1
                ).int()
                if int(w3.max()) < bufs.comp_c4[ratio].shape[0] * bufs.comp_c4[ratio].shape[2]:
                    wrong["other_ratio_row_stride"] = _excess(
                        st.run(ws, compact=True, idx=w3), ref, atol
                    )
                else:  # ratio 2 at small s: the ratio-1 stride points past the view (not a callable index)
                    na.append("other_ratio_row_stride(past the view)")
                L = KVQ_L[ratio]
                bufs.comp_c[L : 2 * L] = torch.randn(
                    L, HEAD_DIM, generator=g, device=DEV, dtype=torch.bfloat16
                )
                st.fill(skip_request=1)
                wrong["request_1_rows_stale"] = _excess(st.run(ws, compact=True), ref, atol)
                st.fill()
            else:
                na += ["other_ratio_row_stride", "request_1_rows_stale"]
        else:
            na += [
                "compressed_renumbered_one_entry_late",
                "other_ratio_row_stride",
                "request_1_rows_stale",
            ]
        print(
            f"kvq mbs controls {cfg} H{num_heads} {layer} B {st.B}: attended -1 {neg}, masked -1 {masked}, "
            f"correct {correct:.3f}, wrong { {k: round(v, 1) for k, v in wrong.items()} }, not applicable {na}"
        )
        for name, e in wrong.items():
            assert e > 10.0 * max(correct, 0.1), (
                f"{name} not discriminated: {e:.2f} vs {correct:.2f}"
            )
            applied.setdefault((cfg[1], name), 0)
            applied[(cfg[1], name)] += 1
    print(f"kvq mbs controls applied (s, control): steps {applied}")
    for s in (1, 2):
        for name in (
            "minus_one_not_zero_backed",
            "window_renumbered_one_row_late",
            "compressed_renumbered_one_entry_late",
        ):
            assert applied.get((s, name), 0) > 0, (s, name)
    for name in ("other_ratio_row_stride", "request_1_rows_stale"):
        assert applied.get((2, name), 0) > 0 and (1, name) not in applied, name


def test_kvq_mbs_smallest_compact_buffers_ending_a_mapping_never_fault():
    """Precondition 8 at the smallest compressed buffers the axis allocates: the views at s = 1, 2, 3
    ([132 s, 1, 64, 512], 8,448 s rows at ratio 2; [132 s, 1, 128, 512], 16,896 s rows at ratio 1), each
    placed so that its last row ends a mapping, its last rows attended and a -1 loaded from it (lengths
    129 / 131 / 133 / 639, H 16 and 64). Nothing faults, and every output is bitwise the output on a
    normal allocation. The smallest view, 8,448 rows, is 66 times the 128 rows a loaded -1's over-read
    (a row in 96..127 from the first row) needs."""
    trials = []
    for seqs in (1, 2, 3):
        for r in (2, 1):
            page = 128 // r
            rows = seqs * KVQ_COLS * page
            assert (
                rows == 16_896 * seqs // r and rows >= KVQ_MBS_SMALLEST_ROWS >= 66 * POOL_MIN_ROWS
            )
            for j, lens in enumerate((129, 131, 133, 639)):
                h = (16, 64)[(j + seqs + r) % 2]
                trials.append(
                    dict(
                        name=f"kvq-mbs-comp-r{r}-s{seqs}-lens{lens}-H{h}",
                        kind="engine",
                        pool="comp",
                        rows=rows,
                        page=page,
                        lens=lens,
                        heads=h,
                        window=0,
                        mapped=rows,
                        seed=2_200_000 + 100 * seqs + 10 * r + j,
                    )
                )
    p, ok, tried = _guard_run(trials)
    names = [c["name"] for c in trials]
    assert p.returncode == 0, f"child failed after {tried[-1:]}: {p.stderr[-2000:]}"
    assert list(ok) == names, (list(ok), names)
    assert all(ok.values()), [k for k, v in ok.items() if not v]
    print(
        f"kvq mbs guard page: {len(names)} trials on the s = 1..3 compressed views (smallest "
        f"{KVQ_MBS_SMALLEST_ROWS} rows), each bitwise = the normal allocation"
    )


# ---------------------------------------------------------------------------------------------
# The context call over the engine envelope (per rank): up to 16384 tokens per step
# (max_num_tokens, E2), up to 256 requests (max_batch_size, E3), and a cached prefix + q_len up to the
# DSpark cache manager's length Lm = L + 14 at max_seq_len L up to 1,048,576 (E4); in a mixed step the
# generation call follows at next_n 1 (DSpark off) or 6 (DSpark on, kv lengths up to Lm + 5). Calls are
# made as the core target makes them (_Step). These tests use their own pools (env_pools_e): a 2-page
# SWA ring per request for 256 requests, and 130 compressed pages per request. A _Step request gets
# pages for every compressed position it can see up to 128 pages -- every position of a 16384-token
# request at ratio 2 (64-row pages) and ratio 1 (128-row pages); a request that can see more uses
# its 128 pages as a ring, so far positions alias nearer rows and an index row can name one row
# twice; the call and the reference both attend it twice.
# ---------------------------------------------------------------------------------------------
E_T = 16384
E_L = (16384, 32768, 65536, 131072, 262144, 1048576)
E_LM = E_L[-1] + 14  # the DSpark manager length at max_seq_len 1,048,576
E_CAP_PAGES = 128
E_SWA_PAGES = 2 * E_B + 32
# 130 pages per request: the native-storage per-request steps below give every request distinct pages for
# all its compressed positions up to 16398 (129 pages at either ratio).
E_COMP_PAGES = E_B * 130 + 64
# A 16384-token chunk that ends at position Lm - 1 of the longest DSpark request.
E_LONG_CACHED = E_LM - E_T


@pytest.fixture(scope="module")
def env_pools_e():
    """Random pools for the engine envelope (see the section comment)."""
    gen = torch.Generator(device=DEV).manual_seed(4343)
    swa = torch.randn(E_SWA_PAGES, 1, SWA_PAGE, HEAD_DIM, generator=gen, device=DEV).to(
        torch.bfloat16
    )
    pools = {"swa": swa, "comp": {}, "atol": {}}
    for layer, (ratio, page) in ENV_LAYERS.items():
        if ratio:
            comp = torch.empty(E_COMP_PAGES, 1, page, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
            for s0 in range(0, E_COMP_PAGES, 4096):
                s1 = min(E_COMP_PAGES, s0 + 4096)
                comp[s0:s1] = torch.randn(s1 - s0, 1, page, HEAD_DIM, generator=gen, device=DEV).to(
                    torch.bfloat16
                )
            pools["atol"][layer] = _atol(swa, comp)
        else:
            # The core target's ratio-0 placeholder shape; NaN, so that any read of it would show.
            comp = torch.full((1, 1, 64, HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=DEV)
            pools["atol"][layer] = _atol(swa)
        pools["comp"][layer] = comp
    yield pools
    del pools
    torch.cuda.empty_cache()


def _e_step(ctx, dec, num_heads, layer, pools, gen, next_n=1):
    return _Step(ctx, dec, num_heads, layer, pools, gen, next_n=next_n, cap_pages=E_CAP_PAGES)


@pytest.mark.parametrize("cached", [0, E_LONG_CACHED])
@pytest.mark.parametrize("layer", ["r0", "r2"])
@pytest.mark.parametrize("num_heads", ALL_HEADS)
def test_e_prefill_every_max_q_len_to_16384(env_pools_e, num_heads, layer, cached):
    """One context request of every q_len 1..16384 (max_q_len = T = q_len), fresh and as the last
    16384 tokens of a request of the DSpark manager length 1,048,590 (positions up to 1,048,589),
    against the fp32 reference. As in test_prefill_every_max_q_len_one_request, row j's result does
    not depend on q_len, so each call is held to the first q_len rows of ONE reference computed for
    the 16384-row request."""
    gen = torch.Generator(device=DEV).manual_seed(
        32000 + num_heads + (cached > 0) + ENV_LAYERS[layer][0]
    )
    st = _e_step([(cached, E_T)], [], num_heads, layer, env_pools_e, gen)
    ref = st.reference()
    den = env_pools_e["atol"][layer] + RTOL * ref.abs()
    ws = _workspace()
    out = torch.empty_like(st.q3)
    worst = torch.zeros(E_T, device=DEV)
    seq = torch.zeros(1, dtype=torch.int32, device=DEV)
    cu = torch.zeros(2, dtype=torch.int32, device=DEV)
    diff = torch.empty_like(ref)
    for n in range(1, E_T + 1):
        seq.fill_(cached + n)
        cu[1].fill_(n)
        op(
            st.q3[:n],
            st.swa,
            st.comp,
            ws,
            st.idx[:n],
            st.active[:n],
            seq,
            st.sinks,
            SCALE,
            1.0,
            cum_seq_lens_q=cu,
            max_q_len=n,
            out=out[:n],
        )
        d = diff[:n]
        d.copy_(out[:n]).sub_(ref[:n]).abs_().div_(den[:n])
        torch.amax(d, out=worst[n - 1])
    torch.cuda.synchronize()
    bad = (worst > 1.0).nonzero().flatten()
    print(
        f"every-q_len-16384 H{num_heads} {layer} cached {cached}: max excess {worst.max().item():.3f}"
    )
    assert bad.numel() == 0, (
        f"q_len {(bad + 1).tolist()[:20]} exceed the gate (max {worst.max().item():.2f})"
    )
    assert torch.isfinite(out).all()


def _e_partition_case(seed):
    """A random step of the engine envelope: k in 1..256 context requests over T in 1..16384
    tokens (k and T log-uniform or uniform), random q_len split and order; a max_seq_len L drawn from
    E_L with DSpark on (Lm = L + 14) or off (Lm = L); each context request fresh (40 %), ending at the
    last position Lm - 1 (15 %) or at a random cached prefix (cached + q_len <= Lm). For half the seeds
    a mixed step: G generation requests after the context rows (k + G <= 256, T + G * next_n <=
    16384) at next_n 6 with kv lengths up to Lm + 5 (DSpark on) or next_n 1 up to L (off).
    Seeds below 1000 run H 16 / 64, seeds from 1000 on the members' H 8 / 32."""
    rng = random.Random(1900000 + seed)
    num_heads = ((16, 64) if seed < 1000 else MEMBER_HEADS)[seed % 2]
    layer = ("r0", "r2", "r1")[(seed // 2) % 3]
    if rng.random() < 0.5:
        k = rng.randint(1, E_B)
    else:
        k = int(round(math.exp(rng.uniform(0.0, math.log(E_B)))))
    if rng.random() < 0.5:
        t = int(round(math.exp(rng.uniform(0.0, math.log(E_T)))))
    else:
        t = rng.randint(1, E_T)
    t = max(t, k)
    cuts = sorted(rng.sample(range(1, t), k - 1)) if k > 1 else []
    q_lens = [b - a for a, b in zip([0] + cuts, cuts + [t])]
    L = E_L[rng.randrange(len(E_L))]
    dspark = rng.random() < 0.5
    lm = L + 14 if dspark else L
    ctx = []
    for q in q_lens:
        u = rng.random()
        ctx.append((0 if u < 0.4 else (lm - q if u < 0.55 else rng.randint(0, lm - q)), q))
    dec, next_n = [], (6 if dspark else 1)
    if seed % 4 >= 2 and k < E_B and E_T - t >= next_n:
        g = rng.randint(1, min(E_B - k, (E_T - t) // next_n))
        top = lm - 1 if dspark else L - 1  # verify: cached + 6 <= Lm + 5; decode: pos + 1 <= L
        dec = [top if rng.random() < 0.2 else rng.randint(0, top) for _ in range(g)]
    return num_heads, layer, ctx, dec, next_n, L, dspark


E_PARTITION_SEEDS = list(range(240))
E_MEMBER_PARTITION_SEEDS = list(range(1000, 1240))
E_PARTITION_SETS = {
    "h16_h64": (E_PARTITION_SEEDS, (16, 64)),
    "h8_h32": (E_MEMBER_PARTITION_SEEDS, MEMBER_HEADS),
}


def _seed_ids(seeds):
    """Test ids: the seed, prefixed 'h8h32-' for the members' head counts (seeds from 1000 on)."""
    return [str(s) if s < 1000 else f"h8h32-{s}" for s in seeds]


@pytest.mark.parametrize(
    "seed",
    E_PARTITION_SEEDS + E_MEMBER_PARTITION_SEEDS,
    ids=_seed_ids(E_PARTITION_SEEDS + E_MEMBER_PARTITION_SEEDS),
)
def test_e_prefill_random_partition(env_pools_e, pq_buffers, seed):
    """At the engine envelope: 240 seeded random partitions of T in [1, 16384]
    into k in [1, 256] context requests at H 16 / 64, and 240 more at H 8 / 32, half of them mixed
    steps whose generation call (next_n 1 or 6) reads the views starting at row tc, all against the
    fp32 reference. Each step is also read in the target's native per-query form (compact window,
    compressed rows t * 512 + k at max_num_tokens 16384; see _PerQueryBuffers), bitwise equal to the
    container read."""
    num_heads, layer, ctx, dec, next_n, L, dspark = _e_partition_case(seed)
    gen = torch.Generator(device=DEV).manual_seed(1500000 + seed)
    st = _e_step(ctx, dec, num_heads, layer, env_pools_e, gen, next_n)
    ws = _workspace()
    o = st.run(ws)
    e = _excess(o, st.reference(), env_pools_e["atol"][layer])
    o_pq = _pq_read(st, pq_buffers, ws)
    print(
        f"e-partition {seed}: H{num_heads} {layer} L {L} dspark {int(dspark)} T {st.T} tc {st.tc} nc {st.nc} "
        f"G {st.G} next_n {next_n} max_q_len {st.max_ctx_q} max kv {int(st.kv_lens[: st.B].max())} "
        f"excess {e:.3f}, per-query read bitwise {torch.equal(_bits(o_pq), _bits(o))}"
    )
    assert torch.isfinite(o).all(), seed
    assert e <= 1.0, (seed, e)
    assert torch.equal(_bits(o_pq), _bits(o)), seed


@pytest.mark.parametrize("seeds", list(E_PARTITION_SETS))
def test_e_prefill_random_partitions_reach_the_envelope(seeds):
    """Each set of 240 partitions spans the engine envelope: k from 1 to >= 250 (and every
    decade), T up to >= 16000, total requests up to 256, context requests ending at Lm - 1 for every L
    with DSpark on and off, verify rows at kv length Lm + 5 of the longest L, mixed steps at next_n 1
    and 6 whose views start at odd rows / odd requests, and every (H, layer) of its head counts."""
    seed_list, heads = E_PARTITION_SETS[seeds]
    cases = [_e_partition_case(s) for s in seed_list]
    ks = [len(c[2]) for c in cases]
    ts = [sum(q for _, q in c[2]) for c in cases]
    mixed = [c for c in cases if c[3]]
    assert min(ks) == 1 and max(ks) >= 250, (min(ks), max(ks))
    assert all(any(lo <= k < 10 * lo for k in ks) for lo in (1, 10, 100)), sorted(ks)
    assert min(ts) <= 4 and max(ts) >= 16000, (min(ts), max(ts))
    assert max(len(c[2]) + len(c[3]) for c in cases) == E_B
    ends = {(c[5], c[6]) for c in cases for cached, q in c[2] if cached + q == c[5] + 14 * c[6]}
    assert ends == {(L, d) for L in E_L for d in (False, True)}, sorted(ends)
    assert any(p + 6 == E_LM + 5 for c in cases if c[6] for p in c[3]), "no verify row at Lm + 5"
    assert {c[4] for c in mixed} == {1, 6}
    odd_tc = sum(sum(q for _, q in c[2]) % 2 for c in mixed)
    odd_nc = sum(len(c[2]) % 2 for c in mixed)
    assert len(mixed) >= 100 and odd_tc >= 20 and odd_nc >= 20, (len(mixed), odd_tc, odd_nc)
    assert {(c[0], c[1]) for c in cases} == {(h, layer) for h in heads for layer in ENV_LAYERS}


@pytest.mark.parametrize("seeds", list(E_PARTITION_SETS))
def test_e_prefill_partition_wrong_readings_fail_the_gate(env_pools_e, seeds):
    """The partition harness sees a wrong reading at the new envelope too. On the first four mixed
    partitions of each seed set with >= 64 requests and a fresh context request of >= 2 tokens, the
    reference built on each wrong reading -- every context query given its request's LAST window
    count, each request's window count from its neighbour's seq_lens, the generation rows' q read
    from row 0 instead of row tc -- lands far outside the gate while the call sits inside it."""
    picked = []
    for seed in E_PARTITION_SETS[seeds][0]:
        num_heads, layer, ctx, dec, next_n, L, dspark = _e_partition_case(seed)
        if dec and len(ctx) >= 64 and any(c == 0 and q >= 2 for c, q in ctx):
            picked.append(seed)
        if len(picked) == 4:
            break
    assert len(picked) == 4, picked
    for seed in picked:
        num_heads, layer, ctx, dec, next_n, L, dspark = _e_partition_case(seed)
        gen = torch.Generator(device=DEV).manual_seed(1500000 + seed)
        st = _e_step(ctx, dec, num_heads, layer, env_pools_e, gen, next_n)
        o = st.run(_workspace())
        atol = env_pools_e["atol"][layer]
        correct = _excess(o, st.reference(), atol)
        q_lens = [q for _, q in st.requests]
        seq = st.kv_lens[: st.B].long()
        last = torch.repeat_interleave(seq.clamp(max=WINDOW), torch.tensor(q_lens, device=DEV))
        rolled = seq.roll(1)
        pos_rolled = torch.cat(
            [torch.arange(q, device=DEV) + int(s) - q for s, q in zip(rolled.tolist(), q_lens)]
        )
        wrong = {
            "non_causal_window": _reference(
                st.q3, st.swa, st.comp, st.idx, st.active, last, st.sinks
            ),
            "neighbour_seq_lens": _reference(
                st.q3,
                st.swa,
                st.comp,
                st.idx,
                st.active,
                (pos_rolled + 1).clamp(1, WINDOW),
                st.sinks,
            ),
        }
        q_lost = st.q3.clone()
        q_lost[st.tc :] = st.q3[: st.T - st.tc]
        wrong["gen_view_from_row_0"] = _reference(
            q_lost, st.swa, st.comp, st.idx, st.active, st.counts, st.sinks
        )
        excess = {k: _excess(o, r, atol) for k, r in wrong.items()}
        print(
            f"e-partition controls seed {seed} (H{num_heads} {layer} nc {st.nc}, G {st.G}, next_n {next_n}): "
            f"correct {correct:.3f}, wrong { {k: round(v, 1) for k, v in excess.items()} }"
        )
        assert correct <= 1.0, correct
        for name, e in excess.items():
            assert e > 10.0 * max(correct, 0.1), (
                f"seed {seed}: {name} not discriminated ({e:.2f} vs {correct:.2f})"
            )


# Kernel map of the context call over the engine envelope (profiler probe of 174,080 calls on
# sm_103, 152 SMs: every (H, width, B) with B 1..256 at every max_q_len 1..160 and at 256, 512, 1024,
# 2048, 4096, 8191, 8192, 8193, 12288 and 16384 - (B - 1), the other B - 1 requests at q_len 1). For B <= 32 it is
# VARLEN_BANDS cell by cell, with the top band running on to 16384 - (B - 1). E_BAND_MAP[(H, width)] =
# 'B0-B1:lo-hi/v,...;...' over runs of B with one band structure; hi '*' is the envelope's top 16384 - (B - 1); v
# indexes E_VARLEN_KERNELS. The H 8 / 32 entries come from the same probe run at those head counts (another
# 174,080 calls); the Q32 kernels (indices 18..23) appear only at H 32.
E_VARLEN_KERNELS = (
    ("HVPerCta128PagedKvDenseDynamicTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen",),
    ("HVPerCta256PagedKvDenseDynamicTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen",),
    ("PagedKvDenseDynamicTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen",),
    ("HVPerCta128PagedKvDenseDynamicTokenSparseP1VarSeqQ16Kv128PersistentSwapsAbForGen",),
    ("HVPerCta256PagedKvDenseDynamicTokenSparseP1VarSeqQ16Kv128PersistentSwapsAbForGen",),
    ("PagedKvDenseDynamicTokenSparseP1VarSeqQ16Kv128PersistentSwapsAbForGen",),
    ("HVPerCta128PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",),
    ("HVPerCta256PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",),
    ("PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",),
    ("HVPerCta128PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",),
    ("HVPerCta256PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",),
    ("PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",),
    ("HVPerCta128PagedKvDenseDynamicTokenSparseP1VarSeqQ64Kv128PersistentKeepsAbForGen",),
    ("HVPerCta256PagedKvDenseDynamicTokenSparseP1VarSeqQ64Kv128PersistentKeepsAbForGen",),
    ("PagedKvDenseDynamicTokenSparseP1VarSeqQ64Kv128PersistentKeepsAbForGen",),
    (
        "HVPerCta128PagedKvDenseDynamicTokenSparseP1MultiCtasKvGmemSepVarSeqQ64Kv128StaticKeepsAbForGen",
        "fmhaReductionKernel<64, 128, false, __nv_bfloat16, __nv_bfloat16>",
    ),
    (
        "HVPerCta256PagedKvDenseDynamicTokenSparseP1MultiCtasKvGmemSepVarSeqQ64Kv128StaticKeepsAbForGen",
        "fmhaReductionKernel<64, 256, false, __nv_bfloat16, __nv_bfloat16>",
    ),
    (
        "PagedKvDenseDynamicTokenSparseP1MultiCtasKvGmemSepVarSeqQ64Kv128StaticKeepsAbForGen",
        "fmhaReductionKernel<64, 512, false, __nv_bfloat16, __nv_bfloat16>",
    ),
    ("HVPerCta128PagedKvDenseDynamicTokenSparseP1VarSeqQ32Kv128PersistentSwapsAbForGen",),
    ("HVPerCta256PagedKvDenseDynamicTokenSparseP1VarSeqQ32Kv128PersistentSwapsAbForGen",),
    ("PagedKvDenseDynamicTokenSparseP1VarSeqQ32Kv128PersistentSwapsAbForGen",),
    ("HVPerCta128PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen",),
    ("HVPerCta256PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen",),
    ("PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen",),
)
E_BAND_MAP = {
    (16, 128): (
        "1-1:1-19/0,20-38/1,39-*/2;"
        "2-2:1-9/0,10-19/1,20-*/2;"
        "3-3:1-6/0,7-12/1,13-*/2;"
        "4-4:1-4/0,5-9/1,10-*/2;"
        "5-5:1-3/0,4-7/1,8-*/2;"
        "6-6:1-3/0,4-6/1,7-*/2;"
        "7-7:1-2/0,3-5/1,6-*/2;"
        "8-9:1-2/0,3-4/1,5-*/2;"
        "10-12:1-1/0,2-3/1,4-*/2;"
        "13-19:1-1/0,2-2/1,3-*/2;"
        "20-25:1-1/3,2-3/4,4-*/5;"
        "26-38:1-1/3,2-2/4,3-*/5;"
        "39-76:1-1/4,2-*/5;"
        "77-256:1-*/5"
    ),
    (16, 640): (
        "1-1:1-6/6,7-12/7,13-38/8,39-*/2;"
        "2-2:1-3/6,4-6/7,7-19/8,20-*/2;"
        "3-3:1-2/6,3-4/7,5-12/8,13-*/2;"
        "4-4:1-3/9,4-6/10,7-19/11,20-*/5;"
        "5-5:1-2/9,3-5/10,6-15/11,16-*/5;"
        "6-6:1-2/9,3-4/10,5-12/11,13-*/5;"
        "7-7:1-1/9,2-3/10,4-10/11,11-*/5;"
        "8-8:1-1/9,2-3/10,4-9/11,10-*/5;"
        "9-9:1-1/9,2-2/10,3-8/11,9-*/5;"
        "10-10:1-1/9,2-2/10,3-7/11,8-*/5;"
        "11-12:1-1/9,2-2/10,3-6/11,7-*/5;"
        "13-15:1-1/10,2-5/11,6-*/5;"
        "16-19:1-1/10,2-4/11,5-*/5;"
        "20-25:1-1/10,2-3/11,4-*/5;"
        "26-38:1-2/11,3-*/5;"
        "39-76:1-1/11,2-*/5;"
        "77-256:1-*/5"
    ),
    (64, 128): (
        "1-1:1-38/12,39-76/13,77-*/14;"
        "2-2:1-19/12,20-38/13,39-*/14;"
        "3-3:1-12/12,13-25/13,26-*/14;"
        "4-4:1-9/12,10-19/13,20-*/14;"
        "5-5:1-7/12,8-15/13,16-*/14;"
        "6-6:1-6/12,7-12/13,13-*/14;"
        "7-7:1-5/12,6-10/13,11-*/14;"
        "8-8:1-4/12,5-9/13,10-*/14;"
        "9-9:1-4/12,5-8/13,9-*/14;"
        "10-10:1-3/12,4-7/13,8-*/14;"
        "11-12:1-3/12,4-6/13,7-*/14;"
        "13-15:1-2/12,3-5/13,6-*/14;"
        "16-19:1-2/12,3-4/13,5-*/14;"
        "20-25:1-1/12,2-3/13,4-*/14;"
        "26-38:1-1/12,2-2/13,3-*/14;"
        "39-76:1-1/13,2-*/14;"
        "77-256:1-*/14"
    ),
    (64, 640): (
        "1-1:1-12/15,13-25/16,26-76/17,77-*/14;"
        "2-2:1-6/15,7-12/16,13-38/17,39-*/14;"
        "3-3:1-4/15,5-8/16,9-25/17,26-*/14;"
        "4-4:1-3/15,4-6/16,7-19/17,20-*/14;"
        "5-5:1-2/15,3-5/16,6-15/17,16-*/14;"
        "6-6:1-2/15,3-4/16,5-12/17,13-*/14;"
        "7-7:1-1/15,2-3/16,4-10/17,11-*/14;"
        "8-8:1-1/15,2-3/16,4-9/17,10-*/14;"
        "9-9:1-1/15,2-2/16,3-8/17,9-*/14;"
        "10-10:1-1/15,2-2/16,3-7/17,8-*/14;"
        "11-12:1-1/15,2-2/16,3-6/17,7-*/14;"
        "13-15:1-1/16,2-5/17,6-*/14;"
        "16-19:1-1/16,2-4/17,5-*/14;"
        "20-25:1-1/16,2-3/17,4-*/14;"
        "26-38:1-2/17,3-*/14;"
        "39-76:1-1/17,2-*/14;"
        "77-256:1-*/14"
    ),
    (8, 128): (
        "1-1:1-38/0,39-76/1,77-*/2;"
        "2-2:1-19/0,20-38/1,39-*/2;"
        "3-3:1-12/0,13-25/1,26-*/2;"
        "4-4:1-9/0,10-19/1,20-*/2;"
        "5-5:1-7/0,8-15/1,16-*/2;"
        "6-6:1-6/0,7-12/1,13-*/2;"
        "7-7:1-5/0,6-10/1,11-*/2;"
        "8-8:1-4/0,5-9/1,10-*/2;"
        "9-9:1-4/0,5-8/1,9-*/2;"
        "10-10:1-3/0,4-7/1,8-*/2;"
        "11-12:1-3/0,4-6/1,7-*/2;"
        "13-15:1-2/0,3-5/1,6-*/2;"
        "16-19:1-2/0,3-4/1,5-*/2;"
        "20-25:1-1/0,2-3/1,4-*/2;"
        "26-38:1-1/0,2-2/1,3-*/2;"
        "39-76:1-1/1,2-*/2;"
        "77-256:1-*/2"
    ),
    (8, 640): (
        "1-1:1-12/6,13-25/7,26-76/8,77-*/2;"
        "2-2:1-6/6,7-12/7,13-38/8,39-*/2;"
        "3-3:1-4/6,5-8/7,9-25/8,26-*/2;"
        "4-4:1-3/6,4-6/7,7-19/8,20-*/2;"
        "5-5:1-2/6,3-5/7,6-15/8,16-*/2;"
        "6-6:1-2/6,3-4/7,5-12/8,13-*/2;"
        "7-7:1-1/6,2-3/7,4-10/8,11-*/2;"
        "8-8:1-1/6,2-3/7,4-9/8,10-*/2;"
        "9-9:1-1/6,2-2/7,3-8/8,9-*/2;"
        "10-10:1-1/6,2-2/7,3-7/8,8-*/2;"
        "11-12:1-1/6,2-2/7,3-6/8,7-*/2;"
        "13-15:1-1/7,2-5/8,6-*/2;"
        "16-19:1-1/7,2-4/8,5-*/2;"
        "20-25:1-1/7,2-3/8,4-*/2;"
        "26-38:1-2/8,3-*/2;"
        "39-76:1-1/8,2-*/2;"
        "77-256:1-*/2"
    ),
    (32, 128): (
        "1-1:1-19/3,20-38/4,39-*/5;"
        "2-2:1-9/3,10-19/4,20-*/5;"
        "3-3:1-6/3,7-12/4,13-*/5;"
        "4-4:1-4/3,5-9/4,10-*/5;"
        "5-5:1-3/3,4-7/4,8-*/5;"
        "6-6:1-3/3,4-6/4,7-*/5;"
        "7-7:1-2/3,3-5/4,6-*/5;"
        "8-9:1-2/3,3-4/4,5-*/5;"
        "10-12:1-1/3,2-3/4,4-*/5;"
        "13-19:1-1/3,2-2/4,3-*/5;"
        "20-25:1-1/18,2-3/19,4-*/20;"
        "26-38:1-1/18,2-2/19,3-*/20;"
        "39-76:1-1/19,2-*/20;"
        "77-256:1-*/20"
    ),
    (32, 640): (
        "1-1:1-6/9,7-12/10,13-38/11,39-*/5;"
        "2-2:1-3/9,4-6/10,7-19/11,20-*/5;"
        "3-3:1-2/9,3-4/10,5-12/11,13-*/5;"
        "4-4:1-3/21,4-6/22,7-19/23,20-*/20;"
        "5-5:1-2/21,3-5/22,6-15/23,16-*/20;"
        "6-6:1-2/21,3-4/22,5-12/23,13-*/20;"
        "7-7:1-1/21,2-3/22,4-10/23,11-*/20;"
        "8-8:1-1/21,2-3/22,4-9/23,10-*/20;"
        "9-9:1-1/21,2-2/22,3-8/23,9-*/20;"
        "10-10:1-1/21,2-2/22,3-7/23,8-*/20;"
        "11-12:1-1/21,2-2/22,3-6/23,7-*/20;"
        "13-15:1-1/22,2-5/23,6-*/20;"
        "16-19:1-1/22,2-4/23,5-*/20;"
        "20-25:1-1/22,2-3/23,4-*/20;"
        "26-38:1-2/23,3-*/20;"
        "39-76:1-1/23,2-*/20;"
        "77-256:1-*/20"
    ),
}


def _e_bands(spec: str) -> dict:
    """Parse E_BAND_MAP: per (H, width), 'B0-B1:lo-hi/v,...;...' with the last band's hi '*' =
    16384 - (B - 1), the envelope's top for B requests."""
    out = {}
    for key, text in spec.items():
        per_b = {}
        for seg in text.split(";"):
            brange, bands = seg.split(":")
            b0, b1 = (int(x) for x in brange.split("-"))
            for b in range(b0, b1 + 1):
                row = []
                for band in bands.split(","):
                    lohi, vid = band.split("/")
                    lo, hi = lohi.split("-")
                    row.append((int(lo), E_T - (b - 1) if hi == "*" else int(hi), int(vid)))
                per_b[b] = tuple(row)
        assert sorted(per_b) == list(range(1, E_B + 1)), key
        for b, row in per_b.items():
            assert row[0][0] == 1 and row[-1][1] == E_T - (b - 1), (key, b, row)
            assert all(row[i][1] + 1 == row[i + 1][0] for i in range(len(row) - 1)), (key, b, row)
        out[key] = per_b
    return out


E_VARLEN_BANDS = _e_bands(E_BAND_MAP)


def _e_band_template(num_heads, layer, cached, pools, gen):
    """The max_q_len request of every band case: E_T rows at `cached`, with ONE fp32 reference; a
    case of max_q_len m uses its first m rows (a row's result does not depend on q_len)."""
    st = _e_step([(cached, E_T)], [], num_heads, layer, pools, gen)
    return st, st.reference()


def _e_band_case(tpl, mq, b, rng, gen, pools, layer):
    """One call of b requests: the template's first mq rows as one request, and b - 1 short
    requests (q_len 1..min(mq, 8), fresh or cached up to E_LM; a 2-page SWA ring and a 4-page
    compressed ring each, as _paged_rows builds them) in random order, within the 16384-row step.
    Returns (q, idx, active, seq_lens, cu, the big request's row slice, the short rows' reference)."""
    st, _ = tpl
    others = [rng.randint(1, min(mq, 8)) for _ in range(b - 1)]
    while sum(others) > E_T - mq:
        i = max(range(b - 1), key=lambda k: others[k])
        others[i] -= 1
    small = [(0 if rng.random() < 0.4 else rng.randint(0, E_LM - q), q) for q in others]
    at = rng.randrange(b)
    q_parts, i_parts, a_parts, seq, qls = [], [], [], [], []
    ref_small = None
    if small:
        ratio, page = ENV_LAYERS[layer]
        ns = len(small)
        swa_bt = torch.randperm(st.swa.shape[0], generator=gen, device=DEV)[: 2 * ns].view(ns, 2)
        comp_bt = (
            torch.randperm(st.comp.shape[0], generator=gen, device=DEV)[: 4 * ns].view(ns, 4)
            if ratio
            else None
        )
        s_idx, s_act, s_seq, s_cu = _paged_rows(
            small, gen, st.width, max(ratio, 1), swa_bt, comp_bt, SWA_PAGE, page
        )
        s_cnt = _window_counts(s_seq, s_cu)
        s_q = (
            torch.randn(s_idx.shape[0], st.q3.shape[1], HEAD_DIM, generator=gen, device=DEV)
            .mul_(2)
            .to(torch.bfloat16)
        )
        ref_small = _reference(s_q, st.swa, st.comp, s_idx, s_act, s_cnt, st.sinks)
        cut = int(s_cu[at])
    for part in ("before", "big", "after"):
        if part == "big":
            q_parts.append(st.q3[:mq])
            i_parts.append(st.idx[:mq])
            a_parts.append(st.active[:mq])
            seq.append(int(st.kv_lens[0]) - E_T + mq)
            qls.append(mq)
        elif small:
            sl = slice(0, cut) if part == "before" else slice(cut, None)
            rs = slice(0, at) if part == "before" else slice(at, None)
            q_parts.append(s_q[sl])
            i_parts.append(s_idx[sl])
            a_parts.append(s_act[sl])
            seq += s_seq[rs].tolist()
            qls += [q for _, q in small[rs]]
    big = slice(cut if small else 0, (cut if small else 0) + mq)
    cu = torch.tensor([0] + qls, dtype=torch.int32, device=DEV).cumsum(0).int()
    return (
        torch.cat(q_parts).contiguous(),
        torch.cat(i_parts).contiguous(),
        torch.cat(a_parts).contiguous(),
        torch.tensor(seq, dtype=torch.int32, device=DEV),
        cu,
        big,
        ref_small,
    )


@pytest.mark.parametrize("width", [128, 640])
@pytest.mark.parametrize("num_heads", ALL_HEADS)
def test_e_prefill_kernel_variant_bands(env_pools_e, num_heads, width):
    """Kernel bands of the context call over the engine envelope: every kernel band of
    E_VARLEN_BANDS at every request count B 1..256, at its lowest, one interior and its highest
    max_q_len (the top band ending at 16384 - (B - 1)), with the kernels read back from the profiler
    and every row held to the fp32 reference. The max_q_len request is a prefix of one 16384-row
    template per (layer, cached prefix) -- fresh, and the last 16384 positions of a 1,048,590-token
    request -- so its reference is computed once; the other B - 1 requests are 1..8 tokens (the
    kernel is a function of (H, width, B, max_q_len), not of the other requests' lengths)."""
    rng = random.Random(num_heads * 1000 + width + 7)
    gen = torch.Generator(device=DEV).manual_seed(num_heads * 1000 + width + 7)
    ws = _workspace()
    layers = ("r0", "r0") if width == WINDOW else ("r2", "r1")
    tpls = [
        (layers[0], _e_band_template(num_heads, layers[0], 0, env_pools_e, gen)),
        (layers[1], _e_band_template(num_heads, layers[1], E_LONG_CACHED, env_pools_e, gen)),
    ]
    cases = []
    for b, bands in E_VARLEN_BANDS[(num_heads, width)].items():
        for lo, hi, vid in bands:
            for mq in sorted({lo, (lo + hi) // 2, hi}):
                cases.append((b, mq, vid))
    seen, worst, wrong = [], 0.0, []
    out = torch.empty(E_T, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for i, (b, mq, vid) in enumerate(cases):
            layer, tpl = tpls[i % 2]
            st, ref_big = tpl
            q, idx, act, seq, cu, big, ref_small = _e_band_case(
                tpl, mq, b, rng, gen, env_pools_e, layer
            )
            n = q.shape[0]
            torch.cuda.synchronize()
            torch.cuda._sleep(10)
            op(
                q,
                st.swa,
                st.comp,
                ws,
                idx,
                act,
                seq,
                st.sinks,
                SCALE,
                1.0,
                cum_seq_lens_q=cu,
                max_q_len=mq,
                out=out[:n],
            )
            torch.cuda._sleep(10)
            atol = env_pools_e["atol"][layer]
            e = _excess(out[big], ref_big[:mq], atol)
            if ref_small is not None:
                o_small = torch.cat([out[: big.start], out[big.stop : n]])
                e = max(e, _excess(o_small, ref_small, atol))
            worst = max(worst, e)
            if e > 1.0:
                wrong.append((b, mq, layer, e))
        torch.cuda.synchronize()
    ev = sorted(
        (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda e: e.time_range.start,
    )
    open_call, cur = False, []
    for e in ev:
        if "spin_kernel" in e.name or "sleep" in e.name:
            if open_call:
                seen.append(_variant_of(cur))
            open_call, cur = not open_call, []
        elif open_call:
            cur.append(e.name)
    assert len(seen) == len(cases), (len(seen), len(cases))
    mism = [
        (c, E_VARLEN_KERNELS[c[2]], s) for c, s in zip(cases, seen) if E_VARLEN_KERNELS[c[2]] != s
    ]
    used = {c[2] for c in cases}
    print(
        f"e-bands H{num_heads} W{width}: {len(cases)} calls, B 1..{E_B}, {len(used)} variants, "
        f"max excess {worst:.3f}"
    )
    assert not mism, mism[:3]
    assert not wrong, wrong[:3]


# ---------------------------------------------------------------------------------------------
# The native-storage read at the engine envelope. Under native KV storage the target hands
# this call bf16 read buffers: the compact window buffer [max_num_tokens * 128, 512] (query t's
# window column c at row t * 128 + c, a -1 column as a zero row), and a compressed buffer in one of
# two forms, whichever holds fewer rows:
# - per request (b * L_r + e, the form of the test_kvq_* cases above): seqs x cols x 128 rows;
# - per query: query t's top-k column k at row t * 512 + k of a [max_num_tokens * 512, 512] buffer,
#   viewed per ratio as [max_num_tokens * 512 / (128 / r), 1, 128 / r, 512]. The index table's
#   compressed half is then the constant arange(max_num_tokens * 512).view(max_num_tokens, 512)[:T]:
#   only the columns below sparse_topk_lens are written each step, and every column past it names a
#   finite row (zero at allocation, a previous step's values after).
# At max_num_tokens 16384 the per-query buffer is 8,388,608 rows (65536 pages of 128 at ratio 1,
# 131072 of 64 at ratio 2) and the window buffer 2,097,152. Each read below is held BITWISE to the
# container read of the same step (int16 view: signed zeros count); the container read is held to
# the fp32 reference.
# ---------------------------------------------------------------------------------------------
E_PQ_ROWS = E_T * INDEX_TOPK


class _PerQueryBuffers:
    """The target's native read buffers at max_num_tokens 16384 in the per-query form, prefilled with
    finite random rows (a previous step's content), with the constant renumbered index table."""

    def __init__(self, gen):
        self.swa_c = torch.randn(
            E_T * WINDOW, HEAD_DIM, generator=gen, device=DEV, dtype=torch.bfloat16
        )
        self.comp_q = torch.empty(E_PQ_ROWS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        for s0 in range(0, E_PQ_ROWS, 1 << 20):
            self.comp_q[s0 : s0 + (1 << 20)] = torch.randn(
                1 << 20, HEAD_DIM, generator=gen, device=DEV, dtype=torch.bfloat16
            )
        self.swa_c4 = self.swa_c.view(E_T, 1, WINDOW, HEAD_DIM)
        self.comp_q4 = {
            r: self.comp_q.view(E_PQ_ROWS // (128 // r), 1, 128 // r, HEAD_DIM) for r in (1, 2)
        }
        win = torch.arange(E_T * WINDOW, dtype=torch.int32, device=DEV).view(E_T, WINDOW)
        topk = torch.arange(E_PQ_ROWS, dtype=torch.int32, device=DEV).view(E_T, INDEX_TOPK)
        self.idx = {0: win, 1: torch.cat([win, topk], 1).contiguous()}
        self.idx[2] = self.idx[1]
        self.dummy = torch.zeros(1, 1, 64, HEAD_DIM, dtype=torch.bfloat16, device=DEV)

    def fill(self, swa, comp, idx, active):
        """What the target's window and top-k dequantizers write for a step whose container index is
        `idx`: row t * 128 + c of swa_c for every query t and window column c (a -1 column as +0.0),
        and row t * 512 + k of comp_q for every compressed column k < active[t] - 128 (the index op
        packs a row's valid entries first; asserted). Rows of the masked columns keep what they held."""
        T = idx.shape[0]
        w = idx[:, :WINDOW].long()
        rows = swa.view(-1, HEAD_DIM)[w.clamp_min(0)]
        rows.masked_fill_((w < 0)[..., None], 0.0)
        self.swa_c[: T * WINDOW] = rows.view(T * WINDOW, HEAD_DIM)
        del rows
        if idx.shape[1] > WINDOW:
            c = idx[:, WINDOW:].long()
            valid = torch.arange(INDEX_TOPK, device=DEV)[None] < (active.long() - WINDOW)[:, None]
            assert bool((c[valid] >= 0).all()) and bool((c[~valid] < 0).all())
            dst = torch.arange(T * INDEX_TOPK, device=DEV).view(T, INDEX_TOPK)[valid]
            self.comp_q[dst] = comp.view(-1, HEAD_DIM)[c[valid]]

    def views(self, ratio, T):
        """(swa pool, compressed pool, index) the target passes in the per-query form."""
        return self.swa_c4, (self.comp_q4[ratio] if ratio else self.dummy), self.idx[ratio][:T]


@pytest.fixture(scope="module")
def pq_buffers():
    bufs = _PerQueryBuffers(torch.Generator(device=DEV).manual_seed(2_300_000))
    yield bufs
    del bufs
    torch.cuda.empty_cache()


def _pq_read(st, bufs, ws, out=None):
    """The step `st` (a _Step on env_pools_e) read in the per-query form."""
    ratio = ENV_LAYERS[st.layer][0]
    bufs.fill(st.swa, st.comp, st.idx, st.active)
    swa, comp, idx = bufs.views(ratio, st.T)
    return st.run(ws, out=out, swa=swa, comp=comp, idx=idx)


def _attended_max(idx, active, counts, cols):
    """The largest index an attended column in `cols` names."""
    col = torch.arange(idx.shape[1], device=DEV)
    att = (col[None] < active[:, None].long()) & (
        (col[None] >= WINDOW) | (col[None] < counts[:, None])
    )
    return int(torch.where(att[:, cols], idx[:, cols], -1).max())


E_PQ_FAR = {
    # one context request of 16384 tokens ending at position 1,048,589 (the DSpark manager length)
    "ctx": ([(E_LONG_CACHED, E_T)], [], 1),
    # a mixed step of 16384 rows: two contexts, then 256 - 2 verify requests up to kv length Lm + 5
    "mixed": (
        [(E_LONG_CACHED, E_T - 6 * 254 - 700), (0, 700)],
        [E_LM - 1 - (i % 7) for i in range(254)],
        6,
    ),
}


@pytest.mark.parametrize("case", list(E_PQ_FAR))
@pytest.mark.parametrize("layer", list(ENV_LAYERS))
@pytest.mark.parametrize("num_heads", ALL_HEADS)
def test_e_native_per_query_far_end(env_pools_e, pq_buffers, num_heads, layer, case):
    """16384-row steps in the per-query form: the last query's attended columns name the window
    buffer's last row (2,097,151) and, at ratio 1 / 2, the per-query buffer's last row (8,388,607).
    The per-query read is bitwise the container read, which is inside the gate."""
    ctx, dec, next_n = E_PQ_FAR[case]
    gen = torch.Generator(device=DEV).manual_seed(
        2_310_000 + num_heads + 10 * ENV_LAYERS[layer][0] + len(dec)
    )
    st = _e_step(ctx, dec, num_heads, layer, env_pools_e, gen, next_n)
    assert st.T == E_T, st.T
    ws = _workspace()
    o_pool = st.run(ws)
    o_pq = _pq_read(st, pq_buffers, ws)
    ratio = ENV_LAYERS[layer][0]
    _, _, idx = pq_buffers.views(ratio, st.T)
    assert _attended_max(idx, st.active, st.counts, slice(0, WINDOW)) == E_T * WINDOW - 1
    if ratio:
        assert _attended_max(idx, st.active, st.counts, slice(WINDOW, None)) == E_PQ_ROWS - 1
    assert torch.equal(_bits(o_pq), _bits(o_pool)), (num_heads, layer, case)
    e = _excess(o_pool, st.reference(), env_pools_e["atol"][layer])
    print(
        f"per-query far end H{num_heads} {layer} {case}: T {st.T} B {st.B}, bitwise = container, excess {e:.3f}"
    )
    assert e <= 1.0, e


@pytest.mark.parametrize("layer", ["r2", "r1"])
@pytest.mark.parametrize("num_heads", ALL_HEADS)
def test_e_native_per_query_generation_cuda_graph(env_pools_e, pq_buffers, num_heads, layer):
    """The generation call in the per-query form captured at every G of GSET x next_n 1 and 6: q,
    lengths and seq_lens are prefixes of step buffers, the index is the constant renumbered table's
    prefix, the output is allocated inside the capture, and one eager call comes first. The first
    replay equals the eager call bitwise; three more on new steps (new buffer rows, q, lengths and
    seq_lens; kv lengths up to Lm + 5 = 1,048,595) each equal a fresh eager per-query call AND the
    container read bitwise, and the container read matches the reference."""
    ratio = ENV_LAYERS[layer][0]
    ws = _workspace()
    sinks = _sinks(num_heads)
    lens_buf = torch.full((E_T,), WINDOW, dtype=torch.int32, device=DEV)
    kvl_buf = torch.full((E_B + 1,), 1, dtype=torch.int32, device=DEV)
    q_buf = torch.zeros(VERIFY_Q * E_B, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    n_graphs, worst = 0, 0.0
    for next_n in (1, VERIFY_Q):
        long = _cached_positions(next_n)
        for G in GSET:
            T = G * next_n
            gen = torch.Generator(device=DEV).manual_seed(
                2_320_000 + 1000 * G + 100 * next_n + num_heads + ratio
            )
            rng = random.Random(2_330_000 + 1000 * G + 100 * next_n + num_heads + ratio)
            swa, comp, idx = pq_buffers.views(ratio, T)

            def make(step):
                pos = [
                    long[(i + step) % len(long)] if rng.random() < 0.5 else rng.randint(0, E_LM - 1)
                    for i in range(G)
                ]
                st = _e_step([], pos, num_heads, layer, env_pools_e, gen, next_n)
                pq_buffers.fill(st.swa, st.comp, st.idx, st.active)
                q_buf[:T].copy_(st.q3)
                lens_buf[:T].copy_(st.active)
                kvl_buf[:G].copy_(st.kv_lens[:G])
                return st

            def call(out):
                return op(
                    q_buf[:T].view(G, next_n, num_heads, HEAD_DIM),
                    swa,
                    comp,
                    ws,
                    idx,
                    lens_buf[:T],
                    kvl_buf[:G],
                    sinks,
                    SCALE,
                    1.0,
                    out=out,
                )

            shape = (G, next_n, num_heads, HEAD_DIM)
            st = make(0)
            eager = call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
            assert torch.equal(_bits(eager.view(T, num_heads, HEAD_DIM)), _bits(st.run(ws)))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                o = torch.empty(T, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
                out = call(o.view(shape))
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(_bits(out), _bits(eager)), (G, next_n)
            prev = eager.clone()
            for it in range(1, 4):
                st = make(it)
                out.fill_(float("nan"))
                graph.replay()
                fresh = call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
                pool = st.run(ws)
                torch.cuda.synchronize()
                assert torch.equal(_bits(out), _bits(fresh)), (G, next_n, it, "replay != eager")
                assert torch.equal(_bits(out.view(T, num_heads, HEAD_DIM)), _bits(pool)), (
                    G,
                    next_n,
                    it,
                )
                assert not torch.equal(_bits(out), _bits(prev)), (G, next_n, it)
                e = _excess(pool, st.reference(), env_pools_e["atol"][layer])
                assert e <= 1.0, (G, next_n, it, e)
                worst = max(worst, e)
                prev = out.clone()
            del graph
            n_graphs += 1
    print(
        f"per-query capture H{num_heads} {layer}: {n_graphs} graphs x 4 replays, replay = eager = container, "
        f"max excess {worst:.3f}"
    )


# The per-request form at the envelope corner where it is the smaller one: max_num_tokens 16384,
# 256 sequences, the DSpark page-table width 132 (max_seq_len 16384): 4,325,376 rows, request b's
# entry e at row b * L_r + e. Steps are the test_kvq_* steps (_KvqStep) on env_pools_e.
E_KVQ_CFG = (E_T, E_B)


@pytest.fixture(scope="module")
def e_kvq_buffers():
    bufs = _KvqBuffers(*E_KVQ_CFG, torch.Generator(device=DEV).manual_seed(2_400_000))
    yield bufs
    del bufs
    torch.cuda.empty_cache()


def _e_kvq_case(seed):
    """_kvq_case at (16384, 256): k in 1..256 requests, prefill-only / generation-only / mixed,
    next_n 1 or 6, a quarter of generation requests in the DSpark tail up to 16397, contexts fresh or
    cached (cached + q_len <= 16384), a quarter of cached contexts with a window floor. Seeds below
    1000 run H 16 / 64, seeds from 1000 on the members' H 8 / 32."""
    rng = random.Random(2_500_000 + seed)
    num_heads = ((16, 64) if seed < 1000 else MEMBER_HEADS)[seed % 2]
    layer = ("r0", "r2", "r1")[(seed // 2) % 3]
    max_t, seqs = E_KVQ_CFG
    k = (
        rng.randint(1, seqs)
        if rng.random() < 0.5
        else int(round(math.exp(rng.uniform(0.0, math.log(seqs)))))
    )
    next_n = rng.choice((1, 6))
    kind = rng.random()
    n_gen = 0 if kind < 0.3 else (k if kind < 0.5 else rng.randint(1, k))
    n_ctx = k - n_gen
    ctx = []
    if n_ctx:
        budget = max_t - n_gen * next_n
        if rng.random() < 0.5:
            t = int(round(math.exp(rng.uniform(0.0, math.log(budget)))))
        else:
            t = rng.randint(n_ctx, budget)
        t = min(max(t, n_ctx), budget)
        cuts = sorted(rng.sample(range(1, t), n_ctx - 1)) if n_ctx > 1 else []
        q_lens = [b - a for a, b in zip([0] + cuts, cuts + [t])]
        ctx = [(_fresh_or_cached(rng, q), q) for q in q_lens]
    top = KVQ_TAIL + 1 - next_n
    gen = [
        rng.randint(ENV_SEQ - next_n, top) if rng.random() < 0.25 else rng.randint(0, top)
        for _ in range(n_gen)
    ]
    floors = [
        rng.randint(max(1, c - WINDOW + 1), c) if c and rng.random() < 0.25 else 0 for c, _ in ctx
    ]
    return num_heads, layer, ctx, gen, next_n, floors


E_KVQ_SEEDS = list(range(60))
# The first 60-seed block from 1000 on (in steps of 60) whose steps reach k >= 250 and T >= 16000.
E_MEMBER_KVQ_SEEDS = list(range(1240, 1300))
E_KVQ_SETS = {"h16_h64": (E_KVQ_SEEDS, (16, 64)), "h8_h32": (E_MEMBER_KVQ_SEEDS, MEMBER_HEADS)}


@pytest.mark.parametrize(
    "seed", E_KVQ_SEEDS + E_MEMBER_KVQ_SEEDS, ids=_seed_ids(E_KVQ_SEEDS + E_MEMBER_KVQ_SEEDS)
)
def test_e_native_per_request_read(env_pools_e, e_kvq_buffers, seed):
    """60 random steps of up to 256 requests and 16384 rows on the per-request buffers at (16384,
    256) at H 16 / 64, and 60 more at H 8 / 32: the compact read is bitwise the container read,
    which is inside the gate."""
    num_heads, layer, ctx, gen_pos, next_n, floors = _e_kvq_case(seed)
    g = torch.Generator(device=DEV).manual_seed(2_600_000 + seed)
    st = _KvqStep(ctx, gen_pos, next_n, num_heads, layer, env_pools_e, e_kvq_buffers, g, floors)
    ws = _workspace()
    st.fill()
    o_pool = st.run(ws, compact=False)
    o_comp = st.run(ws, compact=True)
    assert torch.equal(_bits(o_comp), _bits(o_pool)), seed
    e = _excess(o_pool, st.reference(), env_pools_e["atol"][layer])
    print(
        f"per-request {seed}: H{num_heads} {layer} T {st.T} tc {st.tc} nc {st.nc} G {st.G} next_n {next_n} "
        f"attended -1 {st.attended_minus_one()} bitwise = container, excess {e:.3f}"
    )
    assert e <= 1.0, (seed, e)


@pytest.mark.parametrize("seeds", list(E_KVQ_SETS))
def test_e_native_per_request_reach(seeds):
    """Each set of 60 steps reaches k 256 and >= 16000 rows, both next_n, floored contexts and every
    (H, layer) of its head counts."""
    seed_list, heads = E_KVQ_SETS[seeds]
    cases = [_e_kvq_case(s) for s in seed_list]
    ks = [len(c[2]) + len(c[3]) for c in cases]
    ts = [sum(q for _, q in c[2]) + len(c[3]) * c[4] for c in cases]
    assert max(ks) >= 250 and max(ts) >= 16000, (max(ks), max(ts))
    assert {c[4] for c in cases if c[3]} == {1, 6} and any(any(c[5]) for c in cases)
    assert {(c[0], c[1]) for c in cases} == {(h, layer) for h in heads for layer in ENV_LAYERS}


def _e_kvq_far_steps():
    """_kvq_far_steps at (16384, 256): T = 16384 with 256 requests -- one long context, 254 3-token
    contexts spread over positions 131..16383 (every request stays inside max_seq_len 16384, so its
    entries stay inside its L_r rows), and a DSpark verify ending at 16397 -- without and with window
    floors on every cached context; and a verify-only step of 256 requests at the tail."""
    max_t, seqs = E_KVQ_CFG
    n_ctx = seqs - 1
    rest = max_t - 6 - 3 * (n_ctx - 1)
    ctx = [(ENV_SEQ - rest, rest)] + [
        (131 + (97 * i) % (ENV_SEQ - 134), 3) for i in range(n_ctx - 1)
    ]
    assert all(c + q <= ENV_SEQ for c, q in ctx)
    floors = [c - 40 for c, _ in ctx]
    verify = [KVQ_TAIL - 5 - (i % 3) for i in range(seqs)]
    return [(ctx, [KVQ_TAIL - 5], 6, None), (ctx, [KVQ_TAIL - 5], 6, floors), ([], verify, 6, None)]


def test_e_native_per_request_far_end(env_pools_e, e_kvq_buffers):
    """_e_kvq_far_steps, H 8 / 16 / 32 / 64 x every layer: the window half names the window buffer's
    last row, the compressed half row 255 * L_r + 16398 // r - 1 (asserted); the compact read is
    bitwise the container read (with floored windows among them), inside the gate."""
    bufs = e_kvq_buffers
    max_t, seqs = E_KVQ_CFG
    ws = _workspace()
    worst, n = 0.0, 0
    for i, (ctx, gen_pos, next_n, floors) in enumerate(_e_kvq_far_steps()):
        for num_heads in ALL_HEADS:
            for layer in ENV_LAYERS:
                g = torch.Generator(device=DEV).manual_seed(
                    2_700_000 + 100 * i + num_heads + ENV_LAYERS[layer][0]
                )
                st = _KvqStep(ctx, gen_pos, next_n, num_heads, layer, env_pools_e, bufs, g, floors)
                if i < 2:
                    assert (
                        st.T == max_t
                        and st.B == seqs
                        and int(st.cidx[:, :WINDOW].max()) == max_t * WINDOW - 1
                    )
                    if st.ratio:
                        far = (seqs - 1) * KVQ_L[st.ratio] + (KVQ_TAIL + 1) // st.ratio - 1
                        assert int(st.cidx[:, WINDOW:].max()) == far
                st.fill()
                o_pool = st.run(ws, compact=False)
                o_comp = st.run(ws, compact=True)
                assert torch.equal(_bits(o_comp), _bits(o_pool)), (i, num_heads, layer)
                e = _excess(o_pool, st.reference(), env_pools_e["atol"][layer])
                assert e <= 1.0, (i, num_heads, layer, e)
                worst, n = max(worst, e), n + 1
    print(
        f"per-request far end {E_KVQ_CFG}: {n} steps, compact == container bitwise, max excess {worst:.3f}"
    )


@pytest.mark.parametrize("layer", list(ENV_LAYERS))
@pytest.mark.parametrize("num_heads", ALL_HEADS)
def test_e_native_per_request_generation_cuda_graph(env_pools_e, e_kvq_buffers, num_heads, layer):
    """The generation call on the per-request buffers at (16384, 256) captured at every G of GSET x
    next_n 1 and 6, as test_kvq_compact_read_cuda_graph_capture_set does it on the (8192, 32)
    buffers: replay = fresh eager compact call = container read, bitwise, inside the gate."""
    ratio = ENV_LAYERS[layer][0]
    bufs = e_kvq_buffers
    max_t, seqs = E_KVQ_CFG
    ws = _workspace()
    sinks = _sinks(num_heads)
    width = WINDOW + (INDEX_TOPK if ratio else 0)
    idx_buf = torch.full((max_t, width), -1, dtype=torch.int32, device=DEV)
    lens_buf = torch.full((max_t,), WINDOW, dtype=torch.int32, device=DEV)
    kvl_buf = torch.full((seqs + 1,), 1, dtype=torch.int32, device=DEV)
    q_buf = torch.zeros(VERIFY_Q * seqs, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    comp = bufs.comp_c4[ratio] if ratio else bufs.dummy
    n_graphs, worst = 0, 0.0
    for next_n in (1, VERIFY_Q):
        for G in GSET:
            T = G * next_n
            gen = torch.Generator(device=DEV).manual_seed(
                2_800_000 + 1000 * G + 100 * next_n + num_heads + ratio
            )
            rng = random.Random(2_900_000 + 1000 * G + 100 * next_n + num_heads + ratio)
            idx = bufs.win_idx[:T] if not ratio else idx_buf[:T]

            def make(step):
                pos = [rng.randint(0, KVQ_TAIL + 1 - next_n) for _ in range(G)]
                pos[0] = (KVQ_TAIL if step % 2 == 0 else ENV_SEQ - 1) + 1 - next_n
                st = _KvqStep([], pos, next_n, num_heads, layer, env_pools_e, bufs, gen)
                st.fill()
                q_buf[:T].copy_(st.q3)
                if ratio:
                    idx_buf[:T].copy_(st.cidx)
                lens_buf[:T].copy_(st.active)
                kvl_buf[:G].copy_(st.kv_lens[:G])
                return st

            def call(out):
                return op(
                    q_buf[:T].view(G, next_n, num_heads, HEAD_DIM),
                    bufs.swa_c4,
                    comp,
                    ws,
                    idx,
                    lens_buf[:T],
                    kvl_buf[:G],
                    sinks,
                    SCALE,
                    1.0,
                    out=out,
                )

            shape = (G, next_n, num_heads, HEAD_DIM)
            st = make(0)
            eager = call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
            assert torch.equal(
                _bits(eager.view(T, num_heads, HEAD_DIM)), _bits(st.run(ws, compact=False))
            )
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                o = torch.empty(T, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
                out = call(o.view(shape))
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(_bits(out), _bits(eager)), (G, next_n)
            prev = eager.clone()
            for it in range(1, 4):
                st = make(it)
                out.fill_(float("nan"))
                graph.replay()
                fresh = call(torch.empty(shape, dtype=torch.bfloat16, device=DEV))
                pool = st.run(ws, compact=False)
                torch.cuda.synchronize()
                assert torch.equal(_bits(out), _bits(fresh)), (G, next_n, it, "replay != eager")
                assert torch.equal(_bits(out.view(T, num_heads, HEAD_DIM)), _bits(pool)), (
                    G,
                    next_n,
                    it,
                )
                assert not torch.equal(_bits(out), _bits(prev)), (G, next_n, it)
                e = _excess(pool, st.reference(), env_pools_e["atol"][layer])
                assert e <= 1.0, (G, next_n, it, e)
                worst = max(worst, e)
                prev = out.clone()
            del graph
            n_graphs += 1
    print(
        f"per-request capture (16384, 256) H{num_heads} {layer}: {n_graphs} graphs x 4 replays, replay = eager = "
        f"container, max excess {worst:.3f}"
    )


# ---------------------------------------------------------------------------------------------
# Envelope E2': decoder bounded replay (on by default at tep4 / tp4). Layers 21-39 run a decoder pass
# whose context call holds the finishing requests' 128-row tails -- up to 256 requests of at most 128
# rows each, so up to 32768 rows, beyond max_num_tokens -- and whose generation call follows on the
# generation rows. To the kernel a tail is a context chunk of <= 128 rows at a cached prefix (with a
# floor-adjusted seq_lens it is one with fewer cached rows).
# ---------------------------------------------------------------------------------------------
E_TD = WINDOW * E_B


def _e_replay_case(seed):
    """Seeds 0..5: the full decoder pass, 256 tails of 128 rows (32768 rows), H 16 / 64 x ratio 0 / 2 /
    1. Seeds 6..35: k in 1..256 tails of 1..128 rows (a third of them exactly 128), each ending at a
    random kv length up to Lm (a quarter at Lm) for a random L with DSpark on or off; for half of them
    a mixed pass with G generation requests after the tails (k + G <= 256) at next_n 6 (DSpark on) or 1.
    Seeds 1000..1035 are the same 36 shapes' kinds at the members' H 8 / 32 (new random draws)."""
    rng = random.Random(3_000_000 + seed)
    num_heads = ((16, 64) if seed < 1000 else MEMBER_HEADS)[seed % 2]
    seed %= 1000
    layer = ("r0", "r2", "r1")[(seed // 2) % 3]
    L = E_L[rng.randrange(len(E_L))]
    dspark = rng.random() < 0.5
    lm = L + 14 if dspark else L
    next_n = 6 if dspark else 1
    if seed < 6:
        q_lens = [WINDOW] * E_B
    else:
        k = (
            rng.randint(1, E_B)
            if rng.random() < 0.5
            else int(round(math.exp(rng.uniform(0.0, math.log(E_B)))))
        )
        q_lens = [WINDOW if rng.random() < 1 / 3 else rng.randint(1, WINDOW) for _ in range(k)]
    ctx = []
    for q in q_lens:
        kv = lm if rng.random() < 0.25 else rng.randint(q, lm)
        ctx.append((kv - q, q))
    dec = []
    if seed >= 6 and seed % 2 and len(ctx) < E_B:
        g = rng.randint(1, E_B - len(ctx))
        top = lm - 1 if dspark else L - 1
        dec = [top if rng.random() < 0.2 else rng.randint(0, top) for _ in range(g)]
    return num_heads, layer, ctx, dec, next_n, L, dspark


E_REPLAY_SEEDS = list(range(36))
E_MEMBER_REPLAY_SEEDS = list(range(1000, 1036))
E_REPLAY_SETS = {
    "h16_h64": (E_REPLAY_SEEDS, (16, 64)),
    "h8_h32": (E_MEMBER_REPLAY_SEEDS, MEMBER_HEADS),
}


@pytest.mark.parametrize(
    "seed",
    E_REPLAY_SEEDS + E_MEMBER_REPLAY_SEEDS,
    ids=_seed_ids(E_REPLAY_SEEDS + E_MEMBER_REPLAY_SEEDS),
)
def test_e_decoder_replay_rows_to_32768(env_pools_e, seed):
    """E2': the decoder pass's context call at up to 32768 rows in up to 256 requests of <= 128 rows
    (and its generation call after it), against the fp32 reference. For a context-only pass the
    kernels are read back from the profiler and equal E_VARLEN_BANDS' cell for (H, width, B,
    max_q_len)."""
    num_heads, layer, ctx, dec, next_n, L, dspark = _e_replay_case(seed)
    gen = torch.Generator(device=DEV).manual_seed(3_100_000 + seed)
    st = _e_step(ctx, dec, num_heads, layer, env_pools_e, gen, next_n)
    ws = _workspace()
    seen = None
    if dec:
        o = st.run(ws)
    else:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            torch.cuda.synchronize()
            torch.cuda._sleep(10)
            o = st.run(ws)
            torch.cuda._sleep(10)
            torch.cuda.synchronize()
        ev = sorted(
            (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
            key=lambda e: e.time_range.start,
        )
        names, inside = [], False
        for e in ev:
            if "spin_kernel" in e.name or "sleep" in e.name:
                inside = not inside
            elif inside:
                names.append(e.name)
        seen = _variant_of(names)
        width = st.width
        band = [
            v for lo, hi, v in E_VARLEN_BANDS[(num_heads, width)][st.nc] if lo <= st.max_ctx_q <= hi
        ]
        assert seen == E_VARLEN_KERNELS[band[0]], (seed, seen, band)
    e = _excess(o, st.reference(), env_pools_e["atol"][layer])
    print(
        f"decoder-pass {seed}: H{num_heads} {layer} L {L} dspark {int(dspark)} T {st.T} tc {st.tc} nc {st.nc} "
        f"max_q_len {st.max_ctx_q} G {st.G} next_n {next_n} max kv {int(st.kv_lens[: st.B].max())} "
        f"kernels {seen} excess {e:.3f}"
    )
    assert torch.isfinite(o).all(), seed
    assert e <= 1.0, (seed, e)


@pytest.mark.parametrize("seeds", list(E_REPLAY_SETS))
def test_e_decoder_replay_cases_reach_the_shape(seeds):
    """Each set of 36 passes reaches 32768 rows in 256 tails of 128 at every (H, layer) of its head
    counts, mixed passes at next_n 1 and 6, and tails ending at Lm for the longest L."""
    seed_list, heads = E_REPLAY_SETS[seeds]
    cases = [_e_replay_case(s) for s in seed_list]
    ts = [sum(q for _, q in c[2]) for c in cases]
    assert max(ts) == E_TD and all(max(q for _, q in c[2]) <= WINDOW for c in cases)
    full = {(c[0], c[1]) for c, t in zip(cases, ts) if t == E_TD}
    assert full == {(h, layer) for h in heads for layer in ENV_LAYERS}, full
    assert {c[4] for c in cases if c[3]} == {1, 6}
    assert {(c[0], c[1]) for c in cases} == {(h, layer) for h in heads for layer in ENV_LAYERS}
    assert any(cached + q == E_LM for c in cases for cached, q in c[2])


# ---------------------------------------------------------------------------------------------
# The workspace over the engine envelope. The target passes one persistent 128 MiB workspace
# (_ATTN_WS_BYTES, zeroed once) to every call, the drafter's included. FlashInfer hands the kernel
# the whole buffer as the multi-CTA split-KV scratch -- per-CTA max/sum statistics ([SMs x 64]
# float2) followed by per-CTA partial O ([CTAs x 64 x headDimPerCta] bf16) -- and checks no size
# (the multi-CTA semaphores live in a separate buffer FlashInfer allocates per call). The kernel
# splits the KV only while the unsplit grid is smaller than the SM count, and never into more CTAs
# than there are SMs, so the scratch does not grow with G, next_n, H, the query rows, the number of
# requests or the kv length: at most SMs x 64 x (8 + 512 x 2) bytes, 10,039,296 on 152 SMs.
# ---------------------------------------------------------------------------------------------
WS_FILLS = (0xA5, 0x5A)


def _ws_probe_calls():
    """(label, kind, H, width, B, rows-per-request or q_lens) for every call shape the test runs."""
    calls = []
    for g in range(1, E_B + 1):
        for nn in (1, VERIFY_Q):
            for h in ALL_HEADS:
                for w in (128, 640):
                    calls.append(("dense", h, w, g, nn))
        calls.append(("draft", DS_HEADS, DS_WIDTH, g, 1))
    for h in ALL_HEADS:
        for w in (128, 640):
            for mq in list(range(1, 161)) + [512, 2048, 8192, 8193, 12288, E_T]:
                calls.append(("varlen", h, w, 1, [mq]))
            for b in range(2, E_B + 1):
                for mq in (1, 2, 3, 4, 6, 8, 12, 13, 19, 20, 25, 26, 38, 39, 76, 77):
                    if b * mq <= E_T:
                        calls.append(("varlen", h, w, b, [mq] * b))
                if b in GSET:
                    calls.append(("varlen", h, w, b, [E_T - (b - 1)] + [1] * (b - 1)))
            calls.append(("varlen", h, w, E_B, [WINDOW] * E_B))  # E2': the 32768-row decoder pass
    return calls


def test_workspace_high_water_mark_over_the_envelope():
    """The 128 MiB workspace covers the engine envelope, and its content does not matter.

    Every call below runs three times, on the workspace filled with 0xA5, with 0x5A and with zeros;
    the high-water mark is the end of the last 8-byte word that differs from the fill. Shapes: dense
    decode (next_n 1) and verify (next_n 6) at every G 1..256, H 8 / 16 / 32 / 64, width 128 / 640
    (the target's generation call); the drafter's [5G, 1, 64, 512] at width 136 for every G 1..256; the
    context call (H 8 / 16 / 32 / 64) at one request of every max_q_len 1..160 and up to 16384, at
    every B 2..256 with B requests of each max_q_len at the kernel's split-KV edges, B in GSET at the
    16384-row top, and the 32768-row decoder pass (256 requests of 128 rows).
    kv lengths are 1,048,595 (decode, verify, context) and 129 (drafter)."""
    sm = torch.cuda.get_device_properties(DEV).multi_processor_count
    bound = sm * 64 * (8 + HEAD_DIM * 2)
    gen = torch.Generator(device=DEV).manual_seed(5150)
    swa = torch.randn(600, 1, SWA_PAGE, HEAD_DIM, generator=gen, device=DEV).to(torch.bfloat16)
    comp = torch.randn(600, 1, 64, HEAD_DIM, generator=gen, device=DEV).to(torch.bfloat16)
    rows_max = E_T * 2
    q_buf = torch.randn(rows_max, 64, HEAD_DIM, generator=gen, device=DEV).to(torch.bfloat16)
    idx_buf = {
        w: torch.randint(0, 600 * 64, (rows_max, w), generator=gen, device=DEV, dtype=torch.int32)
        for w in (WINDOW, DS_WIDTH, WINDOW + INDEX_TOPK)
    }
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    zws = torch.zeros_like(ws)
    words = ws.view(torch.int64)
    word_idx = torch.arange(words.numel(), device=DEV)
    fill_word = {f: int.from_bytes(bytes([f]) * 8, "little", signed=True) for f in WS_FILLS}
    worst, touched, bad = (0, None), 0, []
    extremes, hw_by_heads = {}, {}
    for kind, h, w, b, per in _ws_probe_calls():
        if kind == "varlen":
            n, mq = sum(per), max(per)
            q = q_buf[:n, :h]
            cu = torch.tensor([0] + per, dtype=torch.int32, device=DEV).cumsum(0).int()
            seq = torch.tensor([1048595 - E_T + x for x in per], dtype=torch.int32, device=DEV)
        else:
            n, mq, cu = b * (DS_BLOCK if kind == "draft" else per), None, None
            rows = n if kind == "draft" else b
            q = q_buf[:n, :h].reshape(rows, n // rows, h, HEAD_DIM)
            seq = torch.full(
                (rows,), 129 if kind == "draft" else 1048595, dtype=torch.int32, device=DEV
            )
        act = torch.full((n,), DS_TOPK if kind == "draft" else w, dtype=torch.int32, device=DEV)
        idx = idx_buf[w][:n]
        sinks = _sinks(h)
        outs, hw = [], 0
        for f in WS_FILLS + (None,):
            if f is None:
                buf = zws
            else:
                ws.fill_(f)
                buf = ws
            outs.append(
                op(
                    q,
                    swa,
                    comp,
                    buf,
                    idx,
                    act,
                    seq,
                    sinks,
                    SCALE,
                    1.0,
                    cum_seq_lens_q=cu,
                    max_q_len=mq,
                )
            )
            if f is not None:
                diff = words != fill_word[f]
                if bool(diff.any()):
                    hw = max(hw, (int(torch.where(diff, word_idx, -1).max()) + 1) * 8)
        label = (kind, h, w, b, mq if kind == "varlen" else per)
        if not (
            torch.equal(outs[0], outs[1])
            and torch.equal(outs[0], outs[2])
            and bool(torch.isfinite(outs[0]).all())
        ):
            bad.append(label)
        touched += hw > 0
        if kind != "draft":
            hw_by_heads[h] = max(hw_by_heads.get(h, 0), hw)
        if hw > worst[0]:
            worst = (hw, label)
        if (kind, b) in (("dense", E_B), ("draft", E_B)) or (kind == "varlen" and n >= E_T):
            extremes[label] = hw
    print(
        f"workspace: {len(_ws_probe_calls())} call shapes, {touched} touched the workspace, high-water mark "
        f"{worst[0]} bytes ({worst[0] / 2**20:.3f} MiB) at {worst[1]}; bound {bound} on {sm} SMs; "
        f"envelope extremes {sorted(set(extremes.values()))}; high-water mark by H (trunk calls) {hw_by_heads}"
    )
    assert not bad, bad[:5]
    assert touched > 0 and worst[0] > 0  # the harness sees the scratch writes it measures
    assert worst[0] <= bound <= 128 * 1024 * 1024 // 8, (worst, bound)
    assert sorted(hw_by_heads) == list(ALL_HEADS) and all(v > 0 for v in hw_by_heads.values()), (
        hw_by_heads
    )
    assert set(extremes.values()) == {0}, {k: v for k, v in extremes.items() if v}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--guard-child":
        _guard_child(json.loads(sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == "--pool-child":
        _pool_child()
