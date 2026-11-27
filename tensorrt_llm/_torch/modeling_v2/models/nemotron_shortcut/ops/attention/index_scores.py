# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score every causal key of a query row against its indexer query, over the paged index-key cache.

The sparse-attention indexer has one learned query head and one key head; a key's score
is their plain dot product -- no scale, activation or position embedding -- over the bf16
projections, accumulated in fp32. Each
query row then keeps its `index_topk` best-scoring causal keys; this op produces the scores the
selection reads, row-major fp32 over the request-local key positions `[0, row_len)`.

A program takes a tile of up to `BM` consecutive rows of one request -- the rows of a prefill
chunk, or a single decode query -- and every `splits`-th key tile of its
causal extent (the tile's last row), one page (or an aligned part of one) per key tile, so the
keys are read once per row tile with `tl.dot`. Positions past a row's own length are not written.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _index_scores_kernel(
    q,
    q_stride,
    cache,
    cache_slot_stride,
    cache_token_stride,
    offsets,
    offsets_stride,
    row_req,
    row_len,
    tile_row0,
    tile_rows,
    logits,
    logits_stride,
    slot_scale,
    num_splits,
    TPB: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
    D: tl.constexpr,
):
    t = tl.program_id(0)
    split = tl.program_id(1)
    row0 = tl.load(tile_row0 + t)
    nrows = tl.load(tile_rows + t)
    rows = row0 + tl.arange(0, BM)
    row_ok = tl.arange(0, BM) < nrows
    b = tl.load(row_req + row0).to(tl.int64)
    lens = tl.load(row_len + rows, mask=row_ok, other=0)
    extent = tl.load(row_len + row0 + nrows - 1)
    d = tl.arange(0, D)
    qt = tl.load(
        q + rows[:, None].to(tl.int64) * q_stride + d[None, :], mask=row_ok[:, None], other=0.0
    )
    n_local = tl.arange(0, BN)
    for kt in range(split, tl.cdiv(extent, BN), num_splits):
        n0 = kt * BN
        page_offset = tl.load(offsets + b * offsets_stride + n0 // TPB)
        slot = (page_offset // slot_scale).to(tl.int64)
        key_ptr = (
            cache
            + slot * cache_slot_stride
            + ((n0 % TPB) + n_local)[:, None] * cache_token_stride
            + d[None, :]
        )
        k = tl.load(key_ptr, mask=(n0 + n_local)[:, None] < extent, other=0.0)
        s = tl.dot(qt, tl.trans(k))
        cols = n0 + n_local
        tl.store(
            logits + rows[:, None].to(tl.int64) * logits_stride + cols[None, :],
            s,
            mask=row_ok[:, None] & (cols[None, :] < lens[:, None]),
        )


class IndexScores(OpWrapper):
    """In place: `logits[r, n] = sum_d q[r, d] * key(row_req[r], n)[d]` in fp32 for every
    `n < row_len[r]`, where `key(b, n)` is request b's cached index key at position n
    (page `n // tpb` at slot `k_offsets[b, n // tpb] // slot_scale` of `cache`). Returns
    `logits`; columns at or past a row's length are left as they were.

    `q` `[N, D]` bf16, unit last-dim stride. `cache` `[slots, tpb, 1, D]` bf16 (see
    ``index_key_store``), every position below a row's length written. `row_req` and
    `row_len` int32 `[N]`. The row tiles (`tile_row0`, `tile_rows`, int32 `[T]`) partition
    the rows into runs of at most `block_m` consecutive rows of one request, each run in
    increasing length order, so its last row bounds the run's keys. `logits` fp32 `[N, ld]`
    with `ld` at least every row's length. `splits` programs share a row tile's keys.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per request",
            spec=dict(tpb=32, rows=(1, 1, 1), past=(5, 600, 2100), splits=4),
        ),
        Cell(
            why="short chunks from two requests",
            spec=dict(tpb=64, rows=(3, 3), past=(700, 1500), splits=8),
        ),
        Cell(
            why="prefill chunks after a prefix",
            spec=dict(tpb=128, rows=(300, 70), past=(0, 900), splits=2),
        ),
    )

    note = """
    The dot runs on the tensor cores over bf16 operands with fp32 accumulation;
    every product of two bf16 values is exact in fp32, so the result differs
    from an fp32 reference only in summation order. A key tile never crosses a
    page: `block_n` divides the page size.
    """

    def __call__(
        self,
        q: torch.Tensor,
        cache: torch.Tensor,
        k_offsets: torch.Tensor,
        slot_scale: int,
        row_req: torch.Tensor,
        row_len: torch.Tensor,
        tile_row0: torch.Tensor,
        tile_rows: torch.Tensor,
        logits: torch.Tensor,
        splits: int = 1,
        block_m: int = 16,
    ) -> torch.Tensor:
        num_tiles = tile_row0.numel()
        if num_tiles == 0:
            return logits
        tpb = cache.shape[1]
        _index_scores_kernel[(num_tiles, splits)](
            q,
            q.stride(0),
            cache,
            cache.stride(0),
            cache.stride(1),
            k_offsets,
            k_offsets.stride(0),
            row_req,
            row_len,
            tile_row0,
            tile_rows,
            logits,
            logits.stride(0),
            slot_scale,
            splits,
            TPB=tpb,
            BN=min(tpb, 64),
            BM=block_m,
            D=q.shape[1],
            num_warps=4,
        )
        return logits

    def reference(
        self,
        q,
        cache,
        k_offsets,
        slot_scale,
        row_req,
        row_len,
        tile_row0,
        tile_rows,
        logits,
        splits=1,
        block_m=16,
    ) -> torch.Tensor:
        out = logits.clone()
        tpb = cache.shape[1]
        for r, (b, n) in enumerate(zip(row_req.tolist(), row_len.tolist())):
            pos = torch.arange(n, device=q.device)
            slots = k_offsets[b, pos // tpb].long() // slot_scale
            keys = cache[slots, pos % tpb, 0].float()
            out[r, :n] = keys @ q[r].float()
        return out

    def is_valid(
        self,
        q,
        cache,
        k_offsets,
        slot_scale,
        row_req,
        row_len,
        tile_row0,
        tile_rows,
        logits,
        splits=1,
        block_m=16,
    ) -> None:
        tpb = cache.shape[1]
        assert (
            q.dim() == 2
            and q.stride(1) == 1
            and q.dtype == cache.dtype == torch.bfloat16
            and cache.dim() == 4
            and cache.shape[2] == 1
            and cache.shape[3] == q.shape[1]
            and cache.stride(3) == 1
            and tpb & (tpb - 1) == 0
            and tpb >= 16
            and logits.dtype == torch.float32
            and logits.stride(1) == 1
            and int(row_len.max()) <= logits.shape[1]
            and int(tile_rows.max()) <= block_m
            and block_m >= 16
            and splits >= 1
        ), "IndexScores received unsupported inputs"


index_scores = IndexScores()
