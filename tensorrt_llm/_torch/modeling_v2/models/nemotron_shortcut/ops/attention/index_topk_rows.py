# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-attention key selection: each query row's `topk` best-scoring causal keys,
as the K and V pool rows the token-sparse attention kernel reads."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .index_pool_rows import index_pool_rows
from .index_scores import index_scores


class IndexTopkRows(OpWrapper):
    """For every row `r`: score request `row_req[r]`'s cached index keys at positions
    `[0, row_len[r])` against `q[r]` (``index_scores``), keep the `topk` highest
    (ties prefer earlier logical positions), and write their K and V pool rows in
    ascending logical order into ``out``. Without an output pool-row buffer, return the
    logical token indices. Short rows keep all causal keys and pad with -1.
    Caller-owned FP32 scores have a 16-byte-aligned row stride. Output and scratch
    buffers are supplied by the caller so generation can use CUDA graphs.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode rows, short and long histories",
            spec=dict(tpb=32, rows=(1, 1, 1), past=(100, 700, 5000)),
        ),
        Cell(
            why="short chunks from two requests", spec=dict(tpb=64, rows=(3, 3), past=(509, 3000))
        ),
        Cell(why="a prefill chunk crossing the top-k", spec=dict(tpb=64, rows=(200,), past=(400,))),
        Cell(
            why="prefill requests with different chunk and history lengths",
            spec=dict(tpb=32, rows=(130, 2), past=(450, 3000), prefill=True),
        ),
    )

    note = """
    Sparse attention reduces the selected K/V rows in selection order. Keeping that
    order stable across ranks and prefix-cache hits preserves the target logits.
    """

    def __call__(
        self,
        q: torch.Tensor,
        cache: torch.Tensor,
        offsets: torch.Tensor,
        slot_scale: int,
        layer_in_pool: int,
        row_req: torch.Tensor,
        row_len: torch.Tensor,
        tile_row0: torch.Tensor,
        tile_rows: torch.Tensor,
        logits: torch.Tensor,
        indices: torch.Tensor,
        out: torch.Tensor | None,
        splits: int = 1,
        block_m: int = 16,
    ) -> torch.Tensor:
        n = q.shape[0]
        index_scores(
            q,
            cache,
            offsets[:, 0],
            slot_scale,
            row_req,
            row_len,
            tile_row0,
            tile_rows,
            logits[:n],
            splits,
            block_m,
        )
        from tensorrt_llm._torch.kernels.blackwell.top_k import selfsampling_topk_run_prefill

        selfsampling_topk_run_prefill(
            logits[:n],
            torch.zeros_like(row_len),
            row_len,
            indices[:n],
            deterministic=True,
            sorting="ascending",
        )
        if out is None:
            return indices[:n]
        return index_pool_rows(indices[:n], row_req, offsets, layer_in_pool, cache.shape[1], out)

    def reference(
        self,
        q,
        cache,
        offsets,
        slot_scale,
        layer_in_pool,
        row_req,
        row_len,
        tile_row0,
        tile_rows,
        logits,
        indices,
        out,
        splits=1,
        block_m=16,
    ) -> torch.Tensor:
        """Score the causal prefix and break cutoff ties by logical position."""
        n, topk = q.shape[0], indices.shape[1]
        tpb = cache.shape[1]
        shape = (n, topk) if out is None else (1, n, 2, topk)
        res = torch.full(shape, -1, dtype=torch.int32, device=q.device)
        for r, (b, length) in enumerate(zip(row_req.tolist(), row_len.tolist())):
            pos = torch.arange(length, device=q.device)
            slots = offsets[b, 0, pos // tpb].long() // slot_scale
            scores = cache[slots, pos % tpb, 0].float() @ q[r].float()
            keep = pos if length <= topk else scores.argsort(descending=True, stable=True)[:topk]
            keep = keep.sort().values
            if out is None:
                res[r, : keep.numel()] = keep.int()
                continue
            page, within = keep // tpb, keep % tpb
            for kv in range(2):
                rows = (offsets[b, kv, page].long() + 2 * layer_in_pool) * tpb + within
                res[0, r, kv, : keep.numel()] = rows.to(torch.int32)
        return res

    def is_valid(
        self,
        q,
        cache,
        offsets,
        slot_scale,
        layer_in_pool,
        row_req,
        row_len,
        tile_row0,
        tile_rows,
        logits,
        indices,
        out,
        splits=1,
        block_m=16,
    ) -> None:
        n, topk = q.shape[0], indices.shape[1]
        assert (
            indices.dtype == torch.int32
            and indices.is_contiguous()
            and indices.shape[0] >= n
            and topk % 4 == 0
            and row_len.dtype == torch.int32
            and row_len.numel() == n
            and (out is None or tuple(out.shape) == (1, n, 2, topk))
            and offsets.dim() == 3
            and slot_scale > 0
            and logits.shape[0] >= n
            and logits.stride(1) == 1
            and logits.stride(0) % 4 == 0
        ), "IndexTopkRows received unsupported inputs"


index_topk_rows = IndexTopkRows()
