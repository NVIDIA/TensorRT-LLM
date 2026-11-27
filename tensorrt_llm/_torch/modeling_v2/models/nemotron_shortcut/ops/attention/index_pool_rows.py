# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn each query row's selected request-local key positions into the K and V pool rows the
token-sparse attention kernel reads."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _index_pool_rows_kernel(
    indices,
    row_req,
    offsets,
    offsets_b_stride,
    offsets_kv_stride,
    out,
    layer_in_pool,
    TPB: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tok = tl.load(indices + r.to(tl.int64) * TOPK + j)
    valid = tok >= 0
    b = tl.load(row_req + r).to(tl.int64)
    page = tl.where(valid, tok // TPB, 0)
    k_page = tl.load(offsets + b * offsets_b_stride + page, mask=valid, other=0)
    v_page = tl.load(offsets + b * offsets_b_stride + offsets_kv_stride + page, mask=valid, other=0)
    within = tok % TPB
    k_row = (k_page + 2 * layer_in_pool) * TPB + within
    v_row = (v_page + 2 * layer_in_pool) * TPB + within
    base = out + r.to(tl.int64) * 2 * TOPK
    tl.store(base + j, tl.where(valid, k_row, -1))
    tl.store(base + TOPK + j, tl.where(valid, v_row, -1))


class IndexPoolRows(OpWrapper):
    """`out[0, r, 0, j]` / `out[0, r, 1, j]` = the K / V pool row of request `row_req[r]`'s key
    at position `indices[r, j]`, or -1 where `indices[r, j]` is -1. Returns `out`
    (int32 `[1, N, 2, topk]`, allocated when not given).

    A pool row is one token of one KV head, `head_dim` elements from the pool base: with one
    KV head and the `[kv heads, tokens per block, head dim]` page layout, block `p`'s K row of
    in-page token `t` is `(offsets[b, 0, p] + 2 * layer_in_pool) * tpb + t` -- the address
    `thop_attention` forms for a dense call, in rows -- and its V row the same with
    `offsets[b, 1, p]`. `offsets` is one pool's block offsets `[B, 2, max_blocks]`
    (`kv_cache_block_offsets[pool]`), `layer_in_pool` the layer's
    `host_kv_cache_pool_mapping[layer, 1]`. `indices` int32 `[N, topk]`, contiguous.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode rows with short and long histories",
            spec=dict(tpb=32, lens=(7, 600, 4000), layer=3),
        ),
        Cell(
            why="prefill rows of one request, 64-token pages",
            spec=dict(tpb=64, lens=tuple(range(500, 540)), layer=0),
        ),
    )

    note = """
    The conversion reads the device block offsets the step already holds, so it
    needs no host block table. The rows fit int32 for pools below 2^31 rows
    (1 TiB at 512-byte rows).
    """

    def __call__(
        self,
        indices: torch.Tensor,
        row_req: torch.Tensor,
        offsets: torch.Tensor,
        layer_in_pool: int,
        tokens_per_block: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n, topk = indices.shape
        if out is None:
            out = torch.empty(1, n, 2, topk, dtype=torch.int32, device=indices.device)
        if n == 0:
            return out
        block = 128 if topk % 128 == 0 else topk
        _index_pool_rows_kernel[(n, topk // block)](
            indices,
            row_req,
            offsets,
            offsets.stride(0),
            offsets.stride(1),
            out,
            layer_in_pool,
            TPB=tokens_per_block,
            TOPK=topk,
            BLOCK=block,
        )
        return out

    def reference(
        self, indices, row_req, offsets, layer_in_pool, tokens_per_block, out=None
    ) -> torch.Tensor:
        n, topk = indices.shape
        res = torch.full((1, n, 2, topk), -1, dtype=torch.int32, device=indices.device)
        for r in range(n):
            b = int(row_req[r])
            tok = indices[r].long()
            ok = tok >= 0
            page = torch.where(ok, tok // tokens_per_block, 0)
            within = tok % tokens_per_block
            for kv in range(2):
                rows = (offsets[b, kv, page].long() + 2 * layer_in_pool) * tokens_per_block + within
                res[0, r, kv] = torch.where(ok, rows, -1).to(torch.int32)
        return res

    def is_valid(
        self, indices, row_req, offsets, layer_in_pool, tokens_per_block, out=None
    ) -> None:
        topk = indices.shape[1]
        assert (
            indices.dtype == torch.int32
            and indices.is_contiguous()
            and row_req.dtype == torch.int32
            and row_req.numel() == indices.shape[0]
            and offsets.dtype == torch.int32
            and offsets.dim() == 3
            and offsets.shape[1] == 2
            and offsets.stride(2) == 1
            and (topk % 128 == 0 or topk & (topk - 1) == 0)
            and (
                out is None
                or (out.is_contiguous() and tuple(out.shape) == (1, indices.shape[0], 2, topk))
            )
        ), "IndexPoolRows received unsupported inputs"


index_pool_rows = IndexPoolRows()
