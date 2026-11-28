# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compressor of an index-owner layer, one launch: state update, pair pooling, RMSNorm, index-K projection, row slots.

A Triton kernel after the engine's `_compress_project_kernel` (`attention/backends/sparse/csa2/compressor.py`), which
fuses the same four steps for the engine's compressed-row layout. This one keeps the token-indexed layout of the target
(one latent per source token, each token carrying the latent of its own pair), so the target writes the rows with the
one `csa2_store_rows` launch and no per-token index arithmetic. It also resolves, per token, the flat GLOBAL / INDEX
pool row the latent is published at and the first position of its pair (the position the row's RoPE uses).
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm

from .csa2_compress import csa2_compress_pairs
from .csa2_rows_io import csa2_row_addr

_STATE_ROWS = 128  # rows per page of the compressor-state pools (one row per token position)
_ROWS = tl.constexpr(_STATE_ROWS)  # the same, for the kernel
_N_BLOCK = 16  # index-K outputs per program: a row's projection spreads over N / 16 programs


@triton.jit
def _compress_decode_kernel(
    X,
    x_stride,
    POS,
    RID,
    CS,
    CE,
    KV_STATE,
    SCORE_STATE,
    KV_TABLE,
    SCORE_TABLE,
    state_stride,
    state_cols,
    NORM_W,
    eps,
    WK,
    G_TABLE,
    g_stride,
    g_cols,
    LATENT,
    INDEX_K,
    SLOTS,
    FIRST,
    D: tl.constexpr,
    N: tl.constexpr,
    RATIO: tl.constexpr,
    RPP: tl.constexpr,
    N_BLOCK: tl.constexpr,
    PROJECT: tl.constexpr,
):
    """Program `(row, block)`: block `block` of `INDEX_K[row] = LATENT[row] @ WK.T`. Block 0 also writes `LATENT[row]`,
    `SLOTS[row]`, `FIRST[row]` and, at ratio 2, this token's kv / score row into the state pools (every row, as the
    engine's compressor keeps every source row: a speculative step can start at an odd position inside the previous
    step, whose partner must then be that step's accepted row, not whatever the page held before; a padding copy of
    the previous row writes nothing)."""
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = tl.arange(0, D)
    outputs = block * N_BLOCK + tl.arange(0, N_BLOCK)
    pos = tl.load(POS + row)
    rid = tl.load(RID + row)
    weight = tl.load(NORM_W + cols).to(tl.float32)
    if RATIO == 2:
        cs = tl.load(CS + row)
        ce = tl.load(CE + row)
        k_own = tl.load(X + row * x_stride + cols).to(tl.float32)
        s_own = tl.load(X + row * x_stride + D + cols).to(tl.float32)
        blk = pos // _ROWS
        in_table = blk < state_cols
        kv_page = tl.load(KV_TABLE + rid * state_stride + blk, mask=in_table, other=-1)
        score_page = tl.load(SCORE_TABLE + rid * state_stride + blk, mask=in_table, other=-1)
        if block == 0:
            off = pos % _ROWS
            # A row repeating the previous row's (request, position) is a padding copy of the dep2 static body: its own
            # row is the live row's slot, but its activations need not be the live row's (its Engram look-back reads
            # the ring, not the step), so only the first row of a position writes the state the next step pairs with.
            prev = tl.maximum(row - 1, 0)
            first_of_pos = (row == 0) | (tl.load(RID + prev) != rid) | (tl.load(POS + prev) != pos)
            tl.store(
                KV_STATE + (kv_page.to(tl.int64) * _ROWS + off) * D + cols,
                k_own,
                mask=(kv_page >= 0) & first_of_pos,
            )
            tl.store(
                SCORE_STATE + (score_page.to(tl.int64) * _ROWS + off) * D + cols,
                s_own,
                mask=(score_page >= 0) & first_of_pos,
            )
        partner = pos ^ 1
        inflight = (partner >= cs) & (partner < ce)
        old = partner < cs
        # The partner's row, kept inside the launch's rows as `csa2_compress_pairs` keeps it: a padding row of the dep2
        # static body (a copy of the step's last row whose `ce` is out of range) finds its partner "in flight", and the
        # last one would read one row past `X` (an illegal access when `X` ends its allocation segment).
        src = tl.minimum(tl.maximum(row + partner - pos, 0), tl.num_programs(0) - 1)
        k_in = tl.load(X + src * x_stride + cols, mask=inflight & (cols < D), other=0.0).to(
            tl.float32
        )
        s_in = tl.load(X + src * x_stride + D + cols, mask=inflight & (cols < D), other=0.0).to(
            tl.float32
        )
        pblk = partner // _ROWS
        p_in_table = pblk < state_cols
        p_kv_page = tl.load(KV_TABLE + rid * state_stride + pblk, mask=p_in_table, other=-1)
        p_score_page = tl.load(SCORE_TABLE + rid * state_stride + pblk, mask=p_in_table, other=-1)
        poff = partner % _ROWS
        k_old = tl.load(
            KV_STATE + (p_kv_page.to(tl.int64) * _ROWS + poff) * D + cols,
            mask=old & (p_kv_page >= 0) & (cols < D),
            other=0.0,
        )
        s_old = tl.load(
            SCORE_STATE + (p_score_page.to(tl.int64) * _ROWS + poff) * D + cols,
            mask=old & (p_score_page >= 0) & (cols < D),
            other=0.0,
        )
        k_p = tl.where(inflight, k_in, k_old)
        s_p = tl.where(inflight | old, tl.where(inflight, s_in, s_old), float("-inf"))
        own_first = (pos & 1) == 0
        a_kv = tl.where(own_first, k_own, k_p)
        b_kv = tl.where(own_first, k_p, k_own)
        a_s = tl.where(own_first, s_own, s_p)
        b_s = tl.where(own_first, s_p, s_own)
        top = tl.maximum(a_s, b_s)
        e_a = tl.exp(a_s - top)
        e_b = tl.exp(b_s - top)
        pooled = ((a_kv * e_a + b_kv * e_b) / (e_a + e_b)).to(tl.bfloat16).to(tl.float32)
        first = pos - (pos & 1)
        row_in_pool = pos // 2
    else:
        pooled = tl.load(X + row * x_stride + cols).to(tl.float32)
        first = pos
        row_in_pool = pos
    inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / D + eps)
    normalized = ((pooled * inv) * weight).to(tl.bfloat16)
    if PROJECT:
        w = tl.load(WK + outputs[:, None] * D + cols[None, :]).to(tl.float32)
        projected = tl.sum(w * normalized.to(tl.float32)[None, :], 1)
        tl.store(INDEX_K + row * N + outputs, projected.to(tl.bfloat16))
    if block == 0:
        tl.store(LATENT + row * D + cols, normalized)
        tl.store(FIRST + row, first)
        gblk = row_in_pool // RPP
        g_page = tl.load(G_TABLE + rid * g_stride + gblk, mask=gblk < g_cols, other=-1)
        slot = tl.where(g_page >= 0, g_page.to(tl.int64) * RPP + row_in_pool % RPP, -1)
        tl.store(SLOTS + row, slot)


class Csa2CompressDecode(OpWrapper):
    """`(latent bf16 [n, 512], index_k bf16 [n, 128], slots int64 [n], first int32 [n])` of an index-owner layer.

    `x` is the token rows' projection: at ratio 2 `[n, 1024]` fp32 or bf16, value then gate (`csa2_compress_pairs`' `kv`
    and `score` side by side; a bf16 projection is promoted to fp32 on load), at ratio 1 bf16 `[n, 512]` (no pooling, no
    state). `pos` int32 `[n]`, `rid` int64 `[n]`, `cs` / `ce` int32 `[n]` and the state pools / tables are those of
    `csa2_compress_pairs`, whose state update and pairing rules apply (every token of the step is written to the state
    pools, in place; a row repeating the previous row's request and position, the dep2 static body's padding copy of a
    step's last row, writes no state). `latent` is the pooled
    pair (the token's own row at ratio 1) rounded to bf16 and RMS-normalised with `norm_w`; `index_k` is
    `latent @ wk.T` rounded to bf16; `slots` the flat row of the latent in the owner's GLOBAL pool
    (`g_table`, `128 // ratio` rows per page; -1 = unmapped) and `first` the first position of the token's pair.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 verify step: one request of six rows after a prefill, ratio 2",
            spec=dict(ratio=2, reqs=((7, 6, 6),)),
        ),
        Cell(
            why="decode: one token per step, so every pair straddles two steps, ratio 2",
            spec=dict(ratio=2, reqs=((1,) * 9,)),
        ),
        Cell(
            why="chunked prefill with odd chunk sizes, pairs straddle chunk edges, ratio 2",
            spec=dict(ratio=2, reqs=((3, 1, 4, 1, 2),)),
        ),
        Cell(
            why="dep8 step: 320 requests of four rows (1280 rows), ratio 2",
            spec=dict(ratio=2, reqs=tuple((9, 4) for _ in range(320))),
        ),
        Cell(
            why="dep32 step: 25 requests of six rows, a request crossing the 128-row state page, ratio 2",
            spec=dict(ratio=2, reqs=((130, 5),) + tuple((6, 6) for _ in range(24))),
        ),
        Cell(
            why="ratio 2 from a bf16 projection (the target's own, one GEMM over value | gate): verify steps",
            spec=dict(ratio=2, reqs=((7, 6, 6),) * 3, bf16=True),
        ),
        Cell(
            why="the ratio-1 owner: four requests of six rows, no pooling",
            spec=dict(ratio=1, reqs=((20, 6),) * 4),
        ),
        Cell(
            why="dep2 context step, ratio 2 from the bf16 projection, index-K as one GEMM: 8 requests of 380 rows",
            spec=dict(ratio=2, reqs=((380,),) * 8, bf16=True, index_gemm=True),
        ),
        Cell(
            why=(
                "dep2 context step at the ratio-1 owner, index-K as "
                "one GEMM: a 1500-row chunk after a first one of 100"
            ),
            spec=dict(ratio=1, reqs=((100, 1500),) * 2, index_gemm=True),
        ),
    )
    note = """
    Latents are bf16 after an fp32 pooling and norm, so the test gates them in bf16 ulps against the plain-torch
    reference; state pools, slots and first positions agree exactly. Tokens of one request must be adjacent and
    ascending inside the step (`is_valid`).
    """

    def __call__(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        cs: torch.Tensor,
        ce: torch.Tensor,
        state_kv: torch.Tensor,
        state_score: torch.Tensor,
        table_kv: torch.Tensor,
        table_score: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        wk: torch.Tensor,
        g_table: torch.Tensor,
        ratio: int,
        index_gemm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n, d = x.shape[0], norm_w.shape[0]
        dev = x.device
        latent = torch.empty(n, d, dtype=torch.bfloat16, device=dev)
        index_k = torch.empty(n, wk.shape[0], dtype=torch.bfloat16, device=dev)
        slots = torch.empty(n, dtype=torch.int64, device=dev)
        first = torch.empty(n, dtype=torch.int32, device=dev)
        rpp = _STATE_ROWS // ratio
        if ratio == 2:
            state = (
                state_kv,
                state_score,
                table_kv,
                table_score,
                table_kv.stride(0),
                table_kv.shape[1],
            )
        else:  # no state at ratio 1: any valid pointers, never read
            state = (x, x, pos, pos, 0, 0)
        if n:
            # `index_gemm` (context steps): one program per row (each `(row, block)` program would pool and normalize
            # the row again), then the index-K projection as one bf16 GEMM, as stock's prefill compressor does
            # (`bf16_projection(index_wk, latent)`)
            _compress_decode_kernel[(n, 1 if index_gemm else wk.shape[0] // _N_BLOCK)](
                x, x.stride(0), pos, rid, cs, ce, *state, norm_w, eps, wk, g_table, g_table.stride(0), g_table.shape[1],
                latent, index_k, slots, first,
                d, wk.shape[0], ratio, rpp, _N_BLOCK, not index_gemm,
                num_warps=4, enable_fp_fusion=False,
            )  # fmt: skip
            if index_gemm:
                index_k = cublas_mm(latent, wk.t())
        return latent, index_k, slots, first

    def reference(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        cs: torch.Tensor,
        ce: torch.Tensor,
        state_kv: torch.Tensor,
        state_score: torch.Tensor,
        table_kv: torch.Tensor,
        table_score: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        wk: torch.Tensor,
        g_table: torch.Tensor,
        ratio: int,
        index_gemm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pairing and state update by `csa2_compress_pairs.reference`, the norm and the projection in fp32, the slots
        by `csa2_row_addr.reference` (state pools are updated in place: pass copies)."""
        d = norm_w.shape[0]
        x = x.float() if ratio == 2 else x
        if ratio == 2:
            pooled = csa2_compress_pairs.reference(
                x[:, :d], x[:, d:], pos, rid, cs, ce, state_kv, state_score, table_kv, table_score
            )
            row, first = pos.long() // 2, pos - pos % 2
        else:
            pooled, row, first = x.float(), pos.long(), pos.clone()
        v = pooled.to(torch.bfloat16).float()
        normalized = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) * norm_w.float()).to(
            torch.bfloat16
        )
        index_k = (normalized.float() @ wk.float().t()).to(torch.bfloat16)
        slots = csa2_row_addr.reference(g_table, rid, row, _STATE_ROWS // ratio)
        return normalized, index_k, slots, first

    def is_valid(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        cs: torch.Tensor,
        ce: torch.Tensor,
        state_kv: torch.Tensor,
        state_score: torch.Tensor,
        table_kv: torch.Tensor,
        table_score: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        wk: torch.Tensor,
        g_table: torch.Tensor,
        ratio: int,
        index_gemm: bool = False,
    ) -> None:
        d = norm_w.shape[0]
        assert ratio in (1, 2) and d == 512 and wk.shape[1] == d and wk.shape[0] % _N_BLOCK == 0, (
            "ratio 1 or 2, 512-wide latents, an index width that is a multiple of 16"
        )
        assert wk.dtype == torch.bfloat16 and norm_w.dtype == torch.bfloat16 and wk.is_contiguous()
        assert norm_w.is_contiguous(), "norm_w is read as consecutive values"
        if ratio == 2:
            assert x.dtype in (torch.float32, torch.bfloat16) and x.shape[1] == 2 * d, (
                "ratio 2: x is [n, 1024] (value then gate), fp32 or bf16 (a bf16 projection is promoted; the state "
                "pools stay fp32)"
            )
            assert state_kv.dtype == torch.float32 and state_score.dtype == torch.float32
            assert state_kv.is_contiguous() and state_score.is_contiguous()
            assert table_kv.dtype == torch.int32 and table_score.dtype == torch.int32
            assert table_kv.shape == table_score.shape and table_kv.stride() == table_score.stride()
            assert table_kv.stride(1) == 1, (
                "both state tables are rows of one stacked table (one row stride)"
            )
        else:
            assert x.dtype == torch.bfloat16 and x.shape[1] == d, "ratio 1: x is bf16 [n, 512]"
        assert x.stride(1) == 1 and pos.dtype == torch.int32 and cs.dtype == ce.dtype == torch.int32
        assert rid.dtype == torch.int64 and g_table.dtype == torch.int32 and g_table.stride(1) == 1
        assert bool(((pos >= cs) & (pos < ce)).all()), (
            "every token must lie in its request's [cs, ce)"
        )
        runs = 1 + int((rid[1:] != rid[:-1]).sum()) if rid.numel() else 0
        assert runs == rid.unique().numel(), (
            "the tokens of one request must form one run (request-major rows): a recurring request id pairs a row "
            "with another request's"
        )
        same = rid[1:] == rid[:-1]
        assert bool((pos[1:][same] == pos[:-1][same] + 1).all()), (
            "the tokens of one request must be adjacent and ascending in the step"
        )


csa2_compress_decode = Csa2CompressDecode()
