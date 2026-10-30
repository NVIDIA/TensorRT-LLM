# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter block attention: each request's draft block attends densely to its paged context and to the
block's own K / V."""

import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_drafter.op  # noqa: F401 — registers torch.ops.trtllm.k3_drafter_attn
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class K3DrafterAttn(OpWrapper):
    """Write ``out`` [R T, num_heads * 64]: request r's T rows of ``qkv`` attend to its ``ctx_len[r]`` cached rows
    (pages: row r of ``page_table``) and to its block's own k / v in ``qkv``. Returns None."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{heads}/{kv} heads, R={r}, T={t}: "
                + ("DSpark max_draft_len=7 block" if t == 7 else "request/block partition"),
                spec=dict(
                    heads=heads,
                    kv=kv,
                    num_requests=r,
                    tokens=t,
                    lengths=tuple(
                        (60, 2041, 0, 127, 1000, 64, 128, 5)[(100 * r + t + heads + 3 * i) % 8]
                        for i in range(r)
                    ),
                ),
            )
            for heads, kv in ((6, 1), (24, 4))
            for r, t in (
                (1, 1),
                (1, 8),
                (2, 4),
                (4, 2),
                (8, 1),
                (3, 1),
                (2, 8),
                (8, 8),
                *((r, 7) for r in range(1, 9)),
            )
        ),
        *(
            Cell(
                why=f"{form} cache/table layout and one request's 1-D table",
                spec=dict(
                    heads=6,
                    kv=1,
                    num_requests=4,
                    tokens=8,
                    lengths=(5, 0, 64, 60),
                    page_pad=page_pad,
                    strided_table=strided_table,
                    table_forms=True,
                ),
            )
            for form, page_pad, strided_table in (
                ("dense", 0, False),
                ("padded", 3 * 2 * 64 * 64, True),
            )
        ),
    )

    note = """
    qkv is contiguous bf16 [R*T,(num_heads+2*num_kv_heads)*64], q then k
    then v; out is contiguous bf16 [R*T,num_heads*64]. Cache pages are bf16
    HND [pages,2,num_kv_heads,64,64], with dense pages and a page stride
    divisible by eight elements. Tables are int32 [>=R,width] with dense
    columns and any row stride, or [width] for one request; lengths are
    int32 [R]. The cells cover TP16's 6/1 and TP4's 24/4 head layouts,
    R,T<=8 and contexts from zero to 2041, including page, 128-row tile
    and 16-tile-round boundaries.

    Attention is non-causal within each block; requests are independent.
    The block's own K/V come from qkv and are not appended to the cache.
    Only out is written; all inputs are read-only. Cache ownership remains
    with the caller: active table entries must name that request's pages.
    Rows beyond ctx_len do not contribute, including NaN-filled tails.
    The kernel may fetch pages through ceil((ctx_len+T)/64), so table
    storage must cover the block as well as its context. Projection,
    q/k normalization, RoPE and storing retained K/V are outside this op.

    No attention metadata is consumed. The process compile cache is keyed
    by head counts, page stride, R>1, normalization/position dtype and PDL.
    Each new build must run eagerly before CUDA-graph capture or raises
    RuntimeError. Lengths, tables and qkv are device-read and may be
    rewritten in place for replay; four such replays are tested.
    Unsupported shapes, dtypes and layouts raise ValueError before launch.
    The kernel needs SM100 tcgen05, TMA and clusters.

    Probabilities and outputs round to bf16. Certification uses a 1e-2
    max-error/output-scale band and checks cache immutability and bitwise
    reruns. The kernel test additionally checks request isolation and the
    replaced DFlash append_paged_kv_cache plus non-causal trtllm-gen path.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        cache: torch.Tensor,
        page_table: torch.Tensor,
        ctx_len: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        out: torch.Tensor,
    ) -> None:
        torch.ops.trtllm.k3_drafter_attn(
            qkv, cache, page_table, ctx_len, num_heads, num_kv_heads, out
        )

    def reference(
        self,
        qkv: torch.Tensor,
        cache: torch.Tensor,
        page_table: torch.Tensor,
        ctx_len: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        out: torch.Tensor,
    ) -> None:
        h, kv = num_heads, num_kv_heads
        t = qkv.shape[0] // ctx_len.numel()
        tables = page_table.unsqueeze(0) if page_table.dim() == 1 else page_table
        outs = []
        with true_fp32_matmul():
            for r, c in enumerate(ctx_len.reshape(-1).tolist()):
                rows = qkv[r * t : (r + 1) * t].float()
                q = rows[:, : h * 64].reshape(t, h, 64)
                k_blk = rows[:, h * 64 : (h + kv) * 64].reshape(t, kv, 64).permute(1, 0, 2)
                v_blk = rows[:, (h + kv) * 64 :].reshape(t, kv, 64).permute(1, 0, 2)
                pages = tables[r, : (c + 63) // 64].long()
                k_ctx = cache[pages, 0].float().permute(1, 0, 2, 3).reshape(kv, -1, 64)[:, :c]
                v_ctx = cache[pages, 1].float().permute(1, 0, 2, 3).reshape(kv, -1, 64)[:, :c]
                k = torch.cat([k_ctx, k_blk], 1).repeat_interleave(h // kv, 0)
                v = torch.cat([v_ctx, v_blk], 1).repeat_interleave(h // kv, 0)
                p = torch.softmax(torch.einsum("thd,hld->thl", q, k) / 8.0, dim=-1)
                outs.append(torch.einsum("thl,hld->thd", p, v).reshape(t, h * 64))
        out.copy_(torch.cat(outs).to(out.dtype).reshape(out.shape))

    def is_valid(
        self,
        qkv: torch.Tensor,
        cache: torch.Tensor,
        page_table: torch.Tensor,
        ctx_len: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        out: torch.Tensor,
    ) -> None:
        tables = page_table.unsqueeze(0) if page_table.dim() == 1 else page_table
        t = qkv.shape[0] // ctx_len.numel()
        for r, c in enumerate(ctx_len.reshape(-1).tolist()):
            assert c >= 0, "context lengths must be nonnegative"
            assert c + t <= tables.shape[1] * 64, "page table must cover context and block"
            pages = tables[r, : (c + 63) // 64]
            assert bool(((pages >= 0) & (pages < cache.shape[0])).all()), (
                "active page-table entries must name cache pages"
            )


k3_drafter_attn = K3DrafterAttn()
