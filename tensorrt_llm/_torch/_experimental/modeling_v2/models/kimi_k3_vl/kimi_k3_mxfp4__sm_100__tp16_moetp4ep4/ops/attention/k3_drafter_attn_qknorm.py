# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter block attention on the raw projection output: per-head q / k RMSNorm and NeoX RoPE inside
the kernel, then each request's draft block attending densely to its paged context and to the block's own K / V."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ...kernels.k3_drafter import (
    op as _k3_drafter_op,  # noqa: F401 — registers torch.ops.trtllm.k3_drafter_attn_qknorm
)


class K3DrafterAttnQknorm(OpWrapper):
    """Write ``out`` [R T, num_heads * 64]: RMSNorm (``q_w`` / ``k_w``, ``eps``) and NeoX RoPE (``rope_base``, one
    position per row of ``qkv``) of every q and k head, computed in the kernel; then request r's T rows attend to its
    ``ctx_len[r]`` cached rows (pages: row r of ``page_table``) and to its block's own k / v. ``qkv`` is not modified.
    Returns None."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"R={r}, T={t}: "
            + ("DSpark max_draft_len=7 block" if t == 7 else "request/block partition")
            + " with int32/int64 NeoX positions",
            spec=dict(
                heads=6,
                kv=1,
                num_requests=r,
                tokens=t,
                lengths=tuple(
                    (60, 2041, 0, 127, 1000, 64, 128, 5)[(300 + 10 * r + t + 3 * i) % 8]
                    for i in range(r)
                ),
                eps=1e-5,
                rope_base=10000.0,
                position_dtypes=(torch.int32, torch.int64),
            ),
        )
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
    )

    note = """
    qkv is contiguous bf16 [R*T,(num_heads+2*num_kv_heads)*64], q then k
    then v; out is contiguous bf16 [R*T,num_heads*64]. Cache pages are bf16
    HND [pages,2,num_kv_heads,64,64], with dense pages and a page stride
    divisible by eight elements. Tables are int32 [>=R,width] with dense
    columns and any row stride, or [width] for one request; lengths are
    int32 [R]. Six query heads share each KV head.

    Attention is non-causal within each block; requests are independent.
    The block's own K/V come from qkv and are not appended to the cache.
    Active table entries must name that request's pages, and the table
    must cover ceil((ctx_len+T)/64) pages. Rows beyond ctx_len do not
    contribute, including NaN-filled tails.

    Cells use 6 query heads and one KV head of width 64, R,T<=8 and
    contexts from zero to 2041. q_w and k_w are contiguous bf16 [64];
    positions has at least R*T int32 or int64 elements, with bit-identical
    results for both dtypes. eps=1e-5 and rope_base=10000 are certified.
    Unsupported normalization weights or positions raise ValueError.

    RMSNorm is per head, followed by full-dimension NeoX RoPE: the two
    32-element halves rotate together, without YaRN. Only the block's
    own q/k are normalized, not cached rows; v is unchanged. The kernel
    uses fused_qk_norm_rope's arithmetic, including bf16 q/k rounding,
    while the reference accumulates in fp32 and rounds only the output.
    qkv and cache are read-only and out is written. Projection and storing
    retained roped K/V remain outside the fusion.

    No metadata is consumed. The process compile cache is keyed by head
    counts, page stride, R>1, normalization, position dtype and PDL. Each
    new build must run eagerly before CUDA-graph capture or raises
    RuntimeError. Captured calls replay rewritten positions, qkv, lengths
    and page tables; four replays are tested. Unsupported shapes, dtypes
    and layouts raise ValueError before launch. The kernel needs SM100
    tcgen05, TMA and clusters.

    The 1e-2 max-error/output-scale comparison retains qkv immutability
    and position-dtype bit checks. The kernel test also covers the dense
    and 1-D page tables and padded page strides, and compares
    fused_qk_norm_rope followed by the kernel's attention without the
    normalization.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        q_w: torch.Tensor,
        k_w: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
        rope_base: float,
        cache: torch.Tensor,
        page_table: torch.Tensor,
        ctx_len: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        out: torch.Tensor,
    ) -> None:
        torch.ops.trtllm.k3_drafter_attn_qknorm(
            qkv,
            q_w,
            k_w,
            positions,
            eps,
            rope_base,
            cache,
            page_table,
            ctx_len,
            num_heads,
            num_kv_heads,
            out,
        )

    def reference(
        self,
        qkv: torch.Tensor,
        q_w: torch.Tensor,
        k_w: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
        rope_base: float,
        cache: torch.Tensor,
        page_table: torch.Tensor,
        ctx_len: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        out: torch.Tensor,
    ) -> None:
        h, kv = num_heads, num_kv_heads
        x = qkv.float().reshape(qkv.shape[0], h + 2 * kv, 64)
        inv_freq = 1.0 / rope_base ** (
            torch.arange(0, 64, 2, dtype=torch.float32, device=qkv.device) / 64
        )
        angle = positions.reshape(-1)[: qkv.shape[0]].float()[:, None] * inv_freq
        cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
        for start, count, w in ((0, h, q_w), (h, kv, k_w)):
            y = x[:, start : start + count]
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps) * w.float()
            y1, y2 = y[..., :32], y[..., 32:]
            x[:, start : start + count] = torch.cat([y1 * cos - y2 * sin, y1 * sin + y2 * cos], -1)
        x = x.reshape(qkv.shape[0], -1)
        t = qkv.shape[0] // ctx_len.numel()
        tables = page_table.unsqueeze(0) if page_table.dim() == 1 else page_table
        outs = []
        with true_fp32_matmul():
            for r, c in enumerate(ctx_len.reshape(-1).tolist()):
                rows = x[r * t : (r + 1) * t]
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
        q_w: torch.Tensor,
        k_w: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
        rope_base: float,
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


k3_drafter_attn_qknorm = K3DrafterAttnQknorm()
