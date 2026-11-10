# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter block attention on the raw projection output: per-head q / k RMSNorm and NeoX RoPE inside
the kernel, then k3_drafter_attn."""

import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_drafter.op  # noqa: F401 — registers torch.ops.trtllm.k3_drafter_attn_qknorm
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .k3_drafter_attn import k3_drafter_attn


class K3DrafterAttnQknorm(OpWrapper):
    """``k3_drafter_attn`` after RMSNorm (``q_w`` / ``k_w``, ``eps``) and NeoX RoPE (``rope_base``, one position per
    row of ``qkv``) of every q and k head, computed in the kernel; ``qkv`` is not modified. Returns None."""

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
    The cache, qkv, output and page-table layouts, request ownership,
    preconditions and non-causal block attention are k3_drafter_attn's.
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

    No metadata is consumed. The process compile cache and eager-before-
    capture rule are k3_drafter_attn's, with normalization and position
    dtype in the build key. Captured calls replay rewritten positions,
    qkv, lengths and page tables; four replays are covered in the paired
    attention test. The 1e-2 max-error/output-scale comparison retains
    qkv immutability and position-dtype bit checks. The kernel test also
    compares fused_qk_norm_rope followed by k3_drafter_attn.
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
        k3_drafter_attn.reference(
            x.reshape(qkv.shape[0], -1),
            cache,
            page_table,
            ctx_len,
            num_heads,
            num_kv_heads,
            out,
        )

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
        k3_drafter_attn.is_valid(qkv, cache, page_table, ctx_len, num_heads, num_kv_heads, out)


k3_drafter_attn_qknorm = K3DrafterAttnQknorm()
