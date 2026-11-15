# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dense layers' per-head RMSNorm + partial NeoX RoPE of the packed [Q | K | V] into FP8, the bits of the BF16
``fused_qk_norm_rope`` output cast to E4M3; optionally also storing the live tokens' K / V into the paged FP8 cache."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope,
)


class M3DenseQkNormRopeFp8(OpWrapper):
    """``qkv`` [T, (q + 2 kv) heads x 128] bf16 -> [T, same] E4M3: Q and K heads normed and roped in fp32, rounded to
    BF16 and then to E4M3 (saturating), V cast; ``qkv`` untouched
    (``trtllm::minimax_m3_sm107tp4_dense_qk_norm_rope_fp8``).

    With ``kv_buffers`` (the layer's HND FP8 pool [pages, 2, kv heads, 128, 128]), ``out_cache_loc`` (the step's int32
    cache slots) and ``num_live_tokens``, the kernel also stores the live tokens' K and V rows into the pool (the bytes
    the MSA backend's KV write stores) and every row's Q into ``q_compact`` [T, q heads, 128]
    (``..._dense_qk_norm_rope_fp8_kv``).
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode token, tp4 heads (16 q, 1 kv)", spec=dict(m=1, q=16, kv=1)),
        Cell(
            why="a 4-request verify (16 rows) storing K / V",
            spec=dict(m=16, q=16, kv=1, kv_write=True),
        ),
        Cell(why="a 48-row decode verify", spec=dict(m=48, q=16, kv=1)),
        Cell(why="a prefill chunk on an FP8 cache", spec=dict(m=2048, q=16, kv=1)),
    )

    note = """
    Bit-exact: `reference` runs the catalog's in-place BF16 ``fused_qk_norm_rope`` on a copy and casts every element to
    E4M3, which is this kernel's rounding sequence (fp32 -> BF16 -> E4M3) for Q and K and the cast for V.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        rotary_dim: int,
        eps: float,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        base: float,
        is_neox: bool,
        position_ids: torch.Tensor,
        use_gemma: bool,
        kv_buffers: Optional[torch.Tensor] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        num_live_tokens: int = 0,
        q_compact: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        args = (
            qkv,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            rotary_dim,
            eps,
            q_weight,
            k_weight,
            base,
            is_neox,
            position_ids,
            use_gemma,
        )
        if kv_buffers is not None:
            return torch.ops.trtllm.minimax_m3_sm107tp4_dense_qk_norm_rope_fp8_kv(
                *args, kv_buffers, out_cache_loc, num_live_tokens, q_compact
            )
        return torch.ops.trtllm.minimax_m3_sm107tp4_dense_qk_norm_rope_fp8(*args)

    def reference(
        self,
        qkv: torch.Tensor,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        rotary_dim: int,
        eps: float,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        base: float,
        is_neox: bool,
        position_ids: torch.Tensor,
        use_gemma: bool,
        kv_buffers: Optional[torch.Tensor] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        num_live_tokens: int = 0,
        q_compact: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The FP8 copy (the K / V store into ``kv_buffers`` is checked by the test against the reference rows)."""
        head_dim = qkv.shape[-1] // (num_heads_q + num_heads_k + num_heads_v)
        out = qkv.clone()
        fused_qk_norm_rope(
            out,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            rotary_dim,
            eps,
            q_weight,
            k_weight,
            base,
            is_neox,
            position_ids,
            1.0,
            0.0,
            0.0,
            1.0,
            True,
            use_gemma,
            False,
            0,
            0,
        )
        return out.to(torch.float8_e4m3fn)


m3_dense_qk_norm_rope_fp8 = M3DenseQkNormRopeFp8()
