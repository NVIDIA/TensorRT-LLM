# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sparse layers' generic horizontal producer: packed QKV/index rows -> FP8 Q / index-Q, K / V / index-K into
the paged caches."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class M3QkvInsert(OpWrapper):
    """The packed ``[Q | K | V | index-Q | index-K]`` bf16 rows of a sparse layer: per-head Gemma RMSNorm and the
    partial NeoX RoPE of Q, K, index-Q and index-K, K / V written to the main paged cache at ``out_cache_loc`` and
    index-K to the FP8 index cache; returns (Q, index-Q) in FP8 E4M3 [M, heads, 128].

    ``kv_scale`` given: the main cache is NVFP4 (packed E2M1 data [pages, 2, kv, 128, 64] uint8 + E4M3 scale pool
    [pages, 2, kv, 128, 8], ``kv_quant_scale`` the fp32 (Q, K, V) quantization scales); None: FP8 E4M3
    [pages, 2, kv, 128, 128]. Token-major and step-type agnostic. ``m3_producer`` takes the NVFP4 cache at the tp8
    heads; this entry takes the FP8 cache and NVFP4 pools outside that layout.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="FP8 main cache, a prefill chunk", spec=dict(m=1024, cache="fp8")),
        Cell(why="FP8 main cache, one decode token", spec=dict(m=1, cache="fp8")),
    )

    note = """
    The arithmetic is the CUDA producer's (fp32 norm and RoPE with bf16 roundings, then the E4M3 casts); `reference`
    returns the fp32 values, and the test holds the FP8 outputs and cache rows within one E4M3 step of them.
    """

    def __call__(
        self,
        packed: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_scale: Optional[torch.Tensor],
        index_k_cache: torch.Tensor,
        out_cache_loc: torch.Tensor,
        kv_quant_scale: Optional[torch.Tensor],
        num_heads: int,
        num_kv_heads: int,
        num_index_heads: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        index_q_norm: torch.Tensor,
        index_k_norm: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        common = (
            num_heads,
            num_kv_heads,
            num_index_heads,
            head_dim,
            rotary_dim,
            eps,
            q_norm,
            k_norm,
            index_q_norm,
            index_k_norm,
            rotary_cos_sin,
            position_ids,
        )
        if kv_scale is not None:
            return torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert(
                packed, kv_cache, kv_scale, index_k_cache, out_cache_loc, kv_quant_scale, *common
            )
        return torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(
            packed, kv_cache, index_k_cache, out_cache_loc, *common
        )

    def reference(
        self,
        packed: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_scale: Optional[torch.Tensor],
        index_k_cache: torch.Tensor,
        out_cache_loc: torch.Tensor,
        kv_quant_scale: Optional[torch.Tensor],
        num_heads: int,
        num_kv_heads: int,
        num_index_heads: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        index_q_norm: torch.Tensor,
        index_k_norm: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """The fp32 values the producer rounds (a new dict, nothing written): Gemma RMSNorm + NeoX partial RoPE of
        Q [M, heads, D], index-Q [M, index heads, D], K [M, kv heads, D] and index-K [M, D]; V [M, kv heads, D] as
        packed. The cache rows at ``out_cache_loc`` and the returned FP8 Q / index-Q are these values cast."""
        m = packed.shape[0]
        sizes = [
            num_heads * head_dim,
            num_kv_heads * head_dim,
            num_kv_heads * head_dim,
            num_index_heads * head_dim,
            head_dim,
        ]
        q, k, v, iq, ik = packed.float().split(sizes, dim=-1)
        cos, sin = rotary_cos_sin[position_ids.long()].unbind(1)
        half = rotary_dim // 2

        def norm_rope(x, w, heads):
            x = x.reshape(m, heads, head_dim)
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())
            x1, x2, rest = x[..., :half], x[..., half:rotary_dim], x[..., rotary_dim:]
            c, s = cos.unsqueeze(1), sin.unsqueeze(1)
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, rest], dim=-1)

        return {
            "q": norm_rope(q, q_norm, num_heads),
            "index_q": norm_rope(iq, index_q_norm, num_index_heads),
            "k": norm_rope(k, k_norm, num_kv_heads),
            "v": v.reshape(m, num_kv_heads, head_dim),
            "index_k": norm_rope(ik, index_k_norm, 1)[:, 0],
        }


m3_qkv_insert = M3QkvInsert()
