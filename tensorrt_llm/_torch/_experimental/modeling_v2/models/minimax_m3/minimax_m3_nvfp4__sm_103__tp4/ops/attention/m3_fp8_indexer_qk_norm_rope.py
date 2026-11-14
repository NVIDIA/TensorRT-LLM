# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unfused sparse layers' index branch: per-head Gemma RMSNorm + partial NeoX RoPE of [index-Q | index-K], index-Q
out in FP8 E4M3 and index-K inserted into the FP8 index-K cache, in one kernel."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope,
)


class M3Fp8IndexerQkNormRope(OpWrapper):
    """``qk`` [T, (heads + 1) * 128] BF16 (the index projection's output): returns index-Q [T, heads, 128] E4M3 and
    writes index-K (E4M3) at ``out_cache_loc`` of ``index_k_cache`` [pages, 1, 128, 128]. Gemma RMSNorm of each head,
    then RoPE on its leading ``rotary_dim`` channels (NeoX).

    The unfused sparse layers (``fuse_qkv_index_projection=False``) run it on the index branch with an FP8 indexer;
    the fused layers' producer writes the index cache itself.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode token, one index head per rank", spec=dict(m=1, heads=1)),
        Cell(why="a 48-row decode verify", spec=dict(m=48, heads=1)),
        Cell(why="a prefill chunk", spec=dict(m=2048, heads=1)),
    )

    note = """
    `reference` is the in-place fused norm + RoPE in BF16 followed by the E4M3 cast (the kernel rounds through BF16
    before the cast); its index-K is what the kernel writes to the cache. The two kernels may round a handful of
    values differently: the test allows fewer than 0.1% of the bytes to differ, as the in-tree test does.
    """

    def __call__(
        self,
        qk: torch.Tensor,
        index_k_cache: torch.Tensor,
        out_cache_loc: torch.Tensor,
        num_heads_q: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        base: float,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.trtllm.minimax_m3_fp8_indexer_qk_norm_rope(
            qk,
            index_k_cache,
            out_cache_loc,
            num_heads_q,
            head_dim,
            rotary_dim,
            eps,
            q_weight,
            k_weight,
            base,
            position_ids,
        )

    def reference(
        self,
        qk: torch.Tensor,
        index_k_cache: torch.Tensor,
        out_cache_loc: torch.Tensor,
        num_heads_q: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        base: float,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(index-Q [T, heads, head_dim] E4M3, index-K [T, head_dim] E4M3): the catalog's in-place kernel on a copy,
        then the E4M3 cast."""
        out = fused_qk_norm_rope(
            qk.clone(),
            num_heads_q,
            1,
            0,
            head_dim,
            rotary_dim,
            eps,
            q_weight,
            k_weight,
            base,
            True,
            position_ids,
            1.0,
            0.0,
            0.0,
            1.0,
            True,
            True,
            False,
            0,
            0,
        )
        q, k = out.split([num_heads_q * head_dim, head_dim], dim=-1)
        return (
            q.reshape(q.shape[0], num_heads_q, head_dim).to(torch.float8_e4m3fn),
            k.to(torch.float8_e4m3fn),
        )


m3_fp8_indexer_qk_norm_rope = M3Fp8IndexerQkNormRope()
