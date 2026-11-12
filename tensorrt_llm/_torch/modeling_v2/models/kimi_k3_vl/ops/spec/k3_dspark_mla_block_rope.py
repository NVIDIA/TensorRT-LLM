# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The block RoPE work of one MLA drafter layer in one CuTe DSL launch: the absorbed query's
rope columns and the block latents normalized, rotated and stored into the paged pool."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._dspark_mla_rope import dspark_mla_block_rope
from .k3_dspark_rmsnorm_rope import rmsnorm_rope_reference


class K3DsparkMlaBlockRope(OpWrapper):
    """Writes ``query[..., kv_lora_rank:]`` = the RoPE of ``q[..., qk_nope_dim:]`` per head and
    ``pool[pages[t], offsets[t]]`` = ``kv[t]`` with its latent RMSNorm (times ``norm_weight``)
    and its rope columns rotated, both with ``freqs[positions[t]]``. Returns None; the rest of
    ``query`` and of the pool is untouched.

    ``q`` is BF16 ``[T, H, qk_nope_dim + rope_dim]``, ``query`` BF16 ``[T, H, kv_lora_rank +
    rope_dim]`` (unit inner strides, heads packed), ``kv`` BF16 ``[T, head_dim]`` row-strided,
    ``pool`` ``[pages, page_size, head_dim]`` BF16, ``positions`` / ``pages`` / ``offsets``
    int64 ``[T]``, ``freqs`` FP32 ``[P, rope_dim // 2, 2]`` contiguous.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS = (
        Cell(
            why="helix16 ULL c4: 4 requests x 8 block tokens (DSpark 7), 64 heads, latents read "
            "in place from the fused [q_a | kv_a] GEMM output (row stride 2112), 128-token pages",
            spec=dict(tokens=32, kv_row_stride=2112),
        ),
        Cell(
            why="helix16 ULL c1: one request, 8 block tokens",
            spec=dict(tokens=8, kv_row_stride=2112),
        ),
        Cell(
            why="helix16 ULL c2: 2 requests x 8 block tokens, kv_a rows from their own projection "
            "(compact, 576)",
            spec=dict(tokens=16, kv_row_stride=576),
        ),
    )

    note = """
    Bitwise ``cute_dsl_dspark_rope_into(q, freqs[positions], query, H, rope_dim, kv_lora_rank)``
    and ``k3_scatter_rows_to_pool(k3_dspark_rmsnorm_rope(kv, norm_weight, freqs[positions], 1,
    rope_dim, eps, True, True, False, kv_lora_rank), pool, pages, offsets)`` in one launch
    (``dspark_tail_v2``). Distinct (page, offset) per token. A process-wide compile cache holds
    one kernel per (H, qk_nope_dim, kv_lora_rank, rope_dim, eps).
    """

    def __call__(
        self,
        q: torch.Tensor,
        query: torch.Tensor,
        kv: torch.Tensor,
        norm_weight: torch.Tensor,
        positions: torch.Tensor,
        freqs: torch.Tensor,
        pool: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
        eps: float,
    ) -> None:
        dspark_mla_block_rope(
            q, query, kv, norm_weight, positions, freqs, pool, pages, offsets, eps
        )

    def reference(
        self,
        q: torch.Tensor,
        query: torch.Tensor,
        kv: torch.Tensor,
        norm_weight: torch.Tensor,
        positions: torch.Tensor,
        freqs: torch.Tensor,
        pool: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(query, pool)`` after the call (clones)."""
        num_tokens, num_heads, q_dim = q.shape
        kv_lora_rank = norm_weight.shape[0]
        rope_dim = query.shape[-1] - kv_lora_rank
        table = freqs[positions.reshape(-1)]
        query_out = query.clone()
        rotated = rmsnorm_rope_reference(
            q[..., q_dim - rope_dim :], None, table, num_heads, rope_dim, 0.0, False, False, False
        )
        query_out[..., kv_lora_rank:] = rotated
        latents = rmsnorm_rope_reference(
            kv, norm_weight, table, 1, rope_dim, eps, True, True, False, kv_lora_rank
        )
        pool_out = pool.clone()
        pool_out[pages.reshape(-1), offsets.reshape(-1)] = latents
        return query_out, pool_out

    def is_valid(
        self,
        q: torch.Tensor,
        query: torch.Tensor,
        kv: torch.Tensor,
        norm_weight: torch.Tensor,
        positions: torch.Tensor,
        freqs: torch.Tensor,
        pool: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
        eps: float,
    ) -> None:
        if positions.numel():
            assert int(positions.max()) < freqs.shape[0] and int(positions.min()) >= 0
            keys = pages.reshape(-1) * pool.shape[1] + offsets.reshape(-1)
            assert keys.unique().numel() == keys.numel(), "two block tokens share a pool slot"
            assert int(offsets.max()) < pool.shape[1] and int(pages.max()) < pool.shape[0]


k3_dspark_mla_block_rope = K3DsparkMlaBlockRope()
