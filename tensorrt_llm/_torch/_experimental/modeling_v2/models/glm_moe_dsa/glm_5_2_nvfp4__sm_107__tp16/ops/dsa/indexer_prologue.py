# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused indexer prologue: k_norm, RoPE, E4M3 Q and K, the scaled weights and the indexer K-cache
write in one Triton launch."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._indexer_prologue import launch_indexer_prologue
from ._indexer_reference import reference_k_rows, reference_q_rows, write_cache_rows

#: GLM-5.2's indexer: 32 heads of 128, the RoPE slice (first 64 values, interleaved pairs), the
#: k_norm eps, the UE8M0 scale format and softmax_scale * n_heads**-0.5. K and the weights are
#: column views of the fused-A output (kv_a_proj_with_mqa 2624 rows, wk 128, weights_proj 32).
_GLM = dict(
    n_heads=32,
    head_dim=128,
    rope_dim=64,
    interleave=True,
    use_ue8m0=True,
    eps=1e-6,
    weight_scale=128**-0.5 * 32**-0.5,
    tokens_per_block=64,
    fused_a_width=2784,
    k_col=2624,
)


def _spec(rows, first_position, **overrides) -> dict:
    spec = dict(
        _GLM,
        rows=rows,
        cache_rows=rows,
        first_position=first_position,
        k_layout="fused_a",
        pool_layout="contiguous",
        pages=16,
        write_cache=True,
        use_pdl=True,
        launch_dependents=True,
        prewait=True,
    )
    spec.update(overrides)
    return spec


class GlmIndexerPrologue(OpWrapper):
    """The whole indexer prologue of a step whose Q comes from a separate wq_b GEMM.

    For every row ``i`` (the K part is ``indexer_k_tail``'s, including the cache write when
    ``k_cache`` is given and ``num_cache_tokens`` > 0):

    - Q: RoPE on ``q[i, h, :rope_dim]`` at ``positions[i]``, rounded to bf16; per head
      ``q_scale[i, h] = max(amax, 1e-12) / 448`` (rounded up to a power of two with
      ``use_ue8m0``) and ``q_fp8[i, h] = e4m3(q * rcp(q_scale))``;
    - weights: ``(weights[i, h] * q_scale[i, h]) * weight_scale`` in fp32.

    Returns ``(q_fp8 [N, H, D], k_fp8 [N, D], k_scale [N, 1], weights [N, H] fp32,
    q_scale [N, H, 1])``, freshly allocated, the contract of the indexer's Q and K inputs to the
    MQA logits. PDL flags as in ``indexer_k_tail``.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, 8 rows: wq_b output Q, fused-A K/weights, cache write, PDL + pre-wait",
            spec=_spec(8, 529_981),
        ),
        Cell(
            why="decode, 8 rows, plain launch",
            spec=_spec(8, 1_048_560, use_pdl=False, launch_dependents=False, prewait=False),
        ),
        Cell(
            why="no cache write (the caller scatters the K rows itself)",
            spec=_spec(8, 300, write_cache=False, launch_dependents=False, prewait=False),
        ),
        Cell(
            why="padded graph rows: 8 rows, the first 3 written to the cache",
            spec=_spec(8, 60, cache_rows=3),
        ),
        Cell(
            why="context chunk: 256 rows of one request over 4+ pages, contiguous K and weights",
            spec=_spec(
                256,
                1_000,
                k_layout="contiguous",
                use_pdl=False,
                launch_dependents=False,
                prewait=False,
            ),
        ),
    )

    note = """
    Metadata consumed: none; every input is an argument (see ``indexer_k_tail`` for the K-side
    operands). ``q`` is the indexer wq_b output viewed as [N, 32, 128] bf16 and ``weights`` the
    weights_proj columns [2752, 2784) of the fused-A output (bf16).

    At the c1 decode cells the target takes the Q part from the wq_b epilogue and runs only
    ``indexer_k_tail``; this entry serves the forwards that epilogue does not.

    Not certified: fp32 K or weights, NeoX RoPE, the plain amax / 448 scale, other head counts or
    dims.
    """

    def __call__(
        self,
        indexer_k: torch.Tensor,
        weights: torch.Tensor,
        q: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
        weight_scale: float,
        rope_dim: int,
        interleave: bool,
        use_ue8m0: bool,
        k_cache: Optional[torch.Tensor],
        slot_mapping_fp8: Optional[torch.Tensor],
        slot_mapping_scale: Optional[torch.Tensor],
        num_cache_tokens: int,
        use_pdl: bool,
        launch_dependents: bool,
        prewait: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return launch_indexer_prologue(
            indexer_k,
            weights,
            q,
            positions,
            cos_sin,
            ln_weight,
            ln_bias,
            eps,
            weight_scale,
            rope_dim,
            interleave,
            use_ue8m0,
            k_cache,
            slot_mapping_fp8,
            slot_mapping_scale,
            num_cache_tokens,
            use_pdl,
            launch_dependents,
            prewait,
        )

    def reference(
        self,
        indexer_k: torch.Tensor,
        weights: torch.Tensor,
        q: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
        weight_scale: float,
        rope_dim: int,
        interleave: bool,
        use_ue8m0: bool,
        k_cache: Optional[torch.Tensor],
        slot_mapping_fp8: Optional[torch.Tensor],
        slot_mapping_scale: Optional[torch.Tensor],
        num_cache_tokens: int,
        use_pdl: bool,
        launch_dependents: bool,
        prewait: bool,
    ) -> Tuple[torch.Tensor, ...]:
        """``(q_fp8, k_fp8, k_scale, weights, q_scale, k_cache)``: the five outputs, then the cache
        after the writes (a fresh tensor; None without ``k_cache``). The launch flags change when
        the kernel runs, not what it computes."""
        q8, q_scale, w = reference_q_rows(
            q, weights, positions, cos_sin, weight_scale, rope_dim, interleave, use_ue8m0
        )
        k8, k_scale = reference_k_rows(
            indexer_k, positions, cos_sin, ln_weight, ln_bias, eps, rope_dim, interleave, use_ue8m0
        )
        cache = None
        if k_cache is not None:
            cache = k_cache.clone()
            if num_cache_tokens > 0:
                write_cache_rows(
                    cache, k8, k_scale, slot_mapping_fp8, slot_mapping_scale, num_cache_tokens
                )
        return q8, k8, k_scale, w, q_scale, cache

    def is_valid(
        self,
        indexer_k: torch.Tensor,
        weights: torch.Tensor,
        q: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
        weight_scale: float,
        rope_dim: int,
        interleave: bool,
        use_ue8m0: bool,
        k_cache: Optional[torch.Tensor],
        slot_mapping_fp8: Optional[torch.Tensor],
        slot_mapping_scale: Optional[torch.Tensor],
        num_cache_tokens: int,
        use_pdl: bool,
        launch_dependents: bool,
        prewait: bool,
    ) -> None:
        rows, n_heads, head_dim = q.shape
        # Every operand is read through a unit inner stride and fixed 128-wide rows.
        assert head_dim == 128 and indexer_k.shape == (rows, 128), (
            f"q must be [N, H, 128] and indexer_k [N, 128], got {tuple(q.shape)} and "
            f"{tuple(indexer_k.shape)}"
        )
        assert q.dtype == torch.bfloat16 and q.stride(2) == 1, "q must be bf16, unit inner stride"
        assert indexer_k.stride(1) == 1 and weights.stride(1) == 1, (
            "indexer_k and weights need a unit inner stride"
        )
        assert weights.shape == (rows, n_heads), (
            f"weights must be [N, H] = [{rows}, {n_heads}], got {tuple(weights.shape)}"
        )
        assert indexer_k.dtype in (torch.bfloat16, torch.float32), indexer_k.dtype
        assert weights.dtype in (torch.bfloat16, torch.float32), weights.dtype
        assert cos_sin.dtype == torch.float32 and cos_sin.dim() == 2, (
            "cos_sin must be a [max_pos, rope_dim] fp32 table"
        )
        # The sin half starts rope_dim / 2 columns into the row.
        assert cos_sin.shape[1] == rope_dim and cos_sin.stride(1) == 1, (
            f"cos_sin rows must be [cos | sin] of rope_dim={rope_dim} contiguous values"
        )
        assert rope_dim % 2 == 0 and 0 < rope_dim <= 128, f"rope_dim={rope_dim}"
        assert positions.dim() == 1 and positions.numel() >= rows, (
            "positions must hold one position per row"
        )
        if rows > 0:
            assert int(positions[:rows].max()) < cos_sin.shape[0], (
                "a position past the cos/sin table reads past its end"
            )
        assert ln_weight.numel() == 128 and ln_bias.numel() == 128, (
            "k_norm weight and bias hold 128 values"
        )
        if k_cache is not None and num_cache_tokens > 0:
            assert k_cache.dim() == 4 and k_cache.element_size() == 1, (
                "k_cache must be the 4-D byte view of the indexer K cache"
            )
            assert slot_mapping_fp8 is not None and slot_mapping_scale is not None, (
                "a cache write needs both slot mappings"
            )
            assert slot_mapping_fp8.dtype == torch.int64, "slot mappings are int64"
            assert slot_mapping_scale.dtype == torch.int64, "slot mappings are int64"
            assert num_cache_tokens <= rows, f"num_cache_tokens={num_cache_tokens} > {rows} rows"
            assert slot_mapping_fp8.numel() >= num_cache_tokens, "too few fp8 slots"
            assert slot_mapping_scale.numel() >= num_cache_tokens, "too few scale slots"
            limit = k_cache.numel()
            assert int(slot_mapping_fp8[:num_cache_tokens].max()) + 128 <= limit, (
                "an fp8 slot writes past the cache"
            )
            assert int(slot_mapping_scale[:num_cache_tokens].max()) + 4 <= limit, (
                "a scale slot writes past the cache"
            )


indexer_prologue = GlmIndexerPrologue()
