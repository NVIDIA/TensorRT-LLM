# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexer K tail: k_norm, RoPE, E4M3 K and the indexer K-cache write in one Triton launch."""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._indexer_prologue import launch_indexer_k_tail
from ._indexer_reference import reference_k_rows, write_cache_rows

#: GLM-5.2's indexer: head dim, RoPE slice (first 64 values, interleaved pairs), k_norm eps, the
#: UE8M0 scale format, and the indexer K cache's page (64 tokens of 128 FP8 bytes, then their 64
#: fp32 scales). The pre-norm K is a column view of the fused-A output, whose rows are the
#: kv_a_proj_with_mqa (2624), indexer wk (128) and weights_proj (32) rows.
_GLM = dict(
    head_dim=128,
    rope_dim=64,
    interleave=True,
    use_ue8m0=True,
    eps=1e-6,
    tokens_per_block=64,
    fused_a_width=2784,
    k_col=2624,
)


def _spec(rows, first_position, **overrides) -> dict:
    """``rows`` K rows of one request at positions first_position, first_position + 1, ..."""
    spec = dict(
        _GLM,
        rows=rows,
        cache_rows=rows,
        first_position=first_position,
        k_layout="fused_a",
        pool_layout="contiguous",
        pages=16,
        negative_slot=False,
        use_pdl=True,
        launch_dependents=True,
        prewait=True,
    )
    spec.update(overrides)
    return spec


class GlmIndexerKTail(OpWrapper):
    """The K part of the fused indexer prologue for rows whose Q part another kernel computes.

    For every row ``i`` of ``indexer_k`` [N, 128] (the pre-norm indexer K):

    1. ``kn = LayerNorm(bf16(indexer_k[i]); ln_weight, ln_bias, eps)``, fp32 math, rounded to bf16;
    2. RoPE on ``kn[:rope_dim]`` at ``positions[i]`` with the ``[cos | sin]`` row of ``cos_sin``
       (interleaved pairs when ``interleave``), ``fma(x, cos, (+-partner) * sin)``, rounded to bf16;
    3. ``k_scale[i] = max(amax, 1e-12) / 448``, rounded up to a power of two with ``use_ue8m0``;
       ``k_fp8[i] = e4m3(k * rcp(k_scale[i]))``;
    4. for ``i < num_cache_tokens`` with both slots >= 0, the 128 FP8 bytes into
       ``k_cache`` at the flat byte offset ``slot_mapping_fp8[i]`` and the 4 scale bytes at
       ``slot_mapping_scale[i]``, each 4-byte chunk unravelled over ``k_cache``'s strided view
       (``indexer_k_cache_scatter_op``'s rule).

    Returns ``(k_fp8, k_scale, k_cache)``, all written in place.

    With ``use_pdl`` the kernel waits on ``griddepcontrol.wait`` before reading its inputs;
    ``launch_dependents`` lets the next kernel of the stream launch right after that wait (only for
    the early-KV decode MQA logits, which read the new cache rows after their own wait); ``prewait``
    loads the positions, cos/sin rows, the LayerNorm parameters and the slot mappings before the
    wait (only when the step's metadata preparation wrote them before the kernels this one waits
    for).
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="verify / draft step 0: 8 rows from fused-A columns, PDL trigger + pre-wait",
            spec=_spec(8, 529_981),
        ),
        Cell(
            why="8 rows at the end of a 1M-token context (cos/sin rows near 2^20)",
            spec=_spec(8, 1_048_560),
        ),
        Cell(
            why="plain launch, PDL off",
            spec=_spec(8, 4_093, use_pdl=False, launch_dependents=False, prewait=False),
        ),
        Cell(
            why="PDL trigger without the pre-wait loads",
            spec=_spec(8, 70_000, prewait=False),
        ),
        Cell(
            why="padded graph rows: 8 K rows, the first 5 written to the cache",
            spec=_spec(8, 130, cache_rows=5),
        ),
        Cell(
            why="a slot of -1 leaves that row out of the cache",
            spec=_spec(8, 640, negative_slot=True),
        ),
        Cell(
            why="cache pages interleaved with other layers' pages (strided page dim)",
            spec=_spec(8, 63, pool_layout="interleaved"),
        ),
        Cell(
            why="contiguous pre-norm K (a separate wk|weights_proj GEMM)",
            spec=_spec(8, 9_000, k_layout="contiguous"),
        ),
    )

    note = """
    Metadata consumed: none; every input is an argument. The target passes the K columns
    [2624, 2752) of the fused-A output, the step's ``position_ids`` (int32), its cos/sin cache
    ``[max_pos, 64]`` fp32 (``[cos | sin]`` per row), the layer's ``k_norm`` weight and bias, the
    layer's indexer K-cache view ``[pages, 64, 1, 132]`` uint8 and the step's
    ``slot_mapping_fp8`` / ``slot_mapping_scale`` (int64 flat byte offsets), with
    ``num_cache_tokens`` = the step's token count.

    ``k_fp8`` / ``k_scale`` are written in full (N rows) even where only the cache is read
    afterwards. A slot < 0 skips the row's cache write, as the scatter op does.

    Not certified: fp32 pre-norm K, NeoX RoPE (``interleave=False``), the plain amax / 448 scale
    (``use_ue8m0=False``), other head or RoPE dims, more than 8 rows.
    """

    def __call__(
        self,
        indexer_k: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
        rope_dim: int,
        interleave: bool,
        use_ue8m0: bool,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_cache_tokens: int,
        use_pdl: bool,
        launch_dependents: bool,
        prewait: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        launch_indexer_k_tail(
            indexer_k,
            positions,
            cos_sin,
            ln_weight,
            ln_bias,
            eps,
            rope_dim,
            interleave,
            use_ue8m0,
            k_fp8,
            k_scale,
            k_cache,
            slot_mapping_fp8,
            slot_mapping_scale,
            num_cache_tokens,
            use_pdl,
            launch_dependents,
            prewait,
        )
        return k_fp8, k_scale, k_cache

    def reference(
        self,
        indexer_k: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
        rope_dim: int,
        interleave: bool,
        use_ue8m0: bool,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_cache_tokens: int,
        use_pdl: bool,
        launch_dependents: bool,
        prewait: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fresh ``(k_fp8, k_scale, k_cache)`` from plain torch, fp32 math rounded where the kernel
        rounds. PDL changes when the kernel runs, not what it computes, so the three launch flags
        are unused."""
        k8, scale = reference_k_rows(
            indexer_k, positions, cos_sin, ln_weight, ln_bias, eps, rope_dim, interleave, use_ue8m0
        )
        cache = k_cache.clone()
        write_cache_rows(cache, k8, scale, slot_mapping_fp8, slot_mapping_scale, num_cache_tokens)
        return k8, scale, cache

    def is_valid(
        self,
        indexer_k: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
        rope_dim: int,
        interleave: bool,
        use_ue8m0: bool,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_cache_tokens: int,
        use_pdl: bool,
        launch_dependents: bool,
        prewait: bool,
    ) -> None:
        rows = indexer_k.shape[0]
        # The kernel reads every operand through a unit inner stride and a fixed 128-wide row.
        assert indexer_k.dim() == 2 and indexer_k.shape[1] == 128, (
            f"indexer_k must be [N, 128], got {tuple(indexer_k.shape)}"
        )
        assert indexer_k.stride(1) == 1, "indexer_k needs a unit inner stride"
        assert indexer_k.dtype in (torch.bfloat16, torch.float32), indexer_k.dtype
        # The sin half starts rope_dim / 2 columns into the row: a wider or narrower table reads
        # the wrong values without failing.
        assert cos_sin.dtype == torch.float32 and cos_sin.dim() == 2, (
            "cos_sin must be a [max_pos, rope_dim] fp32 table"
        )
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
        # The outputs are written as dense [N, 128] / [N, 1] rows.
        assert k_fp8.shape == (rows, 128) and k_fp8.is_contiguous(), (
            f"k_fp8 must be a contiguous [N, 128] tensor, got {tuple(k_fp8.shape)}"
        )
        assert k_fp8.dtype == torch.float8_e4m3fn, k_fp8.dtype
        assert k_scale.shape == (rows, 1) and k_scale.is_contiguous(), (
            f"k_scale must be a contiguous [N, 1] tensor, got {tuple(k_scale.shape)}"
        )
        assert k_scale.dtype == torch.float32, k_scale.dtype
        assert k_cache.dim() == 4 and k_cache.element_size() == 1, (
            "k_cache must be the 4-D byte view of the indexer K cache"
        )
        assert slot_mapping_fp8.dtype == torch.int64 and slot_mapping_scale.dtype == torch.int64, (
            "slot mappings are int64 flat byte offsets"
        )
        assert 0 < num_cache_tokens <= rows, (
            f"num_cache_tokens={num_cache_tokens} must be in [1, {rows}]"
        )
        assert slot_mapping_fp8.numel() >= num_cache_tokens, "too few fp8 slots"
        assert slot_mapping_scale.numel() >= num_cache_tokens, "too few scale slots"
        limit = k_cache.numel()
        written = slot_mapping_fp8[:num_cache_tokens]
        assert int(written.max()) + 128 <= limit, "an fp8 slot writes past the cache"
        assert int(slot_mapping_scale[:num_cache_tokens].max()) + 4 <= limit, (
            "a scale slot writes past the cache"
        )


indexer_k_tail = GlmIndexerKTail()
