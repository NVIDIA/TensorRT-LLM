# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer FP8 prologue: quantize the indexer Q and K, scale the head weights and write the
keys into the paged index-key cache, in one launch (CuTe DSL)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.dsa_fp8_decode_prologue import (  # noqa: E501
    HEAD_DIM,
    dsa_fp8_decode_prologue_into,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.fused_cat_fp8 import (
    fused_cat_fp8,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_k_cache_scatter_op import (  # noqa: E501
    indexer_k_cache_scatter_op,
)

#: Token counts of the decode steps the K3.1 bar recipes run (requests x tokens per request).
BAR_TOKENS = (6, 8, 16, 24, 32, 72, 98, 224)


def _spec(tokens, **overrides) -> dict:
    spec = dict(
        tokens=tokens,
        rows=None,  # rows of q / k / weights (>= tokens: the padded projection outputs)
        heads=32,
        tokens_per_block=64,
        num_pages=64,
        page_stride_pad=0,
        data="randn",
        negative_slots=False,
        latent=False,  # also the MLA latent_cache copy (kv_lora_rank 512 + rope 64)
    )
    spec.update(overrides)
    return spec


class DsaFp8DecodePrologue(OpWrapper):
    """For the first ``num_tokens`` tokens ``t``:

    - ``q_fp8`` [T * H, 128] E4M3, ``q_scale`` [T * H, 1] fp32: ``fused_cat_fp8`` of each head row
      ``q[t, h]`` (the two 64-wide halves of one 128-wide row), one scale per row;
    - ``weights`` [T, H] fp32: ``weights_bf16[t] * q_scale[t * H : (t + 1) * H] * weight_scale``;
    - ``k_fp8`` [T, 128] E4M3, ``k_scale`` [T, 1] fp32: ``fused_cat_fp8`` of ``k[t]``;
    - ``k_cache``: ``indexer_k_cache_scatter_op`` of ``(k_fp8, k_scale)`` at the slot mappings (a
      token with a negative key or scale slot is not written), in place;
    - with ``compressed_kv`` and ``k_pe`` (bf16 [>= T, C] and [>= T, R], 16-byte aligned rows,
      ``C`` and ``R`` multiples of 8), ``latent_cache`` [T, C + R] bf16 =
      ``cat([compressed_kv[:T], k_pe[:T]], -1)``; else None.

    ``q`` is bf16 [>= T, H, 128] (contiguous heads, 16-byte aligned rows), ``k`` bf16 [>= T, 128]
    (16-byte aligned rows), ``weights_bf16`` bf16 [>= T, H] with a unit inner stride (a column
    view of the fused wk | weights_proj output), ``k_cache`` the [pages, P, 1, 132] uint8 view with
    contiguous tokens within a page, the slot mappings int64 [>= T] flat byte offsets. Returns
    ``(q_fp8, q_scale, weights, k_fp8, k_scale, latent_cache)``, newly allocated.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"a {tokens}-token decode step of the bar recipes",
                spec=_spec(tokens),
            )
            for tokens in BAR_TOKENS
        ),
        Cell(
            why="padded projection outputs: 27 of 40 rows are the step's tokens",
            spec=_spec(27, rows=40),
        ),
        Cell(
            why="negative slots: those tokens' keys stay out of the cache",
            spec=_spec(16, negative_slots=True),
        ),
        Cell(
            why="all-zero rows: the 1e-12 amax floor",
            spec=_spec(8, data="zeros_row"),
        ),
        Cell(
            why="large-magnitude rows: scales far from 1",
            spec=_spec(32, data="wide"),
        ),
        Cell(
            why="a padded page stride: the V2 manager's per-layer view of a wider page",
            spec=_spec(32, page_stride_pad=256),
        ),
        Cell(
            why="a page stride that is 4- but not 8-byte aligned: 4-byte cache stores",
            spec=_spec(24, page_stride_pad=4),
        ),
        Cell(
            why="32-token pages",
            spec=_spec(24, tokens_per_block=32),
        ),
        Cell(
            why="128-token pages",
            spec=_spec(24, tokens_per_block=128),
        ),
        Cell(
            why="a 3000-token prefill chunk over random pages",
            spec=_spec(3000, num_pages=96),
        ),
        *(
            Cell(
                why=f"with the latent_cache copy: {tokens} tokens",
                spec=_spec(tokens, latent=True),
            )
            for tokens in (8, 98)
        ),
        Cell(
            why="with the latent_cache copy: padded rows, 27 of 40 are the step's tokens",
            spec=_spec(27, rows=40, latent=True),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).

    Replaces, per DSA layer per step, the chain fused_cat_fp8(Q) + fused_cat_fp8(K) + the torch
    weight product + indexer_k_cache_scatter_op: the target passes the wq_b output viewed
    [T, 32, 128], the k LayerNorm output [T, 128], the weights_proj columns of the fused
    wk | weights_proj output, weight_scale = 128**-0.5 * 32**-0.5, the layer's index-key cache
    view, the metadata's slot mappings and the step's token count; the scales are UE8M0 (the
    indexer's scale_fmt). The kernel uses the chain's instructions (rcp.approx.ftz reciprocal, RN
    satfinite E4M3 conversion, two RN fp32 multiplies for the weights) and every output -- q_fp8,
    q_scale, weights, k_fp8, k_scale and every cache byte -- equals the chain's bit for bit. Byte
    stores only into the slots; distinct tokens must map to distinct slots.
    With compressed_kv and k_pe the same launch also writes the MLA latent_cache (the
    torch.concat of the two), a byte copy; pass None for both to skip it.

    One launch with PDL (griddepcontrol.wait before the first global read, then the dependent
    grid is released); compiles once per (heads, page size, cache store width, latent widths),
    which must happen outside CUDA graph capture (the target's warmup step).
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights_bf16: torch.Tensor,
        weight_scale: float,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
        compressed_kv: Optional[torch.Tensor],
        k_pe: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        heads = q.shape[1]
        device = q.device
        q_fp8 = torch.empty(num_tokens * heads, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
        q_scale = torch.empty(num_tokens * heads, 1, dtype=torch.float32, device=device)
        weights = torch.empty(num_tokens, heads, dtype=torch.float32, device=device)
        k_fp8 = torch.empty(num_tokens, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
        k_scale = torch.empty(num_tokens, 1, dtype=torch.float32, device=device)
        latent_cache = None
        if compressed_kv is not None:
            latent_cache = torch.empty(
                num_tokens,
                compressed_kv.shape[-1] + k_pe.shape[-1],
                dtype=compressed_kv.dtype,
                device=device,
            )
        dsa_fp8_decode_prologue_into(
            q,
            k,
            weights_bf16,
            weight_scale,
            num_tokens,
            q_fp8,
            q_scale,
            weights,
            k_fp8,
            k_scale,
            k_cache,
            slot_mapping_fp8,
            slot_mapping_scale,
            compressed_kv,
            k_pe,
            latent_cache,
        )
        return q_fp8, q_scale, weights, k_fp8, k_scale, latent_cache

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights_bf16: torch.Tensor,
        weight_scale: float,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
        compressed_kv: Optional[torch.Tensor],
        k_pe: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """The chain this op replaces, run on the first ``num_tokens`` rows: the family's
        ``fused_cat_fp8`` and ``indexer_k_cache_scatter_op`` and the torch weight product, in the
        indexer's order, and the torch.concat of the MLA latent. Writes ``k_cache`` in place."""
        heads = q.shape[1]
        half = HEAD_DIM // 2
        # The indexer has no rope: each row is passed as its two 64-wide halves.
        q_lo, q_hi = q[:num_tokens].split([half, half], dim=-1)
        k_lo, k_hi = k[:num_tokens].split([half, half], dim=-1)
        q_fp8, q_scale = fused_cat_fp8(q_lo, q_hi)
        k_fp8, k_scale = fused_cat_fp8(k_lo, k_hi)
        weights = (
            weights_bf16[:num_tokens]
            * q_scale.view(num_tokens, heads, 1).squeeze(-1)
            * weight_scale
        )
        indexer_k_cache_scatter_op(
            k_fp8,
            k_scale.view(torch.int32),
            k_cache,
            slot_mapping_fp8,
            slot_mapping_scale,
            num_tokens,
        )
        latent_cache = None
        if compressed_kv is not None:
            latent_cache = torch.concat([compressed_kv[:num_tokens], k_pe[:num_tokens]], dim=-1)
        return q_fp8, q_scale, weights, k_fp8, k_scale, latent_cache

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights_bf16: torch.Tensor,
        weight_scale: float,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
        compressed_kv: Optional[torch.Tensor],
        k_pe: Optional[torch.Tensor],
    ) -> None:
        # Rows past the inputs are read from whatever follows them.
        assert q.shape[0] >= num_tokens and k.shape[0] >= num_tokens, "num_tokens exceeds q or k"
        assert weights_bf16.shape[0] >= num_tokens and weights_bf16.shape[1] == q.shape[1], (
            "weights_bf16 must hold one value per head of each token"
        )
        assert k.shape[-1] == HEAD_DIM, f"k rows must be {HEAD_DIM} wide, got {k.shape[-1]}"
        assert slot_mapping_fp8.dtype == torch.int64 and slot_mapping_scale.dtype == torch.int64, (
            "slot mappings are int64 flat byte offsets"
        )
        assert (
            slot_mapping_fp8.shape[0] >= num_tokens and slot_mapping_scale.shape[0] >= num_tokens
        ), "num_tokens exceeds the slot mappings"
        if compressed_kv is not None:
            assert k_pe is not None, "the latent_cache copy takes both compressed_kv and k_pe"
            assert compressed_kv.shape[0] >= num_tokens and k_pe.shape[0] >= num_tokens, (
                "num_tokens exceeds compressed_kv or k_pe"
            )
        # Offsets past the cache's logical size write into whatever follows it.
        if num_tokens > 0:
            logical = k_cache.numel()
            assert int(slot_mapping_fp8[:num_tokens].max()) + HEAD_DIM <= logical, (
                "a key slot past the cache"
            )
            assert int(slot_mapping_scale[:num_tokens].max()) + 4 <= logical, (
                "a scale slot past the cache"
            )
            # Two tokens on one slot race: which one the cache keeps is unspecified.
            written = slot_mapping_fp8[:num_tokens]
            written = written[written >= 0]
            assert written.unique().numel() == written.numel(), "two tokens share a key slot"


dsa_fp8_decode_prologue = DsaFp8DecodePrologue()
