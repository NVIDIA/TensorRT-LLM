# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs the GLM indexer entries' tests share: the indexer K cache, its slot mappings and block
table, the RoPE table, and the comparison of E4M3 rows quantized with per-row scales.

The cache layout is the engine's: an INDEX_KEY page holds 64 tokens' 128 FP8 bytes, then their 64
fp32 scales, viewed as [pages, 64, 1, 132] uint8; a token's bytes are found through the slot
mappings, flat byte offsets computed exactly as the DSA metadata does.
"""

from __future__ import annotations

from typing import Tuple

import torch

PAGE = 64
HEAD_DIM = 128
TOKEN_BYTES = HEAD_DIM + 4
SENTINEL = 0xA5


def cos_sin_table(rows: int, rope_dim: int = 64, theta: float = 8.0e6) -> torch.Tensor:
    """[rows, rope_dim] fp32, ``[cos | sin]`` of the default RoPE at every position."""
    exponents = torch.arange(0, rope_dim, 2, dtype=torch.float64, device="cuda") / rope_dim
    inv_freq = 1.0 / (theta**exponents)
    angles = torch.arange(rows, dtype=torch.float64, device="cuda")[:, None] * inv_freq[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=1).float()


def k_cache_pool(pages: int, layout: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """(this layer's [pages, 64, 1, 132] uint8 view, the whole allocation), filled with a
    sentinel. ``interleaved`` puts two other layers' pages between this layer's pages (a strided
    page dimension); ``contiguous`` is one layer's own pool."""
    if layout == "interleaved":
        whole = torch.full((pages, 3, PAGE * TOKEN_BYTES), SENTINEL, dtype=torch.uint8).cuda()
        view = whole[:, 1].view(pages, PAGE, 1, TOKEN_BYTES)
        assert not view.is_contiguous()
        return view, whole
    whole = torch.full((pages, PAGE, 1, TOKEN_BYTES), SENTINEL, dtype=torch.uint8).cuda()
    return whole, whole


def block_row(positions: torch.Tensor, pool_pages: int, gen: torch.Generator) -> torch.Tensor:
    """One request's INDEX_KEY block-table row up to the page of its last position: the pages the
    positions touch map to distinct random physical pages (so no two rows share a slot), the
    others to page 0."""
    pages = positions.long() // PAGE
    touched = torch.unique(pages)
    assert touched.numel() <= pool_pages, "the pool is smaller than the pages the rows touch"
    row = torch.zeros(int(pages.max()) + 1, dtype=torch.int32)
    row[touched] = torch.randperm(pool_pages, generator=gen)[: touched.numel()].to(torch.int32)
    return row


def slot_mappings(positions: torch.Tensor, row: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(fp8 slots, scale slots) int64 flat byte offsets of each position's K row, the metadata's
    ``_compute_slot_mappings`` for one request."""
    pos = positions.long().cpu()
    block_ids = row.long()[pos // PAGE]
    in_page = pos % PAGE
    page_stride = PAGE * TOKEN_BYTES
    fp8 = block_ids * page_stride + in_page * HEAD_DIM
    scale = block_ids * page_stride + PAGE * HEAD_DIM + in_page * 4
    return fp8.cuda(), scale.cuda()


def e4m3_step(units: torch.Tensor) -> torch.Tensor:
    """Spacing of E4M3 values at magnitude ``units`` (in units of the scale)."""
    exponent = torch.floor(torch.log2(units.clamp_min(2.0**-6)))
    return torch.pow(2.0, exponent - 3)


def assert_quantized_close(
    q8: torch.Tensor,
    scale: torch.Tensor,
    q8_ref: torch.Tensor,
    scale_ref: torch.Tensor,
    max_mismatch: float = 0.01,
) -> None:
    """E4M3 rows with per-row scales against a reference that rounds at the same points.

    The kernel and the reference compute the same fp32 expressions, but the LayerNorm reduction
    order and the MUFU rsqrt can move an fp32 intermediate by an ulp, and a bf16 rounding of that
    intermediate then flips with probability of about 2^-16 per value. A flipped bf16 value moves
    its E4M3 code by at most one step (and the row scale, a power of two, only when the row's
    maximum crosses one). So every dequantized value is within one E4M3 step of the reference, at
    the coarser of the two scales, and almost all codes match exactly.
    """
    assert q8.shape == q8_ref.shape and scale.shape == scale_ref.shape
    deq = q8.float() * scale
    deq_ref = q8_ref.float() * scale_ref
    step = torch.maximum(
        scale * e4m3_step(q8.float().abs()), scale_ref * e4m3_step(q8_ref.float().abs())
    )
    excess = ((deq - deq_ref).abs() - step).clamp_min(0)
    assert excess.max().item() == 0, (
        f"a value is more than one E4M3 step from the reference (by {excess.max().item():.3e})"
    )
    mismatch = (q8.view(torch.uint8) != q8_ref.view(torch.uint8)).float().mean().item()
    assert mismatch <= max_mismatch, f"{mismatch:.2%} of the E4M3 codes differ"
