# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged-cache writes shared by the MiniMax-M3 kernels and backends."""

from __future__ import annotations

from typing import Literal

import torch


def write_kv_slots(
    cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    values: torch.Tensor,
    *,
    layout: Literal["NHD", "HND"] = "NHD",
) -> None:
    """Write per-token values into a K, V, or index-K cache at given slots.

    Handles a 3-D flat-slot cache and a 4-D paged view. `layout` sets the paged
    axis order: "NHD" is [num_pages, tokens_per_block, num_heads, channel],
    "HND" is [num_pages, num_heads, tokens_per_block, channel]. The paged view
    is non-contiguous, so the slot id is split into (page, within) and written
    by multi-dim assignment. `values` is always [num_tokens, num_heads, channel].

    Callers must provide valid slots for every live token. The production M3
    mapping satisfies this contract: ``get_block_ids_per_seq`` canonicalizes
    padded ``BAD_PAGE_INDEX`` entries before ``build_paged_kv_slot_mapping``
    selects only the allocated live-token positions.
    """
    with torch.no_grad():
        if cache.ndim >= 4:
            token_axis = 2 if layout == "HND" else 1
            tokens_per_block = int(cache.shape[token_axis])
            out_long = out_cache_loc.to(torch.long)
            page = out_long // tokens_per_block
            within = out_long % tokens_per_block
            if layout == "HND":
                # Advanced indices on dims 0 and 2 broadcast to [num_tokens] and
                # move front, giving a [num_tokens, num_heads, channel] target.
                cache[page, :, within, :] = values.to(cache.dtype)
            else:
                cache[page, within] = values.to(cache.dtype)
        else:
            cache.index_copy_(0, out_cache_loc.to(torch.long), values.to(cache.dtype))


__all__ = ["write_kv_slots"]
