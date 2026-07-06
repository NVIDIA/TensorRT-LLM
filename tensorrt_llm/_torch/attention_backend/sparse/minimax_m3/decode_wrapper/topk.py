# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-driven top-k KV-block selection for M3 sparse decode.

Replaces ``fmha_sm100.sparse_topk_select`` on the decode path.  The MSA
kernel takes ``num_valid_pages`` as a **single global host int** (the
max over the batch), which is both CUDA-graph-hostile (it varies per
step) and imprecise for heterogeneous batches (the forced "local"
window is anchored at the *global* last block instead of each row's
own last valid block).

This implementation reads per-row valid page counts from a device
tensor, matching the in-tree Triton reference semantics
(``backend.py:_index_attention_and_select``: force first
``init_blocks`` and last ``local_blocks`` *valid* blocks per row, mask
invalid blocks, pick top-k, ascending indices, ``-1`` tail padding)
while staying pure torch ops — fully CUDA-graph-capturable.
"""

from __future__ import annotations

from typing import Optional

import torch

# Finite sentinel mirroring MSA's FLT_MAX forced-score marker (avoids
# inf arithmetic edge cases in torch.topk).
_FORCE_SCORE = torch.finfo(torch.float32).max
# Sort key sentinel that pushes -1 (invalid) entries to the tail while
# staying well inside int32.
_PAD_KEY = 0x40000000


def select_topk_blocks(
    max_score_kv: torch.Tensor,
    valid_pages: torch.Tensor,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select per-(token, kv-head) top-k KV block indices.

    Parameters
    ----------
    max_score_kv : ``[num_kv_heads, max_k_tiles, total_q]`` float32.
        Per-128-token-block max scores (already group-reduced to KV-head
        granularity).  Blocks beyond a row's valid count may hold any
        value — they are masked here.
    valid_pages : ``[total_q]`` int32/int64 device tensor.
        Per-token count of valid KV blocks (``ceil(kv_len / page_size)``).
    topk : number of blocks to select (M3: 16).
    init_blocks : always include blocks ``[0, init_blocks)``.
    local_blocks : always include blocks
        ``[valid - local_blocks, valid)`` (per row).
    out : optional ``[total_q, num_kv_heads, topk]`` int32 buffer.

    Returns
    -------
    ``[total_q, num_kv_heads, topk]`` int32, ascending block indices,
    ``-1`` padded at the tail (the sparse FMHA kernel's
    ``kv_block_indexes`` contract).
    """
    num_kv_heads, max_k_tiles, total_q = max_score_kv.shape
    if max_k_tiles < topk:
        raise ValueError(f"max_k_tiles ({max_k_tiles}) must be >= topk ({topk})")

    scores = max_score_kv.permute(2, 0, 1)  # [total_q, H_kv, K]
    k_idx = torch.arange(max_k_tiles, device=scores.device, dtype=torch.int64)
    valid = valid_pages.to(torch.int64).view(total_q, 1, 1)

    forced = k_idx < init_blocks
    if local_blocks > 0:
        forced = forced | ((k_idx >= valid - local_blocks) & (k_idx < valid))
    else:
        forced = forced.expand(total_q, 1, max_k_tiles)

    scores = torch.where(forced, scores.new_full((), _FORCE_SCORE), scores)
    scores = torch.where(k_idx >= valid, scores.new_full((), float("-inf")), scores)

    top_scores, top_idx = torch.topk(scores, topk, dim=-1)
    # Rows with fewer than topk valid blocks pick -inf slots: pad them.
    top_idx = torch.where(top_scores == float("-inf"), -1, top_idx)

    # Ascending by block index with -1 at the tail.
    sort_key = torch.where(top_idx < 0, _PAD_KEY, top_idx)
    sort_key, _ = torch.sort(sort_key, dim=-1)
    result = torch.where(sort_key == _PAD_KEY, -1, sort_key).to(torch.int32)

    if out is not None:
        out.copy_(result)
        return out
    return result


__all__ = ["select_topk_blocks"]
