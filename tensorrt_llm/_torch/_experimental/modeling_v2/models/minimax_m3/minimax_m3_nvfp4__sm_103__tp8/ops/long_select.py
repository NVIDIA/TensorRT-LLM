# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-16 block select for MiniMax-M3 decode rows of 129 to 9216 blocks.

Same contract and bit-identical output as ``torch.ops.trtllm.minimax_m3_select_blocks`` on per-KV-head scores
[num_kv_heads, n_blocks, total_q]: init / local forcing, per-query valid-block count, ties to the smaller block index,
forced -inf picks emitted as -1, ascending block ids with -1 tail padding.

One 1024-thread CTA per (query, KV head) row holds the row's ordered score keys in registers (2, 4 or 9 per
thread, by the row's valid count). The weakest of the 16 maxima of the interleaved groups ``block % 16`` is a lower
bound on the 16th-largest key, so the top 16 is among the keys at or above it; those are compacted to shared memory
and ranked exactly by (key, block). More than 1024 keys tied at the bound (degenerate rows) rank every candidate
against the whole row (re-read).

The kernel is ``torch.ops.trtllm.minimax_m3_select_long``, called through this target's ``m3_select_long`` entry.
"""

from typing import Optional

import torch

from ..ops.attention.m3_select_long import MAX_BLOCKS, m3_select_long


def long_select_blocks(
    max_score_kv: torch.Tensor,
    *,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    use_pdl: bool = False,
    early_trigger: bool = False,
) -> Optional[torch.Tensor]:
    """Top-16 select over per-KV-head scores [num_kv_heads, n_blocks, total_q] with 129..9216 blocks.

    Returns the head-major backed [total_q, num_kv_heads, 16] int32 table, or None outside this kernel's shapes
    (callers keep the reference op). ``early_trigger`` releases the dependent sparse decode right after the select's
    grid wait (``SELECT_EARLY_MAX_ROWS`` in modeling.py).
    """
    if (
        max_score_kv.dtype != torch.float32
        or not max_score_kv.is_cuda
        or max_score_kv.dim() != 3
        or not 128 < max_score_kv.shape[1] <= MAX_BLOCKS
        or n_valid_blocks.dtype != torch.int32
        or not n_valid_blocks.is_cuda
        or n_valid_blocks.shape[0] < max_score_kv.shape[2]
    ):
        return None
    num_kv_heads, _, total_q = max_score_kv.shape
    out = torch.empty(
        (num_kv_heads, total_q, 16), dtype=torch.int32, device=max_score_kv.device
    ).permute(1, 0, 2)
    return m3_select_long(
        max_score_kv,
        n_valid_blocks.contiguous(),
        out,
        init_blocks,
        local_blocks,
        use_pdl,
        early_trigger,
    )
