# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused index score + top-16 select of a decode span of at most ``dispatch.FUSED_SCORE_SELECT_MAX_REQUESTS``
requests (``ops/attention/m3_fused_score_select``), with its scratch and arrival counters.

One scratch [2, 8, ceil4(max_tiles)] and one zeroed counter pair per (device, engine block count) serve every sparse
layer: the layers run one after another on one stream, and the kernel returns each counter to zero. They are
created on an eager decode step; a CUDA-graph capture only reads them.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ..ops.attention.m3_fused_score_select import (
    MAX_DECODE_QUERY_LEN,
    MAX_ROWS,
    m3_fused_score_select,
    ring_fits,
    scratch_shape,
)

_BUFFERS: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def supported(
    idx_q: torch.Tensor,
    idx_k_pages: torch.Tensor,
    decode_query_len: int,
    num_kv_heads: int,
    max_tiles: int,
):
    """None when the kernel takes this span's geometry, else the reason it does not."""
    if (
        idx_q.dim() != 3
        or idx_q.dtype != torch.float8_e4m3fn
        or idx_k_pages.dtype != torch.float8_e4m3fn
    ):
        return "the index Q / index-K cache is not FP8"
    heads = int(idx_q.shape[1])
    rows = decode_query_len * num_kv_heads
    if (
        idx_q.shape[2] != 128
        or idx_k_pages.dim() != 3
        or tuple(idx_k_pages.shape[1:]) != (128, 128)
        or not 1 <= decode_query_len <= MAX_DECODE_QUERY_LEN
        or heads % num_kv_heads
        or heads * MAX_DECODE_QUERY_LEN > MAX_ROWS
        or rows > MAX_ROWS
    ):
        return f"{heads} index heads x {decode_query_len} query rows is not the kernel's geometry"
    if not ring_fits(rows, max_tiles):
        return f"{max_tiles} blocks of select scratch do not fit the stage ring"
    return None


def buffers(device: torch.device, max_tiles: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(scratch, counters) for this engine's block count; None during a capture that finds them missing."""
    key = (str(device), int(max_tiles))
    found = _BUFFERS.get(key)
    if found is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        found = _BUFFERS[key] = (
            torch.empty(scratch_shape(max_tiles), dtype=torch.float32, device=device),
            torch.zeros(scratch_shape(max_tiles)[0], dtype=torch.int32, device=device),
        )
    return found


def fused_score_select(
    idx_q,
    idx_k_pages,
    block_table,
    seq_lens,
    n_valid,
    *,
    decode_query_len,
    num_kv_heads,
    init_blocks,
    local_blocks,
    max_tiles,
    scratch,
    counters,
):
    """The [total_q, num_kv_heads, 16] table (a view of a head-major [num_kv_heads, total_q, 16] backing)."""
    total_q = int(idx_q.shape[0])
    out = torch.empty((num_kv_heads, total_q, 16), dtype=torch.int32, device=idx_q.device).permute(
        1, 0, 2
    )
    return m3_fused_score_select(
        idx_q,
        idx_k_pages,
        block_table,
        seq_lens,
        n_valid,
        out,
        scratch,
        counters,
        int(decode_query_len),
        int(num_kv_heads),
        int(init_blocks),
        int(local_blocks),
        int(max_tiles),
        get_env_enable_pdl(),
    )
