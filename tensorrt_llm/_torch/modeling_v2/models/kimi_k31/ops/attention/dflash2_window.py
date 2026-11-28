# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The exact per-query sliding window of the DFlash2 MLA drafter's block decode.

The drafter's attention is causal with a sliding window of ``W`` positions: the block query at
position ``p`` attends the keys at positions ``[max(0, p - W + 1), p]``, the context rows below the
block start ``L`` and the block's own rows ``L .. p`` alike. Every key lives in the request's
paged latent pool (the block's latents are stored at ``L + j`` before the decode), so the window
of a query is a contiguous position range, and the decode is a sparse MLA over per-row index
lists of global pool rows (``page * page_size + offset``), ``-1`` past each row's live length.

:func:`dflash2_window_rows` builds those lists (device ops only, fixed shapes: CUDA-graph
capturable). :func:`sparse_mla_rows_eager` is the exact eager decode over them.
"""

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_ring import (
    dflash2_ring_positions,
)


def dflash2_window_rows(
    ctx_len: torch.Tensor,
    page_tables: torch.Tensor,
    block_size: int,
    page_size: int,
    window: int,
    width: int,
    ring_rows: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-query pool rows of the causal sliding window, and their live lengths.

    Args:
        ctx_len: ``[B]`` committed context length ``L`` of each request (the block starts there).
        page_tables: ``[B, max_pages]`` pool pages of each request, by position page.
        block_size: queries per request (block positions ``L .. L + block_size - 1``).
        page_size: tokens per pool page.
        window: positions a query attends, itself included (``sliding_window``).
        width: index-list width, at least ``window``; entries past a row's length are ``-1``.
        ring_rows: rows per request of a context ring (:mod:`..spec.dflash2_ring`): position ``p``
            is read from request row ``p mod ring_rows``. None for a dense table.

    Returns:
        ``(rows, lengths)``: int32 ``[B * block_size, width]`` global pool rows (ascending
        positions, request-major query rows) and int32 ``[B * block_size]`` live lengths.
    """
    if width < window:
        raise ValueError(f"index width {width} cannot hold a {window}-position window")
    if ring_rows and ring_rows < window + block_size - 1:
        raise ValueError(
            f"a {ring_rows}-row ring cannot hold a {window}-position window and its block"
        )
    device = ctx_len.device
    batch = ctx_len.shape[0]
    pos_q = ctx_len.long().view(batch, 1) + torch.arange(block_size, device=device)
    start = (pos_q - (window - 1)).clamp_min(0)
    lengths = pos_q - start + 1
    offs = torch.arange(width, device=device)
    pos = start.unsqueeze(-1) + offs
    valid = offs < lengths.unsqueeze(-1)
    pos = torch.where(valid, pos, torch.zeros_like(pos))
    pos = dflash2_ring_positions(pos, ring_rows)
    page_col = (pos // page_size).clamp_max(page_tables.shape[1] - 1)
    pages = page_tables.long().gather(1, page_col.view(batch, -1)).view_as(pos)
    rows = pages * page_size + pos % page_size
    rows = torch.where(valid, rows, torch.full_like(rows, -1))
    return (
        rows.view(batch * block_size, width).to(torch.int32),
        lengths.view(-1).to(torch.int32),
    )


def sparse_mla_rows_eager(
    query: torch.Tensor,
    pool_rows: torch.Tensor,
    rows: torch.Tensor,
    softmax_scale: float,
    latent_dim: int,
    kv_scale: float = 1.0,
) -> torch.Tensor:
    """Absorbed MLA over per-query pool rows, in FP32: the reference of the sparse kernels.

    Args:
        query: ``[R, H, latent_dim + rope_dim]`` absorbed queries (``[q_nope W_UK | q_pe]``).
        pool_rows: ``[N, latent_dim + rope_dim]`` latent rows (BF16, or FP8 read times
            ``kv_scale``).
        rows: int32 ``[R, K]`` rows each query attends; ``-1`` entries are masked.
        softmax_scale: multiplies the scores.
        latent_dim: the leading latent width the output is read from.
        kv_scale: dequant scale of an FP8 pool.

    Returns:
        ``[R, H, latent_dim]`` in the query's dtype.
    """
    valid = rows >= 0
    kv = pool_rows[rows.clamp_min(0).long()].float()
    if kv_scale != 1.0:
        kv = kv * kv_scale
    scores = torch.einsum("rhc,rkc->rhk", query.float(), kv) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("rhk,rkc->rhc", probs, kv[..., :latent_dim])
    return out.to(query.dtype)


__all__ = ["dflash2_window_rows", "sparse_mla_rows_eager"]
