# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""CUTE DSL launchers for paged fp8 decode forward."""

from __future__ import annotations

import torch

from .atten_fwd import run_decode_attention
from .combine import run_decode_combine


def decode_forward_paged_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    request_indices: torch.Tensor,
    qo_tile_indices: torch.Tensor,
    kv_tile_indices: torch.Tensor,
    block_valid_mask: torch.Tensor,
    split_counts: torch.Tensor,
    o_indptr: torch.Tensor,
    merge_indptr: torch.Tensor,
    O_partial: torch.Tensor | None,
    LSE_partial: torch.Tensor | None,
    *,
    softmax_scale: float,
    seqlen_q: int,
    page_size: int,
    kv_chunk_size_pages: int,
    max_split_count: int,
    split_kv: bool,
    causal: bool,
    return_lse: bool = True,
    O_partial_dummy: torch.Tensor | None = None,
    LSE_partial_dummy: torch.Tensor | None = None,
) -> None:
    """Launch dense paged fp8 decode forward and optional compressed combine.

    ``O_partial_dummy`` / ``LSE_partial_dummy`` are caller-provided pre-allocated
    placeholder buffers for the non-split path.  When supplied, ``run_decode_attention``
    skips the per-call ``torch.empty`` it would otherwise need to satisfy the
    kernel's positional arg signature, saving ~5us on small-kv calls.
    """

    run_decode_attention(
        q,
        k,
        v,
        page_table,
        seqused_k,
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        block_valid_mask,
        split_counts,
        o_indptr,
        out,
        lse,
        O_partial,
        LSE_partial,
        softmax_scale=float(softmax_scale),
        seqlen_q=int(seqlen_q),
        page_size=int(page_size),
        kv_chunk_size_pages=int(kv_chunk_size_pages),
        split_kv=bool(split_kv),
        causal=bool(causal),
        return_lse=bool(return_lse),
        O_partial_dummy=O_partial_dummy,
        LSE_partial_dummy=LSE_partial_dummy,
    )
    if split_kv:
        if O_partial is None or LSE_partial is None:
            raise ValueError("split decode requires O_partial and LSE_partial")
        qhead_per_kv = q.shape[1] // k.shape[1]
        q_tokens_per_group = 128 // int(qhead_per_kv)
        run_decode_combine(
            O_partial,
            LSE_partial,
            split_counts,
            o_indptr,
            out,
            lse,
            seqlen_q=int(seqlen_q),
            q_tokens_per_group=q_tokens_per_group,
            max_split_count=int(max_split_count),
        )


__all__ = ["decode_forward_paged_fp8", "run_decode_attention", "run_decode_combine"]
