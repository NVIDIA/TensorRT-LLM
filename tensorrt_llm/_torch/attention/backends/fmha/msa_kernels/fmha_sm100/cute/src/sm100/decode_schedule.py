# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Split-KV schedule for paged fp8 decode attention.

The public PageKV representation remains this repo's rectangular page table:
``page_table [B, max_pages]`` plus ``seqused_k [B]``.  The schedule only
describes how query tiles and KV chunks are split into work items.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DecodeAttentionSchedule:
    split_kv: bool
    cta_tile_q: int
    num_q_tiles: int
    kv_chunk_size_pages: int
    kv_chunk_size_tokens: int
    work_count: int
    padded_work_count: int
    partial_rows: int
    max_split_count: int
    max_grid_size: int
    active_blocks_per_sm: int
    num_sms: int
    base_cta: int
    request_indices: torch.Tensor
    qo_tile_indices: torch.Tensor
    kv_tile_indices: torch.Tensor
    merge_indptr: torch.Tensor
    o_indptr: torch.Tensor
    block_valid_mask: torch.Tensor
    kv_pages: torch.Tensor
    split_counts: torch.Tensor


def _require_i32_cuda_1d(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be rank-1")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def prepare_decode_schedule(
    *,
    seqused_k: torch.Tensor,
    page_size: int,
    seqlen_q: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    enable_cuda_graph: bool = False,
    max_grid_size: Optional[int] = None,
    fixed_split_size: Optional[int] = None,
    disable_split_kv: bool = False,
) -> DecodeAttentionSchedule:
    """Build paged decode split-KV schedule on the GPU.

    A single CUDA kernel reads ``seqused_k`` on device and writes all
    schedule index arrays.  Only a small summary tensor is D2H-synced so
    the wrapper can size O_partial / pick the kernel grid / choose the
    split-vs-non-split compile path.

    ``max_seqlen_k`` is the host-side worst-case bound used to pad the
    work-tile arrays.  It must satisfy ``max(seqused_k) <= max_seqlen_k``.
    """
    _require_i32_cuda_1d(seqused_k, name="seqused_k")
    # Hard cap: current single-CTA schedule kernel stores per-batch state
    # in shared memory.  Larger batches require a multi-CTA cooperative
    # scheduler (unimplemented).  Fail fast at the Python boundary so the
    # error doesn't surface from inside the CUDA extension.
    if int(seqused_k.shape[0]) > 1024:
        raise NotImplementedError(
            "decode schedule currently supports batch <= 1024 "
            f"(got batch={int(seqused_k.shape[0])}). Larger batches need "
            "the multi-CTA scheduler — not yet implemented."
        )
    # Two API-boundary checks tied to the kernel's packed-GQA layout
    # (q_tokens_per_group = m_block_size / qhead_per_kv = 128/16 = 8):
    #
    # (1) seqused_k[b] >= seqlen_q.  The kernel computes the causal mask as
    #     col_limit = row_idx + seqlen_k - seqlen_q + 1.  For row 0 (first
    #     q-token in the packed group) this is col_limit = seqlen_k - seqlen_q
    #     + 1, which goes <= 0 whenever seqlen_k < seqlen_q.  That all-masked
    #     row then enters a mask-codegen path with PTX-undefined shift counts
    #     and the kernel hangs.  The condition is also semantically invalid
    #     in batched-decode: you can't emit seqlen_q new tokens with fewer
    #     than seqlen_q total context tokens (seqlen_k includes them).
    #
    # (2) seqused_k[b] % page_size ∈ {0, 8, 16, ..., 120}.  Same hang fires
    #     when the LAST partial page has < q_tokens_per_group=8 valid
    #     columns, because then the *last MMA tile* hits the same all-masked
    #     row case for the trailing q-tokens.
    #
    # Both are tracked as a separate kernel-level TODO (un-pack the
    # all-masked row → skip mask call, or saturate causal_col_limit at >= 1
    # in mask.py).  Until then, fail fast at the Python boundary with a
    # clear message rather than letting the kernel timeout.
    seqlen_q_i = int(seqlen_q)
    bad_q = seqused_k < seqlen_q_i
    if bool(bad_q.any().item()):
        bad_idx = int(torch.nonzero(bad_q, as_tuple=True)[0][0].item())
        bad_val = int(seqused_k[bad_idx].item())
        raise ValueError(
            f"decode kernel requires seqused_k[b] >= seqlen_q (= {seqlen_q_i}) "
            f"for every batch.  Got seqused_k[{bad_idx}]={bad_val}.  "
            f"This is also a batched-decode invariant: seqlen_k must include "
            f"the seqlen_q new tokens being emitted."
        )
    rem = seqused_k % int(page_size)
    bad_rem = (rem > 0) & (rem < seqlen_q_i)
    if bool(bad_rem.any().item()):
        bad_idx = int(torch.nonzero(bad_rem, as_tuple=True)[0][0].item())
        bad_val = int(seqused_k[bad_idx].item())
        raise ValueError(
            f"decode kernel requires seqused_k[b] % page_size ∈ "
            f"{{0, {seqlen_q_i}, {seqlen_q_i*2}, ..., {(page_size//seqlen_q_i)*seqlen_q_i}}}.  "
            f"Got seqused_k[{bad_idx}]={bad_val}, last partial page has "
            f"{bad_val % int(page_size)} valid columns (< seqlen_q={seqlen_q_i}). "
            f"Round seqused_k up to the next multiple of {seqlen_q_i} OR to "
            f"a multiple of {page_size}."
        )
    if int(page_size) <= 0:
        raise ValueError("page_size must be positive")
    if int(seqlen_q) <= 0:
        raise ValueError("seqlen_q must be positive")
    if int(num_qo_heads) <= 0 or int(num_kv_heads) <= 0:
        raise ValueError("head counts must be positive")
    if int(num_qo_heads) % int(num_kv_heads) != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    if int(num_qo_heads) // int(num_kv_heads) != 16:
        raise NotImplementedError("decode schedule currently supports only qhead_per_kv=16")
    if int(head_dim) != 128:
        raise NotImplementedError("decode schedule currently supports only head_dim=128")
    if int(max_seqlen_k) <= 0:
        raise ValueError("max_seqlen_k must be positive")

    from src.sm100.fwd_decode.build_decode_schedule import build_decode_schedule

    raw = build_decode_schedule(
        seqused_k,
        page_size=int(page_size),
        seqlen_q=int(seqlen_q),
        num_qo_heads=int(num_qo_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        max_seqlen_k=int(max_seqlen_k),
        enable_cuda_graph=bool(enable_cuda_graph),
        max_grid_size=0 if max_grid_size is None else int(max_grid_size),
        fixed_split_size=-1 if fixed_split_size is None else int(fixed_split_size),
        disable_split_kv=bool(disable_split_kv),
    )
    return DecodeAttentionSchedule(
        split_kv=bool(raw["split_kv"]),
        cta_tile_q=int(raw["cta_tile_q"]),
        num_q_tiles=int(raw["num_q_tiles"]),
        kv_chunk_size_pages=int(raw["kv_chunk_size_pages"]),
        kv_chunk_size_tokens=int(raw["kv_chunk_size_tokens"]),
        work_count=int(raw["work_count"]),
        padded_work_count=int(raw["padded_work_count"]),
        partial_rows=int(raw["partial_rows"]),
        max_split_count=int(raw["max_split_count"]),
        max_grid_size=int(raw["max_grid_size"]),
        active_blocks_per_sm=int(raw["active_blocks_per_sm"]),
        num_sms=int(raw["num_sms"]),
        base_cta=int(raw["base_cta"]),
        request_indices=raw["request_indices"],
        qo_tile_indices=raw["qo_tile_indices"],
        kv_tile_indices=raw["kv_tile_indices"],
        merge_indptr=raw["merge_indptr"],
        o_indptr=raw["o_indptr"],
        block_valid_mask=raw["block_valid_mask"],
        kv_pages=raw["kv_pages"],
        split_counts=raw["split_counts"],
    )


__all__ = [
    "DecodeAttentionSchedule",
    "prepare_decode_schedule",
]
