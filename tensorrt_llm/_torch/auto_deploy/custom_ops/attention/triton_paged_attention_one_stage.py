# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton Paged Attention

This module provides a Triton-based paged attention implementation with:
- Combined KV cache with HND layout: [num_blocks, 2, num_kv_heads, page_size, head_dim]
- Context/prefill kernel with causal masking
"""

import math
from typing import List, Literal, Optional, Tuple

import flashinfer
import torch
import triton
import triton.language as tl
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from tensorrt_llm.llmapi.llm_args import KvCacheConfig

from ...utils.logger import ad_logger
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    KVPagedResourceHandler,
    MHACallable,
    PrepareMetadataCallable,
    ResourceHandlerDict,
)

# =============================================================================
# MODULE CONSTANTS
# =============================================================================

# KV cache layout: "HND" to match FlashInfer
# Shape: [num_blocks, 2, num_kv_heads, page_size, head_dim]
# Dimension 1: 0 = K, 1 = V
KV_LAYOUT: Literal["HND", "NHD"] = "HND"


def _get_sm_scale(head_dim: int, scale: Optional[float]) -> float:
    """Get softmax scale, computing default if not provided."""
    return scale if scale is not None else 1.0 / math.sqrt(head_dim)


# =============================================================================
# TRITON KERNELS - CACHE UPDATE
# =============================================================================


# TODO: add num_warps for auto-tuning?
# TODO: what about Instead of 1 token per kernel instance, process multiple tokens together?
@triton.jit
def _update_paged_kv_cache_kernel(
    # Input K, V
    k_ptr,
    v_ptr,
    # Metadata
    batch_indices_ptr,
    positions_ptr,
    # KV cache (combined)
    kv_cache_ptr,
    # Page table
    kv_indices_ptr,
    kv_indptr_ptr,
    # Constants
    NUM_TOKENS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    # Strides for kv_cache: [num_blocks, 2, num_kv_heads, page_size, head_dim]
    cache_stride_block: tl.constexpr,
    cache_stride_kv: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_token: tl.constexpr,
):
    """Update combined KV cache with new tokens.

    Cache layout (HND): [num_blocks, 2, num_kv_heads, page_size, head_dim]
    - Dimension 1: 0 = K, 1 = V
    """
    token_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    if token_id >= NUM_TOKENS:
        return

    # Load metadata for the token
    batch_idx = tl.load(batch_indices_ptr + token_id)
    position = tl.load(positions_ptr + token_id)

    # Compute page and offset within page
    page_idx_in_seq = position // PAGE_SIZE
    offset_in_page = position % PAGE_SIZE

    # Get physical page from page table
    page_start = tl.load(kv_indptr_ptr + batch_idx)
    physical_page = tl.load(kv_indices_ptr + page_start + page_idx_in_seq)

    # Load K, V for the token and head
    head_offsets = tl.arange(0, HEAD_DIM)
    kv_offset = token_id * N_KV_HEADS * HEAD_DIM + head_id * HEAD_DIM + head_offsets

    k = tl.load(k_ptr + kv_offset)
    v = tl.load(v_ptr + kv_offset)

    # Compute cache offset (use int64 to avoid overflow when physical_page * stride > 2^31)
    cache_base = (
        physical_page.to(tl.int64) * cache_stride_block
        + head_id * cache_stride_head
        + offset_in_page.to(tl.int64) * cache_stride_token
        + head_offsets
    )

    # Store K (kv_idx=0) and V (kv_idx=1)
    tl.store(kv_cache_ptr + cache_base, k)
    tl.store(kv_cache_ptr + cache_base + cache_stride_kv, v)


def update_paged_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
) -> None:
    """Update the combined paged KV cache with new K, V tensors.

    Args:
        k: Key tensor [num_tokens, n_kv_heads, head_dim]
        v: Value tensor [num_tokens, n_kv_heads, head_dim]
        batch_indices: Batch index for each token [num_tokens]
        positions: Position in sequence for each token [num_tokens]
        kv_cache: Combined cache [num_blocks, 2, n_kv_heads, page_size, head_dim]
        kv_indices: Physical page indices (flattened)
        kv_indptr: Cumulative page counts per sequence [num_seq + 1]
    """
    num_tokens, n_kv_heads, head_dim = k.shape
    page_size = kv_cache.shape[3]

    if num_tokens == 0:
        return

    grid = (num_tokens, n_kv_heads)

    _update_paged_kv_cache_kernel[grid](
        k,
        v,
        batch_indices,
        positions,
        kv_cache,
        kv_indices,
        kv_indptr,
        NUM_TOKENS=num_tokens,
        N_KV_HEADS=n_kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        cache_stride_block=kv_cache.stride(0),
        cache_stride_kv=kv_cache.stride(1),
        cache_stride_head=kv_cache.stride(2),
        cache_stride_token=kv_cache.stride(3),
    )


# =============================================================================
# TRITON KERNELS - SINGLE-STAGE DECODE
# =============================================================================
# TODO(perf): Consider FlashDecoding-style two-phase approach for both decode and context:
#             Phase 1: Parallelize across KV blocks (each computes local_max, local_sum, local_acc)
#             Phase 2: Reduction kernel to combine results with max corrections.
#             This enables parallelism over the KV sequence dimension for long sequences.
@triton.autotune(
    configs=[
        triton.Config({"SEQ_BLOCK_SIZE": 32}, num_warps=4, num_stages=2),
        triton.Config({"SEQ_BLOCK_SIZE": 64}, num_warps=4, num_stages=3),
        triton.Config({"SEQ_BLOCK_SIZE": 128}, num_warps=8, num_stages=3),
        triton.Config({"SEQ_BLOCK_SIZE": 256}, num_warps=8, num_stages=4),
    ],
    key=["HEAD_DIM", "MAX_SEQ_LEN", "PAGE_SIZE"],
)
@triton.jit
def _single_stage_paged_decode_kernel(
    # Query input
    q_ptr,
    # KV cache (combined)
    kv_cache_ptr,
    # Page table
    kv_indices_ptr,
    kv_indptr_ptr,
    kv_last_page_len_ptr,
    # Output
    o_ptr,
    # Strides
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    o_stride_batch: tl.constexpr,
    o_stride_head: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_kv: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_token: tl.constexpr,
    # Constants
    SM_SCALE: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SEQ_BLOCK_SIZE: tl.constexpr,
):
    """Single-stage paged decode with paged KV cache.

    Grid: (batch_size, n_heads)
    Each program processes one (batch, head) pair using online softmax.
    """
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    # GQA: Map query head to KV head
    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS
    kv_head_id = head_id // HEAD_RATIO

    # Get sequence length from page table
    page_start = tl.load(kv_indptr_ptr + batch_id)
    page_end = tl.load(kv_indptr_ptr + batch_id + 1)
    num_pages = page_end - page_start
    last_page_len = tl.load(kv_last_page_len_ptr + batch_id)
    seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len

    dhead_offsets = tl.arange(0, HEAD_DIM)

    if seq_len <= 0:
        # Zero output for empty sequence
        o_offset = batch_id * o_stride_batch + head_id * o_stride_head + dhead_offsets
        tl.store(o_ptr + o_offset, tl.zeros([HEAD_DIM], dtype=tl.float32))
        return

    # Load Q (stays in registers)
    q_offset = batch_id * q_stride_batch + head_id * q_stride_head + dhead_offsets
    q = tl.load(q_ptr + q_offset)

    # Initialize online softmax accumulators
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    m_i = float("-inf")
    l_i = 0.0

    # Process KV in blocks
    # TODO(perf): Iterate by page instead of by block to eliminate div/mod for page indexing.
    #             Use page-aligned iteration: for page_idx in range(num_pages), then load
    #             physical_page directly from page table. Handle last_page_len for partial pages.
    num_blocks = (seq_len + SEQ_BLOCK_SIZE - 1) // SEQ_BLOCK_SIZE

    for block_idx in range(num_blocks):
        seq_start = block_idx * SEQ_BLOCK_SIZE
        seq_offsets = seq_start + tl.arange(0, SEQ_BLOCK_SIZE)
        seq_mask = seq_offsets < seq_len

        # Compute page indices and offsets
        page_idx_in_seq = seq_offsets // PAGE_SIZE
        offset_in_page = seq_offsets % PAGE_SIZE

        # Load physical page indices
        physical_pages = tl.load(
            kv_indices_ptr + page_start + page_idx_in_seq,
            mask=seq_mask & (page_idx_in_seq < num_pages),
            other=0,
        )

        # Compute cache offsets for K and V (int64 to avoid overflow)
        cache_base = (
            physical_pages[:, None].to(tl.int64) * cache_stride_block
            + kv_head_id * cache_stride_head
            + offset_in_page[:, None].to(tl.int64) * cache_stride_token
            + dhead_offsets[None, :]
        )

        kv_mask = seq_mask[:, None]

        # Load K and V
        k = tl.load(kv_cache_ptr + cache_base, mask=kv_mask, other=0.0)
        v = tl.load(kv_cache_ptr + cache_base + cache_stride_kv, mask=kv_mask, other=0.0)

        # Compute attention scores
        attn = tl.sum(q[None, :] * k, axis=1) * SM_SCALE
        attn = tl.where(seq_mask, attn, float("-inf"))

        # Online softmax update
        m_ij = tl.max(attn, axis=0)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(attn - m_i_new)

        # Update accumulator
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_i_new

    # Finalize
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_i

    # Store output
    o_offset = batch_id * o_stride_batch + head_id * o_stride_head + dhead_offsets
    tl.store(o_ptr + o_offset, o)


def triton_paged_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    sm_scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single-stage paged decode with paged KV cache.

    Args:
        q: Query tensor [batch_size, n_heads, head_dim]
        kv_cache: Combined cache [num_blocks, 2, n_kv_heads, page_size, head_dim]
        kv_indices: Physical page indices (flattened)
        kv_indptr: Cumulative page counts [batch_size + 1]
        kv_last_page_len: Valid tokens in last page [batch_size]
        sm_scale: Softmax scale factor
        out: Optional output tensor [batch_size, n_heads, head_dim]

    Returns:
        Output tensor [batch_size, n_heads, head_dim]
    """
    batch_size, n_heads, head_dim = q.shape
    _, _, n_kv_heads, page_size, _ = kv_cache.shape
    max_pages = kv_indices.shape[0]
    max_seq_len = max_pages * page_size

    output = out if out is not None else torch.empty_like(q)

    grid = (batch_size, n_heads)

    _single_stage_paged_decode_kernel[grid](
        q,
        kv_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        output,
        q.stride(0),
        q.stride(1),
        output.stride(0),
        output.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        SM_SCALE=sm_scale,
        MAX_SEQ_LEN=max_seq_len,
        N_HEADS=n_heads,
        N_KV_HEADS=n_kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
    )

    return output


# =============================================================================
# TRITON KERNELS - CONTEXT/PREFILL
# =============================================================================


# TODO: add auto-tuning? And consider tune the Q_BLOCK and KV_BLOCK?
@triton.jit
def _paged_context_kernel(
    # Inputs
    q_ptr,
    kv_cache_ptr,
    # Metadata
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    kv_last_page_len_ptr,
    seq_len_with_cache_ptr,
    # Output
    o_ptr,
    # Strides
    q_stride_token: tl.constexpr,
    q_stride_head: tl.constexpr,
    o_stride_token: tl.constexpr,
    o_stride_head: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_kv: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_token: tl.constexpr,
    # Constants
    SM_SCALE: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
):
    """Context/prefill attention with paged KV cache and causal masking.

    Grid: (num_seq, n_heads, num_q_blocks)
    """
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    q_block_id = tl.program_id(axis=2)

    # GQA
    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS
    kv_head_id = head_id // HEAD_RATIO

    # Load metadata
    q_start = tl.load(qo_indptr_ptr + batch_id)
    q_end = tl.load(qo_indptr_ptr + batch_id + 1)
    q_len = q_end - q_start

    kv_page_start = tl.load(kv_indptr_ptr + batch_id)
    kv_page_end = tl.load(kv_indptr_ptr + batch_id + 1)
    num_kv_pages = kv_page_end - kv_page_start
    total_kv_len = tl.load(seq_len_with_cache_ptr + batch_id)

    cache_len = total_kv_len - q_len

    # Q block range
    q_block_start = q_block_id * Q_BLOCK
    q_offsets = q_block_start + tl.arange(0, Q_BLOCK)
    q_mask = q_offsets < q_len

    if tl.sum(q_mask.to(tl.int32)) == 0:
        return

    # Load Q block
    dhead_offsets = tl.arange(0, HEAD_DIM)
    q_load_offsets = (
        (q_start + q_offsets[:, None]) * q_stride_token
        + head_id * q_stride_head
        + dhead_offsets[None, :]
    )
    q_load_mask = q_mask[:, None]
    q = tl.load(q_ptr + q_load_offsets, mask=q_load_mask, other=0.0)

    # Initialize accumulators
    acc = tl.zeros([Q_BLOCK, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([Q_BLOCK], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([Q_BLOCK], dtype=tl.float32)

    # Loop over KV blocks
    num_kv_blocks = (total_kv_len + KV_BLOCK - 1) // KV_BLOCK

    # TODO: Instead of iterating by KV_BLOCK, iterate by page?
    for kv_block_id in range(num_kv_blocks):
        kv_block_start = kv_block_id * KV_BLOCK
        kv_offsets = kv_block_start + tl.arange(0, KV_BLOCK)
        kv_mask = kv_offsets < total_kv_len

        # Map to physical pages
        page_idx = kv_offsets // PAGE_SIZE
        offset_in_page = kv_offsets % PAGE_SIZE

        physical_pages = tl.load(
            kv_indices_ptr + kv_page_start + page_idx,
            mask=kv_mask & (page_idx < num_kv_pages),
            other=0,
        )

        # Compute cache offsets (int64 to avoid overflow)
        cache_offsets = (
            physical_pages[:, None].to(tl.int64) * cache_stride_block
            + kv_head_id * cache_stride_head
            + offset_in_page[:, None].to(tl.int64) * cache_stride_token
            + dhead_offsets[None, :]
        )

        kv_load_mask = kv_mask[:, None]

        # Load K, V
        k = tl.load(kv_cache_ptr + cache_offsets, mask=kv_load_mask, other=0.0)
        v = tl.load(
            kv_cache_ptr + cache_offsets + cache_stride_kv,
            mask=kv_load_mask,
            other=0.0,
        )

        # Compute QK^T
        qk = tl.dot(q, tl.trans(k)) * SM_SCALE

        # Causal mask: q_pos + cache_len >= kv_pos
        q_positions = q_offsets[:, None] + cache_len
        kv_positions = kv_offsets[None, :]
        causal_mask = q_positions >= kv_positions

        # Combined mask
        full_mask = q_mask[:, None] & causal_mask & kv_mask[None, :]
        qk = tl.where(full_mask, qk, float("-inf"))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # Finalize
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_i[:, None]

    # Store output
    o_store_offsets = (
        (q_start + q_offsets[:, None]) * o_stride_token
        + head_id * o_stride_head
        + dhead_offsets[None, :]
    )
    tl.store(o_ptr + o_store_offsets, o, mask=q_load_mask)


def triton_paged_context(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    sm_scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Context/prefill attention with paged KV cache.

    Args:
        q: Query tensor [total_tokens, n_heads, head_dim]
        kv_cache: Combined cache [num_blocks, 2, n_kv_heads, page_size, head_dim]
        qo_indptr: Cumulative token counts [num_seq + 1]
        kv_indptr: Cumulative page counts [num_seq + 1]
        kv_indices: Physical page indices
        kv_last_page_len: Valid tokens in last page [num_seq]
        seq_len_with_cache: Total sequence length including cache [num_seq]
        sm_scale: Softmax scale
        out: Optional output tensor [total_tokens, n_heads, head_dim]

    Returns:
        Output tensor [total_tokens, n_heads, head_dim]
    """
    total_tokens, n_heads, head_dim = q.shape
    _, _, n_kv_heads, page_size, _ = kv_cache.shape
    num_seq = qo_indptr.shape[0] - 1

    output = out if out is not None else torch.empty_like(q)

    # TODO: make these configurable?
    Q_BLOCK = 32
    KV_BLOCK = 32

    # Compute max q_len for grid sizing
    q_lens = qo_indptr[1:] - qo_indptr[:-1]
    max_q_len = int(q_lens.max().item()) if num_seq > 0 else 0
    num_q_blocks = (max_q_len + Q_BLOCK - 1) // Q_BLOCK

    if num_seq == 0 or max_q_len == 0:
        return output

    grid = (num_seq, n_heads, num_q_blocks)

    _paged_context_kernel[grid](
        q,
        kv_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_len_with_cache,
        output,
        q.stride(0),
        q.stride(1),
        output.stride(0),
        output.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        SM_SCALE=sm_scale,
        N_HEADS=n_heads,
        N_KV_HEADS=n_kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        Q_BLOCK=Q_BLOCK,
        KV_BLOCK=KV_BLOCK,
    )

    return output


# =============================================================================
# METADATA PREPARATION
# =============================================================================


@torch.library.custom_op("auto_deploy::triton_paged_one_stage_prepare_metadata", mutates_args=())
def prepare_triton_paged_one_stage_metadata(
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
) -> List[torch.Tensor]:
    """Prepare metadata for Triton paged attention.

    Computes batch_indices and positions for cache updates.
    """

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_tokens = num_prefill_tokens + num_decode

    qo_indptr = cu_seqlen[: num_seq + 1]

    # Use FlashInfer's optimized Triton kernel for batch indices and positions
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr, seq_len_with_cache[:num_seq], num_tokens
    )

    return batch_indices, positions


@prepare_triton_paged_one_stage_metadata.register_fake
def prepare_triton_paged_one_stage_metadata_fake(
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
):
    num_tokens = position_ids.shape[0] * position_ids.shape[1]
    return (
        torch.empty(num_tokens, dtype=torch.int32, device=position_ids.device),
        torch.empty(num_tokens, dtype=torch.int32, device=position_ids.device),
    )


@torch.library.custom_op("auto_deploy::triton_paged_one_stage_mha_with_cache", mutates_args=())
def triton_paged_one_stage_mha_with_cache(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    # EXTRA METADATA
    triton_batch_indices: torch.Tensor,
    triton_positions: torch.Tensor,
    # CACHES - combined KV cache
    kv_cache: torch.Tensor,
    # CONSTANTS
    scale: Optional[float],
) -> torch.Tensor:
    """Triton paged attention with mixed batch support.

    Handles prefill, decode, or mixed batches by checking batch_info_host
    and dispatching to the appropriate kernel.

    Args:
        q, k, v: Query, Key, Value tensors [batch, seq, heads, head_dim]
        batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
        kv_cache: Combined cache [num_blocks, 2, n_kv_heads, page_size, head_dim]
        scale: Softmax scale (default: 1/sqrt(head_dim))

    Returns:
        Attention output with same shape as q
    """
    # Extract shapes
    head_dim = kv_cache.shape[-1]
    q_shape_og = q.shape
    b, s = q_shape_og[:2]

    # Reshape to [total_tokens, heads, head_dim]
    q = q.reshape(b * s, -1, head_dim).contiguous()
    k = k.reshape(b * s, -1, head_dim).contiguous()
    v = v.reshape(b * s, -1, head_dim).contiguous()

    # Extract batch info
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_total_tokens = num_prefill_tokens + num_decode

    sm_scale = _get_sm_scale(head_dim, scale)

    # Update KV cache with new tokens
    update_paged_kv_cache(
        k[:num_total_tokens],
        v[:num_total_tokens],
        triton_batch_indices[:num_total_tokens],
        triton_positions[:num_total_tokens],
        kv_cache,
        cache_loc,
        cu_num_pages[: num_seq + 1],
    )

    # Allocate output tensor once
    y = torch.empty_like(q)

    # Process prefill tokens if any
    if num_prefill > 0:
        cu_seqlen = cu_seqlen_host[: num_prefill + 1].to(q.device, non_blocking=True)
        seq_len_with_cache = seq_len_with_cache_host[:num_prefill].to(q.device, non_blocking=True)
        triton_paged_context(
            q[:num_prefill_tokens],
            kv_cache,
            cu_seqlen,
            cu_num_pages[: num_prefill + 1],
            cache_loc,
            last_page_len[:num_prefill],
            seq_len_with_cache,
            sm_scale,
            out=y[:num_prefill_tokens],
        )

    # Process decode tokens if any
    if num_decode > 0:
        triton_paged_decode(
            q[num_prefill_tokens:num_total_tokens],
            kv_cache,
            cache_loc,
            cu_num_pages[num_prefill : num_seq + 1],
            last_page_len[num_prefill:num_seq],
            sm_scale,
            out=y[num_prefill_tokens:num_total_tokens],
        )

    return y.view(q_shape_og)


@triton_paged_one_stage_mha_with_cache.register_fake
def triton_paged_one_stage_mha_with_cache_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    triton_batch_indices: torch.Tensor,
    triton_positions: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: Optional[float],
) -> torch.Tensor:
    return torch.empty_like(q.contiguous())


# =============================================================================
# DESCRIPTOR CLASS
# =============================================================================


@AttentionRegistry.register("triton_paged_one_stage")
class TritonPagedAttentionOneStage(AttentionDescriptor):
    """Descriptor for Triton Paged Attention backend."""

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the backend."""
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of qkv arguments expected by the source op."""
        return 3

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        """Get the source attention op that we target for replacement."""
        return torch.ops.auto_deploy.torch_attention

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.triton_paged_one_stage_mha_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        """Standard metadata args - matching FlashInfer for consistency."""
        return [
            "batch_info_host",
            "cu_seqlen_host",
            "cu_num_pages",
            "cu_num_pages_host",
            "cache_loc",
            "last_page_len",
            "last_page_len_host",
            "seq_len_with_cache_host",
        ]

    @classmethod
    def get_prepare_extra_metadata_info(
        cls, any_source_attn_node: Node
    ) -> Tuple[Optional[PrepareMetadataCallable], int, List[Constant]]:
        return (
            torch.ops.auto_deploy.triton_paged_one_stage_prepare_metadata.default,
            2,
            [],
        )

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        """Create paged resource handler for combined KV cache."""
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]

        return {
            "kv_cache": KVPagedResourceHandler(
                num_kv_heads,
                head_dim,
                dtype=cls.resolve_cache_dtype(cache_config.dtype, k_fake.dtype),
                kv_factor=2,
                kv_layout=KV_LAYOUT,
            )
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Extract constants from the source attention node."""
        # Sanity check: layout == "bsnd"
        layout = source_attn_node.kwargs.get("layout", None)
        if (
            layout is None
            and len(source_attn_node.args) > 0
            and isinstance(source_attn_node.args[-1], str)
        ):
            layout = source_attn_node.args[-1]
        if layout != "bsnd":
            raise RuntimeError(
                f"Expected torch_attention layout='bsnd' but got {layout!r} "
                f"for node: {source_attn_node.format_node()}"
            )

        # Check other arguments
        attn_mask, dropout_p, is_causal = extract_op_args(
            source_attn_node, "attn_mask", "dropout_p", "is_causal"
        )
        if attn_mask is not None or dropout_p != 0.0 or not is_causal:
            ad_logger.debug(
                "Unsupported attention arguments for "
                f"{source_attn_node=}: {attn_mask=}, {dropout_p=}, {is_causal=}"
            )

        # Get scale
        if len(source_attn_node.args) > 6:
            scale = source_attn_node.args[6]
        else:
            scale = source_attn_node.kwargs.get("scale", None)

        if not (isinstance(scale, float) or scale is None):
            ad_logger.warning(f"Provided {scale=}, is not a float. Using default scale instead.")
            scale = None

        return [scale]
