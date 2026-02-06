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

KV_LAYOUT: Literal["HND", "NHD"] = "HND"


def _get_sm_scale(head_dim: int, scale: Optional[float]) -> float:
    """Get softmax scale, computing default if not provided."""
    return scale if scale is not None else 1.0 / math.sqrt(head_dim)


@triton.jit
def _update_paged_kv_cache_kernel(
    # Input K, V
    k_ptr,
    v_ptr,
    # Metadata
    batch_indices_ptr,
    positions_ptr,
    # KV cache
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
    """Update combined KV cache with new tokens."""
    token_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    if token_id >= NUM_TOKENS:
        return

    batch_idx = tl.load(batch_indices_ptr + token_id)
    position = tl.load(positions_ptr + token_id)

    page_idx_in_seq = position // PAGE_SIZE
    offset_in_page = position % PAGE_SIZE

    page_start = tl.load(kv_indptr_ptr + batch_idx)
    physical_page = tl.load(kv_indices_ptr + page_start + page_idx_in_seq)

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
    """Update the combined paged KV cache with new K, V tensors."""
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
# FLASH DECODE - HELPERS
# =============================================================================


def _get_num_splits(max_seq_len: int, batch_size: int, n_kv_heads: int, page_size: int) -> int:
    """Compute optimal number of KV splits for FlashDecoding.

    With GQA batching, the grid is (batch, n_kv_heads, num_splits).
    We want enough blocks to saturate the GPU.
    """
    if max_seq_len <= 0:
        return 1

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    existing_parallelism = batch_size * n_kv_heads

    # Already enough parallelism
    if existing_parallelism >= num_sms * 2:
        return 1

    # Target ~4 waves of thread blocks
    target_blocks = num_sms * 4
    num_splits = max(1, (target_blocks + existing_parallelism - 1) // existing_parallelism)

    # Cap splits so each block has at least 2 pages of work.
    # With fewer pages, the per-block overhead (Q load, accumulator init,
    # partial_o/lse store, plus stage2 reduction cost) dominates the useful
    # compute (page-loop iterations). 2 pages is a conservative lower bound
    # to keep the overhead-to-work ratio acceptable.
    max_pages = max_seq_len // page_size
    max_splits = max(1, max_pages // 2)
    num_splits = min(num_splits, max_splits)

    # Round to next power of 2 for Triton compile caching
    if num_splits > 1:
        num_splits = 2 ** math.ceil(math.log2(num_splits))

    return min(num_splits, 128)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["HEAD_DIM", "PAGE_SIZE", "HEAD_RATIO"],
)
@triton.jit
def _flash_decode_stage1_kernel(
    # Query input
    q_ptr,
    # KV cache (combined)
    kv_cache_ptr,
    # Page table
    kv_indices_ptr,
    kv_indptr_ptr,
    kv_last_page_len_ptr,
    # Intermediate outputs
    partial_o_ptr,
    partial_lse_ptr,
    # Q strides: [batch, n_heads, head_dim]
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    # Partial output strides: [batch, n_heads, num_splits, head_dim]
    po_stride_batch: tl.constexpr,
    po_stride_head: tl.constexpr,
    po_stride_split: tl.constexpr,
    # Partial LSE strides: [batch, n_heads, num_splits]
    plse_stride_batch: tl.constexpr,
    plse_stride_head: tl.constexpr,
    plse_stride_split: tl.constexpr,
    # Cache strides: [num_blocks, 2, n_kv_heads, page_size, head_dim]
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
    HEAD_RATIO: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """
    Key optimizations:
    - Loads KV once for HEAD_RATIO Q heads
    - Iterates by page for contiguous memory access
    - Splits KV sequence across blocks for GPU utilization
    """
    batch_id = tl.program_id(axis=0)
    kv_head_id = tl.program_id(axis=1)
    split_id = tl.program_id(axis=2)

    # Get sequence info from page table
    kv_page_start = tl.load(kv_indptr_ptr + batch_id)
    kv_page_end = tl.load(kv_indptr_ptr + batch_id + 1)
    num_pages = kv_page_end - kv_page_start
    last_page_len = tl.load(kv_last_page_len_ptr + batch_id)

    # Compute this split's page range (page-aligned splits)
    pages_per_split = (num_pages + NUM_SPLITS - 1) // NUM_SPLITS
    page_split_start = split_id * pages_per_split
    page_split_end = tl.minimum(page_split_start + pages_per_split, num_pages)

    dhead_offsets = tl.arange(0, HEAD_DIM)
    head_ids = kv_head_id * HEAD_RATIO + tl.arange(0, HEAD_RATIO)

    # Handle inactive splits (beyond the sequence's pages)
    if page_split_start >= num_pages:
        # Store zeros + -inf LSE for all HEAD_RATIO Q heads
        po_offsets = (
            batch_id * po_stride_batch
            + head_ids[:, None] * po_stride_head
            + split_id * po_stride_split
            + dhead_offsets[None, :]
        )
        tl.store(
            partial_o_ptr + po_offsets,
            tl.zeros([HEAD_RATIO, HEAD_DIM], dtype=tl.float32),
        )
        plse_offsets = (
            batch_id * plse_stride_batch
            + head_ids * plse_stride_head
            + split_id * plse_stride_split
        )
        tl.store(
            partial_lse_ptr + plse_offsets,
            tl.zeros([HEAD_RATIO], dtype=tl.float32) + float("-inf"),
        )
        return

    # Load Q for all HEAD_RATIO heads sharing this KV head: [HEAD_RATIO, HEAD_DIM]
    q_offsets = (
        batch_id * q_stride_batch + head_ids[:, None] * q_stride_head + dhead_offsets[None, :]
    )
    q_all = tl.load(q_ptr + q_offsets)  # [HEAD_RATIO, HEAD_DIM]

    acc = tl.zeros([HEAD_RATIO, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([HEAD_RATIO], dtype=tl.float32) + float("-inf")
    l_i = tl.zeros([HEAD_RATIO], dtype=tl.float32)

    num_pages_this_split = page_split_end - page_split_start
    for local_page_idx in range(num_pages_this_split):
        page_idx = page_split_start + local_page_idx
        physical_page = tl.load(kv_indices_ptr + kv_page_start + page_idx)

        # Determine valid tokens in this page
        is_last_page_of_seq = page_idx == (num_pages - 1)
        valid_tokens = tl.where(is_last_page_of_seq, last_page_len, PAGE_SIZE)

        page_offsets = tl.arange(0, PAGE_SIZE)
        page_mask = page_offsets < valid_tokens

        # Compute cache offset (use int64 to avoid overflow when physical_page * stride > 2^31)
        cache_base = (
            physical_page.to(tl.int64) * cache_stride_block
            + kv_head_id * cache_stride_head
            + page_offsets[:, None] * cache_stride_token
            + dhead_offsets[None, :]
        )
        page_mask_2d = page_mask[:, None]

        k = tl.load(
            kv_cache_ptr + cache_base, mask=page_mask_2d, other=0.0
        )  # [PAGE_SIZE, HEAD_DIM]
        v = tl.load(
            kv_cache_ptr + cache_base + cache_stride_kv,
            mask=page_mask_2d,
            other=0.0,
        )  # [PAGE_SIZE, HEAD_DIM]

        # [HEAD_RATIO, HEAD_DIM] @ [HEAD_DIM, PAGE_SIZE] -> [HEAD_RATIO, PAGE_SIZE]
        attn = tl.dot(q_all, tl.trans(k)) * SM_SCALE
        attn = tl.where(page_mask[None, :], attn, float("-inf"))

        # Online softmax update (vectorized over HEAD_RATIO)
        m_ij = tl.max(attn, axis=1)  # [HEAD_RATIO]
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(attn - m_i_new[:, None])  # [HEAD_RATIO, PAGE_SIZE]

        # [HEAD_RATIO, PAGE_SIZE] @ [PAGE_SIZE, HEAD_DIM] -> [HEAD_RATIO, HEAD_DIM]
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # Finalize: normalize and compute LSE
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    partial_o_val = acc / l_i_safe[:, None]  # [HEAD_RATIO, HEAD_DIM]
    lse_val = m_i + tl.log(l_i_safe)  # [HEAD_RATIO]

    # Store results for all HEAD_RATIO Q heads at once (2D store)
    po_offsets = (
        batch_id * po_stride_batch
        + head_ids[:, None] * po_stride_head
        + split_id * po_stride_split
        + dhead_offsets[None, :]
    )
    tl.store(partial_o_ptr + po_offsets, partial_o_val)

    plse_offsets = (
        batch_id * plse_stride_batch + head_ids * plse_stride_head + split_id * plse_stride_split
    )
    tl.store(partial_lse_ptr + plse_offsets, lse_val)


@triton.jit
def _flash_decode_stage2_kernel(
    # Partial results
    partial_o_ptr,
    partial_lse_ptr,
    # Final output
    o_ptr,
    # Partial output strides: [batch, n_heads, num_splits, head_dim]
    po_stride_batch: tl.constexpr,
    po_stride_head: tl.constexpr,
    po_stride_split: tl.constexpr,
    # Partial LSE strides: [batch, n_heads, num_splits]
    plse_stride_batch: tl.constexpr,
    plse_stride_head: tl.constexpr,
    plse_stride_split: tl.constexpr,
    # Output strides: [batch, n_heads, head_dim]
    o_stride_batch: tl.constexpr,
    o_stride_head: tl.constexpr,
    # Constants
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """
    Each program combines results from all splits for one (batch, head) pair.
    """
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    dhead_offsets = tl.arange(0, HEAD_DIM)

    # Find global maximum LSE across splits for numerical stability
    global_max_lse = float("-inf")
    for split_id in range(NUM_SPLITS):
        plse_offset = (
            batch_id * plse_stride_batch + head_id * plse_stride_head + split_id * plse_stride_split
        )
        lse = tl.load(partial_lse_ptr + plse_offset)
        global_max_lse = tl.maximum(global_max_lse, lse)

    # Guard: if all splits had -inf LSE (empty sequence), output zeros
    o_offset = batch_id * o_stride_batch + head_id * o_stride_head + dhead_offsets
    if global_max_lse == float("-inf"):
        tl.store(o_ptr + o_offset, tl.zeros([HEAD_DIM], dtype=tl.float32))
        return

    # Weighted combination: weight_i = exp(lse_i - global_max)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    total_weight = 0.0

    for split_id in range(NUM_SPLITS):
        plse_offset = (
            batch_id * plse_stride_batch + head_id * plse_stride_head + split_id * plse_stride_split
        )
        lse = tl.load(partial_lse_ptr + plse_offset)
        weight = tl.exp(lse - global_max_lse)

        po_base = batch_id * po_stride_batch + head_id * po_stride_head + split_id * po_stride_split
        partial_o = tl.load(partial_o_ptr + po_base + dhead_offsets)

        acc += weight * partial_o
        total_weight += weight

    # Normalize and store
    total_weight = tl.where(total_weight == 0.0, 1.0, total_weight)
    o = acc / total_weight
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
    """Optimized paged decode with GQA batching + FlashDecoding + page-aligned iteration.

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
    head_ratio = n_heads // n_kv_heads

    max_pages = kv_indices.shape[0]
    max_seq_len = max_pages * page_size

    output = out if out is not None else torch.empty_like(q)

    if batch_size == 0:
        return output

    num_splits = _get_num_splits(max_seq_len, batch_size, n_kv_heads, page_size)

    # Allocate intermediate buffers for split-K
    partial_o = torch.empty(
        batch_size,
        n_heads,
        num_splits,
        head_dim,
        dtype=torch.float32,
        device=q.device,
    )
    partial_lse = torch.empty(
        batch_size,
        n_heads,
        num_splits,
        dtype=torch.float32,
        device=q.device,
    )

    # Stage 1: GQA-batched parallel KV processing
    _flash_decode_stage1_kernel[(batch_size, n_kv_heads, num_splits)](
        q,
        kv_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        partial_o,
        partial_lse,
        # Q strides
        q.stride(0),
        q.stride(1),
        # Partial output strides
        partial_o.stride(0),
        partial_o.stride(1),
        partial_o.stride(2),
        # Partial LSE strides
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        # Cache strides
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        # Constants
        SM_SCALE=sm_scale,
        N_HEADS=n_heads,
        N_KV_HEADS=n_kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        HEAD_RATIO=head_ratio,
        NUM_SPLITS=num_splits,
    )

    # Stage 2: Combine partial results
    _flash_decode_stage2_kernel[(batch_size, n_heads)](
        partial_o,
        partial_lse,
        output,
        # Partial output strides
        partial_o.stride(0),
        partial_o.stride(1),
        partial_o.stride(2),
        # Partial LSE strides
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        # Output strides
        output.stride(0),
        output.stride(1),
        # Constants
        HEAD_DIM=head_dim,
        NUM_SPLITS=num_splits,
    )

    return output


# =============================================================================
# TRITON KERNELS - CONTEXT/PREFILL (page-aligned)
# =============================================================================
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
):
    """Context/prefill attention with paged KV cache, causal masking, and page-aligned iteration.

    Page-aligned iteration benefits over arbitrary KV_BLOCK:
    - 1 scalar page table load per page
    """
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    q_block_id = tl.program_id(axis=2)

    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS
    kv_head_id = head_id // HEAD_RATIO

    q_start = tl.load(qo_indptr_ptr + batch_id)
    q_end = tl.load(qo_indptr_ptr + batch_id + 1)
    q_len = q_end - q_start

    kv_page_start = tl.load(kv_indptr_ptr + batch_id)
    kv_page_end = tl.load(kv_indptr_ptr + batch_id + 1)
    num_kv_pages = kv_page_end - kv_page_start
    total_kv_len = tl.load(seq_len_with_cache_ptr + batch_id)

    cache_len = total_kv_len - q_len

    q_block_start = q_block_id * Q_BLOCK
    q_offsets = q_block_start + tl.arange(0, Q_BLOCK)
    q_mask = q_offsets < q_len

    if tl.sum(q_mask.to(tl.int32)) == 0:
        return

    dhead_offsets = tl.arange(0, HEAD_DIM)
    q_load_offsets = (
        (q_start + q_offsets[:, None]) * q_stride_token
        + head_id * q_stride_head
        + dhead_offsets[None, :]
    )
    q_load_mask = q_mask[:, None]
    q = tl.load(q_ptr + q_load_offsets, mask=q_load_mask, other=0.0)

    acc = tl.zeros([Q_BLOCK, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([Q_BLOCK], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([Q_BLOCK], dtype=tl.float32)

    page_offsets = tl.arange(0, PAGE_SIZE)

    # Page-aligned iteration: one page per loop iteration
    for page_idx in range(num_kv_pages):
        # Single scalar page table load (vs vector load of KV_BLOCK entries)
        physical_page = tl.load(kv_indices_ptr + kv_page_start + page_idx)

        # Valid tokens in this page
        kv_base_pos = page_idx * PAGE_SIZE
        valid_tokens = tl.minimum(PAGE_SIZE, total_kv_len - kv_base_pos)
        page_mask = page_offsets < valid_tokens

        # Contiguous KV load: all tokens in a page are adjacent in memory
        cache_base = (
            physical_page.to(tl.int64) * cache_stride_block
            + kv_head_id * cache_stride_head
            + page_offsets[:, None] * cache_stride_token
            + dhead_offsets[None, :]
        )
        page_mask_2d = page_mask[:, None]

        k = tl.load(kv_cache_ptr + cache_base, mask=page_mask_2d, other=0.0)
        v = tl.load(
            kv_cache_ptr + cache_base + cache_stride_kv,
            mask=page_mask_2d,
            other=0.0,
        )

        # QK^T: [Q_BLOCK, HEAD_DIM] @ [HEAD_DIM, PAGE_SIZE] -> [Q_BLOCK, PAGE_SIZE]
        qk = tl.dot(q, tl.trans(k)) * SM_SCALE

        # Causal mask: q_pos (in full sequence) >= kv_pos
        q_positions = q_offsets[:, None] + cache_len
        kv_positions = kv_base_pos + page_offsets[None, :]
        causal_mask = q_positions >= kv_positions

        full_mask = q_mask[:, None] & causal_mask & page_mask[None, :]
        qk = tl.where(full_mask, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        # P@V: [Q_BLOCK, PAGE_SIZE] @ [PAGE_SIZE, HEAD_DIM] -> [Q_BLOCK, HEAD_DIM]
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_i[:, None]

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
    """Context/prefill attention with paged KV cache."""
    total_tokens, n_heads, head_dim = q.shape
    _, _, n_kv_heads, page_size, _ = kv_cache.shape
    num_seq = qo_indptr.shape[0] - 1

    output = out if out is not None else torch.empty_like(q)

    Q_BLOCK = 32

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
    )

    return output


@torch.library.custom_op("auto_deploy::triton_paged_prepare_metadata", mutates_args=())
def prepare_triton_paged_metadata(
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
) -> List[torch.Tensor]:
    """Prepare metadata for Triton paged attention."""
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_tokens = num_prefill_tokens + num_decode

    qo_indptr = cu_seqlen[: num_seq + 1]

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr, seq_len_with_cache[:num_seq], num_tokens
    )

    return batch_indices, positions


@prepare_triton_paged_metadata.register_fake
def prepare_triton_paged_metadata_fake(
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


@torch.library.custom_op("auto_deploy::triton_paged_mha_with_cache", mutates_args=())
def triton_paged_mha_with_cache(
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
    """Triton paged attention with mixed batch support."""
    head_dim = kv_cache.shape[-1]
    q_shape_og = q.shape
    b, s = q_shape_og[:2]

    q = q.reshape(b * s, -1, head_dim).contiguous()
    k = k.reshape(b * s, -1, head_dim).contiguous()
    v = v.reshape(b * s, -1, head_dim).contiguous()

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


@triton_paged_mha_with_cache.register_fake
def triton_paged_mha_with_cache_fake(
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


@AttentionRegistry.register("triton_paged")
class TritonPagedAttention(AttentionDescriptor):
    """Descriptor for Triton Paged Attention backend.

    Optimized with GQA head batching, FlashDecoding, and page-aligned iteration.
    """

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        return 3

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_attention

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.triton_paged_mha_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
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
            torch.ops.auto_deploy.triton_paged_prepare_metadata.default,
            2,
            [],
        )

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
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

        attn_mask, dropout_p, is_causal = extract_op_args(
            source_attn_node, "attn_mask", "dropout_p", "is_causal"
        )
        if attn_mask is not None or dropout_p != 0.0 or not is_causal:
            ad_logger.debug(
                "Unsupported attention arguments for "
                f"{source_attn_node=}: {attn_mask=}, {dropout_p=}, {is_causal=}"
            )

        if len(source_attn_node.args) > 6:
            scale = source_attn_node.args[6]
        else:
            scale = source_attn_node.kwargs.get("scale", None)

        if not (isinstance(scale, float) or scale is None):
            ad_logger.warning(f"Provided {scale=}, is not a float. Using default scale instead.")
            scale = None

        return [scale]
