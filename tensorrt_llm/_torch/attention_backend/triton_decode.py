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
"""Triton FlashDecoding for paged KV cache.

Used as a fallback when the FlashInfer paged decode kernels cannot handle the
layer's head_dim, e.g. Gemma4's head_dim=512 full-attention layers on
architectures without trtllm-gen cubins (anything that is not datacenter
Blackwell). Companion to ``triton_prefill.py``, which covers the context phase
for the same layers.

Ported from the AutoDeploy Triton attention backend
(``tensorrt_llm/_torch/auto_deploy/custom_ops/attention/triton_attention.py``),
which uses the identical combined HND cache layout
``[num_pages, 2, num_kv_heads, page_size, head_dim]``. Stripped of the
AutoDeploy ``AttentionDescriptor``/``AttentionRegistry`` wiring and the
context-phase kernels; only the decode path is kept.

The KV cache is read with a cast to the query dtype, so an FP8 cache is
dequantized in-kernel and needs no separate conversion pass.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

# Cache SM count to avoid repeated get_device_properties calls
_NUM_SMS: Optional[int] = None

# Narrowest K that tl.dot accepts; page tiles are widened to at least this.
_MIN_TL_DOT_K = 16


def _get_num_sms() -> int:
    """Get the number of SMs on the current GPU (cached)."""
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    return _NUM_SMS


def _get_page_block(page_size: int) -> int:
    """Return the page tile width used by Triton dot kernels."""
    return max(_MIN_TL_DOT_K, 1 << (page_size - 1).bit_length())


def _get_num_splits(max_seq_len: int, batch_size: int, n_kv_heads: int, page_size: int) -> int:
    """Compute optimal number of KV splits for FlashDecoding.

    With GQA batching, the grid is (batch, n_kv_heads, num_splits).
    We want enough blocks to saturate the GPU.
    """
    if max_seq_len <= 0:
        return 1

    num_sms = _get_num_sms()
    existing_parallelism = batch_size * n_kv_heads

    # Already enough parallelism
    if existing_parallelism >= num_sms * 2:
        return 1

    # Target ~4 waves of thread blocks
    target_blocks = num_sms * 4
    num_splits = max(1, (target_blocks + existing_parallelism - 1) // existing_parallelism)

    # Cap splits so each block has at least 2 pages of work. With fewer pages,
    # the per-block overhead (Q load, accumulator init, partial_o/lse store,
    # plus stage2 reduction cost) dominates the useful compute (page-loop
    # iterations). 2 pages is a conservative lower bound to keep the
    # overhead-to-work ratio acceptable.
    max_pages = max_seq_len // page_size
    max_splits = max(1, max_pages // 2)
    num_splits = min(num_splits, max_splits)

    # Round to next power of 2 for Triton compile caching
    if num_splits > 1:
        num_splits = 2 ** math.ceil(math.log2(num_splits))

    return min(num_splits, 128)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["HEAD_DIM", "PAGE_SIZE", "PAGE_BLOCK", "HEAD_RATIO_PADDED", "SLIDING_WINDOW"],
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
    PAGE_BLOCK: tl.constexpr,
    HEAD_RATIO: tl.constexpr,
    HEAD_RATIO_PADDED: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr = 0,
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

    # Sliding window: restrict attention to pages within the window.
    # Compute the total sequence length and the first valid KV position.
    seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len
    if SLIDING_WINDOW > 0:
        first_valid_pos = tl.maximum(0, seq_len - SLIDING_WINDOW)
        first_window_page = first_valid_pos // PAGE_SIZE
    else:
        first_valid_pos = 0
        first_window_page = 0

    # Only split over pages within the window
    window_pages = num_pages - first_window_page
    pages_per_split = (window_pages + NUM_SPLITS - 1) // NUM_SPLITS
    page_split_start = first_window_page + split_id * pages_per_split
    page_split_end = tl.minimum(page_split_start + pages_per_split, num_pages)

    dhead_offsets = tl.arange(0, HEAD_DIM)
    # Use padded range for Triton power-of-2 requirement; mask out-of-bounds heads
    head_local = tl.arange(0, HEAD_RATIO_PADDED)
    head_ids = kv_head_id * HEAD_RATIO + head_local
    head_mask = head_local < HEAD_RATIO

    # Handle inactive splits (beyond the sequence's pages)
    if page_split_start >= num_pages:
        # Store zeros + -inf LSE for valid HEAD_RATIO Q heads only
        po_offsets = (
            batch_id * po_stride_batch
            + head_ids[:, None] * po_stride_head
            + split_id * po_stride_split
            + dhead_offsets[None, :]
        )
        tl.store(
            partial_o_ptr + po_offsets,
            tl.zeros([HEAD_RATIO_PADDED, HEAD_DIM], dtype=tl.float32),
            mask=head_mask[:, None],
        )
        plse_offsets = (
            batch_id * plse_stride_batch
            + head_ids * plse_stride_head
            + split_id * plse_stride_split
        )
        tl.store(
            partial_lse_ptr + plse_offsets,
            tl.zeros([HEAD_RATIO_PADDED], dtype=tl.float32) + float("-inf"),
            mask=head_mask,
        )
        return

    # Load Q for HEAD_RATIO heads sharing this KV head: [HEAD_RATIO_PADDED, HEAD_DIM]
    # Padded rows get zeros, producing zero attention scores (harmless, never stored)
    q_offsets = (
        batch_id * q_stride_batch + head_ids[:, None] * q_stride_head + dhead_offsets[None, :]
    )
    q_all = tl.load(q_ptr + q_offsets, mask=head_mask[:, None], other=0.0)

    acc = tl.zeros([HEAD_RATIO_PADDED, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([HEAD_RATIO_PADDED], dtype=tl.float32) + float("-inf")
    l_i = tl.zeros([HEAD_RATIO_PADDED], dtype=tl.float32)

    num_pages_this_split = page_split_end - page_split_start
    for local_page_idx in range(num_pages_this_split):
        page_idx = page_split_start + local_page_idx
        physical_page = tl.load(kv_indices_ptr + kv_page_start + page_idx)

        # Determine valid tokens in this page
        is_last_page_of_seq = page_idx == (num_pages - 1)
        valid_tokens = tl.where(is_last_page_of_seq, last_page_len, PAGE_SIZE)

        page_offsets = tl.arange(0, PAGE_BLOCK)
        page_mask = page_offsets < valid_tokens

        # Compute cache offset (use int64 to avoid overflow when
        # physical_page * stride > 2^31)
        cache_base = (
            physical_page.to(tl.int64) * cache_stride_block
            + kv_head_id * cache_stride_head
            + page_offsets[:, None] * cache_stride_token
            + dhead_offsets[None, :]
        )
        page_mask_2d = page_mask[:, None]

        k = tl.load(kv_cache_ptr + cache_base, mask=page_mask_2d, other=0.0).to(
            q_all.dtype
        )  # [PAGE_BLOCK, HEAD_DIM]; cast from fp8 if kv cache is fp8
        v = tl.load(
            kv_cache_ptr + cache_base + cache_stride_kv,
            mask=page_mask_2d,
            other=0.0,
        ).to(k.dtype)  # [PAGE_BLOCK, HEAD_DIM]; cast from fp8 if kv cache is fp8

        # [HEAD_RATIO_PADDED, HEAD_DIM] @ [HEAD_DIM, PAGE_BLOCK]
        #   -> [HEAD_RATIO_PADDED, PAGE_BLOCK]
        attn = tl.dot(q_all, tl.trans(k)) * SM_SCALE

        # Combine validity mask with sliding window mask
        if SLIDING_WINDOW > 0:
            global_pos = page_idx * PAGE_SIZE + page_offsets
            window_mask = global_pos >= first_valid_pos
            attn = tl.where(page_mask[None, :] & window_mask[None, :], attn, float("-inf"))
        else:
            attn = tl.where(page_mask[None, :], attn, float("-inf"))

        # Online softmax update (vectorized over HEAD_RATIO_PADDED)
        m_ij = tl.max(attn, axis=1)  # [HEAD_RATIO_PADDED]
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(attn - m_i_new[:, None])  # [HEAD_RATIO_PADDED, PAGE_BLOCK]

        # [HEAD_RATIO_PADDED, PAGE_BLOCK] @ [PAGE_BLOCK, HEAD_DIM]
        #   -> [HEAD_RATIO_PADDED, HEAD_DIM]
        acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # Finalize: normalize and compute LSE
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    partial_o_val = acc / l_i_safe[:, None]  # [HEAD_RATIO_PADDED, HEAD_DIM]
    lse_val = m_i + tl.log(l_i_safe)  # [HEAD_RATIO_PADDED]

    # Store results for valid HEAD_RATIO Q heads only (masked 2D store)
    po_offsets = (
        batch_id * po_stride_batch
        + head_ids[:, None] * po_stride_head
        + split_id * po_stride_split
        + dhead_offsets[None, :]
    )
    tl.store(partial_o_ptr + po_offsets, partial_o_val, mask=head_mask[:, None])

    plse_offsets = (
        batch_id * plse_stride_batch + head_ids * plse_stride_head + split_id * plse_stride_split
    )
    tl.store(partial_lse_ptr + plse_offsets, lse_val, mask=head_mask)


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


def triton_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    sm_scale: float,
    sliding_window: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Paged decode with GQA batching + FlashDecoding + page-aligned iteration.

    Args:
        q: Query tensor [batch_size, n_heads, head_dim]
        kv_cache: Combined cache [num_blocks, 2, n_kv_heads, page_size, head_dim]
        kv_indices: Physical page indices (flattened)
        kv_indptr: Cumulative page counts [batch_size + 1]
        kv_last_page_len: Valid tokens in last page [batch_size]
        sm_scale: Softmax scale factor
        sliding_window: If set, only attend to the last sliding_window tokens
        out: Optional output tensor [batch_size, n_heads, head_dim]

    Returns:
        Output tensor [batch_size, n_heads, head_dim]
    """
    batch_size, n_heads, head_dim = q.shape
    _, _, n_kv_heads, page_size, _ = kv_cache.shape
    head_ratio = n_heads // n_kv_heads
    head_ratio_padded = max(1, 2 ** math.ceil(math.log2(head_ratio))) if head_ratio > 1 else 1
    page_block = _get_page_block(page_size)

    max_pages = kv_indices.shape[0]
    max_seq_len = max_pages * page_size
    # Normalize sliding_window: None/non-positive -> 0 (full attention)
    sw = sliding_window if isinstance(sliding_window, int) and sliding_window > 0 else 0

    output = out if out is not None else torch.empty_like(q)

    if batch_size == 0:
        return output

    # Use effective sequence length (capped by sliding window) for split-K heuristic
    effective_seq_len = min(max_seq_len, sw) if sw > 0 else max_seq_len
    num_splits = _get_num_splits(effective_seq_len, batch_size, n_kv_heads, page_size)

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
        PAGE_BLOCK=page_block,
        HEAD_RATIO=head_ratio,
        HEAD_RATIO_PADDED=head_ratio_padded,
        NUM_SPLITS=num_splits,
        SLIDING_WINDOW=sw,
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
