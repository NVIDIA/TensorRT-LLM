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
import triton.language.extra.cuda.libdevice as tl_libdevice
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

# Cache SM count to avoid repeated get_device_properties calls
_NUM_SMS: Optional[int] = None


def _get_num_sms() -> int:
    """Get the number of SMs on the current GPU (cached)."""
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    return _NUM_SMS


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
    HEAD_DIM_PADDED: tl.constexpr,
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

    head_offsets = tl.arange(0, HEAD_DIM_PADDED)
    head_dim_mask = head_offsets < HEAD_DIM
    kv_offset = token_id * N_KV_HEADS * HEAD_DIM + head_id * HEAD_DIM + head_offsets

    k = tl.load(k_ptr + kv_offset, mask=head_dim_mask, other=0.0)
    v = tl.load(v_ptr + kv_offset, mask=head_dim_mask, other=0.0)

    # Compute cache offset (use int64 to avoid overflow when physical_page * stride > 2^31)
    cache_base = (
        physical_page.to(tl.int64) * cache_stride_block
        + head_id * cache_stride_head
        + offset_in_page.to(tl.int64) * cache_stride_token
        + head_offsets
    )

    tl.store(kv_cache_ptr + cache_base, k, mask=head_dim_mask)
    tl.store(kv_cache_ptr + cache_base + cache_stride_kv, v, mask=head_dim_mask)


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
        HEAD_DIM_PADDED=triton.next_power_of_2(head_dim),
        PAGE_SIZE=page_size,
        cache_stride_block=kv_cache.stride(0),
        cache_stride_kv=kv_cache.stride(1),
        cache_stride_head=kv_cache.stride(2),
        cache_stride_token=kv_cache.stride(3),
    )


# =============================================================================
# FLASH DECODE - HELPERS
# =============================================================================


def _get_num_splits(
    max_seq_len: int,
    batch_size: int,
    n_kv_heads: int,
    page_size: int,
    sliding_window: int = 0,
) -> int:
    """Compute optimal number of KV splits for FlashDecoding.

    With GQA batching, the grid is (batch, n_kv_heads, num_splits).
    We want enough blocks to saturate the GPU.
    """
    if max_seq_len <= 0:
        return 1

    num_sms = _get_num_sms()
    existing_parallelism = batch_size * n_kv_heads

    # Use a lower splits=1 threshold for non-SW shapes: non-SW inner loops are cheaper
    # (no per-page window_mask ops), so fewer longer splits are fine. SW shapes keep the
    # higher 2x num_sms threshold to avoid long SW-masked loops.
    splits1_threshold = num_sms if sliding_window == 0 else num_sms * 2
    if existing_parallelism >= splits1_threshold:
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

    # Cap at 32 to limit stage2 serial reduction iterations for long-context shapes.
    # Stage2 loops over num_splits sequentially; 32 is sufficient to saturate the GPU
    # for all practical batch sizes while keeping stage2 fast.
    return min(num_splits, 32)


@triton.autotune(
    configs=[
        # Original 6 configs
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        # expand to num_warps=16, num_stages=4/5
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=1, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=5),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=5),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=4),
        # deeper pipeline stages for long inner loops (S3/S4/L shapes).
        # Higher num_stages hides HBM latency by prefetching more K/V tiles in advance.
        triton.Config({}, num_warps=4, num_stages=6),
        triton.Config({}, num_warps=4, num_stages=7),
        triton.Config({}, num_warps=8, num_stages=6),
        triton.Config({}, num_warps=8, num_stages=7),
        triton.Config({}, num_warps=16, num_stages=5),
        triton.Config({}, num_warps=16, num_stages=6),
    ],
    key=[
        "HEAD_DIM",
        "HEAD_DIM_PADDED",
        "PAGE_SIZE",
        "HEAD_RATIO_PADDED",
        "SLIDING_WINDOW",
        "LOGIT_CAP",
        "SKIP_SW_MASK",
        "WRITE_DIRECT",
    ],
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
    # Intermediate outputs (unused when WRITE_DIRECT=True)
    partial_o_ptr,
    partial_lse_ptr,
    # Direct output pointer: only used when WRITE_DIRECT=True (num_splits==1 path)
    direct_o_ptr,
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
    # Direct output strides: [batch, n_heads, head_dim] — only used when WRITE_DIRECT=True
    do_stride_batch: tl.constexpr,
    do_stride_head: tl.constexpr,
    # Constants
    SM_SCALE: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_RATIO: tl.constexpr,
    HEAD_RATIO_PADDED: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr = 0,
    LOGIT_CAP: tl.constexpr = 0.0,
    SKIP_SW_MASK: tl.constexpr = False,
    WRITE_DIRECT: tl.constexpr = False,  # Write output directly to final buffer, skipping partial bufs
    OUT_DTYPE: tl.constexpr = tl.bfloat16,  # Output dtype (must match Q tensor dtype)
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

    # HEAD_DIM_PADDED >= HEAD_DIM, rounded to next power-of-2 for tl.arange compatibility.
    # Elements in [HEAD_DIM, HEAD_DIM_PADDED) are masked to zero so dot products are unaffected.
    dhead_offsets = tl.arange(0, HEAD_DIM_PADDED)
    head_dim_mask = dhead_offsets < HEAD_DIM
    # Use padded range for Triton power-of-2 requirement; mask out-of-bounds heads
    head_local = tl.arange(0, HEAD_RATIO_PADDED)
    head_ids = kv_head_id * HEAD_RATIO + head_local
    head_mask = head_local < HEAD_RATIO

    # Handle inactive splits (beyond the sequence's pages)
    if page_split_start >= num_pages:
        if WRITE_DIRECT:
            # Write zeros directly to output (inactive split, WRITE_DIRECT path).
            do_offsets = (
                batch_id * do_stride_batch
                + head_ids[:, None] * do_stride_head
                + dhead_offsets[None, :]
            )
            tl.store(
                direct_o_ptr + do_offsets,
                tl.zeros([HEAD_RATIO_PADDED, HEAD_DIM_PADDED], dtype=OUT_DTYPE),
                mask=head_mask[:, None] & head_dim_mask[None, :],
            )
        else:
            # Store zeros + -inf LSE for valid HEAD_RATIO Q heads only
            po_offsets = (
                batch_id * po_stride_batch
                + head_ids[:, None] * po_stride_head
                + split_id * po_stride_split
                + dhead_offsets[None, :]
            )
            tl.store(
                partial_o_ptr + po_offsets,
                tl.zeros([HEAD_RATIO_PADDED, HEAD_DIM_PADDED], dtype=tl.float32),
                mask=head_mask[:, None] & head_dim_mask[None, :],
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

    # Load Q for HEAD_RATIO heads sharing this KV head: [HEAD_RATIO_PADDED, HEAD_DIM_PADDED]
    # Padded rows (head_local >= HEAD_RATIO) and padded cols (dhead >= HEAD_DIM) are zero.
    q_offsets = (
        batch_id * q_stride_batch + head_ids[:, None] * q_stride_head + dhead_offsets[None, :]
    )
    q_all = tl.load(q_ptr + q_offsets, mask=head_mask[:, None] & head_dim_mask[None, :], other=0.0)

    acc = tl.zeros([HEAD_RATIO_PADDED, HEAD_DIM_PADDED], dtype=tl.float32)
    m_i = tl.zeros([HEAD_RATIO_PADDED], dtype=tl.float32) + float("-inf")
    l_i = tl.zeros([HEAD_RATIO_PADDED], dtype=tl.float32)

    num_pages_this_split = page_split_end - page_split_start
    page_offsets = tl.arange(0, PAGE_SIZE)
    # precompute loop-invariant offset table [PAGE_SIZE, HEAD_DIM_PADDED] outside loop.
    # Saves one vector multiply per iteration vs computing inside the loop.
    kv_head_base = kv_head_id * cache_stride_head
    local_kv = page_offsets[:, None] * cache_stride_token + dhead_offsets[None, :]

    # SKIP_SW_MASK is a compile-time constexpr set by the launcher when all sequences
    # in the batch satisfy seq_len <= SLIDING_WINDOW (so first_valid_pos=0 for every thread).
    # When True, the SW-masking branch is dead code and the compiler elides it entirely,
    # giving the same register footprint as SLIDING_WINDOW=0 kernels.
    # When False (seq_len > SW possible), fall back to the first_valid_pos_in_page dual-loop.
    if SLIDING_WINDOW > 0 and not SKIP_SW_MASK:
        # window_mask = page_offsets >= first_valid_pos_in_page. When first_valid_pos_in_page <= 0
        # (entire split within window), window_mask is all-True, equivalent to no masking.
        first_valid_pos_in_page = first_valid_pos - page_split_start * PAGE_SIZE
        for local_page_idx in range(num_pages_this_split):
            page_idx = page_split_start + local_page_idx
            physical_page = tl.load(kv_indices_ptr + kv_page_start + page_idx)
            is_last_page_of_seq = page_idx == (num_pages - 1)
            valid_tokens = tl.where(is_last_page_of_seq, last_page_len, PAGE_SIZE)
            page_mask = page_offsets < valid_tokens
            cache_page_base = physical_page.to(tl.int64) * cache_stride_block + kv_head_base
            kv_mask_2d = page_mask[:, None] & head_dim_mask[None, :]
            k = tl.load(kv_cache_ptr + cache_page_base + local_kv, mask=kv_mask_2d, other=0.0).to(
                q_all.dtype
            )  # cast from fp8 if kv cache is fp8
            v = tl.load(
                kv_cache_ptr + cache_page_base + cache_stride_kv + local_kv,
                mask=kv_mask_2d,
                other=0.0,
            ).to(q_all.dtype)  # cast from fp8 if kv cache is fp8
            attn = tl.dot(q_all, tl.trans(k)) * SM_SCALE
            if LOGIT_CAP > 0.0:
                attn = LOGIT_CAP * tl_libdevice.tanh(attn / LOGIT_CAP)
            window_mask = page_offsets >= first_valid_pos_in_page
            attn = tl.where(page_mask[None, :] & window_mask[None, :], attn, float("-inf"))
            first_valid_pos_in_page -= PAGE_SIZE
            m_ij = tl.max(attn, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(attn - m_i_new[:, None])
            acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_i_new
    else:
        # SKIP_SW_MASK=True or SLIDING_WINDOW=0: no SW masking. Compiler eliminates the
        # dual-loop block above, keeping register pressure identical to SW=0 kernels.
        for local_page_idx in range(num_pages_this_split):
            page_idx = page_split_start + local_page_idx
            physical_page = tl.load(kv_indices_ptr + kv_page_start + page_idx)

            # Determine valid tokens in this page
            is_last_page_of_seq = page_idx == (num_pages - 1)
            valid_tokens = tl.where(is_last_page_of_seq, last_page_len, PAGE_SIZE)
            page_mask = page_offsets < valid_tokens

            # Compute cache offset (use int64 to avoid overflow when physical_page * stride > 2^31)
            # cache_page_base is a scalar int64; local_kv broadcasts to [PAGE_SIZE, HEAD_DIM_PADDED]
            cache_page_base = physical_page.to(tl.int64) * cache_stride_block + kv_head_base
            kv_mask_2d = page_mask[:, None] & head_dim_mask[None, :]

            k = tl.load(
                kv_cache_ptr + cache_page_base + local_kv, mask=kv_mask_2d, other=0.0
            )  # [PAGE_SIZE, HEAD_DIM_PADDED]
            v = tl.load(
                kv_cache_ptr + cache_page_base + cache_stride_kv + local_kv,
                mask=kv_mask_2d,
                other=0.0,
            )  # [PAGE_SIZE, HEAD_DIM_PADDED]

            # [HEAD_RATIO_PADDED, HEAD_DIM_PADDED] @ [HEAD_DIM_PADDED, PAGE_SIZE] -> [HEAD_RATIO_PADDED, PAGE_SIZE]
            attn = tl.dot(q_all, tl.trans(k)) * SM_SCALE

            # logit soft-capping (Gemma-4: logit_cap=50.0). Apply BEFORE masking
            # so that -inf from page/window masks propagates cleanly through softmax.
            # use tl_libdevice.tanh (__nv_tanhf, 1 SFU instruction) instead of
            # 6-op sigmoid-based tanh: cap*tanh(x/cap) = cap * libdevice.tanh(x/cap).
            if LOGIT_CAP > 0.0:
                attn = LOGIT_CAP * tl_libdevice.tanh(attn / LOGIT_CAP)

            attn = tl.where(page_mask[None, :], attn, float("-inf"))

            # Online softmax update (vectorized over HEAD_RATIO_PADDED)
            m_ij = tl.max(attn, axis=1)  # [HEAD_RATIO_PADDED]
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(attn - m_i_new[:, None])  # [HEAD_RATIO_PADDED, PAGE_SIZE]

            # [HEAD_RATIO_PADDED, PAGE_SIZE] @ [PAGE_SIZE, HEAD_DIM_PADDED] -> [HEAD_RATIO_PADDED, HEAD_DIM_PADDED]
            acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_i_new

    # Finalize: normalize, then either write directly to output (WRITE_DIRECT) or to partial buffers.
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)

    if WRITE_DIRECT:
        # Write output dtype output directly — skips partial_o fp32 write and stage2/copy_.
        # Eliminates ~5-6µs kernel-launch overhead for splits=1 shapes.
        out_val = (acc / l_i_safe[:, None]).to(OUT_DTYPE)
        do_offsets = (
            batch_id * do_stride_batch + head_ids[:, None] * do_stride_head + dhead_offsets[None, :]
        )
        tl.store(
            direct_o_ptr + do_offsets, out_val, mask=head_mask[:, None] & head_dim_mask[None, :]
        )
    else:
        partial_o_val = acc / l_i_safe[:, None]  # [HEAD_RATIO_PADDED, HEAD_DIM_PADDED]
        lse_val = m_i + tl.log(l_i_safe)  # [HEAD_RATIO_PADDED]

        # Store results for valid HEAD_RATIO Q heads and valid head_dim elements
        po_offsets = (
            batch_id * po_stride_batch
            + head_ids[:, None] * po_stride_head
            + split_id * po_stride_split
            + dhead_offsets[None, :]
        )
        tl.store(
            partial_o_ptr + po_offsets,
            partial_o_val,
            mask=head_mask[:, None] & head_dim_mask[None, :],
        )

        plse_offsets = (
            batch_id * plse_stride_batch
            + head_ids * plse_stride_head
            + split_id * plse_stride_split
        )
        tl.store(partial_lse_ptr + plse_offsets, lse_val, mask=head_mask)


@triton.autotune(
    configs=[
        # Original 6 configs
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=1, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=5),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=5),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=4),
        # deeper pipeline stages for long inner loops (S3/S4/L shapes)
        triton.Config({}, num_warps=4, num_stages=6),
        triton.Config({}, num_warps=4, num_stages=7),
        triton.Config({}, num_warps=8, num_stages=6),
        triton.Config({}, num_warps=8, num_stages=7),
        triton.Config({}, num_warps=16, num_stages=5),
        triton.Config({}, num_warps=16, num_stages=6),
    ],
    key=[
        "HEAD_DIM",
        "HD_CHUNK1",
        "HD_CHUNK2",
        "PAGE_SIZE",
        "HEAD_RATIO_PADDED",
        "SLIDING_WINDOW",
        "LOGIT_CAP",
        "SKIP_SW_MASK",
        "WRITE_DIRECT",
    ],
)
@triton.jit
def _flash_decode_stage1_two_chunk_kernel(
    # Query input
    q_ptr,
    # KV cache (combined)
    kv_cache_ptr,
    # Page table
    kv_indices_ptr,
    kv_indptr_ptr,
    kv_last_page_len_ptr,
    # Intermediate outputs (unused when WRITE_DIRECT=True)
    partial_o_ptr,
    partial_lse_ptr,
    # Direct output pointer: only used when WRITE_DIRECT=True (num_splits==1 path)
    direct_o_ptr,
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
    # Direct output strides: [batch, n_heads, head_dim] — only used when WRITE_DIRECT=True
    do_stride_batch: tl.constexpr,
    do_stride_head: tl.constexpr,
    # Constants
    SM_SCALE: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_RATIO: tl.constexpr,
    HEAD_RATIO_PADDED: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HD_CHUNK1: tl.constexpr,  # First chunk size (power-of-2, covers indices 0..HD_CHUNK1-1)
    HD_CHUNK2: tl.constexpr,  # Second chunk size (power-of-2, covers HD_CHUNK1..HD_CHUNK1+HD_CHUNK2-1)
    SLIDING_WINDOW: tl.constexpr = 0,
    LOGIT_CAP: tl.constexpr = 0.0,
    SKIP_SW_MASK: tl.constexpr = False,
    WRITE_DIRECT: tl.constexpr = False,  # Write output directly to final buffer, skipping partial bufs
    OUT_DTYPE: tl.constexpr = tl.bfloat16,  # Output dtype (must match Q tensor dtype)
):
    """Two-chunk stage1 for non-power-of-2 head_dim (e.g. 176 = 128 + 64).

    Splits head_dim into two power-of-2 chunks so Triton tl.dot and tl.arange
    work without padding. For head_dim=176: HD_CHUNK1=128, HD_CHUNK2=64 covers
    indices 0..191, with masking for 176..191 in chunk2.
    Saves 25% KV bandwidth vs HEAD_DIM_PADDED=256 approach.
    """
    batch_id = tl.program_id(axis=0)
    kv_head_id = tl.program_id(axis=1)
    split_id = tl.program_id(axis=2)

    kv_page_start = tl.load(kv_indptr_ptr + batch_id)
    kv_page_end = tl.load(kv_indptr_ptr + batch_id + 1)
    num_pages = kv_page_end - kv_page_start
    last_page_len = tl.load(kv_last_page_len_ptr + batch_id)

    seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len
    if SLIDING_WINDOW > 0:
        first_valid_pos = tl.maximum(0, seq_len - SLIDING_WINDOW)
        first_window_page = first_valid_pos // PAGE_SIZE
    else:
        first_valid_pos = 0
        first_window_page = 0

    window_pages = num_pages - first_window_page
    pages_per_split = (window_pages + NUM_SPLITS - 1) // NUM_SPLITS
    page_split_start = first_window_page + split_id * pages_per_split
    page_split_end = tl.minimum(page_split_start + pages_per_split, num_pages)

    # Head-dim chunk offsets and mask for chunk2 (chunk1 is fully valid)
    dhead_c1 = tl.arange(0, HD_CHUNK1)  # indices 0..HD_CHUNK1-1 (all valid)
    dhead_c2 = HD_CHUNK1 + tl.arange(0, HD_CHUNK2)  # indices HD_CHUNK1..HD_CHUNK1+HD_CHUNK2-1
    head_dim_mask_c2 = dhead_c2 < HEAD_DIM  # mask out padding in chunk2

    head_local = tl.arange(0, HEAD_RATIO_PADDED)
    head_ids = kv_head_id * HEAD_RATIO + head_local
    head_mask = head_local < HEAD_RATIO

    if page_split_start >= num_pages:
        if WRITE_DIRECT:
            # write zeros directly to output (no partial buffers).
            do_base = batch_id * do_stride_batch + head_ids[:, None] * do_stride_head
            tl.store(
                direct_o_ptr + do_base + dhead_c1[None, :],
                tl.zeros([HEAD_RATIO_PADDED, HD_CHUNK1], dtype=OUT_DTYPE),
                mask=head_mask[:, None],
            )
            tl.store(
                direct_o_ptr + do_base + dhead_c2[None, :],
                tl.zeros([HEAD_RATIO_PADDED, HD_CHUNK2], dtype=OUT_DTYPE),
                mask=head_mask[:, None] & head_dim_mask_c2[None, :],
            )
        else:
            # Inactive split: store zeros + -inf LSE
            po_base = (
                batch_id * po_stride_batch
                + head_ids[:, None] * po_stride_head
                + split_id * po_stride_split
            )
            tl.store(
                partial_o_ptr + po_base + dhead_c1[None, :],
                tl.zeros([HEAD_RATIO_PADDED, HD_CHUNK1], dtype=tl.float32),
                mask=head_mask[:, None],
            )
            tl.store(
                partial_o_ptr + po_base + dhead_c2[None, :],
                tl.zeros([HEAD_RATIO_PADDED, HD_CHUNK2], dtype=tl.float32),
                mask=head_mask[:, None] & head_dim_mask_c2[None, :],
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

    # Load Q in two chunks: [HEAD_RATIO_PADDED, HD_CHUNK1] and [HEAD_RATIO_PADDED, HD_CHUNK2]
    q_base = batch_id * q_stride_batch + head_ids[:, None] * q_stride_head
    q_c1 = tl.load(q_ptr + q_base + dhead_c1[None, :], mask=head_mask[:, None], other=0.0)
    q_c2 = tl.load(
        q_ptr + q_base + dhead_c2[None, :],
        mask=head_mask[:, None] & head_dim_mask_c2[None, :],
        other=0.0,
    )

    # Per-chunk accumulators
    acc_c1 = tl.zeros([HEAD_RATIO_PADDED, HD_CHUNK1], dtype=tl.float32)
    acc_c2 = tl.zeros([HEAD_RATIO_PADDED, HD_CHUNK2], dtype=tl.float32)
    m_i = tl.zeros([HEAD_RATIO_PADDED], dtype=tl.float32) + float("-inf")
    l_i = tl.zeros([HEAD_RATIO_PADDED], dtype=tl.float32)

    page_offsets = tl.arange(0, PAGE_SIZE)
    num_pages_this_split = page_split_end - page_split_start
    # precompute loop-invariant KV load offsets [PAGE_SIZE, HD_CHUNK{1,2}] outside loop.
    # kv_head_base is a scalar; local_kv_c{1,2} broadcast when added to page scalar.
    kv_head_base = kv_head_id * cache_stride_head
    local_kv_c1 = (
        page_offsets[:, None] * cache_stride_token + dhead_c1[None, :]
    )  # [PAGE_SIZE, HD_C1]
    local_kv_c2 = (
        page_offsets[:, None] * cache_stride_token + dhead_c2[None, :]
    )  # [PAGE_SIZE, HD_C2]
    # SKIP_SW_MASK is a compile-time constexpr set by the launcher when all sequences
    # in the batch satisfy seq_len <= SLIDING_WINDOW (so first_valid_pos=0 for every thread).
    # When True, the SW-masking branch is dead code and the compiler elides it entirely,
    # giving the same register footprint as SLIDING_WINDOW=0 kernels.
    if SLIDING_WINDOW > 0 and not SKIP_SW_MASK:
        # window_mask = page_offsets >= first_valid_pos_in_page. When first_valid_pos_in_page <= 0
        # (entire split within window), window_mask is all-True, equivalent to no masking.
        first_valid_pos_in_page = first_valid_pos - page_split_start * PAGE_SIZE
        for local_page_idx in range(num_pages_this_split):
            page_idx = page_split_start + local_page_idx
            physical_page = tl.load(kv_indices_ptr + kv_page_start + page_idx)
            is_last_page_of_seq = page_idx == (num_pages - 1)
            valid_tokens = tl.where(is_last_page_of_seq, last_page_len, PAGE_SIZE)
            page_mask = page_offsets < valid_tokens
            cache_page_base = physical_page.to(tl.int64) * cache_stride_block + kv_head_base
            k_c1 = tl.load(
                kv_cache_ptr + cache_page_base + local_kv_c1, mask=page_mask[:, None], other=0.0
            )
            k_c2 = tl.load(
                kv_cache_ptr + cache_page_base + local_kv_c2,
                mask=page_mask[:, None] & head_dim_mask_c2[None, :],
                other=0.0,
            )
            attn = (tl.dot(q_c1, tl.trans(k_c1)) + tl.dot(q_c2, tl.trans(k_c2))) * SM_SCALE
            if LOGIT_CAP > 0.0:
                attn = LOGIT_CAP * tl_libdevice.tanh(attn / LOGIT_CAP)
            window_mask = page_offsets >= first_valid_pos_in_page
            attn = tl.where(page_mask[None, :] & window_mask[None, :], attn, float("-inf"))
            first_valid_pos_in_page -= PAGE_SIZE
            m_ij = tl.max(attn, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(attn - m_i_new[:, None])
            v_c1 = tl.load(
                kv_cache_ptr + cache_page_base + cache_stride_kv + local_kv_c1,
                mask=page_mask[:, None],
                other=0.0,
            )
            v_c2 = tl.load(
                kv_cache_ptr + cache_page_base + cache_stride_kv + local_kv_c2,
                mask=page_mask[:, None] & head_dim_mask_c2[None, :],
                other=0.0,
            )
            acc_c1 = tl.dot(p.to(v_c1.dtype), v_c1, acc=acc_c1 * alpha[:, None])
            acc_c2 = tl.dot(p.to(v_c2.dtype), v_c2, acc=acc_c2 * alpha[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_i_new
    else:
        # SKIP_SW_MASK=True or SLIDING_WINDOW=0: no SW masking. Compiler eliminates the
        # dual-loop block above entirely, preserving the same register footprint as SW=0.
        for local_page_idx in range(num_pages_this_split):
            page_idx = page_split_start + local_page_idx
            physical_page = tl.load(kv_indices_ptr + kv_page_start + page_idx)

            is_last_page_of_seq = page_idx == (num_pages - 1)
            valid_tokens = tl.where(is_last_page_of_seq, last_page_len, PAGE_SIZE)
            page_mask = page_offsets < valid_tokens

            # Cache page base: scalar int64 — local_kv_c{1,2} broadcast to [PAGE_SIZE, HD_CHUNK{1,2}]
            cache_page_base = physical_page.to(tl.int64) * cache_stride_block + kv_head_base

            # Load K in two chunks: [PAGE_SIZE, HD_CHUNK1] and [PAGE_SIZE, HD_CHUNK2]
            k_c1 = tl.load(
                kv_cache_ptr + cache_page_base + local_kv_c1,
                mask=page_mask[:, None],
                other=0.0,
            )
            k_c2 = tl.load(
                kv_cache_ptr + cache_page_base + local_kv_c2,
                mask=page_mask[:, None] & head_dim_mask_c2[None, :],
                other=0.0,
            )

            # QK: sum contributions from both chunks → [HEAD_RATIO_PADDED, PAGE_SIZE]
            attn = (tl.dot(q_c1, tl.trans(k_c1)) + tl.dot(q_c2, tl.trans(k_c2))) * SM_SCALE

            # logit soft-capping; iter 24: libdevice.tanh (1 SFU) vs 6-op sigmoid approx.
            if LOGIT_CAP > 0.0:
                attn = LOGIT_CAP * tl_libdevice.tanh(attn / LOGIT_CAP)

            attn = tl.where(page_mask[None, :], attn, float("-inf"))

            m_ij = tl.max(attn, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(attn - m_i_new[:, None])  # [HEAD_RATIO_PADDED, PAGE_SIZE]

            # Load V in two chunks (using precomputed local_kv_c{1,2} offsets)
            v_c1 = tl.load(
                kv_cache_ptr + cache_page_base + cache_stride_kv + local_kv_c1,
                mask=page_mask[:, None],
                other=0.0,
            )
            v_c2 = tl.load(
                kv_cache_ptr + cache_page_base + cache_stride_kv + local_kv_c2,
                mask=page_mask[:, None] & head_dim_mask_c2[None, :],
                other=0.0,
            )

            # PV: update per-chunk accumulators
            acc_c1 = tl.dot(p.to(v_c1.dtype), v_c1, acc=acc_c1 * alpha[:, None])
            acc_c2 = tl.dot(p.to(v_c2.dtype), v_c2, acc=acc_c2 * alpha[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_i_new

    # Normalize and store: either directly to bf16 output (WRITE_DIRECT) or to partial buffers.
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)

    if WRITE_DIRECT:
        # write bf16 output directly — skips partial_o fp32 write and stage2/copy_.
        do_base = batch_id * do_stride_batch + head_ids[:, None] * do_stride_head
        tl.store(
            direct_o_ptr + do_base + dhead_c1[None, :],
            (acc_c1 / l_i_safe[:, None]).to(OUT_DTYPE),
            mask=head_mask[:, None],
        )
        tl.store(
            direct_o_ptr + do_base + dhead_c2[None, :],
            (acc_c2 / l_i_safe[:, None]).to(OUT_DTYPE),
            mask=head_mask[:, None] & head_dim_mask_c2[None, :],
        )
    else:
        po_base = (
            batch_id * po_stride_batch
            + head_ids[:, None] * po_stride_head
            + split_id * po_stride_split
        )
        tl.store(
            partial_o_ptr + po_base + dhead_c1[None, :],
            acc_c1 / l_i_safe[:, None],
            mask=head_mask[:, None],
        )
        tl.store(
            partial_o_ptr + po_base + dhead_c2[None, :],
            acc_c2 / l_i_safe[:, None],
            mask=head_mask[:, None] & head_dim_mask_c2[None, :],
        )
        plse_offsets = (
            batch_id * plse_stride_batch
            + head_ids * plse_stride_head
            + split_id * plse_stride_split
        )
        tl.store(partial_lse_ptr + plse_offsets, m_i + tl.log(l_i_safe), mask=head_mask)


@triton.autotune(
    configs=[
        # sweep BLOCK_HD and num_warps for stage2
        triton.Config({"BLOCK_HD": 32}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_HD": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_HD": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HD": 64}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_HD": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_HD": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HD": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_HD": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HD": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HD": 256}, num_warps=8, num_stages=1),
    ],
    key=["HEAD_DIM", "HEAD_DIM_PADDED", "NUM_SPLITS"],
)
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
    HEAD_DIM_PADDED: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    """Combine partial outputs from all splits for one (batch, head, hd_block) triple.

    Grid axis 2 tiles HEAD_DIM into BLOCK_HD-element chunks, giving more SM parallelism
    vs the original (batch, n_heads) 2D grid. Each program reads all NUM_SPLITS LSE
    scalars and its BLOCK_HD slice of each split's partial output.

    Two-pass algorithm: pass 1 finds global max LSE (cheap scalar ops), pass 2
    computes the weighted sum using the known max (one tl.exp per split vs two for
    the one-pass online approach). Batch/head offsets hoisted outside both loops.
    """
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    hd_block_id = tl.program_id(axis=2)

    # This tile's head_dim range
    hd_start = hd_block_id * BLOCK_HD
    dhead_offsets = hd_start + tl.arange(0, BLOCK_HD)
    head_dim_mask = dhead_offsets < HEAD_DIM

    # Early exit: entire tile is padding (only happens when HEAD_DIM_PADDED > HEAD_DIM)
    if hd_start >= HEAD_DIM:
        return

    # Hoist batch/head base offsets outside both loops.
    plse_head_base = batch_id * plse_stride_batch + head_id * plse_stride_head
    po_head_base = batch_id * po_stride_batch + head_id * po_stride_head
    o_offset = batch_id * o_stride_batch + head_id * o_stride_head + dhead_offsets

    # Pass 1: find global max LSE across all splits (scalar ops only).
    global_max_lse = float("-inf")
    for split_id in range(NUM_SPLITS):
        lse = tl.load(partial_lse_ptr + plse_head_base + split_id * plse_stride_split)
        global_max_lse = tl.maximum(global_max_lse, lse)

    # Guard: if all splits had -inf LSE (empty sequence), output zeros.
    if global_max_lse == float("-inf"):
        tl.store(o_ptr + o_offset, tl.zeros([BLOCK_HD], dtype=tl.float32), mask=head_dim_mask)
        return

    # Pass 2: weighted sum using global max — only ONE tl.exp per split.
    norm = 0.0
    acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
    for split_id in range(NUM_SPLITS):
        lse = tl.load(partial_lse_ptr + plse_head_base + split_id * plse_stride_split)
        weight = tl.exp(lse - global_max_lse)
        partial_o = tl.load(
            partial_o_ptr + po_head_base + split_id * po_stride_split + dhead_offsets,
            mask=head_dim_mask,
            other=0.0,
        )
        acc += weight * partial_o
        norm += weight

    # Normalize and store.
    norm = tl.where(norm == 0.0, 1.0, norm)
    tl.store(o_ptr + o_offset, acc / norm, mask=head_dim_mask)


def triton_paged_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    sm_scale: float,
    sliding_window: Optional[int] = None,
    logit_cap: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    max_decode_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """Optimized paged decode with GQA batching + FlashDecoding + page-aligned iteration.

    Args:
        q: Query tensor [batch_size, n_heads, head_dim]
        kv_cache: Combined cache [num_blocks, 2, n_kv_heads, page_size, head_dim]
        kv_indices: Physical page indices (flattened)
        kv_indptr: Cumulative page counts [batch_size + 1]
        kv_last_page_len: Valid tokens in last page [batch_size]
        sm_scale: Softmax scale factor
        sliding_window: If set, only attend to the last sliding_window tokens
        logit_cap: If set (>0), apply logit soft-capping: cap * tanh(score / cap)
        out: Optional output tensor [batch_size, n_heads, head_dim]
        max_decode_seq_len: Optional caller-provided max sequence length across the batch
            (CPU int, no GPU sync needed). When provided and <= sliding_window, enables the
            SKIP_SW_MASK=True kernel variant which eliminates SW-masking overhead entirely.
            When None, conservatively uses SKIP_SW_MASK=False (correct, no regression).

    Returns:
        Output tensor [batch_size, n_heads, head_dim]
    """
    batch_size, n_heads, head_dim = q.shape
    _, _, n_kv_heads, page_size, _ = kv_cache.shape
    head_ratio = n_heads // n_kv_heads
    head_ratio_padded = max(1, 2 ** math.ceil(math.log2(head_ratio))) if head_ratio > 1 else 1
    head_dim_padded = triton.next_power_of_2(head_dim)

    max_pages = kv_indices.shape[0]
    max_seq_len = max_pages * page_size
    # Normalize sliding_window: None/non-positive → 0 (full attention)
    sw = sliding_window if isinstance(sliding_window, int) and sliding_window > 0 else 0
    # Normalize logit_cap: None/non-positive → 0.0 (no capping)
    lc = float(logit_cap) if logit_cap is not None and logit_cap > 0.0 else 0.0

    output = out if out is not None else torch.empty_like(q)

    if batch_size == 0:
        return output

    # Use effective sequence length (capped by sliding window) for split-K heuristic
    effective_seq_len = min(max_seq_len, sw) if sw > 0 else max_seq_len
    num_splits = _get_num_splits(effective_seq_len, batch_size, n_kv_heads, page_size, sw)

    # WRITE_DIRECT=True for num_splits==1 — stage1 writes final bf16 output directly,
    # skipping the partial_o fp32 write and the subsequent copy_() kernel. Saves ~5-6µs of
    # kernel-launch overhead for splits=1 shapes (D6-D10, S4-S5 for Gemma-4/H100).
    write_direct = num_splits == 1

    if write_direct:
        # Dummy 1-element tensors; stage1 will not write to them (WRITE_DIRECT=True path).
        partial_o = torch.empty(1, dtype=torch.float32, device=q.device)
        partial_lse = torch.empty(1, dtype=torch.float32, device=q.device)
        po_s0, po_s1, po_s2 = 0, 0, 0
        plse_s0, plse_s1, plse_s2 = 0, 0, 0
    else:
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
        po_s0, po_s1, po_s2 = partial_o.stride(0), partial_o.stride(1), partial_o.stride(2)
        plse_s0, plse_s1, plse_s2 = (
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
        )

    # Stage 1: GQA-batched parallel KV processing.
    # For non-power-of-2 head_dim, use the two-chunk kernel which splits head_dim into
    # two power-of-2 chunks (e.g. 176 = 128 + 64), reducing KV bandwidth by 25% vs
    # the HEAD_DIM_PADDED=256 approach.
    stage1_common_args = (
        q,
        kv_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        partial_o,
        partial_lse,
        output,  # direct_o_ptr: output tensor passed to both kernels; only used if WRITE_DIRECT
        q.stride(0),
        q.stride(1),
        po_s0,
        po_s1,
        po_s2,
        plse_s0,
        plse_s1,
        plse_s2,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        output.stride(0),  # do_stride_batch
        output.stride(1),  # do_stride_head
    )
    # SKIP_SW_MASK=True eliminates SW-masking branch at compile time (same register
    # footprint as SLIDING_WINDOW=0 kernels). Set when caller guarantees max_seq_len <= sw.
    # Use max_decode_seq_len if provided (CPU int, no GPU sync). Otherwise conservative False
    # (correct, no regression vs iter 26). Note: sw==0 case handled by SLIDING_WINDOW constexpr.
    skip_sw_mask = sw == 0 or (max_decode_seq_len is not None and max_decode_seq_len <= sw)

    out_dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    stage1_common_kwargs = dict(
        SM_SCALE=sm_scale,
        N_HEADS=n_heads,
        N_KV_HEADS=n_kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        HEAD_RATIO=head_ratio,
        HEAD_RATIO_PADDED=head_ratio_padded,
        NUM_SPLITS=num_splits,
        SLIDING_WINDOW=sw,
        LOGIT_CAP=lc,
        SKIP_SW_MASK=skip_sw_mask,
        WRITE_DIRECT=write_direct,
        OUT_DTYPE=out_dtype,
    )

    if head_dim_padded == head_dim:
        # Power-of-2 head_dim: use original kernel (no padding waste)
        _flash_decode_stage1_kernel[(batch_size, n_kv_heads, num_splits)](
            *stage1_common_args,
            HEAD_DIM_PADDED=head_dim_padded,
            **stage1_common_kwargs,
        )
    else:
        # Non-power-of-2 head_dim: two-chunk kernel for ~25% bandwidth savings.
        # Split head_dim into two power-of-2 chunks: e.g. head_dim=176 → 128 + 64.
        # Chunk2 is padded to the next power-of-2 >= (head_dim - chunk1); padding masked out.
        hd_chunk1 = (
            head_dim_padded // 2
        )  # Largest power-of-2 that fits in head_dim (< head_dim_padded)
        hd_chunk2 = triton.next_power_of_2(
            head_dim - hd_chunk1
        )  # Covers remainder with minimal padding
        _flash_decode_stage1_two_chunk_kernel[(batch_size, n_kv_heads, num_splits)](
            *stage1_common_args,
            HD_CHUNK1=hd_chunk1,
            HD_CHUNK2=hd_chunk2,
            **stage1_common_kwargs,
        )

    if not write_direct:
        # Stage 2: Combine partial results — tiled over HEAD_DIM for more SM parallelism.
        # BLOCK_HD is autotuned; use lambda grid so it adapts to the chosen config.
        _flash_decode_stage2_kernel[
            lambda meta: (batch_size, n_heads, triton.cdiv(head_dim_padded, meta["BLOCK_HD"]))
        ](
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
            HEAD_DIM_PADDED=head_dim_padded,
            NUM_SPLITS=num_splits,
        )
    # WRITE_DIRECT=True (write_direct): stage1 already wrote to output directly; nothing to do.

    return output


# =============================================================================
# TRITON KERNELS - CONTEXT/PREFILL (page-aligned, causal skip, autotuned)
# =============================================================================
@triton.autotune(
    configs=[
        # Original 6 configs
        triton.Config({"Q_BLOCK": 64}, num_stages=2, num_warps=2),
        triton.Config({"Q_BLOCK": 64}, num_stages=2, num_warps=4),
        triton.Config({"Q_BLOCK": 64}, num_stages=4, num_warps=4),
        triton.Config({"Q_BLOCK": 128}, num_stages=2, num_warps=4),
        triton.Config({"Q_BLOCK": 128}, num_stages=2, num_warps=8),
        triton.Config({"Q_BLOCK": 128}, num_stages=3, num_warps=8),
        # additional Q_BLOCK=32 for very short prefills, and more stages
        triton.Config({"Q_BLOCK": 32}, num_stages=2, num_warps=2),
        triton.Config({"Q_BLOCK": 32}, num_stages=3, num_warps=4),
        triton.Config({"Q_BLOCK": 64}, num_stages=3, num_warps=4),
        triton.Config({"Q_BLOCK": 64}, num_stages=4, num_warps=8),
        triton.Config({"Q_BLOCK": 128}, num_stages=4, num_warps=8),
        # Note: num_warps=16 (512 threads) removed — triggers register exhaustion for
        # HEAD_DIM_PADDED=256 (head_dim=176): 158 regs/thread > 128 SM limit (Triton 3.6.0).
    ],
    # add SLIDING_WINDOW to key so sw=0 and sw>0 shapes get different configs.
    # Sliding-window path has extra per-token masking and different phase1/phase2 balance.
    # add LOGIT_CAP to key — softcap path has tanh vs no-op, different perf profile.
    key=[
        "HEAD_DIM",
        "HEAD_DIM_PADDED",
        "PAGE_SIZE",
        "SLIDING_WINDOW",
        "LOGIT_CAP",
    ],
)
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
    # Autotuned
    Q_BLOCK: tl.constexpr,
    # Constants
    SM_SCALE: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr = 0,
    LOGIT_CAP: tl.constexpr = 0.0,
):
    """Context/prefill attention with paged KV cache, causal skip, and page-aligned iteration.

    Grid: (num_seq, n_heads, num_q_blocks)

    Optimizations:
    - Page-aligned iteration: 1 scalar page table load per page, no div/mod,
      contiguous KV memory access within each page.
    - Causal skip: pages entirely beyond the last Q position are skipped,
      saving ~50% of KV loads on average for causal attention.
    - Autotuned Q_BLOCK, num_stages, num_warps for best tile/pipeline config.
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

    dhead_offsets = tl.arange(0, HEAD_DIM_PADDED)
    head_dim_mask = dhead_offsets < HEAD_DIM
    q_load_offsets = (
        (q_start + q_offsets[:, None]) * q_stride_token
        + head_id * q_stride_head
        + dhead_offsets[None, :]
    )
    q_load_mask = q_mask[:, None] & head_dim_mask[None, :]
    q = tl.load(q_ptr + q_load_offsets, mask=q_load_mask, other=0.0)

    acc = tl.zeros([Q_BLOCK, HEAD_DIM_PADDED], dtype=tl.float32)
    m_i = tl.zeros([Q_BLOCK], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([Q_BLOCK], dtype=tl.float32)

    page_offsets = tl.arange(0, PAGE_SIZE)

    # Two-phase page loop:
    # Phase 1 (full pages): pages entirely before the first Q position need no causal mask.
    #   First Q position in KV coords = q_block_start + cache_len.
    #   A page ending at (page_idx+1)*PAGE_SIZE - 1 is fully attended if that's <= first Q pos.
    # Phase 2 (boundary pages): remaining pages up to last Q position need causal masking.
    first_q_kv_pos = q_block_start + cache_len
    max_q_pos = q_block_start + Q_BLOCK - 1 + cache_len

    # Number of full pages (all tokens in these pages are attended by all Q tokens)
    num_full_pages = first_q_kv_pos // PAGE_SIZE

    # Sliding window: compute the first page within the window for Phase 1 pruning.
    # Each query at position q_pos attends to KV in [q_pos - W + 1, q_pos].
    # The most restrictive query is the first one (q_block_start), so:
    #   first_valid_pos = max(0, first_q_kv_pos - SLIDING_WINDOW + 1)
    if SLIDING_WINDOW > 0:
        first_valid_pos = tl.maximum(0, first_q_kv_pos - SLIDING_WINDOW + 1)
        first_window_page = first_valid_pos // PAGE_SIZE
    else:
        first_window_page = 0

    # Check if this is a full Q block (no q_mask needed)
    is_full_q_block = (q_block_start + Q_BLOCK) <= q_len

    # hoist q_positions_2d before both Phase 1 and Phase 2 loops — it is used in
    # both the SLIDING_WINDOW Phase 1 mask (as q_kv_pos) and the Phase 2 causal mask.
    # Saves one [Q_BLOCK, 1] vector add per Phase 1 iteration when SLIDING_WINDOW > 0.
    q_positions_2d = q_offsets[:, None] + cache_len  # [Q_BLOCK, 1], loop-invariant

    # Phase 1: Full pages — no causal mask, no validity mask
    # Process one page at a time with a clean inner loop
    kv_head_offset = kv_head_id * cache_stride_head
    local_kv = page_offsets[:, None] * cache_stride_token + dhead_offsets[None, :]

    for page_idx in range(first_window_page, num_full_pages):
        physical_page = tl.load(kv_indices_ptr + kv_page_start + page_idx)

        # Use int64 to avoid overflow when physical_page * stride > 2^31
        page_base = physical_page.to(tl.int64) * cache_stride_block + kv_head_offset

        # When sliding window is active, the first window page may partially overlap the
        # window boundary, requiring per-token masking.
        if SLIDING_WINDOW > 0:
            k = tl.load(
                kv_cache_ptr + page_base + local_kv,
                mask=head_dim_mask[None, :],
                other=0.0,
            )
            v = tl.load(
                kv_cache_ptr + page_base + local_kv + cache_stride_kv,
                mask=head_dim_mask[None, :],
                other=0.0,
            )

            qk = tl.dot(q, tl.trans(k)) * SM_SCALE
            # logit soft-capping. Apply BEFORE masking.
            if LOGIT_CAP > 0.0:
                qk = LOGIT_CAP * tl_libdevice.tanh(qk / LOGIT_CAP)

            # Per-query sliding window mask: each query position q_pos
            # can attend to KV in [q_pos - W + 1, q_pos].
            kv_positions = page_idx * PAGE_SIZE + page_offsets[None, :]
            sw_mask = (q_positions_2d - kv_positions) < SLIDING_WINDOW
            full_mask_p1 = q_mask[:, None] & sw_mask
            qk = tl.where(full_mask_p1, qk, float("-inf"))
        elif HEAD_DIM_PADDED == HEAD_DIM:
            # Power-of-2 head_dim: use block_ptr for best memory access pattern
            k_block_ptr = tl.make_block_ptr(
                base=kv_cache_ptr + page_base,
                shape=(PAGE_SIZE, HEAD_DIM),
                strides=(cache_stride_token, 1),
                offsets=(0, 0),
                block_shape=(PAGE_SIZE, HEAD_DIM),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=kv_cache_ptr + page_base + cache_stride_kv,
                shape=(PAGE_SIZE, HEAD_DIM),
                strides=(cache_stride_token, 1),
                offsets=(0, 0),
                block_shape=(PAGE_SIZE, HEAD_DIM),
                order=(1, 0),
            )
            k = tl.load(k_block_ptr).to(q.dtype)  # cast from fp8 if kv cache is fp8
            v = tl.load(v_block_ptr).to(q.dtype)  # cast from fp8 if kv cache is fp8

            qk = tl.dot(q, tl.trans(k)) * SM_SCALE
            if LOGIT_CAP > 0.0:
                qk = LOGIT_CAP * tl_libdevice.tanh(qk / LOGIT_CAP)

            if not is_full_q_block:
                qk = tl.where(q_mask[:, None], qk, float("-inf"))
        else:
            # Non-power-of-2 head_dim: use masked loads with head_dim_mask
            k = tl.load(
                kv_cache_ptr + page_base + local_kv,
                mask=head_dim_mask[None, :],
                other=0.0,
            )
            v = tl.load(
                kv_cache_ptr + page_base + local_kv + cache_stride_kv,
                mask=head_dim_mask[None, :],
                other=0.0,
            )

            qk = tl.dot(q, tl.trans(k)) * SM_SCALE
            if LOGIT_CAP > 0.0:
                qk = LOGIT_CAP * tl_libdevice.tanh(qk / LOGIT_CAP)

            if not is_full_q_block:
                qk = tl.where(q_mask[:, None], qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        if SLIDING_WINDOW > 0:
            # Guard against NaN when m_i == m_i_new == -inf (no valid tokens seen
            # yet for a query whose window doesn't overlap this page at all).
            alpha = tl.where(m_i > float("-inf"), tl.exp(m_i - m_i_new), 0.0)
            p = tl.where(m_i_new[:, None] > float("-inf"), tl.exp(qk - m_i_new[:, None]), 0.0)
        else:
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new[:, None])
        acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # Phase 2: Boundary pages — need causal mask and validity mask.
    # q_positions_2d already computed before Phase 1 loop (iter 19 hoist).
    # tighten upper loop bound to max_q_pos // PAGE_SIZE + 1 so the
    # causal skip (kv_base_pos > max_q_pos) is encoded in the loop bound.
    # This eliminates the per-page `if kv_base_pos <= max_q_pos:` branch and,
    # for early Q blocks in long prefill, skips all pages beyond the Q window.
    phase2_end = tl.minimum(num_kv_pages, max_q_pos // PAGE_SIZE + 1)
    for page_idx in range(num_full_pages, phase2_end):
        kv_base_pos = page_idx * PAGE_SIZE

        physical_page = tl.load(kv_indices_ptr + kv_page_start + page_idx)
        valid_tokens = tl.minimum(PAGE_SIZE, total_kv_len - kv_base_pos)
        page_mask = page_offsets < valid_tokens

        # Use int64 to avoid overflow when physical_page * stride > 2^31
        page_base = physical_page.to(tl.int64) * cache_stride_block + kv_head_offset
        kv_mask_2d = page_mask[:, None] & head_dim_mask[None, :]
        k = tl.load(kv_cache_ptr + page_base + local_kv, mask=kv_mask_2d, other=0.0).to(
            q.dtype
        )  # cast from fp8 if kv cache is fp8
        v = tl.load(
            kv_cache_ptr + page_base + local_kv + cache_stride_kv,
            mask=kv_mask_2d,
            other=0.0,
        ).to(q.dtype)  # cast from fp8 if kv cache is fp8

        qk = tl.dot(q, tl.trans(k)) * SM_SCALE
        # logit soft-capping. Apply BEFORE masking.
        if LOGIT_CAP > 0.0:
            qk = LOGIT_CAP * tl_libdevice.tanh(qk / LOGIT_CAP)
        kv_positions = kv_base_pos + page_offsets[None, :]
        causal_mask = q_positions_2d >= kv_positions
        if SLIDING_WINDOW > 0:
            sliding_mask = (q_positions_2d - kv_positions) < SLIDING_WINDOW
            full_mask = q_mask[:, None] & causal_mask & sliding_mask & page_mask[None, :]
        else:
            full_mask = q_mask[:, None] & causal_mask & page_mask[None, :]
        qk = tl.where(full_mask, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        if SLIDING_WINDOW > 0:
            alpha = tl.where(m_i > float("-inf"), tl.exp(m_i - m_i_new), 0.0)
            p = tl.where(m_i_new[:, None] > float("-inf"), tl.exp(qk - m_i_new[:, None]), 0.0)
        else:
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new[:, None])
        acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_i[:, None]
    o_store_offsets = (
        (q_start + q_offsets[:, None]) * o_stride_token
        + head_id * o_stride_head
        + dhead_offsets[None, :]
    )
    # q_load_mask already includes head_dim_mask, so reuse it for the store
    tl.store(o_ptr + o_store_offsets, o, mask=q_load_mask)


@triton.jit
def _fast_gather_sdpa_kernel(
    kv_cache_ptr,
    kv_indices_ptr,
    out_k_ptr,
    out_v_ptr,
    # Strides
    cache_stride_block: tl.constexpr,
    cache_stride_kv: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_token: tl.constexpr,
    out_stride_seq: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_token: tl.constexpr,
    # Constants
    MAX_PAGES: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    HD_CHUNK1: tl.constexpr,  # First chunk size (0 = single-chunk mode)
    HD_CHUNK2: tl.constexpr,  # Second chunk size
):
    """Gather scattered pages into separate K, V buffers in SDPA layout.

    Grid: (total_pages, N_KV_HEADS)
    Each program copies one page for one KV head into contiguous K and V
    outputs shaped [num_seq, n_kv_heads, max_kv_len, head_dim].

    When HD_CHUNK1 > 0 (non-power-of-2 head_dim), uses two-chunk loading to
    reduce 25% of bandwidth vs HEAD_DIM_PADDED=256 approach.
    """
    page_global_idx = tl.program_id(0)
    kv_head_id = tl.program_id(1)

    seq_id = page_global_idx // MAX_PAGES
    local_page = page_global_idx % MAX_PAGES

    physical_page = tl.load(kv_indices_ptr + page_global_idx)
    token_offsets = tl.arange(0, PAGE_SIZE)

    src_base = physical_page.to(tl.int64) * cache_stride_block + kv_head_id * cache_stride_head
    local_token_start = local_page * PAGE_SIZE
    dst_token_base = (
        seq_id * out_stride_seq
        + kv_head_id * out_stride_head
        + (local_token_start + token_offsets[:, None]) * out_stride_token
    )

    if HD_CHUNK1 == 0:
        # Single-chunk path (power-of-2 head_dim)
        head_offsets = tl.arange(0, HEAD_DIM_PADDED)
        head_dim_mask = head_offsets < HEAD_DIM
        src_offsets = token_offsets[:, None] * cache_stride_token + head_offsets[None, :]
        k_data = tl.load(
            kv_cache_ptr + src_base + src_offsets, mask=head_dim_mask[None, :], other=0.0
        )
        v_data = tl.load(
            kv_cache_ptr + src_base + cache_stride_kv + src_offsets,
            mask=head_dim_mask[None, :],
            other=0.0,
        )
        dst_base = dst_token_base + head_offsets[None, :]
        tl.store(out_k_ptr + dst_base, k_data, mask=head_dim_mask[None, :])
        tl.store(out_v_ptr + dst_base, v_data, mask=head_dim_mask[None, :])
    else:
        # Two-chunk path (non-power-of-2 head_dim): reduce bandwidth by ~25%
        dhead_c1 = tl.arange(0, HD_CHUNK1)  # [0, HD_CHUNK1) — all valid
        dhead_c2 = HD_CHUNK1 + tl.arange(0, HD_CHUNK2)  # [HD_CHUNK1, HD_CHUNK1+HD_CHUNK2)
        head_dim_mask_c2 = dhead_c2 < HEAD_DIM

        src_c1 = token_offsets[:, None] * cache_stride_token + dhead_c1[None, :]
        src_c2 = token_offsets[:, None] * cache_stride_token + dhead_c2[None, :]

        k_c1 = tl.load(kv_cache_ptr + src_base + src_c1)
        k_c2 = tl.load(
            kv_cache_ptr + src_base + src_c2,
            mask=head_dim_mask_c2[None, :],
            other=0.0,
        )
        v_c1 = tl.load(kv_cache_ptr + src_base + cache_stride_kv + src_c1)
        v_c2 = tl.load(
            kv_cache_ptr + src_base + cache_stride_kv + src_c2,
            mask=head_dim_mask_c2[None, :],
            other=0.0,
        )

        dst_c1 = dst_token_base + dhead_c1[None, :]
        dst_c2 = dst_token_base + dhead_c2[None, :]
        tl.store(out_k_ptr + dst_c1, k_c1)
        tl.store(out_k_ptr + dst_c2, k_c2, mask=head_dim_mask_c2[None, :])
        tl.store(out_v_ptr + dst_c1, v_c1)
        tl.store(out_v_ptr + dst_c2, v_c2, mask=head_dim_mask_c2[None, :])


def triton_paged_context(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    sm_scale: float,
    sliding_window: Optional[int] = None,
    logit_cap: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Context/prefill attention with paged KV cache."""
    total_tokens, n_heads, head_dim = q.shape
    _, _, n_kv_heads, page_size, _ = kv_cache.shape
    num_seq = qo_indptr.shape[0] - 1

    output = out if out is not None else torch.empty_like(q)

    if num_seq == 0 or total_tokens == 0:
        return output

    # Compute max_q_len without GPU sync for single-sequence batches (most common
    # in serving). For multi-sequence batches, we must use .item() because
    # total_tokens // num_seq gives the average, not the max — variable-length
    # sequences can produce wrong results (under-launched Q blocks or wrong SDPA reshape).
    if num_seq == 1:
        max_q_len = total_tokens
    else:
        q_lens = qo_indptr[1:] - qo_indptr[:-1]
        max_q_len = int(q_lens.max().item())

    # Adaptive dispatch: gather + cuDNN SDPA for seq>=512 (outperforms paged kernel),
    # paged Triton kernel for shorter sequences where gather overhead dominates.
    # Compute max_pages from max_q_len without GPU sync
    # (assumes pure prefill where q_len == kv_len for each seq)
    # Normalize sliding_window for kernel constexpr: None/non-positive → 0
    sw = sliding_window if isinstance(sliding_window, int) and sliding_window > 0 else 0
    # Normalize logit_cap: None/non-positive → 0.0 (no capping)
    lc = float(logit_cap) if logit_cap is not None and logit_cap > 0.0 else 0.0

    max_pages = (max_q_len + page_size - 1) // page_size
    total_expected_pages = num_seq * max_pages
    use_sdpa = (
        max_q_len >= 512
        and num_seq <= 64
        and max_pages > 0
        and kv_indices.shape[0] == total_expected_pages  # all seqs same page count
        and sw == 0  # SDPA doesn't support sliding window natively
        and lc == 0.0  # cuDNN SDPA doesn't support logit soft-capping
    )

    if use_sdpa:
        # Fast Triton gather: scattered pages → separate K, V in SDPA layout
        # Single alloc for both K and V, single kernel to fill
        max_kv_len = max_pages * page_size
        kv_buf = torch.empty(
            2,
            num_seq,
            n_kv_heads,
            max_kv_len,
            head_dim,
            dtype=kv_cache.dtype,
            device=kv_cache.device,
        )
        k_sdpa = kv_buf[0]
        v_sdpa = kv_buf[1]
        # Cast k/v to query dtype if kv cache uses a different dtype (e.g., fp8)
        if kv_cache.dtype != q.dtype:
            k_sdpa = k_sdpa.to(q.dtype)
            v_sdpa = v_sdpa.to(q.dtype)
        head_dim_padded = triton.next_power_of_2(head_dim)
        if head_dim_padded == head_dim:
            hd_gather_chunk1, hd_gather_chunk2 = 0, 0  # single-chunk
        else:
            hd_gather_chunk1 = head_dim_padded // 2
            hd_gather_chunk2 = triton.next_power_of_2(head_dim - hd_gather_chunk1)
        _fast_gather_sdpa_kernel[(total_expected_pages, n_kv_heads)](
            kv_cache,
            kv_indices,
            k_sdpa,
            v_sdpa,
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            k_sdpa.stride(0),
            k_sdpa.stride(1),
            k_sdpa.stride(2),
            MAX_PAGES=max_pages,
            N_KV_HEADS=n_kv_heads,
            PAGE_SIZE=page_size,
            HEAD_DIM=head_dim,
            HEAD_DIM_PADDED=head_dim_padded,
            HD_CHUNK1=hd_gather_chunk1,
            HD_CHUNK2=hd_gather_chunk2,
        )

        # SDPA with GQA
        o_sdpa = torch.nn.functional.scaled_dot_product_attention(
            q.view(num_seq, max_q_len, n_heads, head_dim).transpose(1, 2),
            k_sdpa,
            v_sdpa,
            scale=sm_scale,
            is_causal=True,
            enable_gqa=True,
        )
        output.view(num_seq, max_q_len, n_heads, head_dim).copy_(o_sdpa.permute(0, 2, 1, 3))
    else:
        # Use paged kernel (better for small workloads)
        def grid_paged(meta):
            q_block = meta["Q_BLOCK"]
            num_q_blocks = (max_q_len + q_block - 1) // q_block
            return (num_seq, n_heads, num_q_blocks)

        _paged_context_kernel[grid_paged](
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
            HEAD_DIM_PADDED=triton.next_power_of_2(head_dim),
            PAGE_SIZE=page_size,
            SLIDING_WINDOW=sw,
            LOGIT_CAP=lc,
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
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
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


@torch.library.custom_op("auto_deploy::triton_paged_mha_with_cache", mutates_args=("kv_cache",))
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
    sliding_window: Optional[int] = None,
    logit_cap: Optional[float] = None,
    # OPTIONAL PRE-ALLOCATED OUTPUT
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Triton paged attention with mixed batch support."""
    head_dim = kv_cache.shape[-1]
    q_shape_og = q.shape
    b, s = q_shape_og[:2]

    q = q.reshape(b * s, -1, head_dim).contiguous()
    k = k.reshape(b * s, -1, head_dim).contiguous()
    v = v.reshape(b * s, -1, head_dim).contiguous()

    from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
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

    if out is not None:
        y = out.view(-1, q.shape[1], head_dim)
    else:
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
            sliding_window=sliding_window,
            logit_cap=logit_cap,
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
            sliding_window=sliding_window,
            logit_cap=logit_cap,
            out=y[num_prefill_tokens:num_total_tokens],
        )

    if out is not None:
        # Zero stale data in padding region for CUDA graph replay stability
        bs = b * s
        if num_total_tokens < bs:
            y[num_total_tokens:].zero_()
        # Return a 0-element dummy to satisfy PyTorch's no-alias constraint.
        # The caller (DynamicOpWrapper._coalesce_output) picks ``out`` over
        # this dummy, so the pre-allocated buffer is used downstream.
        return out.new_empty(0)

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
    sliding_window: Optional[int] = None,
    logit_cap: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is not None:
        return out.new_empty(0)
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
        (layout,) = extract_op_args(source_attn_node, "layout")
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

        sliding_window = extract_op_args(source_attn_node, "sliding_window")[0]
        logit_cap = extract_op_args(source_attn_node, "logit_cap")[0]

        return [scale, sliding_window, logit_cap]
