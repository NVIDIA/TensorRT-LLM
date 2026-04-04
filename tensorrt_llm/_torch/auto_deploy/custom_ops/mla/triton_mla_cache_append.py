# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Triton kernels for MLA paged cache operations.

Provides a CUDA-graph-compatible cache append for combined MLA paged caches
with layout [num_pages, page_size, kv_lora_rank + qk_rope_head_dim].

The kernel computes per-token physical page destinations from decomposed
paging metadata (cu_seqlen, cu_num_pages, cache_loc, input_pos) entirely
on GPU, avoiding host-device synchronization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _find_seq_idx(token_idx, cu_seqlen_ptr, num_seq: tl.constexpr):
    """Binary search for the sequence that owns token_idx."""
    lo = 0
    hi = num_seq - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tl.load(cu_seqlen_ptr + mid) <= token_idx:
            lo = mid
        else:
            hi = mid - 1
    return lo


@triton.jit
def _append_paged_mla_cache_kernel(
    kv_cache_ptr,
    compressed_kv_ptr,
    kpe_ptr,
    cu_seqlen_ptr,
    cu_num_pages_ptr,
    cache_loc_ptr,
    input_pos_ptr,
    num_tokens,
    cache_stride_page,  # kv_cache.stride(0) = page_size * total_dim
    cache_stride_slot,  # kv_cache.stride(1) = total_dim
    ckv_stride_token,  # compressed_kv.stride(0)
    kpe_stride_token,  # kpe.stride(0)
    page_size,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    num_seq: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return

    # Find which sequence this token belongs to (binary search over cu_seqlen)
    seq_idx = _find_seq_idx(token_idx, cu_seqlen_ptr, num_seq)

    # Compute offset within sequence and absolute cache position
    seq_start = tl.load(cu_seqlen_ptr + seq_idx)
    token_offset = token_idx - seq_start
    pos0 = tl.load(input_pos_ptr + seq_idx)
    abs_pos = pos0 + token_offset

    # Map to physical page and offset within page
    page_in_seq = abs_pos // page_size
    offset_in_page = abs_pos % page_size
    flat_page_idx = tl.load(cu_num_pages_ptr + seq_idx) + page_in_seq
    phys_page = tl.load(cache_loc_ptr + flat_page_idx)

    # Destination pointer in kv_cache
    dst_base = kv_cache_ptr + phys_page * cache_stride_page + offset_in_page * cache_stride_slot

    ckv_base = compressed_kv_ptr + token_idx * ckv_stride_token
    d_offsets = tl.arange(0, BLOCK_D)

    # Write kv_lora_rank elements
    ckv_mask = d_offsets < kv_lora_rank
    ckv_vals = tl.load(ckv_base + d_offsets, mask=ckv_mask, other=0.0)
    tl.store(dst_base + d_offsets, ckv_vals, mask=ckv_mask)

    # Write qk_rope_head_dim elements
    kpe_base = kpe_ptr + token_idx * kpe_stride_token
    kpe_offsets = tl.arange(0, BLOCK_D)
    kpe_mask = kpe_offsets < qk_rope_head_dim
    kpe_vals = tl.load(kpe_base + kpe_offsets, mask=kpe_mask, other=0.0)
    tl.store(dst_base + kv_lora_rank + kpe_offsets, kpe_vals, mask=kpe_mask)


def append_paged_mla_cache(
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    cu_seqlen: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    input_pos: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_lora_rank: int,
    num_tokens: int,
    num_seq: int,
) -> None:
    """Append tokens to a combined MLA paged cache using a Triton kernel.

    All inputs must be device tensors. No host-device synchronization is
    performed, making this compatible with CUDA graph capture.

    Args:
        compressed_kv: [total_tokens, kv_lora_rank] compressed KV to append.
        kpe: [total_tokens, qk_rope_head_dim] positional key embeddings.
        cu_seqlen: [num_seq + 1] cumulative token counts (device, int32).
        cu_num_pages: [num_seq + 1] cumulative page counts (device, int32).
        cache_loc: [total_pages] physical page indices (device, int32).
        input_pos: [num_seq] starting cache position per sequence (device, int32).
        kv_cache: [num_pages, page_size, kv_lora_rank + qk_rope_head_dim] cache to write into.
        kv_lora_rank: dimension of compressed KV.
        num_tokens: number of tokens to append.
        num_seq: number of sequences in the batch.
    """
    if num_tokens == 0:
        return

    qk_rope_head_dim = kpe.shape[-1]
    page_size = kv_cache.shape[1]

    # BLOCK_D must be a power of 2 >= max(kv_lora_rank, qk_rope_head_dim)
    BLOCK_D = triton.next_power_of_2(max(kv_lora_rank, qk_rope_head_dim))

    grid = (num_tokens,)
    _append_paged_mla_cache_kernel[grid](
        kv_cache,
        compressed_kv,
        kpe,
        cu_seqlen,
        cu_num_pages,
        cache_loc,
        input_pos,
        num_tokens,
        kv_cache.stride(0),
        kv_cache.stride(1),
        compressed_kv.stride(0),
        kpe.stride(0),
        page_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        num_seq=num_seq,
        BLOCK_D=BLOCK_D,
    )
