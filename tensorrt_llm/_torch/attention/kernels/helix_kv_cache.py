# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _append_helix_kv(
    key,
    value,
    key_cache,
    value_cache,
    batch_indices,
    positions,
    page_indices,
    page_indptr,
    KEY_TOKEN_STRIDE: tl.constexpr,
    KEY_HEAD_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    VALUE_TOKEN_STRIDE: tl.constexpr,
    VALUE_HEAD_STRIDE: tl.constexpr,
    VALUE_DIM_STRIDE: tl.constexpr,
    CACHE_PAGE_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    CACHE_HEAD_STRIDE: tl.constexpr,
    CACHE_DIM_STRIDE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_index = tl.program_id(0)
    position = tl.load(positions + token_index)
    if position >= 0:
        request_index = tl.load(batch_indices + token_index)
        page_start = tl.load(page_indptr + request_index)
        page_index = tl.load(page_indices + page_start + position // PAGE_SIZE)
        offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        heads = offsets // HEAD_DIM
        dimensions = offsets % HEAD_DIM
        valid = offsets < NUM_HEADS * HEAD_DIM
        key_values = tl.load(
            key
            + token_index * KEY_TOKEN_STRIDE
            + heads * KEY_HEAD_STRIDE
            + dimensions * KEY_DIM_STRIDE,
            valid,
            other=0,
        )
        value_values = tl.load(
            value
            + token_index * VALUE_TOKEN_STRIDE
            + heads * VALUE_HEAD_STRIDE
            + dimensions * VALUE_DIM_STRIDE,
            valid,
            other=0,
        )
        cache_offsets = (
            page_index.to(tl.int64) * CACHE_PAGE_STRIDE
            + (position % PAGE_SIZE) * CACHE_TOKEN_STRIDE
            + heads * CACHE_HEAD_STRIDE
            + dimensions * CACHE_DIM_STRIDE
        )
        tl.store(key_cache + cache_offsets, key_values, valid)
        tl.store(value_cache + cache_offsets, value_values, valid)


def append_helix_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    page_indices: torch.Tensor,
    page_indptr: torch.Tensor,
    kv_layout: str,
) -> None:
    """Append rank-owned KV rows; negative positions leave the cache untouched."""
    key_cache, value_cache = paged_kv_cache.unbind(1)
    if kv_layout == "HND":
        page_stride, head_stride, token_stride, dim_stride = key_cache.stride()
        page_size = key_cache.shape[2]
    else:
        page_stride, token_stride, head_stride, dim_stride = key_cache.stride()
        page_size = key_cache.shape[1]
    num_heads, head_dim = key.shape[1:]
    block_size = 256
    _append_helix_kv[(key.shape[0], triton.cdiv(num_heads * head_dim, block_size))](
        key,
        value,
        key_cache,
        value_cache,
        batch_indices,
        positions,
        page_indices,
        page_indptr,
        *key.stride(),
        *value.stride(),
        page_stride,
        token_stride,
        head_stride,
        dim_stride,
        num_heads,
        head_dim,
        page_size,
        block_size,
    )
