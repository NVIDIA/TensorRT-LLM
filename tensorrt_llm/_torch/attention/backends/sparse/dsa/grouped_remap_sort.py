# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in physical index ordering for shared-index generation groups.

The producer reads the current indices/page table on every launch, sorts the
shared physical base once, and applies each member's dynamic layer offset.
Invalid slots and duplicate selections are retained. Reordering affects
floating-point reduction order in attention and is not bitwise equivalent.
"""

import os
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from triton.language.core import TRITON_MAX_TENSOR_NUMEL

_PRESERVE_SELECTION_ORDER = any(
    os.environ.get(name, "0") == "1"
    for name in ("FORCE_DETERMINISTIC", "FORCE_ATTENTION_KERNEL_DETERMINISTIC")
)

if TYPE_CHECKING:
    from .metadata import DSAtrtllmAttentionMetadata


@triton.jit
def _grouped_remap_sort_kernel(
    requests,
    pages,
    indices,
    layers,
    output,
    ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    SORT_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    PAGE_SCALE: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, SORT_SIZE)
    request = tl.load(requests + row).to(tl.int64)
    token = tl.load(indices + row * TOPK + cols, cols < TOPK, -1)
    block = token // TOKENS_PER_BLOCK
    valid = ((token >= 0) & (block < MAX_BLOCKS)
             & (request >= 0) & (request < NUM_REQUESTS))
    page = tl.load(pages + request * MAX_BLOCKS + block, valid, -1)
    physical = (page.to(tl.int64) * (PAGE_SCALE * TOKENS_PER_BLOCK)
                + token % TOKENS_PER_BLOCK)
    base = tl.where(valid & (page >= 0) & (physical < POOL_TOKENS), physical, -1).to(tl.int32)
    ordered = tl.sort(tl.where(base >= 0, base, 2147483647), descending=False)
    for group in tl.static_range(GROUP_SIZE):
        layer = tl.load(layers + group)
        physical = ordered.to(tl.int64) + layer.to(tl.int64) * TOKENS_PER_BLOCK
        valid = ((ordered != 2147483647) & (layer >= 0) & (layer < PAGE_SCALE)
                 & (physical < POOL_TOKENS))
        value = tl.where(valid, physical, -1).to(tl.int32)
        group_offset = tl.full((), group, tl.int64) * ROWS * TOPK
        tl.store(output + group_offset + row * TOPK + cols, value, cols < TOPK)


def try_grouped_remap_sort(
    topk_indices: torch.Tensor,
    metadata: "DSAtrtllmAttentionMetadata",
    layer_ids: torch.Tensor,
    page_index_scale: int,
) -> torch.Tensor | None:
    """Return sorted int32 [group_size, rows, topk] indices, or None.

    This generation-only helper is called after the existing backend grouping
    checks. The manager remains responsible for valid request IDs and layer
    offsets, just as for the native grouped operator. Neither query-head count
    nor KV dtype affects this integer permutation. Arbitrary positive TopK is
    padded to a power of two for sorting; padding is never emitted.
    All checks use tensor metadata; no device values are copied to the host.
    """
    # Physical page placement can change between runs; deterministic attention
    # must retain the indexer's logical-token order instead.
    if (
        _PRESERVE_SELECTION_ORDER
        or getattr(metadata.sparse_metadata_params, "gvr_deterministic_topk", False)
        or metadata.in_mtp_draft_loop
        or topk_indices.ndim != 2
        or not 0 < topk_indices.shape[0] <= 2**31 - 1
        or topk_indices.shape[1] <= 0
        or triton.next_power_of_2(topk_indices.shape[1]) > TRITON_MAX_TENSOR_NUMEL
        or layer_ids.ndim != 1
        or layer_ids.numel() < 2
        or page_index_scale <= 0
    ):
        return None
    metadata._ensure_pool_view_cached()
    requests = metadata._cached_req_idx_gen
    pages = metadata._cached_block_table_gen
    tokens_per_block = metadata._cached_tokens_per_block
    pool_tokens = metadata._cached_num_pool_tokens
    if (
        tokens_per_block <= 0
        or not 0 < pool_tokens <= 2**31 - 1
        or page_index_scale * tokens_per_block > pool_tokens
        or requests is None
        or pages is None
        or requests.ndim != 1
        or requests.numel() != topk_indices.shape[0]
        or pages.ndim != 2
        or pages.shape[0] == 0
        or pages.shape[1] == 0
        or any(
            tensor.dtype != torch.int32
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.device != topk_indices.device
            for tensor in (requests, pages, topk_indices, layer_ids)
        )
    ):
        return None
    rows, topk = topk_indices.shape
    group_size = layer_ids.numel()
    output = torch.empty((group_size, rows, topk), device=topk_indices.device, dtype=torch.int32)
    _grouped_remap_sort_kernel[(topk_indices.shape[0],)](
        requests, pages, topk_indices, layer_ids, output,
        rows, topk, triton.next_power_of_2(topk), group_size,
        pages.shape[0], pages.shape[1], tokens_per_block, page_index_scale,
        pool_tokens, num_warps=8,
    )
    return output
