# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

########################################################
# Index gather kernel
########################################################


@triton.jit
def _convert_req_index_to_global_index_kernel_with_stride_factor(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_kv_heads, num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_kv_heads, num_tokens, kv_factor, NUM_TOPK_TOKENS]
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    stride_factor: tl.constexpr,  # elements per physical page in pool
    layer_id: tl.constexpr,  # for multi-layer KV cache
    num_kv_heads: tl.constexpr,
    kv_factor: tl.constexpr,
    bt_stride0,  # stride for batch dim
    bt_stride1,  # stride for page dim
    ti_stride0,
    ti_stride1,
    ti_stride2,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
):
    """
    Convert request-local token indices to global KV cache pool indices.

    KV cache pool layout: [max_num_pages, num_layers, kv_factor, num_kv_heads, tokens_per_block, head_dim]
    block_table: [num_requests, max_pages_per_req]
      - stores memPoolBlockIdx (physical page index, first dim of pool)
      - same block_table for K and V (K/V share the same physical page)

    stride_factor = num_layers * kv_factor * num_kv_heads * BLOCK_SIZE
      (elements per physical page, excluding head_dim)

    Global index:
      memPoolBlockIdx * stride_factor
        + layer_id * kv_factor * num_kv_heads * BLOCK_SIZE
        + kv_factor_idx * num_kv_heads * BLOCK_SIZE
        + kv_head_idx * BLOCK_SIZE
        + token_in_page
    """
    kv_head_idx = tl.program_id(0)
    token_id = tl.program_id(1)
    tile_id = tl.program_id(2)

    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req = tl.load(req_id_ptr + token_id)

    # Load local token indices: token_indices[kv_head_idx, token_id, indice_id]
    ti_ptr = (token_indices_ptr + kv_head_idx * ti_stride0 +
              token_id * ti_stride1 + indice_id * ti_stride2)
    tok = tl.load(ti_ptr)

    is_invalid_tok = tok < 0
    page_idx = tok // BLOCK_SIZE
    token_in_page = tok % BLOCK_SIZE
    valid_page = page_idx < max_num_blocks_per_req

    # block_table[req, page_idx] → memPoolBlockIdx (physical page index)
    bt_ptr = (block_table_ptr + req * bt_stride0 + page_idx * bt_stride1)
    mem_pool_idx = tl.load(bt_ptr, mask=valid_page, other=0)

    # Base offset within physical page (invariant across kv_factor loop)
    base_off = (layer_id * kv_factor * num_kv_heads * BLOCK_SIZE +
                kv_head_idx * BLOCK_SIZE + token_in_page)

    for kv_factor_idx in tl.static_range(kv_factor):
        # Within-page offset: layer → kv_factor → kv_head → token
        inpage_off = base_off + kv_factor_idx * num_kv_heads * BLOCK_SIZE

        global_idx = mem_pool_idx * stride_factor + inpage_off

        out_val = tl.where(is_invalid_tok | (~valid_page), -1, global_idx)

        out_ptr_ij = (out_ptr + kv_head_idx * out_stride0 +
                      token_id * out_stride1 + kv_factor_idx * out_stride2 +
                      indice_id * out_stride3)
        tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.
    Tensor,  # int32 [num_requests, kv_factor, max_num_blocks_per_req]
    token_indices: torch.
    Tensor,  # int32 [num_kv_heads, num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
    stride_factor: int = None,  # elements per block in pool
    layer_id: int = 0,  # for multi-layer KV cache
    num_kv_heads: int = 1,
    kv_factor: int = 1,
):
    """
    Convert request-local token indices to global KV cache pool indices.

    Accepts both 2D and 3D token_indices:
      - 2D [num_tokens, topk]: MLA path (num_kv_heads=1 implicit)
      - 3D [num_kv_heads, num_tokens, topk]: MQA/GQA path

    Output shape: [num_kv_heads, num_tokens, kv_factor, topk]

    Args:
        block_table: [num_requests, max_pages] — physical page indices.
        stride_factor: Elements per physical page
                       (num_layers * kv_factor * num_kv_heads * BLOCK_SIZE).
        num_kv_heads: Number of KV heads.
        kv_factor: 2 for MQA/GQA, 1 for MLA.
    """
    if stride_factor is None:
        stride_factor = kv_factor * num_kv_heads * BLOCK_SIZE
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.ndim in (2, 3), \
        f"Expected 2D [num_tokens, topk] or 3D [num_kv_heads, num_tokens, topk], got {token_indices.ndim}D"
    assert block_table.ndim == 2, f"Expected 2D [batch, max_pages], got {block_table.ndim}D"
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, \
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty((num_kv_heads, num_tokens, kv_factor, NUM_TOPK_TOKENS),
                      dtype=torch.int32,
                      device=token_indices.device)

    bt_stride0, bt_stride1 = block_table_c.stride()
    # 2D input: ti_stride0=0 (kv_head_idx is always 0, term vanishes)
    if token_indices_c.ndim == 2:
        ti_stride0 = 0
        ti_stride1, ti_stride2 = token_indices_c.stride()
    else:
        ti_stride0, ti_stride1, ti_stride2 = token_indices_c.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

    grid = (num_kv_heads, num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel_with_stride_factor[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        stride_factor,
        layer_id,
        num_kv_heads,
        kv_factor,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        ti_stride2,
        out_stride0,
        out_stride1,
        out_stride2,
        out_stride3,
    )
    return out


@triton.jit
def _triton_gather_k_cache_kernel(
    k_cache_ptr,
    slot_fp8_ptr,
    slot_scale_ptr,
    out_fp8_ptr,
    out_scale_ptr,
    k_token_start,
    num_k_tokens,
    HEAD_DIM: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offsets = (pid * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)).to(
        tl.int64)
    token_mask = token_offsets < num_k_tokens

    fp8_base = tl.load(slot_fp8_ptr + k_token_start + token_offsets,
                       mask=token_mask,
                       other=0)
    scale_base = tl.load(slot_scale_ptr + k_token_start + token_offsets,
                         mask=token_mask,
                         other=0)

    byte_offsets = tl.arange(0, HEAD_DIM).to(tl.int64)
    src_fp8 = fp8_base[:, None] + byte_offsets[None, :]
    dst_fp8 = token_offsets[:, None] * HEAD_DIM + byte_offsets[None, :]
    gather_mask = token_mask[:, None]

    fp8_data = tl.load(k_cache_ptr + src_fp8, mask=gather_mask, other=0)
    tl.store(out_fp8_ptr + dst_fp8, fp8_data, mask=gather_mask)

    scale_byte_offsets = tl.arange(0, SCALE_BYTES).to(tl.int64)
    src_scale = scale_base[:, None] + scale_byte_offsets[None, :]
    dst_scale = token_offsets[:,
                              None] * SCALE_BYTES + scale_byte_offsets[None, :]

    scale_data = tl.load(k_cache_ptr + src_scale, mask=gather_mask, other=0)
    tl.store(out_scale_ptr + dst_scale, scale_data, mask=gather_mask)


def triton_gather_k_cache(
    k_cache: torch.Tensor,
    slot_mapping_fp8: torch.Tensor,
    slot_mapping_scale: torch.Tensor,
    k_token_start: int,
    k_token_end: int,
    head_dim: int,
):
    """Gather K FP8 values and scales from the indexer K cache for a chunk.

    Replaces ``_gather_k_cache_for_chunk``, fusing ~8-12 small PyTorch ops
    (arange, unsqueeze, broadcast add, _unravel_indices, advanced indexing)
    into a single Triton kernel that directly gathers from flat byte offsets.
    This is purely data movement — bit-exact with the original.

    Args:
        k_cache: Indexer K cache pool data (2D contiguous), uint8.
        slot_mapping_fp8: Flat byte indices for FP8 data
            ``[total_kv_len]``, int64.
        slot_mapping_scale: Flat byte indices for scale data
            ``[total_kv_len]``, int64.
        k_token_start: Start index into slot mapping arrays.
        k_token_end: End index into slot mapping arrays.
        head_dim: FP8 head dimension (typically 128).

    Returns:
        Tuple of (k_fp8, k_scale):
            k_fp8: ``[num_k_tokens, head_dim]``, float8_e4m3fn.
            k_scale: ``[num_k_tokens, 1]``, float32.
    """
    num_k_tokens = k_token_end - k_token_start
    device = k_cache.device

    if num_k_tokens == 0:
        return (
            torch.empty(0, head_dim, dtype=torch.float8_e4m3fn, device=device),
            torch.empty(0, 1, dtype=torch.float32, device=device),
        )

    SCALE_BYTES = 4
    BLOCK_TOKENS = 32

    k_cache_flat = k_cache.reshape(-1)
    out_fp8 = torch.empty(num_k_tokens,
                          head_dim,
                          dtype=torch.uint8,
                          device=device)
    out_scale = torch.empty(num_k_tokens,
                            SCALE_BYTES,
                            dtype=torch.uint8,
                            device=device)

    grid = (triton.cdiv(num_k_tokens, BLOCK_TOKENS), )
    _triton_gather_k_cache_kernel[grid](
        k_cache_flat,
        slot_mapping_fp8,
        slot_mapping_scale,
        out_fp8.view(-1),
        out_scale.view(-1),
        k_token_start,
        num_k_tokens,
        HEAD_DIM=head_dim,
        SCALE_BYTES=SCALE_BYTES,
        BLOCK_TOKENS=BLOCK_TOKENS,
    )

    k_fp8 = out_fp8.view(torch.float8_e4m3fn)
    k_scale = out_scale.view(torch.float32).view(num_k_tokens, 1)
    return k_fp8, k_scale
