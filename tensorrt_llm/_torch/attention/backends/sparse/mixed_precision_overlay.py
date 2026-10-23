# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained-token rows for sparse MLA backends whose history lives in a native NVFP4 pool.

Such a backend already dequantizes the rows its top-k selected into an FP8 scratch
before attention. Mixed precision adds the standard manager's retained-token twin
layer next to that pool: the KV append also stores the token's row in the twin
(first_n / last_n tokens survive there, the rest is released by the twin's window),
and after the gather the rows the twin still holds replace the dequantized ones.
The twin's page table comes from the manager's existing builder; page >= 0 means
retained.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _write_rows_kernel(
    rows_ptr,
    req_ptr,
    pos_ptr,
    table_ptr,
    table_stride,
    pool_ptr,
    rows_per_page,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    req = tl.load(req_ptr + token)
    pos = tl.load(pos_ptr + token)
    # pos < 0 marks a padding row the caller does not want stored.
    page = tl.load(table_ptr + req * table_stride + pos // rows_per_page, mask=pos >= 0, other=-1)
    if page >= 0:
        offs = tl.arange(0, BLOCK)
        mask = offs < ROW
        row = tl.load(rows_ptr + token.to(tl.int64) * ROW + offs, mask=mask, other=0.0)
        slot = page.to(tl.int64) * rows_per_page + pos % rows_per_page
        tl.store(pool_ptr + slot * ROW + offs, row.to(pool_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _overlay_rows_kernel(
    topk_ptr,
    topk_stride,
    compact_ptr,
    compact_stride,
    req_ptr,
    table_ptr,
    table_stride,
    pool_ptr,
    scratch_ptr,
    rows_per_page,
    K,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    row_id = pair // K
    col = pair % K
    tok = tl.load(topk_ptr + row_id.to(tl.int64) * topk_stride + col)
    compact = tl.load(compact_ptr + row_id.to(tl.int64) * compact_stride + col)
    if (tok >= 0) & (compact >= 0):
        req = tl.load(req_ptr + row_id)
        page = tl.load(table_ptr + req * table_stride + tok // rows_per_page)
        if page >= 0:
            offs = tl.arange(0, BLOCK)
            mask = offs < ROW
            slot = page.to(tl.int64) * rows_per_page + tok % rows_per_page
            row = tl.load(pool_ptr + slot * ROW + offs, mask=mask, other=0.0)
            tl.store(
                scratch_ptr + compact.to(tl.int64) * ROW + offs,
                row.to(scratch_ptr.dtype.element_ty),
                mask=mask,
            )


def _retained_pool(metadata, layer_idx: int) -> Optional[torch.Tensor]:
    """``[pages, tokens_per_block, row]`` view of the layer's retained pool, else None."""
    table = getattr(metadata, "mixed_high_block_table", None)
    manager = metadata.kv_cache_manager
    high_layers = getattr(manager, "high_layers", None)
    if table is None or not high_layers:
        return None
    local = manager.layer_offsets.get(layer_idx)
    if local not in high_layers:
        return None
    # [pages, kv_factor, rows_per_page, heads, width] with one head and no value half.
    pool = manager._get_mixed_precision_buffers(local, "NHD", "high")
    return pool.view(pool.shape[0], pool.shape[2], -1)


def _block(width: int) -> int:
    return max(16, triton.next_power_of_2(width))


def write_rows(
    metadata, layer_idx: int, rows: torch.Tensor, req_idx: torch.Tensor, positions: torch.Tensor
) -> None:
    """Store ``rows`` (one per token, RoPE already applied) in the retained pool."""
    pool = _retained_pool(metadata, layer_idx)
    if pool is None:
        return
    rows = rows.reshape(rows.shape[0], -1)
    if rows.shape[0] == 0:
        return
    if not rows.is_contiguous():
        rows = rows.contiguous()
    width = pool.shape[-1]
    assert rows.shape[1] == width, (rows.shape, width)
    table = metadata.mixed_high_block_table
    _write_rows_kernel[(rows.shape[0],)](
        rows,
        req_idx,
        positions,
        table,
        table.stride(0),
        pool,
        pool.shape[1],
        ROW=width,
        BLOCK=_block(width),
    )


def write_context_rows(attention, metadata, latent_cache: torch.Tensor) -> None:
    """Context phase: the append kernel wrote the rotated k_pe back into ``latent_cache``."""
    positions = getattr(metadata, "mixed_ctx_positions", None)
    if positions is None:
        return
    num_ctx = metadata.num_ctx_tokens
    write_rows(
        metadata,
        attention.layer_idx,
        latent_cache[:num_ctx],
        metadata.req_idx_per_token[:num_ctx],
        positions,
    )


def write_generation_rows(
    attention, metadata, latent_cache: torch.Tensor, position_ids: torch.Tensor
) -> None:
    """Generation phase: the fused kernel leaves ``latent_cache`` unrotated, so rotate k_pe here.

    Same interleaved-pair rotation and cos/sin table as the append kernel.
    """
    if _retained_pool(metadata, attention.layer_idx) is None:
        return
    rows = latent_cache.reshape(latent_cache.shape[0], -1)
    num_rows = rows.shape[0]
    if num_rows == 0:
        return
    assert position_ids is not None, "the retained-row write needs the generation positions"
    # Callers pass either the whole batch's positions or only the generation
    # slice; the generation tokens are the trailing ones either way.
    positions = position_ids.reshape(-1)[-num_rows:]
    num_ctx = metadata.num_ctx_tokens
    rope_dim = attention.mla_params.qk_rope_head_dim
    rotated = rows.float()
    k = rotated[:, -rope_dim:].view(num_rows, rope_dim // 2, 2)
    table = attention.rotary_cos_sin.view(-1, rope_dim, 2)[positions.long(), : rope_dim // 2]
    cos, sin = table[..., 0], table[..., 1]
    x, y = k[..., 0].clone(), k[..., 1].clone()
    k[..., 0] = x * cos - y * sin
    k[..., 1] = y * cos + x * sin
    write_rows(
        metadata,
        attention.layer_idx,
        # The append kernel rounds the rotated values to the activation dtype
        # before quantizing; do the same so the rows match a plain FP8 pool.
        rotated.to(rows.dtype),
        metadata.req_idx_per_token[num_ctx : num_ctx + num_rows],
        positions,
    )


def write_compressed_rows(
    compressor,
    metadata,
    rows: torch.Tensor,
    num_comp_tokens: torch.Tensor,
    cu_new_comp_kv: torch.Tensor,
    start_pos: torch.Tensor,
    compressed_mask: torch.Tensor,
    batch_size: int,
) -> None:
    """DeepSeek-V4 compressor: store this step's new compressed rows (one per ratio tokens).

    Row ``i`` belongs to the request whose ``cu_new_comp_kv`` span holds it and
    lands at compressed position ``start_pos[request] + offset``; rows past a
    request's ``num_comp_tokens`` or outside ``compressed_mask`` are padding.
    """
    if _retained_pool(metadata, compressor.layer_idx) is None:
        return
    total = rows.shape[0]
    if total == 0:
        return
    token_idx = torch.arange(total, dtype=torch.int32, device=rows.device)
    cu = cu_new_comp_kv[: batch_size + 1].to(torch.int32)
    batch_idx = torch.searchsorted(cu[1:], token_idx, right=True).clamp(max=batch_size - 1)
    local_idx = token_idx - cu[batch_idx]
    valid = compressed_mask[:total].to(torch.bool) & (local_idx < num_comp_tokens[batch_idx])
    positions = torch.where(valid, start_pos[batch_idx] + local_idx, torch.full_like(local_idx, -1))
    write_rows(
        metadata, compressor.layer_idx, rows, batch_idx.to(torch.int32), positions.to(torch.int32)
    )


def overlay_rows(
    attention,
    metadata,
    topk_indices: torch.Tensor,
    req_idx: torch.Tensor,
    compact_indices: torch.Tensor,
    scratch: torch.Tensor,
) -> None:
    """After the backend's gather: copy the retained rows over the dequantized scratch rows.

    ``topk_indices`` holds each query row's selected token positions inside its
    request (-1 = none), ``req_idx`` that row's request in batch order and
    ``compact_indices`` the scratch row the gather used for each pair (-1 = none).
    """
    pool = _retained_pool(metadata, attention.layer_idx)
    if pool is None:
        return
    num_rows, k = topk_indices.shape
    if num_rows == 0 or k == 0:
        return
    assert topk_indices.stride(1) == 1 and compact_indices.stride(1) == 1
    width = pool.shape[-1]
    table = metadata.mixed_high_block_table
    _overlay_rows_kernel[(num_rows * k,)](
        topk_indices,
        topk_indices.stride(0),
        compact_indices,
        compact_indices.stride(0),
        req_idx,
        table,
        table.stride(0),
        pool,
        scratch,
        pool.shape[1],
        k,
        ROW=width,
        BLOCK=_block(width),
    )
