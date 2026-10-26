# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained-token rows for sparse MLA backends whose history lives in a native NVFP4 pool.

The KV append also writes each token's row into the manager's retained-token twin layer;
after the backend's top-k gather, the rows the twin still holds replace the dequantized
rows in its FP8 scratch. Twin page table: page >= 0 means retained.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row(
    src_ptr,
    src_row,
    dst_ptr,
    dst_row,
    scale_ptr,
    APPLY_SCALE: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < ROW
    row = tl.load(src_ptr + src_row.to(tl.int64) * ROW + offs, mask=mask, other=0.0)
    if APPLY_SCALE:
        # Per-tensor FP8 KV scale, as the native append kernel applies it.
        row = row.to(tl.float32) * tl.load(scale_ptr)
    tl.store(
        dst_ptr + dst_row.to(tl.int64) * ROW + offs, row.to(dst_ptr.dtype.element_ty), mask=mask
    )


@triton.jit
def _write_rows_kernel(
    rows_ptr,
    req_ptr,
    pos_ptr,
    table_ptr,
    table_stride,
    table_rows,
    table_width,
    pool_ptr,
    pool_pages,
    rows_per_page,
    scale_ptr,
    APPLY_SCALE: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    req = tl.load(req_ptr + token)
    pos = tl.load(pos_ptr + token)
    # pos < 0 marks a padding row the caller does not want stored.
    column = pos // rows_per_page
    valid = (pos >= 0) & (column < table_width) & (req >= 0) & (req < table_rows)
    page = tl.load(table_ptr + req * table_stride + column, mask=valid, other=-1)
    if (page >= 0) & (page < pool_pages):
        slot = page.to(tl.int64) * rows_per_page + pos % rows_per_page
        _copy_row(rows_ptr, token, pool_ptr, slot, scale_ptr, APPLY_SCALE, ROW, BLOCK)


@triton.jit
def _write_compressed_rows_kernel(
    rows_ptr,
    cu_ptr,
    num_comp_ptr,
    start_pos_ptr,
    mask_ptr,
    batch_size,
    table_ptr,
    table_stride,
    table_rows,
    table_width,
    pool_ptr,
    pool_pages,
    rows_per_page,
    scale_ptr,
    APPLY_SCALE: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Find the row's request from the cumulative counts, then its compressed position.
    token = tl.program_id(0)
    req = 0
    for b in range(1, batch_size):
        if token >= tl.load(cu_ptr + b):
            req = b
    local = token - tl.load(cu_ptr + req)
    valid = (tl.load(mask_ptr + token) != 0) & (local < tl.load(num_comp_ptr + req))
    valid = valid & (req < table_rows)
    pos = tl.load(start_pos_ptr + req) + local
    column = pos // rows_per_page
    valid = valid & (pos >= 0) & (column < table_width)
    page = tl.load(table_ptr + req * table_stride + column, mask=valid, other=-1)
    if (page >= 0) & (page < pool_pages):
        slot = page.to(tl.int64) * rows_per_page + pos % rows_per_page
        _copy_row(rows_ptr, token, pool_ptr, slot, scale_ptr, APPLY_SCALE, ROW, BLOCK)


@triton.jit
def _overlay_rows_kernel(
    topk_ptr,
    topk_stride,
    compact_ptr,
    compact_stride,
    req_ptr,
    table_ptr,
    table_stride,
    table_rows,
    table_width,
    pool_ptr,
    pool_pages,
    scratch_ptr,
    scratch_rows,
    rows_per_page,
    K,
    scale_ptr,
    APPLY_SCALE: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    row_id = pair // K
    col = pair % K
    tok = tl.load(topk_ptr + row_id.to(tl.int64) * topk_stride + col)
    compact = tl.load(compact_ptr + row_id.to(tl.int64) * compact_stride + col)
    req = tl.load(req_ptr + row_id)
    # Index tensors may hold stale padding: bound them like the backends' own conversion.
    column = tok // rows_per_page
    valid = (tok >= 0) & (column < table_width) & (compact >= 0) & (compact < scratch_rows)
    valid = valid & (req >= 0) & (req < table_rows)
    if valid:
        page = tl.load(table_ptr + req * table_stride + column)
        if (page >= 0) & (page < pool_pages):
            slot = page.to(tl.int64) * rows_per_page + tok % rows_per_page
            _copy_row(pool_ptr, slot, scratch_ptr, compact, scale_ptr, APPLY_SCALE, ROW, BLOCK)


def _retained_table(metadata, owner) -> Optional[torch.Tensor]:
    """Retained page table ``owner`` uses (one per compression ratio on DeepSeek-V4)."""
    table = getattr(metadata, "mixed_high_block_table", None)
    if isinstance(table, dict):
        table = table.get(getattr(owner, "compress_ratio", None))
    return table


def _retained_pool(metadata, owner) -> Optional[torch.Tensor]:
    """``[pages, rows_per_page, row]`` view of ``owner``'s retained pool, else None."""
    if _retained_table(metadata, owner) is None:
        return None
    manager = metadata.kv_cache_manager
    high_layers = getattr(manager, "high_layers", None)
    if not high_layers:
        return None
    local = manager.layer_offsets.get(owner.layer_idx)
    if local not in high_layers:
        return None
    # [pages, kv_factor, rows_per_page, heads, width] with one head and no value half.
    pool = manager._get_mixed_precision_buffers(local, "NHD", "high")
    return pool.view(pool.shape[0], pool.shape[2], -1)


def _block(width: int) -> int:
    return max(16, triton.next_power_of_2(width))


def write_rows(
    metadata,
    owner,
    rows: torch.Tensor,
    req_idx: torch.Tensor,
    positions: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> None:
    """Store ``rows`` (RoPE applied) in ``owner``'s retained pool (FP8 pool: ``row * scale``)."""
    pool = _retained_pool(metadata, owner)
    if pool is None:
        return
    rows = rows.reshape(rows.shape[0], -1)
    if rows.shape[0] == 0:
        return
    if not rows.is_contiguous():
        rows = rows.contiguous()
    width = pool.shape[-1]
    assert rows.shape[1] == width, (rows.shape, width)
    table = _retained_table(metadata, owner)
    apply_scale = scale is not None and pool.dtype == torch.float8_e4m3fn
    _write_rows_kernel[(rows.shape[0],)](
        rows,
        req_idx,
        positions,
        table,
        table.stride(0),
        table.shape[0],
        table.shape[1],
        pool,
        pool.shape[0],
        pool.shape[1],
        scale if apply_scale else pool,
        APPLY_SCALE=apply_scale,
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
        attention,
        latent_cache[:num_ctx],
        metadata.req_idx_per_token[:num_ctx],
        positions,
        attention.kv_scale_orig_quant,
    )


def write_generation_rows(
    attention, metadata, latent_cache: torch.Tensor, position_ids: torch.Tensor
) -> None:
    """Generation: rotate k_pe like the append kernel (the fused kernel leaves it unrotated)."""
    if _retained_pool(metadata, attention) is None:
        return
    rows = latent_cache.reshape(latent_cache.shape[0], -1)
    num_rows = rows.shape[0]
    if num_rows == 0:
        return
    assert position_ids is not None, "the retained-row write needs the generation positions"
    # Generation tokens are the trailing positions whichever slice the caller passes.
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
        attention,
        # Round to the activation dtype first, like the append kernel.
        rotated.to(rows.dtype),
        metadata.req_idx_per_token[num_ctx : num_ctx + num_rows],
        positions,
        attention.kv_scale_orig_quant,
    )


def retained_row_buffer(metadata, like: torch.Tensor) -> torch.Tensor:
    """Persistent ``[max tokens, row]`` buffer for the DeepSeek-V4 compressors (one per metadata).

    Allocated before CUDA graph capture: a per-call buffer on the compressor's side stream
    could be recycled while the compressor is still writing it.
    """
    buf = getattr(metadata, "_retained_rows", None)
    rows = max(int(metadata.max_num_tokens), like.shape[0])
    if (
        buf is None
        or buf.shape[0] < rows
        or buf.shape[1:] != like.shape[1:]
        or buf.dtype != like.dtype
    ):
        assert not torch.cuda.is_current_stream_capturing(), (
            "the retained-row buffer must be allocated before CUDA graph capture"
        )
        buf = torch.empty((rows, *like.shape[1:]), dtype=like.dtype, device=like.device)
        metadata._retained_rows = buf
    return buf[: like.shape[0]]


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
    """DeepSeek-V4 compressor: store this step's new compressed rows in the retained twin.

    Row ``i`` belongs to the request whose ``cu_new_comp_kv`` span holds it, at compressed
    position ``start_pos[request] + offset``; the native FP8 rows carry no per-tensor scale.
    """
    pool = _retained_pool(metadata, compressor)
    if pool is None:
        return
    total = rows.shape[0]
    if total == 0 or batch_size == 0:
        return
    width = pool.shape[-1]
    assert rows.shape[1] == width and rows.is_contiguous(), (rows.shape, width)
    table = _retained_table(metadata, compressor)
    _write_compressed_rows_kernel[(total,)](
        rows,
        cu_new_comp_kv,
        num_comp_tokens,
        start_pos,
        compressed_mask.view(torch.uint8),
        batch_size,
        table,
        table.stride(0),
        table.shape[0],
        table.shape[1],
        pool,
        pool.shape[0],
        pool.shape[1],
        pool,
        APPLY_SCALE=False,
        ROW=width,
        BLOCK=_block(width),
    )


def overlay_rows(
    attention,
    metadata,
    topk_indices: torch.Tensor,
    req_idx: torch.Tensor,
    compact_indices: torch.Tensor,
    scratch: torch.Tensor,
) -> None:
    """After the gather: copy the retained rows over the dequantized scratch rows.

    ``topk_indices``: selected positions per query row (-1 = none); ``req_idx``: that row's
    request; ``compact_indices``: the scratch row the gather used for each pair (-1 = none).
    """
    pool = _retained_pool(metadata, attention)
    if pool is None:
        return
    num_rows, k = topk_indices.shape
    if num_rows == 0 or k == 0:
        return
    assert topk_indices.stride(1) == 1 and compact_indices.stride(1) == 1
    width = pool.shape[-1]
    table = _retained_table(metadata, attention)
    apply_scale = pool.dtype != scratch.dtype
    _overlay_rows_kernel[(num_rows * k,)](
        topk_indices,
        topk_indices.stride(0),
        compact_indices,
        compact_indices.stride(0),
        req_idx,
        table,
        table.stride(0),
        table.shape[0],
        table.shape[1],
        pool,
        pool.shape[0],
        scratch,
        scratch.numel() // width,
        pool.shape[1],
        k,
        attention.kv_scale_orig_quant if apply_scale else pool,
        APPLY_SCALE=apply_scale,
        ROW=width,
        BLOCK=_block(width),
    )
