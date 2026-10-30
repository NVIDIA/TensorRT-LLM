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
def _copy_kv_row(
    src_ptr,
    src_row,
    dst_ptr,
    dst_row,
    scale_ptr,
    cos_sin_ptr,
    cos_sin_stride,
    position,
    APPLY_SCALE: tl.constexpr,
    ROTATE: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < ROW_WIDTH
    src = src_ptr + src_row.to(tl.int64) * ROW_WIDTH
    row = tl.load(src + offsets, mask=mask, other=0.0)
    if ROTATE:
        # Rotate k_pe like the append kernel: float math, round, scale.
        rope = offsets - (ROW_WIDTH - ROPE_DIM)
        in_rope = mask & (rope >= 0)
        x = row.to(tl.float32)
        partner = tl.load(src + (offsets ^ 1), mask=in_rope, other=0.0).to(tl.float32)
        cos_sin = cos_sin_ptr + position.to(tl.int64) * cos_sin_stride + (rope // 2) * 2
        cos = tl.load(cos_sin, mask=in_rope, other=1.0)
        sin = tl.load(cos_sin + 1, mask=in_rope, other=0.0)
        rotated = tl.where(rope % 2 == 0, x * cos - partner * sin, x * cos + partner * sin)
        row = tl.where(in_rope, rotated.to(src_ptr.dtype.element_ty), row)
    if APPLY_SCALE:
        # Per-tensor FP8 KV scale, as the native append kernel applies it.
        row = row.to(tl.float32) * tl.load(scale_ptr)
    tl.store(
        dst_ptr + dst_row.to(tl.int64) * ROW_WIDTH + offsets,
        row.to(dst_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _store_retained_rows_kernel(
    rows_ptr,
    request_ptr,
    position_ptr,
    block_table_ptr,
    block_table_stride,
    num_requests,
    max_blocks,
    kv_cache_ptr,
    num_blocks,
    tokens_per_block,
    scale_ptr,
    cos_sin_ptr,
    cos_sin_stride,
    APPLY_SCALE: tl.constexpr,
    ROTATE: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per token: store its row at its position in the retained pool, if held."""
    token = tl.program_id(0)
    request = tl.load(request_ptr + token)
    position = tl.load(position_ptr + token)
    block = position // tokens_per_block
    valid = (position >= 0) & (block < max_blocks) & (request >= 0) & (request < num_requests)
    page = tl.load(block_table_ptr + request * block_table_stride + block, mask=valid, other=-1)
    if (page >= 0) & (page < num_blocks):
        slot = page.to(tl.int64) * tokens_per_block + position % tokens_per_block
        _copy_kv_row(
            rows_ptr,
            token,
            kv_cache_ptr,
            slot,
            scale_ptr,
            cos_sin_ptr,
            cos_sin_stride,
            position,
            APPLY_SCALE,
            ROTATE,
            ROW_WIDTH,
            ROPE_DIM,
            BLOCK_SIZE,
        )


# The epoch changes every call: keep it out of the specialization key.
@triton.jit(do_not_specialize=["claim_epoch"])
def _overlay_retained_rows_kernel(
    topk_ptr,
    topk_stride,
    compact_ptr,
    compact_stride,
    request_ptr,
    block_table_ptr,
    block_table_stride,
    num_requests,
    max_blocks,
    kv_cache_ptr,
    num_blocks,
    scratch_ptr,
    scratch_rows,
    claim_ptr,
    claim_epoch,
    tokens_per_block,
    topk,
    scale_ptr,
    APPLY_SCALE: tl.constexpr,
    DEDUP: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """One program per (query row, BLOCK_K selected tokens): copy the rows the twin holds."""
    query = tl.program_id(0)
    ks = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    in_row = ks < topk
    request = tl.load(request_ptr + query)
    position = tl.load(topk_ptr + query.to(tl.int64) * topk_stride + ks, mask=in_row, other=-1)
    scratch_row = tl.load(
        compact_ptr + query.to(tl.int64) * compact_stride + ks, mask=in_row, other=-1
    )
    # Padded entries of the index tensors hold stale values; bound them like the backends do.
    block = position // tokens_per_block
    valid = in_row & (position >= 0) & (block < max_blocks)
    valid = valid & (scratch_row >= 0) & (scratch_row < scratch_rows)
    valid = valid & (request >= 0) & (request < num_requests)
    page = tl.load(block_table_ptr + request * block_table_stride + block, mask=valid, other=-1)
    valid = valid & (page >= 0) & (page < num_blocks)
    slot = page.to(tl.int64) * tokens_per_block + position % tokens_per_block
    if DEDUP:
        # Context scratch rows are shared by every query that selected the token: record the
        # row's twin slot (same value from every pair, no atomics); a per-row pass copies it once.
        # Most pairs find the row already recorded and skip the store.
        record = (claim_epoch.to(tl.int64) << 32) | slot
        current = tl.load(claim_ptr + scratch_row, mask=valid, other=0)
        tl.store(claim_ptr + scratch_row, record, mask=valid & (current != record))
        return
    src = kv_cache_ptr + slot[:, None] * ROW_WIDTH
    dst = scratch_ptr + scratch_row.to(tl.int64)[:, None] * ROW_WIDTH
    if APPLY_SCALE:
        scale = tl.load(scale_ptr)
    for start in range(0, ROW_WIDTH, BLOCK_W):
        cols = start + tl.arange(0, BLOCK_W)
        mask = valid[:, None] & (cols < ROW_WIDTH)[None, :]
        row = tl.load(src + cols[None, :], mask=mask)
        if APPLY_SCALE:
            row = row.to(tl.float32) * scale
        tl.store(dst + cols[None, :], row.to(scratch_ptr.dtype.element_ty), mask=mask)


# The epoch changes every call: keep it out of the specialization key.
@triton.jit(do_not_specialize=["claim_epoch"])
def _copy_recorded_rows_kernel(
    claim_ptr,
    claim_epoch,
    kv_cache_ptr,
    scratch_ptr,
    scratch_rows,
    scale_ptr,
    APPLY_SCALE: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """One program per BLOCK_R scratch rows: copy the twin row recorded for this call's epoch."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    recorded = tl.load(claim_ptr + rows, mask=rows < scratch_rows, other=0)
    valid = (rows < scratch_rows) & ((recorded >> 32) == claim_epoch.to(tl.int64))
    slot = recorded & 0xFFFFFFFF
    src = kv_cache_ptr + slot[:, None] * ROW_WIDTH
    dst = scratch_ptr + rows.to(tl.int64)[:, None] * ROW_WIDTH
    if APPLY_SCALE:
        scale = tl.load(scale_ptr)
    for start in range(0, ROW_WIDTH, BLOCK_W):
        cols = start + tl.arange(0, BLOCK_W)
        mask = valid[:, None] & (cols < ROW_WIDTH)[None, :]
        row = tl.load(src + cols[None, :], mask=mask)
        if APPLY_SCALE:
            row = row.to(tl.float32) * scale
        tl.store(dst + cols[None, :], row.to(scratch_ptr.dtype.element_ty), mask=mask)


def _retained_block_table(metadata, owner) -> Optional[torch.Tensor]:
    """Retained page table ``owner`` uses (one per compression ratio on DeepSeek-V4)."""
    table = getattr(metadata, "retained_block_table", None)
    if isinstance(table, dict):
        table = table.get(getattr(owner, "compress_ratio", None))
    return table


def _retained_kv_cache(metadata, owner) -> Optional[torch.Tensor]:
    """``[num_blocks, tokens_per_block, row]`` view of ``owner``'s retained pool, else None."""
    if _retained_block_table(metadata, owner) is None:
        return None
    manager = metadata.kv_cache_manager
    local = manager.layer_offsets[owner.layer_idx]
    if local not in manager.high_layers:
        # A layer without a twin (e.g. a drafter with its own architecture) keeps a plain cache.
        return None
    # [blocks, kv_factor, tokens, heads, width] with one head and no value half.
    pool = manager._get_mixed_precision_buffers(local, "NHD", "high")
    return pool.view(pool.shape[0], pool.shape[2], -1)


def _block_size(row_width: int) -> int:
    return max(16, triton.next_power_of_2(row_width))


def store_retained_rows(
    metadata,
    owner,
    rows: torch.Tensor,
    requests: torch.Tensor,
    positions: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    rotary_cos_sin: Optional[torch.Tensor] = None,
    rope_dim: int = 0,
) -> None:
    """Store ``rows`` in ``owner``'s retained pool (FP8 pool: ``row * scale``).

    With ``rotary_cos_sin`` the trailing ``rope_dim`` of every row is rotated at its position first.
    """
    kv_cache = _retained_kv_cache(metadata, owner)
    if kv_cache is None:
        return
    rows = rows.reshape(rows.shape[0], -1)
    if rows.shape[0] == 0:
        return
    if not rows.is_contiguous():
        rows = rows.contiguous()
    row_width = kv_cache.shape[-1]
    assert rows.shape[1] == row_width, (rows.shape, row_width)
    block_table = _retained_block_table(metadata, owner)
    apply_scale = scale is not None and kv_cache.dtype == torch.float8_e4m3fn
    rotate = rotary_cos_sin is not None
    if rotate:
        # (cos, sin) per pair and position; the first rope_dim // 2 pairs are used.
        rotary_cos_sin = rotary_cos_sin.view(-1, rope_dim, 2)
        assert rotary_cos_sin.stride(1) == 2 and rotary_cos_sin.stride(2) == 1
    _store_retained_rows_kernel[(rows.shape[0],)](
        rows,
        requests,
        positions,
        block_table,
        block_table.stride(0),
        block_table.shape[0],
        block_table.shape[1],
        kv_cache,
        kv_cache.shape[0],
        kv_cache.shape[1],
        scale if apply_scale else kv_cache,
        rotary_cos_sin if rotate else kv_cache,
        rotary_cos_sin.stride(0) if rotate else 0,
        APPLY_SCALE=apply_scale,
        ROTATE=rotate,
        ROW_WIDTH=row_width,
        ROPE_DIM=rope_dim,
        BLOCK_SIZE=_block_size(row_width),
        # Separate roundings like the append kernel; FMA contraction would flip FP8 bits.
        enable_fp_fusion=False,
    )


def store_retained_context_rows(attention, metadata, latent_cache: torch.Tensor) -> None:
    """Context phase: the append kernel wrote the rotated k_pe back into ``latent_cache``."""
    positions = getattr(metadata, "retained_context_positions", None)
    if positions is None:
        return
    num_ctx = metadata.num_ctx_tokens
    store_retained_rows(
        metadata,
        attention,
        latent_cache[:num_ctx],
        metadata.req_idx_per_token[:num_ctx],
        positions,
        attention.kv_scale_orig_quant,
    )


def store_retained_generation_rows(
    attention, metadata, latent_cache: torch.Tensor, position_ids: torch.Tensor
) -> None:
    """Generation: the fused kernel rotates k_pe in-kernel only, so the store rotates it as well."""
    num_rows = latent_cache.shape[0]
    if num_rows == 0:
        return
    assert position_ids is not None, "the retained-row store needs the generation positions"
    # Generation tokens are the trailing positions whichever slice the caller passes.
    positions = position_ids.reshape(-1)[-num_rows:]
    num_ctx = metadata.num_ctx_tokens
    store_retained_rows(
        metadata,
        attention,
        latent_cache,
        metadata.req_idx_per_token[num_ctx : num_ctx + num_rows],
        positions,
        attention.kv_scale_orig_quant,
        attention.rotary_cos_sin,
        attention.mla_params.qk_rope_head_dim,
    )


@triton.jit
def _retained_compressed_row_mask_kernel(
    out_ptr,
    block_table_ptr,
    block_table_stride,
    max_blocks,
    tokens_per_block,
    num_comp_tokens_ptr,
    cu_new_comp_kv_ptr,
    start_pos_ptr,
    compressed_mask_ptr,
    batch_size,
    total_rows,
    SEARCH_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < total_rows
    # request = number of cu_new_comp_kv[1:batch_size + 1] entries <= row (binary search).
    lo = tl.zeros([BLOCK], dtype=tl.int32)
    hi = tl.zeros([BLOCK], dtype=tl.int32) + batch_size
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        searching = lo < hi
        bound = tl.load(cu_new_comp_kv_ptr + 1 + mid, mask=searching & (mid < batch_size), other=0)
        right = searching & (bound <= rows)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(searching & ~right, mid, hi)
    request = tl.minimum(lo, batch_size - 1)
    local = rows - tl.load(cu_new_comp_kv_ptr + request)
    position = tl.load(start_pos_ptr + request) + local
    block = position // tokens_per_block
    # Blocks past the table's width are pages the twin never held.
    in_table = in_range & (position >= 0) & (block < max_blocks)
    page = tl.load(block_table_ptr + request * block_table_stride + block, mask=in_table, other=-1)
    selected = tl.load(compressed_mask_ptr + rows, mask=in_range, other=0) != 0
    keep = selected & (local < tl.load(num_comp_tokens_ptr + request)) & in_table & (page >= 0)
    tl.store(out_ptr + rows, keep, mask=in_range)


def _retained_compressed_row_mask(
    block_table: torch.Tensor,
    tokens_per_block: int,
    num_comp_tokens: torch.Tensor,
    cu_new_comp_kv: torch.Tensor,
    start_pos: torch.Tensor,
    compressed_mask: torch.Tensor,
    batch_size: int,
    total_rows: int,
) -> torch.Tensor:
    """``compressed_mask`` narrowed to the rows whose page the retained twin holds.

    The native scatter addresses pages straight from the table it is given and does not
    understand -1, so the rows of pages the twin released must be masked out.
    """
    mask = torch.empty(total_rows, dtype=torch.bool, device=block_table.device)
    block = 256
    _retained_compressed_row_mask_kernel[(triton.cdiv(total_rows, block),)](
        mask,
        block_table,
        block_table.stride(0),
        block_table.shape[1],
        tokens_per_block,
        num_comp_tokens,
        cu_new_comp_kv,
        start_pos,
        compressed_mask,
        batch_size,
        total_rows,
        SEARCH_STEPS=max(1, batch_size.bit_length()),
        BLOCK=block,
    )
    return mask


def store_retained_compressed_rows(
    compressor,
    metadata,
    kv_comp: torch.Tensor,
    position_ids: torch.Tensor,
    num_comp_tokens: torch.Tensor,
    cu_new_comp_kv: torch.Tensor,
    start_pos: torch.Tensor,
    compressed_mask: torch.Tensor,
    batch_size: int,
) -> None:
    """DeepSeek-V4: run the native compressor scatter a second time, into the retained twin."""
    from .deepseek_v4.compressor import KVCacheDtype

    kv_cache = _retained_kv_cache(metadata, compressor)
    total_rows = kv_comp.shape[0]
    if kv_cache is None or total_rows == 0 or batch_size == 0:
        return
    block_table = _retained_block_table(metadata, compressor)
    tokens_per_block = kv_cache.shape[1]
    mask = _retained_compressed_row_mask(
        block_table,
        tokens_per_block,
        num_comp_tokens,
        cu_new_comp_kv,
        start_pos,
        compressed_mask,
        batch_size,
        total_rows,
    )
    cache_dtype = (
        KVCacheDtype.FP8_PERTENSOR if kv_cache.dtype == torch.float8_e4m3fn else KVCacheDtype.NONE
    )
    torch.ops.trtllm.compressor_postprocess_scatter(
        kv_comp,
        None,
        compressor.norm.weight,
        compressor.norm.variance_epsilon,
        compressor.rotary_emb.rotary_cos_sin,
        position_ids,
        compressor.nope_head_dim,
        compressor.rope_head_dim,
        kv_cache,
        None,
        num_comp_tokens,
        cu_new_comp_kv,
        start_pos,
        block_table,
        mask,
        tokens_per_block,
        int(cache_dtype),
        1.0,
        0,
        compressor.rotate_activation,
        None,
        None,
    )


_OVERLAY_BLOCK_K = 64
# Per-device scratch-row records for context overlays, and the epoch of the last overlay.
_overlay_claims = {}


def _overlay_claim(device: torch.device, rows: int):
    """Per-row records covering ``rows`` scratch rows and a fresh epoch (no per-call clearing)."""
    claims, epoch = _overlay_claims.get(device, (None, 0))
    epoch += 1
    if claims is None or claims.numel() < rows or epoch >= 2**31 - 1:
        claims = torch.zeros(max(rows, 1 << 16), dtype=torch.int64, device=device)
        epoch = 1
    _overlay_claims[device] = (claims, epoch)
    return claims, epoch


def overlay_retained_rows(
    attention,
    metadata,
    topk_indices: torch.Tensor,
    requests: torch.Tensor,
    compact_indices: torch.Tensor,
    scratch: torch.Tensor,
    shared_rows: bool = False,
) -> None:
    """After the gather: copy the retained rows over the dequantized scratch rows.

    ``topk_indices``: selected positions per query row (-1 = none); ``requests``: that row's
    request; ``compact_indices``: the scratch row the gather used for each pair (-1 = none).
    ``shared_rows``: the context gather deduplicates, so many pairs point at one scratch row.
    """
    kv_cache = _retained_kv_cache(metadata, attention)
    if kv_cache is None:
        return
    num_rows, topk = topk_indices.shape
    if num_rows == 0 or topk == 0:
        return
    assert topk_indices.stride(1) == 1 and compact_indices.stride(1) == 1
    row_width = kv_cache.shape[-1]
    scratch_rows = scratch.numel() // row_width
    block_table = _retained_block_table(metadata, attention)
    apply_scale = kv_cache.dtype != scratch.dtype
    # A captured graph would replay one epoch; there every pair copies its row instead.
    dedup = shared_rows and not torch.cuda.is_current_stream_capturing()
    claims, epoch = _overlay_claim(scratch.device, scratch_rows) if dedup else (block_table, 0)
    _overlay_retained_rows_kernel[(num_rows, triton.cdiv(topk, _OVERLAY_BLOCK_K))](
        topk_indices,
        topk_indices.stride(0),
        compact_indices,
        compact_indices.stride(0),
        requests,
        block_table,
        block_table.stride(0),
        block_table.shape[0],
        block_table.shape[1],
        kv_cache,
        kv_cache.shape[0],
        scratch,
        scratch_rows,
        claims,
        epoch,
        kv_cache.shape[1],
        topk,
        attention.kv_scale_orig_quant if apply_scale else kv_cache,
        APPLY_SCALE=apply_scale,
        DEDUP=dedup,
        ROW_WIDTH=row_width,
        BLOCK_K=_OVERLAY_BLOCK_K,
        BLOCK_W=128,
    )
    if dedup:
        block_rows = 32
        _copy_recorded_rows_kernel[(triton.cdiv(scratch_rows, block_rows),)](
            claims,
            epoch,
            kv_cache,
            scratch,
            scratch_rows,
            attention.kv_scale_orig_quant if apply_scale else kv_cache,
            APPLY_SCALE=apply_scale,
            ROW_WIDTH=row_width,
            BLOCK_R=block_rows,
            BLOCK_W=128,
        )
