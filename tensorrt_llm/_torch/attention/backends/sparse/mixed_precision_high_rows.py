# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision high pool of the sparse MLA backends and the cold-page window.

New tokens are written to the high pool as well; with an NVFP4 history its rows overwrite the
dequantized scratch after the top-k gather, with a plain history (cold-page window) the held
positions are read from its rows in place. High page table: the page index if held, else -1.
"""

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from ..mixed_precision_rows import copy_rows

if TYPE_CHECKING:
    from ..trtllm import TrtllmAttention
    from .deepseek_v4.compressor import Compressor
    from .dsa.metadata import DSAtrtllmAttentionMetadata


@triton.jit
def _store_mixed_precision_high_rows_kernel(
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
    """One program per token: store its row in the mixed-precision high pool if its page is held."""
    token = tl.program_id(0)
    request = tl.load(request_ptr + token)
    position = tl.load(position_ptr + token)
    block = position // tokens_per_block
    valid = (position >= 0) & (block < max_blocks) & (request >= 0) & (request < num_requests)
    page = tl.load(block_table_ptr + request * block_table_stride + block, mask=valid, other=-1)
    if (page >= 0) & (page < num_blocks):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < ROW_WIDTH
        src = rows_ptr + token.to(tl.int64) * ROW_WIDTH
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
            row = tl.where(in_rope, rotated.to(rows_ptr.dtype.element_ty), row)
        if APPLY_SCALE:
            # Per-tensor FP8 KV scale, as the native append kernel applies it.
            row = row.to(tl.float32) * tl.load(scale_ptr)
        slot = page.to(tl.int64) * tokens_per_block + position % tokens_per_block
        tl.store(
            kv_cache_ptr + slot * ROW_WIDTH + offsets,
            row.to(kv_cache_ptr.dtype.element_ty),
            mask=mask,
        )


# The epoch changes every call: keep it out of the specialization key.
@triton.jit(do_not_specialize=["claim_epoch"])
def _overlay_mixed_precision_high_rows_kernel(
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
    """Per (query row, BLOCK_K tokens): copy rows the mixed-precision high pool holds."""
    query = tl.program_id(0).to(tl.int64)
    ks = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    in_row = ks < topk
    request = tl.load(request_ptr + query)
    position = tl.load(topk_ptr + query * topk_stride + ks, mask=in_row, other=-1)
    scratch_row = tl.load(compact_ptr + query * compact_stride + ks, mask=in_row, other=-1)
    # Padded entries of the index tensors hold stale values; bound them like the backends do.
    block = position // tokens_per_block
    valid = (position >= 0) & (block < max_blocks) & (request >= 0) & (request < num_requests)
    valid = valid & (scratch_row >= 0) & (scratch_row < scratch_rows)
    page = tl.load(block_table_ptr + request * block_table_stride + block, mask=valid, other=-1)
    valid = valid & (page >= 0) & (page < num_blocks)
    slot = page.to(tl.int64) * tokens_per_block + position % tokens_per_block
    if DEDUP:
        # Context scratch rows are shared by every query that selected the token: record the row's
        # high pool slot (same value from every pair, no atomics); a per-row pass copies it once.
        record = (claim_epoch.to(tl.int64) << 32) | slot
        current = tl.load(claim_ptr + scratch_row, mask=valid, other=0)
        tl.store(claim_ptr + scratch_row, record, mask=valid & (current != record))
        return
    copy_rows(
        kv_cache_ptr,
        slot,
        scratch_ptr,
        scratch_row,
        valid,
        scale_ptr,
        APPLY_SCALE,
        ROW_WIDTH,
        BLOCK_W,
    )


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
    """One program per BLOCK_R scratch rows: copy the mixed-precision high pool row recorded now."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    recorded = tl.load(claim_ptr + rows, mask=rows < scratch_rows, other=0)
    valid = (rows < scratch_rows) & ((recorded >> 32) == claim_epoch.to(tl.int64))
    copy_rows(
        kv_cache_ptr,
        recorded & 0xFFFFFFFF,
        scratch_ptr,
        rows,
        valid,
        scale_ptr,
        APPLY_SCALE,
        ROW_WIDTH,
        BLOCK_W,
    )


def _mixed_precision_high_pool(
    metadata: "DSAtrtllmAttentionMetadata", owner: "TrtllmAttention | Compressor"
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """`owner`'s mixed-precision high pool `[num_blocks, tokens_per_block, row]` and page table.

    Both are None when `owner` has no mixed-precision high pool.
    """
    table = metadata.mixed_precision_high_block_table
    if isinstance(table, dict):
        # DeepSeek-V4: one page table per compression ratio.
        table = table.get(owner.compress_ratio)
    if table is None:
        return None, None
    manager = metadata.kv_cache_manager
    local = manager.layer_offsets[owner.layer_idx]
    if local not in manager.mixed_precision_high_layers:
        # No mixed-precision high pool (e.g. an Eagle3 drafter layer): a plain cache.
        return None, None
    # [blocks, kv_factor, tokens, heads, width] with one head and no value half.
    pool = manager._get_mixed_precision_buffers(local, "NHD", "high")
    return pool.view(pool.shape[0], pool.shape[2], -1), table


def store_mixed_precision_high_rows(
    attention: "TrtllmAttention",
    metadata: "DSAtrtllmAttentionMetadata",
    latent_cache: torch.Tensor,
    position_ids: torch.Tensor,
    is_generation: bool,
) -> None:
    """Store this phase's tokens in the mixed-precision high pool where it holds their pages.

    FP8 pools store `row * scale`. Context: the append kernel has already rotated k_pe and written
    it back into `latent_cache`. Generation: the fused kernel rotates k_pe only inside the kernel,
    so the store rotates it too.

    Args:
        attention: Attention layer whose mixed-precision high pool receives the rows.
        metadata: Attention metadata of the batch.
        latent_cache: `[num_tokens, kv_lora_rank + qk_rope_head_dim]` compressed KV and k_pe in
            the activation dtype: the batch's tokens, context first, or only the generation ones.
        position_ids: int32 positions of the batch, context tokens first, in any shape that
            flattens to one dimension; a context token's position is its KV position.
        is_generation: Whether to store the generation tokens instead of the context tokens.
    """
    num_ctx = metadata.num_ctx_tokens
    num_rows = latent_cache.shape[0] if is_generation else num_ctx
    if num_rows == 0:
        return
    assert position_ids is not None, "positions are required"
    positions = position_ids.reshape(-1)
    # Generation tokens are the trailing positions whichever slice the caller passes.
    positions = positions[-num_rows:] if is_generation else positions[:num_ctx]
    first = num_ctx if is_generation else 0
    kv_cache, block_table = _mixed_precision_high_pool(metadata, attention)
    if kv_cache is None:
        return
    rows = latent_cache[:num_rows].flatten(1).contiguous()
    row_width = kv_cache.shape[-1]
    assert rows.shape[1] == row_width, (rows.shape, row_width)
    scale = attention.kv_scale_orig_quant
    apply_scale = scale is not None and kv_cache.dtype == torch.float8_e4m3fn
    rope_dim = attention.mla_params.qk_rope_head_dim if is_generation else 0
    rotary_cos_sin = attention.rotary_cos_sin if is_generation else None
    rotate = rotary_cos_sin is not None
    if rotate:
        # (cos, sin) per pair and position; the first rope_dim // 2 pairs are used.
        rotary_cos_sin = rotary_cos_sin.view(-1, rope_dim, 2)
        assert rotary_cos_sin.stride(1) == 2 and rotary_cos_sin.stride(2) == 1
    _store_mixed_precision_high_rows_kernel[(rows.shape[0],)](
        rows,
        metadata.req_idx_per_token[first : first + num_rows],
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
        BLOCK_SIZE=max(16, triton.next_power_of_2(row_width)),
        # Separate roundings like the append kernel; FMA contraction would flip FP8 bits.
        enable_fp_fusion=False,
    )


@triton.jit
def _mixed_precision_high_compressed_row_mask_kernel(
    out_ptr,
    block_table_ptr,
    block_table_stride,
    max_blocks,
    tokens_per_block,
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
    # Blocks past the table's width are pages the mixed-precision high pool never held.
    in_table = in_range & (position >= 0) & (block < max_blocks)
    page = tl.load(block_table_ptr + request * block_table_stride + block, mask=in_table, other=-1)
    # compressed_mask already limits every request to its new compressed rows.
    selected = tl.load(compressed_mask_ptr + rows, mask=in_range, other=0) != 0
    tl.store(out_ptr + rows, selected & in_table & (page >= 0), mask=in_range)


def store_mixed_precision_high_compressed_rows(
    compressor: "Compressor",
    metadata: "DSAtrtllmAttentionMetadata",
    kv_comp: torch.Tensor,
    position_ids: torch.Tensor,
    num_comp_tokens: torch.Tensor,
    cu_new_comp_kv: torch.Tensor,
    start_pos: torch.Tensor,
    compressed_mask: torch.Tensor,
    batch_size: int,
) -> None:
    """DeepSeek-V4: run the compressor scatter again, into the mixed-precision high pool.

    Args:
        compressor: Main (non-indexer) compressor of a layer with a mixed-precision high pool.
        metadata: Attention metadata of the batch.
        kv_comp: `[total_rows, head_dim]` compressed KV rows before RMSNorm and RoPE, in the
            activation dtype.
        position_ids: `[total_rows]` int32 positions of the compressed rows.
        num_comp_tokens: `[batch_size]` int32 new compressed rows of each request.
        cu_new_comp_kv: int32 offsets of each request's rows in `kv_comp`; at least
            `batch_size + 1` entries.
        start_pos: `[batch_size]` int32 compressed rows each request had cached before this step.
        compressed_mask: `[total_rows]` bool rows the native scatter writes.
        batch_size: Number of requests in the batch.
    """
    from .deepseek_v4.compressor import KVCacheDtype

    kv_cache, block_table = _mixed_precision_high_pool(metadata, compressor)
    total_rows = kv_comp.shape[0]
    if kv_cache is None or total_rows == 0 or batch_size == 0:
        return
    tokens_per_block = kv_cache.shape[1]
    # Narrow compressed_mask to rows whose page the mixed-precision high pool holds: the native
    # scatter addresses pages straight from the table it is given and does not understand -1.
    mask = torch.empty(total_rows, dtype=torch.bool, device=block_table.device)
    _mixed_precision_high_compressed_row_mask_kernel[(triton.cdiv(total_rows, 256),)](
        mask,
        block_table,
        block_table.stride(0),
        block_table.shape[1],
        tokens_per_block,
        cu_new_comp_kv,
        start_pos,
        compressed_mask,
        batch_size,
        total_rows,
        SEARCH_STEPS=max(1, batch_size.bit_length()),
        BLOCK=256,
    )
    cache_dtype = (
        KVCacheDtype.FP8_PERTENSOR if kv_cache.dtype == torch.float8_e4m3fn else KVCacheDtype.NONE
    )
    torch.ops.trtllm.compressor_postprocess_scatter(
        kv_comp,
        None,  # kv_out
        compressor.norm.weight,
        compressor.norm.variance_epsilon,
        compressor.rotary_emb.rotary_cos_sin,
        position_ids,
        compressor.nope_head_dim,
        compressor.rope_head_dim,
        kv_cache,
        None,  # kv_cache_scale
        num_comp_tokens,
        cu_new_comp_kv,
        start_pos,
        block_table,
        mask,
        tokens_per_block,
        int(cache_dtype),
        1.0,  # nvfp4_global_scale (unused: the mixed-precision high pool is not NVFP4)
        0,  # nvfp4_residual_dim (must be 0: the pool is not NVFP4)
        compressor.rotate_activation,
        None,  # quant_output
        None,  # scale_output
    )


_OVERLAY_BLOCK_K = 64
# Per-device scratch-row records (`epoch << 32 | mixed-precision high pool slot`) for context
# overlays, and the epoch of the last overlay. Epochs start at 1, so zeroed records never match.
_overlay_claims: dict[torch.device, tuple[torch.Tensor, int]] = {}


def _overlay_claim(device: torch.device, rows: int) -> tuple[torch.Tensor, int]:
    """Per-row records covering `rows` scratch rows and a fresh epoch (no per-call clearing)."""
    claims, epoch = _overlay_claims.get(device, (None, 0))
    epoch += 1
    # Restart before the epoch outgrows int32: it fills the upper half of an int64 record.
    if claims is None or claims.numel() < rows or epoch >= 2**31 - 1:
        # At least 64K rows, so that smaller batches reuse the buffer.
        claims = torch.zeros(max(rows, 1 << 16), dtype=torch.int64, device=device)
        epoch = 1
    _overlay_claims[device] = (claims, epoch)
    return claims, epoch


def overlay_mixed_precision_high_rows(
    attention: "TrtllmAttention",
    metadata: "DSAtrtllmAttentionMetadata",
    topk_indices: torch.Tensor,
    requests: torch.Tensor,
    compact_indices: torch.Tensor,
    scratch: torch.Tensor,
    shared_rows: bool = False,
) -> None:
    """After the gather: copy mixed-precision high pool rows over the FP8 `scratch` rows.

    Args:
        attention: Attention layer whose mixed-precision high pool supplies the rows.
        metadata: Attention metadata of the batch.
        topk_indices: `[num_rows, topk]` int32 KV positions each query token selected, -1 for
            none; contiguous in the last dimension.
        requests: `[num_rows]` int32 batch index of each query token's request.
        compact_indices: `[num_rows, topk]` int32 scratch row the gather used for each selected
            position, -1 for none; contiguous in the last dimension.
        scratch: `torch.float8_e4m3fn` gather output, read as rows of the mixed-precision high
            pool's row width.
        shared_rows: Whether many (query token, top-k entry) pairs may point at one scratch row,
            as after the deduplicating context gather.
    """
    kv_cache, block_table = _mixed_precision_high_pool(metadata, attention)
    if kv_cache is None:
        return
    num_rows, topk = topk_indices.shape
    if num_rows == 0 or topk == 0:
        return
    assert topk_indices.stride(1) == 1 and compact_indices.stride(1) == 1
    row_width = kv_cache.shape[-1]
    scratch_rows = scratch.numel() // row_width
    apply_scale = kv_cache.dtype != scratch.dtype
    # A captured graph would replay one epoch; there every pair copies its row instead.
    dedup = shared_rows and not torch.cuda.is_current_stream_capturing()
    # Without deduplication the kernel never reads the records; the page table is a placeholder.
    claims, epoch = _overlay_claim(scratch.device, scratch_rows) if dedup else (block_table, 0)
    _overlay_mixed_precision_high_rows_kernel[(num_rows, triton.cdiv(topk, _OVERLAY_BLOCK_K))](
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


def mixed_precision_high_restore_rows(
    spans: list[tuple[int, int, int]], tokens_per_row: int = 1
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Flat `(batch rows, row positions)` of the whole rows that `spans` cover.

    Args:
        spans: `(batch row, first token, end token)` per request, as
            `KVCacheManagerV2.take_mixed_precision_restore_spans` returns them.
        tokens_per_row: 1, or the DeepSeek-V4 compression ratio: a compressed row exists once all
            of its tokens are cached.

    Returns:
        `[num_rows]` int32 batch rows and row positions on the current CUDA device, or None.
    """
    requests, positions = [], []
    for row, start, end in spans:
        first, last = start // tokens_per_row, end // tokens_per_row
        requests += [row] * (last - first)
        positions += range(first, last)
    if not positions:
        return None
    device = torch.device("cuda", torch.cuda.current_device())
    return (
        torch.tensor(requests, dtype=torch.int32).to(device),
        torch.tensor(positions, dtype=torch.int32).to(device),
    )


def restore_mixed_precision_high_rows(
    owner: "TrtllmAttention",
    metadata: "DSAtrtllmAttentionMetadata",
    rows: torch.Tensor,
    requests: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Store FP8 `rows`, decoded as the read decodes them, at `(request, position)` unscaled.

    Args:
        owner: The attention layer whose mixed-precision high pool receives the rows.
        metadata: The sparse attention metadata of the step.
        rows: `[num_rows, ...]` in the high pool's dtype; each flattens to one high pool row.
        requests: `[num_rows]` int32 on the GPU: the batch row of each row.
        positions: `[num_rows]` int32 on the GPU: the position of each row, a compressed row
            position under DeepSeek-V4.
    """
    kv_cache, block_table = _mixed_precision_high_pool(metadata, owner)
    if kv_cache is None:
        return
    rows = rows.flatten(1)
    row_width = kv_cache.shape[-1]
    assert rows.shape[1] == row_width, (rows.shape, row_width)
    _store_mixed_precision_high_rows_kernel[(rows.shape[0],)](
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
        kv_cache,  # scale (unused)
        kv_cache,  # cos/sin (unused)
        0,
        APPLY_SCALE=False,
        ROTATE=False,
        ROW_WIDTH=row_width,
        ROPE_DIM=0,
        BLOCK_SIZE=max(16, triton.next_power_of_2(row_width)),
    )


def mixed_precision_high_row_indices(
    metadata: "DSAtrtllmAttentionMetadata",
    attention: "TrtllmAttention",
    topk_indices: torch.Tensor,
    topk_indices_global: torch.Tensor,
    requests: torch.Tensor,
) -> torch.Tensor:
    """Point the selected positions the mixed-precision high pool holds at its rows, in place.

    Args:
        metadata: The sparse attention metadata of the step.
        attention: The calling attention layer; its `layer_idx` selects the layer's high layer
            (and its `compress_ratio` the page table under DeepSeek-V4).
        topk_indices: `[num_rows, topk]` int32 with stride-1 rows: the selected KV positions of
            each query row, -1 for none.
        topk_indices_global: `[num_rows, topk]` int32 with stride-1 rows: those positions' pool
            rows, rewritten in place.
        requests: `[num_rows]` int32: the batch index of each query row.

    Returns:
        `topk_indices_global`.
    """
    block_table = metadata.mixed_precision_high_block_table
    if isinstance(block_table, dict):
        # DeepSeek-V4: one page table per compression ratio.
        block_table = block_table.get(attention.compress_ratio)
    if block_table is None:
        return topk_indices_global
    manager = metadata.kv_cache_manager
    local = manager.layer_offsets[attention.layer_idx]
    layer_offset = manager.mixed_precision_high_layer_offsets.get(local)
    if layer_offset is None:
        # No mixed-precision high pool (e.g. an Eagle3 drafter layer): a plain cache.
        return topk_indices_global
    num_rows, topk = topk_indices.shape
    if num_rows == 0 or topk == 0:
        return topk_indices_global
    assert topk_indices.stride(1) == 1
    rows_per_page = manager._mixed_precision_rows_per_page(local)
    # The page table already carries the page scale, so a page is one block of rows.
    high_rows = torch.ops.trtllm.convert_req_index_to_global(
        requests, block_table, topk_indices, rows_per_page, topk, rows_per_page, layer_offset
    )
    return torch.where(high_rows >= 0, high_rows, topk_indices_global, out=topk_indices_global)
