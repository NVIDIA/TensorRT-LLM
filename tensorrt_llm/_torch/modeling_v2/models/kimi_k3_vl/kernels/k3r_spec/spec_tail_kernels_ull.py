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
"""Speculative-decoding kernels and feature selection for Kimi K3 ULL.

The family spec tail provides the common acceptance, latent-storage and token
kernels. ULL binds its profile settings and uses specialized launchers for its
draft cache and vocabulary-sharded embedding.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from ...ops.common.k3_ull_features import active as _features
from .spec_tail_kernels import (
    _BLOCK_TABLE_BLOCK,
    BLOCK_TABLE_CHUNK,
    TAIL_V2_MAX_GENS,
    DraftBlockPlacement,
    PagedLatentPrepare,
    _sum_block_counts_kernel,
    _vocab_shard_embedding_kernel,
    dflash_ctx_positions,
    dflash_noise_embedding,
    dflash_paged_latent_prepare,
    fused_next_new_tokens,
    fused_next_new_tokens_rewind,
    fused_select_accept,
    fused_strict_accept,
    scatter_rows_to_pool,
    store_context_latents,
    store_mla_context_latents,
    strict_accept,
)

__all__ = [
    "BLOCK_TABLE_CHUNK",
    "DraftBlockPlacement",
    "PagedLatentPrepare",
    "TAIL_V2_MAX_GENS",
    "decode_draft_block_table",
    "decode_draft_block_table_parallel",
    "dflash_ctx_positions",
    "dflash_gen_prepare",
    "dflash_noise_embedding",
    "dflash_paged_latent_prepare",
    "fused_next_new_tokens",
    "fused_next_new_tokens_rewind",
    "fused_select_accept",
    "fused_strict_accept",
    "scatter_rows_to_pool",
    "spec_tail_fusion_enabled",
    "store_context_latents",
    "store_mla_context_latents",
    "strict_accept",
    "tail_v2_enabled",
    "vocab_shard_embedding",
]


@triton.jit
def _decode_block_table_kernel(
    src_ptr,
    stride_src_row,
    stride_src_col,
    n_blocks,
    divisor,
    table_ptr,
    stride_table,
    counts_ptr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    src_row = src_ptr + row.to(tl.int64) * stride_src_row
    table_row = table_ptr + row.to(tl.int64) * stride_table
    count = tl.zeros([BLOCK], dtype=tl.int64)
    for start in range(0, n_blocks, BLOCK):
        cols = start + offs
        mask = cols < n_blocks
        encoded = tl.load(src_row + cols * stride_src_col, mask=mask, other=-1).to(tl.int64)
        count += (encoded >= 0).to(tl.int64)
        tl.store(table_row + cols, (tl.maximum(encoded, 0) // divisor).to(tl.int32), mask=mask)
    tl.store(counts_ptr + row, tl.sum(count, axis=0))


def decode_draft_block_table(
    src: torch.Tensor, divisor: int, table: torch.Tensor, counts: torch.Tensor
) -> None:
    """``counts[i] = (src[i] >= 0).sum()``; ``table[i] = src[i].clamp(min=0) // divisor``.

    Args:
        src: ``[rows, blocks]`` encoded block offsets (any strides).
        divisor: encoded offset -> pool block index divisor.
        table: ``[>= rows, blocks]`` int32 output, unit inner stride.
        counts: ``[>= rows]`` int64 output.
    """
    rows, n_blocks = src.shape
    assert table.shape[-1] == n_blocks and table.stride(-1) == 1
    if rows == 0:
        return
    _decode_block_table_kernel[(rows,)](
        src,
        src.stride(0),
        src.stride(1),
        n_blocks,
        int(divisor),
        table,
        table.stride(0),
        counts,
        BLOCK=min(_BLOCK_TABLE_BLOCK, triton.next_power_of_2(n_blocks)),
        num_warps=4,
    )


def spec_tail_fusion_enabled() -> bool:
    """Whether the ULL profile enables fused speculative tail kernels."""
    return _features().spec_tail_fusion


def tail_v2_enabled() -> bool:
    """Whether the ULL profile enables the second set of fused tail kernels.

    These kernels cover context and block RoPE with in-kernel table lookups,
    direct storage of block latents into the paged pool, and column-parallel
    draft block-table decoding with shape-only operands built once. A TP-sharded
    drafter computes context latents on a side stream, concurrently with target
    token selection and acceptance.
    """
    return spec_tail_fusion_enabled() and _features().dspark_tail_v2


@triton.jit
def _decode_block_table_chunk_kernel(
    src_ptr,
    stride_src,
    stride_src_col,
    n_blocks,
    divisor,
    table_ptr,
    stride_table,
    partial_ptr,
    BLOCK: tl.constexpr,
):
    # Program (row, chunk) decodes columns [chunk * BLOCK, (chunk + 1) * BLOCK)
    # and records how many of them are allocated (encoded >= 0).
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_blocks
    src_row = src_ptr + row.to(tl.int64) * stride_src
    encoded = tl.load(src_row + cols * stride_src_col, mask=mask, other=-1).to(tl.int64)
    decoded = tl.maximum(encoded, 0) // divisor
    tl.store(table_ptr + row.to(tl.int64) * stride_table + cols, decoded.to(tl.int32), mask=mask)
    count = tl.sum((encoded >= 0).to(tl.int32), axis=0)
    tl.store(partial_ptr + row * tl.num_programs(1) + chunk, count)


def decode_draft_block_table_parallel(
    src: torch.Tensor,
    divisor: int,
    table: torch.Tensor,
    counts: torch.Tensor,
    partial: torch.Tensor,
) -> None:
    """:func:`decode_draft_block_table` with the columns of a row spread over programs.

    Same outputs (integer arithmetic, order-free). ``partial`` is an int32
    scratch of at least ``rows * cdiv(blocks, _BLOCK_TABLE_BLOCK)`` elements.
    """
    rows, n_blocks = src.shape
    assert table.shape[-1] == n_blocks and table.stride(-1) == 1
    if rows == 0:
        return
    n_chunks = triton.cdiv(n_blocks, _BLOCK_TABLE_BLOCK)
    assert partial.dtype == torch.int32 and partial.numel() >= rows * n_chunks
    _decode_block_table_chunk_kernel[(rows, n_chunks)](
        src,
        src.stride(0),
        src.stride(1),
        n_blocks,
        int(divisor),
        table,
        table.stride(0),
        partial,
        BLOCK=_BLOCK_TABLE_BLOCK,
        num_warps=8,
    )
    _sum_block_counts_kernel[(1,)](
        partial,
        n_chunks,
        counts,
        rows,
        BLOCK_R=triton.next_power_of_2(rows),
        BLOCK_C=triton.next_power_of_2(n_chunks),
        num_warps=4,
    )


@triton.jit
def _dflash_gen_prepare_kernel(
    num_accepted_ptr,
    accepted_ptr,
    stride_accepted,
    slots_ptr,
    ctx_len_ptr,
    block_counts_ptr,
    table_ptr,
    stride_table,
    max_blocks,
    seq_lens_ptr,
    bonus_ptr,
    embed_ids_ptr,
    query_pos_ptr,
    ctx_pos_ptr,
    store_pages_ptr,
    store_offsets_ptr,
    store_mask_ptr,
    num_ctx_ptr,
    blk_pages_ptr,
    blk_offsets_ptr,
    blk_seq_lens_ptr,
    blk_kv_bounds_ptr,
    blk_owned_ptr,
    position_offset_ptr,
    num_gens,
    num_ctx_reqs,
    dummy_slot,
    max_ctx,
    mask_token_id,
    cp_rank,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    CP_SIZE: tl.constexpr,
    HAS_POSITION_OFFSET: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    g = tl.arange(0, BLOCK_G)
    j = tl.arange(0, BLOCK_W)
    gm = g < num_gens
    row = num_ctx_reqs + g
    nacc = tl.load(num_accepted_ptr + row, mask=gm, other=1).to(tl.int64)
    slot = tl.load(slots_ptr + g, mask=gm, other=0).to(tl.int64)
    old = tl.load(ctx_len_ptr + slot, mask=gm, other=0).to(tl.int64)
    counts = tl.load(block_counts_ptr + row, mask=gm, other=0).to(tl.int64)
    table_row = table_ptr + row.to(tl.int64) * stride_table

    # Bonus = last accepted token; the embedding input appends the mask token.
    bidx = tl.maximum(nacc - 1, 0)
    bonus = tl.load(accepted_ptr + row.to(tl.int64) * stride_accepted + bidx, mask=gm, other=0)
    bonus = bonus.to(tl.int64)
    tl.store(bonus_ptr + g, bonus, mask=gm)
    tl.store(embed_ids_ptr + g, bonus, mask=gm)
    tl.store(embed_ids_ptr + num_gens, mask_token_id.to(tl.int64))

    # Block query positions continue from the advanced context length.
    now = tl.minimum(old + nacc, max_ctx)
    qmask = gm[:, None] & (j[None, :] < BLOCK_SIZE)
    tl.store(
        query_pos_ptr + g[:, None] * BLOCK_SIZE + j[None, :], now[:, None] + j[None, :], mask=qmask
    )

    # Context-latent store of this step's K+1 target tokens: RoPE positions,
    # destination page / offset (column clamped into the allocation) and mask.
    # Under helix (CP_SIZE > 1) global page p lives on rank p % CP_SIZE as its
    # local page p // CP_SIZE; positions on other ranks' pages are skipped (-1).
    wmask = gm[:, None] & (j[None, :] < WIDTH)
    flat = g[:, None] * WIDTH + j[None, :]
    pos = old[:, None] + j[None, :]
    tl.store(ctx_pos_ptr + flat, pos, mask=wmask)
    capacity = counts * PAGE_SIZE
    gpage = pos // PAGE_SIZE
    col = (gpage // CP_SIZE) * PAGE_SIZE + pos % PAGE_SIZE
    col = tl.maximum(tl.minimum(col, capacity[:, None] - 1), 0)
    page_idx = tl.minimum(col // PAGE_SIZE, max_blocks - 1)
    page = tl.load(table_row[:, None] + page_idx, mask=wmask, other=0).to(tl.int64)
    tl.store(store_pages_ptr + flat, page, mask=wmask)
    tl.store(store_offsets_ptr + flat, col % PAGE_SIZE, mask=wmask)
    store = (j[None, :] < nacc[:, None]).to(tl.int8)
    store = tl.where(gpage % CP_SIZE == cp_rank, store, -1)
    tl.store(store_mask_ptr + flat, store, mask=wmask)

    # Advance the context length; the shared padding / dummy slot stays empty.
    # A length clamped at max_ctx moves the history's position offset instead.
    new_len = tl.where(slot == dummy_slot, 0, now)
    if HAS_POSITION_OFFSET:
        live = gm & (slot != dummy_slot)
        offset = tl.load(position_offset_ptr + slot, mask=live, other=0)
        overflow = tl.maximum(old + nacc - max_ctx, 0)
        tl.store(position_offset_ptr + slot, offset + overflow, mask=live)
    tl.debug_barrier()
    tl.store(ctx_len_ptr + slot, new_len, mask=gm)
    tl.store(ctx_len_ptr + dummy_slot, tl.zeros([], dtype=tl.int64))

    # Advertised context, capped so the block still fits in the allocation
    # (CP_SIZE local pages cover CP_SIZE times as many global positions).
    allocated = tl.maximum(capacity * CP_SIZE - BLOCK_SIZE, 0)
    num_ctx = tl.minimum(new_len, allocated)
    tl.store(num_ctx_ptr + g, num_ctx, mask=gm)
    seq_lens = tl.full([BLOCK_G], WIDTH, tl.int32).to(seq_lens_ptr.dtype.element_ty)
    tl.store(seq_lens_ptr + row, seq_lens, mask=gm)

    # Block-decode operands: where the block's own latents go, and its bounds
    # (rank-local slots and lengths under helix, as HelixDraftShard computes).
    bpos = num_ctx[:, None] + j[None, :]
    bgpage = bpos // PAGE_SIZE
    bslot = (bgpage // CP_SIZE) * PAGE_SIZE + bpos % PAGE_SIZE
    bpage = tl.load(
        table_row[:, None] + tl.minimum(bslot // PAGE_SIZE, max_blocks - 1), mask=qmask, other=0
    )
    tl.store(blk_pages_ptr + g[:, None] * BLOCK_SIZE + j[None, :], bpage.to(tl.int64), mask=qmask)
    tl.store(blk_offsets_ptr + g[:, None] * BLOCK_SIZE + j[None, :], bslot % PAGE_SIZE, mask=qmask)
    total = num_ctx + BLOCK_SIZE
    if CP_SIZE > 1:
        tl.store(
            blk_owned_ptr + g[:, None] * BLOCK_SIZE + j[None, :],
            (bgpage % CP_SIZE == cp_rank).to(tl.int8),
            mask=qmask,
        )
        span = PAGE_SIZE * CP_SIZE
        rem = total % span - cp_rank * PAGE_SIZE
        total = (total // span) * PAGE_SIZE + tl.minimum(tl.maximum(rem, 0), PAGE_SIZE)
    seq_len = total.to(tl.int32)
    tl.store(blk_seq_lens_ptr + g, seq_len, mask=gm)
    tl.store(
        blk_kv_bounds_ptr + g[:, None] * BLOCK_SIZE + j[None, :],
        tl.zeros([BLOCK_G, BLOCK_W], dtype=tl.int32) + seq_len[:, None],
        mask=qmask,
    )


def dflash_gen_prepare(
    num_accepted_tokens: torch.Tensor,
    accepted_tokens: torch.Tensor,
    slots: torch.Tensor,
    ctx_len: torch.Tensor,
    block_counts: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_cuda: torch.Tensor,
    num_contexts: int,
    num_gens: int,
    dummy_slot: int,
    max_ctx: int,
    mask_token_id: int,
    page_size: int,
    block_size: int,
    width: int,
    position_offset: Optional[torch.Tensor] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> dict:
    """The DFlash paged-context bookkeeping of one generation step in one launch.

    Computes, for each generation request ``g`` (batch row ``num_contexts + g``),
    exactly what ``DFlashWorker.prepare_1st_drafter_inputs`` and
    ``_build_mla_block_fixup`` compute eagerly when the drafter context lives in
    the manager's paged pool: the bonus token, the block query positions, the
    RoPE positions / destination pages / write mask of the ``width`` target
    latents, the advanced ``ctx_len`` (written in place; the dummy slot is
    zeroed), the advertised context length, ``seq_lens_cuda = width`` and the
    block-decode page / offset / bound operands.

    ``position_offset`` (the unified draft cache's per-slot history offset)
    takes what a length clamped at ``max_ctx`` loses, in place. With
    ``cp_size > 1`` the context pages are helix-sharded (``HelixDraftShard``):
    the store mask is -1 on other ranks' positions, the block operands are
    rank-local and ``blk_owned`` marks the block positions this rank holds.
    """
    device = ctx_len.device
    long = dict(dtype=torch.long, device=device)
    out = dict(
        bonus=torch.empty((num_gens,), **long),
        embed_ids=torch.empty((num_gens + 1,), **long),
        query_positions=torch.empty((num_gens, block_size), **long),
        ctx_positions=torch.empty((num_gens * width,), **long),
        store_pages=torch.empty((num_gens * width,), **long),
        store_offsets=torch.empty((num_gens * width,), **long),
        store_mask=torch.empty((num_gens * width,), dtype=torch.int8, device=device),
        num_ctx=torch.empty((num_gens,), **long),
        blk_pages=torch.empty((num_gens, block_size), **long),
        blk_offsets=torch.empty((num_gens, block_size), **long),
        blk_seq_lens=torch.empty((num_gens,), dtype=torch.int32, device=device),
        blk_kv_bounds=torch.empty((num_gens * block_size,), dtype=torch.int32, device=device),
    )
    if cp_size > 1:
        out["blk_owned"] = torch.empty((num_gens, block_size), dtype=torch.int8, device=device)
    assert accepted_tokens.stride(-1) == 1 and block_tables.stride(-1) == 1
    _dflash_gen_prepare_kernel[(1,)](
        num_accepted_tokens,
        accepted_tokens,
        accepted_tokens.stride(0),
        slots,
        ctx_len,
        block_counts,
        block_tables,
        block_tables.stride(0),
        block_tables.shape[1],
        seq_lens_cuda,
        out["bonus"],
        out["embed_ids"],
        out["query_positions"],
        out["ctx_positions"],
        out["store_pages"],
        out["store_offsets"],
        out["store_mask"],
        out["num_ctx"],
        out["blk_pages"],
        out["blk_offsets"],
        out["blk_seq_lens"],
        out["blk_kv_bounds"],
        out.get("blk_owned", out["store_mask"]),
        ctx_len if position_offset is None else position_offset,
        num_gens,
        num_contexts,
        dummy_slot,
        max_ctx,
        mask_token_id,
        cp_rank,
        PAGE_SIZE=page_size,
        BLOCK_SIZE=block_size,
        WIDTH=width,
        CP_SIZE=cp_size,
        HAS_POSITION_OFFSET=position_offset is not None,
        BLOCK_G=triton.next_power_of_2(max(num_gens, 1)),
        BLOCK_W=triton.next_power_of_2(max(width, block_size)),
        num_warps=4,
    )
    return out


def vocab_shard_embedding(embedding, ids: torch.Tensor) -> Optional[torch.Tensor]:
    """``embedding(ids)`` for a COLUMN (vocab-sharded) TP ``Embedding``: this
    rank's rows of the table, zero for ids outside its shard, then the module's
    all-reduce. One lookup kernel instead of the eager mask / lookup / fill
    chain; same values. Returns None for an embedding this does not cover."""
    from ...common.parallel import TensorParallelMode

    weight = getattr(embedding, "weight", None)
    if (
        getattr(embedding, "tp_size", 1) <= 1
        or getattr(embedding, "tp_mode", None) != TensorParallelMode.COLUMN
        or getattr(embedding, "all_reduce", None) is None
        or weight is None
        or not weight.is_contiguous()
        or ids.dim() != 1
    ):
        return None
    hidden = weight.shape[1]
    out = torch.empty((ids.shape[0], hidden), dtype=weight.dtype, device=weight.device)
    if ids.shape[0] > 0:
        block_h = 1024
        _vocab_shard_embedding_kernel[(ids.shape[0], triton.cdiv(hidden, block_h))](
            ids,
            weight,
            out,
            embedding.vocab_start_index,
            embedding.vocab_end_index,
            hidden,
            BLOCK_H=block_h,
        )
    return embedding.all_reduce(out)
