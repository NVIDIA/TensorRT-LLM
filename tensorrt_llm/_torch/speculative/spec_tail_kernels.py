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
"""Triton kernels for the bookkeeping of a one-model speculative-decoding step.

Each kernel replaces a chain of small eager PyTorch ops with one launch and
produces the same values:

* :func:`strict_accept` -- strict (token-equality) acceptance of the previous
  draft against the sampled target tokens.
* :func:`decode_draft_block_table` -- the DFlash draft-KV block-table decode
  (blocks per request, clamped and divided block offsets).
* :func:`dflash_paged_latent_prepare` -- the per-request bookkeeping of a
  DFlash generation step whose drafter context is a paged single-latent cache.
* :func:`store_context_latents` -- the masked store of the context latents of
  every drafter layer into its page pool.
* :func:`scatter_rows_to_pool` -- rows into a paged pool at given pages and
  offsets, optionally only the rows a mask selects.
* :func:`draft_fp8_cast` -- ``to_draft_fp8`` of a contiguous tensor.

Launch shapes depend only on host batch sizes, so all of them are CUDA-graph
capturable.
"""

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from .dflash_attention import DRAFT_FP8_MAX

_BLOCK_TABLE_BLOCK = 2048
_DRAFT_FP8_MAX = tl.constexpr(DRAFT_FP8_MAX)


@triton.jit
def draft_fp8_saturate(v):
    """``to_draft_fp8`` of FP32 ``v`` in a kernel: clamped to the E4M3 range with NaN kept
    (``tl.minimum`` / ``tl.maximum`` alone would drop it), rounded to nearest even."""
    clamped = tl.minimum(tl.maximum(v, -_DRAFT_FP8_MAX), _DRAFT_FP8_MAX)
    return tl.where(v != v, v, clamped).to(tl.float8e4nv)


# Rows per program of _strict_accept_kernel. Fixed, like the arguments in its do_not_specialize list, so
# one compiled variant per draft length serves every batch: serving changes the request counts on every
# iteration, and a new variant compiles and loads a module on the sampling path.
_STRICT_ACCEPT_BLOCK_B = 128


@triton.jit(
    do_not_specialize=["draft_ptr", "stride_draft_row", "stride_draft_col", "num_ctx", "num_gen"]
)
def _strict_accept_kernel(
    target_ptr,
    draft_ptr,
    stride_draft_row,
    stride_draft_col,
    accepted_ptr,
    num_accepted_ptr,
    num_ctx,
    num_gen,
    DRAFT_LEN: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    width = DRAFT_LEN + 1
    rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)

    # Context requests: accept the sampled token only.
    ctx_mask = rows < num_ctx
    t_ctx = tl.load(target_ptr + rows, mask=ctx_mask, other=0)
    tl.store(accepted_ptr + rows * width, t_ctx.to(tl.int32), mask=ctx_mask)
    tl.store(num_accepted_ptr + rows, tl.full([BLOCK_B], 1, tl.int32), mask=ctx_mask)

    # Generation requests: the accepted row is the K+1 target tokens, and the
    # count is 1 + the number of leading draft tokens equal to them.
    gen_mask = rows < num_gen
    tile_mask = gen_mask[:, None] & (cols[None, :] < width)
    t_gen = tl.load(
        target_ptr + num_ctx + rows[:, None] * width + cols[None, :], mask=tile_mask, other=0
    ).to(tl.int32)
    tl.store(
        accepted_ptr + (num_ctx + rows[:, None]) * width + cols[None, :], t_gen, mask=tile_mask
    )
    draft_mask = gen_mask[:, None] & (cols[None, :] < DRAFT_LEN)
    draft = tl.load(
        draft_ptr + rows[:, None] * stride_draft_row + cols[None, :] * stride_draft_col,
        mask=draft_mask,
        other=0,
    ).to(tl.int32)
    mismatch = draft_mask & (draft != t_gen)
    first_mismatch = tl.min(tl.where(mismatch, cols[None, :], DRAFT_LEN), axis=1)
    tl.store(num_accepted_ptr + num_ctx + rows, (1 + first_mismatch).to(tl.int32), mask=gen_mask)


def strict_accept(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict acceptance of a uniform-width draft in one launch.

    Args:
        target_tokens: ``[num_contexts + num_gens * (K + 1)]`` sampled target
            tokens (any integer dtype).
        draft_tokens: ``[num_gens, K]`` previous draft tokens.
        num_contexts, batch_size: request counts.
    Returns:
        ``accepted_tokens [batch_size, K + 1]`` and ``num_accepted_tokens
        [batch_size]``, both int32. Context rows define only column 0.
    """
    num_gens = batch_size - num_contexts
    draft_len = draft_tokens.shape[-1]
    device = target_tokens.device
    accepted = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=device)
    num_accepted = torch.empty((batch_size,), dtype=torch.int32, device=device)
    if batch_size == 0:
        return accepted, num_accepted
    block_b = _STRICT_ACCEPT_BLOCK_B
    grid = (triton.cdiv(max(num_contexts, num_gens, 1), block_b),)
    _strict_accept_kernel[grid](
        target_tokens,
        draft_tokens,
        draft_tokens.stride(0),
        draft_tokens.stride(1),
        accepted,
        num_accepted,
        num_contexts,
        num_gens,
        DRAFT_LEN=draft_len,
        BLOCK_B=block_b,
        BLOCK_K=triton.next_power_of_2(draft_len + 1),
        num_warps=4,
    )
    return accepted, num_accepted


@triton.jit
def _decode_block_table_kernel(
    src_ptr,
    stride_src_row,
    stride_src_col,
    n_blocks,
    divisor,
    block_offset,
    table_ptr,
    stride_table,
    counts_ptr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    src_row = src_ptr + row.to(tl.int64) * stride_src_row
    table_row = table_ptr + row.to(tl.int64) * stride_table
    extent = tl.zeros([BLOCK], dtype=tl.int64)
    for start in range(0, n_blocks, BLOCK):
        cols = start + offs
        mask = cols < n_blocks
        encoded = tl.load(src_row + cols * stride_src_col, mask=mask, other=-1).to(tl.int64)
        encoded -= block_offset
        extent = tl.maximum(extent, tl.where(encoded >= 0, cols.to(tl.int64) + 1, 0))
        tl.store(table_row + cols, (tl.maximum(encoded, 0) // divisor).to(tl.int32), mask=mask)
    tl.store(counts_ptr + row, tl.max(extent, axis=0))


def decode_draft_block_table(
    src: torch.Tensor,
    divisor: int,
    table: torch.Tensor,
    counts: torch.Tensor,
    *,
    block_offset: int = 0,
) -> None:
    """Decode layer-relative block offsets without modifying ``src``.

    For ``encoded = src - block_offset``, computes
    ``table[i] = encoded[i].clamp(min=0) // divisor`` and ``counts[i]`` = one
    past the last ``j`` with ``encoded[i, j] >= 0`` (0 when none).

    The extent, not the number of backed entries: a sliding-window pool
    releases a row's low entries, and positions are indexed from entry 0, so
    only the extent bounds the positions a request may address.

    Args:
        src: ``[rows, blocks]`` encoded block offsets (any strides).
        divisor: encoded offset -> pool block index divisor.
        table: ``[>= rows, blocks]`` int32 output, unit inner stride.
        counts: ``[>= rows]`` int64 output.
        block_offset: this layer's K-buffer offset within each physical slot.
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
        int(block_offset),
        table,
        table.stride(0),
        counts,
        BLOCK=min(_BLOCK_TABLE_BLOCK, triton.next_power_of_2(n_blocks)),
        num_warps=4,
    )


class DraftBlockPlacement(NamedTuple):
    """Where a paged drafter context places each request's own block.

    Block position ``j`` of request ``g`` goes to page ``pages[g, j]`` at
    ``offsets[g, j]``, right after the ``num_ctx[g]`` context positions; the
    block attends ``seq_lens[g] = num_ctx[g] + block`` keys, repeated per query
    token in ``kv_bounds``.
    """

    pages: torch.Tensor  # [G, block] int64
    offsets: torch.Tensor  # [G, block] int64
    seq_lens: torch.Tensor  # [G] int32
    kv_bounds: torch.Tensor  # [G * block] int32
    # Recorded after the step's context latent store when it runs on a side
    # stream; the drafter waits on it before its first draft-pool write.
    ctx_stored: Optional[torch.cuda.Event] = None
    # Helix-sharded context: [G * block, 1] bool, whether this rank holds each
    # block position; ``pages``, ``offsets`` and the bounds are then rank-local.
    owned: Optional[torch.Tensor] = None


class PagedLatentPrepare(NamedTuple):
    """What :func:`dflash_paged_latent_prepare` computes for ``G`` generation
    requests drafting ``block`` positions from ``width = K + 1`` target tokens."""

    bonus: torch.Tensor  # [G] int64, the last accepted token
    query_positions: torch.Tensor  # [G, block] int64
    ctx_positions: torch.Tensor  # [G * width] int64, RoPE positions of the target latents
    store_pages: torch.Tensor  # [G * width] int64, destination page of each latent
    store_offsets: torch.Tensor  # [G * width] int64, offset inside that page
    store_mask: torch.Tensor  # [G * width] int8, 1 accepted, 0 zeroed, -1 another rank's
    num_ctx: torch.Tensor  # [G] int64, context length the block attends
    blk_pages: torch.Tensor  # [G, block] int64, page of each block position
    blk_offsets: torch.Tensor  # [G, block] int64
    blk_seq_lens: torch.Tensor  # [G] int32, num_ctx + block
    blk_kv_bounds: torch.Tensor  # [G * block] int32, blk_seq_lens per query token
    blk_owned: torch.Tensor  # [G * block] int8, 1 where this rank holds the block position
    rebase_pages: torch.Tensor  # [G] int64, row entries the block decode skips


@triton.jit
def _paged_latent_prepare_kernel(
    num_accepted_ptr,
    accepted_ptr,
    stride_accepted,
    slots_ptr,
    ctx_len_ptr,
    base_pages_ptr,
    block_counts_ptr,
    table_ptr,
    stride_table,
    max_blocks,
    seq_lens_ptr,
    bonus_ptr,
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
    rebase_ptr,
    num_gens,
    first_row,
    dummy_slot,
    max_ctx,
    cp_rank,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    CP_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    g = tl.arange(0, BLOCK_G)
    j = tl.arange(0, BLOCK_W)
    gm = g < num_gens
    row = first_row + g
    nacc = tl.load(num_accepted_ptr + row, mask=gm, other=1).to(tl.int64)
    slot = tl.load(slots_ptr + g, mask=gm, other=0).to(tl.int64)
    old = tl.load(ctx_len_ptr + slot, mask=gm, other=0).to(tl.int64)
    # Global positions the row's pages stand for. Under helix (CP_SIZE > 1)
    # global page p is local page p // CP_SIZE of rank p % CP_SIZE.
    capacity = tl.load(block_counts_ptr + row, mask=gm, other=0).to(tl.int64) * (
        PAGE_SIZE * CP_SIZE
    )
    table_row = table_ptr + row.to(tl.int64) * stride_table

    bonus = tl.load(
        accepted_ptr + row.to(tl.int64) * stride_accepted + tl.maximum(nacc - 1, 0),
        mask=gm,
        other=0,
    )
    tl.store(bonus_ptr + g, bonus.to(tl.int64), mask=gm)

    # The block continues from the context length advanced by this step.
    now = tl.minimum(old + nacc, max_ctx)
    qmask = gm[:, None] & (j[None, :] < BLOCK_SIZE)
    blk = g[:, None] * BLOCK_SIZE + j[None, :]
    tl.store(query_pos_ptr + blk, now[:, None] + j[None, :], mask=qmask)

    # Context latents of this step's target tokens: RoPE position, destination
    # (column clamped into the request's allocation, then rank-local) and write
    # mask (-1 on another rank's position: nothing written).
    wmask = gm[:, None] & (j[None, :] < WIDTH)
    flat = g[:, None] * WIDTH + j[None, :]
    pos = old[:, None] + j[None, :]
    tl.store(ctx_pos_ptr + flat, pos, mask=wmask)
    col = tl.maximum(tl.minimum(pos, capacity[:, None] - 1), 0)
    store = (j[None, :] < nacc[:, None]).to(tl.int8)
    if CP_SIZE > 1:
        gpage = col // PAGE_SIZE
        store = tl.where(gpage % CP_SIZE == cp_rank, store, -1).to(tl.int8)
        col = (gpage // CP_SIZE) * PAGE_SIZE + col % PAGE_SIZE
    page = tl.load(table_row[:, None] + tl.minimum(col // PAGE_SIZE, max_blocks - 1), mask=wmask)
    tl.store(store_pages_ptr + flat, page.to(tl.int64), mask=wmask)
    tl.store(store_offsets_ptr + flat, col % PAGE_SIZE, mask=wmask)
    tl.store(store_mask_ptr + flat, store, mask=wmask)

    # Advance the context length; the shared dummy slot stays empty.
    new_len = tl.where(slot == dummy_slot, 0, now)
    # Every lane has read its old length before any lane writes a new one.
    tl.debug_barrier()
    tl.store(ctx_len_ptr + slot, new_len, mask=gm)
    tl.store(ctx_len_ptr + dummy_slot, tl.zeros([], dtype=tl.int64))

    # Advertised context, capped so the block still fits in the allocation,
    # then rebased past the slot's base pages (dflash_rebased_ctx_view): the
    # block decode reads row entries from ``rebase`` on.
    num_ctx = tl.minimum(new_len, tl.maximum(capacity - BLOCK_SIZE, 0))
    base = tl.load(base_pages_ptr + slot, mask=gm, other=0).to(tl.int64)
    rebase = tl.minimum(base, num_ctx // PAGE_SIZE)
    num_ctx = tl.maximum(num_ctx - base * PAGE_SIZE, 0)
    tl.store(rebase_ptr + g, rebase, mask=gm)
    tl.store(num_ctx_ptr + g, num_ctx, mask=gm)
    tl.store(
        seq_lens_ptr + row,
        tl.full([BLOCK_G], WIDTH, tl.int32).to(seq_lens_ptr.dtype.element_ty),
        mask=gm,
    )

    # Block-decode operands: where the block's own latents go, and its bounds
    # (rank-local slots and lengths under helix, as HelixDraftShard computes).
    bpos = num_ctx[:, None] + j[None, :]
    total = num_ctx + BLOCK_SIZE
    if CP_SIZE > 1:
        bgpage = bpos // PAGE_SIZE
        tl.store(blk_owned_ptr + blk, (bgpage % CP_SIZE == cp_rank).to(tl.int8), mask=qmask)
        bpos = (bgpage // CP_SIZE) * PAGE_SIZE + bpos % PAGE_SIZE
        span = PAGE_SIZE * CP_SIZE
        tail = tl.minimum(tl.maximum(total % span - cp_rank * PAGE_SIZE, 0), PAGE_SIZE)
        total = (total // span) * PAGE_SIZE + tail
    else:
        tl.store(blk_owned_ptr + blk, tl.full([BLOCK_G, BLOCK_W], 1, tl.int8), mask=qmask)
    bentry = tl.minimum(rebase[:, None] + bpos // PAGE_SIZE, max_blocks - 1)
    bpage = tl.load(table_row[:, None] + bentry, mask=qmask)
    tl.store(blk_pages_ptr + blk, bpage.to(tl.int64), mask=qmask)
    tl.store(blk_offsets_ptr + blk, bpos % PAGE_SIZE, mask=qmask)
    seq_len = total.to(tl.int32)
    tl.store(blk_seq_lens_ptr + g, seq_len, mask=gm)
    tl.store(
        blk_kv_bounds_ptr + blk,
        tl.zeros([BLOCK_G, BLOCK_W], tl.int32) + seq_len[:, None],
        mask=qmask,
    )


def dflash_paged_latent_prepare(
    num_accepted_tokens: torch.Tensor,
    accepted_tokens: torch.Tensor,
    slots: torch.Tensor,
    ctx_len: torch.Tensor,
    base_pages: torch.Tensor,
    block_counts: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_cuda: torch.Tensor,
    first_row: int,
    dummy_slot: int,
    max_ctx: int,
    page_size: int,
    block_size: int,
    width: int,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> PagedLatentPrepare:
    """The DFlash generation-step bookkeeping for a paged single-latent context.

    For generation request ``g`` (batch row ``first_row + g``, context slot
    ``slots[g]``) computes what ``DFlashWorker.prepare_1st_drafter_inputs`` and
    ``_build_mla_block_fixup`` compute eagerly, and applies the same side
    effects: ``ctx_len`` advanced in place (the dummy slot zeroed) and
    ``seq_lens_cuda`` of the generation rows set to ``width``.

    With ``cp_size > 1`` the pages are helix-sharded (``HelixDraftShard``): a
    row of ``block_tables`` lists this rank's local pages, ``store_mask`` is -1
    on the other ranks' positions and the block operands are rank-local.

    Args:
        num_accepted_tokens: ``[batch]`` accepted counts.
        accepted_tokens: ``[batch, width]`` accepted tokens, unit inner stride.
        slots: ``[G]`` context slots.
        ctx_len: ``[slots]`` int64 context lengths.
        base_pages: ``[slots]`` int64 pages below each slot's first stored
            position; ``num_ctx`` and the block placement are rebased past them
            and ``rebase_pages`` says by how many row entries.
        block_counts: ``[batch]`` extent of the pages held per batch row
            (``decode_draft_block_table``).
        block_tables: ``[batch, max_blocks]`` int32 page tables, unit inner stride.
        seq_lens_cuda: ``[batch]`` attention sequence lengths.
        first_row: batch row of generation request 0.
        dummy_slot: the shared padding slot.
        max_ctx: context length ceiling.
        page_size, block_size, width: page size, drafted block positions and
            target tokens per request.
        cp_size, cp_rank: the helix CP group the pages are sharded over.
    """
    num_gens = slots.shape[0]
    device = ctx_len.device
    i64 = dict(dtype=torch.int64, device=device)
    out = PagedLatentPrepare(
        bonus=torch.empty((num_gens,), **i64),
        query_positions=torch.empty((num_gens, block_size), **i64),
        ctx_positions=torch.empty((num_gens * width,), **i64),
        store_pages=torch.empty((num_gens * width,), **i64),
        store_offsets=torch.empty((num_gens * width,), **i64),
        store_mask=torch.empty((num_gens * width,), dtype=torch.int8, device=device),
        num_ctx=torch.empty((num_gens,), **i64),
        blk_pages=torch.empty((num_gens, block_size), **i64),
        blk_offsets=torch.empty((num_gens, block_size), **i64),
        blk_seq_lens=torch.empty((num_gens,), dtype=torch.int32, device=device),
        blk_kv_bounds=torch.empty((num_gens * block_size,), dtype=torch.int32, device=device),
        blk_owned=torch.empty((num_gens * block_size,), dtype=torch.int8, device=device),
        rebase_pages=torch.empty((num_gens,), **i64),
    )
    if num_gens == 0:
        return out
    assert accepted_tokens.stride(-1) == 1 and block_tables.stride(-1) == 1
    _paged_latent_prepare_kernel[(1,)](
        num_accepted_tokens,
        accepted_tokens,
        accepted_tokens.stride(0),
        slots,
        ctx_len,
        base_pages,
        block_counts,
        block_tables,
        block_tables.stride(0),
        block_tables.shape[1],
        seq_lens_cuda,
        *out,
        num_gens,
        first_row,
        dummy_slot,
        max_ctx,
        cp_rank,
        PAGE_SIZE=page_size,
        BLOCK_SIZE=block_size,
        WIDTH=width,
        CP_SIZE=cp_size,
        BLOCK_G=triton.next_power_of_2(num_gens),
        BLOCK_W=triton.next_power_of_2(max(width, block_size)),
        num_warps=4,
    )
    return out


@triton.jit
def _store_context_latents_kernel(
    latent_ptr,
    stride_latent_tok,
    stride_latent_layer,
    pool_ptrs_ptr,
    stride_pool_page,
    stride_pool_off,
    pages_ptr,
    offsets_ptr,
    mask_ptr,
    head_dim,
    BLOCK_D: tl.constexpr,
    POOL_FP8: tl.constexpr,
):
    tok = tl.program_id(0)
    layer = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    dmask = offs < head_dim
    page = tl.load(pages_ptr + tok)
    off = tl.load(offsets_ptr + tok)
    flag = tl.load(mask_ptr + tok)
    keep = (flag > 0).to(tl.float32)
    src = latent_ptr + tok.to(tl.int64) * stride_latent_tok + layer * stride_latent_layer
    vals = tl.load(src + offs, mask=dmask, other=0.0)
    # The eager path multiplies by the mask in the latent dtype: x * 1 is x and
    # x * 0 a signed zero either way.
    vals = (vals.to(tl.float32) * keep).to(vals.dtype)
    if POOL_FP8:
        vals = draft_fp8_saturate(vals.to(tl.float32))
    pool = tl.load(pool_ptrs_ptr + layer).to(tl.pointer_type(vals.dtype))
    dst = pool + page * stride_pool_page + off * stride_pool_off
    tl.store(dst + offs, vals, mask=dmask & (flag >= 0))


def store_context_latents(
    latent: torch.Tensor,
    pools: list[torch.Tensor],
    pool_ptrs: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """``pools[l][pages[t], 0, 0, offsets[t]] = latent[t, l] * mask[t]`` for every
    token ``t`` and layer ``l``.

    Args:
        latent: ``[N, L, D]`` per-token, per-layer latents (unit inner stride).
        pools: the ``L`` per-layer ``[pages, 1, 1, page_size, D]`` pool views,
            all with the same strides.
        pool_ptrs: ``[L]`` int64 base addresses of ``pools``.
        pages, offsets: ``[N]`` destination page and offset in the page.
        mask: ``[N]`` int8 write mask; 0 stores a zeroed latent, -1 leaves the
            slot untouched (another helix rank's position).

    An FP8 E4M3 pool stores ``to_draft_fp8`` of the masked latent.
    """
    num_tokens, num_layers, head_dim = latent.shape
    if num_tokens == 0:
        return
    pool0 = pools[0]
    assert latent.stride(-1) == 1 and pool0.stride(-1) == 1
    assert all(p.stride() == pool0.stride() and p.dtype == pool0.dtype for p in pools)
    pool_fp8 = pool0.dtype == torch.float8_e4m3fn
    assert pool_fp8 or pool0.dtype == latent.dtype
    _store_context_latents_kernel[(num_tokens, num_layers)](
        latent,
        latent.stride(0),
        latent.stride(1),
        pool_ptrs,
        pool0.stride(0),
        pool0.stride(3),
        pages,
        offsets,
        mask,
        head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        POOL_FP8=pool_fp8,
        num_warps=4,
    )


@triton.jit
def _scatter_rows_kernel(
    src_ptr,
    stride_src,
    dst_ptr,
    stride_dst_page,
    stride_dst_off,
    pages_ptr,
    offsets_ptr,
    rows_mask_ptr,
    width,
    BLOCK_D: tl.constexpr,
    POOL_FP8: tl.constexpr,
    HAS_ROWS_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < width
    if HAS_ROWS_MASK:
        mask = mask & (tl.load(rows_mask_ptr + row) != 0)
    page = tl.load(pages_ptr + row)
    off = tl.load(offsets_ptr + row)
    vals = tl.load(src_ptr + row.to(tl.int64) * stride_src + offs, mask=mask)
    if POOL_FP8:
        vals = draft_fp8_saturate(vals.to(tl.float32))
    tl.store(dst_ptr + page * stride_dst_page + off * stride_dst_off + offs, vals, mask=mask)


def scatter_rows_to_pool(
    src: torch.Tensor,
    pool: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    rows_mask: Optional[torch.Tensor] = None,
) -> None:
    """``pool[pages[i], offsets[i]] = src[i]`` for a ``[pages, page_size, D]`` pool view.

    The same writes as the eager advanced-index assignment; an FP8 E4M3 pool
    stores ``to_draft_fp8(src[i])``. Rows where ``rows_mask`` (bool or integer
    ``[rows]``) is 0 write nothing, so their slots keep their values.
    """
    rows, width = src.shape
    if rows == 0:
        return
    assert src.stride(-1) == 1 and pool.stride(-1) == 1 and pool.shape[-1] == width
    pool_fp8 = pool.dtype == torch.float8_e4m3fn
    assert pool_fp8 or pool.dtype == src.dtype
    if rows_mask is not None:
        rows_mask = rows_mask.reshape(-1)
        assert rows_mask.numel() == rows and rows_mask.is_contiguous()
        if rows_mask.dtype == torch.bool:
            rows_mask = rows_mask.view(torch.uint8)
    _scatter_rows_kernel[(rows,)](
        src,
        src.stride(0),
        pool,
        pool.stride(0),
        pool.stride(1),
        pages,
        offsets,
        pages if rows_mask is None else rows_mask,
        width,
        BLOCK_D=triton.next_power_of_2(width),
        POOL_FP8=pool_fp8,
        HAS_ROWS_MASK=rows_mask is not None,
        num_warps=4,
    )


@triton.jit
def _draft_fp8_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    v = tl.load(src_ptr + offs, mask=mask).to(tl.float32)
    tl.store(dst_ptr + offs, draft_fp8_saturate(v), mask=mask)


def draft_fp8_cast(x: torch.Tensor) -> torch.Tensor:
    """``to_draft_fp8(x)`` (saturated E4M3 at unit scale) of a contiguous tensor, one launch."""
    assert x.is_contiguous() and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    numel = x.numel()
    if numel:
        block = 4096
        _draft_fp8_kernel[(triton.cdiv(numel, block),)](x, out, numel, BLOCK=block, num_warps=4)
    return out
