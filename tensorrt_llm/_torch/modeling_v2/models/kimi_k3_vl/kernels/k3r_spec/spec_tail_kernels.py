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

Launch shapes depend only on host batch sizes, so all of them are CUDA-graph
capturable.

The kernels below the main ones are fused tails a caller selects with the
target's constants: ``spec_tail_fusion`` (:func:`fused_strict_accept`) and
``dspark_tail_v2`` (fused selection + acceptance, the column-parallel block-table
decode, the MLA drafter's context/block preparation and its context latents on a
side stream). Without them, the main kernels above are what runs.
"""

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

_VP_INT32_MAX = tl.constexpr(2**31 - 1)

_BLOCK_TABLE_BLOCK = 2048


@triton.jit
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
    block_b = min(triton.next_power_of_2(max(num_contexts, num_gens, 1)), 128)
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
    count = tl.zeros([BLOCK], dtype=tl.int64)
    for start in range(0, n_blocks, BLOCK):
        cols = start + offs
        mask = cols < n_blocks
        encoded = tl.load(src_row + cols * stride_src_col, mask=mask, other=-1).to(tl.int64)
        encoded -= block_offset
        count += (encoded >= 0).to(tl.int64)
        tl.store(table_row + cols, (tl.maximum(encoded, 0) // divisor).to(tl.int32), mask=mask)
    tl.store(counts_ptr + row, tl.sum(count, axis=0))


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
    ``counts[i] = (encoded[i] >= 0).sum()`` and
    ``table[i] = encoded[i].clamp(min=0) // divisor``.

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


class PagedLatentPrepare(NamedTuple):
    """What :func:`dflash_paged_latent_prepare` computes for ``G`` generation
    requests drafting ``block`` positions from ``width = K + 1`` target tokens."""

    bonus: torch.Tensor  # [G] int64, the last accepted token
    query_positions: torch.Tensor  # [G, block] int64
    ctx_positions: torch.Tensor  # [G * width] int64, RoPE positions of the target latents
    store_pages: torch.Tensor  # [G * width] int64, destination page of each latent
    store_offsets: torch.Tensor  # [G * width] int64, offset inside that page
    store_mask: torch.Tensor  # [G * width] int8, 1 for accepted tokens
    num_ctx: torch.Tensor  # [G] int64, context length the block attends
    blk_pages: torch.Tensor  # [G, block] int64, page of each block position
    blk_offsets: torch.Tensor  # [G, block] int64
    blk_seq_lens: torch.Tensor  # [G] int32, num_ctx + block
    blk_kv_bounds: torch.Tensor  # [G * block] int32, blk_seq_lens per query token


@triton.jit
def _paged_latent_prepare_kernel(
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
    num_gens,
    first_row,
    dummy_slot,
    max_ctx,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
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
    capacity = tl.load(block_counts_ptr + row, mask=gm, other=0).to(tl.int64) * PAGE_SIZE
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
    # (column clamped into the request's allocation) and write mask.
    wmask = gm[:, None] & (j[None, :] < WIDTH)
    flat = g[:, None] * WIDTH + j[None, :]
    pos = old[:, None] + j[None, :]
    tl.store(ctx_pos_ptr + flat, pos, mask=wmask)
    col = tl.maximum(tl.minimum(pos, capacity[:, None] - 1), 0)
    page = tl.load(table_row[:, None] + tl.minimum(col // PAGE_SIZE, max_blocks - 1), mask=wmask)
    tl.store(store_pages_ptr + flat, page.to(tl.int64), mask=wmask)
    tl.store(store_offsets_ptr + flat, col % PAGE_SIZE, mask=wmask)
    tl.store(store_mask_ptr + flat, (j[None, :] < nacc[:, None]).to(tl.int8), mask=wmask)

    # Advance the context length; the shared dummy slot stays empty.
    new_len = tl.where(slot == dummy_slot, 0, now)
    # Every lane has read its old length before any lane writes a new one.
    tl.debug_barrier()
    tl.store(ctx_len_ptr + slot, new_len, mask=gm)
    tl.store(ctx_len_ptr + dummy_slot, tl.zeros([], dtype=tl.int64))

    # Advertised context, capped so the block still fits in the allocation.
    num_ctx = tl.minimum(new_len, tl.maximum(capacity - BLOCK_SIZE, 0))
    tl.store(num_ctx_ptr + g, num_ctx, mask=gm)
    tl.store(
        seq_lens_ptr + row,
        tl.full([BLOCK_G], WIDTH, tl.int32).to(seq_lens_ptr.dtype.element_ty),
        mask=gm,
    )

    # Block-decode operands: where the block's own latents go, and its bounds.
    bpos = num_ctx[:, None] + j[None, :]
    bpage = tl.load(table_row[:, None] + tl.minimum(bpos // PAGE_SIZE, max_blocks - 1), mask=qmask)
    tl.store(blk_pages_ptr + blk, bpage.to(tl.int64), mask=qmask)
    tl.store(blk_offsets_ptr + blk, bpos % PAGE_SIZE, mask=qmask)
    seq_len = (num_ctx + BLOCK_SIZE).to(tl.int32)
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
    block_counts: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_cuda: torch.Tensor,
    first_row: int,
    dummy_slot: int,
    max_ctx: int,
    page_size: int,
    block_size: int,
    width: int,
) -> PagedLatentPrepare:
    """The DFlash generation-step bookkeeping for a paged single-latent context.

    For generation request ``g`` (batch row ``first_row + g``, context slot
    ``slots[g]``) computes what ``K3DFlashWorker.prepare_1st_drafter_inputs`` and
    ``_build_mla_block_fixup`` compute eagerly, and applies the same side
    effects: ``ctx_len`` advanced in place (the dummy slot zeroed) and
    ``seq_lens_cuda`` of the generation rows set to ``width``.

    Args:
        num_accepted_tokens: ``[batch]`` accepted counts.
        accepted_tokens: ``[batch, width]`` accepted tokens, unit inner stride.
        slots: ``[G]`` context slots.
        ctx_len: ``[slots]`` int64 context lengths.
        block_counts: ``[batch]`` pages held per batch row.
        block_tables: ``[batch, max_blocks]`` int32 page tables, unit inner stride.
        seq_lens_cuda: ``[batch]`` attention sequence lengths.
        first_row: batch row of generation request 0.
        dummy_slot: the shared padding slot.
        max_ctx: context length ceiling.
        page_size, block_size, width: page size, drafted block positions and
            target tokens per request.
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
        PAGE_SIZE=page_size,
        BLOCK_SIZE=block_size,
        WIDTH=width,
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
):
    tok = tl.program_id(0)
    layer = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    dmask = offs < head_dim
    page = tl.load(pages_ptr + tok)
    off = tl.load(offsets_ptr + tok)
    keep = tl.load(mask_ptr + tok).to(tl.float32)
    src = latent_ptr + tok.to(tl.int64) * stride_latent_tok + layer * stride_latent_layer
    vals = tl.load(src + offs, mask=dmask, other=0.0)
    # The eager path multiplies by the mask in the latent dtype: x * 1 is x and
    # x * 0 a signed zero either way.
    vals = (vals.to(tl.float32) * keep).to(vals.dtype)
    pool = tl.load(pool_ptrs_ptr + layer).to(tl.pointer_type(vals.dtype))
    tl.store(pool + page * stride_pool_page + off * stride_pool_off + offs, vals, mask=dmask)


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
        mask: ``[N]`` int8 write mask; 0 stores a zeroed latent.
    """
    num_tokens, num_layers, head_dim = latent.shape
    if num_tokens == 0:
        return
    pool0 = pools[0]
    assert latent.stride(-1) == 1 and pool0.stride(-1) == 1
    assert all(p.stride() == pool0.stride() and p.dtype == latent.dtype for p in pools)
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
        num_warps=4,
    )


# --------------------------------------------------------------------------------------------------
# Fused tails (the target's spec_tail_fusion / dspark_tail_v2 constants select them).
# --------------------------------------------------------------------------------------------------

# Largest number of generation requests the one-program fused tail kernels
# (selection + acceptance, the Markov steps) take; larger batches keep the
# unfused, row-parallel kernels. Measured in the decode graph at 1-4 requests.
TAIL_V2_MAX_GENS = 16

# Columns per program of the parallel draft block-table decode.
BLOCK_TABLE_CHUNK = _BLOCK_TABLE_BLOCK


@triton.jit
def _fused_strict_accept_kernel(
    target_ptr,
    draft_ptr,
    stride_draft,
    accepted_ptr,
    num_accepted_ptr,
    num_ctx,
    num_gen,
    pool_ptr,
    counter_ptr,
    base_total,
    frac,
    max_total,
    DRAFT_LEN: tl.constexpr,
    FORCE: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    POOL_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    width = DRAFT_LEN + 1
    rows = tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)

    # Context requests: accept the sampled token only.
    ctx_mask = rows < num_ctx
    t_ctx = tl.load(target_ptr + rows, mask=ctx_mask, other=0)
    tl.store(accepted_ptr + rows * width, t_ctx.to(tl.int32), mask=ctx_mask)
    tl.store(num_accepted_ptr + rows, tl.full([BLOCK_B], 1, tl.int32), mask=ctx_mask)

    # Generation requests: accepted row = the K+1 target tokens; the accepted
    # length is 1 + the number of leading draft tokens equal to them.
    gen_mask = rows < num_gen
    tile_mask = gen_mask[:, None] & (cols[None, :] < width)
    t_gen = tl.load(
        target_ptr + num_ctx + rows[:, None] * width + cols[None, :], mask=tile_mask, other=0
    ).to(tl.int32)
    tl.store(
        accepted_ptr + (num_ctx + rows[:, None]) * width + cols[None, :], t_gen, mask=tile_mask
    )
    draft_mask = gen_mask[:, None] & (cols[None, :] < DRAFT_LEN)
    d = tl.load(
        draft_ptr + rows[:, None] * stride_draft + cols[None, :], mask=draft_mask, other=0
    ).to(tl.int32)
    mismatch = draft_mask & (d != t_gen)
    first_mismatch = tl.min(tl.where(mismatch, cols[None, :], DRAFT_LEN), axis=1)
    n = 1 + first_mismatch
    if FORCE == 1:
        n = tl.full([BLOCK_B], 0, tl.int32) + base_total
    elif FORCE == 2:
        # Same draw as SpecWorkerBase._apply_force_accepted_tokens: bump the
        # device counter, hash (counter, slot) into the fixed uniform pool.
        counter = tl.load(counter_ptr) + 1
        # Every warp reads the old value before any warp writes the new one.
        tl.debug_barrier()
        tl.store(counter_ptr, counter)
        idx = (counter * COUNTER_STRIDE + rows.to(tl.int64) * SLOT_STRIDE) & POOL_MASK
        rand = tl.load(pool_ptr + idx, mask=gen_mask, other=1.0)
        extra = (rand < frac).to(tl.int32)
        n = tl.minimum(base_total + extra, max_total)
    tl.store(num_accepted_ptr + num_ctx + rows, n.to(tl.int32), mask=gen_mask)


def fused_strict_accept(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
    force: Optional[tuple] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict acceptance in one launch.

    Args:
        target_tokens: ``[num_contexts + num_gens * (K + 1)]`` sampled target
            tokens (any integer dtype).
        draft_tokens: ``[num_gens, K]`` previous draft tokens (unit inner stride).
        num_contexts, batch_size: request counts.
        force: ``None`` for no override, ``(base_total,)`` for a constant
            accepted length, or ``(base_total, frac, max_total, pool, counter,
            counter_stride, slot_stride)`` for the fractional override.
    Returns:
        ``accepted_tokens [batch_size, K + 1]`` int32 and
        ``num_accepted_tokens [batch_size]`` int32, matching
        ``SpecWorkerBase._sample_and_accept_draft_tokens_base`` (context rows
        only define column 0, as there).
    """
    num_gens = batch_size - num_contexts
    draft_len = draft_tokens.shape[-1]
    device = target_tokens.device
    accepted = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=device)
    num_accepted = torch.empty((batch_size,), dtype=torch.int32, device=device)
    if batch_size == 0:
        return accepted, num_accepted
    assert draft_tokens.dim() == 2 and (num_gens == 0 or draft_tokens.stride(-1) == 1)
    force_mode, base_total, frac, max_total = 0, 0, 0.0, 0
    pool, counter, counter_stride, slot_stride, pool_mask = target_tokens, target_tokens, 1, 1, 0
    if force is not None and len(force) == 1:
        force_mode, base_total = 1, int(force[0])
    elif force is not None:
        base_total, frac, max_total, pool, counter, counter_stride, slot_stride = force
        force_mode = 2
        pool_mask = pool.numel() - 1
    _fused_strict_accept_kernel[(1,)](
        target_tokens,
        draft_tokens,
        draft_tokens.stride(0) if num_gens > 0 else draft_len,
        accepted,
        num_accepted,
        num_contexts,
        num_gens,
        pool,
        counter,
        base_total,
        float(frac),
        max_total,
        DRAFT_LEN=draft_len,
        FORCE=force_mode,
        COUNTER_STRIDE=counter_stride,
        SLOT_STRIDE=slot_stride,
        POOL_MASK=pool_mask,
        BLOCK_B=triton.next_power_of_2(max(num_contexts, num_gens, 1)),
        BLOCK_K=triton.next_power_of_2(draft_len + 1),
        num_warps=4,
    )
    return accepted, num_accepted


@triton.jit
def _pair_argmax(vals, idx, axis: tl.constexpr):
    # The index the vocab-parallel _argmax_combine reduction selects, from
    # native reductions: NaN beats every number and the lowest index wins among
    # NaNs; otherwise the largest value wins and the lowest index among equal
    # values (+0.0 == -0.0). The selected index is unique, so it does not depend
    # on the reduction order.
    is_nan = vals != vals
    any_nan = tl.max(is_nan.to(tl.int32), axis=axis) > 0
    vmax = tl.max(tl.where(is_nan, float("-inf"), vals), axis=axis)
    cand = tl.where(
        tl.expand_dims(any_nan, axis), is_nan, (vals == tl.expand_dims(vmax, axis)) & ~is_nan
    )
    return tl.min(tl.where(cand, idx, _VP_INT32_MAX), axis=axis)


@triton.jit
def _select_accept_kernel(
    pairs_ptr,
    stride_row,
    n_slots,
    draft_ptr,
    stride_draft,
    accepted_ptr,
    num_accepted_ptr,
    num_ctx,
    num_gen,
    pool_ptr,
    counter_ptr,
    base_total,
    frac,
    max_total,
    prev_ptr,
    stride_prev,
    state_idx_ptr,
    dummy_ptr,
    kv_lens_ptr,
    rewind_ptr,
    DRAFT_LEN: tl.constexpr,
    FORCE: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    POOL_MASK: tl.constexpr,
    KDA_REPLAY: tl.constexpr,
    KV_REWIND: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    # Pairs are (value, index) float32 couples, slot s of a row at columns
    # [2 s, 2 s + 1]; a row is loaded as one contiguous run and split.
    width = DRAFT_LEN + 1
    rows = tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)
    gen_mask = rows < num_gen
    tile_mask = gen_mask[:, None] & (cols[None, :] < width)
    draft_mask = gen_mask[:, None] & (cols[None, :] < DRAFT_LEN)

    # Everything that does not depend on the selection is read first.
    d = tl.load(
        draft_ptr + rows[:, None] * stride_draft + cols[None, :], mask=draft_mask, other=0
    ).to(tl.int32)
    if KDA_REPLAY:
        slot = tl.load(state_idx_ptr + num_ctx + rows, mask=gen_mask, other=0).to(tl.int64)
        dummy = tl.load(dummy_ptr + num_ctx + rows, mask=gen_mask, other=0) != 0
        current = tl.load(prev_ptr + slot * stride_prev, mask=gen_mask, other=0)
    if KV_REWIND:
        kv = tl.load(kv_lens_ptr + num_ctx + rows, mask=gen_mask)
    if FORCE == 2:
        counter = tl.load(counter_ptr) + 1
        idx = (counter * COUNTER_STRIDE + rows.to(tl.int64) * SLOT_STRIDE) & POOL_MASK
        rand = tl.load(pool_ptr + idx, mask=gen_mask, other=1.0)
        # Every warp reads the old value before any warp writes the new one.
        tl.debug_barrier()
        tl.store(counter_ptr, counter)

    lanes = tl.arange(0, 2 * BLOCK_SLOTS)
    lane_ok = lanes < 2 * n_slots

    # Context requests: accept the selected token only.
    c = tl.arange(0, BLOCK_C)
    cmask = c < num_ctx
    cm2 = cmask[:, None] & lane_ok[None, :]
    craw = tl.load(
        pairs_ptr + c[:, None].to(tl.int64) * stride_row + lanes[None, :], mask=cm2, other=0.0
    )
    cvals, cidxf = tl.split(tl.reshape(craw, [BLOCK_C, BLOCK_SLOTS, 2]))
    cok = cmask[:, None] & (tl.arange(0, BLOCK_SLOTS)[None, :] < n_slots)
    t_ctx = _pair_argmax(
        tl.where(cok, cvals, float("-inf")), tl.where(cok, cidxf.to(tl.int32), _VP_INT32_MAX), 1
    )
    tl.store(accepted_ptr + c * width, t_ctx, mask=cmask)
    tl.store(num_accepted_ptr + c, tl.full([BLOCK_C], 1, tl.int32), mask=cmask)

    # Generation requests: the K+1 selected tokens of each request (the global
    # argmax / draw of every row's exchanged pairs, as the vocab-parallel
    # selection reduces them), then strict acceptance as _strict_accept_kernel.
    prow = num_ctx + rows[:, None] * width + cols[None, :]
    m3 = tile_mask[:, :, None] & lane_ok[None, None, :]
    graw = tl.load(
        pairs_ptr + prow[:, :, None].to(tl.int64) * stride_row + lanes[None, None, :],
        mask=m3,
        other=0.0,
    )
    gvals, gidxf = tl.split(tl.reshape(graw, [BLOCK_B, BLOCK_K, BLOCK_SLOTS, 2]))
    gok = tile_mask[:, :, None] & (tl.arange(0, BLOCK_SLOTS)[None, None, :] < n_slots)
    t_gen = _pair_argmax(
        tl.where(gok, gvals, float("-inf")), tl.where(gok, gidxf.to(tl.int32), _VP_INT32_MAX), 2
    )
    tl.store(
        accepted_ptr + (num_ctx + rows[:, None]) * width + cols[None, :], t_gen, mask=tile_mask
    )
    mismatch = draft_mask & (d != t_gen)
    first_mismatch = tl.min(tl.where(mismatch, cols[None, :], DRAFT_LEN), axis=1)
    n = 1 + first_mismatch
    if FORCE == 1:
        n = tl.full([BLOCK_B], 0, tl.int32) + base_total
    elif FORCE == 2:
        extra = (rand < frac).to(tl.int32)
        n = tl.minimum(base_total + extra, max_total)
    n = n.to(tl.int32)
    tl.store(num_accepted_ptr + num_ctx + rows, n, mask=gen_mask)

    if KDA_REPLAY:
        # KDAReplayState.record_acceptance: record the accepted draft prefix per
        # state slot; dummy requests keep the slot's current value. All slots
        # were read before any is written, as gather-then-put does.
        accepted_drafts = tl.maximum(n - 1, 0).to(current.dtype)
        tl.debug_barrier()
        tl.store(
            prev_ptr + slot * stride_prev,
            tl.where(dummy, current, accepted_drafts),
            mask=gen_mask,
        )
    if KV_REWIND:
        # K3DFlashWorker._prepare_kv_for_draft_forward.
        tl.store(rewind_ptr + rows, 1 - n, mask=gen_mask)
        tl.store(kv_lens_ptr + num_ctx + rows, kv + 1, mask=gen_mask)


def fused_select_accept(
    pairs: tuple,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
    force: Optional[tuple] = None,
    kda_replay: Optional[tuple] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Vocab-parallel target selection, strict acceptance and the per-step
    bookkeeping that follows it, in one launch.

    Args:
        pairs: ``(pairs, row stride, slot stride, slots)`` from
            ``K3VocabShardedLogits.select_pairs``: every rank's ``(value, global
            index)`` candidates of the ``num_contexts + num_gens * (K + 1)`` rows.
        draft_tokens, num_contexts, batch_size, force: as :func:`fused_strict_accept`.
        kda_replay: ``(prev_num_accepted_tokens, state_indices, dummy_mask)`` to
            also apply ``KDAReplayState.record_acceptance``, or None.
            ``prev_num_accepted_tokens`` is a 1-D int32 view with any slot stride
            (the replay count is the first int32 of each slot's record);
            ``state_indices`` and ``dummy_mask`` cover the whole batch, contexts first.
        kv_lens: ``attn_metadata.kv_lens_cuda`` to also apply
            ``K3DFlashWorker._prepare_kv_for_draft_forward`` (its gen rows += 1), or None.
    Returns:
        ``accepted_tokens [batch_size, K + 1]`` int32, ``num_accepted_tokens
        [batch_size]`` int32 (the values of the reduce + :func:`fused_strict_accept`
        path) and, with ``kv_lens``, the ``[num_gens]`` int32 KV rewind amounts
        ``1 - num_accepted``.
    """
    pair_buf, stride_row, stride_slot, n_slots = pairs
    num_gens = batch_size - num_contexts
    draft_len = draft_tokens.shape[-1]
    device = pair_buf.device
    accepted = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=device)
    num_accepted = torch.empty((batch_size,), dtype=torch.int32, device=device)
    rewind = (
        torch.empty((num_gens,), dtype=torch.int32, device=device) if kv_lens is not None else None
    )
    if batch_size == 0:
        return accepted, num_accepted, rewind
    assert draft_tokens.dim() == 2 and (num_gens == 0 or draft_tokens.stride(-1) == 1)
    force_mode, base_total, frac, max_total = 0, 0, 0.0, 0
    pool, counter, counter_stride, slot_stride, pool_mask = pair_buf, pair_buf, 1, 1, 0
    if force is not None and len(force) == 1:
        force_mode, base_total = 1, int(force[0])
    elif force is not None:
        base_total, frac, max_total, pool, counter, counter_stride, slot_stride = force
        force_mode = 2
        pool_mask = pool.numel() - 1
    prev, state_idx, dummy = kda_replay if kda_replay is not None else (accepted,) * 3
    if kda_replay is not None:
        assert prev.dim() == 1 and prev.dtype == torch.int32
        assert state_idx.stride(-1) == 1 and dummy.stride(-1) == 1
    # Interleaved (value, index) slots, as the MNNVL / single-rank exchange lays them out.
    assert stride_slot == 2 and pair_buf.stride(-1) == 1
    _select_accept_kernel[(1,)](
        pair_buf,
        stride_row,
        n_slots,
        draft_tokens,
        draft_tokens.stride(0) if num_gens > 0 else draft_len,
        accepted,
        num_accepted,
        num_contexts,
        num_gens,
        pool,
        counter,
        base_total,
        float(frac),
        max_total,
        prev,
        prev.stride(0),
        state_idx,
        dummy,
        kv_lens if kv_lens is not None else accepted,
        rewind if rewind is not None else accepted,
        DRAFT_LEN=draft_len,
        FORCE=force_mode,
        COUNTER_STRIDE=counter_stride,
        SLOT_STRIDE=slot_stride,
        POOL_MASK=pool_mask,
        KDA_REPLAY=kda_replay is not None,
        KV_REWIND=kv_lens is not None,
        BLOCK_C=triton.next_power_of_2(max(num_contexts, 1)),
        BLOCK_B=triton.next_power_of_2(max(num_gens, 1)),
        BLOCK_K=triton.next_power_of_2(draft_len + 1),
        BLOCK_SLOTS=triton.next_power_of_2(n_slots),
        num_warps=4,
    )
    return accepted, num_accepted, rewind


@triton.jit
def _decode_block_table_chunk_kernel(
    src_ptr,
    stride_src,
    stride_src_col,
    n_blocks,
    divisor,
    block_offset,
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
    encoded -= block_offset
    decoded = tl.maximum(encoded, 0) // divisor
    tl.store(table_ptr + row.to(tl.int64) * stride_table + cols, decoded.to(tl.int32), mask=mask)
    count = tl.sum((encoded >= 0).to(tl.int32), axis=0)
    tl.store(partial_ptr + row * tl.num_programs(1) + chunk, count)


@triton.jit
def _sum_block_counts_kernel(
    partial_ptr, n_chunks, counts_ptr, rows, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr
):
    r = tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)
    mask = (r[:, None] < rows) & (c[None, :] < n_chunks)
    partial = tl.load(partial_ptr + r[:, None] * n_chunks + c[None, :], mask=mask, other=0)
    tl.store(counts_ptr + r, tl.sum(partial.to(tl.int64), axis=1), mask=r < rows)


def decode_draft_block_table_parallel(
    src: torch.Tensor,
    divisor: int,
    table: torch.Tensor,
    counts: torch.Tensor,
    partial: torch.Tensor,
    *,
    block_offset: int = 0,
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
        int(block_offset),
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
def _ctx_positions_kernel(
    ctx_len_ptr, slots_ptr, out_ptr, num_gens, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    mask = i < num_gens * WIDTH
    g = i // WIDTH
    slot = tl.load(slots_ptr + g, mask=mask, other=0).to(tl.int64)
    old = tl.load(ctx_len_ptr + slot, mask=mask, other=0).to(tl.int64)
    tl.store(out_ptr + i, old + (i - g * WIDTH), mask=mask)


def dflash_ctx_positions(ctx_len: torch.Tensor, slots: torch.Tensor, width: int) -> torch.Tensor:
    """``ctx_len[slots[g]] + j`` for the ``width`` tokens ``j`` of each request
    ``g``: the RoPE positions :func:`dflash_gen_prepare` assigns the step's
    context latents (it reads ``ctx_len`` before advancing it)."""
    num_gens = slots.shape[0]
    out = torch.empty((num_gens * width,), dtype=torch.long, device=ctx_len.device)
    if num_gens > 0:
        _ctx_positions_kernel[(1,)](
            ctx_len,
            slots,
            out,
            num_gens,
            WIDTH=width,
            BLOCK=triton.next_power_of_2(num_gens * width),
            num_warps=1,
        )
    return out


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
    new_len = tl.where(slot == dummy_slot, 0, now)
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
    cp_size: int = 1,
    cp_rank: int = 0,
) -> dict:
    """The DFlash paged-context bookkeeping of one generation step in one launch.

    Computes, for each generation request ``g`` (batch row ``num_contexts + g``),
    exactly what ``K3DFlashWorker.prepare_1st_drafter_inputs`` and
    ``_build_mla_block_fixup`` compute eagerly when the drafter context lives in
    the manager's paged pool: the bonus token, the block query positions, the
    RoPE positions / destination pages / write mask of the ``width`` target
    latents, the advanced ``ctx_len`` (written in place; the dummy slot is
    zeroed), the advertised context length, ``seq_lens_cuda = width`` and the
    block-decode page / offset / bound operands.

    With ``cp_size > 1`` the context pages are helix-sharded (``HelixDraftShard``):
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
        BLOCK_G=triton.next_power_of_2(max(num_gens, 1)),
        BLOCK_W=triton.next_power_of_2(max(width, block_size)),
        num_warps=4,
    )
    return out


@triton.jit
def _noise_embedding_kernel(
    embed_ptr, out_ptr, num_gens, hidden, BLOCK_SIZE: tl.constexpr, BLOCK_H: tl.constexpr
):
    g = tl.program_id(0)
    j = tl.program_id(1)
    offs = tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < hidden
    src_row = tl.where(j == 0, g, num_gens).to(tl.int64)
    vals = tl.load(embed_ptr + src_row * hidden + offs, mask=mask)
    tl.store(out_ptr + (g.to(tl.int64) * BLOCK_SIZE + j) * hidden + offs, vals, mask=mask)


def dflash_noise_embedding(combined_embed: torch.Tensor, block_size: int) -> torch.Tensor:
    """``[G, block, H]``: the mask-token embedding everywhere and the bonus
    embedding at block slot 0, from ``combined_embed = embed([bonus..., mask])``."""
    num_gens = combined_embed.shape[0] - 1
    hidden = combined_embed.shape[-1]
    combined_embed = combined_embed.contiguous()
    out = torch.empty(
        (num_gens, block_size, hidden), dtype=combined_embed.dtype, device=combined_embed.device
    )
    if num_gens > 0:
        block_h = 1024
        _noise_embedding_kernel[(num_gens, block_size, triton.cdiv(hidden, block_h))](
            combined_embed, out, num_gens, hidden, BLOCK_SIZE=block_size, BLOCK_H=block_h
        )
    return out


@triton.jit
def _store_mla_context_kernel(
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
    # Same bf16 multiply as the eager k.mul_(mask): exact for x * 1, signed zero for x * 0.
    vals = (vals.to(tl.float32) * keep).to(vals.dtype)
    if POOL_FP8:
        # to_draft_fp8: saturated E4M3 at unit scale, round to nearest even.
        vals = tl.minimum(tl.maximum(vals.to(tl.float32), -448.0), 448.0).to(tl.float8e4nv)
    pool = tl.load(pool_ptrs_ptr + layer).to(tl.pointer_type(vals.dtype))
    dst = pool + page * stride_pool_page + off * stride_pool_off
    tl.store(dst + offs, vals, mask=dmask & (flag >= 0))


def store_mla_context_latents(
    latent: torch.Tensor,
    pools: list,
    pool_ptrs: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """``pools[l][pages[t], 0, 0, offsets[t]] = latent[t, l] * mask[t]`` for all t, l.

    Args:
        latent: ``[N, L, D]`` per-token, per-layer MLA latents (any token and
            layer strides, unit inner stride).
        pools: the L per-layer ``[pages, 1, 1, page_size, D]`` pool views
            (identical strides).
        pool_ptrs: ``[L]`` int64 base addresses of ``pools``.
        pages, offsets: ``[N]`` destination page and in-page offset.
        mask: ``[N]`` int8 write mask (0 writes a zeroed latent, as eagerly;
            -1 leaves the slot untouched).

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
    _store_mla_context_kernel[(num_tokens, num_layers)](
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
        # to_draft_fp8: saturated E4M3 at unit scale, round to nearest even.
        vals = tl.minimum(tl.maximum(vals.to(tl.float32), -448.0), 448.0).to(tl.float8e4nv)
    tl.store(dst_ptr + page * stride_dst_page + off * stride_dst_off + offs, vals, mask=mask)


def scatter_rows_to_pool(
    src: torch.Tensor,
    pool: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    rows_mask: Optional[torch.Tensor] = None,
) -> None:
    """``pool[pages[i], offsets[i]] = src[i]`` for a ``[pages, page_size, D]`` pool view.

    The same writes as the eager advanced-index assignment (a pure copy); an
    FP8 E4M3 pool stores ``to_draft_fp8(src[i])``. With ``rows_mask`` (bool or
    integer ``[rows]``), rows where it is 0 / False write nothing: the slot
    keeps its value, as ``torch.where(rows_mask, src, pool[pages, offsets])``
    written back would leave it.
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
def _next_new_tokens_kernel(
    accepted_ptr,
    stride_accepted,
    num_accepted_ptr,
    draft_ptr,
    stride_draft,
    out_ptr,
    batch_size,
    DRAFT_LEN: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)
    rmask = rows < batch_size
    nacc = tl.load(num_accepted_ptr + rows, mask=rmask, other=1)
    last = tl.load(accepted_ptr + rows.to(tl.int64) * stride_accepted + (nacc - 1), mask=rmask)
    width = DRAFT_LEN + 1
    tl.store(out_ptr + rows * width, last.to(out_ptr.dtype.element_ty), mask=rmask)
    dmask = rmask[:, None] & (cols[None, :] < DRAFT_LEN)
    d = tl.load(draft_ptr + rows[:, None].to(tl.int64) * stride_draft + cols[None, :], mask=dmask)
    tl.store(
        out_ptr + rows[:, None] * width + 1 + cols[None, :],
        d.to(out_ptr.dtype.element_ty),
        mask=dmask,
    )


@triton.jit
def _next_new_tokens_rewind_kernel(
    accepted_ptr,
    stride_accepted,
    num_accepted_ptr,
    draft_ptr,
    stride_draft,
    out_ptr,
    batch_size,
    kv_lens_ptr,
    rewind_ptr,
    num_ctx,
    num_gen,
    DRAFT_LEN: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # _next_new_tokens_kernel for a batch in one program, plus
    # K3DFlashWorker._apply_kv_rewind_after_draft on the gen rows.
    rows = tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)
    rmask = rows < batch_size
    nacc = tl.load(num_accepted_ptr + rows, mask=rmask, other=1)
    last = tl.load(accepted_ptr + rows.to(tl.int64) * stride_accepted + (nacc - 1), mask=rmask)
    width = DRAFT_LEN + 1
    tl.store(out_ptr + rows * width, last.to(out_ptr.dtype.element_ty), mask=rmask)
    dmask = rmask[:, None] & (cols[None, :] < DRAFT_LEN)
    d = tl.load(draft_ptr + rows[:, None].to(tl.int64) * stride_draft + cols[None, :], mask=dmask)
    tl.store(
        out_ptr + rows[:, None] * width + 1 + cols[None, :],
        d.to(out_ptr.dtype.element_ty),
        mask=dmask,
    )
    gmask = rows < num_gen
    kv = tl.load(kv_lens_ptr + num_ctx + rows, mask=gmask)
    amount = tl.load(rewind_ptr + rows, mask=gmask)
    tl.store(kv_lens_ptr + num_ctx + rows, tl.maximum(kv - amount, 0), mask=gmask)


def fused_next_new_tokens_rewind(
    accepted_tokens: torch.Tensor,
    next_draft_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    kv_lens: torch.Tensor,
    rewind: torch.Tensor,
    num_contexts: int,
) -> torch.Tensor:
    """:func:`fused_next_new_tokens` and ``kv_lens[num_contexts:][:num_gens] =
    (kv_lens - rewind).clamp(min=0)`` in one launch (one program)."""
    batch_size, draft_len = next_draft_tokens.shape
    num_gens = rewind.shape[0]
    dtype = torch.promote_types(accepted_tokens.dtype, next_draft_tokens.dtype)
    out = torch.empty((batch_size, draft_len + 1), dtype=dtype, device=accepted_tokens.device)
    if batch_size == 0 and num_gens == 0:
        return out
    assert accepted_tokens.stride(-1) == 1 and next_draft_tokens.stride(-1) == 1
    assert kv_lens.stride(-1) == 1 and rewind.is_contiguous()
    _next_new_tokens_rewind_kernel[(1,)](
        accepted_tokens,
        accepted_tokens.stride(0),
        num_accepted_tokens,
        next_draft_tokens,
        next_draft_tokens.stride(0),
        out,
        batch_size,
        kv_lens,
        rewind,
        num_contexts,
        num_gens,
        DRAFT_LEN=draft_len,
        BLOCK_B=triton.next_power_of_2(max(batch_size, num_gens, 1)),
        BLOCK_K=triton.next_power_of_2(max(draft_len, 1)),
        num_warps=4,
    )
    return out


def fused_next_new_tokens(
    accepted_tokens: torch.Tensor,
    next_draft_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    """``cat([accepted[b, nacc[b] - 1], next_draft[b]])`` per request, in one launch
    (``SpecWorkerBase._prepare_next_new_tokens`` with identity batch indices)."""
    batch_size, draft_len = next_draft_tokens.shape
    dtype = torch.promote_types(accepted_tokens.dtype, next_draft_tokens.dtype)
    out = torch.empty((batch_size, draft_len + 1), dtype=dtype, device=accepted_tokens.device)
    if batch_size == 0:
        return out
    assert accepted_tokens.stride(-1) == 1 and next_draft_tokens.stride(-1) == 1
    block_b = 128
    _next_new_tokens_kernel[(triton.cdiv(batch_size, block_b),)](
        accepted_tokens,
        accepted_tokens.stride(0),
        num_accepted_tokens,
        next_draft_tokens,
        next_draft_tokens.stride(0),
        out,
        batch_size,
        DRAFT_LEN=draft_len,
        BLOCK_B=block_b,
        BLOCK_K=triton.next_power_of_2(max(draft_len, 1)),
        num_warps=4,
    )
    return out


@triton.jit
def _vocab_shard_embedding_kernel(
    ids_ptr, weight_ptr, out_ptr, vocab_start, vocab_end, hidden, BLOCK_H: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < hidden
    token = tl.load(ids_ptr + row).to(tl.int64)
    local = (token >= vocab_start) & (token < vocab_end)
    src = tl.where(local, token - vocab_start, 0)
    vals = tl.load(weight_ptr + src * hidden + offs, mask=mask & local, other=0.0)
    tl.store(out_ptr + row.to(tl.int64) * hidden + offs, vals, mask=mask)


def vocab_shard_embedding_applies(embedding, ids: torch.Tensor) -> bool:
    """Whether ``embedding(ids)`` of a vocab-parallel ``layers.EmbeddingTable`` can run
    as :func:`vocab_shard_lookup` of this rank's rows followed by the table's all-reduce."""
    weight = getattr(embedding, "weight", None)
    return (
        getattr(embedding, "vocab_parallel", False)
        and getattr(embedding, "all_reduce", None) is not None
        and weight is not None
        and weight.is_contiguous()
        and ids.dim() == 1
    )


def vocab_shard_lookup(
    ids: torch.Tensor, weight: torch.Tensor, vocab_start: int, vocab_end: int
) -> torch.Tensor:
    """This rank's rows ``weight[ids - vocab_start]`` of a vocab shard ``[vocab_start,
    vocab_end)``, zero for ids outside it: one lookup kernel instead of the eager mask / lookup /
    fill chain, same values. The table's all-reduce of the result is ``embedding(ids)``."""
    hidden = weight.shape[1]
    out = torch.empty((ids.shape[0], hidden), dtype=weight.dtype, device=weight.device)
    if ids.shape[0] > 0:
        block_h = 1024
        _vocab_shard_embedding_kernel[(ids.shape[0], triton.cdiv(hidden, block_h))](
            ids,
            weight,
            out,
            vocab_start,
            vocab_end,
            hidden,
            BLOCK_H=block_h,
        )
    return out
