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

* :func:`decode_draft_block_table` -- the DFlash draft-KV block-table decode
  (blocks per request, clamped and divided block offsets).
* :func:`dflash_paged_latent_prepare` -- the per-request bookkeeping of a
  DFlash generation step whose drafter context is a paged single-latent cache.
* :func:`store_context_latents` -- the masked store of the context latents of
  every drafter layer into its page pool.

Launch shapes depend only on host batch sizes, so all of them are CUDA-graph
capturable.
"""

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from ...ops.common.rubin_ull_features import active as k31_features

# Largest number of generation requests the one-program fused Markov step
# (``dspark_markov_v2``) takes; larger batches keep the unfused, row-parallel
# kernels.
TAIL_V2_MAX_GENS = 16

_BLOCK_TABLE_BLOCK = 2048
# Columns per program of the parallel draft block-table decode.
BLOCK_TABLE_CHUNK = _BLOCK_TABLE_BLOCK


def spec_tail_fusion_enabled() -> bool:
    """Whether the tail fusions that are not on by default for every model run (the
    MLA drafter's V un-absorb written by ``bmm_out``): ``spec_tail_fusion`` (on for every
    K3.1 generation server)."""
    return bool(k31_features().spec_tail_fusion)


def tail_v2_enabled() -> bool:
    """Whether ``dspark_tail_v2`` selects the second set of fused tail
    kernels. They write bitwise the values of the paths they replace: the MLA
    drafter's context and block RoPE kernels with in-kernel table lookups and
    the block latents stored straight into the paged pool, the column-parallel
    draft block-table decode, and shape-only operands built once instead of
    every step."""
    return bool(k31_features().dspark_tail_v2)


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
    # Helix: [G, block, 1] bool, True where this rank's pages hold the block
    # position; pages / offsets and the lengths above are then rank-local.
    owned: Optional[torch.Tensor] = None


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
    blk_owned: Optional[torch.Tensor] = None  # helix: [G, block] int8, 1 where this rank owns it


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
    blk_owned_ptr,
    position_offset_ptr,
    num_gens,
    first_row,
    dummy_slot,
    max_ctx,
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
    # (column clamped into the request's allocation) and write mask. Under helix
    # (CP_SIZE > 1) global page p lives on rank p % CP_SIZE as its local page
    # p // CP_SIZE; positions on other ranks' pages are skipped (mask -1).
    wmask = gm[:, None] & (j[None, :] < WIDTH)
    flat = g[:, None] * WIDTH + j[None, :]
    pos = old[:, None] + j[None, :]
    tl.store(ctx_pos_ptr + flat, pos, mask=wmask)
    gpage = pos // PAGE_SIZE
    col = (gpage // CP_SIZE) * PAGE_SIZE + pos % PAGE_SIZE
    col = tl.maximum(tl.minimum(col, capacity[:, None] - 1), 0)
    page = tl.load(table_row[:, None] + tl.minimum(col // PAGE_SIZE, max_blocks - 1), mask=wmask)
    tl.store(store_pages_ptr + flat, page.to(tl.int64), mask=wmask)
    tl.store(store_offsets_ptr + flat, col % PAGE_SIZE, mask=wmask)
    store = (j[None, :] < nacc[:, None]).to(tl.int8)
    store = tl.where(gpage % CP_SIZE == cp_rank, store, -1)
    tl.store(store_mask_ptr + flat, store, mask=wmask)

    # Advance the context length; the shared dummy slot stays empty. A length
    # clamped at max_ctx moves the history's position offset instead.
    new_len = tl.where(slot == dummy_slot, 0, now)
    if HAS_POSITION_OFFSET:
        live = gm & (slot != dummy_slot)
        offset = tl.load(position_offset_ptr + slot, mask=live, other=0)
        overflow = tl.maximum(old + nacc - max_ctx, 0)
        tl.store(position_offset_ptr + slot, offset + overflow, mask=live)
    # Every lane has read its old length before any lane writes a new one.
    tl.debug_barrier()
    tl.store(ctx_len_ptr + slot, new_len, mask=gm)
    tl.store(ctx_len_ptr + dummy_slot, tl.zeros([], dtype=tl.int64))

    # Advertised context, capped so the block still fits in the allocation
    # (CP_SIZE local pages cover CP_SIZE times as many global positions).
    num_ctx = tl.minimum(new_len, tl.maximum(capacity * CP_SIZE - BLOCK_SIZE, 0))
    tl.store(num_ctx_ptr + g, num_ctx, mask=gm)
    tl.store(
        seq_lens_ptr + row,
        tl.full([BLOCK_G], WIDTH, tl.int32).to(seq_lens_ptr.dtype.element_ty),
        mask=gm,
    )

    # Block-decode operands: where the block's own latents go, and its bounds
    # (rank-local slots and lengths under helix, as HelixDraftShard computes).
    bpos = num_ctx[:, None] + j[None, :]
    bgpage = bpos // PAGE_SIZE
    bslot = (bgpage // CP_SIZE) * PAGE_SIZE + bpos % PAGE_SIZE
    bpage = tl.load(table_row[:, None] + tl.minimum(bslot // PAGE_SIZE, max_blocks - 1), mask=qmask)
    tl.store(blk_pages_ptr + blk, bpage.to(tl.int64), mask=qmask)
    tl.store(blk_offsets_ptr + blk, bslot % PAGE_SIZE, mask=qmask)
    total = num_ctx + BLOCK_SIZE
    if CP_SIZE > 1:
        tl.store(blk_owned_ptr + blk, (bgpage % CP_SIZE == cp_rank).to(tl.int8), mask=qmask)
        span = PAGE_SIZE * CP_SIZE
        rem = total % span - cp_rank * PAGE_SIZE
        total = (total // span) * PAGE_SIZE + tl.minimum(tl.maximum(rem, 0), PAGE_SIZE)
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
    block_counts: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_cuda: torch.Tensor,
    first_row: int,
    dummy_slot: int,
    max_ctx: int,
    page_size: int,
    block_size: int,
    width: int,
    position_offset: Optional[torch.Tensor] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> PagedLatentPrepare:
    """The DFlash generation-step bookkeeping for a paged single-latent context.

    For generation request ``g`` (batch row ``first_row + g``, context slot
    ``slots[g]``) computes what ``DFlashWorker.prepare_1st_drafter_inputs`` and
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
        position_offset: the unified draft cache's ``[slots]`` history position
            offset; takes what a length clamped at ``max_ctx`` loses, in place.
        cp_size, cp_rank: helix sharding of the context pages
            (``HelixDraftShard``). With ``cp_size > 1`` the store mask is -1 on
            other ranks' positions, the block operands are rank-local and
            ``blk_owned`` marks the block positions this rank holds.
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
        blk_owned=(
            torch.empty((num_gens, block_size), dtype=torch.int8, device=device)
            if cp_size > 1
            else None
        ),
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
        *out[:-1],
        out.store_mask if out.blk_owned is None else out.blk_owned,
        ctx_len if position_offset is None else position_offset,
        num_gens,
        first_row,
        dummy_slot,
        max_ctx,
        cp_rank,
        PAGE_SIZE=page_size,
        BLOCK_SIZE=block_size,
        WIDTH=width,
        CP_SIZE=cp_size,
        HAS_POSITION_OFFSET=position_offset is not None,
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
    SKIP_UNACCEPTED: tl.constexpr = False,
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
        # to_draft_fp8: saturated E4M3 at unit scale, round to nearest even.
        vals = tl.minimum(tl.maximum(vals.to(tl.float32), -448.0), 448.0).to(tl.float8e4nv)
    pool = tl.load(pool_ptrs_ptr + layer).to(tl.pointer_type(vals.dtype))
    # A -1 flag leaves the slot untouched (another helix rank owns it); with
    # SKIP_UNACCEPTED a 0 flag does too (the helix8 context fork, dspark_h8.ctx_fork_enabled).
    if SKIP_UNACCEPTED:
        written = flag > 0
    else:
        written = flag >= 0
    tl.store(
        pool + page * stride_pool_page + off * stride_pool_off + offs,
        vals,
        mask=dmask & written,
    )


def store_context_latents(
    latent: torch.Tensor,
    pools: list[torch.Tensor],
    pool_ptrs: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    mask: torch.Tensor,
    skip_unaccepted: bool = False,
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
            slot untouched.
        skip_unaccepted: a 0 flag also leaves the slot untouched. The 0 rows of a
            generation step (the target tokens past the accepted ones) lie in the
            slots the drafter block's own latents take next, so a caller whose
            block store follows may skip them (``dspark_h8.ctx_fork_enabled``).

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
        SKIP_UNACCEPTED=bool(skip_unaccepted),
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


def vocab_shard_embedding(embedding, ids: torch.Tensor) -> Optional[torch.Tensor]:
    """``embedding(ids)`` for a COLUMN (vocab-sharded) TP ``Embedding``: this
    rank's rows of the table, zero for ids outside its shard, then the module's
    all-reduce. One lookup kernel instead of the eager mask / lookup / fill
    chain; same values. Returns None for an embedding this does not cover."""

    weight = getattr(embedding, "weight", None)
    if (
        getattr(embedding, "tp_size", 1) <= 1
        or getattr(embedding, "tp_mode", None) != "column"
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
    # SpecWorker._prepare_next_new_tokens for a batch in one program, plus
    # DFlashWorker._apply_kv_rewind_after_draft on the gen rows.
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
    """The next step's input tokens ``[accepted[i, num_accepted[i] - 1], *next_draft[i]]``
    (``_prepare_next_new_tokens`` with ``batch_indices = arange``) and
    ``kv_lens[num_contexts:][:num_gens] = (kv_lens - rewind).clamp(min=0)`` in one launch
    (one program). Integer copies and arithmetic: the values of the unfused ops."""
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
