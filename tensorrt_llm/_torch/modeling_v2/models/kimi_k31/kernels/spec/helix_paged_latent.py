# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of the DFlash worker's generation-step preparation over one Helix CP rank's context pages.

Global page ``g`` of a request lives on CP rank ``g % CP_SIZE`` as its local page ``g // CP_SIZE``
(``ops/spec/helix_draft_shard.py``). The preparation gives other ranks' positions store flag -1, the store
leaves those slots untouched, and the advertised context is capped at ``capacity * CP_SIZE - BLOCK_SIZE``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.speculative.spec_tail_kernels import PagedLatentPrepare


# The generation count, the first generation row and the slot view sliced at
# that row follow the batch composition. Specializing them (a count of 1 or a
# multiple of 16, a 16-byte-aligned pointer) would compile kernel variants
# mid-serving.
@triton.jit(
    do_not_specialize=["num_gens", "first_row"],
    do_not_specialize_on_alignment=["slots_ptr"],
)
def _helix_paged_latent_prepare_kernel(
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
    """``_paged_latent_prepare_kernel`` of ``speculative/spec_tail_kernels.py`` with the Helix page mapping."""
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

    # Advance the context length; the shared dummy slot stays empty.
    new_len = tl.where(slot == dummy_slot, 0, now)
    # Every lane has read its old length before any lane writes a new one.
    tl.debug_barrier()
    tl.store(ctx_len_ptr + slot, new_len, mask=gm)
    tl.store(ctx_len_ptr + dummy_slot, tl.zeros([], dtype=tl.int64))

    # Advertised context, capped so the block still fits in the allocation
    # (CP_SIZE local pages cover CP_SIZE times as many global positions), then
    # rebased past the slot's base pages (dflash_rebased_ctx_view): the block
    # decode reads row entries from ``rebase`` on.
    num_ctx = tl.minimum(new_len, tl.maximum(capacity * CP_SIZE - BLOCK_SIZE, 0))
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

    # Block-decode operands: where the block's own latents go, and its bounds.
    bpos = num_ctx[:, None] + j[None, :]
    bentry = tl.minimum(rebase[:, None] + bpos // PAGE_SIZE, max_blocks - 1)
    bpage = tl.load(table_row[:, None] + bentry, mask=qmask)
    tl.store(blk_pages_ptr + blk, bpage.to(tl.int64), mask=qmask)
    tl.store(blk_offsets_ptr + blk, bpos % PAGE_SIZE, mask=qmask)
    seq_len = (num_ctx + BLOCK_SIZE).to(tl.int32)
    tl.store(blk_seq_lens_ptr + g, seq_len, mask=gm)
    tl.store(
        blk_kv_bounds_ptr + blk,
        tl.zeros([BLOCK_G, BLOCK_W], tl.int32) + seq_len[:, None],
        mask=qmask,
    )


def helix_paged_latent_prepare(
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
    cp_size: int,
    cp_rank: int,
) -> PagedLatentPrepare:
    """``dflash_paged_latent_prepare`` of ``speculative/spec_tail_kernels.py`` on CP rank ``cp_rank`` of ``cp_size``."""
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
        rebase_pages=torch.empty((num_gens,), **i64),
    )
    if num_gens == 0:
        return out
    assert accepted_tokens.stride(-1) == 1 and block_tables.stride(-1) == 1
    _helix_paged_latent_prepare_kernel[(1,)](
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
def _helix_store_context_latents_kernel(
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
    """``_store_context_latents_kernel`` of ``speculative/spec_tail_kernels.py``; a -1 flag skips the slot."""
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
    pool = tl.load(pool_ptrs_ptr + layer).to(tl.pointer_type(vals.dtype))
    # A -1 flag leaves the slot untouched (another helix rank owns it).
    tl.store(
        pool + page * stride_pool_page + off * stride_pool_off + offs,
        vals,
        mask=dmask & (flag >= 0),
    )


def helix_store_context_latents(
    latent: torch.Tensor,
    pools: list[torch.Tensor],
    pool_ptrs: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """``store_context_latents`` of ``speculative/spec_tail_kernels.py``; ``mask`` -1 leaves the slot untouched."""
    num_tokens, num_layers, head_dim = latent.shape
    if num_tokens == 0:
        return
    pool0 = pools[0]
    assert latent.stride(-1) == 1 and pool0.stride(-1) == 1
    assert all(p.stride() == pool0.stride() and p.dtype == latent.dtype for p in pools)
    _helix_store_context_latents_kernel[(num_tokens, num_layers)](
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
