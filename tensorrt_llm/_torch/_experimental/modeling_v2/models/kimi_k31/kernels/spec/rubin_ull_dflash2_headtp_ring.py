# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernel of the head-parallel DFlash2 drafter's ring (``ops/spec/rubin_ull_dflash2_headtp_ring.py``):
one generation step's ring targets in one launch.

Request ``g`` (batch row ``rows[g]``) has context slot ``slot_table[rows[g]]``; its verify latents
sit at positions ``ctx_positions[g * width + i]`` and its block at ``num_ctx[g] + j``. Position
``p`` of slot ``s`` is ring page ``s * pages + (p mod R) // page``, offset ``p mod page`` (``R =
pages * page``). Writes the slots and the two (page, offset) lists.
"""

import triton
import triton.language as tl


@triton.jit
def _step_targets(
    slot_table_ptr,
    rows_ptr,
    ctx_pos_ptr,
    num_ctx_ptr,
    slots_ptr,
    v_pages_ptr,
    v_offs_ptr,
    b_pages_ptr,
    b_offs_ptr,
    groups,
    pages,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    BG: tl.constexpr,
    BW: tl.constexpr,
):
    g = tl.arange(0, BG)
    gm = g < groups
    row = tl.load(rows_ptr + g, mask=gm, other=0).to(tl.int64)
    slot = tl.load(slot_table_ptr + row, mask=gm, other=0).to(tl.int64)
    tl.store(slots_ptr + g, slot, mask=gm)
    ring_rows = pages * PAGE
    base = slot * pages
    j = tl.arange(0, BW)
    vm = gm[:, None] & (j[None, :] < WIDTH)
    vi = g[:, None] * WIDTH + j[None, :]
    pos = tl.load(ctx_pos_ptr + vi, mask=vm, other=0).to(tl.int64)
    r = pos % ring_rows
    tl.store(v_pages_ptr + vi, base[:, None] + r // PAGE, mask=vm)
    tl.store(v_offs_ptr + vi, r % PAGE, mask=vm)
    bm = gm[:, None] & (j[None, :] < BLOCK)
    bi = g[:, None] * BLOCK + j[None, :]
    n = tl.load(num_ctx_ptr + g, mask=gm, other=0).to(tl.int64)
    r = (n[:, None] + j[None, :]) % ring_rows
    tl.store(b_pages_ptr + bi, base[:, None] + r // PAGE, mask=bm)
    tl.store(b_offs_ptr + bi, r % PAGE, mask=bm)


def step_targets(
    slot_table, rows, ctx_positions, num_ctx, slots, v_pages, v_offs, b_pages, b_offs, pages, page
):
    """``slots [G]``, ``v_pages / v_offs [G * width]``, ``b_pages / b_offs [G * block]`` (int64) of
    ``G = rows.shape[0]`` requests; ``width = ctx_positions.numel() // G``, ``block =
    b_pages.numel() // G``."""
    groups = rows.shape[0]
    width = ctx_positions.numel() // groups
    block = b_pages.numel() // groups
    _step_targets[(1,)](
        slot_table,
        rows,
        ctx_positions,
        num_ctx,
        slots,
        v_pages,
        v_offs,
        b_pages,
        b_offs,
        groups,
        pages,
        PAGE=page,
        WIDTH=width,
        BLOCK=block,
        BG=triton.next_power_of_2(groups),
        BW=triton.next_power_of_2(max(width, block)),
        num_warps=1,
    )


# ---- first-GEN fill ----------------------------------------------------------------------------
#
# The window ``[first, L)`` (``first = max(0, L - W)``) of each newly generating request, from the
# history pages to every CP rank's ring. CP rank ``r`` owns the global pages ``g`` with ``g % cp ==
# r`` (local page ``g // cp`` of its page table). Its share has ``capacity = pages_share * PAGE``
# entries: entry ``i`` is offset ``i % PAGE`` of the ``i // PAGE``-th page it owns from ``first //
# PAGE`` on, a position of the window or a hole. The pack writes each rank's share as bytes, the CP
# group all-gathers the shares, and the scatter writes every rank's share into this rank's ring;
# both compute the positions from ``(first, L, r)``, so only the latents travel.


@triton.jit
def _share_position(length, i, r, WINDOW: tl.constexpr, PAGE: tl.constexpr, CP: tl.constexpr):
    first = tl.maximum(length - WINDOW, 0)
    p0 = first // PAGE
    g = p0 + (r - p0 % CP + CP) % CP + (i // PAGE) * CP
    pos = g * PAGE + i % PAGE
    return pos, g, (pos >= first) & (pos < length)


@triton.jit
def _fill_pack(
    meta_ptr,
    stride_meta,
    pool_ptrs_ptr,
    stride_pool_page,
    stride_pool_off,
    wire_ptr,
    cp_rank,
    CAPACITY: tl.constexpr,
    LAYERS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK_B: tl.constexpr,
    WINDOW: tl.constexpr,
    PAGE: tl.constexpr,
    CP: tl.constexpr,
):
    g = tl.program_id(0)
    i = tl.program_id(1)
    layer = tl.program_id(2)
    meta = meta_ptr + g.to(tl.int64) * stride_meta
    length = tl.load(meta + 1).to(tl.int64)
    pos, gpage, valid = _share_position(length, i, cp_rank, WINDOW, PAGE, CP)
    local = tl.load(meta + 2 + gpage // CP, mask=valid, other=0).to(tl.int64)
    b = tl.arange(0, BLOCK_B)
    bm = b < ROW_BYTES
    pool = tl.load(pool_ptrs_ptr + layer).to(tl.pointer_type(tl.uint8))
    src = pool + local * stride_pool_page + (pos % PAGE) * stride_pool_off
    vals = tl.load(src + b, mask=bm & valid, other=0)
    dst = wire_ptr + ((g.to(tl.int64) * LAYERS + layer) * CAPACITY + i) * ROW_BYTES
    tl.store(dst + b, vals, mask=bm)


@triton.jit
def _fill_scatter(
    meta_ptr,
    stride_meta,
    gathered_ptr,
    ring_ptr,
    ring_pages,
    slot_pages,
    groups,
    CAPACITY: tl.constexpr,
    LAYERS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK_B: tl.constexpr,
    WINDOW: tl.constexpr,
    PAGE: tl.constexpr,
    CP: tl.constexpr,
):
    rg = tl.program_id(0)
    i = tl.program_id(1)
    layer = tl.program_id(2)
    r = rg // groups
    g = rg % groups
    meta = meta_ptr + g.to(tl.int64) * stride_meta
    slot = tl.load(meta).to(tl.int64)
    length = tl.load(meta + 1).to(tl.int64)
    pos, _, valid = _share_position(length, i, r, WINDOW, PAGE, CP)
    row = pos % (slot_pages * PAGE)
    page = slot * slot_pages + row // PAGE
    b = tl.arange(0, BLOCK_B)
    bm = b < ROW_BYTES
    src = gathered_ptr + (((r * groups + g).to(tl.int64) * LAYERS + layer) * CAPACITY + i) * ROW_BYTES
    vals = tl.load(src + b, mask=bm & valid, other=0)
    dst = ring_ptr + ((layer.to(tl.int64) * ring_pages + page) * PAGE + row % PAGE) * ROW_BYTES
    tl.store(dst + b, vals, mask=bm & valid)


def fill_pack(meta, pool_ptrs, pool_page_bytes, pool_off_bytes, wire, cp_rank, window, page, cp):
    """This rank's share of every request's window into ``wire [G, layers, capacity, row bytes]``
    (uint8; holes zero). ``meta [G, 2 + table]`` int32 rows ``(slot, length, local page
    table...)``; ``pool_ptrs [layers]`` the history pools' bases, ``pool_*_bytes`` their page and
    offset strides in bytes."""
    groups, layers, capacity, row_bytes = wire.shape
    _fill_pack[(groups, capacity, layers)](
        meta,
        meta.stride(0),
        pool_ptrs,
        pool_page_bytes,
        pool_off_bytes,
        wire,
        cp_rank,
        CAPACITY=capacity,
        LAYERS=layers,
        ROW_BYTES=row_bytes,
        BLOCK_B=triton.next_power_of_2(row_bytes),
        WINDOW=window,
        PAGE=page,
        CP=cp,
        num_warps=1,
    )


def fill_scatter(meta, gathered, ring_bytes, slot_pages, window, page, cp):
    """Every CP rank's share (``gathered [cp, G, layers, capacity, row bytes]`` uint8) into the
    ring (``ring_bytes [layers, slots * slot_pages, 1, 1, page, row bytes]`` uint8) at each
    request's slot (``meta[:, 0]``); holes untouched."""
    _, groups, layers, capacity, row_bytes = gathered.shape
    _fill_scatter[(cp * groups, capacity, layers)](
        meta,
        meta.stride(0),
        gathered,
        ring_bytes,
        ring_bytes.shape[1],
        slot_pages,
        groups,
        CAPACITY=capacity,
        LAYERS=layers,
        ROW_BYTES=row_bytes,
        BLOCK_B=triton.next_power_of_2(row_bytes),
        WINDOW=window,
        PAGE=page,
        CP=cp,
        num_warps=1,
    )
