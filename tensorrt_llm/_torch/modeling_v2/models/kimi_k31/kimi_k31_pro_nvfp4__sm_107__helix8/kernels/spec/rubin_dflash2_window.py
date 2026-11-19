# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of the DFlash2 drafter's sliding-window attention: plan, body and tail.

Block query ``j`` of a request with ``L`` committed history positions sits at ``p = L + j`` and
attends the positions ``[lo_j, p]``, ``lo_j = max(0, p - W + 1)``. With page size ``P``,
``lo_max = lo_{blk-1}`` and ``S = min(L, ceil(lo_max / P) * P)``, the window splits into three
disjoint parts that cover it exactly (``lo_j <= lo_max <= S <= L <= p``):

- body: history positions ``[S, L)``, the same for every query of the request;
- head: history positions ``[lo_j, S)``, at most ``(blk - 1) + (P - 1)`` keys;
- block: positions ``[L, p]``, the block's own latents, read from a scratch buffer (never the
  pool).

The plan kernel derives all three from ``L`` once per step and reads only the page-table
columns of the window. The body kernel attends every query row of a request to one page-aligned
chunk of its body; the number of chunks is a static shape of the step (one per CUDA graph), so
no grid depends on a live length. It writes a normalized partial and natural-log
``(max, sum)`` stats per chunk. The tail kernel attends the head and block keys and merges them
with the body partials by log-sum-exp. Every merge selects (``where``) rather than multiplying a
zero weight into a partial, so an empty part never turns a row into NaN. The tail reads no body
partial of a request whose plan has no body key, so a body producer may leave those rows
unwritten.

Under helix (``CP_SIZE > 1``) position ``x`` lives on CP rank ``(x // P) % CP_SIZE`` as its local
page ``x // P // CP_SIZE`` (the rule of ``HelixDraftShard`` and of the fused prepare's store). Each
part keeps only the rank's own keys, so the ranks' partials merge to the exact window softmax. A
row with no local key is written as a zero partial with ``(-inf, 0)`` stats, the values the
helix exchange's empty-row sanitization would write.

Compile hygiene: lengths, batch rows and strides are runtime values or ``do_not_specialize``;
only the model and graph constants (``blk``, heads, page size, window, CP size, the split
count and the tile shapes) are ``constexpr``, so a new batch or new lengths never compile.

``vec_io`` (off by default) moves the query, pool, block, partial and output rows of the body and
tail kernels as 16-byte vectors: the launcher checks that every row stride, column offset and base
address involved is a multiple of 16 and passes ``ALIGNED``, under which the kernels hint the row
offsets with ``tl.multiple_of``. The same bytes reach the same arithmetic, so no output byte
changes; with ``ALIGNED`` false (``vec_io`` off, or a stride that refuses) the kernels are today's
text. ``ALIGNED`` follows the strides of persistent buffers, so it is fixed per graph as well.

``pdl`` (off by default, and only under the engine-wide PDL switch) launches the tail as a
programmatic dependent of the body inside :func:`launch_window_triton`: the body, a normal launch
(it starts after everything before it), lets its dependents launch as soon as each of its programs
has started (``PDL_TRIGGER``), and the tail runs everything that reads no body output (plan, query,
pool and block rows, the score dots, the row maximum) before it waits for the body grid
(``griddepcontrol.wait``, ``PDL_WAIT``) right before its first read of ``partial_stats``. One
variable sets ``PDL_WAIT`` and the launch attribute, so a tail never waits without being a
dependent, or the reverse.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

#: Query rows (``(j, h)`` pairs of one request) per program of the body and tail kernels.
BLOCK_M = 64
#: Latent columns of the output per program; the latent width is split over programs.
BLOCK_DV = 256
#: Latent columns per score-dot step.
BLOCK_KC = 128
#: Warps per body or tail program: a 64 x 256 float32 accumulator is 64 registers a thread at 8.
NUM_WARPS = 8
#: Smallest tile edge ``tl.dot`` takes.
_MIN_DOT = 16
#: log2(e): the kernels exponentiate in base 2; stats are natural-log.
_LOG2E = 1.4426950408889634


class DFlash2WindowPlan(NamedTuple):
    """One step's window plan for requests ``[0, B)``; filled by :func:`launch_window_plan`.

    Every field is int32. Shapes are static per graph: ``NWL`` = :func:`body_page_count`,
    ``HW`` = :func:`head_width`, ``num_splits`` = ``chunk_bounds.shape[1] - 1``.
    """

    #: ``[B, NWL]`` pool pages of this rank's body, in position order; ``-1`` after the last.
    body_pages: torch.Tensor
    #: ``[B]`` this rank's body keys; its pages are full except the last.
    body_len: torch.Tensor
    #: ``[B, num_splits + 1]`` page-aligned bounds of the split chunks of ``[0, body_len)``.
    chunk_bounds: torch.Tensor
    #: ``[B, HW]`` pool row ``page * P + offset`` of position ``lo_0 + t``, ``-1`` unless that
    #: position is a head key of this rank.
    head_rows: torch.Tensor
    #: ``[B, blk]`` first head index inside query ``j``'s window (``lo_j - lo_0``).
    head_first: torch.Tensor
    #: ``[B]`` bit ``k`` set when this rank owns block position ``L + k``.
    block_bits: torch.Tensor


class DFlash2WindowWorkspace(NamedTuple):
    """The body kernel's split partials, sized once for the largest step (before capture)."""

    #: float32 ``[max_requests, max_splits, blk * H, latent]`` normalized chunk partials.
    partial_out: torch.Tensor
    #: float32 ``[max_requests, max_splits, blk * H, 2]`` natural-log ``(max, sum)`` per chunk.
    partial_stats: torch.Tensor


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def body_page_count(window: int, page_size: int, block_size: int, cp_size: int) -> int:
    """Most body pages one rank holds.

    The body is page-aligned and holds at most ``W - blk`` keys (its start ``S`` is at least
    ``lo_max = L + blk - W``), so it spans at most ``ceil((W - blk) / P)`` consecutive pages, of
    which one CP rank owns every ``cp_size``-th.
    """
    return max(1, _cdiv(max(1, _cdiv(window - block_size, page_size)), cp_size))


def head_width(page_size: int, block_size: int) -> int:
    """Most head keys of a request: ``S - lo_0 <= (P - 1) + (blk - 1)``."""
    return max(1, page_size + block_size - 2)


def plan_buffers(
    max_requests: int,
    block_size: int,
    window: int,
    page_size: int,
    cp_size: int,
    max_splits: int,
    device: torch.device,
) -> DFlash2WindowPlan:
    """Persistent plan buffers for up to ``max_requests`` requests: allocate once, before capture."""
    nwl = body_page_count(window, page_size, block_size, cp_size)
    hw = head_width(page_size, block_size)
    i32 = dict(dtype=torch.int32, device=device)
    return DFlash2WindowPlan(
        body_pages=torch.full((max_requests, nwl), -1, **i32),
        body_len=torch.zeros(max_requests, **i32),
        chunk_bounds=torch.zeros(max_requests, max_splits + 1, **i32),
        head_rows=torch.full((max_requests, hw), -1, **i32),
        head_first=torch.zeros(max_requests, block_size, **i32),
        block_bits=torch.zeros(max_requests, **i32),
    )


def window_workspace(
    max_requests: int,
    block_size: int,
    num_heads: int,
    latent_dim: int,
    max_splits: int,
    device: torch.device,
) -> DFlash2WindowWorkspace:
    """The body partials for up to ``max_requests`` requests and ``max_splits`` chunks.

    Allocate once, before capture. The tail kernel reads only the chunks the plan of the step
    has, and every body launch rewrites them, so the buffers need no initialization.
    """
    rows = block_size * num_heads
    f32 = dict(dtype=torch.float32, device=device)
    return DFlash2WindowWorkspace(
        partial_out=torch.empty(max_requests, max_splits, rows, latent_dim, **f32),
        partial_stats=torch.empty(max_requests, max_splits, rows, 2, **f32),
    )


def plan_view(plan: DFlash2WindowPlan, num_requests: int, num_splits: int) -> DFlash2WindowPlan:
    """The rows of ``num_requests`` requests and the bounds of ``num_splits`` chunks (views)."""
    return DFlash2WindowPlan(
        body_pages=plan.body_pages[:num_requests],
        body_len=plan.body_len[:num_requests],
        chunk_bounds=plan.chunk_bounds[:num_requests, : num_splits + 1],
        head_rows=plan.head_rows[:num_requests],
        head_first=plan.head_first[:num_requests],
        block_bits=plan.block_bits[:num_requests],
    )


# ---------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "stride_table",
        "num_cols",
        "stride_pages",
        "stride_chunks",
        "stride_head",
        "stride_first",
        "cp_rank",
    ]
)
def _window_plan_kernel(
    num_ctx_ptr,
    table_ptr,
    table_rows_ptr,
    stride_table,
    num_cols,
    body_pages_ptr,
    stride_pages,
    body_len_ptr,
    chunk_ptr,
    stride_chunks,
    head_rows_ptr,
    stride_head,
    head_first_ptr,
    stride_first,
    block_bits_ptr,
    cp_rank,
    WINDOW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CP_SIZE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    HEAD: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HAS_ROWS: tl.constexpr,
    PAD_PAGES: tl.constexpr,
    PAD_HEAD: tl.constexpr,
    PAD_SPLITS: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    hist = tl.load(num_ctx_ptr + b).to(tl.int64)
    if HAS_ROWS:
        row = tl.load(table_rows_ptr + b).to(tl.int64)
    else:
        row = b.to(tl.int64)
    table_row = table_ptr + row * stride_table

    # S: the page-aligned body start; lo0: the first window position of query 0.
    lo_max = tl.maximum(hist + BLOCK_SIZE - WINDOW, 0)
    body_start = tl.minimum(hist, (lo_max + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE)
    lo0 = tl.maximum(hist - WINDOW + 1, 0)

    # Body: global pages [S / P, ceil(L / P)), this rank's every CP_SIZE-th, in order.
    first_page = body_start // PAGE_SIZE
    end_page = tl.where(body_start < hist, (hist + PAGE_SIZE - 1) // PAGE_SIZE, first_page)
    first_owned = first_page + (cp_rank - first_page % CP_SIZE + CP_SIZE) % CP_SIZE
    i = tl.arange(0, PAD_PAGES)
    gpage = first_owned + i * CP_SIZE
    live = (i < NUM_PAGES) & (gpage < end_page)
    col = tl.minimum(gpage // CP_SIZE, num_cols - 1)
    page = tl.load(table_row + col, mask=live, other=0).to(tl.int32)
    tl.store(body_pages_ptr + b * stride_pages + i, tl.where(live, page, -1), mask=i < NUM_PAGES)
    keys = tl.where(live, tl.minimum(hist - gpage * PAGE_SIZE, PAGE_SIZE), 0)
    body_len = tl.sum(keys, axis=0)
    tl.store(body_len_ptr + b, body_len.to(tl.int32))

    # Page-aligned chunks: chunk s takes local pages [s * n / NUM_SPLITS, (s + 1) * n / NUM_SPLITS).
    num_pages = tl.sum(live.to(tl.int64), axis=0)
    s = tl.arange(0, PAD_SPLITS)
    bound = tl.minimum((s * num_pages // NUM_SPLITS) * PAGE_SIZE, body_len)
    tl.store(chunk_ptr + b * stride_chunks + s, bound.to(tl.int32), mask=s <= NUM_SPLITS)

    # Head: positions lo0 + t below S that this rank owns.
    t = tl.arange(0, PAD_HEAD)
    pos = lo0 + t
    hpage = pos // PAGE_SIZE
    hlive = (t < HEAD) & (pos < body_start) & (hpage % CP_SIZE == cp_rank)
    hcol = tl.minimum(hpage // CP_SIZE, num_cols - 1)
    hp = tl.load(table_row + hcol, mask=hlive, other=0).to(tl.int64)
    hrow = tl.where(hlive, hp * PAGE_SIZE + pos % PAGE_SIZE, -1)
    tl.store(head_rows_ptr + b * stride_head + t, hrow.to(tl.int32), mask=t < HEAD)

    # Per query: the first head index of its window; and the block positions this rank owns.
    j = tl.arange(0, PAD_BLOCK)
    lo_j = tl.maximum(hist + j - WINDOW + 1, 0)
    tl.store(head_first_ptr + b * stride_first + j, (lo_j - lo0).to(tl.int32), mask=j < BLOCK_SIZE)
    owned = (j < BLOCK_SIZE) & (((hist + j) // PAGE_SIZE) % CP_SIZE == cp_rank)
    bits = tl.sum(tl.where(owned, 1 << j, 0), axis=0)
    tl.store(block_bits_ptr + b, bits.to(tl.int32))


def launch_window_plan(
    num_ctx: torch.Tensor,
    page_table: torch.Tensor,
    table_rows: Optional[torch.Tensor],
    block_size: int,
    window: int,
    page_size: int,
    cp_size: int,
    cp_rank: int,
    num_splits: int,
    plan: DFlash2WindowPlan,
) -> DFlash2WindowPlan:
    """Fill ``plan``'s first ``num_ctx.numel()`` rows; returns the views of this step."""
    batch = num_ctx.shape[0]
    view = plan_view(plan, batch, num_splits)
    nwl = view.body_pages.shape[1]
    hw = view.head_rows.shape[1]
    _window_plan_kernel[(batch,)](
        num_ctx,
        page_table,
        num_ctx if table_rows is None else table_rows,
        page_table.stride(0),
        page_table.shape[1],
        view.body_pages,
        view.body_pages.stride(0),
        view.body_len,
        view.chunk_bounds,
        view.chunk_bounds.stride(0),
        view.head_rows,
        view.head_rows.stride(0),
        view.head_first,
        view.head_first.stride(0),
        view.block_bits,
        cp_rank,
        WINDOW=window,
        BLOCK_SIZE=block_size,
        PAGE_SIZE=page_size,
        CP_SIZE=cp_size,
        NUM_PAGES=nwl,
        HEAD=hw,
        NUM_SPLITS=num_splits,
        HAS_ROWS=table_rows is not None,
        PAD_PAGES=triton.next_power_of_2(nwl),
        PAD_HEAD=triton.next_power_of_2(hw),
        PAD_SPLITS=triton.next_power_of_2(num_splits + 1),
        PAD_BLOCK=triton.next_power_of_2(block_size),
        num_warps=1,
    )
    return view


def window_plan_torch(
    num_ctx: torch.Tensor,
    page_table: torch.Tensor,
    table_rows: Optional[torch.Tensor],
    block_size: int,
    window: int,
    page_size: int,
    cp_size: int,
    cp_rank: int,
    num_splits: int,
) -> DFlash2WindowPlan:
    """The plan in plain torch, from the window definition (the plan kernel's reference)."""
    device = num_ctx.device
    hist = num_ctx.long()
    rows = torch.arange(hist.numel(), device=device) if table_rows is None else table_rows.long()
    table = page_table.long()[rows]
    num_cols = table.shape[1]
    nwl = body_page_count(window, page_size, block_size, cp_size)
    hw = head_width(page_size, block_size)
    lo_max = (hist + block_size - window).clamp_min(0)
    body_start = torch.minimum(hist, (lo_max + page_size - 1) // page_size * page_size)
    lo0 = (hist - window + 1).clamp_min(0)

    gpage = (body_start // page_size)[:, None] + torch.arange(nwl * cp_size, device=device)
    in_body = (gpage * page_size < hist[:, None]) & (body_start < hist)[:, None]
    owned = gpage % cp_size == cp_rank
    live = in_body & owned
    # This rank's pages, in order: stable-sort the live columns to the front.
    order = torch.argsort((~live).to(torch.int8), dim=1, stable=True)[:, :nwl]
    gsel = gpage.gather(1, order)
    lsel = live.gather(1, order)
    cols = (gsel // cp_size).clamp_max(num_cols - 1)
    body_pages = torch.where(lsel, table.gather(1, cols), torch.full_like(gsel, -1))
    keys = torch.where(lsel, (hist[:, None] - gsel * page_size).clamp(max=page_size), 0)
    body_len = keys.sum(1)
    num_pages = lsel.sum(1)
    s = torch.arange(num_splits + 1, device=device)
    chunk_bounds = torch.minimum(
        (s[None, :] * num_pages[:, None] // num_splits) * page_size, body_len[:, None]
    )

    pos = lo0[:, None] + torch.arange(hw, device=device)
    hpage = pos // page_size
    hlive = (pos < body_start[:, None]) & (hpage % cp_size == cp_rank)
    hcols = (hpage // cp_size).clamp_max(num_cols - 1)
    head_rows = torch.where(
        hlive, table.gather(1, hcols) * page_size + pos % page_size, torch.full_like(pos, -1)
    )
    j = torch.arange(block_size, device=device)
    lo_j = (hist[:, None] + j - window + 1).clamp_min(0)
    head_first = lo_j - lo0[:, None]
    bown = ((hist[:, None] + j) // page_size) % cp_size == cp_rank
    block_bits = (bown.long() << j).sum(1)
    i32 = torch.int32
    return DFlash2WindowPlan(
        body_pages=body_pages.to(i32),
        body_len=body_len.to(i32),
        chunk_bounds=chunk_bounds.to(i32),
        head_rows=head_rows.to(i32),
        head_first=head_first.to(i32),
        block_bits=block_bits.to(i32),
    )


# ---------------------------------------------------------------------------------------------
# body: one page-aligned chunk of the body for BLOCK_M query rows of one request
# ---------------------------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "stride_qr",
        "stride_qh",
        "stride_pg",
        "stride_pt",
        "stride_pages",
        "stride_chunks",
        "stride_ob",
        "stride_os",
        "stride_oi",
        "stride_sb",
        "stride_ss",
        "stride_si",
    ]
)
def _window_body_kernel(
    q_ptr,
    stride_qr,
    stride_qh,
    pool_ptr,
    stride_pg,
    stride_pt,
    body_pages_ptr,
    stride_pages,
    chunk_ptr,
    stride_chunks,
    po_ptr,
    stride_ob,
    stride_os,
    stride_oi,
    ps_ptr,
    stride_sb,
    stride_ss,
    stride_si,
    scale_log2,
    v_scale,
    HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    NUM_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_KC: tl.constexpr,
    ALIGNED: tl.constexpr = False,
    PDL_TRIGGER: tl.constexpr = False,
):
    if PDL_TRIGGER:
        # The PDL tail may start once every program of this grid has started: before its grid
        # wait it reads nothing this grid writes.
        tl.extra.cuda.gdc_launch_dependents()
    b = tl.program_id(0)
    tile = tl.program_id(1)
    split = tl.program_id(2) // NUM_DV
    dv0 = (tl.program_id(2) % NUM_DV) * BLOCK_DV

    i = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_ok = i < BLOCK_SIZE * HEADS
    q_off = (b * BLOCK_SIZE + i // HEADS).to(tl.int64) * stride_qr + (i % HEADS) * stride_qh
    # ALIGNED (launcher-checked): every row offset is a multiple of 16 elements, every base
    # 16-byte aligned, so the rows move as 16-byte vectors. Else today's text.
    if ALIGNED:
        q_row = q_ptr + tl.multiple_of(q_off, 16)
    else:
        q_row = q_ptr + q_off
    cdt = q_ptr.dtype.element_ty

    start = tl.load(chunk_ptr + b * stride_chunks + split)
    end = tl.load(chunk_ptr + b * stride_chunks + split + 1)

    n = tl.arange(0, PAGE_SIZE)
    kc = tl.arange(0, BLOCK_KC)
    kr = tl.arange(0, ROPE)
    dv = dv0 + tl.arange(0, BLOCK_DV)
    q_rope = tl.load(q_row[:, None] + LATENT + kr[None, :], mask=rows_ok[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)
    for t0 in range(start, end, PAGE_SIZE):
        page = tl.load(body_pages_ptr + b * stride_pages + t0 // PAGE_SIZE).to(tl.int64)
        k_ok = (t0 + n) < end
        if ALIGNED:
            k_row = pool_ptr + tl.multiple_of(page * stride_pg + n.to(tl.int64) * stride_pt, 16)
        else:
            k_row = pool_ptr + page * stride_pg + n.to(tl.int64) * stride_pt
        scores = tl.zeros([BLOCK_M, PAGE_SIZE], tl.float32)
        for c in tl.static_range(LATENT // BLOCK_KC):
            q_c = tl.load(
                q_row[:, None] + c * BLOCK_KC + kc[None, :], mask=rows_ok[:, None], other=0.0
            )
            k_c = tl.load(
                k_row[:, None] + c * BLOCK_KC + kc[None, :], mask=k_ok[:, None], other=0.0
            )
            scores += tl.dot(q_c, tl.trans(k_c.to(cdt)))
        k_r = tl.load(k_row[:, None] + LATENT + kr[None, :], mask=k_ok[:, None], other=0.0)
        scores += tl.dot(q_rope, tl.trans(k_r.to(cdt)))
        scores = tl.where(k_ok[None, :], scores * scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(k_row[:, None] + dv[None, :], mask=k_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(cdt), v.to(cdt))
        m_i = m_new

    # A chunk with no key keeps m = -inf and l = 0: a zero partial, selected away by the tail.
    has = l_i > 0.0
    out = tl.where(has[:, None], acc * (v_scale / tl.where(has, l_i, 1.0))[:, None], 0.0)
    if ALIGNED:
        o_off = tl.multiple_of(
            b.to(tl.int64) * stride_ob + split * stride_os + i.to(tl.int64) * stride_oi, 16
        )
    else:
        o_off = b.to(tl.int64) * stride_ob + split * stride_os + i.to(tl.int64) * stride_oi
    tl.store(po_ptr + o_off[:, None] + dv[None, :], out, mask=rows_ok[:, None])
    if dv0 == 0:
        s_off = b.to(tl.int64) * stride_sb + split * stride_ss + i.to(tl.int64) * stride_si
        tl.store(
            ps_ptr + s_off, tl.where(has, m_i * 0.6931471805599453, float("-inf")), mask=rows_ok
        )
        tl.store(ps_ptr + s_off + 1, l_i, mask=rows_ok)


# ---------------------------------------------------------------------------------------------
# tail: head and block keys, merged with the body partials
# ---------------------------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "stride_qr",
        "stride_qh",
        "stride_pg",
        "stride_pt",
        "stride_blk",
        "stride_head",
        "stride_first",
        "stride_ob",
        "stride_os",
        "stride_oi",
        "stride_sb",
        "stride_ss",
        "stride_si",
        "stride_or",
        "stride_oh",
        "stride_tr",
        "stride_th",
    ]
)
def _window_tail_kernel(
    q_ptr,
    stride_qr,
    stride_qh,
    pool_ptr,
    stride_pg,
    stride_pt,
    blk_ptr,
    stride_blk,
    head_rows_ptr,
    stride_head,
    head_first_ptr,
    stride_first,
    block_bits_ptr,
    body_len_ptr,
    po_ptr,
    stride_ob,
    stride_os,
    stride_oi,
    ps_ptr,
    stride_sb,
    stride_ss,
    stride_si,
    out_ptr,
    stride_or,
    stride_oh,
    stats_ptr,
    stride_tr,
    stride_th,
    scale_log2,
    v_scale,
    HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    HEAD: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_DV: tl.constexpr,
    WRITE_STATS: tl.constexpr,
    PAD_HEAD: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_KC: tl.constexpr,
    ALIGNED: tl.constexpr = False,
    PDL_WAIT: tl.constexpr = False,
):
    b = tl.program_id(0)
    tile = tl.program_id(1)
    dv0 = tl.program_id(2) * BLOCK_DV

    i = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_ok = i < BLOCK_SIZE * HEADS
    jq = i // HEADS
    hq = i % HEADS
    # ALIGNED: as in the body kernel; each hinted row offset below, else today's text.
    if ALIGNED:
        q_row = q_ptr + tl.multiple_of(
            (b * BLOCK_SIZE + jq).to(tl.int64) * stride_qr + hq * stride_qh, 16
        )
    else:
        q_row = q_ptr + (b * BLOCK_SIZE + jq).to(tl.int64) * stride_qr + hq * stride_qh
    cdt = q_ptr.dtype.element_ty
    kc = tl.arange(0, BLOCK_KC)
    kr = tl.arange(0, ROPE)
    dv = dv0 + tl.arange(0, BLOCK_DV)

    # Head keys: rows lo_0 + t of the pool; query j sees t >= head_first[j].
    t = tl.arange(0, PAD_HEAD)
    hrow = tl.load(head_rows_ptr + b * stride_head + t, mask=t < HEAD, other=-1).to(tl.int64)
    h_ok = hrow >= 0
    hsafe = tl.where(h_ok, hrow, 0)
    if ALIGNED:
        h_ptr = pool_ptr + tl.multiple_of(
            (hsafe // PAGE_SIZE) * stride_pg + (hsafe % PAGE_SIZE) * stride_pt, 16
        )
    else:
        h_ptr = pool_ptr + (hsafe // PAGE_SIZE) * stride_pg + (hsafe % PAGE_SIZE) * stride_pt
    first = tl.load(head_first_ptr + b * stride_first + jq, mask=rows_ok, other=0)
    h_vis = h_ok[None, :] & (t[None, :] >= first[:, None])

    # Block keys: rows b * blk + k of the scratch; query j sees owned k <= j.
    kb = tl.arange(0, PAD_BLOCK)
    bits = tl.load(block_bits_ptr + b)
    b_ok = (kb < BLOCK_SIZE) & (((bits >> kb) & 1) != 0)
    if ALIGNED:
        b_ptr = blk_ptr + tl.multiple_of(
            (b * BLOCK_SIZE + tl.minimum(kb, BLOCK_SIZE - 1)).to(tl.int64) * stride_blk, 16
        )
    else:
        b_ptr = (
            blk_ptr + (b * BLOCK_SIZE + tl.minimum(kb, BLOCK_SIZE - 1)).to(tl.int64) * stride_blk
        )
    b_vis = b_ok[None, :] & (kb[None, :] <= jq[:, None])

    s_h = tl.zeros([BLOCK_M, PAD_HEAD], tl.float32)
    s_b = tl.zeros([BLOCK_M, PAD_BLOCK], tl.float32)
    for c in tl.static_range(LATENT // BLOCK_KC):
        q_c = tl.load(q_row[:, None] + c * BLOCK_KC + kc[None, :], mask=rows_ok[:, None], other=0.0)
        kh = tl.load(h_ptr[:, None] + c * BLOCK_KC + kc[None, :], mask=h_ok[:, None], other=0.0)
        s_h += tl.dot(q_c, tl.trans(kh.to(cdt)))
        kbl = tl.load(b_ptr[:, None] + c * BLOCK_KC + kc[None, :], mask=b_ok[:, None], other=0.0)
        s_b += tl.dot(q_c, tl.trans(kbl.to(cdt)))
    q_r = tl.load(q_row[:, None] + LATENT + kr[None, :], mask=rows_ok[:, None], other=0.0)
    kh_r = tl.load(h_ptr[:, None] + LATENT + kr[None, :], mask=h_ok[:, None], other=0.0)
    s_h += tl.dot(q_r, tl.trans(kh_r.to(cdt)))
    kb_r = tl.load(b_ptr[:, None] + LATENT + kr[None, :], mask=b_ok[:, None], other=0.0)
    s_b += tl.dot(q_r, tl.trans(kb_r.to(cdt)))
    s_h = tl.where(h_vis, s_h * scale_log2, float("-inf"))
    s_b = tl.where(b_vis, s_b * scale_log2, float("-inf"))

    # The row maximum over the tail keys and every non-empty body chunk (log2 units). A request
    # without body keys has no non-empty chunk, whatever its partials hold: a body producer may
    # leave them unwritten.
    has_body = tl.load(body_len_ptr + b) > 0
    m_row = tl.maximum(tl.max(s_h, axis=1), tl.max(s_b, axis=1))
    if ALIGNED:
        o_base = tl.multiple_of(b.to(tl.int64) * stride_ob + i.to(tl.int64) * stride_oi, 16)
    else:
        o_base = b.to(tl.int64) * stride_ob + i.to(tl.int64) * stride_oi
    s_base = b.to(tl.int64) * stride_sb + i.to(tl.int64) * stride_si
    if PDL_WAIT:
        # Above: the plan, the query, the pool and block_kv, all complete before the body grid
        # started. Below: the body grid's partials, and every store.
        tl.extra.cuda.gdc_wait()
    for s in tl.static_range(NUM_SPLITS):
        m_s = tl.load(ps_ptr + s_base + s * stride_ss, mask=rows_ok, other=float("-inf"))
        l_s = tl.load(ps_ptr + s_base + s * stride_ss + 1, mask=rows_ok, other=0.0)
        l_s = tl.where(has_body, l_s, 0.0)
        m_row = tl.where(l_s > 0.0, tl.maximum(m_row, m_s * 1.4426950408889634), m_row)
    any_key = m_row > float("-inf")
    m_safe = tl.where(any_key, m_row, 0.0)

    p_h = tl.where(h_vis, tl.exp2(s_h - m_safe[:, None]), 0.0)
    p_b = tl.where(b_vis, tl.exp2(s_b - m_safe[:, None]), 0.0)
    l_row = tl.sum(p_h, axis=1) + tl.sum(p_b, axis=1)
    vh = tl.load(h_ptr[:, None] + dv[None, :], mask=h_ok[:, None], other=0.0)
    vb = tl.load(b_ptr[:, None] + dv[None, :], mask=b_ok[:, None], other=0.0)
    acc = tl.dot(p_h.to(cdt), vh.to(cdt)) + tl.dot(p_b.to(cdt), vb.to(cdt))
    acc = acc * v_scale
    for s in tl.static_range(NUM_SPLITS):
        m_s = tl.load(ps_ptr + s_base + s * stride_ss, mask=rows_ok, other=float("-inf"))
        l_s = tl.load(ps_ptr + s_base + s * stride_ss + 1, mask=rows_ok, other=0.0)
        l_s = tl.where(has_body, l_s, 0.0)
        live = l_s > 0.0
        w = tl.where(
            live, l_s * tl.exp2(tl.where(live, m_s * 1.4426950408889634, 0.0) - m_safe), 0.0
        )
        if ALIGNED:
            o_s = tl.load(
                po_ptr + tl.multiple_of(o_base + s * stride_os, 16)[:, None] + dv[None, :],
                mask=rows_ok[:, None] & live[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            o_s = tl.load(
                po_ptr + o_base[:, None] + s * stride_os + dv[None, :],
                mask=rows_ok[:, None] & live[:, None],
                other=0.0,
            ).to(tl.float32)
        acc += tl.where(live[:, None], w[:, None] * o_s, 0.0)
        l_row += w

    has = l_row > 0.0
    out = tl.where(has[:, None], acc / tl.where(has, l_row, 1.0)[:, None], 0.0)
    if ALIGNED:
        out_row = out_ptr + tl.multiple_of(
            (b * BLOCK_SIZE + jq).to(tl.int64) * stride_or + hq * stride_oh, 16
        )
    else:
        out_row = out_ptr + (b * BLOCK_SIZE + jq).to(tl.int64) * stride_or + hq * stride_oh
    tl.store(
        out_row[:, None] + dv[None, :], out.to(out_ptr.dtype.element_ty), mask=rows_ok[:, None]
    )
    if WRITE_STATS:
        if dv0 == 0:
            st_row = stats_ptr + (b * BLOCK_SIZE + jq).to(tl.int64) * stride_tr + hq * stride_th
            tl.store(st_row, tl.where(has, m_row * 0.6931471805599453, float("-inf")), mask=rows_ok)
            tl.store(st_row + 1, tl.where(has, l_row, 0.0), mask=rows_ok)


def _tile_args(block_size: int, num_heads: int, latent_dim: int, block_m: int, block_dv: int):
    dv = min(block_dv, latent_dim)
    if latent_dim % dv:
        raise ValueError(f"latent width {latent_dim} is not a multiple of {dv}")
    return triton.cdiv(block_size * num_heads, block_m), latent_dim // dv, dv


def _aligned16(*values: int) -> bool:
    """Whether every value is a multiple of 16."""
    return all(int(v) % 16 == 0 for v in values)


def _body_vec_io_values(
    query: torch.Tensor,
    pool: torch.Tensor,
    partial_out: torch.Tensor,
    latent_dim: int,
    dv: int,
    block_kc: int,
) -> dict:
    """What the body's ``ALIGNED`` needs to be multiples of 16: the element strides of the query,
    pool and partial rows, the column offsets inside them (``c * BLOCK_KC``, ``LATENT``, ``dv0``)
    and the byte addresses of their bases (Triton vectorizes only from a 16-byte aligned base)."""
    return {
        "query.stride(0)": query.stride(0),
        "query.stride(1)": query.stride(1),
        "pool.stride(0)": pool.stride(0),
        "pool.stride(1)": pool.stride(1),
        "partial_out.stride(0)": partial_out.stride(0),
        "partial_out.stride(1)": partial_out.stride(1),
        "partial_out.stride(2)": partial_out.stride(2),
        "latent_dim": latent_dim,
        "BLOCK_DV": dv,
        "BLOCK_KC": block_kc,
        "query address": query.data_ptr(),
        "pool address": pool.data_ptr(),
        "partial_out address": partial_out.data_ptr(),
    }


def _tail_vec_io_values(
    query: torch.Tensor,
    pool: torch.Tensor,
    block_kv: torch.Tensor,
    partial_out: torch.Tensor,
    out_strides: tuple,
    out_address: int,
    latent_dim: int,
    dv: int,
    block_kc: int,
) -> dict:
    """The tail's: the body's, the block rows' and the output rows'."""
    return {
        **_body_vec_io_values(query, pool, partial_out, latent_dim, dv, block_kc),
        "block_kv.stride(0)": block_kv.stride(0),
        "out.stride(0)": out_strides[0],
        "out.stride(1)": out_strides[1],
        "block_kv address": block_kv.data_ptr(),
        "out address": out_address,
    }


def _vec_io(kernel: str, vec_io: bool, values: dict) -> bool:
    """One launch's ``ALIGNED``: ``vec_io`` and every value a multiple of 16. A ``vec_io`` request
    the values refuse runs today's scalar loads and stores, logged once per process and kernel."""
    if not vec_io:
        return False
    bad = [f"{name} {int(v)}" for name, v in values.items() if int(v) % 16]
    if not bad:
        return True
    from tensorrt_llm.logger import logger  # here: the interpreted tests load this file alone

    logger.warning_once(
        f"DFlash2 window {kernel}: vec_io asked for, but not a multiple of 16: {', '.join(bad)}; "
        "the kernel runs its scalar loads and stores (ALIGNED=False)",
        key=f"dflash2_window_vec_io_refused_{kernel}",
    )
    return False


def window_vec_io_aligned(
    query: torch.Tensor,
    pool: torch.Tensor,
    block_kv: torch.Tensor,
    workspace: DFlash2WindowWorkspace,
    latent_dim: int,
    *,
    block_dv: int = BLOCK_DV,
    block_kc: int = BLOCK_KC,
) -> bool:
    """Whether :func:`launch_window_triton` with ``vec_io`` runs both kernels with ``ALIGNED`` on
    these inputs: the tail's conditions, which hold the body's, with the output being the
    launcher's own contiguous ``[R, H, latent]`` allocation (the caching allocator's base)."""
    _, num_heads, _ = query.shape
    dv, kc = min(block_dv, latent_dim), min(block_kc, latent_dim)
    out_strides = (num_heads * latent_dim, latent_dim)
    values = _tail_vec_io_values(
        query, pool, block_kv, workspace.partial_out, out_strides, 0, latent_dim, dv, kc
    )
    return _aligned16(*values.values())


def _pdl_on(query: torch.Tensor) -> bool:
    """The engine-wide PDL switch, on a GPU with programmatic dependent launch (SM 90 and up)."""
    if not query.is_cuda:
        return False
    # Imported here: the interpreted tests load this file alone.
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
    from tensorrt_llm._utils import get_sm_version

    return get_env_enable_pdl() and get_sm_version() >= 90


def _launch_window_tail(
    query: torch.Tensor,
    pool: torch.Tensor,
    block_kv: torch.Tensor,
    plan: DFlash2WindowPlan,
    partial_out: torch.Tensor,
    partial_stats: torch.Tensor,
    softmax_scale: float,
    latent_dim: int,
    kv_scale: float,
    stats: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    vec_io: bool,
    pdl: bool,
    block_m: int,
    block_dv: int,
    block_kc: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """The tail launch into ``out``. ``pdl``: a programmatic dependent of the launch just before it
    on the stream, which must be the body grid that writes ``partial_out`` / ``partial_stats``
    (:func:`launch_window_triton`): it waits for that grid only before its first partial read."""
    batch, block_size = plan.head_first.shape
    _, num_heads, head_dim = query.shape
    tiles, num_dv, dv = _tile_args(block_size, num_heads, latent_dim, block_m, block_dv)
    hw = plan.head_rows.shape[1]
    kc = min(block_kc, latent_dim)
    aligned = _vec_io(
        "tail",
        vec_io,
        _tail_vec_io_values(
            query,
            pool,
            block_kv,
            partial_out,
            (out.stride(0), out.stride(1)),
            out.data_ptr(),
            latent_dim,
            dv,
            kc,
        ),
    )
    _window_tail_kernel[(batch, tiles, num_dv)](
        query,
        query.stride(0),
        query.stride(1),
        pool,
        pool.stride(0),
        pool.stride(1),
        block_kv,
        block_kv.stride(0),
        plan.head_rows,
        plan.head_rows.stride(0),
        plan.head_first,
        plan.head_first.stride(0),
        plan.block_bits,
        plan.body_len,
        partial_out,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_stats,
        partial_stats.stride(0),
        partial_stats.stride(1),
        partial_stats.stride(2),
        out,
        out.stride(0),
        out.stride(1),
        out if stats is None else stats,
        0 if stats is None else stats.stride(0),
        0 if stats is None else stats.stride(1),
        softmax_scale * kv_scale * _LOG2E,
        kv_scale,
        HEADS=num_heads,
        BLOCK_SIZE=block_size,
        PAGE_SIZE=pool.shape[1],
        LATENT=latent_dim,
        ROPE=head_dim - latent_dim,
        HEAD=hw,
        NUM_SPLITS=partial_stats.shape[1],
        NUM_DV=num_dv,
        WRITE_STATS=stats is not None,
        PAD_HEAD=max(_MIN_DOT, triton.next_power_of_2(hw)),
        PAD_BLOCK=max(_MIN_DOT, triton.next_power_of_2(block_size)),
        BLOCK_M=block_m,
        BLOCK_DV=dv,
        BLOCK_KC=kc,
        ALIGNED=aligned,
        PDL_WAIT=pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        launch_pdl=pdl,
    )
    return out


def launch_window_triton(
    query: torch.Tensor,
    pool: torch.Tensor,
    block_kv: torch.Tensor,
    plan: DFlash2WindowPlan,
    softmax_scale: float,
    latent_dim: int,
    kv_scale: float,
    workspace: DFlash2WindowWorkspace,
    stats: Optional[torch.Tensor],
    *,
    vec_io: bool = False,
    pdl: bool = False,
    block_m: int = BLOCK_M,
    block_dv: int = BLOCK_DV,
    block_kc: int = BLOCK_KC,
    num_warps: int = NUM_WARPS,
    num_stages: int = 1,
) -> torch.Tensor:
    """Body (split-KV) and tail launches of one drafter layer; returns ``[R, H, latent]``.

    ``vec_io``: both kernels with 16-byte vector loads and stores where every stride allows
    (``ALIGNED``, checked per kernel; the module doc). ``pdl`` (under the engine-wide PDL switch):
    the tail as a programmatic dependent of the body, launched right after it on the same stream;
    the body stays a normal launch, so it starts only after every input of both kernels is
    complete. The output bytes are the same with either lever.
    """
    batch, block_size = plan.head_first.shape
    num_splits = plan.chunk_bounds.shape[1] - 1
    rows, num_heads, head_dim = query.shape
    partial_out = workspace.partial_out[:batch, :num_splits]
    partial_stats = workspace.partial_stats[:batch, :num_splits]
    tiles, num_dv, dv = _tile_args(block_size, num_heads, latent_dim, block_m, block_dv)
    kc = min(block_kc, latent_dim)
    # Allocated before the body: no host operation between the two launches.
    out = torch.empty(rows, num_heads, latent_dim, dtype=query.dtype, device=query.device)
    # One variable: the body's trigger, the tail's wait and the tail's launch attribute.
    use_pdl = pdl and _pdl_on(query)
    aligned = _vec_io(
        "body", vec_io, _body_vec_io_values(query, pool, partial_out, latent_dim, dv, kc)
    )
    _window_body_kernel[(batch, tiles, num_splits * num_dv)](
        query,
        query.stride(0),
        query.stride(1),
        pool,
        pool.stride(0),
        pool.stride(1),
        plan.body_pages,
        plan.body_pages.stride(0),
        plan.chunk_bounds,
        plan.chunk_bounds.stride(0),
        partial_out,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_stats,
        partial_stats.stride(0),
        partial_stats.stride(1),
        partial_stats.stride(2),
        softmax_scale * kv_scale * _LOG2E,
        kv_scale,
        HEADS=num_heads,
        BLOCK_SIZE=block_size,
        PAGE_SIZE=pool.shape[1],
        LATENT=latent_dim,
        ROPE=head_dim - latent_dim,
        NUM_DV=num_dv,
        BLOCK_M=block_m,
        BLOCK_DV=dv,
        BLOCK_KC=kc,
        ALIGNED=aligned,
        PDL_TRIGGER=use_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return _launch_window_tail(
        query,
        pool,
        block_kv,
        plan,
        partial_out,
        partial_stats,
        softmax_scale,
        latent_dim,
        kv_scale,
        stats,
        out,
        vec_io=vec_io,
        pdl=use_pdl,
        block_m=block_m,
        block_dv=block_dv,
        block_kc=block_kc,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# ---------------------------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------------------------


def _gather_rows(pool: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Pool rows ``page * P + offset`` (``-1`` reads row 0) in float32, whatever the pool dtype."""
    page_size = pool.shape[1]
    safe = rows.clamp_min(0).long()
    pages, offsets = safe // page_size, safe % page_size
    if pool.dtype == torch.float8_e4m3fn:
        # Byte view: advanced indexing of FP8 tensors is not supported everywhere.
        return pool.view(torch.uint8)[pages, offsets].view(torch.float8_e4m3fn).float()
    return pool[pages, offsets].float()


def window_reference(
    query: torch.Tensor,
    pool: torch.Tensor,
    block_kv: torch.Tensor,
    plan: DFlash2WindowPlan,
    softmax_scale: float,
    latent_dim: int,
    kv_scale: float,
    stats: Optional[torch.Tensor] = None,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Each query row's exact softmax attention over the keys its plan assigns, in one pass.

    No split, no merge order: the body, head and block keys of a row are gathered together and
    attended with one softmax in ``compute_dtype``, rounded once to the query's dtype. A row
    with no key gets a zero output and, in ``stats``, ``(-inf, 0)``; otherwise ``stats`` gets
    the natural-log maximum of its scaled scores and the sum of their exponentials.
    """
    batch, block_size = plan.head_first.shape
    rows, num_heads, head_dim = query.shape
    page_size = pool.shape[1]
    device = query.device
    span = plan.body_pages.shape[1] * page_size
    t = torch.arange(span, device=device)
    body_pages = plan.body_pages.long()
    body_rows = body_pages.gather(1, (t // page_size)[None].expand(batch, -1)) * page_size + (
        t % page_size
    )
    body_ok = t[None] < plan.body_len.long()[:, None]
    body_rows = torch.where(body_ok, body_rows, torch.full_like(body_rows, -1))
    head_rows = plan.head_rows.long()
    j = torch.arange(block_size, device=device)
    k = torch.arange(block_size, device=device)
    bits = plan.block_bits.long()
    block_ok = ((bits[:, None] >> k) & 1) != 0
    keys = torch.cat((_gather_rows(pool, body_rows), _gather_rows(pool, head_rows)), dim=1)
    keys = torch.cat((keys, block_kv.reshape(batch, block_size, head_dim).float()), dim=1)
    # A key outside the plan is never read: zero what the gather fetched in its place.
    key_ok = torch.cat((body_ok, head_rows >= 0, block_ok), dim=1)
    keys = torch.where(key_ok[..., None], keys, 0).to(compute_dtype) * kv_scale

    hw = head_rows.shape[1]
    head_vis = (head_rows >= 0)[:, None, :] & (
        torch.arange(hw, device=device)[None, None, :] >= plan.head_first.long()[:, :, None]
    )
    block_vis = block_ok[:, None, :] & (k[None, :] <= j[:, None])[None]
    visible = torch.cat(
        (body_ok[:, None, :].expand(batch, block_size, span), head_vis, block_vis), dim=2
    )

    q = query.reshape(batch, block_size, num_heads, head_dim).to(compute_dtype)
    scores = torch.einsum("bjhd,bkd->bjhk", q, keys) * softmax_scale
    scores = scores.masked_fill(~visible[:, :, None, :], float("-inf"))
    peak = scores.amax(dim=-1, keepdim=True)
    has = torch.isfinite(peak)
    probs = torch.where(visible[:, :, None, :], torch.exp(scores - torch.where(has, peak, 0)), 0)
    total = probs.sum(-1, keepdim=True)
    out = torch.einsum("bjhk,bkc->bjhc", probs, keys[..., :latent_dim])
    out = torch.where(has, out / torch.where(has, total, 1), 0)
    if stats is not None:
        st = torch.stack(
            (
                torch.where(has, peak, float("-inf"))[..., 0],
                torch.where(has, total, 0)[..., 0],
            ),
            dim=-1,
        )
        stats.copy_(st.reshape(rows, num_heads, 2).to(stats.dtype))
    return out.reshape(rows, num_heads, latent_dim).to(query.dtype)


__all__ = [
    "BLOCK_DV",
    "BLOCK_KC",
    "BLOCK_M",
    "DFlash2WindowPlan",
    "DFlash2WindowWorkspace",
    "body_page_count",
    "head_width",
    "launch_window_plan",
    "launch_window_triton",
    "plan_buffers",
    "plan_view",
    "window_plan_torch",
    "window_reference",
    "window_vec_io_aligned",
    "window_workspace",
]
