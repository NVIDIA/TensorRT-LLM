# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash drafter block attention over a sliding window of the pooled context.

The DFlash drafter attends its block of ``M`` query rows (the last accepted
token and the ``K`` drafts) non-causally over the request's pooled context
K/V, which sits in a dense per-slot arena ``[slots, capacity, Hkv, D]`` sized
for the served ``max_seq_len`` (about 1M tokens here). Every drafter layer of
the published DFlash 2 checkpoint is a two-sided 2048-token sliding window,
so a block only ever reads the last ``window_left + M`` keys.

The built-in path is FlashAttention-2's ``flash_attn_with_kvcache``. Its
split-KV heuristic sizes the split count and each split's key range from the
arena *capacity*, not from the window: at capacity ~1M, eight query heads and
152 SMs it picks 33 splits of ~32k keys each, the window falls inside one of
them, and that split's CTA walks the ~2k window keys alone, 128 keys per step,
with its 64-row tile holding 8 real rows. The other splits exit at once and the
combine kernel still reduces all 33.

This entry bounds the work to the window instead. The key range
``[max(0, L - M - window_left), L)`` (``L`` = keys including the block, read
on device so the call is CUDA-graph safe) is cut into fixed-size splits, one
program per (split, request, KV head); each program packs every query head of
its KV head with every block row into one tile (``GROUP * M`` rows, 64 for
eight heads and an eight-row block), so the window's K/V is read once per KV
head, and computes the masked partial softmax for its keys. A second kernel
merges the splits' (output, log-sum-exp) pairs, one program per output row
reading all of its partials in one load. Splits wholly past ``L`` (short
contexts) exit without writing and are not read back; a window that fits one
split is written directly, without the merge. The merge holds at most
``MAX_SPLITS`` partials per row in registers, so a window wider than
``MAX_SPLITS`` 32-key splits gets longer splits (several 32-key tiles each)
rather than more of them; windows past ``MAX_WINDOW_SPAN`` keys are refused.

Optionally (``k_new`` / ``v_new``) the op also performs the block's K/V append
the drafter otherwise does with two ``index_put`` launches: the block's keys
and values are read from ``k_new`` / ``v_new`` rather than the arena, and the
first split's program of each (request, KV head) stores them into the arena at
``[L - M, L)``. No program reads those arena columns. Given one slot per
request, each block column is then written by exactly one program, and the
arena ends up holding exactly what the ``index_put`` would have written.
Requests that share a slot (the drafter's CUDA-graph padding entries all map to
one dummy slot, with the same ``L``) write the same columns from different
programs, in no defined order -- as ``index_put`` with duplicate indices does.
Nothing reads a dummy slot's block, so only which padding row lands there
differs.

The context may instead live in pages, as a KV cache manager keeps it
(``page_table``): the caches are then pools ``[pages, page_size, Hkv, D]`` (the
planes of an HND pool, transposed), and ``cache_batch_idx`` names each request's
table row, which lists its pages in token order, so column ``c`` is row
``c % page_size`` of page ``page_table[row, c // page_size]``. A paged call
differs from a dense one only in which cache row a column is read from or
appended to; on the same values the two are bitwise equal.

Numerics follow FlashAttention-2's split path: fp32 scores, base-2 online
softmax, the probabilities rounded to the operand dtype for the P.V product
with the row sum taken from the fp32 values, each split normalized and merged
in fp32 by log-sum-exp weights, one rounding to the operand dtype at the end.
The split boundaries differ from FlashAttention's, so the result is not bitwise
FlashAttention's: at the TP8 serving shape about a third of the output
elements differ in their last bits, and the mean error against a float64
evaluation of the same bf16 inputs is lower than FlashAttention's (fewer,
larger splits merged in fp32). The tests gate every cell against
FlashAttention's own error on the same inputs: a maximum error within one unit
in the last place of the row scale above FlashAttention's, and a mean error
within 10 % of it.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

#: The published DFlash 2 drafter at TP8: 64 query / 8 KV heads over eight
#: ranks, head width 128, a 2048-token two-sided window (FlashAttention bounds
#: ``(2047, 2047)``), a block of K + 1 = 8 rows.
NUM_Q_HEADS_PER_RANK, NUM_KV_HEADS_PER_RANK, HEAD_DIM = 8, 1, 128
WINDOW = (2047, 2047)

#: Operand dtypes the kernels take.
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
#: Largest packed tile (query heads per KV head x block rows) one program holds.
MAX_TILE_ROWS = 128
#: Keys per tile of a split.
SPLIT_BLOCK_N = 32
#: Most splits per (request, KV head): the merge loads every split's partial
#: output for a row as one [MAX_SPLITS, D] fp32 tile. 128 x 128 fits in
#: registers without spilling (about 155 registers per thread at four warps);
#: 256 splits spill.
MAX_SPLITS = 128
#: Most 32-key tiles one split walks; with MAX_SPLITS this bounds the window.
MAX_TILES = 16
#: Most keys a block row's window may span (window_size[0] + M).
MAX_WINDOW_SPAN = MAX_SPLITS * MAX_TILES * SPLIT_BLOCK_N


@triton.jit
def _context_rows(table, cols, mask, page_rows, PAGE: tl.constexpr):
    """The cache rows of context columns ``cols``: the columns of a dense slot, or row
    ``c % PAGE`` of page ``table[c // PAGE]`` in a pool whose pages are ``page_rows`` apart."""
    if PAGE:
        page = tl.load(table + cols // PAGE, mask=mask, other=0).to(tl.int64)
        rows = page * page_rows + cols % PAGE
    else:
        rows = cols.to(tl.int64)
    return rows


def _page_arguments(k_cache, v_cache, page_table, cache_batch_idx):
    """The split kernel's ``(page_table, PAGE, page_rows, stride_pt, stride_ks, stride_vs)``.

    A dense call reads its slot's columns (``PAGE`` 0, the table unused); a paged
    call reads rows of the whole pool through its table row (slot strides 0).
    """
    if page_table is None:
        return cache_batch_idx, 0, 0, 0, k_cache.stride(0), v_cache.stride(0)
    page_rows = k_cache.stride(0) // k_cache.stride(1)
    return page_table, k_cache.shape[1], page_rows, page_table.stride(0), 0, 0


@triton.jit
def _window_attention_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    k_new_ptr,
    v_new_ptr,
    seqlens_ptr,
    batch_idx_ptr,
    page_table_ptr,
    out_ptr,
    o_part_ptr,
    lse_part_ptr,
    scale_log2,
    num_bh,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_ks,
    stride_kn,
    stride_kh,
    stride_vs,
    stride_vn,
    stride_vh,
    stride_pt,
    page_rows,
    stride_knt,
    stride_knh,
    stride_vnt,
    stride_vnh,
    stride_ob,
    stride_om,
    stride_oh,
    M: tl.constexpr,
    M_P2: tl.constexpr,
    GROUP: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    APPEND: tl.constexpr,
    DIRECT: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    PAGE: tl.constexpr,
):
    split = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HKV
    kvh = bh % HKV
    if LAUNCH_WITH_PDL:
        # Before any global read: q and the block's K/V are the previous
        # kernels' outputs. The merge kernel may launch now; it waits for this
        # whole grid (griddepcontrol.wait) before it reads a partial.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    seqlen = tl.load(seqlens_ptr + b)
    lo = tl.maximum(seqlen - M - WINDOW_LEFT, 0)
    start = lo + split * (TILES * BLOCK_N)
    if start >= seqlen:
        return
    slot = tl.load(batch_idx_ptr + b).to(tl.int64)
    dims = tl.arange(0, D)
    # A paged call's slot names the request's page-table row; its slot strides are 0.
    table = page_table_ptr + slot * stride_pt
    k_base = k_ptr + slot * stride_ks + kvh * stride_kh
    v_base = v_ptr + slot * stride_vs + kvh * stride_vh
    block_start = seqlen - M
    new_row0 = b.to(tl.int64) * M
    if APPEND:
        if split == 0:
            # The append: the block's K/V into the arena at [L - M, L). Nothing
            # in this grid reads those columns (the block keys come from
            # k_new / v_new below). One writer per column given one slot per
            # request (see the module docstring for shared padding slots).
            j = tl.arange(0, M_P2)
            j_ok = j < M
            new_rows = new_row0 + j
            kv_rows = _context_rows(table, block_start + j, j_ok, page_rows, PAGE)
            k_blk = tl.load(
                k_new_ptr + new_rows[:, None] * stride_knt + kvh * stride_knh + dims[None, :],
                mask=j_ok[:, None],
            )
            v_blk = tl.load(
                v_new_ptr + new_rows[:, None] * stride_vnt + kvh * stride_vnh + dims[None, :],
                mask=j_ok[:, None],
            )
            tl.store(
                k_base + kv_rows[:, None] * stride_kn + dims[None, :], k_blk, mask=j_ok[:, None]
            )
            tl.store(
                v_base + kv_rows[:, None] * stride_vn + dims[None, :], v_blk, mask=j_ok[:, None]
            )

    # Packed tile row r = g * M + i: query head kvh * GROUP + g, block row i,
    # which sits at key position seqlen - M + i (bottom-right aligned).
    rows = tl.arange(0, ROWS)
    g = rows // M
    i = rows % M
    row_ok = rows < GROUP * M
    q_pos = block_start + i
    first_key = q_pos - WINDOW_LEFT
    q = tl.load(
        q_ptr
        + b.to(tl.int64) * stride_qb
        + i[:, None] * stride_qm
        + (kvh * GROUP + g)[:, None] * stride_qh
        + dims[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )

    m_i = tl.full([ROWS], float("-inf"), tl.float32)
    l_i = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, D], tl.float32)
    for t in tl.static_range(TILES):
        cols = start + t * BLOCK_N + tl.arange(0, BLOCK_N)
        col_ok = cols < seqlen
        kv_rows = _context_rows(table, cols, col_ok, page_rows, PAGE)
        # Both loads are issued before the products: V's latency hides behind
        # QK^T and the softmax.
        if APPEND:
            from_new = cols >= block_start
            in_arena = col_ok & (cols < block_start)
            new_rows = new_row0 + (cols - block_start)
            k = tl.load(
                k_base + kv_rows[:, None] * stride_kn + dims[None, :],
                mask=in_arena[:, None],
                other=0.0,
            )
            k = tl.where(
                from_new[:, None],
                tl.load(
                    k_new_ptr + new_rows[:, None] * stride_knt + kvh * stride_knh + dims[None, :],
                    mask=(col_ok & from_new)[:, None],
                    other=0.0,
                ),
                k,
            )
            v = tl.load(
                v_base + kv_rows[:, None] * stride_vn + dims[None, :],
                mask=in_arena[:, None],
                other=0.0,
            )
            v = tl.where(
                from_new[:, None],
                tl.load(
                    v_new_ptr + new_rows[:, None] * stride_vnt + kvh * stride_vnh + dims[None, :],
                    mask=(col_ok & from_new)[:, None],
                    other=0.0,
                ),
                v,
            )
        else:
            k = tl.load(
                k_base + kv_rows[:, None] * stride_kn + dims[None, :],
                mask=col_ok[:, None],
                other=0.0,
            )
            v = tl.load(
                v_base + kv_rows[:, None] * stride_vn + dims[None, :],
                mask=col_ok[:, None],
                other=0.0,
            )
        s = tl.dot(q, tl.trans(k))
        keep = col_ok[None, :] & (cols[None, :] >= first_key[:, None])
        if WINDOW_RIGHT >= 0:
            keep = keep & (cols[None, :] <= (q_pos + WINDOW_RIGHT)[:, None])
        s = tl.where(keep, s * scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # A row with no key yet keeps m = -inf; shift by 0 so exp2 gives 0, not NaN.
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_use)
        p = tl.math.exp2(s - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    has_key = l_i > 0.0
    l_use = tl.where(has_key, l_i, 1.0)
    o = acc / l_use[:, None]
    if DIRECT:
        tl.store(
            out_ptr
            + b.to(tl.int64) * stride_ob
            + i[:, None] * stride_om
            + (kvh * GROUP + g)[:, None] * stride_oh
            + dims[None, :],
            o.to(out_ptr.dtype.element_ty),
            mask=row_ok[:, None],
        )
    else:
        part = (split * num_bh + bh).to(tl.int64) * (GROUP * M)
        tl.store(o_part_ptr + (part + rows)[:, None] * D + dims[None, :], o, mask=row_ok[:, None])
        lse = tl.where(has_key, m_i + tl.math.log2(l_use), float("-inf"))
        tl.store(lse_part_ptr + part + rows, lse, mask=row_ok)


@triton.jit
def _window_attention_merge_kernel(
    o_part_ptr,
    lse_part_ptr,
    seqlens_ptr,
    out_ptr,
    num_bh,
    stride_ob,
    stride_om,
    stride_oh,
    M: tl.constexpr,
    GROUP: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    SPLITS_P2: tl.constexpr,
    SPLIT_KEYS: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    ENTRY_TRIGGER: tl.constexpr = False,
):
    # One program per output row: every split's partial for the row in one
    # [splits, D] load, so the merge costs one L2 round trip, not one per split.
    bh = tl.program_id(0)
    row = tl.program_id(1)
    b = bh // HKV
    kvh = bh % HKV
    if LAUNCH_WITH_PDL:
        if ENTRY_TRIGGER:
            # Release the dependent (the output projection, which streams its
            # weight before its own wait) before this kernel's wait.
            tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    seqlen = tl.load(seqlens_ptr + b)
    lo = tl.maximum(seqlen - M - WINDOW_LEFT, 0)
    # The splits the split kernel wrote: those starting before seqlen.
    num_splits = (seqlen - lo + SPLIT_KEYS - 1) // SPLIT_KEYS
    splits = tl.arange(0, SPLITS_P2)
    split_ok = splits < num_splits
    part = (splits * num_bh + bh).to(tl.int64) * (GROUP * M) + row
    lse = tl.load(lse_part_ptr + part, mask=split_ok, other=float("-inf"))
    dims = tl.arange(0, D)
    o = tl.load(o_part_ptr + part[:, None] * D + dims[None, :], mask=split_ok[:, None], other=0.0)
    lse_max = tl.max(lse, 0)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    weight = tl.math.exp2(lse - lse_max)
    denom = tl.sum(weight, 0)
    out = tl.sum(weight[:, None] * o, 0) / tl.where(denom > 0.0, denom, 1.0)
    g = row // M
    i = row % M
    tl.store(
        out_ptr + b.to(tl.int64) * stride_ob + i * stride_om + (kvh * GROUP + g) * stride_oh + dims,
        out.to(out_ptr.dtype.element_ty),
    )


def default_launch_config(span: int, tile_rows: int) -> dict:
    """Split geometry for at most ``span`` keys per request and ``tile_rows`` packed rows.

    ``block_n * tiles`` keys per split. At the TP8 drafter shape (64 packed
    rows, a 2,055-key span) the default is many short splits -- 65 programs
    of one 32-key tile -- because each program's cost is its load latency,
    not its arithmetic, so wider tiles and multi-tile splits only lengthen
    each program. A span wider than ``MAX_SPLITS``
    such splits keeps ``MAX_SPLITS`` splits and gives each several tiles, so
    the merge's per-row tile stays at ``MAX_SPLITS`` partials.
    """
    tiles = max(1, triton.cdiv(triton.cdiv(span, SPLIT_BLOCK_N), MAX_SPLITS))
    return dict(block_n=SPLIT_BLOCK_N, tiles=tiles, num_warps=8, merge_warps=4)


def window_attention_launch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_batch_idx: torch.Tensor,
    window_size: tuple[int, int],
    softmax_scale: float | None,
    enable_pdl: bool,
    k_new: torch.Tensor | None = None,
    v_new: torch.Tensor | None = None,
    *,
    block_n: int,
    tiles: int,
    num_warps: int,
    merge_warps: int,
    merge_entry_trigger: bool = False,
    page_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the split and merge kernels with an explicit split geometry.

    ``merge_entry_trigger`` (with ``enable_pdl``): the merge kernel releases its
    dependent before its own wait instead of at its exit (the PDL chain's ``pdl_merge``).
    ``page_table``: the caches are pools of pages (see the module docstring).

    Raises:
        ValueError: if the geometry cuts the window into more than ``MAX_SPLITS`` splits.
    """
    batch, m, num_q_heads, d = q.shape
    num_kv_heads = k_cache.shape[2]
    group = num_q_heads // num_kv_heads
    window_left, window_right = window_size
    tile_rows = group * m
    split_keys = block_n * tiles
    # Every row's keys lie in [L - M - window_left, L): at most window_left + M.
    num_splits = triton.cdiv(window_left + m, split_keys)
    if num_splits > MAX_SPLITS:
        raise ValueError(
            f"{window_left + m} keys in {split_keys}-key splits is {num_splits} splits; "
            f"the merge takes at most {MAX_SPLITS}"
        )
    scale = d**-0.5 if softmax_scale is None else softmax_scale
    out = torch.empty_like(q)
    num_bh = batch * num_kv_heads
    if batch == 0:
        return out
    append = k_new is not None
    if not append:
        k_new = v_new = k_cache  # unused
    page_table, page, page_rows, stride_pt, stride_ks, stride_vs = _page_arguments(
        k_cache, v_cache, page_table, cache_batch_idx
    )
    direct = num_splits == 1
    if direct:
        o_part = lse_part = out  # unused
    else:
        o_part = torch.empty(
            (num_splits, num_bh, tile_rows, d), dtype=torch.float32, device=q.device
        )
        lse_part = torch.empty(
            (num_splits, num_bh, tile_rows), dtype=torch.float32, device=q.device
        )
    _window_attention_split_kernel[(num_splits, num_bh)](
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        cache_seqlens,
        cache_batch_idx,
        page_table,
        out,
        o_part,
        lse_part,
        scale * math.log2(math.e),
        num_bh,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        stride_ks,
        k_cache.stride(1),
        k_cache.stride(2),
        stride_vs,
        v_cache.stride(1),
        v_cache.stride(2),
        stride_pt,
        page_rows,
        k_new.stride(0),
        k_new.stride(1),
        v_new.stride(0),
        v_new.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        M=m,
        M_P2=max(2, triton.next_power_of_2(m)),
        GROUP=group,
        HKV=num_kv_heads,
        D=d,
        ROWS=max(16, triton.next_power_of_2(tile_rows)),
        BLOCK_N=block_n,
        TILES=tiles,
        WINDOW_LEFT=window_left,
        WINDOW_RIGHT=window_right,
        APPEND=append,
        DIRECT=direct,
        LAUNCH_WITH_PDL=enable_pdl,
        PAGE=page,
        num_warps=num_warps,
        launch_pdl=enable_pdl,
    )
    if not direct:
        _window_attention_merge_kernel[(num_bh, tile_rows)](
            o_part,
            lse_part,
            cache_seqlens,
            out,
            num_bh,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            M=m,
            GROUP=group,
            HKV=num_kv_heads,
            D=d,
            SPLITS_P2=triton.next_power_of_2(num_splits),
            SPLIT_KEYS=split_keys,
            WINDOW_LEFT=window_left,
            LAUNCH_WITH_PDL=enable_pdl,
            ENTRY_TRIGGER=enable_pdl and merge_entry_trigger,
            num_warps=merge_warps,
            launch_pdl=enable_pdl,
        )
    return out


def _spec(**over):
    """A cell spec: the TP8 serving shape unless overridden."""
    spec = dict(batch=1, m=8, hq=8, hkv=1, d=128, window=WINDOW, ctx=[5_000], capacity=8_200)
    spec["slots"] = 2
    spec.update(over)
    return spec


class DflashWindowAttention(OpWrapper):
    """Non-causal sliding-window attention of a DFlash block over its slot's pooled K/V.

    ``__call__(q, k_cache, v_cache, cache_seqlens, cache_batch_idx, window_size,
    softmax_scale=None, enable_pdl=False, k_new=None, v_new=None,
    merge_entry_trigger=False, page_table=None)`` takes the arguments the drafter
    hands ``flash_attn_with_kvcache`` (non-causal):

    * ``q`` ``[B, M, Hq, D]``: the block's queries, unit last-dim stride;
    * ``k_cache`` / ``v_cache`` ``[slots, capacity, Hkv, D]``: one layer's view
      of the arena (any outer strides). Without ``k_new`` / ``v_new`` they
      already hold the block's own K/V at ``[L - M, L)``;
    * ``cache_seqlens`` ``[B]`` int32: ``L``, keys per request including the
      block; ``cache_batch_idx`` ``[B]`` int32: the arena slot per request;
    * ``window_size = (left, right)``: FlashAttention's inclusive bounds -- row
      ``i`` (at position ``L - M + i``) sees keys ``j`` with
      ``L - M + i - left <= j <= L - M + i + right`` and ``j < L``;
      ``right = -1`` is unbounded;
    * ``softmax_scale``: ``D ** -0.5`` when None;
    * ``k_new`` / ``v_new`` ``[B * M, Hkv, D]`` (any row / head strides, unit
      last-dim stride): the block's K/V, appended by the op -- written into
      ``k_cache`` / ``v_cache`` at ``[L - M, L)`` of each request's slot and
      attended from these tensors;
    * ``merge_entry_trigger`` (with ``enable_pdl``, the PDL chain's ``pdl_merge``): the merge
      kernel triggers its dependents before its ``griddepcontrol.wait``
      instead of at its exit. It reads only the split's partials and the
      lengths, all after that wait; its dependent must read the output after
      its own wait. The same values;
    * ``page_table`` ``[rows, max_pages]`` int32, unit last stride: the caches
      are pools ``[pages, page_size, Hkv, D]`` whose page stride is the same
      whole number of rows for both, and ``cache_batch_idx`` names table rows
      (see the module docstring).

    Returns ``[B, M, Hq, D]`` in ``q``'s dtype, contiguous.

    Fusion boundary: the output projection (after) stays outside; so does the
    append when ``k_new`` is None.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = (
        "Not bitwise flash_attn_with_kvcache: different split boundaries change the "
        "accumulation order; gated against FlashAttention's own error on the same "
        "inputs (max within 1 ulp of the row scale above it, mean within 10 %). "
        "Causal masks, unbounded left windows and windows past MAX_WINDOW_SPAN keys "
        "are not taken; the drafter adapter sends those to the built-in."
    )

    def is_valid(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_batch_idx: torch.Tensor,
        window_size: tuple[int, int],
        softmax_scale: float | None = None,
        enable_pdl: bool = False,
        k_new: torch.Tensor | None = None,
        v_new: torch.Tensor | None = None,
        merge_entry_trigger: bool = False,
        page_table: torch.Tensor | None = None,
    ) -> None:
        reason = unsupported_reason(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            cache_batch_idx,
            window_size,
            k_new,
            v_new,
            page_table,
        )
        if reason is not None:
            raise ValueError(f"dflash_window_attention: {reason}")

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_batch_idx: torch.Tensor,
        window_size: tuple[int, int],
        softmax_scale: float | None = None,
        enable_pdl: bool = False,
        k_new: torch.Tensor | None = None,
        v_new: torch.Tensor | None = None,
        merge_entry_trigger: bool = False,
        page_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m, num_q_heads = q.shape[1], q.shape[2]
        group = num_q_heads // k_cache.shape[2]
        config = default_launch_config(window_size[0] + m, group * m)
        return window_attention_launch(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            cache_batch_idx,
            window_size,
            softmax_scale,
            enable_pdl,
            k_new,
            v_new,
            **config,
            merge_entry_trigger=merge_entry_trigger,
            page_table=page_table,
        )

    def reference(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_batch_idx: torch.Tensor,
        window_size: tuple[int, int],
        softmax_scale: float | None = None,
        enable_pdl: bool = False,
        k_new: torch.Tensor | None = None,
        v_new: torch.Tensor | None = None,
        merge_entry_trigger: bool = False,
    ) -> torch.Tensor:
        """Dense masked softmax attention in fp32, one rounding at the end.

        The block's keys and values come from ``k_new`` / ``v_new`` when given;
        the reference does not write the arena. ``enable_pdl`` and
        ``merge_entry_trigger`` change how the op launches, not what it
        computes; ignored.
        """
        return attention_reference(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            cache_batch_idx,
            window_size,
            softmax_scale,
            torch.float32,
            k_new,
            v_new,
        ).to(q.dtype)


def attention_reference(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    cache_batch_idx,
    window_size,
    softmax_scale,
    dtype,
    k_new=None,
    v_new=None,
) -> torch.Tensor:
    """The block attention computed densely in ``dtype`` (fp32 or fp64), unrounded."""
    batch, m, num_q_heads, d = q.shape
    group = num_q_heads // k_cache.shape[2]
    window_left, window_right = window_size
    scale = d**-0.5 if softmax_scale is None else softmax_scale
    out = torch.empty(q.shape, dtype=dtype, device=q.device)
    seqlens = cache_seqlens.tolist()
    slots = cache_batch_idx.tolist()
    for b in range(batch):
        seqlen, slot = int(seqlens[b]), int(slots[b])
        lo = max(0, seqlen - m - window_left) if window_left >= 0 else 0
        keys = k_cache[slot, lo:seqlen].to(dtype)
        values = v_cache[slot, lo:seqlen].to(dtype)
        if k_new is not None:
            first = seqlen - m - lo
            keys[first:] = k_new[b * m : (b + 1) * m].to(dtype)
            values[first:] = v_new[b * m : (b + 1) * m].to(dtype)
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        scores = torch.einsum("mhd,nhd->hmn", q[b].to(dtype), keys) * scale
        cols = torch.arange(lo, seqlen, device=q.device)
        q_pos = seqlen - m + torch.arange(m, device=q.device)
        keep = cols[None, :] < seqlen
        if window_left >= 0:
            keep = keep & (cols[None, :] >= (q_pos - window_left)[:, None])
        if window_right >= 0:
            keep = keep & (cols[None, :] <= (q_pos + window_right)[:, None])
        scores = scores.masked_fill(~keep[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hmn,nhd->mhd", probs, values)
    return out


def geometry_unsupported_reason(
    m: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    window_size: tuple[int, int],
) -> str | None:
    """Why the kernels would not take this call geometry, or None.

    The tensor-free part of ``unsupported_reason``: a caller that knows the
    shapes before it has the tensors (the drafter install) checks these once.
    The limits only grow with ``m``, so a geometry valid at some block size is
    valid at every smaller one.
    """
    if dtype not in SUPPORTED_DTYPES:
        return f"operands must share one of {SUPPORTED_DTYPES}, got {dtype}"
    if head_dim not in (64, 128, 256):
        return f"head width must be 64, 128 or 256, got {head_dim}"
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        return "query heads must be a multiple of KV heads"
    if m < 1 or (num_q_heads // num_kv_heads) * m > MAX_TILE_ROWS:
        return f"query heads per KV head times block rows must be in [1, {MAX_TILE_ROWS}]"
    left, right = window_size
    if left < 0:
        return "the left window must be bounded (window_size[0] >= 0)"
    if right < -1:
        return "window_size[1] must be >= -1"
    if left + m > MAX_WINDOW_SPAN:
        return f"the window spans {left + m} keys; at most {MAX_WINDOW_SPAN} are taken"
    return None


def unsupported_reason(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_batch_idx: torch.Tensor,
    window_size: tuple[int, int],
    k_new: torch.Tensor | None = None,
    v_new: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
) -> str | None:
    """Why the kernels would answer these inputs wrongly, or None if they take them."""
    if q.dim() != 4 or k_cache.dim() != 4 or v_cache.dim() != 4:
        return "q, k_cache and v_cache must be 4-D ([B, M, Hq, D] / [slots, capacity, Hkv, D])"
    if not (q.dtype == k_cache.dtype == v_cache.dtype):
        return f"operands must share one of {SUPPORTED_DTYPES}, got {q.dtype}/{k_cache.dtype}"
    if k_cache.shape != v_cache.shape:
        return "k_cache and v_cache shapes differ"
    batch, m, num_q_heads, d = q.shape
    if k_cache.shape[3] != d:
        return f"head width must match the cache, got {d} and {k_cache.shape[3]}"
    num_kv_heads = k_cache.shape[2]
    reason = geometry_unsupported_reason(m, num_q_heads, num_kv_heads, d, q.dtype, window_size)
    if reason is not None:
        return reason
    if q.stride(3) != 1 or k_cache.stride(3) != 1 or v_cache.stride(3) != 1:
        return "the head dimension must have unit stride"
    if cache_seqlens.dtype != torch.int32 or cache_batch_idx.dtype != torch.int32:
        return "cache_seqlens and cache_batch_idx must be int32"
    if cache_seqlens.shape != (batch,) or cache_batch_idx.shape != (batch,):
        return "cache_seqlens and cache_batch_idx must be [B]"
    if not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and cache_seqlens.is_cuda):
        return "all tensors must be on the GPU"
    if (k_new is None) != (v_new is None):
        return "k_new and v_new come together"
    if k_new is not None:
        expected = (batch * m, num_kv_heads, d)
        if tuple(k_new.shape) != expected or tuple(v_new.shape) != expected:
            return f"k_new / v_new must be {list(expected)}"
        if k_new.dtype != q.dtype or v_new.dtype != q.dtype:
            return "k_new / v_new must have the operands' dtype"
        if k_new.stride(2) != 1 or v_new.stride(2) != 1:
            return "k_new / v_new head dimension must have unit stride"
    if page_table is not None:
        if page_table.dtype != torch.int32 or page_table.dim() != 2 or page_table.stride(1) != 1:
            return "page_table must be [rows, max_pages] int32 with unit last stride"
        if not page_table.is_cuda:
            return "all tensors must be on the GPU"
        page_rows = k_cache.stride(0) // k_cache.stride(1)
        if (k_cache.stride(0), v_cache.stride(0)) != (
            page_rows * k_cache.stride(1),
            page_rows * v_cache.stride(1),
        ):
            return "a page's stride must be the same whole number of rows for K and V"
    return None


dflash_window_attention = DflashWindowAttention()
