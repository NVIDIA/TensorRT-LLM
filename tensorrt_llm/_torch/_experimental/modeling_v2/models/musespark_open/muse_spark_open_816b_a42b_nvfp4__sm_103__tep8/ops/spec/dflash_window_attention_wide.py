# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash drafter block attention over the sliding window, for decode steps of several requests.

The same computation as ``dflash_window_attention`` (non-causal block attention
of ``M`` query rows over the last ``window_left + M`` keys of the request's slot,
optionally appending the block's K/V to the arena), laid out for many requests
per call. ``dflash_window_attention`` is tuned for one request: one 32-key tile
per split, about 65 splits per (request, KV head), each writing a ``[GROUP * M, D]``
FP32 partial -- at a TP4 rank and 16 requests that is 2,080 programs writing
twice the window's K/V bytes as partials for the merge to read back. Here:

* a split walks its keys with a runtime ``tl.range`` loop of ``BLOCK_N``-key tiles
  with ``STAGES`` loads in flight, so a split may be long without unrolling, and a
  request is cut into a handful of splits (``wide_launch_config``) -- the FP32
  partials shrink to a fraction of the window's K/V;
* the block's own ``M`` keys form one separate tail tile, read from ``k_new`` /
  ``v_new`` (or from the arena without the append) by the one split whose key
  range holds the block's first key; the arena loop of every split stops before
  the block, so no tile merges arena and block keys through a select.

One program per (split, request, KV head); each packs every query head of its KV
head with every block row into one tile (``GROUP * M`` rows), as the single-request
op does. The split partials are merged by ``dflash_window_attention``'s merge
kernel (one program per output row, fp32 log-sum-exp weights); a window that fits
one split is written directly.

The append (``k_new`` / ``v_new``): the first split's program of each (request, KV
head) stores the block's K/V into the arena at ``[L - M, L)``; no program of the
grid reads those columns (the tail tile reads ``k_new`` / ``v_new``). One writer
per column given one slot per request; CUDA-graph padding entries share one dummy
slot and write it in no defined order, as ``index_put`` with duplicate indices
does.

The context may live in pages (``page_table``), addressed as the single-request
op addresses them; on the same values a paged call is bitwise the dense call.

Numerics follow the single-request op: fp32 scores, base-2 online softmax, the
probabilities rounded to the operand dtype for the P.V product with the row sum
taken from the fp32 values, each split normalized and merged in fp32 by
log-sum-exp weights, one rounding at the end. Only the split boundaries and the
tile grouping differ (the block keys are one tile of their own), so the output
is not bitwise ``dflash_window_attention``'s or FlashAttention's; it is gated
against FlashAttention's own error on the same inputs, as that op is (maximum
error within one unit in the last place of the row scale above FlashAttention's,
mean error within 10 % of it). With and without the append the keys are grouped
the same way, so those two calls are bitwise equal.
"""

from __future__ import annotations

import functools
import math

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from .dflash_window_attention import (
    MAX_SPLITS,
    WINDOW,
    _context_rows,
    _page_arguments,
    _window_attention_merge_kernel,
    attention_reference,
    unsupported_reason,
)

#: Keys of the block's tail tile: the block rows padded to the smallest dot width.
MIN_TAIL_KEYS = 16
#: Block rows the tail tile takes.
MAX_BLOCK_ROWS = 16


@triton.jit
def _window_attention_wide_split_kernel(
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
    split_keys,
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
    TAIL: tl.constexpr,
    GROUP: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGES: tl.constexpr,
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
        # Before any global read: q and the block's K/V are the previous kernels'
        # outputs. The merge may launch now; it waits for this whole grid before it
        # reads a partial.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    seqlen = tl.load(seqlens_ptr + b)
    lo = tl.maximum(seqlen - M - WINDOW_LEFT, 0)
    start = lo + split * split_keys
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
    jj = tl.arange(0, TAIL)
    jj_ok = jj < M
    if APPEND:
        if split == 0:
            # The append: the block's K/V into the arena at [L - M, L). No program of
            # this grid reads those columns (the tail tile reads k_new / v_new).
            new_rows = new_row0 + jj
            kv_rows = _context_rows(table, block_start + jj, jj_ok, page_rows, PAGE)
            k_blk = tl.load(
                k_new_ptr + new_rows[:, None] * stride_knt + kvh * stride_knh + dims[None, :],
                mask=jj_ok[:, None],
            )
            v_blk = tl.load(
                v_new_ptr + new_rows[:, None] * stride_vnt + kvh * stride_vnh + dims[None, :],
                mask=jj_ok[:, None],
            )
            tl.store(
                k_base + kv_rows[:, None] * stride_kn + dims[None, :], k_blk, mask=jj_ok[:, None]
            )
            tl.store(
                v_base + kv_rows[:, None] * stride_vn + dims[None, :], v_blk, mask=jj_ok[:, None]
            )

    # Packed tile row r = g * M + i: query head kvh * GROUP + g, block row i at key
    # position seqlen - M + i (bottom-right aligned).
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
    end = tl.minimum(start + split_keys, seqlen)
    # The arena's keys of this split, before the block; the block's keys go to the tail.
    arena_end = tl.minimum(end, block_start)
    for c0 in tl.range(start, arena_end, BLOCK_N, num_stages=STAGES):
        cols = c0 + tl.arange(0, BLOCK_N)
        ok = cols < arena_end
        kv_rows = _context_rows(table, cols, ok, page_rows, PAGE)
        k = tl.load(
            k_base + kv_rows[:, None] * stride_kn + dims[None, :], mask=ok[:, None], other=0.0
        )
        v = tl.load(
            v_base + kv_rows[:, None] * stride_vn + dims[None, :], mask=ok[:, None], other=0.0
        )
        s = tl.dot(q, tl.trans(k))
        keep = ok[None, :] & (cols[None, :] >= first_key[:, None])
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

    # The block's M keys: one tail tile, owned by the split whose range holds block_start.
    if (start <= block_start) & (end > block_start):
        cols = block_start + jj
        if APPEND:
            new_rows = new_row0 + jj
            k = tl.load(
                k_new_ptr + new_rows[:, None] * stride_knt + kvh * stride_knh + dims[None, :],
                mask=jj_ok[:, None],
                other=0.0,
            )
            v = tl.load(
                v_new_ptr + new_rows[:, None] * stride_vnt + kvh * stride_vnh + dims[None, :],
                mask=jj_ok[:, None],
                other=0.0,
            )
        else:
            kv_rows = _context_rows(table, cols, jj_ok, page_rows, PAGE)
            k = tl.load(
                k_base + kv_rows[:, None] * stride_kn + dims[None, :],
                mask=jj_ok[:, None],
                other=0.0,
            )
            v = tl.load(
                v_base + kv_rows[:, None] * stride_vn + dims[None, :],
                mask=jj_ok[:, None],
                other=0.0,
            )
        s = tl.dot(q, tl.trans(k))
        keep = jj_ok[None, :] & (cols[None, :] >= first_key[:, None])
        if WINDOW_RIGHT >= 0:
            keep = keep & (cols[None, :] <= (q_pos + WINDOW_RIGHT)[:, None])
        s = tl.where(keep, s * scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
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
        # A split whose range holds only block keys it does not own writes an empty
        # partial (lse -inf), which the merge weighs at zero.
        part = (split * num_bh + bh).to(tl.int64) * (GROUP * M)
        tl.store(o_part_ptr + (part + rows)[:, None] * D + dims[None, :], o, mask=row_ok[:, None])
        lse = tl.where(has_key, m_i + tl.math.log2(l_use), float("-inf"))
        tl.store(lse_part_ptr + part + rows, lse, mask=row_ok)


#: Split geometry by requests per call: (fewest requests, key-tile width, target splits
#: per request, warps, pipeline stages). The last row whose fewest requests the call
#: reaches applies. The fastest geometry of a sweep at a 2,047-key window on GB300 (TP4
#: rank shape, cold L2, both orders): the split grid (requests x KV heads x splits)
#: stays within about one wave of 8-warp programs up to 16 requests, two waves of
#: 4-warp programs above.
WIDE_GEOMETRY = (
    (1, 64, 11, 8, 3),
    (7, 64, 11, 4, 3),
    (8, 64, 7, 8, 3),
    (12, 64, 6, 8, 3),
    (14, 64, 5, 8, 3),
    (16, 64, 4, 8, 3),
    (20, 64, 7, 4, 3),
    (24, 64, 6, 4, 3),
    (28, 64, 5, 4, 3),
    (32, 64, 4, 4, 3),
    (40, 64, 3, 4, 3),
    (64, 64, 2, 4, 3),
)


#: The same sweep on R200 (212 SMs, TP4 rank shape, cold L2, confirm pass in reverse
#: order): eight-warp programs win at almost every request count; the split count steps
#: down as the split grid (requests x KV heads x splits) passes about one, then two
#: waves of the larger device.
WIDE_GEOMETRY_212_SMS = (
    (1, 64, 11, 8, 3),
    (10, 64, 7, 8, 3),
    (16, 64, 6, 8, 3),
    (20, 64, 5, 8, 3),
    (24, 64, 7, 8, 3),
    (28, 64, 7, 4, 3),
    (32, 64, 6, 4, 3),
    (40, 64, 5, 8, 3),
    (48, 64, 4, 8, 3),
    (64, 64, 3, 8, 3),
    (96, 64, 2, 8, 3),
    (128, 64, 3, 8, 3),
)
#: The geometry table per device SM count; a device of another count takes ``WIDE_GEOMETRY``.
WIDE_GEOMETRY_BY_SMS = {152: WIDE_GEOMETRY, 212: WIDE_GEOMETRY_212_SMS}


@functools.lru_cache(maxsize=None)
def _device_sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def wide_geometry(device: torch.device) -> tuple:
    """The geometry table for ``device``: ``WIDE_GEOMETRY_BY_SMS`` by its SM count."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return WIDE_GEOMETRY_BY_SMS.get(_device_sms(index), WIDE_GEOMETRY)


def split_keys_for(span: int, block_n: int, splits: int) -> int:
    """Keys per split cutting ``span`` keys into at most ``splits`` splits of whole tiles."""
    return block_n * max(1, triton.cdiv(triton.cdiv(span, block_n), splits))


def wide_launch_config(batch: int, span: int, geometry: tuple = WIDE_GEOMETRY) -> dict:
    """The split geometry for ``batch`` requests of at most ``span`` keys each from ``geometry``."""
    row = geometry[0]
    for candidate in geometry:
        if batch >= candidate[0]:
            row = candidate
    _, block_n, splits, warps, stages = row
    split_keys = split_keys_for(span, block_n, splits)
    # The merge holds at most MAX_SPLITS partials per row.
    split_keys = max(split_keys, split_keys_for(span, block_n, MAX_SPLITS))
    return dict(
        block_n=block_n, split_keys=split_keys, num_warps=warps, stages=stages, merge_warps=4
    )


def wide_window_attention_launch(
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
    split_keys: int,
    num_warps: int,
    stages: int,
    merge_warps: int,
    page_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the split and merge kernels with an explicit geometry.

    ``page_table``: the caches are pools of pages (``dflash_window_attention``'s
    module docstring).

    Raises:
        ValueError: if ``split_keys`` is not a whole number of tiles or the
            geometry cuts the window into more than ``MAX_SPLITS`` splits.
    """
    batch, m, num_q_heads, d = q.shape
    num_kv_heads = k_cache.shape[2]
    group = num_q_heads // num_kv_heads
    window_left, window_right = window_size
    if split_keys <= 0 or split_keys % block_n:
        raise ValueError(f"split_keys {split_keys} is not a whole number of {block_n}-key tiles")
    num_splits = triton.cdiv(window_left + m, split_keys)
    if num_splits > MAX_SPLITS:
        raise ValueError(
            f"{window_left + m} keys in {split_keys}-key splits is {num_splits} splits; "
            f"the merge takes at most {MAX_SPLITS}"
        )
    tile_rows = group * m
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
    _window_attention_wide_split_kernel[(num_splits, num_bh)](
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
        split_keys,
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
        TAIL=max(MIN_TAIL_KEYS, triton.next_power_of_2(m)),
        GROUP=group,
        HKV=num_kv_heads,
        D=d,
        ROWS=max(16, triton.next_power_of_2(tile_rows)),
        BLOCK_N=block_n,
        STAGES=stages,
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
            ENTRY_TRIGGER=False,
            num_warps=merge_warps,
            launch_pdl=enable_pdl,
        )
    return out


def _spec(**over):
    """A cell spec: a TP4 rank of the served drafter (16 query / 2 KV heads), block 8.

    By default every request's slot holds a full window, as the windowed arena's
    slots do at steady state (capacity = window + block + 16 + 8).
    """
    spec = dict(batch=8, m=8, hq=16, hkv=2, d=128, window=WINDOW, ctx=None, capacity=2_080)
    spec.update(over)
    if spec["ctx"] is None:
        spec["ctx"] = [2_064 - spec["m"]] * spec["batch"]
    spec.setdefault("slots", spec["batch"] + 1)
    return spec


class DflashWindowAttentionWide(OpWrapper):
    """Non-causal sliding-window attention of several DFlash blocks over their slots' pooled K/V.

    ``__call__(q, k_cache, v_cache, cache_seqlens, cache_batch_idx, window_size,
    softmax_scale=None, enable_pdl=False, k_new=None, v_new=None, page_table=None)``:
    the arguments and the result of ``dflash_window_attention`` (see there), without
    its single-request merge entry trigger. The split geometry follows the request
    count and the device's SM count (``wide_geometry``, ``wide_launch_config``).

    Fusion boundary: the output projection (after) stays outside; so does the
    append when ``k_new`` is None.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    note = (
        "Not bitwise dflash_window_attention or flash_attn_with_kvcache: different split "
        "boundaries and the block keys in a tile of their own change the accumulation "
        "order; gated against FlashAttention's own error on the same inputs (max within "
        "1 ulp of the row scale above it, mean within 10 %). Causal masks, unbounded left "
        "windows, blocks of more than 16 rows and windows past MAX_WINDOW_SPAN keys are "
        "not taken."
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
        page_table: torch.Tensor | None = None,
    ) -> None:
        reason = wide_unsupported_reason(
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
            raise ValueError(f"dflash_window_attention_wide: {reason}")

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
        page_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        config = wide_launch_config(
            q.shape[0], window_size[0] + q.shape[1], wide_geometry(q.device)
        )
        return wide_window_attention_launch(
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
    ) -> torch.Tensor:
        """Dense masked softmax attention in fp32, one rounding at the end.

        ``enable_pdl`` changes how the op launches, not what it computes; ignored.
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


def wide_geometry_unsupported_reason(m: int) -> str | None:
    """Why this op would not take blocks of ``m`` rows beyond ``dflash_window_attention``'s limits."""
    if m > MAX_BLOCK_ROWS:
        return f"the block's tail tile takes at most {MAX_BLOCK_ROWS} rows, got {m}"
    return None


def wide_unsupported_reason(
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
    reason = unsupported_reason(
        q, k_cache, v_cache, cache_seqlens, cache_batch_idx, window_size, k_new, v_new, page_table
    )
    if reason is not None:
        return reason
    return wide_geometry_unsupported_reason(q.shape[1])


dflash_window_attention_wide = DflashWindowAttentionWide()
