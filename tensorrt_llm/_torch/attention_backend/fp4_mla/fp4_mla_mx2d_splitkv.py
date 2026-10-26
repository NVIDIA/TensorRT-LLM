# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-KV plan and merge for the persistent MXFP4-2D FP4 MLA decode (opt-in).

``TRTLLM_FP4_MLA_MXFP4_2D_SPLITKV=1`` runs a small MXFP4-2D decode batch (at most
``TRTLLM_FP4_MLA_MXFP4_2D_SPLITKV_MAX_QUERIES`` single-token queries, default 64) as
(query, KV-tile window) tasks so that a few long queries fill the GPU.  The plan kernel
balances each query's KV tiles over windows, in the longest-KV-first order; the persistent
clusters take the tasks from their work queue.  A query with one window is written by the
decode itself; the others store normalized FP32 partials and their row softmax state, and
the merge kernel combines them (a rounding-level change: the merge order and each window's
own anchor).  The plan and merge are device-side (no host sync) and CUDA-graph capturable;
the buffers are allocated by the first eager call, never during capture.

Variant label: H95 (split-KV MXFP4-2D decode).
"""

import os
from typing import Any, NamedTuple, Optional

import torch
import triton
import triton.language as tl

FP4_MLA_MXFP4_2D_SPLITKV_ENV = "TRTLLM_FP4_MLA_MXFP4_2D_SPLITKV"
FP4_MLA_MXFP4_2D_SPLITKV_MAX_QUERIES_ENV = "TRTLLM_FP4_MLA_MXFP4_2D_SPLITKV_MAX_QUERIES"
# Plan knobs: the shortest window (KV tiles), the windows per query at most (rounded up to a
# power of two) and the task waves (each wave is one task per persistent cluster).
FP4_MLA_MXFP4_2D_SPLITKV_MIN_TILES_ENV = "TRTLLM_FP4_MLA_MXFP4_2D_SPLITKV_MIN_TILES"
FP4_MLA_MXFP4_2D_SPLITKV_MAX_WINDOWS_ENV = "TRTLLM_FP4_MLA_MXFP4_2D_SPLITKV_MAX_WINDOWS"
FP4_MLA_MXFP4_2D_SPLITKV_WAVES_ENV = "TRTLLM_FP4_MLA_MXFP4_2D_SPLITKV_WAVES"
_DEFAULT_MAX_QUERIES = 64
_DEFAULT_MIN_TILES = 4
_KV_TILE = 256
_PLAN_KEY_ATTR = "_fp4_mla_mx2d_split_plan_key"

# Per device: (task map [tasks, 4] int32, query ranges [queries, 2] int32,
# [work-queue counter header | FP32 workspace]), sized for every batch up to the query cap.
_BUFFERS: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


class SplitKvLaunch(NamedTuple):
    """One split decode: the kernel's ``split_kv`` arguments and what the merge needs."""

    task_map: torch.Tensor
    buffer: torch.Tensor
    ranges: torch.Tensor
    max_windows: int
    header_ints: int


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def split_kv_requested() -> bool:
    """Whether TRTLLM_FP4_MLA_MXFP4_2D_SPLITKV is on (read per call; a CUDA graph keeps the
    launch it captured)."""
    return os.environ.get(FP4_MLA_MXFP4_2D_SPLITKV_ENV, "0").lower() in ("1", "true", "yes", "on")


def split_kv_max_queries() -> int:
    value = _env_int(FP4_MLA_MXFP4_2D_SPLITKV_MAX_QUERIES_ENV)
    return _DEFAULT_MAX_QUERIES if value is None else value


@triton.jit
def _split_kv_plan_kernel(
    kv_lens_ptr,
    order_ptr,
    task_map_ptr,
    ranges_ptr,
    num_queries,
    num_tasks,
    min_tiles,
    max_windows,
    KV_TILE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Program i places the i-th longest query.

    Every program derives the shared window length from all queries' KV tiles: the smallest
    piece >= min_tiles (and >= ceil(longest / max_windows)) for which the queries'
    max(1, ceil(t / piece)) windows fit the tasks (a binary search under the closed-form
    bound max(ceil(total / (tasks - queries)), that floor), which always fits).  A query of
    t tiles gets that many windows of balanced length (they differ by at most one tile).
    Tasks are laid out in the order (longest query first, its windows together), a query
    with one window is direct (its decode task stores the output row), and the slots past
    the last window are empty (query 0, window [0, 0), not direct: a fully masked tile into
    an unread partial slot).  ``ranges`` holds each query's (first task, window count).
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_Q)
    live = offs < num_queries
    qs = tl.load(order_ptr + offs, mask=live, other=0)
    lens = tl.load(kv_lens_ptr + qs, mask=live, other=0)
    tiles = tl.maximum((tl.maximum(lens, 0) + (KV_TILE - 1)) // KV_TILE, 1)
    tiles = tl.where(live, tiles, 0)
    total = tl.sum(tiles, axis=0)
    spare = tl.maximum(num_tasks - num_queries, 1)
    # At most max_windows windows per query (the merge loads them at once).
    lo = tl.maximum(
        tl.maximum(min_tiles, 1), (tl.max(tiles, axis=0) + max_windows - 1) // max_windows
    )
    hi = tl.maximum((total + spare - 1) // spare, lo)
    for _ in tl.static_range(17):
        mid = (lo + hi) // 2
        need = tl.sum(tl.where(live, tl.maximum((tiles + mid - 1) // mid, 1), 0), axis=0)
        fits = need <= num_tasks
        hi = tl.where(fits, mid, hi)
        lo = tl.where(fits, lo, mid + 1)
    piece = hi
    counts = tl.where(live, tl.maximum((tiles + piece - 1) // piece, 1), 0)
    ends = tl.cumsum(counts, axis=0)
    mine = offs == pid
    my_query = tl.sum(tl.where(mine, qs, 0), axis=0)
    my_tiles = tl.sum(tl.where(mine, tiles, 0), axis=0)
    my_count = tl.sum(tl.where(mine, counts, 0), axis=0)
    my_first = tl.sum(tl.where(mine, ends, 0), axis=0) - my_count
    base = my_tiles // my_count
    extra = my_tiles - base * my_count
    slot = tl.arange(0, BLOCK_T)
    used = slot < my_count
    begin = slot * base + tl.minimum(slot, extra)
    end = begin + base + (slot < extra).to(tl.int32)
    rows = task_map_ptr + (my_first + slot) * 4
    tl.store(rows, tl.zeros_like(slot) + my_query, mask=used)
    tl.store(rows + 1, begin, mask=used)
    tl.store(rows + 2, end, mask=used)
    tl.store(rows + 3, tl.zeros_like(slot) + (my_count == 1).to(tl.int32), mask=used)
    tl.store(ranges_ptr + my_query * 2, my_first)
    tl.store(ranges_ptr + my_query * 2 + 1, my_count)
    if pid == num_queries - 1:
        empty = (slot >= my_first + my_count) & (slot < num_tasks)
        zero = tl.zeros_like(slot)
        rows = task_map_ptr + slot * 4
        tl.store(rows, zero, mask=empty)
        tl.store(rows + 1, zero, mask=empty)
        tl.store(rows + 2, zero, mask=empty)
        tl.store(rows + 3, zero, mask=empty)


@triton.jit
def _split_kv_merge_kernel(
    workspace_ptr,
    ranges_ptr,
    out_ptr,
    out_stride_query,
    out_stride_head,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    MAX_WINDOWS: tl.constexpr,
):
    """Merge one query's windows for HEAD_BLOCK heads, in FP32.

    out = sum_j w_j o_j / sum_j w_j with w_j = 2^(lse_j - max lse), lse_j = m2_j +
    log2(row sum_j) and o_j the window's normalized output rows.  One [windows, heads, DIM]
    tile load (at most MAX_WINDOWS windows, the plan's cap for this launch) puts every
    window's rows in flight at once.  A window with a zero row sum (no valid key) stored
    O = 0 and gets weight 0; a row with none writes 0, as the unsplit decode does.  Queries
    with one window were written by the decode itself and are skipped.
    """
    query = tl.program_id(0)
    head_block = tl.program_id(1)
    first = tl.load(ranges_ptr + query * 2)
    count = tl.load(ranges_ptr + query * 2 + 1)
    if count > 1:
        task_floats: tl.constexpr = HEADS * (DIM + 2)
        heads = head_block * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
        cols = tl.arange(0, DIM)
        windows = tl.arange(0, MAX_WINDOWS)
        live = windows < count
        task = workspace_ptr + (first + windows).to(tl.int64) * task_floats
        o = tl.load(
            task[:, None, None] + heads[None, :, None] * DIM + cols[None, None, :],
            mask=live[:, None, None],
            other=0.0,
        )
        stat = task[:, None] + HEADS * DIM + heads[None, :]
        m2 = tl.load(stat, mask=live[:, None], other=float("-inf"))
        row_sum = tl.load(stat + HEADS, mask=live[:, None], other=0.0)
        valid = live[:, None] & (row_sum > 0.0)
        lse = tl.where(valid, m2 + tl.log2(tl.where(valid, row_sum, 1.0)), float("-inf"))
        top = tl.max(lse, axis=0)
        top = tl.where(top > float("-inf"), top, 0.0)
        w = tl.where(valid, tl.exp2(lse - top[None, :]), 0.0)
        total = tl.sum(w, axis=0)
        acc = tl.sum(tl.where(valid[:, :, None], o * w[:, :, None], 0.0), axis=0)
        out = acc / tl.where(total > 0.0, total, 1.0)[:, None]
        out = tl.where((total > 0.0)[:, None], out, 0.0)
        dst = (
            out_ptr
            + query.to(tl.int64) * out_stride_query
            + heads[:, None] * out_stride_head
            + cols[None, :]
        )
        tl.store(dst, out.to(out_ptr.dtype.element_ty))


def split_kv_max_windows(num_queries: int, clusters: int) -> int:
    """Windows per query at most, a power of two: the largest one <= clusters / queries, in
    [4, 16] (16 at 4 queries, 8 at 8, 4 from 16 with 106 clusters), or _MAX_WINDOWS rounded up."""
    override = _env_int(FP4_MLA_MXFP4_2D_SPLITKV_MAX_WINDOWS_ENV)
    if override:
        return triton.next_power_of_2(max(2, override))
    per_query = max(1, clusters // max(1, num_queries))
    return min(16, max(4, 1 << (per_query.bit_length() - 1)))


def split_kv_tasks(num_queries: int, clusters: int) -> int:
    """Task slots of a split launch of ``num_queries`` queries (fixed per CUDA-graph batch):
    whole waves of clusters, at least two slots per query, at most the window cap each."""
    waves = max(
        _env_int(FP4_MLA_MXFP4_2D_SPLITKV_WAVES_ENV) or 1,
        (2 * num_queries + clusters - 1) // clusters,
    )
    tasks = min(waves * clusters, split_kv_max_windows(num_queries, clusters) * num_queries)
    return max(tasks, num_queries + 1)


def prepare_split_kv(
    metadata: Any,
    module: Any,
    kv_lens: torch.Tensor,
    order: torch.Tensor,
    *,
    num_queries: int,
    refresh_plan: bool,
) -> Optional[SplitKvLaunch]:
    """Plan a split MXFP4-2D decode of ``num_queries`` single-token queries, or return None.

    ``order`` is the device longest-KV-first query order and ``kv_lens`` the int32 KV lengths.
    The plan runs on the device when ``refresh_plan`` (the order was recomputed: the first
    decoding layer of a forward step, or of a CUDA-graph capture) or when the launch shape
    changed; the other layers reuse it.  None when the batch exceeds the query cap or when
    no eager call has allocated the buffers before a CUDA-graph capture.
    """
    if not getattr(module, "SUPPORTS_MX2D_SPLIT_KV", False):
        return None
    max_queries = split_kv_max_queries()
    if max_queries <= 0:
        return None
    device = kv_lens.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    clusters = module.persistent_max_clusters(device)
    buffers = _BUFFERS.get(index)
    capacity_tasks = max(split_kv_tasks(b, clusters) for b in range(1, max_queries + 1))
    if (
        buffers is None or buffers[0].shape[0] < capacity_tasks or buffers[1].shape[0] < max_queries
    ) and not torch.cuda.is_current_stream_capturing():
        # Sized for every batch up to the query cap at the first eager call of any batch (the
        # warmup), so that a later capture of a small batch finds them; never allocated during
        # capture.
        header = module.MX2D_SPLIT_WS_HEADER_INTS
        task_floats = 128 * (module.TRTLLM_V_HEAD_DIM + module.SPLIT_KV_STAT_COLS)
        buffers = (
            torch.zeros(
                (capacity_tasks, module.SPLIT_KV_TASK_MAP_COLS), dtype=torch.int32, device=device
            ),
            torch.zeros((max_queries, 2), dtype=torch.int32, device=device),
            # [the work-queue counter (zero between launches; the kernel resets it) | workspace]
            torch.zeros(
                (header + capacity_tasks * task_floats,), dtype=torch.float32, device=device
            ),
        )
        _BUFFERS[index] = buffers
    if num_queries <= 0 or num_queries > max_queries:
        return None
    num_tasks = split_kv_tasks(num_queries, clusters)
    if buffers is None or buffers[0].shape[0] < num_tasks or buffers[1].shape[0] < num_queries:
        # No eager call allocated them before this capture.
        return None
    task_map_buf, ranges_buf, buffer = buffers
    task_map = task_map_buf[:num_tasks]
    ranges = ranges_buf[:num_queries]
    max_windows = split_kv_max_windows(num_queries, clusters)
    key = (
        task_map.data_ptr(),
        order.data_ptr(),
        kv_lens.data_ptr(),
        num_queries,
        num_tasks,
        max_windows,
    )
    if refresh_plan or getattr(metadata, _PLAN_KEY_ATTR, None) != key:
        min_tiles = max(1, _env_int(FP4_MLA_MXFP4_2D_SPLITKV_MIN_TILES_ENV) or _DEFAULT_MIN_TILES)
        _split_kv_plan_kernel[(num_queries,)](
            kv_lens,
            order,
            task_map,
            ranges,
            num_queries,
            num_tasks,
            min_tiles,
            max_windows,
            KV_TILE=_KV_TILE,
            BLOCK_Q=triton.next_power_of_2(num_queries),
            BLOCK_T=triton.next_power_of_2(num_tasks),
            num_warps=4,
        )
        setattr(metadata, _PLAN_KEY_ATTR, key)
    return SplitKvLaunch(task_map, buffer, ranges, max_windows, module.MX2D_SPLIT_WS_HEADER_INTS)


def merge_split_kv(launch: SplitKvLaunch, output: torch.Tensor) -> None:
    """Merge the split queries' windows into their ``output`` rows ([queries, 128, 512]):
    one program per (query, head block), a [max_windows, 16 // max_windows heads, 512] tile."""
    num_heads = output.shape[1]
    head_block = max(1, 16 // launch.max_windows)
    _split_kv_merge_kernel[(launch.ranges.shape[0], num_heads // head_block)](
        launch.buffer[launch.header_ints :],
        launch.ranges,
        output,
        output.stride(0),
        output.stride(1),
        HEADS=num_heads,
        DIM=output.shape[2],
        HEAD_BLOCK=head_block,
        MAX_WINDOWS=launch.max_windows,
        num_warps=8,
    )
