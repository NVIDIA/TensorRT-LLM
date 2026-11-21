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
"""specDSA large-batch kernels, host side: DeepSeek-V4-Flash decode for many rows on B200 (SM100, 148 SMs).

Two launches per call, back to back on the current stream. The first streams each row's indexer K cache (FP8 or MXFP4)
once, computes the indexer logits and selects the exact top-512 compressed positions in the stream (a floor learned
from the candidates already admitted in the row segment, the exact top-512 of the ones kept, exact fallbacks when too
many are kept), maps them to compressed-KV rows and publishes finished rows to a ready queue. The second, started with
programmatic dependent launch, runs the sparse MLA attention over the selected rows plus the 128-row sliding window and
the attention sink, claiming rows from the queue as they complete.

``prepare`` builds the per-step state (the CTA schedule from the compressed lengths, the expanded sinks);
``run`` launches both kernels and writes ``topk`` (int32 compressed positions) and ``out`` (bf16). The kernels are
compiled on first use per device, outside CUDA-graph capture, or ahead of it by ``precompile``.

Contract:
- A state belongs to one batch size and one caller. Its workspace (ready queue, selection scratch) is reset by the
  kernels at the end of every launch, so launches that share a state, and ``prepare`` calls that refresh it, must be
  stream-ordered.
- ``topk`` and ``out`` must not alias any input. ``out`` should be 32-byte aligned (every torch allocation is): the
  attention epilogue of an FP4 spec writes 32-byte stores into it, and an ``out`` that is only 16-byte aligned runs the
  narrower epilogue, a little slower.
- The indexer launch is persistent and needs all 148 SMs; it fails at launch on a partitioned device (MPS limits,
  green contexts).
- Block tables must be contiguous int32 and hold valid page ids for every page a row touches: ``ceil(n / 32)``
  entries for the indexer and the compressed-KV tables, ``ceil(kv_len / 128)`` for the window table.
- The kernels do not trap. A schedule that does not match ``ctx_lens`` (a stale ``prepare``) makes the launch refuse
  every row: ``topk`` is -1, ``out`` is zero and ``error_bits`` is non-zero.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch

from .config import (
    DS_V4_FLASH,
    DS_V4_FLASH_FP4,
    NUM_SMS,
    SpecDsaConfig,
    check_device,
    supported_spec,
)

__all__ = [
    "ERR_SCHEDULE",
    "SpecDsaLargeBatchState",
    "error_bits",
    "is_supported",
    "precompile",
    "prepare",
    "run",
]

# Bit of ``error_bits`` (the kernel status word): the schedule does not match ctx_lens, i.e. prepare() was not
# refreshed for these lengths.
ERR_SCHEDULE = 1

# Schedule granule: a row costs ceil(n / SCHED_UNIT) units, at least SCHED_MIN_UNITS; the CTAs split the total evenly.
SCHED_UNIT = 256
SCHED_MIN_UNITS = 128
# Below SCHED_ALIGN_BATCH rows the schedule aligns the rows to the CTAs instead (see ``_schedule``), when that costs
# the busiest CTA at most SCHED_ALIGN_SLACK_PCT percent more stream work. It gains nothing measurable above that and
# costs about 60 us of host time per call.
SCHED_ALIGN_BATCH = 80
SCHED_ALIGN_SLACK_PCT = 3
# A CTA that streams pieces of several rows pays a fresh start of the selection (floor learning, four compactions) for
# every row it touches, on top of its stream time: about 6 us (BS 32) to 15 us (BS 256) per extra row segment, i.e.
# 14 to 41 schedule units. ``_balance_cuts`` moves the cuts so that stream units plus this cost are even over the CTAs.
# Per spec: (batch size, cost of a row segment in units) points, linear in between; no entry = the even split.
_SEG_UNITS: dict = {
    DS_V4_FLASH_FP4.kernel_key: ((1, 14), (64, 14), (100, 24), (128, 38), (256, 28), (1024, 28))
}
SELECT_BINS = 512
SEGMENT_WORDS = 520  # int64 words per CTA and slot of the selection scratch
# int32 words per row of the ready queue; word AQ_ENTRY + k holds the k-th finished row.
READY_QUEUE_WORDS = 16

_KERNEL_CONFIG = dict(
    phys_block_kv=DS_V4_FLASH.index_page_tokens,
    desc=True,
    sel_fin=True,
    gath=True,
    rot=True,
    ahint=1,
    late_release=False,
    apace=10,
    sel_idle_ns=128,
    rsort=True,
    policy=1,
    cbins=SELECT_BINS,
    trig0=768,
    trig=1536,
    attn=False,
    kv_stages=4,
    q_stages=2,
)

# The knobs of the indexer kernel per spec (kernel_key). An FP4 K tile is half the bytes of an FP8 one: twice the
# stages keep the same bytes in flight. merge1: the last piece of a split row merges its row's
# pieces in one selection (two sources gathered at a time), not in one two-way selection per
# piece; fast_verdict: the schedule verdict at kernel entry issues its loads before its first branch;
# trig0: the first compaction of a segment waits for 1536 candidates; sel_fast / cmp_fast: the exact selection of the
# finalization and the in-stream compaction issue the independent work of an iteration together; sel_hand2: the selector
# warps finalize the last segment of a CTA into their own result list instead of waiting for the row warps to finish the
# earlier segment (the wait made the call 5 - 12 us slower at 40 - 92 rows). lim_loop / lim_loop2: the tile loops of the
# math warps / of the K-loading and UMMA warps end on one compare with a limit computed once per segment instead of
# the three-term condition; nolock: the two UMMA warps meet at the named barrier at a row change only, not after every
# tile; floor_mask: a math warp re-reads its candidate floor when (tile & floor_mask) == 0. All four are off here (the
# base table is the behaviour below the first variant) and on in the variants.
_KNOBS = {
    DS_V4_FLASH.kernel_key: _KERNEL_CONFIG,
    DS_V4_FLASH_FP4.kernel_key: dict(
        _KERNEL_CONFIG,
        kv_stages=8,
        merge1=2,
        fast_verdict=True,
        trig0=1536,
        sel_fast=True,
        cmp_fast=True,
        sel_hand2=True,
        lim_loop=False,
        lim_loop2=False,
        nolock=False,
        floor_mask=6,
    ),
}

# Batch-size ranges that run the indexer kernel with other knobs: (first batch size, overrides), the last range whose
# first batch size is not above the batch size wins. A range is a second compile of the same kernel class. The first
# range starts at the first large-batch size, so the base knobs above only serve launches forced below it. The four
# selector warps poll the candidate rings every sel_idle_ns: the polls compete with the TMA / MMA barriers of the SM and
# with the power budget (a 1.5 us sleep is 1 - 2 % faster at 100 - 150 rows and 4 - 7 % from 192 rows on), and six K
# stages (12 tiles in flight instead of 16) share the HBM more evenly between the SMs (the slowest SMs set the end of
# the call): with the loop-control and lockstep knobs that is 0.9 - 2 % faster than the base knobs at 27 - 56 rows
# (eight stages with those knobs are 3 - 4 % slower at 47 - 64 rows). At 512 rows and more the SM clock sits at about
# 1.4 GHz (power cap) and the stream is bound by the ~290 SM cycles a math warp spends per 128-position tile
# (hand-offs, the 64-head epilogue, floor read, compare, loop control), more than the HBM time of a tile (184 ns): the
# loop-control, lockstep and floor-read knobs take instructions off that chain (a 4 times rarer floor read from 192
# rows: every 16th tile instead of every 4th), and from 768 rows the selector warps sleep 3 us between polls and the K
# stages are eight again (in the nsys protocol six stages are better at 384 - 767 rows, eight at 768 and more).
_VARIANTS = {
    DS_V4_FLASH_FP4.kernel_key: (
        (
            DS_V4_FLASH_FP4.megakernel_max_batch + 1,
            dict(sel_idle_ns=1536, kv_stages=6, lim_loop=True, lim_loop2=True, nolock=True),
        ),
        (
            192,
            dict(
                sel_idle_ns=1536,
                kv_stages=6,
                lim_loop=True,
                lim_loop2=True,
                nolock=True,
                floor_mask=30,
            ),
        ),
        (
            768,
            dict(sel_idle_ns=3072, lim_loop=True, lim_loop2=True, nolock=True, floor_mask=30),
        ),
    ),
}

# Knobs of the attention kernel per spec (kernel_key); no entry = the shipped kernel. exit_early: the "last CTA to leave
# resets the ready queue" accounting runs right after the CTA's final claim, off the critical path of the last row.
# wide_store: the output epilogue writes 32-byte stores, half the store instructions of 16-byte ones (the epilogue is
# ~3 us of the ~13 us row latency that ends every call); it needs a 32-byte aligned ``out``, so a second compile without
# it serves an ``out`` that is only 16-byte aligned.
_TAIL_KNOBS: dict = {DS_V4_FLASH_FP4.kernel_key: dict(exit_early=True, wide_store=True)}

_lock = threading.Lock()
_engines: dict[tuple, dict] = {}


def _kernel_modules() -> tuple:
    from . import large_batch, large_batch_lib

    return large_batch, large_batch_lib


def is_supported(
    device: torch.device | None = None, config: SpecDsaConfig | None = None
) -> tuple[bool, str]:
    """Whether the large-batch kernels serve ``config`` (default: DS-V4-Flash) on ``device``.

    That takes a supported spec with large-batch kernels, CuTe DSL installed and SM100 with all 148 SMs (one CTA per
    SM).
    """
    try:
        if not supported_spec(config).large_batch:
            return False, "the spec has no large-batch kernels"
    except NotImplementedError as e:
        return False, str(e)
    return check_device(device)


def _check_batch_size(batch_size: int, config: SpecDsaConfig) -> None:
    if not 1 <= batch_size <= config.max_batch:
        raise ValueError(f"batch size {batch_size} not in [1, {config.max_batch}]")


def _device_index(device: torch.device) -> int:
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def _engine(device: torch.device, config: SpecDsaConfig | None = None) -> dict:
    config = supported_spec(config)
    if not config.large_batch:
        raise ValueError(f"{config.name} has no large-batch kernels")
    key = (_device_index(device), config.kernel_key)
    with _lock:
        if key not in _engines:
            lb, _ = _kernel_modules()
            knobs = _KNOBS[config.kernel_key]
            tail_knobs = _TAIL_KNOBS.get(config.kernel_key, {})
            _engines[key] = {
                "indexer": lb.IndexerKernel(num_sms=NUM_SMS, config=config, **knobs),
                "indexer_fn": None,
                # [first batch size, kernel, compiled function]
                "variants": [
                    [
                        first,
                        lb.IndexerKernel(num_sms=NUM_SMS, config=config, **{**knobs, **over}),
                        None,
                    ]
                    for first, over in _VARIANTS.get(config.kernel_key, ())
                ],
                "tail": lb.AttentionTail(config=config, **tail_knobs),
                "tail_fn": None,
                # the tail for an ``out`` that is not 32-byte aligned (only with wide_store)
                "tail_narrow": (
                    lb.AttentionTail(config=config, **{**tail_knobs, "wide_store": False})
                    if tail_knobs.get("wide_store")
                    else None
                ),
                "tail_narrow_fn": None,
            }
        return _engines[key]


def _not_compiled(e: dict) -> bool:
    return (
        e["indexer_fn"] is None
        or e["tail_fn"] is None
        or (e.get("tail_narrow") is not None and e.get("tail_narrow_fn") is None)
        or any(v[2] is None for v in e.get("variants", ()))
    )


def precompile(device: torch.device, config: SpecDsaConfig | None = None) -> None:
    """Compile both kernels for ``device`` (the indexer kernel takes about a minute per knob range).

    The compiled kernels serve every batch size (a knob range of the indexer kernel is a compile of its own). Must run
    outside CUDA-graph capture; ``run`` compiles on first use otherwise.
    """
    lb, _ = _kernel_modules()
    e = _engine(device, config)
    with _lock:
        if _not_compiled(e):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "spec_dsa (large batch): compile the kernels (precompile) before "
                    "CUDA-graph capture"
                )
            if e["indexer_fn"] is None:
                e["indexer_fn"] = lb.compile_indexer(e["indexer"])
            for v in e.get("variants", ()):
                if v[2] is None:
                    v[2] = lb.compile_indexer(v[1])
            if e["tail_fn"] is None:
                e["tail_fn"] = lb.compile_tail(e["tail"])
            if e.get("tail_narrow") is not None and e.get("tail_narrow_fn") is None:
                e["tail_narrow_fn"] = lb.compile_tail(e["tail_narrow"])


@dataclass
class SpecDsaLargeBatchState:
    """Per-step metadata and workspace of one caller; see ``prepare``.

    ``sched`` is the CTA schedule (row and unit offset where each CTA starts, one extra entry for the end),
    ``sinks`` the per-row copy of the attention sink. ``ready_queue`` carries finished rows from the indexer to the
    attention kernel. ``selection_ws`` and ``counters`` are per-row selection scratch; ``gidx`` holds the 640 KV row
    indices of each row (128 window rows, then the selection). ``status`` is the kernel status word (``error_bits``).
    ``lg``, ``req`` and ``stamps`` are kernel arguments that carry no state. ``config`` is the spec the state was built
    for.
    """

    batch_size: int
    sched: torch.Tensor
    sinks: torch.Tensor
    req: torch.Tensor
    gidx: torch.Tensor
    selection_ws: torch.Tensor
    counters: torch.Tensor
    ready_queue: torch.Tensor
    stamps: torch.Tensor
    lg: torch.Tensor
    status: torch.Tensor
    config: SpecDsaConfig


def _seg_units(config: SpecDsaConfig, batch_size: int) -> int:
    """Cost of one row segment of a CTA in schedule units for ``batch_size`` rows (0 = not modelled)."""
    points = _SEG_UNITS.get(config.kernel_key)
    if not points:
        return 0
    xs, ys = zip(*points)
    return int(round(float(np.interp(batch_size, xs, ys))))


def _balance_cuts(prefix: np.ndarray, num_ctas: int, seg_units: int) -> np.ndarray:
    """CTA cuts (positions in the concatenated unit space, ``num_ctas + 1`` of them) that make the stream units of a CTA
    plus ``seg_units`` per row it touches even over the CTAs.

    A CTA that covers ``[a, b)`` touches the row boundaries strictly inside plus one row, so its cost is
    ``G(b) - G(a) + seg_units`` with ``G(x) = x + seg_units * #{row starts below x}``: an even split of ``G`` is the
    balanced one. A cut that falls into the jump of ``G`` at a row start is placed on that row start.
    """
    total = int(prefix[-1])
    inner = prefix[1:-1]  # starts of rows 1..B-1
    s = np.arange(num_ctas + 1, dtype=np.float64)
    y = s * ((total + seg_units * inner.size) / num_ctas)
    if inner.size == 0:
        x = y
    else:
        g_left = inner + seg_units * np.arange(inner.size)  # G just below the start of row j + 1
        m = np.searchsorted(g_left + seg_units, y, side="left")  # starts passed completely
        x = y - seg_units * m
        mc = np.minimum(m, inner.size - 1)
        x = np.where((m < inner.size) & (y > g_left[mc]), inner[mc], x)
    cuts = np.rint(x).astype(np.int64)
    cuts[0], cuts[-1] = 0, total
    return np.maximum.accumulate(cuts)


def _schedule(ctx_lens: list[int], num_ctas: int, seg_units: int = 0) -> np.ndarray:
    """CTA schedule: CTA ``s`` starts at unit ``s * quota + min(s, rem)`` of the rows' concatenated unit ranges.

    Row ``r`` spans ``max(ceil(n_r / SCHED_UNIT), SCHED_MIN_UNITS)`` units; entry ``s`` is ``(row, unit offset in
    the row)`` of that start (offset 0 for rows with fewer than ``SCHED_MIN_UNITS`` real units, which are never
    split) and entry ``num_ctas`` is ``(batch_size, 0)``. With ``seg_units`` > 0 the cuts are moved by
    ``_balance_cuts`` so that a CTA that touches more rows gets fewer units.

    With fewer than ``SCHED_ALIGN_BATCH`` rows the even split cuts rows at arbitrary units, and every extra piece of
    a row is one more candidate list for the last finisher of the row to merge. ``_aligned_schedule`` then cuts the
    rows into fewer pieces, one per CTA, and is used when it has fewer pieces and costs the busiest CTA at most
    ``SCHED_ALIGN_SLACK_PCT`` percent more stream work.
    """
    n = np.asarray(ctx_lens, dtype=np.int64)
    batch = n.size
    units = -(-n // SCHED_UNIT)
    cost = np.maximum(units, SCHED_MIN_UNITS)
    prefix = np.concatenate([[0], np.cumsum(cost)])
    quota, rem = divmod(int(prefix[-1]), num_ctas)
    s = np.arange(num_ctas + 1, dtype=np.int64)
    start = s * quota + np.minimum(s, rem)
    if seg_units > 0:
        start = _balance_cuts(prefix, num_ctas, seg_units)
    row = np.searchsorted(prefix[1:], start, side="right")
    inside = row < batch
    row_c = np.minimum(row, batch - 1)
    offset = np.where(inside & (units[row_c] >= SCHED_MIN_UNITS), start - prefix[row_c], 0)
    sched = np.stack([np.where(inside, row, batch), offset], axis=1).astype(np.int32)
    if batch >= SCHED_ALIGN_BATCH or batch > num_ctas:
        return sched
    aligned = _aligned_schedule(units, num_ctas)
    pieces = int((aligned[:, 0] < batch).sum())
    # selection pieces of the even split: the rows it touches plus the rows it cuts in the middle
    first_row, first_off = sched[:-1, 0], sched[:-1, 1]
    next_row, next_off = sched[1:, 0], sched[1:, 1]
    moved = (next_row != first_row) | (next_off != first_off)
    even_pieces = int(((next_row - first_row) + (next_off > 0))[moved].sum())
    busiest_aligned = _busiest_cta_units(units, aligned, seg_units)
    busiest_even = _busiest_cta_units(units, sched, seg_units)
    if pieces < even_pieces and busiest_aligned * 100 <= busiest_even * (
        100 + SCHED_ALIGN_SLACK_PCT
    ):
        return aligned
    return sched


def _busiest_cta_units(units: np.ndarray, sched: np.ndarray, seg_units: int = 0) -> int:
    """Stream units (``SCHED_UNIT`` positions of the rows, without the minimum cost) of the busiest CTA of ``sched``,
    plus ``seg_units`` for every row segment of the CTA."""
    work = np.concatenate([[0], np.cumsum(units)])
    pos = work[sched[:, 0]] + sched[:, 1]
    cost = np.diff(pos)
    if seg_units > 0:
        rows = (np.diff(sched[:, 0]) + (sched[1:, 1] > 0)).clip(min=0)
        cost = cost + seg_units * np.where(cost > 0, rows, 0)
    return int(cost.max())


def _aligned_schedule(units: np.ndarray, num_ctas: int) -> np.ndarray:
    """Cut every row of at least ``SCHED_MIN_UNITS`` units into the fewest equal pieces that fit one per CTA.

    A smallest piece size is searched such that the pieces of all rows (a shorter row stays in one piece) number at
    most ``num_ctas``; a row of ``u`` units is cut into ``ceil(u / size)`` pieces that differ by at most one unit.
    Returns the entries of ``_schedule`` for one piece per CTA, the unused CTAs and the end entry being
    ``(batch_size, 0)``.
    """
    batch = units.size
    small = units < SCHED_MIN_UNITS
    long_rows = units[~small]
    lo = max(1, int(units[small].max()) if small.any() else 1)
    hi = int(units.max())
    while lo < hi:
        mid = (lo + hi) // 2
        if batch - long_rows.size + int(((long_rows + (mid - 1)) // mid).sum()) <= num_ctas:
            hi = mid
        else:
            lo = mid + 1
    count = np.where(small, 1, (units + (lo - 1)) // lo)
    row = np.repeat(np.arange(batch), count)
    nth = np.arange(row.size) - (np.cumsum(count) - count)[row]  # index of the piece inside its row
    base, extra = np.divmod(units, count)
    entries = np.full((num_ctas + 1, 2), 0, dtype=np.int32)
    entries[:, 0] = batch
    entries[: row.size, 0] = row
    entries[: row.size, 1] = nth * base[row] + np.minimum(nth, extra[row])
    return entries


def _host_ints(t: torch.Tensor, count: int, name: str) -> list[int]:
    v = t.reshape(-1)
    if v.numel() < count:
        raise ValueError(f"{name} has {v.numel()} entries, need {count}")
    return [int(x) for x in v[:count].tolist()]


def prepare(
    ctx_lens: torch.Tensor,
    sinks: torch.Tensor,
    state: SpecDsaLargeBatchState | None = None,
    *,
    config: SpecDsaConfig | None = None,
) -> SpecDsaLargeBatchState:
    """Build (or refresh in place) the per-step state for ``B = ctx_lens.numel()`` rows.

    Args:
        ctx_lens: int32 [B], compressed (indexer) length of each row, in [512, 2**18] for DS-V4-Flash.
        sinks: float32 [64] on the target device, per-head attention sink.
        state: a state of the same batch size to refresh; keeps its workspace and its spec.
        config: the spec (default: DS-V4-Flash, the only one the kernels serve).

    The lengths are read on the host: pass a host (CPU) tensor to avoid a device-to-host copy; a device tensor works
    but synchronizes.
    """
    if state is not None:
        if config is not None and config.kernel_key != state.config.kernel_key:
            raise ValueError(f"state is for {state.config.name!r}, got {config.name!r}")
        config = state.config
    config = supported_spec(config)
    batch_size = int(ctx_lens.numel())
    _check_batch_size(batch_size, config)
    device = sinks.device
    if device.type != "cuda":
        raise ValueError("sinks must be a CUDA tensor")
    if sinks.numel() != config.attn_heads:
        raise ValueError(f"sinks must have {config.attn_heads} entries, got {tuple(sinks.shape)}")
    ctx_host = _host_ints(ctx_lens, batch_size, "ctx_lens")
    if any(not config.min_index_len <= n <= config.max_index_len for n in ctx_host):
        raise ValueError(
            f"compressed lengths must lie in [{config.min_index_len}, {config.max_index_len}]"
        )
    if state is not None and state.batch_size != batch_size:
        raise ValueError(f"state is for {state.batch_size} rows, got {batch_size}")
    _, lib = _kernel_modules()
    if state is None:
        ready_queue = torch.zeros((batch_size, READY_QUEUE_WORDS), dtype=torch.int32, device=device)
        ready_queue.view(-1)[lib.AQ_ENTRY : lib.AQ_ENTRY + batch_size] = -1
        state = SpecDsaLargeBatchState(
            batch_size=batch_size,
            sched=torch.empty((NUM_SMS + 1, 2), dtype=torch.int32, device=device),
            sinks=torch.empty((batch_size, config.attn_heads), dtype=torch.float32, device=device),
            req=torch.arange(batch_size, dtype=torch.int32, device=device),
            gidx=torch.full(
                (batch_size, config.selected_rows), -9, dtype=torch.int32, device=device
            ),
            selection_ws=torch.zeros(3 * NUM_SMS * SEGMENT_WORDS, dtype=torch.int64, device=device),
            counters=torch.zeros(batch_size * 16, dtype=torch.int64, device=device),
            ready_queue=ready_queue,
            stamps=torch.zeros(NUM_SMS * 256, dtype=torch.int64, device=device),
            lg=torch.empty((1, 256), dtype=torch.float32, device=device),
            status=torch.full((4,), -7, dtype=torch.int32, device=device),
            config=config,
        )
    state.sched.copy_(
        torch.from_numpy(_schedule(ctx_host, NUM_SMS, _seg_units(config, batch_size))),
        non_blocking=True,
    )
    state.sinks.copy_(sinks.reshape(1, config.attn_heads).expand(batch_size, config.attn_heads))
    return state


def error_bits(state: SpecDsaLargeBatchState) -> torch.Tensor:
    """Status word of the last launch of ``state`` (``ERR_*`` bits): an int32 [1] device view, no sync.

    Zero means the last launch ran normally; a fresh state reads -7 until its first launch.
    """
    return state.status[:1]


def _check_numel(name: str, t: torch.Tensor, numel: int, dtype: torch.dtype | None = None) -> None:
    if t.numel() != numel or (dtype is not None and t.dtype != dtype) or not t.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous with {numel} elements"
            + (f" of {dtype}" if dtype is not None else "")
            + f", got {t.dtype} {tuple(t.shape)}"
        )


def _check_rows(name: str, t: torch.Tensor, row_bytes: int) -> None:
    if t.element_size() != 1 or t.dim() < 2 or t.shape[-1] != row_bytes or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous 1-byte tensor with {row_bytes}-byte rows, "
            f"got {t.dtype} {tuple(t.shape)}"
        )


def _check_block_table(name: str, t: torch.Tensor, batch_size: int) -> None:
    if t.dtype != torch.int32 or t.dim() != 2 or t.shape[0] < batch_size or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous int32 [B, pages] tensor with B >= {batch_size}, "
            f"got {t.dtype} {tuple(t.shape)}"
        )


def run(
    state: SpecDsaLargeBatchState,
    q_idx: torch.Tensor,
    w_idx: torch.Tensor,
    k_cache: torch.Tensor,
    idx_block_table: torch.Tensor,
    ctx_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    q_att: torch.Tensor,
    swa_pool: torch.Tensor,
    bt_swa: torch.Tensor,
    c_pool: torch.Tensor,
    bt_c: torch.Tensor,
    topk: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    q_scale: torch.Tensor | None = None,
) -> None:
    """Launch the two kernels on the current stream.

    Args:
        state: from ``prepare`` for these lengths.
        q_idx: float8_e4m3fn [B, 64, 128] (or [B, 1, 64, 128]), indexer queries; for an FP4 spec packed E2M1, 1 byte
            per two values ([B, 64, 64]).
        w_idx: float32 [B, 64], indexer head weights.
        k_cache: uint8 [pages, ...] with ``config.index_page_bytes`` bytes per page: 4224 for FP8 (32 x 128 FP8 values,
            then 32 float32 scales), 2176 for FP4 (32 x 64 bytes of E2M1 pairs, then 32 int32 of four UE8M0 scales, one
            per 32 values).
        idx_block_table: int32 [B, pages], indexer K pages of each row.
        ctx_lens: int32 [B] on the device, the same lengths as given to ``prepare``.
        kv_lens: int32 [B] on the device, uncompressed length of each row (>= 128); the window is its last 128
            positions.
        q_att: float8_e4m3fn [B, 64, 512], attention queries.
        swa_pool: float8_e4m3fn [..., 512], sliding-window KV rows (128 rows per page); bt_swa: int32 [B, pages].
        c_pool: float8_e4m3fn [..., 512], compressed KV rows (32 rows per page); bt_c: int32 [B, pages].
        topk: int32 [B, 512] output, the selected compressed positions (in no particular order).
        out: bfloat16 [B, 64 * 512] output (32-byte aligned for the fastest epilogue; 16 bytes is enough).
        softmax_scale: attention softmax scale (default: ``1 / sqrt(kv_dim)`` of the spec).
        q_scale: int32 [B, 64] (or [B, 1, 64]), the four UE8M0 query scales of each head in one int32 (FP4 specs only).

    The shapes are those of the DS-V4-Flash spec; ``state.config`` gives the sizes the checks use.
    """
    config = state.config
    b = state.batch_size
    fp4 = config.index_k_dtype == "fp4"
    if softmax_scale is None:
        softmax_scale = config.default_softmax_scale
    q_values = config.index_head_dim // 2 if fp4 else config.index_head_dim
    _check_numel("q_idx", q_idx, b * config.index_heads * q_values)
    if fp4:
        if q_scale is None:
            raise ValueError("an FP4 spec needs q_scale (the UE8M0 scales of the FP4 queries)")
        _check_numel("q_scale", q_scale, b * config.index_heads, torch.int32)
    elif q_scale is not None:
        raise ValueError("q_scale belongs to FP4 specs")
    if q_idx.element_size() != 1:
        raise ValueError(f"q_idx must be FP8 (1 byte per value), got {q_idx.dtype}")
    _check_numel("w_idx", w_idx, b * config.index_heads, torch.float32)
    _check_numel("q_att", q_att, b * config.attn_heads * config.kv_dim)
    if q_att.element_size() != 1:
        raise ValueError(f"q_att must be FP8 (1 byte per value), got {q_att.dtype}")
    _check_numel("ctx_lens", ctx_lens, b, torch.int32)
    _check_numel("kv_lens", kv_lens, b, torch.int32)
    _check_numel("topk", topk, b * config.topk, torch.int32)
    _check_numel("out", out, b * config.attn_heads * config.kv_dim, torch.bfloat16)
    _check_rows("swa_pool", swa_pool, config.kv_dim)
    _check_rows("c_pool", c_pool, config.kv_dim)
    for name, table in (("idx_block_table", idx_block_table), ("bt_swa", bt_swa), ("bt_c", bt_c)):
        _check_block_table(name, table, b)
    pages = int(k_cache.shape[0])
    if k_cache.element_size() != 1 or k_cache.numel() != pages * config.index_page_bytes:
        raise ValueError(
            f"k_cache must hold {config.index_page_bytes} bytes per page, got {tuple(k_cache.shape)}"
        )
    e = _engine(q_idx.device, config)
    if _not_compiled(e):
        precompile(q_idx.device, config)
    indexer_fn = e["indexer_fn"]
    for first, _, fn in e.get("variants", ()):
        if b >= first:
            indexer_fn = fn
    q_att8 = q_att.view(torch.uint8).view(b, config.attn_heads, config.kv_dim)
    swa8 = swa_pool.view(torch.uint8).view(-1, config.kv_dim)
    c8 = c_pool.view(torch.uint8).view(-1, config.kv_dim)
    out3 = out.view(b, config.attn_heads, config.kv_dim)
    indexer_fn(
        k_cache.view(pages, -1),
        q_idx.reshape(-1)
        .as_strided(
            (config.index_heads, q_values, b),
            (q_values, 1, config.index_heads * q_values),
        )
        .view(torch.uint8),
        w_idx.reshape(-1).as_strided((config.index_heads, b), (1, config.index_heads)),
        state.lg,
        idx_block_table[:b],
        ctx_lens,
        state.sched,
        kv_lens,
        state.req,
        bt_swa[:b],
        bt_c[:b],
        topk,
        state.gidx,
        state.selection_ws,
        state.counters,
        state.ready_queue,
        state.stamps,
        q_att8,
        swa8,
        c8,
        state.sinks,
        out3,
        state.status,
        softmax_scale,
        pages,
        b,
        *(
            (q_scale.reshape(-1).as_strided((config.index_heads, b), (1, config.index_heads)),)
            if fp4
            else ()
        ),
    )
    tail_fn = e["tail_fn"]
    if e.get("tail_narrow") is not None and out.data_ptr() % 32:
        tail_fn = e["tail_narrow_fn"]
    tail_fn(
        q_att8,
        swa8,
        c8,
        state.gidx,
        state.sinks,
        out3,
        state.ready_queue,
        softmax_scale,
        NUM_SMS,
    )
