# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-stream FP4 MLA decode planner: split one decode call between the 4x kernel and the 2x packed kernel.

The 4-CTA-cluster (4x) decode kernel can only use the 92 TPC pairs that host 4-CTA clusters (184 of the 212 SMs on
the 106-TPC part); the 28 SMs on the 14 unpaired TPCs are idle for its whole duration.  The 2x packed kernel runs on
2-CTA clusters and can use them.  When the batch does not fill the 4x kernel's last wave, handing the last ``r2``
requests of the batch to the 2x packed kernel on a concurrent stream shortens the critical path: the 4x kernel loses
one whole wave while the 2x kernel finishes its share on the SMs the 4x kernel cannot use.

Measured (hecate0369, graph, sustained): B128/q4/30K 4x-only 502 us -> two-stream (r2 = 13) 467 us (-7 %), 2x
packed 535 us; B128/q4/16K 291.5 -> 275.4 us.  The gain only exists when the 4x grid is launched first (or carries a
higher priority) so that it takes the pairable TPCs before the 2x pairs land on them; see ``run_two_stream`` in
``__init__.py``.

A second split needs no cost model (``one_wave_split``): a call that is one wave for the 2x kernel and two for the 4x
kernel runs as ONE wave of the machine, a FULL-launch 4x wave next to one 2x round on the SMs that wave leaves free.

This module is host-only arithmetic (no torch import) so it can be unit-tested anywhere.  All knobs are environment
variables read at call time:

``TRTLLM_FP4_MLA_TWOSTREAM``          1 (default): automatic; 0: never split the call.
``TRTLLM_FP4_MLA_TWOSTREAM_R2``       force the number of 2x requests (0 = force the plain 4x path).
``TRTLLM_FP4_MLA_TWOSTREAM_MIN_GAIN`` minimum modelled gain over the plain 4x path (default 0.03 = 3 %).
``TRTLLM_FP4_MLA_TWOSTREAM_MAX_R2``   cap on r2 (default 36 = the largest share that still leaves the 4x arm
                                      the longer one at q4).
``TRTLLM_FP4_MLA_TWOSTREAM_T2_SCALE`` relative cost of this tree's 2x packed kernel (see ``DEFAULT_T2_SCALE``).
"""

from __future__ import annotations

import functools
import math
import os
from typing import NamedTuple, Optional

TWOSTREAM_ENV = "TRTLLM_FP4_MLA_TWOSTREAM"
TWOSTREAM_R2_ENV = "TRTLLM_FP4_MLA_TWOSTREAM_R2"
TWOSTREAM_MIN_GAIN_ENV = "TRTLLM_FP4_MLA_TWOSTREAM_MIN_GAIN"
TWOSTREAM_MAX_R2_ENV = "TRTLLM_FP4_MLA_TWOSTREAM_MAX_R2"

KV_TILE = 256
# Cost model (us) fitted to the in-kernel item timelines of 2026-11-02 (B128/q4 at 2K/4K/16K/30K, q2 at 16K):
# a 4x item = one 4-CTA cluster over 256 packed rows and the request's KV; a 2x item = one 2-CTA pair over a
# 128-row m-tile and the request's KV.  Fixed costs are prologue/epilogue, slopes are per 256-key tile.
# Not re-fitted for the 4x kernel of 2026-11-10 (shared memory by sub-bank, four-bank V ring, K128 QK tail): its 4x
# and two-stream cells run 2-14 % faster in the dispatcher matrices while the 2x kernel is unchanged, so T4 now
# overstates the 4x item.  The bias is towards a split: the 4x arm is modelled longer than it is, so the
# critical-path rule of ``_plan_two_stream`` can pass a split whose 2x arm is the longer one on the part (the gain
# gate moves less: T4 is on both sides of it).  Such a split cannot be slower than it was, only slower than plain 4x
# has become; the splits of the matrix shapes (q4: B48 r2 = 2, B64 r2 = 1, B96 r2 = 4, B256 r2 = 3) measured ahead
# of 2x in the 2026-11-10 matrix of ``fp4_mla_dispatch``.  A re-fit needs the 4x item timelines of the new kernel
# at the fitted shapes and a plain 4x arm next to the two-stream cells.
T4_FIXED_US, T4_TILE_US = 13.0, 0.64
T2_FIXED_US, T2_TILE_US = 10.0, 0.62
# Relative speed of the 2x packed kernel this tree ships against the kernel the model was fitted on (the internal
# rubin-advance mufu16 build). The GitHub-main packed path measured 15-25 % slower at B >= 48 on 2026-11-02 (B96/q4/30K
# 477 vs 392 us, B128/q4/30K 593 vs 520; the 4x kernel is identical), which makes the 2x arm the critical path of a
# B128 split there (r2 10-13 all ~490 us = plain 4x). TRTLLM_FP4_MLA_TWOSTREAM_T2_SCALE overrides; set it back to 1.0
# once the packed path is restored.
TWOSTREAM_T2_SCALE_ENV = "TRTLLM_FP4_MLA_TWOSTREAM_T2_SCALE"
DEFAULT_T2_SCALE = 1.2
# When both arms run, each slows the other (power-capped clock + memory-system contention).  Measured through the
# production two-stream path: the 4x arm's waves stretch by ~5 % at B128/q4 (2x arm busy for ~75 % of the call) and
# ~3 % at B96/q4 (busy ~45 %), so the 4x factor scales with the fraction of the call during which the 2x arm runs;
# the 2x items stretch by up to ~32 % (104-109 us vs 83 alone, single-kernel timelines).
CONTENTION_4X_FULL_OVERLAP, CONTENTION_2X = 0.07, 1.32
# The plain 4x path is not "ceil(waves)": above one wave its device split plan cuts the tail requests so the leftover
# cluster slots work on the fractional wave (B128/q4/30K: 5.57 waves measured as 484 us = 5.6 x 86 us, B96: 4.17 waves
# = 360 us), at the price of extra tasks and the reducer (~12 us).  One wave or less runs the FULL-only launch.
SPLIT_OVERHEAD_US = 12.0
# 4-CTA clusters the part can host at once (46 = 92 pairable TPC pairs / 2) and the 2-CTA pairs on the SMs left over.
LANES_4X = 46
ROWS_PER_4X_ITEM = 256
ROWS_PER_2X_ITEM = 128
DEFAULT_MIN_GAIN = 0.03
DEFAULT_MAX_R2 = 36
# Page-table capacity (tokens) from which the host launches a 4x wave of fewer than ``LANES_4X`` pair tasks in split
# mode instead of the FULL-only launch (``fp4_mla_cutedsl_4x_host.use_split_mode``: the same knob and default,
# repeated here because this module stays importable without the kernel host; the dispatch tests hold the two equal).
SPLIT_MIN_KV_ONEWAVE_ENV = "TRTLLM_FP4_MLA_4X_SPLIT_MIN_KV_ONEWAVE"
SPLIT_MIN_KV_ONEWAVE_DEFAULT = 65536


class TwoStreamPlan(NamedTuple):
    """Result of :func:`plan_two_stream`.  ``r2 == 0`` means: run the plain 4x path."""

    r2: int
    num_requests: int
    waves_4x_plain: int
    waves_4x: int
    rounds_2x: int
    model_plain_us: float
    model_us: float
    reason: str

    @property
    def enabled(self) -> bool:
        return self.r2 > 0

    @property
    def num_requests_4x(self) -> int:
        return self.num_requests - self.r2


def _env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def two_stream_enabled() -> bool:
    value = os.getenv(TWOSTREAM_ENV)
    if value is None or value.strip() == "":
        return True
    return value.strip().lower() not in ("0", "false", "off", "no")


def lanes_2x(sm_count: int, lanes_4x: int = LANES_4X) -> int:
    """2-CTA pairs available next to ``lanes_4x`` resident 4-CTA clusters (14 on the 212-SM part)."""
    return max(0, (sm_count - 4 * lanes_4x) // 2)


def t2_scale() -> float:
    value = os.getenv(TWOSTREAM_T2_SCALE_ENV)
    if value is None or value.strip() == "":
        return DEFAULT_T2_SCALE
    try:
        return max(0.1, float(value))
    except ValueError:
        return DEFAULT_T2_SCALE


def item_costs_us(kv_tiles: int) -> tuple[float, float]:
    """(4x item, 2x item) modelled duration for a request of ``kv_tiles`` 256-key tiles."""
    tiles = max(1, int(kv_tiles))
    return T4_FIXED_US + T4_TILE_US * tiles, (T2_FIXED_US + T2_TILE_US * tiles) * t2_scale()


def plain_4x_us(
    num_requests: int, rows_per_request: int, kv_tiles: int, *, lanes_4x: int = LANES_4X
) -> tuple[float, int]:
    """Modelled duration of the plain 4x path: one FULL wave, or the device split plan spreading the fractional wave
    over the leftover slots (fractional waves x item + split overhead).  Returns (us, whole waves)."""
    t4, _ = item_costs_us(kv_tiles)
    n4 = num_requests * (rows_per_request // ROWS_PER_4X_ITEM)
    if n4 <= lanes_4x:
        return t4, 1 if n4 > 0 else 0
    return n4 / lanes_4x * t4 + SPLIT_OVERHEAD_US, math.ceil(n4 / lanes_4x)


def makespan_us(
    num_requests: int,
    r2: int,
    rows_per_request: int,
    kv_tiles: int,
    *,
    lanes_4x: int = LANES_4X,
    lanes_2x_pairs: int,
) -> tuple[float, int, int]:
    """Modelled two-stream makespan with the last ``r2`` requests on the 2x arm: (us, 4x waves, 2x rounds).  The 4x
    arm runs whole waves (its requests are never split), slowed by the 2x arm for the fraction of the call the 2x arm
    is busy; the 2x arm's rounds carry their full contention factor.  ``r2 == 0`` returns the plain-path model."""
    if r2 <= 0:
        us, waves = plain_4x_us(num_requests, rows_per_request, kv_tiles, lanes_4x=lanes_4x)
        return us, waves, 0
    t4, t2 = item_costs_us(kv_tiles)
    items_per_request_4x = rows_per_request // ROWS_PER_4X_ITEM
    items_per_request_2x = rows_per_request // ROWS_PER_2X_ITEM
    n4 = (num_requests - r2) * items_per_request_4x
    n2 = r2 * items_per_request_2x
    waves_4x = math.ceil(n4 / lanes_4x) if n4 > 0 else 0
    rounds_2x = (
        math.ceil(n2 / lanes_2x_pairs)
        if (n2 > 0 and lanes_2x_pairs > 0)
        else (0 if n2 == 0 else 1 << 30)
    )
    arm_4x = waves_4x * t4
    arm_2x = rounds_2x * t2 * CONTENTION_2X
    overlap = min(1.0, arm_2x / arm_4x) if arm_4x > 0 else 0.0
    arm_4x *= 1.0 + CONTENTION_4X_FULL_OVERLAP * overlap
    return max(arm_4x, arm_2x), waves_4x, rounds_2x


def one_wave_split(
    *,
    num_requests: int,
    rows_per_request: int,
    max_kv_len: int,
    kv_capacity: int,
    sm_count: int,
    lanes_4x: int = LANES_4X,
) -> Optional[TwoStreamPlan]:
    """The split that runs a call as ONE wave of the machine, or None where there is none.

    The whole requests that fit one 4x wave go to the 4x kernel as a FULL launch and every other request to the 2x
    kernel, whose m-tiles must fit the 2-CTA pairs that wave leaves free.  A 2x wave is ``sm_count // 2`` clusters,
    two per four-CTA cluster of the 4x wave plus the unpaired SMs, so such a batch is one the 2x kernel serves in one
    wave and the 4x kernel no longer does (212 SMs: 47 to 53 pair tasks), and its 2x arm always fits.

    The 4x wave must be a FULL launch: one that whole requests fill exactly, or an under-filled one on a page table
    (``kv_capacity``) under the host's split-mode bound (``SPLIT_MIN_KV_ONEWAVE_ENV``).  In split mode the wave pays
    the plan and the reducer and its tasks take the cluster slots a FULL launch leaves to the 2x arm: measured
    11-28 % slower than 2x up to KV 2K, and 44-62 % at every KV once the 2x arm no longer fits the pairs left.

    Wave arithmetic only.  ``plan_two_stream`` declines this split because its model has one contended 2x round
    outlast one 4x wave; but what the call runs without it is not a 4x arm with the 2x arm off its critical path: it
    is a second 4x wave in split mode, or the 2x kernel alone, whose own single wave bounds the 2x arm.  Measured
    against the 2x kernel (graph replays, requests of 1 to 4 pair tasks): the split is 14-30 % faster up to KV 2K
    and level with 2x or ahead up to 30K (+2 % at worst), where the plain 4x path is 11-51 % slower up to 4K; the
    plain path ties the split at 12K (16K with one pair task per request) and wins beyond.  The KV span the split
    serves is the caller's (``fp4_mla_dispatch.extra_wave_floor_kv``).

    ``TRTLLM_FP4_MLA_TWOSTREAM=0``, a forced ``TRTLLM_FP4_MLA_TWOSTREAM_R2`` and ``TRTLLM_FP4_MLA_TWOSTREAM_MAX_R2``
    are honoured (None: the planner's answer stands).  The modelled times of the plan are informational.
    """
    if _env_int(TWOSTREAM_R2_ENV) is not None or not two_stream_enabled():
        return None
    items_4x = rows_per_request // ROWS_PER_4X_ITEM
    if rows_per_request % ROWS_PER_4X_ITEM != 0 or not 0 < items_4x <= lanes_4x:
        return None
    requests_4x = lanes_4x // items_4x
    pairs_4x = requests_4x * items_4x
    split_mode_kv = _env_int(SPLIT_MIN_KV_ONEWAVE_ENV)
    split_mode_kv = SPLIT_MIN_KV_ONEWAVE_DEFAULT if split_mode_kv is None else split_mode_kv
    if pairs_4x < lanes_4x and kv_capacity >= split_mode_kv:
        return None  # the host launches this under-filled wave in split mode
    r2 = num_requests - requests_4x
    max_r2 = _env_int(TWOSTREAM_MAX_R2_ENV)
    max_r2 = DEFAULT_MAX_R2 if max_r2 is None else max(0, max_r2)
    free_pairs = lanes_2x(sm_count, pairs_4x)
    m_tiles_2x = r2 * (rows_per_request // ROWS_PER_2X_ITEM)
    if not 1 <= r2 <= max_r2 or m_tiles_2x > free_pairs:
        return None
    kv_tiles = max(1, math.ceil(max(1, int(max_kv_len)) / KV_TILE))
    plain_us, plain_waves = plain_4x_us(num_requests, rows_per_request, kv_tiles, lanes_4x=lanes_4x)
    us, waves_4x, rounds_2x = makespan_us(
        num_requests, r2, rows_per_request, kv_tiles, lanes_4x=lanes_4x, lanes_2x_pairs=free_pairs
    )
    reason = (
        f"one wave of the machine: {pairs_4x} four-CTA clusters on the 4x kernel, {m_tiles_2x} m-tiles on the "
        f"{free_pairs} 2-CTA pairs they leave free"
    )
    return TwoStreamPlan(r2, num_requests, plain_waves, waves_4x, rounds_2x, plain_us, us, reason)


def plan_two_stream(
    *,
    num_requests: int,
    rows_per_request: int,
    max_kv_len: int,
    sm_count: int,
    lanes_4x: int = LANES_4X,
) -> TwoStreamPlan:
    """Pick how many of the LAST requests go to the 2x packed kernel (memoized on the inputs and the env knobs: a
    decode step calls this once per layer).  See :func:`_plan_two_stream`."""
    knobs = tuple(
        os.getenv(name)
        for name in (
            TWOSTREAM_ENV,
            TWOSTREAM_R2_ENV,
            TWOSTREAM_MIN_GAIN_ENV,
            TWOSTREAM_MAX_R2_ENV,
            TWOSTREAM_T2_SCALE_ENV,
        )
    )
    return _plan_two_stream_cached(
        int(num_requests),
        int(rows_per_request),
        int(max_kv_len),
        int(sm_count),
        int(lanes_4x),
        knobs,
    )


@functools.lru_cache(maxsize=256)
def _plan_two_stream_cached(
    num_requests: int,
    rows_per_request: int,
    max_kv_len: int,
    sm_count: int,
    lanes_4x: int,
    knobs: tuple,
) -> TwoStreamPlan:
    return _plan_two_stream(
        num_requests=num_requests,
        rows_per_request=rows_per_request,
        max_kv_len=max_kv_len,
        sm_count=sm_count,
        lanes_4x=lanes_4x,
    )


def _plan_two_stream(
    *,
    num_requests: int,
    rows_per_request: int,
    max_kv_len: int,
    sm_count: int,
    lanes_4x: int = LANES_4X,
) -> TwoStreamPlan:
    """Pick how many of the LAST requests go to the 2x packed kernel.

    The caller has already decided that the 4x kernel takes this call (static envelope).  ``rows_per_request`` is
    the packed Q rows of one request (heads * query_len, a multiple of 256), ``max_kv_len`` the KV capacity the
    kernels are compiled for (the live length is unknown under CUDA-graph capture; the decision only depends on it
    through the per-tile slopes, so capacity is a safe stand-in).

    Rules (all host arithmetic, deterministic):
      * the 2x arm must finish no later than the 4x arm even with the measured mutual slowdowns applied
        (``CONTENTION_2X`` / ``CONTENTION_4X``): its share is the free SMs, never the critical path;
      * the modelled makespan must improve by at least ``TRTLLM_FP4_MLA_TWOSTREAM_MIN_GAIN`` (3 %) over the plain 4x
        path, modelled WITH its device split plan (fractional waves + split overhead) and with the measured mutual
        slowdowns (``CONTENTION_4X_FULL_OVERLAP``, ``CONTENTION_2X``);
      * the split must drop a whole 4x wave: with the wave count unchanged the model credits it with nothing but
        the split-mode overhead of the plain path, which a multi-wave 4x arm pays as well (such splits measured
        level with the plain path at best, and 5-17 % behind it at KV 4K-8K on batches that fill their 4x waves
        exactly);
      * the smallest r2 that reaches the best 4x wave count wins (fewest requests moved);
      * ``TRTLLM_FP4_MLA_TWOSTREAM_R2`` overrides everything (0 forces the plain path).
    """
    kv_tiles = max(1, math.ceil(max(1, int(max_kv_len)) / KV_TILE))
    pairs_2x = lanes_2x(sm_count, lanes_4x)
    plain_us, plain_waves, _ = makespan_us(
        num_requests,
        0,
        rows_per_request,
        kv_tiles,
        lanes_4x=lanes_4x,
        lanes_2x_pairs=max(1, pairs_2x),
    )

    def result(r2: int, reason: str) -> TwoStreamPlan:
        us, w4, w2 = makespan_us(
            num_requests,
            r2,
            rows_per_request,
            kv_tiles,
            lanes_4x=lanes_4x,
            lanes_2x_pairs=max(1, pairs_2x),
        )
        return TwoStreamPlan(r2, num_requests, plain_waves, w4, w2, plain_us, us, reason)

    forced = _env_int(TWOSTREAM_R2_ENV)
    if forced is not None:
        r2 = max(0, min(forced, num_requests - 1))
        return result(r2, f"forced by {TWOSTREAM_R2_ENV}")
    if not two_stream_enabled():
        return result(0, f"disabled by {TWOSTREAM_ENV}")
    if pairs_2x <= 0 or num_requests < 2 or rows_per_request % ROWS_PER_4X_ITEM != 0:
        return result(0, "no 2x lanes or degenerate batch")
    min_gain = _env_float(TWOSTREAM_MIN_GAIN_ENV, DEFAULT_MIN_GAIN)
    max_r2 = _env_int(TWOSTREAM_MAX_R2_ENV)
    max_r2 = DEFAULT_MAX_R2 if max_r2 is None else max(0, max_r2)
    t4, t2 = item_costs_us(kv_tiles)
    best: Optional[tuple[float, int]] = None
    for r2 in range(1, min(max_r2, num_requests - 1) + 1):
        us, w4, w2 = makespan_us(
            num_requests, r2, rows_per_request, kv_tiles, lanes_4x=lanes_4x, lanes_2x_pairs=pairs_2x
        )
        if w2 * t2 * CONTENTION_2X > w4 * t4 * (1.0 + CONTENTION_4X_FULL_OVERLAP):
            break  # from here on the 2x arm is the critical path; more requests only make it worse
        if w4 < plain_waves and (best is None or us < best[0] - 1e-9):
            best = (us, r2)
    if best is None:
        return result(0, "no split drops a 4x wave with the 2x arm off the critical path")
    gain = 1.0 - best[0] / plain_us if plain_us > 0 else 0.0
    if gain < min_gain:
        return result(0, f"modelled gain {gain:.1%} below {min_gain:.0%}")
    return result(
        best[1],
        f"modelled gain {gain:.1%} ({plain_waves} -> {result(best[1], '').waves_4x} 4x waves)",
    )
