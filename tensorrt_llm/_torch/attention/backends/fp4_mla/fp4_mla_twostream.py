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

This module is host-only arithmetic (no torch import) so it can be unit-tested anywhere.  All knobs are environment
variables read at call time:

``TRTLLM_FP4_MLA_TWOSTREAM``          1 (default): automatic; 0: never split the call.
``TRTLLM_FP4_MLA_TWOSTREAM_R2``       force the number of 2x requests (0 = force the plain 4x path).
``TRTLLM_FP4_MLA_TWOSTREAM_MIN_GAIN`` minimum modelled gain over the plain 4x path (default 0.08 = 8 %).
``TRTLLM_FP4_MLA_TWOSTREAM_MAX_R2``   cap on r2 (default 36 = the largest share that still leaves the 4x arm the longer one at q4).
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
T4_FIXED_US, T4_TILE_US = 13.0, 0.64
T2_FIXED_US, T2_TILE_US = 10.0, 0.62
# When both arms run, each slows the other (power-capped clock + memory-system contention): measured per item at
# B128/q4/30K, 4x 95-98 us vs 84-87 alone, 2x 104-109 us vs 83 alone.  The critical-path rule uses these factors so the
# 2x arm never becomes the longer one in practice; the gain gate keeps the standalone model (its margin covers them).
CONTENTION_4X, CONTENTION_2X = 1.15, 1.32
# 4-CTA clusters the part can host at once (46 = 92 pairable TPC pairs / 2) and the 2-CTA pairs on the SMs left over.
LANES_4X = 46
ROWS_PER_4X_ITEM = 256
ROWS_PER_2X_ITEM = 128
DEFAULT_MIN_GAIN = 0.08
DEFAULT_MAX_R2 = 36


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


def item_costs_us(kv_tiles: int) -> tuple[float, float]:
    """(4x item, 2x item) modelled duration for a request of ``kv_tiles`` 256-key tiles."""
    tiles = max(1, int(kv_tiles))
    return T4_FIXED_US + T4_TILE_US * tiles, T2_FIXED_US + T2_TILE_US * tiles


def makespan_us(
    num_requests: int,
    r2: int,
    rows_per_request: int,
    kv_tiles: int,
    *,
    lanes_4x: int = LANES_4X,
    lanes_2x_pairs: int,
) -> tuple[float, int, int]:
    """Modelled makespan with the last ``r2`` requests on the 2x arm: (us, 4x waves, 2x rounds)."""
    t4, t2 = item_costs_us(kv_tiles)
    items_per_request_4x = rows_per_request // ROWS_PER_4X_ITEM
    items_per_request_2x = rows_per_request // ROWS_PER_2X_ITEM
    n4 = (num_requests - r2) * items_per_request_4x
    n2 = r2 * items_per_request_2x
    waves_4x = math.ceil(n4 / lanes_4x) if n4 > 0 else 0
    rounds_2x = math.ceil(n2 / lanes_2x_pairs) if (n2 > 0 and lanes_2x_pairs > 0) else (0 if n2 == 0 else 1 << 30)
    return max(waves_4x * t4, rounds_2x * t2), waves_4x, rounds_2x


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
    knobs = tuple(os.getenv(name) for name in (TWOSTREAM_ENV, TWOSTREAM_R2_ENV, TWOSTREAM_MIN_GAIN_ENV, TWOSTREAM_MAX_R2_ENV))
    return _plan_two_stream_cached(int(num_requests), int(rows_per_request), int(max_kv_len), int(sm_count), int(lanes_4x), knobs)


@functools.lru_cache(maxsize=256)
def _plan_two_stream_cached(
    num_requests: int, rows_per_request: int, max_kv_len: int, sm_count: int, lanes_4x: int, knobs: tuple
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
      * the modelled makespan must improve by at least ``TRTLLM_FP4_MLA_TWOSTREAM_MIN_GAIN`` over the plain 4x path
        (the model ignores the ~10 % slowdown the 4x arm sees from the 2x arm's power/memory pressure, so a margin
        is required: measured gains are about half the modelled ones);
      * the smallest r2 that reaches the best 4x wave count wins (fewest requests moved);
      * ``TRTLLM_FP4_MLA_TWOSTREAM_R2`` overrides everything (0 forces the plain path).
    """
    kv_tiles = max(1, math.ceil(max(1, int(max_kv_len)) / KV_TILE))
    pairs_2x = lanes_2x(sm_count, lanes_4x)
    plain_us, plain_waves, _ = makespan_us(
        num_requests, 0, rows_per_request, kv_tiles, lanes_4x=lanes_4x, lanes_2x_pairs=max(1, pairs_2x)
    )

    def result(r2: int, reason: str) -> TwoStreamPlan:
        us, w4, w2 = makespan_us(
            num_requests, r2, rows_per_request, kv_tiles, lanes_4x=lanes_4x, lanes_2x_pairs=max(1, pairs_2x)
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
        if w2 * t2 * CONTENTION_2X > w4 * t4 * CONTENTION_4X:
            break  # from here on the 2x arm is the critical path; more requests only make it worse
        if best is None or us < best[0] - 1e-9:
            best = (us, r2)
    if best is None:
        return result(0, "no split keeps the 2x arm off the critical path")
    gain = 1.0 - best[0] / plain_us if plain_us > 0 else 0.0
    if gain < min_gain:
        return result(0, f"modelled gain {gain:.1%} below {min_gain:.0%}")
    return result(best[1], f"modelled gain {gain:.1%} ({plain_waves} -> {result(best[1], '').waves_4x} 4x waves)")
