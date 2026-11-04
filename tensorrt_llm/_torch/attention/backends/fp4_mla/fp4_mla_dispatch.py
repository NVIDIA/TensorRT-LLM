# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA decode path selection: ``2x`` packed (default) | ``4x`` four-CTA-cluster | ``twostream`` (4x arm + 2x arm).

The decision is host arithmetic on static shapes plus the KV length, so it is the same in the eager warm-up, at
CUDA-graph capture and (through the graph that was captured) at replay.  Measured basis (hecate0358/0369, Rubin
212 SMs, 128 heads, graph replays, 2026-11-02; see the two-stream planner for the item cost model):

======================================  ==========  ===========  ===================  ===================
shape                                   2x packed   4x one-shot  two-stream           best
======================================  ==========  ===========  ===================  ===================
B128/q4, KV 30K                         520 us      484 us       456 us (r2 = 13)     two-stream (-12 %)
B128/q4, KV 16K                         299         286          272 (r2 = 13)        two-stream (-9 %)
B96/q4, KV 30K                          392         360          349 (r2 = 4)         two-stream (-11 %)
B64/q4, KV 30K                          274         244          = 4x (r2 = 0)        4x (-10 %)
B32/q4, KV 30K                          163         131          = 4x                 4x (-19 %)
B128/q2, KV 16K                         195         159          = 4x                 4x (-18 %)
B128/q4, KV 8K                          172         176          171 (r2 = 13)        tie (within 2 %)
B128/q4, KV 4K                          101         102          102                  tie
======================================  ==========  ===========  ===================  ===================

Graph-replay medians throughout (the 2x kernel's first replays at 8K are faster, 158 us cold, which is the only
sense in which 2x "wins" there); on the main tree's slower 2x packed kernel (#20576) the 8K row read 190 / 191 /
192 and 4K 106 / 107 / 107 on 2026-11-02 (hecate0358) and 182 / 178 / 178 and 110 / 116 / 116 on 2026-11-03
(hecate0341, the matrix below): B128 at 8K is a tie within the node-to-node spread either way, which is why the
many-wave floor sits there.

Below the batch's KV floor (``four_x_floor_kv``) the 2x packed kernel keeps the call: the 4x kernel pays ~13 us of
fixed cost per 256-row item (task identity, barrier init, exposed Q load, pipeline fill and drain) and every wave
beyond the first adds a task per item, so the KV it needs to win grows with the batch's 4x wave count.  Graph-replay
matrix of 2026-11-03 (q4, H128, uniform KV, main tree; ratio = 2x / 4x-or-two-stream):

=========================  ======  ======  ======  ======  ======  ======  ==========
batch (4x waves)           1K      2K      4K      8K      12K     16K     floor
=========================  ======  ======  ======  ======  ======  ======  ==========
B16 (0.7, one wave)        1.23x   2.51x   2.23x   1.93x   1.83x   1.73x   0
B32 (1.4)                  0.95x   1.03x   1.13x   1.22x   1.33x   1.41x   2K
B48 (2.1, two-stream r2=2) 0.90x   0.91x   1.03x   1.10x   1.17x   1.27x   4K
B64 (2.8)                  0.90x   1.01x   1.08x   1.17x   1.25x   1.27x   4K
B96 (4.2)                  0.86x   0.87x   0.92x   1.06x   1.18x   1.18x   8K
B128 (5.6)                 0.86x   0.88x   0.95x   1.02x   1.14x   1.15x   8K
B256 (11.1)                        0.88x   0.95x   1.09x           1.19x   8K
=========================  ======  ======  ======  ======  ======  ======  ==========

Above the floor the static 4x envelope (``fp4_mla_cutedsl_4x_host.envelope_reason_static``:
small-batch rules, Helix / softmax-stats / packed-V requirements, split-mode bounds) decides whether the 4x kernel
may run, and the two-stream planner (``fp4_mla_twostream``) decides whether handing the last requests to the 2x
kernel on a concurrent stream drops a whole 4x wave.

KV length used for the decision: ``live_max_kv`` when the caller knows the step's real longest KV (eager mode
narrows the page-table capacity to it; a graph runner that keys its graphs by KV regime can pass the regime
ceiling through ``Fp4MlaState.kv_regime_max_kv``), otherwise the page-table capacity.  The kernels themselves are
always compiled and planned for the capacity.

Environment:
``TRTLLM_FP4_MLA_CUTEDSL_4X``       master switch (default off): 0 -> always ``2x``.
``TRTLLM_FP4_MLA_DECODE_PATH``      ``auto`` (default) | ``2x`` | ``4x`` | ``twostream``: force a path (a forced 4x or
                                    two-stream path still needs the static envelope; ``twostream`` falls back to ``4x``
                                    when the planner finds no useful split).
``TRTLLM_FP4_MLA_4X_MIN_KV``        one flat KV length (tokens) below which every call stays on ``2x``, replacing
                                    the per-batch floor of ``four_x_floor_kv`` (0 / 2048 / 4096 / 8192 by 4x waves).
``TRTLLM_FP4_MLA_KV_REGIME_GRID``   extra KV regime ceilings (tokens, comma-separated) a graph runner keys its
                                    decode graphs by between a batch's floor and the capacity (default none: one 2x
                                    regime under the floor and one 4x-eligible regime up to the capacity): see
                                    ``kv_regimes_for_batch``.
"""

from __future__ import annotations

import os
from typing import Callable, NamedTuple, Optional, Sequence

from .fp4_mla_twostream import LANES_4X, ROWS_PER_4X_ITEM, TwoStreamPlan, plan_two_stream

DECODE_PATH_ENV = "TRTLLM_FP4_MLA_DECODE_PATH"
MIN_KV_4X_ENV = "TRTLLM_FP4_MLA_4X_MIN_KV"
# KV floor (tokens) of the 4x paths by the batch's 4x wave count (``four_x_waves``): (waves at most, floor) in
# ascending order, ``FLOOR_KV_4X_MANY_WAVES`` beyond the last bound.  From the matrix in the module docstring.
FLOOR_KV_4X_BY_WAVES = ((1.0, 0), (1.5, 2048), (3.0, 4096))
FLOOR_KV_4X_MANY_WAVES = 8192

PATH_2X = "2x"
PATH_4X = "4x"
PATH_TWO_STREAM = "twostream"
PATHS = (PATH_2X, PATH_4X, PATH_TWO_STREAM)

# KV regimes a graph runner can key its decode graphs by (tokens): ``kv_regimes_for_batch`` lists the regimes worth
# a graph for one batch shape, ``kv_regime_for_step`` maps a step's longest live KV onto them; the page-table
# capacity itself is always the last regime, the grid adds optional points between a batch's floor and the capacity.
KV_REGIME_GRID_ENV = "TRTLLM_FP4_MLA_KV_REGIME_GRID"
DEFAULT_KV_REGIME_GRID: tuple[int, ...] = ()


class DecodeStrategy(NamedTuple):
    path: str
    r2: int  # requests on the 2x arm when path == twostream, else 0
    reason: str
    kv_len_decision: int  # KV length the decision was taken on
    kv_source: str  # "live" | "capacity" | "forced"
    plan: Optional[TwoStreamPlan]

    @property
    def num_requests_2x(self) -> int:
        return self.r2 if self.path == PATH_TWO_STREAM else 0

    @property
    def record_tag(self) -> Optional[str]:
        """Tag for the dispatch recorder (``ts<r2>`` for two-stream, ``kv<decision>`` when live KV was used)."""
        parts = []
        if self.path == PATH_TWO_STREAM:
            parts.append(f"ts{self.r2}")
        if self.kv_source != "capacity":
            parts.append(f"kv{self.kv_len_decision}")
        return "|".join(parts) if parts else None


def _env_path() -> str:
    value = (os.getenv(DECODE_PATH_ENV) or "auto").strip().lower()
    return value if value in PATHS else "auto"


def min_kv_4x() -> Optional[int]:
    """Flat KV floor from ``TRTLLM_FP4_MLA_4X_MIN_KV``; None (the default) grades the floor by the batch."""
    value = os.getenv(MIN_KV_4X_ENV)
    if value is None or value.strip() == "":
        return None
    try:
        return max(0, int(value))
    except ValueError:
        return None


def four_x_waves(num_requests: int, rows_per_seq: int, sm_count: int) -> float:
    """4x waves of one call: its 256-row pair tasks over the four-CTA clusters the part hosts at once."""
    del sm_count  # the resident cluster count is a property of the part the kernel is tuned for
    pairs = max(0, int(num_requests)) * max(0, int(rows_per_seq)) // ROWS_PER_4X_ITEM
    return pairs / LANES_4X


def four_x_floor_kv(num_requests: int, rows_per_seq: int, sm_count: int) -> int:
    """KV length (tokens) from which the 4x paths may take a call of this shape; shorter steps stay on 2x.

    Graded by the call's 4x wave count (``FLOOR_KV_4X_BY_WAVES``, measured table in the module docstring): one wave
    wins at every KV, each further wave adds a fixed-cost task per item that only longer KV amortizes.
    ``TRTLLM_FP4_MLA_4X_MIN_KV`` replaces the grading by one flat floor.
    """
    flat = min_kv_4x()
    if flat is not None:
        return flat
    waves = four_x_waves(num_requests, rows_per_seq, sm_count)
    for bound, floor in FLOOR_KV_4X_BY_WAVES:
        if waves <= bound:
            return floor
    return FLOOR_KV_4X_MANY_WAVES


def kv_regime_grid() -> tuple[int, ...]:
    """Extra regime ceilings between floor and capacity: ``TRTLLM_FP4_MLA_KV_REGIME_GRID`` (default none)."""
    value = os.getenv(KV_REGIME_GRID_ENV)
    if value is None or value.strip() == "":
        return DEFAULT_KV_REGIME_GRID
    try:
        ceilings = sorted({int(item) for item in value.split(",") if item.strip() != ""})
    except ValueError:
        return DEFAULT_KV_REGIME_GRID
    return tuple(ceiling for ceiling in ceilings if ceiling > 0)


def kv_regimes_for_batch(
    *,
    num_requests: int,
    query_len: int,
    num_heads: int,
    rows_per_seq: int,
    kv_capacity: int,
    sm_count: int,
    envelope_reason: Callable[..., Optional[str]],
) -> tuple[int, ...]:
    """Ceilings of the KV regimes a graph runner captures one decode graph of this batch shape for.

    Ascending, the page-table capacity last.  Only regimes whose graphs would decide differently earn one: a shape
    the 4x paths can never take (a forced ``2x`` / ``4x`` path, a shape outside the static envelope on the capacity,
    among them every ``query_len`` whose rows are no multiple of 256 and every batch over the split-mode pair bound)
    or always may (floor 0) has the single regime of the capacity; otherwise the 2x regime ends right under the
    shape's KV floor, the 4x-eligible span above it is split wherever the two-stream plan changes (the plan is taken
    on the regime ceiling, and ``plan_two_stream`` drops the split at long KV where its gain fades below the gate:
    B48/q4 keeps r2 = 2 up to ~50K and runs plain 4x beyond, so a 160K page table gets a two-stream regime for the
    short steps), ``kv_regime_grid`` adds its points, and the capacity closes the list.  ``envelope_reason`` as in
    ``choose_decode_strategy``.
    """
    capacity = int(kv_capacity)
    if _env_path() in (PATH_2X, PATH_4X):
        return (capacity,)
    reason = envelope_reason(
        num_heads=num_heads,
        query_len=query_len,
        num_requests=num_requests,
        max_kv_len=capacity,
        sm_count=sm_count,
    )
    if reason is not None:
        return (capacity,)
    floor = four_x_floor_kv(num_requests, rows_per_seq, sm_count)
    if floor <= 1 or capacity < floor:
        return (capacity,)
    ceilings = {floor - 1}
    ceilings.update(ceiling for ceiling in kv_regime_grid() if floor < ceiling < capacity)
    ceilings.update(
        _plan_flip_ceilings(
            num_requests=num_requests,
            rows_per_seq=rows_per_seq,
            floor=floor,
            capacity=capacity,
            sm_count=sm_count,
        )
    )
    return tuple(sorted(ceilings)) + (capacity,)


# Candidate KV lengths (tokens) the two-stream plan is probed at between a shape's floor and its capacity: a power
# of two and its midpoint per octave, enough to place a plan flip within ~30 % of where the planner moves.
_PLAN_PROBE_KVS = tuple(
    sorted({1 << e for e in range(10, 21)} | {3 << (e - 1) for e in range(10, 21)})
)
# A plan flip earns a regime boundary only when the plan it protects is worth it: the modelled two-stream gain of
# the lower regime's plan at one of its probes must reach this much (the planner's own gate is 3 %; B48/q4 is
# ~20 % at 4K, B64/q4's r2 = 1 and B96/q4's r2 = 4 are 4-6 %, which the matrix measured as a wash).
PLAN_FLIP_MIN_GAIN = 0.08


def _plan_flip_ceilings(
    *, num_requests: int, rows_per_seq: int, floor: int, capacity: int, sm_count: int
) -> list[int]:
    """Regime ceilings that keep one two-stream plan per regime: the last probe KV before ``plan_two_stream`` changes
    its ``r2`` on the way from ``floor`` up to ``capacity``, when the plan being left behind is worth
    ``PLAN_FLIP_MIN_GAIN`` (probes start at the floor and stay inside (floor, capacity); the capacity's own plan is
    the one of the last regime)."""
    probes = [floor] + [kv for kv in _PLAN_PROBE_KVS if floor < kv < capacity]

    def plan_at(kv: int) -> TwoStreamPlan:
        return plan_two_stream(
            num_requests=num_requests,
            rows_per_request=rows_per_seq,
            max_kv_len=kv,
            sm_count=sm_count,
        )

    def gain(plan: TwoStreamPlan) -> float:
        if plan.r2 == 0 or plan.model_plain_us <= 0:
            return 0.0
        return 1.0 - plan.model_us / plan.model_plain_us

    flips = []
    previous = plan_at(probes[0])
    regime_gain = gain(previous)
    for kv, next_kv in zip(probes, probes[1:] + [capacity]):
        current = plan_at(next_kv)
        if current.r2 != previous.r2:
            if regime_gain >= PLAN_FLIP_MIN_GAIN and kv > floor:
                flips.append(kv)
            regime_gain = gain(current)
        else:
            regime_gain = max(regime_gain, gain(current))
        previous = current
    return flips


def kv_regime_for_step(max_kv_len: int, ceilings: Sequence[int]) -> int:
    """Regime of a step whose longest live KV is ``max_kv_len``: the smallest ceiling covering it.

    Shorter steps share the first regime; a step beyond the list (never, when ``ceilings`` ends with the
    capacity) gets the last one, the graph of the capacity.
    """
    for ceiling in ceilings:
        if max_kv_len <= ceiling:
            return int(ceiling)
    return int(ceilings[-1])


def choose_decode_strategy(
    *,
    num_requests: int,
    query_len: int,
    num_heads: int,
    rows_per_seq: int,
    kv_capacity: int,
    sm_count: int,
    envelope_reason: Callable[..., Optional[str]],
    live_max_kv: Optional[int] = None,
    helix: bool = False,
    softmax_stats: bool = False,
    v_packed: bool = True,
    four_x_enabled: bool = True,
) -> DecodeStrategy:
    """Pick the decode path for one packed-Q decode call.

    ``envelope_reason`` is ``fp4_mla_cutedsl_4x_host.envelope_reason_static`` (injected so this module stays
    importable without the kernel host); it is always evaluated on the page-table capacity, the geometry the
    kernels are compiled for.
    """
    forced = _env_path()
    kv_source = "capacity"
    kv_decision = int(kv_capacity)
    if live_max_kv is not None and live_max_kv > 0:
        kv_decision = min(int(live_max_kv), int(kv_capacity))
        kv_source = "live"

    def two_x(reason: str) -> DecodeStrategy:
        return DecodeStrategy(PATH_2X, 0, reason, kv_decision, kv_source, None)

    if not four_x_enabled:
        return two_x("TRTLLM_FP4_MLA_CUTEDSL_4X=0")
    if forced == PATH_2X:
        return DecodeStrategy(
            PATH_2X, 0, f"forced by {DECODE_PATH_ENV}", kv_decision, "forced", None
        )
    floor = four_x_floor_kv(num_requests, rows_per_seq, sm_count)
    if forced == "auto" and kv_decision < floor:
        if min_kv_4x() is not None:
            source = f"{MIN_KV_4X_ENV} flat floor"
        else:
            waves = four_x_waves(num_requests, rows_per_seq, sm_count)
            source = f"floor of {waves:.1f} 4x waves; {MIN_KV_4X_ENV} overrides"
        return two_x(
            f"KV {kv_decision} < {floor} ({source}): the 2x packed kernel has the lower fixed cost"
        )
    reason = envelope_reason(
        num_heads=num_heads,
        query_len=query_len,
        num_requests=num_requests,
        max_kv_len=kv_capacity,
        sm_count=sm_count,
        helix=helix,
        softmax_stats=softmax_stats,
        v_packed=v_packed,
    )
    if reason is not None:
        return two_x(reason)
    if forced == PATH_4X:
        return DecodeStrategy(
            PATH_4X, 0, f"forced by {DECODE_PATH_ENV}", kv_decision, "forced", None
        )
    plan = plan_two_stream(
        num_requests=num_requests,
        rows_per_request=rows_per_seq,
        max_kv_len=kv_decision,
        sm_count=sm_count,
    )
    if plan.enabled:
        prefix_reason = envelope_reason(
            num_heads=num_heads,
            query_len=query_len,
            num_requests=plan.num_requests_4x,
            max_kv_len=kv_capacity,
            sm_count=sm_count,
            helix=False,
            softmax_stats=False,
            v_packed=True,
        )
        if prefix_reason is not None:
            plan = plan._replace(
                r2=0,
                reason=f"4x prefix of {plan.num_requests_4x} requests outside the envelope: {prefix_reason}",
            )
    if plan.enabled:
        return DecodeStrategy(
            PATH_TWO_STREAM,
            plan.r2,
            plan.reason,
            kv_decision,
            kv_source if forced == "auto" else "forced",
            plan,
        )
    return DecodeStrategy(
        PATH_4X,
        0,
        plan.reason if forced == "auto" else f"forced {forced}; {plan.reason}",
        kv_decision,
        kv_source if forced == "auto" else "forced",
        plan,
    )


def describe(strategy: DecodeStrategy) -> str:
    extra = f" r2={strategy.r2}" if strategy.path == PATH_TWO_STREAM else ""
    return f"{strategy.path}{extra} (kv {strategy.kv_len_decision} from {strategy.kv_source}): {strategy.reason}"
