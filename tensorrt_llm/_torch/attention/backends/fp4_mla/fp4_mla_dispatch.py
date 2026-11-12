# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA decode path selection: ``2x`` packed | ``4x`` four-CTA-cluster | ``twostream`` (4x arm + 2x arm).

The decision is host arithmetic on static shapes plus the step's KV lengths, so it is the same in the eager warm-up,
at CUDA-graph capture and (through the graph that was captured) at replay.  Measured basis (hecate0358/0369, Rubin
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
(hecate0341, the first matrix below): with the 4x kernel of that date B128 at 8K was a tie within the node-to-node
spread either way, which is where the many-wave floor was put.  The 4x and two-stream columns of the table above
are that kernel's; the kernel of 2026-11-10 is faster and the many-wave tie sits at 4K (second matrix below).

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

Re-validated on 2026-11-10 (hecate0087) for the 4x kernel of that date: shared memory laid out by sub-bank, the
four-bank V ring and the K128 QK tail make the 4x and two-stream cells 2-14 % faster, the 2x kernel is unchanged.
Same shapes, graph-replay medians; ratio = 2x / the 4x paths with the floor forced to 0
(``TRTLLM_FP4_MLA_4X_MIN_KV=0``); ``*`` marks a cell under its floor, where the dispatcher keeps 2x:

===========================  ======  ======  ======  ======  ======  ======  ==========
batch (4x waves)             1K      2K      4K      8K      12K     16K     floor
===========================  ======  ======  ======  ======  ======  ======  ==========
B16 (0.7, one wave)          1.33x   2.66x   2.35x   2.09x                   0
B32 (1.4)                    0.97x*  1.07x   1.15x   1.31x   1.40x           2K
B48 (2.1, two-stream r2=2)   0.92x*  0.99x*  1.04x   1.18x   1.27x   1.30x   4K
B64 (2.8)                    0.91x*  0.98x*  1.08x   1.23x   1.30x   1.33x   4K
B96 (4.2)                    0.90x*  0.94x*  1.02x*  1.17x   1.25x           8K
B128 (5.6)                   0.90x*  0.95x*  0.99x*  1.12x   1.20x   1.24x   8K
B256 (11.1)                          0.98x*  1.02x*  1.16x           1.25x   8K
===========================  ======  ======  ======  ======  ======  ======  ==========

Above the floors the 4x path was two-stream at B48 (r2 = 2), B64 at 4K (r2 = 1), B96 at 8K and 12K (r2 = 4) and
B256 at 8K (r2 = 3), plain 4x in the other cells.  No floor moves (a floor moves on a gain clearly outside the
run-to-run noise of about 2 %): the best cell under a floor is 1.02x (B96 and B256 at 4K), every cell at a floor is
ahead (1.04x to 1.17x).  What changed is the margin of the many-wave floor: B128 at 8K was the tie (1.02x) and is
1.12x now, and these shapes tie at 4K instead (0.99x-1.02x).  The two-stream planner's item cost model was not
re-fitted to this kernel: see the note at ``T4_FIXED_US`` in ``fp4_mla_twostream``.

Extra-wave windows.  A 4x wave holds fewer requests than a 2x wave (46 pair tasks of 256 rows against
``sm_count // 2`` m-tiles of 128, i.e. 53 pair tasks on 212 SMs), so for the batches in between the 4x paths need a
wave more than the 2x kernel, and the plain 4x path pays for that wave at short KV.  Under ``extra_wave_floor_kv``
such a call is therefore taken only by a
two-stream split that drops the extra wave: the one-wave split of ``fp4_mla_twostream.one_wave_split`` where the
host launches its 4x wave FULL (the call is then one wave of the machine, which needs no floor), else the planner's
split from the graded floor, else 2x.

Above the floor the static 4x envelope (``fp4_mla_cutedsl_4x_host.envelope_reason_static``:
small-batch rules, Helix / softmax-stats / packed-V requirements, split-mode bounds) decides whether the 4x kernel
may run, and the two-stream planner (``fp4_mla_twostream``) decides whether handing the last requests to the 2x
kernel on a concurrent stream drops a whole 4x wave.  A Helix call inside the envelope always runs plain ``4x``:
the two-stream arms carry neither the per-token KV bounds nor the softmax statistics.

Whether the 4x kernel can run here at all is not this module's knowledge either: the caller passes the reason it
cannot as ``four_x_unavailable`` (``fp4_mla_cutedsl_4x_host.device_unsupported_reason``: what the kernel needs from
the device, today the shared memory a block may use), and every call then stays on ``2x`` with that reason.  A graph
runner on such a device keys its decode graphs by no KV regime, as it does for Helix.

KV lengths used for the decision: the longest (``live_max_kv``) and the mean (``live_mean_kv``) KV of the step's
requests where the caller knows them (an eager step its own; a graph runner that keys its graphs by KV regime the
regime's, ``kv_regime_bounds``), otherwise the page-table capacity for both.  The floors, the split plans and the
launch of an under-filled 4x wave are taken on the mean: a floor weighs a fixed cost per task against a gain per KV
tile, and the tiles a kernel runs are the mean's, whichever request holds them.  The longest decides one thing
more.  A launch that runs every request uncut (``_runs_uncut``) lasts as long as its longest request, which the plan
of the 2x kernel cuts into pieces for the clusters the shorter requests leave idle: such a launch takes a step only
while its longest request is at most ``SPREAD_MAX_UNCUT_KV`` above the mean, and a more uneven step stays on 2x.
The kernels themselves are always compiled and planned for the capacity.

Environment:
``TRTLLM_FP4_MLA_CUTEDSL_4X_H128``  master switch of the 128-head decode, the head count of the tables above: on
                                    unless set to a non-empty value other than ``1``, ``true``, ``yes`` or ``on``.
``TRTLLM_FP4_MLA_CUTEDSL_4X``       master switch of every other head count: off unless set to ``1``. Neither
                                    switch is read for the other's head counts. Both are honoured by the callers
                                    (``decode.py`` and the decode graphs): where the switch of a call is off this
                                    module is not consulted and the call stays on ``2x``.
``TRTLLM_FP4_MLA_DECODE_PATH``      ``auto`` (default) | ``2x`` | ``4x`` | ``twostream``: force a path (a forced 4x or
                                    two-stream path still needs the static envelope and a device that can host the
                                    4x kernel, ``four_x_unavailable``; ``twostream`` falls back to ``4x`` when the
                                    planner finds no useful split).
``TRTLLM_FP4_MLA_4X_MIN_KV``        one flat KV length (tokens) below which every call stays on ``2x``, replacing
                                    the per-batch floor of ``four_x_floor_kv`` (0 / 2048 / 4096 / 8192 by 4x waves),
                                    the extra-wave rule of ``extra_wave_floor_kv`` and the page-table floor
                                    ``NARROW_KV_CAPACITY``.
``TRTLLM_FP4_MLA_KV_REGIME_GRID``   extra KV regime ceilings (tokens, comma-separated) a graph runner keys its
                                    decode graphs by between a batch's floor and the capacity (default none: one 2x
                                    regime under the floor and one 4x-eligible regime up to the capacity): see
                                    ``kv_regimes_for_batch``.
"""

from __future__ import annotations

import math
import os
from typing import Callable, NamedTuple, Optional

from .fp4_mla_twostream import (
    KV_TILE,
    LANES_4X,
    ROWS_PER_2X_ITEM,
    ROWS_PER_4X_ITEM,
    TwoStreamPlan,
    one_wave_split,
    plan_two_stream,
    split_launch_min_kv,
)

DECODE_PATH_ENV = "TRTLLM_FP4_MLA_DECODE_PATH"
MIN_KV_4X_ENV = "TRTLLM_FP4_MLA_4X_MIN_KV"
# KV floor (tokens) of the 4x paths by the batch's 4x wave count (``four_x_waves``): (waves at most, floor) in
# ascending order, ``FLOOR_KV_4X_MANY_WAVES`` beyond the last bound.  From the 2026-11-03 matrix of the module
# docstring; its 2026-11-10 matrix (the 4x kernel with the sub-bank shared-memory layout) moves none of them.
FLOOR_KV_4X_BY_WAVES = ((1.0, 0), (1.5, 2048), (3.0, 4096))
FLOOR_KV_4X_MANY_WAVES = 8192
# KV floor (tokens) of the PLAIN 4x path for a call that needs a wave more on the 4x kernel than on the 2x kernel
# (``extra_wave_floor_kv``), by the 2x kernel's wave count: (2x waves at most, floor) in ascending order,
# ``FLOOR_KV_4X_MANY_WAVES`` beyond.  Two waves against one: plain 4x ties both 2x and the one-wave split at 12K.
# Three against two: it ties 2x at 7K, and before 6K when a request is a single pair task, which sets the floor.
FLOOR_KV_4X_EXTRA_WAVE = ((1, 12288), (2, 6144))
# Page-table capacity (tokens) up to which every call stays on the 2x kernel.  On a page table of two KV tiles an
# eager 2x launch is one task per request, without plan or reducer (``packed_plan._select_packed_split_count``).
# Measured against 2x at 128 to 512 tokens (graph replays, 128 heads, q2 / q4): on such a page table the one-wave
# split is 1-30 % slower and a FULL-launched 4x wave from 3 % faster to 26 % slower; at the same KV on page tables of
# 640 tokens and more both are ahead (by 11-30 % and 14-66 %).  So a short step stays on 2x in an eager call, which
# runs on its page table narrowed to the step, and may take a 4x path in a graph, which runs on the engine's.
NARROW_KV_CAPACITY = 2 * KV_TILE
# Tokens the longest request of a step may be above the step's mean KV for a launch that runs every request uncut
# (``_runs_uncut``) to take the step.  Measured against 2x (graph replays, 128 heads, q2 to q8; one request to half
# the batch longer than the others, ramps, heavy tails): up to the bound the uncut launches are ahead or level, but
# for one 4x wave of 14 to 17 q4 requests whose others hold at most two KV tiles, which is level at 4000 tokens
# above the mean and up to 6 % behind at the bound.  A lower bound loses more than it saves: ramps to 8K run
# 0.42-0.50 of 2x at 3900-4090 tokens above their mean.  From 4096 tokens the first layouts are up to 6 % behind,
# from 5120 up to 17 %, and one request of 30K among 256-token ones costs 1.5x to 3.2x.
SPREAD_MAX_UNCUT_KV = 4096

PATH_2X = "2x"
PATH_4X = "4x"
PATH_TWO_STREAM = "twostream"
PATHS = (PATH_2X, PATH_4X, PATH_TWO_STREAM)

# KV regimes a graph runner can key its decode graphs by (tokens of mean KV): ``kv_regimes_for_batch`` lists the
# regimes worth a graph for one batch shape, ``kv_regime_for_step`` maps a step's mean and longest live KV onto
# them; the page-table capacity itself is always the last regime, the grid adds optional points between a batch's
# floor and the capacity.
KV_REGIME_GRID_ENV = "TRTLLM_FP4_MLA_KV_REGIME_GRID"
DEFAULT_KV_REGIME_GRID: tuple[int, ...] = ()
# The regime of mean KV 0, which no step has: the 2x regime of the steps too uneven for an uncut launch, on a batch
# shape with no 2x regime of its own (``kv_regime_bounds``).
KV_REGIME_UNEVEN = 0


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


def two_x_waves(num_requests: int, rows_per_seq: int, sm_count: int) -> int:
    """2x waves of one call: its 128-row m-tiles over the 2-CTA clusters the device hosts at once (the 2x kernel's
    time steps at this cluster count, not at whole requests per wave)."""
    m_tiles = max(0, int(num_requests)) * max(0, int(rows_per_seq)) // ROWS_PER_2X_ITEM
    return math.ceil(m_tiles / max(1, int(sm_count) // 2))


def extra_wave_floor_kv(num_requests: int, rows_per_seq: int, sm_count: int) -> int:
    """KV length (tokens) from which the plain 4x path may take a call that needs a wave more on the 4x kernel than
    on the 2x kernel; 0 for every other call, and under ``TRTLLM_FP4_MLA_4X_MIN_KV`` (one flat floor for all).

    Shorter steps of such a call go to a two-stream split that drops the extra wave, or stay on 2x.  Graded by the
    2x wave count (``FLOOR_KV_4X_EXTRA_WAVE``, measured basis at the constant): the extra wave doubles a call of one
    2x wave and weighs less on every further one.  Never under ``four_x_floor_kv``, and the same floor
    beyond three 4x waves.
    """
    if min_kv_4x() is not None:
        return 0
    waves_2x = two_x_waves(num_requests, rows_per_seq, sm_count)
    if math.ceil(four_x_waves(num_requests, rows_per_seq, sm_count)) <= waves_2x:
        return 0
    for bound, floor in FLOOR_KV_4X_EXTRA_WAVE:
        if waves_2x <= bound:
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


class KvRegimes(tuple):
    """KV regimes of one batch shape (``kv_regimes_for_batch``): ceilings of the step's mean KV, ascending, the
    page-table capacity last.  ``uncut``: the ceilings whose decision launches every request uncut
    (``_runs_uncut``); a step too uneven for them replays the first regime (``kv_regime_for_step``)."""

    uncut: frozenset = frozenset()

    def __new__(cls, ceilings, uncut=()):
        regimes = super().__new__(cls, ceilings)
        regimes.uncut = frozenset(uncut)
        return regimes


def kv_regime_bounds(regime: int) -> tuple[Optional[int], int]:
    """(longest, mean) KV the graph of a regime decides on (``choose_decode_strategy``'s ``live_max_kv``,
    ``live_mean_kv``): every request at the ceiling, and for ``KV_REGIME_UNEVEN`` one request as long as the page
    table among empty ones, which no uncut launch takes."""
    if regime == KV_REGIME_UNEVEN:
        return None, 0
    return int(regime), int(regime)


def _runs_uncut(path: str, pair_tasks: int) -> bool:
    """Whether ``path`` runs every request of a call of ``pair_tasks`` whole: a two-stream split (its 2x arm plans
    one task per request), a single 4x wave and plain 4x on waves whole requests fill exactly (the device plan of the
    4x kernel cuts the requests of a last wave that is not full, and nothing else)."""
    if path == PATH_TWO_STREAM:
        return True
    return path == PATH_4X and (pair_tasks <= LANES_4X or pair_tasks % LANES_4X == 0)


def kv_regimes_for_batch(
    *,
    num_requests: int,
    query_len: int,
    num_heads: int,
    rows_per_seq: int,
    kv_capacity: int,
    sm_count: int,
    envelope_reason: Callable[..., Optional[str]],
) -> KvRegimes:
    """Ceilings of the KV regimes a graph runner captures one decode graph of this batch shape for.

    Ascending, the page-table capacity last.  Only regimes whose graphs would decide differently earn one: a shape
    the 4x paths can never take (a forced ``2x`` / ``4x`` path, a shape outside the static envelope on the capacity,
    among them every ``query_len`` whose rows are no multiple of 256 and every batch over the split-mode pair bound)
    or always may (floor 0) has the regime of the capacity; otherwise the 2x regime ends right under the
    shape's KV floor, the 4x-eligible span above it is split wherever the two-stream plan changes (the plan is taken
    on the regime ceiling, and ``plan_two_stream`` drops the split at long KV where its gain fades below the gate:
    B48/q4 keeps r2 = 2 up to ~50K and runs plain 4x beyond, so a 160K page table gets a two-stream regime for the
    short steps), ``kv_regime_grid`` adds its points, and the capacity closes the list.  In an extra-wave window
    (``extra_wave_floor_kv``) that floor ends a regime as well, and each of the two floors does only where the
    decision changes across it: the one-wave split runs under the graded floor, a planner split through the window
    floor.  The host's bound for the split-mode launch of an under-filled 4x wave (``split_launch_min_kv``) ends the
    regime that spans it where that regime launches such a wave.  Where a regime launches every request uncut, a
    step too uneven for it replays a 2x regime (``KvRegimes``): the shape's own under its floor, else
    ``KV_REGIME_UNEVEN`` in front of the list.
    ``envelope_reason`` as in ``choose_decode_strategy``.  A runner whose steps no regime can move keys its graphs
    by none and does not ask: Helix (only one-wave shapes leave ``2x``; their graded floor is 0 and they are in no
    extra-wave window; a flat ``TRTLLM_FP4_MLA_4X_MIN_KV`` is then compared with the capacity), a device where the
    4x kernel is unavailable.
    """
    capacity = int(kv_capacity)
    if _env_path() in (PATH_2X, PATH_4X):
        return KvRegimes((capacity,))
    reason = envelope_reason(
        num_heads=num_heads,
        query_len=query_len,
        num_requests=num_requests,
        max_kv_len=capacity,
        sm_count=sm_count,
    )
    if reason is not None:
        return KvRegimes((capacity,))

    def decision(regime: int) -> DecodeStrategy:
        live_max_kv, live_mean_kv = kv_regime_bounds(regime)
        return choose_decode_strategy(
            num_requests=num_requests,
            query_len=query_len,
            num_heads=num_heads,
            rows_per_seq=rows_per_seq,
            kv_capacity=capacity,
            sm_count=sm_count,
            envelope_reason=envelope_reason,
            four_x_unavailable=None,
            live_max_kv=live_max_kv,
            live_mean_kv=live_mean_kv,
        )

    floor = four_x_floor_kv(num_requests, rows_per_seq, sm_count)
    # The extra-wave rule is the automatic choice's; in a window the one-wave split runs under the graded floor.
    window_floor = (
        extra_wave_floor_kv(num_requests, rows_per_seq, sm_count) if _env_path() == "auto" else 0
    )
    ceilings = set()
    if floor > 1 and (capacity >= floor or window_floor):
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
    if window_floor:
        # Extra-wave window: a boundary of its rule stays only where the decision changes across it.
        floors = {floor - 1, window_floor - 1}
        ordered = sorted(ceiling for ceiling in ceilings | floors if ceiling < capacity)
        ceilings = {
            ceiling
            for ceiling, following in zip(ordered, ordered[1:] + [capacity])
            if ceiling not in floors or decision(ceiling)[:2] != decision(following)[:2]
        }
    regimes = sorted(ceilings) + [capacity]
    pair_tasks = num_requests * rows_per_seq // ROWS_PER_4X_ITEM
    bounds = sorted({split_launch_min_kv(1) - 1, split_launch_min_kv(LANES_4X - 1) - 1})
    for split_bound in bounds:
        if not 0 < split_bound < capacity or split_bound in regimes:
            continue
        spanning = decision(min(regime for regime in regimes if regime > split_bound))
        arm = pair_tasks - spanning.r2 * (rows_per_seq // ROWS_PER_4X_ITEM)
        launched_by_kv = spanning.path != PATH_2X and arm < LANES_4X
        if launched_by_kv and split_launch_min_kv(arm) - 1 == split_bound:
            regimes = sorted(regimes + [split_bound])
    if _env_path() != "auto" or capacity <= SPREAD_MAX_UNCUT_KV:
        # the uneven rule is the automatic choice's, on a page table with room for it
        return KvRegimes(regimes)
    uncut = [regime for regime in regimes if _runs_uncut(decision(regime).path, pair_tasks)]
    if uncut and decision(regimes[0]).path != PATH_2X:
        regimes.insert(0, KV_REGIME_UNEVEN)
    return KvRegimes(regimes, uncut)


# Candidate KV lengths (tokens) the two-stream plan is probed at between a shape's floor and its capacity: a power
# of two and its midpoint per octave, enough to place a plan flip within ~30 % of where the planner moves.
_PLAN_PROBE_KVS = tuple(
    sorted({1 << e for e in range(10, 21)} | {3 << (e - 1) for e in range(10, 21)})
)
# A plan flip earns a regime boundary only when the plan it protects is worth it: the modelled two-stream gain of
# the lower regime's plan at one of its probes must reach this much (the planner's own gate is 3 %; B48/q4 is
# ~20 % at 4K, B64/q4's r2 = 1 and B96/q4's r2 = 4 are 4-6 %, which the matrix measured as a wash).
# The gains are the model's (``fp4_mla_twostream``, constants of 2026-11-02) and the wash is the 2026-11-03 matrix:
# neither was re-taken for the 4x kernel of 2026-11-10, whose matrices have no plain 4x arm on the split cells.  A
# lower T4 moves the modelled gain of the small splits up towards this gate, so re-read it with any re-fit of T4.
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


def kv_regime_for_step(
    max_kv_len: int, ceilings: KvRegimes, mean_kv_len: Optional[float] = None
) -> int:
    """Regime of a step whose longest live KV is ``max_kv_len`` and whose mean is ``mean_kv_len`` (the longest when
    not given): the smallest ceiling covering the mean, else the last.  A step too uneven for the uncut launch of
    that regime, by the test of ``choose_decode_strategy``, replays the first regime, the 2x regime of its shape."""
    mean_kv_len = max_kv_len if mean_kv_len is None else mean_kv_len
    regime = next((int(c) for c in ceilings if mean_kv_len <= c), int(ceilings[-1]))
    if max_kv_len - math.ceil(mean_kv_len) > SPREAD_MAX_UNCUT_KV and regime in ceilings.uncut:
        return int(ceilings[0])
    return regime


def choose_decode_strategy(
    *,
    num_requests: int,
    query_len: int,
    num_heads: int,
    rows_per_seq: int,
    kv_capacity: int,
    sm_count: int,
    envelope_reason: Callable[..., Optional[str]],
    four_x_unavailable: Optional[str],
    live_max_kv: Optional[int] = None,
    live_mean_kv: Optional[float] = None,
    helix: bool = False,
    softmax_stats: bool = False,
    v_packed: bool = True,
) -> DecodeStrategy:
    """Pick the decode path for one packed-Q decode call.

    ``envelope_reason`` is ``fp4_mla_cutedsl_4x_host.envelope_reason_static`` (injected so this module stays
    importable without the kernel host); it is always evaluated on the page-table capacity, the geometry the
    kernels are compiled for.  ``four_x_unavailable`` is why the 4x kernel cannot run at all where the call executes
    (None when it can; required, so that no caller assumes it can by omission): it wins over a forced path, and
    becomes the strategy's reason.  ``live_max_kv`` / ``live_mean_kv``: the step's longest and mean KV (module
    docstring); the strategy's ``kv_len_decision`` is the mean, the KV length its 4x wave is launched for.
    """
    forced = _env_path()
    kv_source = "capacity"
    kv_longest = int(kv_capacity)
    if live_max_kv is not None and live_max_kv > 0:
        kv_longest = min(int(live_max_kv), int(kv_capacity))
        kv_source = "live"
    kv_decision = kv_longest
    if live_mean_kv is not None:
        kv_decision = min(max(1, math.ceil(live_mean_kv)), kv_longest)
        kv_source = "live"
    # Too uneven for a launch that runs every request uncut; the rule is the automatic choice's.
    uneven = forced == "auto" and kv_longest - kv_decision > SPREAD_MAX_UNCUT_KV

    def two_x(reason: str) -> DecodeStrategy:
        return DecodeStrategy(PATH_2X, 0, reason, kv_decision, kv_source, None)

    if four_x_unavailable is not None:
        return two_x(four_x_unavailable)
    if forced == PATH_2X:
        return DecodeStrategy(
            PATH_2X, 0, f"forced by {DECODE_PATH_ENV}", kv_decision, "forced", None
        )
    if forced == "auto" and min_kv_4x() is None and kv_capacity <= NARROW_KV_CAPACITY:
        return two_x(
            f"page table of {int(kv_capacity)} tokens: the 2x packed kernel runs one task per request there; "
            f"{MIN_KV_4X_ENV} overrides"
        )
    floor = four_x_floor_kv(num_requests, rows_per_seq, sm_count)
    # Extra-wave window: under its floor only a two-stream split that drops the extra wave takes the call, and the
    # one-wave split does so under the graded floor too.
    window_floor = extra_wave_floor_kv(num_requests, rows_per_seq, sm_count)
    under_window_floor = forced == "auto" and kv_decision < window_floor
    plan = None
    if under_window_floor:
        plan = one_wave_split(
            num_requests=num_requests,
            rows_per_request=rows_per_seq,
            max_kv_len=kv_decision,
            sm_count=sm_count,
        )
    if forced == "auto" and kv_decision < floor and plan is None:
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
    if helix or softmax_stats:
        # The envelope admitted a Helix call. No split is planned for it, forced or not: the two-stream arms carry
        # neither the KV bounds nor the softmax statistics.
        return DecodeStrategy(
            PATH_4X,
            0,
            "Helix: plain 4x, the two-stream arms carry no KV bounds or softmax statistics",
            kv_decision,
            kv_source if forced == "auto" else "forced",
            None,
        )
    if plan is None:
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
    if under_window_floor and not plan.enabled:
        return two_x(
            f"KV {kv_decision} < {window_floor}: the 4x paths need a wave more than the 2x packed kernel and no "
            f"two-stream split drops it ({plan.reason}); {MIN_KV_4X_ENV} overrides"
        )
    path = PATH_TWO_STREAM if plan.enabled else PATH_4X
    if uneven and _runs_uncut(path, num_requests * rows_per_seq // ROWS_PER_4X_ITEM):
        return two_x(
            f"longest KV {kv_longest} is more than {SPREAD_MAX_UNCUT_KV} above the mean KV {kv_decision}: "
            f"{path} would run it uncut, the 2x packed plan cuts it"
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
