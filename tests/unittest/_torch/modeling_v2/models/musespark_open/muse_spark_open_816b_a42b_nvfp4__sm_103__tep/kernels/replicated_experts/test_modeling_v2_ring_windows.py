# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ring-window placement rule (``ring_windows.py``): ``three_window_placement``'s three windows.

Host-only. Claims gated here:

* Optimality: the maximum ``balance`` reaches is the smallest any assignment
  of the selected experts to their windows can reach -- checked against
  Hall's condition over every subset of blocks (256 subsets, independent of
  the arc argument the rule uses), on a K=7 verify-step census (3,600 cells),
  exhaustively over small loads, on random loads and on constructed skews.
* Achievability: the moves stay in their windows, take no more experts from a
  block than it has, and leave every rank at most at that maximum.
* The kernel's closed form for three windows (``PLACE_BALANCE3``):
  ``m[b] = min(c[b], load[b])`` leave block ``b`` and ``x2[b] = max(0, c[b+1] -
  load[b+1]) <= m[b]`` of them go two ranks on.
* Two windows reproduce ``ring_buddy_assign`` exactly: the same
  maximum, the same flow, the same local ids.
* ``place``: every (token, slot) runs on exactly one rank, inside that rank's
  96-expert window; the highest local indices move farthest; ties and the ring
  wrap are deterministic.
* On the census the means are 4.311 (two windows) and 4.140 (three) experts on
  the busiest rank, the values of an independent placement simulation.
"""

from __future__ import annotations

import itertools
import os
import random
import statistics

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
    windows as ring_windows,
)

pytestmark = pytest.mark.cpu_only

RANKS = ring_windows.RANKS
BLOCK = ring_windows.BLOCK
CENSUS = os.path.join(os.path.dirname(__file__), "verify_step_block_loads.txt")


def _census():
    with open(CENSUS) as f:
        return [[int(c, 16) for c in line.strip()] for line in f if not line.startswith("#")]


def _hall(load, windows):
    """The smallest feasible per-rank capacity by Hall's condition over every block subset."""
    best = 0
    for mask in range(1, 1 << RANKS):
        blocks = [b for b in range(RANKS) if mask >> b & 1]
        reach = {(b + d) % RANKS for b in blocks for d in range(windows)}
        demand = sum(load[b] for b in blocks)
        best = max(best, -(-demand // len(reach)))
    return best


def _check(load, windows):
    t, moved = ring_windows.balance(load, windows)
    assert t == _hall(load, windows), (load, windows)
    for b in range(RANKS):
        assert moved[b][0] == 0 and all(v >= 0 for v in moved[b]), (load, moved)
        assert sum(moved[b]) <= load[b], (load, moved)
    after = ring_windows.loads_after(load, moved)
    assert sum(after) == sum(load)
    assert max(after) == t, (load, after, t)
    return t, moved


def test_the_census_is_the_served_one():
    cells = _census()
    assert len(cells) == 3600 and all(len(c) == RANKS for c in cells)
    assert abs(statistics.fmean(sum(c) for c in cells) - 29.4325) < 1e-9


@pytest.mark.parametrize("windows", [2, 3])
def test_optimal_and_achievable_on_the_census(windows):
    maxima = [_check(load, windows)[0] for load in _census()]
    # An independent placement simulation on the same census: 4.311 / 4.140.
    assert round(statistics.fmean(maxima), 3) == {2: 4.311, 3: 4.140}[windows]


@pytest.mark.parametrize("windows", [2, 3])
def test_optimal_and_achievable_exhaustively_on_small_loads(windows):
    for load in itertools.product(range(4), repeat=RANKS):
        _check(list(load), windows)


@pytest.mark.parametrize("windows", [2, 3])
def test_optimal_and_achievable_on_random_and_skewed_loads(windows):
    rng = random.Random(33 + windows)
    loads = [[rng.randint(0, BLOCK) for _ in range(RANKS)] for _ in range(3000)]
    loads += [[rng.choice((0, 0, 1, 9, 17)) for _ in range(RANKS)] for _ in range(3000)]
    for heavy in range(RANKS):  # every selected expert in one block, any block (the wrap)
        for k in (1, 7, 8, 9, 23, 32):
            loads.append([k if b == heavy else 0 for b in range(RANKS)])
    loads += [[BLOCK] * RANKS, [0] * RANKS, [1, 0] * 4, [0, 0, 0, 0, 0, 0, 5, 9]]
    for load in loads:
        _check(load, windows)


def test_three_windows_match_the_kernels_closed_form():
    """``PLACE_BALANCE3`` computes m and x2 from the carries, not the general levels."""
    rng = random.Random(7)
    loads = _census() + [[rng.randint(0, 12) for _ in range(RANKS)] for _ in range(5000)]
    for load in loads:
        t, moved = ring_windows.balance(load, 3)
        c = ring_windows.carries(load, t)
        for b in range(RANKS):
            nb = (b + 1) % RANKS
            m = min(c[b], load[b])
            x2 = max(0, c[nb] - load[nb])
            assert x2 <= m
            assert moved[b][1] + moved[b][2] == m and moved[b][2] == x2, (load, b)


def test_two_windows_are_the_ring_buddy_placement():
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
        ring_buddy_assign as rba,
    )

    rng = random.Random(10)
    for load in _census() + [[rng.randint(0, 12) for _ in range(RANKS)] for _ in range(3000)]:
        t2, x = rba.balance_ring(load)
        t, moved = ring_windows.balance(load, 2)
        assert t == t2 and [row[1] for row in moved] == x, load


def _selection(rows, rng, skew=None):
    out = []
    for _ in range(rows):
        pool = range(ring_windows.EXPERTS) if skew is None else skew
        out.extend(rng.sample(list(pool), ring_windows.TOP_K))
    return out


def test_two_windows_place_like_ring_buddy_assign():
    torch = pytest.importorskip("torch")
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
        ring_buddy_assign,
    )

    rng = random.Random(11)
    for trial in range(300):
        rows = 1 + trial % 8
        flat = _selection(rows, rng, None if trial % 3 else range(64, 120))
        selected = torch.tensor(flat, dtype=torch.int32).reshape(rows, ring_windows.TOP_K)
        weights = torch.rand(rows, ring_windows.TOP_K)
        for rank in range(RANKS):
            ids, _, stats = ring_buddy_assign.reference(selected, weights, rank)
            local, place_stats = ring_windows.place(flat, rank, 2)
            assert ids.reshape(-1).tolist() == local and stats.tolist() == place_stats


@pytest.mark.parametrize("windows", [2, 3])
def test_place_partitions_the_slots_inside_the_windows(windows):
    rng = random.Random(12 + windows)
    span = ring_windows.window_experts(windows)
    for trial in range(400):
        rows = 1 + trial % 8
        skew = None if trial % 4 else range(32 * (trial % 8), 32 * (trial % 8) + 40)
        flat = [e % ring_windows.EXPERTS for e in _selection(rows, rng, skew)]
        per_rank = [ring_windows.place(flat, rank, windows) for rank in range(RANKS)]
        stats = per_rank[0][1]
        assert all(s == stats for _, s in per_rank), "every rank reports the same loads"
        distinct = [set() for _ in range(RANKS)]
        for slot, expert in enumerate(flat):
            runners = [r for r in range(RANKS) if per_rank[r][0][slot] < span]
            assert len(runners) == 1, (flat, slot, runners)
            rank = runners[0]
            local = per_rank[rank][0][slot]
            hops = (rank - expert // BLOCK) % RANKS
            assert hops < windows and local == hops * BLOCK + expert % BLOCK
            for other in range(RANKS):
                if other != rank:
                    assert span <= per_rank[other][0][slot] < ring_windows.EXPERTS
            distinct[rank].add(expert)
        assert [len(d) for d in distinct] == stats[RANKS:], (flat, stats)
        assert max(stats[RANKS:]) == ring_windows.smallest_max(stats[:RANKS], windows)


def test_place_moves_the_highest_local_indices_farthest():
    rng = random.Random(13)
    for _ in range(500):
        flat = _selection(8, rng, range(0, 96))
        hops_of = {}
        for rank in range(RANKS):
            local, _ = ring_windows.place(flat, rank, 3)
            for expert, lid in zip(flat, local):
                if lid < 96:
                    hops_of[expert] = lid // BLOCK
        for block in range(RANKS):
            mine = sorted(e for e in hops_of if e // BLOCK == block)
            hops = [hops_of[e] for e in mine]
            assert hops == sorted(hops), (block, mine, hops)


def test_ties_and_wrap_are_deterministic():
    # Block 7 heavy: its carry wraps to ranks 0 and 1.
    flat = list(range(7 * BLOCK, 7 * BLOCK + 24)) + list(range(0, 8)) + list(range(40, 48))
    runs = {}
    for rank in range(RANKS):
        local, stats = ring_windows.place(flat, rank, 3)
        assert local == ring_windows.place(flat, rank, 3)[0]
        for expert, lid in zip(flat, local):
            if lid < 96:
                runs[expert] = rank
    load = stats[:RANKS]
    assert load == [8, 8, 0, 0, 0, 0, 0, 24]
    # T = 8 (block 7 alone reaches three ranks). The carries are 16 out of
    # ranks 7, 0, 1 and 8 out of rank 2; each rank forwards its own block before
    # the carried ones, so block 7 fills ranks 7, 0, 1 and blocks 0 and 1 run
    # two ranks on.
    assert ring_windows.smallest_max(load, 3) == 8
    assert stats[RANKS:] == [8, 8, 8, 8, 0, 0, 0, 8]
    assert {runs[e] for e in range(7 * BLOCK, 7 * BLOCK + 24)} == {7, 0, 1}
    assert {runs[e] for e in range(0, 8)} == {2} and {runs[e] for e in range(40, 48)} == {3}
