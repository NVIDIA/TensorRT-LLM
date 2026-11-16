# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ring-window expert placement in plain Python: the rule every placement kernel computes.

Under expert parallelism over ``RANKS`` ranks block ``b`` (global experts
``[BLOCK*b, BLOCK*(b+1))``) lives on rank ``b``. With ``w`` ring windows each
rank also holds the blocks of its ``w - 1`` ring predecessors, stacked as
``[own | b-1 | b-2 | ...]``, so an expert of block ``b`` can run on ranks
``b, b + 1, ..., b + w - 1``. ``replicated_experts`` is ``w = 2``
(``ops/moe/ring_buddy_assign``); ``three_window_placement`` is ``w = 3``, computed by the
``PLACE_BALANCE3`` mode of ``ops/comm/pool_gather_route``.

Placement of one step, from the per-block counts ``load[b]`` of distinct
selected experts:

* The smallest feasible maximum ``T``: a capacity ``T`` per rank is feasible
  exactly when every set of blocks fits on the ranks its windows reach (Hall's
  condition). A window is an arc of the ring, so only arcs need checking: a set
  with a gap of at most ``w - 1`` blocks reaches the same ranks as the arc that
  fills the gap, and sets with wider gaps split into independent conditions.
  An arc of ``n`` blocks reaches ``min(n + w - 1, RANKS)`` ranks, so
  ``T = max(ceil(D / RANKS), max over arcs ceil(arc / (n + w - 1)))``.
* The work crossing the ring edge ``r -> r + 1``: the pointwise smallest flow,
  ``c[r] = max(0, max_n (sum of the n blocks ending at r) - n * T)``, which
  every flow with maximum ``T`` dominates. Rank ``r`` then runs
  ``load[r] + c[r - 1] - c[r]`` experts.
* Which blocks cross: at each rank the latest deadline crosses first (the
  rank's own block before the blocks carried into it), the earliest-deadline
  order that keeps every carried expert inside its window whenever ``T`` is
  feasible. For ``w = 3``: ``m[b] = min(c[b], load[b])`` experts of block
  ``b`` leave rank ``b``, of which ``x2[b] = max(0, c[b+1] - load[b+1])`` go on
  to rank ``b + 2``.
* Within a block the experts with the highest local index move farthest:
  the ``x2[b]`` highest present experts run on ``b + 2``, the next
  ``m[b] - x2[b]`` on ``b + 1``, the rest on ``b`` (for ``w = 2`` this is
  ``ring_buddy_assign``'s rule: the ``x[b]`` highest move).

Every rank computes this from the same selection, so the ranks agree on the
placement and each ``(token, slot)`` pair runs on exactly one rank.

An expert of block ``b`` that rank ``r`` runs has the local id
``((r - b) mod RANKS) * BLOCK + i`` in the stacked tensor; an expert another
rank runs is given an id in ``[w * BLOCK, EXPERTS)``, outside the window.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

EXPERTS = 256
RANKS = 8
BLOCK = EXPERTS // RANKS
TOP_K = 8


def window_experts(windows: int) -> int:
    """Local experts of a rank's stacked tensor with ``windows`` ring windows."""
    return windows * BLOCK


def smallest_max(load: Sequence[int], windows: int) -> int:
    """The smallest per-rank maximum any placement with ``windows`` ring windows reaches."""
    ranks = len(load)
    t = -(-sum(load) // ranks)
    for r in range(ranks):
        arc = 0
        for n in range(1, ranks):
            arc += load[(r - n + 1) % ranks]
            reach = min(n + windows - 1, ranks)
            t = max(t, -(-arc // reach))
    return t


def carries(load: Sequence[int], t: int) -> List[int]:
    """``c[r]``: experts crossing the ring edge ``r -> r + 1`` in the smallest flow of maximum ``t``."""
    ranks = len(load)
    out = []
    for r in range(ranks):
        arc, need = 0, 0
        for n in range(1, ranks + 1):
            arc += load[(r - n + 1) % ranks]
            need = max(need, arc - n * t)
        out.append(need)
    return out


def balance(load: Sequence[int], windows: int) -> Tuple[int, List[List[int]]]:
    """``(T, moved)``: ``moved[b][d]`` experts of block ``b`` run on rank ``b + d``, ``d >= 1``.

    ``moved[b][0]`` is 0. Raises ``AssertionError`` if a carried expert would
    leave its window, which the choice of ``T`` rules out.
    """
    ranks = len(load)
    t = smallest_max(load, windows)
    c = carries(load, t)
    # leaving[j][b]: experts of block b that cross the edge b + j -> b + j + 1.
    leaving = [[min(c[b], load[b]) for b in range(ranks)]]
    for j in range(1, windows - 1):
        level = []
        for b in range(ranks):
            edge = (b + j) % ranks
            # what crosses edge b + j beyond the blocks with a later deadline
            later = sum(leaving[i][(edge - i) % ranks] for i in range(j))
            level.append(min(max(0, c[edge] - later), leaving[j - 1][b]))
        leaving.append(level)
    for b in range(ranks):
        edge = (b + windows - 1) % ranks
        crossing = sum(leaving[i][(edge - i) % ranks] for i in range(windows - 1))
        assert crossing == c[edge], (
            f"placement of {list(load)} with {windows} windows carries an expert out of "
            f"its window at rank {edge}"
        )
    moved = []
    for b in range(ranks):
        row = [0] * windows
        for d in range(1, windows):
            further = leaving[d][b] if d < windows - 1 else 0
            row[d] = leaving[d - 1][b] - further
        moved.append(row)
    return t, moved


def loads_after(load: Sequence[int], moved: Sequence[Sequence[int]]) -> List[int]:
    """Experts each rank runs under ``moved``."""
    ranks = len(load)
    windows = len(moved[0])
    return [
        load[r] - sum(moved[r][1:]) + sum(moved[(r - d) % ranks][d] for d in range(1, windows))
        for r in range(ranks)
    ]


def place(selected: Sequence[int], rank: int, windows: int) -> Tuple[List[int], List[int]]:
    """``(local ids, stats)`` of one step's flat selection for ``rank``.

    ``stats`` is the per-rank loads before then after placement, as the
    placement kernels report them.
    """
    present = sorted(set(int(v) for v in selected))
    load = [0] * RANKS
    for expert in present:
        load[expert // BLOCK] += 1
    _, moved = balance(load, windows)
    runs_on = {}
    for block in range(RANKS):
        in_block = sorted((e for e in present if e // BLOCK == block), reverse=True)
        position = 0
        for d in range(windows - 1, 0, -1):
            for expert in in_block[position : position + moved[block][d]]:
                runs_on[expert] = (block + d) % RANKS
            position += moved[block][d]
        for expert in in_block[position:]:
            runs_on[expert] = block
    span = window_experts(windows)
    local = []
    for expert in (int(v) for v in selected):
        block, index = divmod(expert, BLOCK)
        if runs_on[expert] == rank:
            local.append(((rank - block) % RANKS) * BLOCK + index)
        else:
            local.append(span + expert % (EXPERTS - span))
    return local, load + loads_after(load, moved)


# --------------------------------------------------------------------------
# The half window (tep4): the predecessor's upper half only
# --------------------------------------------------------------------------
#: tep4: 4 ranks of 64-expert blocks. With the half window rank ``r`` also holds the
#: upper half (local experts ``HALF_FROM..63``) of its ring predecessor's block, stacked
#: as ``[own block | predecessor's upper half]`` (``HALF_SPAN`` local experts).
HALF_RANKS = 4
HALF_BLOCK = EXPERTS // HALF_RANKS
HALF_FROM = HALF_BLOCK // 2
HALF_SPAN = HALF_BLOCK + (HALF_BLOCK - HALF_FROM)


def half_smallest_max(load: Sequence[int], movable: Sequence[int]) -> int:
    """The smallest per-rank maximum with the half window.

    ``load[b]`` counts block ``b``'s distinct selected experts and ``movable[b]``
    those of them in its upper half, the only ones rank ``b + 1`` holds. Only
    block ``b``'s experts cross the ring edge ``b -> b + 1``, so a maximum ``T`` is
    feasible exactly when the two-window smallest flow ``carries(load, T)``
    (which every flow of maximum ``T`` dominates) moves at most ``movable[b]``
    across each edge: ``T >= ceil((arc - movable[r]) / n)`` for every arc of
    ``n`` blocks ending at ``r``, on top of the two-window bound.
    """
    ranks = len(load)
    t = smallest_max(load, 2)
    for r in range(ranks):
        arc = 0
        for n in range(1, ranks + 1):
            arc += load[(r - n + 1) % ranks]
            t = max(t, -(-(arc - movable[r]) // n))
    return t


def place_half(selected: Sequence[int], rank: int) -> Tuple[List[int], List[int]]:
    """``(local ids, stats)`` of one step's flat selection for ``rank`` with the half window.

    Within a block the ``c[b]`` highest present experts move to rank ``b + 1``
    (they lie in the upper half: ``c[b] <= movable[b]``). Local ids: an own
    expert keeps its index ``i`` (``0..63``); a predecessor's expert this rank
    runs is ``i + HALF_BLOCK - HALF_FROM`` (``64..95``); an expert another rank
    runs gets an id in ``[HALF_SPAN, EXPERTS)``. ``stats`` is the per-rank loads
    before then after placement.
    """
    present = sorted(set(int(v) for v in selected))
    load = [0] * HALF_RANKS
    movable = [0] * HALF_RANKS
    for expert in present:
        block, index = divmod(expert, HALF_BLOCK)
        load[block] += 1
        movable[block] += index >= HALF_FROM
    t = half_smallest_max(load, movable)
    c = carries(load, t)
    assert all(c[b] <= movable[b] for b in range(HALF_RANKS)), (load, movable, t, c)
    runs_on = {}
    for block in range(HALF_RANKS):
        in_block = sorted((e for e in present if e // HALF_BLOCK == block), reverse=True)
        for position, expert in enumerate(in_block):
            runs_on[expert] = (block + 1) % HALF_RANKS if position < c[block] else block
    local = []
    for expert in (int(v) for v in selected):
        block, index = divmod(expert, HALF_BLOCK)
        if runs_on[expert] != rank:
            local.append(HALF_SPAN + expert % (EXPERTS - HALF_SPAN))
        elif block == rank:
            local.append(index)
        else:
            local.append(index + HALF_BLOCK - HALF_FROM)
    after = [load[r] - c[r] + c[(r - 1) % HALF_RANKS] for r in range(HALF_RANKS)]
    return local, load + after
