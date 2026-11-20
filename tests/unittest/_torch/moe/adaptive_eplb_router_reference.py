# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference of the adaptive balancer's route as the MegaMoE dispatch routers apply it.

A layer that routes in the dispatch router keeps the model's own route launch: the router reads the
expert ids of the launch bucket, maps each pick to a slot through the layer's route tables (the
packed ``counts | starts | slots | cumulative_shares`` rows of :func:`~.policy.build_route_tables`)
and counts the picks of the live rows into the layer's statistics, in the launch that dispatches
them. This module is the definition of the route the router implements, in plain integers: the
reference of its tests and of the GPU check, with no torch or numpy. The statistics follow
``adaptive_eplb_reference.record``, the definition the staged route's record op shares.

The copy of a pick of expert ``e`` in token row ``t`` on EP rank ``r`` is chosen by the point
``p = e * K_EXPERT + r * K_RANK + t * K_TOKEN mod 2^32``: copy ``k`` takes the points in
``[c[k-1], c[k])`` of its expert's cumulative shares (``c[-1] = 0``; the last copy and a bound of
``2^32 - 1`` reach ``2^32``). Every input decides the slot (deterministic); over the 2^32 points
(``K_TOKEN`` is odd, so ``t`` runs over all of them) a copy takes exactly its share of Q32, a share of
0 never takes a pick and a share of 1 takes every pick. Unlike ``adaptive_eplb_route_into``, the
point does not depend on the pick's ordinal among its expert's picks, which the router's warps (one
token each, spread over its CTAs) do not know; the realized split of a few picks deviates from the
shares as a hashed one does.
"""

from __future__ import annotations

from typing import Sequence

K_EXPERT = 0x632BE5AB  # odd; adaptiveEplbKernels.cu kExpertMultiplier
K_RANK = 0x85EBCA6B  # odd; adaptiveEplbKernels.cu kRankMultiplier
K_TOKEN = 0x9E3779B9  # floor(2^32 / golden ratio), odd: consecutive rows spread far apart
_MASK = 0xFFFFFFFF
_TOP = 1 << 32


def unpack(packed: Sequence[int], num_experts: int, num_slots: int):
    """``(counts, starts, slots, cumulative_shares)`` of a packed table row; shares as uint32."""
    e, s = num_experts, num_slots
    packed = [int(v) for v in packed]
    return (
        packed[:e],
        packed[e : 2 * e],
        packed[2 * e : 2 * e + s],
        [v & _MASK for v in packed[2 * e + s : 2 * e + 2 * s]],
    )


def bound(value: int) -> int:
    """A cumulative share as the router compares it: ``2^32 - 1`` (a share that ends the range) is 2^32."""
    return _TOP if value == _MASK else value


def point(expert: int, token: int, ep_rank: int) -> int:
    return (expert * K_EXPERT + ep_rank * K_RANK + token * K_TOKEN) & _MASK


def copy_of(p: int, copies: int, cumulative: Sequence[int]) -> int:
    """The copy (0 .. copies-1) that takes point ``p``: the first whose bound exceeds it."""
    k = 0
    while k < copies - 1 and p >= bound(cumulative[k]):
        k += 1
    return k


def slot(expert: int, token: int, ep_rank: int, tables, num_experts: int, num_slots: int) -> int:
    """The slot of one pick, or -1 for an id outside the experts (the router drops it)."""
    counts, starts, slots, cumulative = tables
    if not 0 <= expert < num_experts:
        return -1
    copies, start = counts[expert], starts[expert]
    if copies <= 0 or start < 0 or start + copies > num_slots:
        return -1
    if copies == 1:
        return slots[start]
    k = copy_of(point(expert, token, ep_rank), copies, cumulative[start : start + copies])
    return slots[start + k]


def slots_of(ids: Sequence[Sequence[int]], ep_rank: int, packed, num_experts: int, num_slots: int):
    """``[rows][top_k]`` slots of ``[rows][top_k]`` expert ids (row index = token row of the bucket)."""
    tables = unpack(packed, num_experts, num_slots)
    return [
        [slot(int(e), t, ep_rank, tables, num_experts, num_slots) for e in row]
        for t, row in enumerate(ids)
    ]


def copy_shares(copies: int, cumulative: Sequence[int]) -> list[int]:
    """Each copy's share of the 2^32 points under :func:`copy_of`, in points."""
    out, below = [], 0
    for k in range(copies):
        top = _TOP if k == copies - 1 else max(bound(cumulative[k]), below)
        out.append(top - below)
        below = top
    return out
