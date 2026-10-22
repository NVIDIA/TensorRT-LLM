# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python reference of the HALO-R replica planner for tests.

It follows the CUDA planner step for step (moeRebalanceHaloRPlanner.h): the
same candidate keys, tie-breaks and helper-slot assignment, so its plan must
equal the kernel's word for word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

ORDER_BITS = 8
ORDER_OPEN = 1 << (ORDER_BITS - 1)
ORDER_MAX = (1 << ORDER_BITS) - 1
LANE_BITS = 5
PLAN_FIELD_STRIDE = 384
PLAN_FIELDS = ("expert", "owner", "destination_mask", "helper_low_mask", "rows", "helper_high_mask")


@dataclass
class Broadcast:
    """One replicated expert: its owner and each receiver's routes and helper slot."""

    expert: int
    owner: int
    rows: Dict[int, int] = field(default_factory=dict)
    helper: Dict[int, int] = field(default_factory=dict)

    @property
    def destination_mask(self) -> int:
        return sum(1 << rank for rank in self.rows)


@dataclass
class Plan:
    broadcasts: List[Broadcast]
    helper_table: List[List[int]]
    load_after: List[int]
    target: int

    def quota(self, histogram: Sequence[int], ep: int) -> List[List[int]]:
        """Routes of each broadcast's expert kept by each rank, owner included."""
        quota = []
        for b in self.broadcasts:
            row = [0] * ep
            for rank, rows in b.rows.items():
                row[rank] = rows
            row[b.owner] = histogram[b.expert] - sum(b.rows.values())
            quota.append(row)
        return quota

    def plan_words(self, epoch: int = 0) -> List[int]:
        """The kernel's plan words: [epoch, broadcasts] then six fields of 384 entries."""
        words = [epoch, len(self.broadcasts)] + [0] * (len(PLAN_FIELDS) * PLAN_FIELD_STRIDE)
        for index, b in enumerate(self.broadcasts):
            low = sum((slot & 1) << rank for rank, slot in b.helper.items())
            high = sum(((slot >> 1) & 1) << rank for rank, slot in b.helper.items())
            values = (b.expert, b.owner, b.destination_mask, low, sum(b.rows.values()), high)
            for f, value in enumerate(values):
                words[2 + f * PLAN_FIELD_STRIDE + index] = _as_int32(value)
        return words


def _as_int32(value: int) -> int:
    return value - (1 << 32) if value >= (1 << 31) else value


def plan_halo_r(
    histogram: Sequence[int], ep: int, helper_slots: int, send_count: int, send_weight: int = 0
) -> Plan:
    """Plan one layer from the global routed-token count of every expert."""
    histogram = [int(x) for x in histogram]
    home = len(histogram) // ep
    hottest = [
        sorted(range(home), key=lambda local: (-histogram[rank * home + local], local))[:send_count]
        for rank in range(ep)
    ]
    load = [sum(histogram[rank * home : (rank + 1) * home]) for rank in range(ep)]
    target = -(-sum(load) // ep)
    received = [0] * ep
    sent = [0] * ep
    exports: List[List[int]] = []
    moves = []
    while True:
        room = [target - load[rank] for rank in range(ep)]
        eligible = [received[rank] < helper_slots and room[rank] > 0 for rank in range(ep)]
        keys = [
            (room[rank] << LANE_BITS) | (31 - rank) if eligible[rank] else 0 for rank in range(ep)
        ]
        best_key = max(keys)
        best_rank = 31 - (best_key & 31) if best_key else -1
        second_key = max(0 if rank == best_rank else keys[rank] for rank in range(ep))

        def room_excluding(rank: int) -> int:
            return (second_key if rank == best_rank else best_key) >> LANE_BITS

        winner_key, winner = 0, None
        for index, (expert, remaining) in enumerate(exports):
            owner = expert // home
            excess = load[owner] - target
            if remaining <= 0 or excess <= 0:
                continue
            rows = min(room_excluding(owner), remaining, excess)
            if rows <= 0:
                continue
            key = ((rows * (1 + send_weight)) << ORDER_BITS) | (ORDER_MAX - index)
            if key > winner_key:
                winner_key, winner = key, (expert, owner, rows, index, False)
        for rank in range(ep):
            excess = load[rank] - target
            if excess <= 0 or sent[rank] >= send_count:
                continue
            expert = rank * home + hottest[rank][sent[rank]]
            if histogram[expert] <= 0:
                continue
            rows = min(room_excluding(rank), histogram[expert], excess)
            if rows <= 0:
                continue
            key = (rows << ORDER_BITS) | (ORDER_MAX - (ORDER_OPEN + rank))
            if key > winner_key:
                winner_key, winner = key, (expert, rank, rows, len(exports), True)
        if winner is None:
            break
        expert, owner, rows, index, opens = winner
        receivers = [
            rank for rank in range(ep) if eligible[rank] and rank != owner and room[rank] >= rows
        ]
        if not receivers:
            raise AssertionError("winning candidate without a receiver")
        destination = receivers[0]
        if opens:
            sent[owner] += 1
            exports.append([expert, histogram[expert]])
        load[owner] -= rows
        load[destination] += rows
        received[destination] += 1
        exports[index][1] -= rows
        moves.append((expert, destination, rows))

    helper_table = []
    for rank in range(ep):
        experts = sorted({expert for expert, dest, _ in moves if dest == rank})
        if len(experts) > helper_slots:
            raise AssertionError(f"rank {rank} receives more experts than helper slots")
        helper_table.append(experts + [-1] * (helper_slots - len(experts)))
    broadcasts = []
    for expert in sorted({expert for expert, _, _ in moves}):
        b = Broadcast(expert=expert, owner=expert // home)
        for moved, dest, rows in moves:
            if moved == expert:
                b.rows[dest] = b.rows.get(dest, 0) + rows
                b.helper[dest] = helper_table[dest].index(expert)
        broadcasts.append(b)
    return Plan(broadcasts=broadcasts, helper_table=helper_table, load_after=load, target=target)
