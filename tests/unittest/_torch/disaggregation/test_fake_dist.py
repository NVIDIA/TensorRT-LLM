# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract of the FakeDist test double: grouping, return shapes and diagnostics."""

import time

import numpy as np
import pytest
from coordinator_harness import CoordinatorHarness
from fake_dist import FakeDistGroup, FakeDistMismatch, FakeDistTimeout

from tensorrt_llm._torch.distributed.communicator import ReduceOp

pytestmark = pytest.mark.cpu_only


def test_tp_collectives_stay_in_the_tp_group_while_allreduce_spans_the_world() -> None:
    group = FakeDistGroup(world_size=4, tp_size=2)

    def step(rank):
        dist = group.rank(rank)
        return dist.tp_allgather(rank), dist.allreduce(rank, op=ReduceOp.SUM)

    results = group.run(step)

    assert [gathered for gathered, _ in results] == [[0, 1], [0, 1], [2, 3], [2, 3]]
    assert [total for _, total in results] == [6, 6, 6, 6]


def test_world_allreduce_reaches_ranks_in_other_tp_groups() -> None:
    """The shape the poison consensus relies on: TP groups of one, MAX over the world."""
    group = FakeDistGroup(world_size=2, tp_size=1)

    flags = group.run(lambda rank: group.rank(rank).allreduce(int(rank == 1), op=ReduceOp.MAX))

    assert flags == [1, 1]


def test_single_rank_tp_group_answers_locally_with_the_real_shapes() -> None:
    """A TP group of one never blocks and returns what a one-rank communicator
    would: a one-element list, the object itself and a (1, n) matrix. Wrapping
    the reduced value would turn a falsy 0 into a truthy [0]."""
    dist = FakeDistGroup(world_size=2, tp_size=1).rank(1)

    assert dist.tp_allgather(5) == [5]
    reduced = dist.tp_allreduce(0, op=ReduceOp.MAX)
    assert reduced == 0 and not reduced
    rows = dist.tp_allgather_int64([True, False])
    assert rows.dtype == np.int64 and rows.tolist() == [[1, 0]]
    assert dist.calls == [
        ("tp_allgather", 5),
        ("tp_allreduce", 0),
        ("tp_allgather_int64", [True, False]),
    ]


def test_int64_allgather_rows_are_ordered_by_tp_rank() -> None:
    group = FakeDistGroup(world_size=2, tp_size=2)

    results = group.run(lambda rank: group.rank(rank).tp_allgather_int64([rank, 10 + rank]))

    for rows in results:
        assert rows.dtype == np.int64
        assert rows.tolist() == [[0, 10], [1, 11]]


def test_gathered_payloads_are_copies() -> None:
    """Ranks receive copies, as after pickling; mutating one rank's result must
    not leak into another rank's view."""
    group = FakeDistGroup(world_size=2, tp_size=2)

    results = group.run(lambda rank: group.rank(rank).tp_allgather([rank]))

    results[0][1].append("mutated")
    assert results[1] == [[0], [1]]


class _MutatedAfterCopy(list):
    """A payload its sender appends to as soon as the fake has copied it.

    Stands in for a sender that modifies its input right after the call
    returns, timed so that the modification always lands before any peer
    could read a shared slot; only a snapshot taken before the wait is immune.
    """

    def __deepcopy__(self, memo):
        snapshot = list(self)
        self.append("late")
        return snapshot


def test_a_sender_mutating_its_input_after_sending_does_not_reach_its_peers() -> None:
    group = FakeDistGroup(world_size=2, tp_size=2)
    payload = _MutatedAfterCopy([1])

    results = group.run(lambda rank: group.rank(rank).tp_allgather(payload if rank == 1 else 0))

    assert payload == [1, "late"]
    assert results == [[0, [1]], [0, [1]]]
    assert group.rank(1).calls == [("tp_allgather", [1])]


def test_ranks_entering_different_collectives_are_reported() -> None:
    group = FakeDistGroup(world_size=2, tp_size=2)

    def step(rank):
        dist = group.rank(rank)
        return dist.tp_allreduce(1, op=ReduceOp.MAX) if rank == 0 else dist.tp_allgather([1])

    with pytest.raises(
        FakeDistMismatch, match=r"rank 0: tp_allreduce\[MAX\], rank 1: tp_allgather"
    ):
        group.run(step)


def test_ranks_reducing_with_different_operations_are_reported() -> None:
    """Entering the same reduce is not enough; the ranks must run the same operation."""
    group = FakeDistGroup(world_size=2, tp_size=2)

    def step(rank):
        return group.rank(rank).tp_allreduce(1, op=ReduceOp.MAX if rank == 0 else ReduceOp.MIN)

    with pytest.raises(
        FakeDistMismatch, match=r"rank 0: tp_allreduce\[MAX\], rank 1: tp_allreduce\[MIN\]"
    ):
        group.run(step)


def test_a_missing_rank_is_named_when_a_collective_times_out(clock) -> None:
    """Only rank 0 drains the timeout consensus. The barrier timeout is
    wall-clock inside ``threading``, so the frozen test clock neither stalls
    nor shortens it."""
    group = FakeDistGroup(world_size=2, tp_size=2, timeout_s=0.2)
    ranks = [CoordinatorHarness(dist=group.rank(i), enable_attention_dp=True) for i in range(2)]

    with pytest.raises(FakeDistTimeout, match=r"Missing: rank 1 \(last call: none\)") as failure:
        group.run(
            lambda rank: ranks[rank].coordinator.handle_timeouts_synced() if rank == 0 else None
        )

    assert "Arrived: rank 0 (tp_allgather_int64)" in str(failure.value)


def test_a_failing_rank_releases_its_peers_and_is_reported_first() -> None:
    group = FakeDistGroup(world_size=2, tp_size=2, timeout_s=5.0)

    def step(rank):
        if rank == 1:
            raise RuntimeError("rank 1 bug")
        group.rank(rank).tp_allreduce(1, op=ReduceOp.MAX)

    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="rank 1 bug"):
        group.run(step)
    assert time.perf_counter() - started < 2.0
