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
"""The collectives of ranks over the rendezvous store.

Ranks run as threads over an in-memory store.
"""

import pickle
import threading

import pytest
from _torch.distributed.store_group_test_utils import InMemoryStore, Unpickled
from mpi4py import MPI

from tensorrt_llm._torch.distributed.membership import SignedStore, StoreGroup, rendezvous

pytestmark = pytest.mark.cpu_only

JOIN_TIMEOUT_S = 10.0


def _group(store, rank, ranks, **options):
    return StoreGroup(store, "group", rank, ranks, **options)


def _run_ranks(calls):
    """Run each rank's call on its own thread, and return its result or exception by rank."""
    results = {}

    def run(rank, call):
        try:
            results[rank] = call()
        except Exception as error:
            results[rank] = error

    threads = [threading.Thread(target=run, args=item, daemon=True) for item in calls.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT_S)
    assert not any(thread.is_alive() for thread in threads), "a rank did not return"
    return results


def test_split_orders_members_by_key():
    """Check that a split orders its members by key, as MPI_Comm_split does."""
    store = InMemoryStore()
    groups = {r: _group(store, r, range(4)) for r in range(4)}

    def split(rank):
        subgroup = groups[rank].Split(color=rank % 2, key=-rank)
        return subgroup.membership, subgroup.allgather(rank)

    results = _run_ranks({r: lambda r=r: split(r) for r in groups})
    assert results[0] == results[2] == ((2, 0), [2, 0])
    assert results[1] == results[3] == ((3, 1), [3, 1])


def test_split_leaves_out_inactive_members():
    """Split the active members only, as an inactive member's color is unknown."""
    store = InMemoryStore()
    active = (0, 1, 3)
    groups = {r: _group(store, r, range(4), active=active) for r in active}

    def split(rank):
        subgroup = groups[rank].Split(color=rank % 2, key=rank)
        return subgroup.membership, subgroup.allgather(rank)

    results = _run_ranks({r: lambda r=r: split(r) for r in groups})
    assert results[0] == ((0,), [0])
    assert results[1] == results[3] == ((1, 3), [1, 3])


def test_split_type_groups_ranks_by_node():
    """Check that a shared-memory split holds the node's ranks, and other split types raise."""
    group = _group(InMemoryStore(), 3, range(6), gpus_per_node=2)
    assert group.Split_type(MPI.COMM_TYPE_SHARED).membership == (2, 3)
    with pytest.raises(NotImplementedError):
        group.Split_type(MPI.COMM_TYPE_SHARED + 1000)


def test_split_type_needs_gpus_per_node():
    """Refuse a node split of a group built without the ranks per node."""
    group = _group(InMemoryStore(), 3, range(6))
    with pytest.raises(ValueError, match="gpus_per_node"):
        group.Split_type(MPI.COMM_TYPE_SHARED)


@pytest.mark.parametrize("op, expected", [("SUM", 6), ("MIN", 1), ("MAX", 3)])
def test_allreduce_over_active_members(op, expected):
    """Reduce over the active members' values only."""
    store = InMemoryStore()
    values = {0: 3, 1: 1, 3: 2}
    groups = {r: _group(store, r, range(4), active=values) for r in values}
    results = _run_ranks(
        {r: lambda r=r: groups[r].allreduce(values[r], getattr(MPI, op)) for r in groups}
    )
    assert results == {r: expected for r in values}


@pytest.mark.parametrize("root", [2, 1], ids=["active", "inactive"])
def test_bcast_over_active_members(root):
    """Broadcast the root's object to the active members, or None from an inactive root."""
    store = InMemoryStore()
    active = (0, 2, 3)
    groups = {r: _group(store, r, range(4), active=active) for r in active}
    results = _run_ranks({r: lambda r=r: groups[r].bcast(f"from {r}", root) for r in groups})
    assert results == {r: f"from {root}" if root in active else None for r in active}


def test_on_wait_error_fails_exchange():
    """Check that an error raised by ``on_wait`` ends the exchange and marks it failed."""
    waiting = []

    def on_wait(mask):
        waiting.append(mask)
        if mask:
            raise TimeoutError("rank 1 stopped")

    group = _group(InMemoryStore(), 0, (0, 1), on_wait=on_wait)
    with pytest.raises(TimeoutError):
        group.allgather(0)
    assert waiting == [0b10, 0]
    assert group.exchange_failed


def test_exchange_failed_tells_latest_exchange():
    """Check that ``exchange_failed`` tells whether the latest exchange raised."""
    failing = [True]

    def on_wait(mask):
        if failing[0]:
            raise TimeoutError("stopped waiting")

    group = _group(InMemoryStore(), 0, (0,), on_wait=on_wait)
    with pytest.raises(TimeoutError):
        group.allgather(0)
    assert group.exchange_failed
    failing[0] = False
    assert group.allgather(1) == [1]
    assert not group.exchange_failed


def test_exchange_keeps_only_latest_call_keys():
    """Delete a call's keys once the next call completes, so only the latest call's keys stay."""
    store = InMemoryStore()
    groups = {r: _group(store, r, (0, 1)) for r in (0, 1)}
    _run_ranks({r: lambda r=r: [groups[r].allgather(call) for call in range(3)] for r in groups})
    assert sorted(store.values) == ["group/2/0", "group/2/1", "group/2/arrived/0"]


def test_waiting_mask_beyond_first_counter(monkeypatch):
    """Report a missing rank 64, whose arrival bit is in the second counter."""
    monkeypatch.setattr(rendezvous, "REPORT_INTERVAL_S", 0.01)
    store = InMemoryStore()
    waiting = {0: [], 63: []}
    rank_63_waits = threading.Event()

    def on_wait(rank):
        def report(mask):
            waiting[rank].append(mask)
            if rank == 63:
                rank_63_waits.set()
            if mask == 1 << 64:
                raise TimeoutError("rank 64 stopped")

        return report

    groups = {r: _group(store, r, (0, 63, 64), on_wait=on_wait(r)) for r in (0, 63)}
    results = _run_ranks(
        {
            0: lambda: rank_63_waits.wait(10) and groups[0].allgather(0),
            63: lambda: groups[63].allgather(63),
        }
    )
    assert all(isinstance(result, TimeoutError) for result in results.values())
    assert waiting[0] == [1 << 64, 0]


@pytest.mark.parametrize("forger_secret", [None, "cd" * 32], ids=["unsigned", "other_secret"])
def test_forged_store_value_is_never_unpickled(monkeypatch, forger_secret):
    """Reject a pickle that a process without the deployment's secret set, before unpickling it."""
    store = InMemoryStore()
    forger = store if forger_secret is None else SignedStore(store, forger_secret)
    signed = SignedStore(store, "ab" * 32)
    monkeypatch.setattr(Unpickled, "calls", [])
    forged = pickle.dumps(Unpickled())
    # Rank 1's value and arrival at the first call of a group of ranks 0 and 1.
    forger.set("forged/0/1", forged)
    forger.add("forged/0/arrived/0", 1 << 1)
    with pytest.raises(ValueError, match="is not signed with this deployment's"):
        StoreGroup(signed, "forged", 0, [0, 1]).allgather("rank 0")
    assert Unpickled.calls == []
    # The deployment's own values pass.
    signed.set("own", forged)
    assert signed.get("own") == forged
