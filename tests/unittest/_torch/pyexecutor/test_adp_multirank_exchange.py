# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Attention-DP host exchanges over a real two-process gloo group.

The router, ``RankState``, ``RequestBroadcaster`` and the ``PyExecutor`` control
request gate run unmodified; only the ``Distributed`` transport is a thin
adapter over ``torch.distributed`` (real collectives, no mocks or patching), so
a rank-divergent collective, a lossy wire format or a width mismatch fails here
the way it would on a cluster.
"""

import contextlib
import os
import tempfile
from datetime import timedelta

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.distributed.communicator import Distributed, ReduceOp
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import RequestBroadcaster
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter
from tensorrt_llm._torch.pyexecutor.scheduler.rank_state import (
    RankIterStatsPayload,
    RankKvCacheStats,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

WORLD_SIZE = 2


class _GlooDist(Distributed):
    """``Distributed`` over torch.distributed (gloo); every call is a real collective."""

    def __init__(self):
        world = dist.get_world_size()
        super().__init__(Mapping(world_size=world, tp_size=world, rank=dist.get_rank()))

    @property
    def rank(self):
        return dist.get_rank()

    @property
    def local_world_size(self):
        return dist.get_world_size()

    def barrier(self):
        dist.barrier()

    tp_barrier = barrier

    def _gather(self, obj):
        out = [None] * dist.get_world_size()
        dist.all_gather_object(out, obj)
        return out

    def broadcast(self, obj, root=0, prefer_cpu=False):
        box = [obj]
        dist.broadcast_object_list(box, src=root)
        return box[0]

    def tp_broadcast(self, obj, root=0, **kwargs):
        return self.broadcast(obj, root)

    def cp_broadcast(self, obj, root=0, **kwargs):
        return obj

    def allgather(self, obj, root=0):
        return self._gather(obj)

    def tp_allgather(self, obj, *, small_payload=False):
        return self._gather(obj)

    def cp_allgather(self, obj, *, small_payload=False):
        return [obj]

    def allreduce(self, obj, op=ReduceOp.SUM):
        values = self._gather(obj)
        return {ReduceOp.SUM: sum, ReduceOp.MAX: max, ReduceOp.MIN: min}[op](values)

    tp_allreduce = allreduce


class _NoHangDetector:
    def pause(self):
        return contextlib.nullcontext()


class _RequestQueue:
    """Rank 0's request queue: hands out what was enqueued and records every poll."""

    def __init__(self, items):
        self.items = list(items)
        self.polls = []

    def get_from_request_queue(self, timeout, wait_for_batch=True):
        self.polls.append(timeout)
        items, self.items = self.items, []
        return items


def _fetching_executor(rank, queued_ids):
    """A PyExecutor whose fetch path runs for real over the gloo group."""
    executor = object.__new__(PyExecutor)
    executor.dist = _GlooDist()
    executor.hang_detector = _NoHangDetector()
    executor.request_broadcaster = RequestBroadcaster(executor.dist, executor.hang_detector)
    executor.executor_request_queue = _RequestQueue(
        [RequestQueueItem(i) for i in queued_ids] if rank == 0 else []
    )
    executor.control_requests = []
    executor.request_accumulated = []
    executor.canceled_req_ids = []
    executor.is_shutdown = False
    executor._disable_mpi = False
    executor.enable_attention_dp = True
    executor.kv_cache_transceiver = None
    executor.async_transfer_manager = None
    executor.kv_connector_manager = None
    return executor


def _kv_snapshot(hit_rate):
    return RankKvCacheStats(1000, 400, 600, 32, 50, 20, 30, 10, hit_rate)


def _worker_rank_states_with_mixed_iteration_payloads(rank):
    """Rank 0 holds a measured KV snapshot, rank 1 only a placeholder."""
    router = DefaultADPRouter(_GlooDist())
    payload = RankIterStatsPayload(
        has_iter_stats=1,
        iter_stats_iter=5,
        kv_cache_stats=_kv_snapshot(0.37) if rank == 0 else None,
    )
    states = router.gather_all_rank_states([], iter_stats_payload=payload, iter_stats_enabled=True)
    assert [s.rank for s in states] == [0, 1]
    assert states[0].iter_stats.kv_cache_stats == _kv_snapshot(0.37)
    assert states[1].iter_stats.kv_cache_stats is None
    assert all(s.iter_stats.iter_stats_iter == 5 for s in states)


def _worker_int64_gathers_carry_the_router_state_and_extras(rank):
    """Without iteration stats the RankState rides the fixed-width int64 gather, alone or with the packed votes."""
    router = DefaultADPRouter(_GlooDist())
    pending = 3 if rank == 0 else 0
    states = router.gather_all_rank_states([], num_pending_new_requests=pending)
    assert [s.rank for s in states] == [0, 1]
    assert [s.num_pending_new_requests for s in states] == [3, 0]
    assert all(s.num_active_requests == 0 for s in states)
    states, extras = router.gather_all_rank_states_with_extras(
        [], [rank + 1, 10], num_pending_new_requests=pending
    )
    assert [s.num_pending_new_requests for s in states] == [3, 0]
    assert extras == [[1, 10], [2, 10]]


def _worker_packed_extras_reject_a_width_mismatch(rank):
    router = DefaultADPRouter(_GlooDist())
    extras = [1, 2] if rank == 0 else [1, 2, 3]
    with pytest.raises(ValueError):
        router.gather_all_rank_states_with_extras([], extras)


def _worker_zero_hint_then_empty_probe_agree(rank):
    broadcaster = RequestBroadcaster(_GlooDist(), _NoHangDetector())
    received = broadcaster.broadcast([], known_nonempty=False)
    assert received == []


def _worker_drain_waits_for_the_busiest_rank(rank):
    """Replicated TP ranks vote: a rank that is locally drained must not fire alone."""
    executor = object.__new__(PyExecutor)
    executor.dist = _GlooDist()
    executor.enable_attention_dp = False
    executor.control_requests = [RequestQueueItem(1, control_requires_drain=True)]
    executor.active_requests = [] if rank == 0 else [object()]
    executor.waiting_queue = FCFSWaitingQueue()
    executor.kv_connector_manager = None
    executor.async_transfer_manager = None
    executor._control_drain_transfer_deadline = None
    executor._handle_control_request()
    # Rank 1 is busy, so the vote is not unanimous: nobody fired, the request stays parked.
    assert len(executor.control_requests) == 1


def _worker_busy_zero_hint_skips_the_fetch_on_every_rank(rank):
    executor = _fetching_executor(rank, queued_ids=[21, 22])
    waiting_queue = FCFSWaitingQueue()
    executor._fetch_and_enqueue_requests(waiting_queue, 4, new_requests_hint=0)
    assert len(waiting_queue) == 0
    # Rank 0 never even polled its queue; the items are still there for the next iteration.
    assert executor.executor_request_queue.polls == []
    assert len(executor.executor_request_queue.items) == (2 if rank == 0 else 0)


def _worker_busy_positive_hint_delivers_to_every_rank(rank):
    executor = _fetching_executor(rank, queued_ids=[21, 22])
    waiting_queue = FCFSWaitingQueue()
    executor._fetch_and_enqueue_requests(waiting_queue, 4, new_requests_hint=2)
    assert sorted(item.id for item in waiting_queue) == [21, 22]


def _worker_busy_without_hint_probes_and_delivers(rank):
    executor = _fetching_executor(rank, queued_ids=[21])
    waiting_queue = FCFSWaitingQueue()
    executor._fetch_and_enqueue_requests(waiting_queue, 4, new_requests_hint=None)
    assert sorted(item.id for item in waiting_queue) == [21]


def _entry(rank, world_size, store, worker_name):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        globals()[worker_name](rank)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run(worker):
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _entry,
            args=(WORLD_SIZE, os.path.join(tmp, "store"), worker.__name__),
            nprocs=WORLD_SIZE,
            join=True,
        )


def test_rank_states_carry_float_hit_rate_and_mixed_payload_widths():
    _run(_worker_rank_states_with_mixed_iteration_payloads)


def test_int64_gathers_carry_the_router_state_and_the_packed_extras():
    _run(_worker_int64_gathers_carry_the_router_state_and_extras)


def test_packed_extras_width_mismatch_fails_on_every_rank():
    _run(_worker_packed_extras_reject_a_width_mismatch)


def test_empty_probe_returns_nothing_on_every_rank():
    _run(_worker_zero_hint_then_empty_probe_agree)


def test_control_drain_waits_for_the_busiest_tp_rank():
    _run(_worker_drain_waits_for_the_busiest_rank)


def test_busy_zero_hint_skips_the_fetch_on_every_rank():
    _run(_worker_busy_zero_hint_skips_the_fetch_on_every_rank)


def test_busy_positive_hint_delivers_the_requests_to_every_rank():
    _run(_worker_busy_positive_hint_delivers_to_every_rank)


def test_busy_without_a_hint_still_probes_and_delivers():
    _run(_worker_busy_without_hint_probes_and_delivers)
