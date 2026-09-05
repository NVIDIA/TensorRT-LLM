# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-iteration attention-DP host exchanges use one int64 collective each,
and every TP rank enters the same collectives regardless of its local batch."""

import threading
import types
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


class _LockstepDist:
    """Two TP ranks driven from two threads; every tp_allgather_int64 is a real rendezvous.

    Each call deposits the caller's vector, meets the peer at a barrier and
    returns the rows built from what both ranks actually sent. A rank that
    skips a collective, or sends a vector of a different length, breaks the
    barrier after ``timeout`` seconds instead of hanging the test.
    """

    def __init__(self, timeout=5.0):
        self.barrier = threading.Barrier(2, timeout=timeout)
        self.lock = threading.Lock()
        self.sent = {}
        self.calls = {0: [], 1: []}
        self.rows = []

    def for_rank(self, rank):
        dist = self

        class _Proxy:
            def tp_allgather_int64(self, values):
                vec = list(values)
                with dist.lock:
                    dist.calls[rank].append(vec)
                    dist.sent[rank] = vec
                dist.barrier.wait()
                with dist.lock:
                    rows = [dist.sent[0], dist.sent[1]]
                    if len(rows[0]) != len(rows[1]):
                        dist.barrier.abort()
                        raise AssertionError(f"ranks sent vectors of different lengths: {rows}")
                    if rank == 0:
                        dist.rows.append(rows)
                dist.barrier.wait()
                return np.array(rows, dtype=np.int64)

        return _Proxy()


def _make_batch(num_generation, num_context=0):
    batch = ScheduledRequests()
    batch.generation_requests = [SimpleNamespace(py_request_id=i) for i in range(num_generation)]
    batch.context_requests_last_chunk = [
        SimpleNamespace(py_request_id=100 + i) for i in range(num_context)
    ]
    return batch


def _make_runner(dist, max_batch=64, dummy_ok=True):
    runner = object.__new__(CUDAGraphRunner)
    runner.enabled = True
    runner.padding_enabled = True
    runner.max_supported_batch_size = max_batch
    runner.enable_encoder_decoder_mixed_cuda_graph = False
    runner.config = SimpleNamespace(
        enable_attention_dp=True,
        mapping=SimpleNamespace(tp_size=2),
        dist=dist,
        batch_size=max_batch,
        use_mrope=False,
    )
    runner._round_up_batch_size_with_draft_len = lambda bs, draft_len: 4 if bs <= 4 else 8
    dummy = SimpleNamespace(py_request_id=-1) if dummy_ok else None
    runner._get_or_create_padding_dummy = lambda rm, draft_len: dummy
    runner._is_mixed_encoder_decoder_batch = Mock(return_value=False)
    runner.get_graph_key = Mock(return_value=None)
    return runner


def _run_iteration(runner, batch):
    padding = CUDAGraphRunner._get_padded_batch(runner, batch, Mock(), 0)
    result = CUDAGraphRunner.maybe_get_cuda_graph(
        runner, batch, enable_spec_decode=False, attn_metadata=object()
    )
    return padding, result


def _run_two_ranks(dist, batches, runner_kwargs=({}, {})):
    """Run one iteration on both ranks concurrently; surface a hang or a rank failure."""
    results = {}

    def run(rank):
        try:
            runner = _make_runner(dist.for_rank(rank), **runner_kwargs[rank])
            results[rank] = _run_iteration(runner, batches[rank])
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            results[rank] = exc

    threads = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads), "a rank is stuck in a collective"
    for rank in (0, 1):
        if isinstance(results[rank], BaseException):
            raise AssertionError(f"rank {rank} failed: {repr(results[rank])}") from results[rank]
    return results[0], results[1]


def _assert_payload_types(dist):
    """The gathered vector is [can_run_cuda_graph, batch_size]: a bool and an int, on both ranks."""
    for calls in dist.calls.values():
        for flag, size in calls:
            assert type(flag) is bool and type(size) is int


def test_pad_batch_pads_to_the_largest_eligible_rank():
    dist = _LockstepDist()
    batch0, batch1 = _make_batch(3), _make_batch(4)

    (padding0, _), (padding1, _) = _run_two_ranks(dist, (batch0, batch1))

    assert (padding0, padding1) == (1, 0)
    assert (batch0.batch_size, batch1.batch_size) == (4, 4)
    assert dist.calls[0] == [[True, 3], [True, 4]]
    assert dist.calls[1] == [[True, 4], [True, 4]]
    assert dist.rows == [[[True, 3], [True, 4]], [[True, 4], [True, 4]]]
    _assert_payload_types(dist)


def test_ineligible_peer_does_not_skip_any_collective():
    # Rank 1 still has a context request, so only rank 0 pads its batch;
    # both ranks must still enter the pad and graph-lookup gathers.
    dist = _LockstepDist()
    batch0, batch1 = _make_batch(3), _make_batch(1, num_context=1)

    (padding0, result0), (padding1, result1) = _run_two_ranks(dist, (batch0, batch1))

    assert (padding0, padding1) == (1, 0)
    assert result0 == result1 == (None, None, None)
    assert dist.calls[0] == [[True, 3], [True, 4]]
    assert dist.calls[1] == [[False, 2], [False, 2]]
    _assert_payload_types(dist)


def test_local_padding_dummy_failure_does_not_skip_any_collective():
    dist = _LockstepDist()
    batch0, batch1 = _make_batch(3), _make_batch(4)

    (padding0, result0), (padding1, result1) = _run_two_ranks(
        dist, (batch0, batch1), runner_kwargs=({"dummy_ok": False}, {})
    )

    assert (padding0, padding1) == (0, 0)
    assert result0 == result1 == (None, None, None)
    assert dist.calls[0] == [[True, 3], [True, 3]]
    assert dist.calls[1] == [[True, 4], [True, 4]]
    _assert_payload_types(dist)


def test_lockstep_fake_fails_fast_when_a_rank_skips_a_collective():
    dist = _LockstepDist(timeout=1.0)
    rank0, rank1 = dist.for_rank(0), dist.for_rank(1)
    errors = {}

    def run(rank, proxy, count):
        try:
            for _ in range(count):
                proxy.tp_allgather_int64([1, 2])
        except threading.BrokenBarrierError as exc:
            errors[rank] = exc

    threads = [
        threading.Thread(target=run, args=(0, rank0, 2), daemon=True),
        threading.Thread(target=run, args=(1, rank1, 1), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    assert 0 in errors, "the rank left alone in the collective must not hang"
    assert dist.rows == [[[1, 2], [1, 2]]]


def test_can_queue_gathers_only_the_batch_size():
    executor = object.__new__(PyExecutor)
    executor.enable_attention_dp = True
    executor.dist = SimpleNamespace(tp_allgather_int64=Mock(return_value=np.array([[2], [0]])))

    can_queue, this_rank = PyExecutor._can_queue(executor, types.SimpleNamespace(batch_size=2))

    executor.dist.tp_allgather_int64.assert_called_once_with([2])
    assert can_queue is False and this_rank is True


class _ObjectPathDist(Distributed):
    """Minimal backend without buffer collectives: exercises the defaults."""

    def __init__(self, mapping, peers):
        super().__init__(mapping)
        self.peers = peers

    @property
    def local_world_size(self):
        return len(self.peers)

    def barrier(self):
        pass

    def tp_barrier(self):
        pass

    def broadcast(self, obj, root=0):
        return obj

    def allgather(self, obj, root=0):
        return [obj] * len(self.peers)

    def allreduce(self, obj, op=None):
        return obj

    def tp_allreduce(self, obj, op=None):
        return obj

    def tp_broadcast(self, obj, root=0, **kwargs):
        return obj

    def cp_broadcast(self, obj, root=0, **kwargs):
        return obj

    def tp_allgather(self, obj, *, small_payload=False):
        return [list(p) for p in self.peers]

    def cp_allgather(self, obj, *, small_payload=False):
        return [obj]


def test_default_int64_helpers_route_through_object_paths():
    mapping = Mapping(world_size=2, rank=0, tp_size=2)
    dist = _ObjectPathDist(mapping, peers=[[5, 6], [7, 8]])

    rows = dist.tp_allgather_int64([5, 6])
    assert rows.dtype == np.int64 and rows.tolist() == [[5, 6], [7, 8]]
    assert dist.tp_cp_allgather_int64([5, 6]).tolist() == [[5, 6], [7, 8]]
    assert dist.broadcast_int64([3]).tolist() == [3]
    assert dist.tp_cp_broadcast_int64([4, 5]).tolist() == [4, 5]
