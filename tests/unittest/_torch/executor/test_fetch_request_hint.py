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
"""Tests for the pending-request hint gating in PyExecutor._fetch_and_enqueue_requests.

The hint is rank 0's request-queue snapshot piggybacked on the ADP rank-state
allgather (RankState.num_pending_new_requests). On the busy path it must:
- skip the fetch entirely when 0 (identically on every rank — the hint comes
  from the same allgather everywhere),
- skip the request-count probe broadcast when > 0,
- be ignored on the idle path, which keeps the blocking probe as the wake
  mechanism, and when it is None (non-ADP callers).
"""

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue


def _make_stub_executor(rank=0, queued_items=None):
    queued_items = queued_items if queued_items is not None else []
    stub = SimpleNamespace()
    stub.control_requests = []
    stub.request_accumulated = []
    stub.canceled_req_ids = []
    stub._disable_mpi = False
    stub.is_shutdown = False
    stub.dist = Mock(rank=rank)
    stub.hang_detector = MagicMock()
    stub.executor_request_queue = Mock()
    stub.executor_request_queue.get_from_request_queue = Mock(return_value=list(queued_items))
    stub.request_broadcaster = Mock()
    # Echo rank 0's requests back, mirroring a world broadcast.
    stub.request_broadcaster.broadcast = Mock(side_effect=lambda reqs, **kw: (list(reqs), None))
    stub._handle_special_queue_items = Mock(side_effect=lambda reqs: reqs)
    # No KV-cache transceiver: the idle wait is unbounded (MPI) or the Ray
    # heartbeat; tests for the transfer-bounded idle wait set these.
    stub.kv_cache_transceiver = None
    stub.async_transfer_manager = None
    stub.enable_attention_dp = False
    stub._idle_kv_transfer_poll_timeout = (
        lambda: PyExecutor._idle_kv_transfer_poll_timeout(stub))
    return stub


def _fetch(stub, waiting_queue, total_num_active_requests, new_requests_hint):
    PyExecutor._fetch_and_enqueue_requests(
        stub,
        waiting_queue,
        total_num_active_requests,
        new_requests_hint=new_requests_hint,
    )


def test_busy_zero_hint_skips_fetch_and_collectives():
    stub = _make_stub_executor(queued_items=[Mock()])
    waiting_queue = FCFSWaitingQueue()

    _fetch(stub, waiting_queue, total_num_active_requests=4, new_requests_hint=0)

    stub.executor_request_queue.get_from_request_queue.assert_not_called()
    stub.request_broadcaster.broadcast.assert_not_called()
    assert len(waiting_queue) == 0


def test_busy_positive_hint_skips_probe_but_fetches():
    items = [Mock(), Mock()]
    stub = _make_stub_executor(queued_items=items)
    waiting_queue = FCFSWaitingQueue()

    _fetch(stub, waiting_queue, total_num_active_requests=4, new_requests_hint=2)

    stub.executor_request_queue.get_from_request_queue.assert_called_once_with(
        datetime.timedelta(0)
    )
    stub.request_broadcaster.broadcast.assert_called_once()
    assert stub.request_broadcaster.broadcast.call_args.kwargs["known_nonempty"] is True
    assert len(waiting_queue) == len(items)


def test_busy_no_hint_keeps_probe_path():
    items = [Mock()]
    stub = _make_stub_executor(queued_items=items)
    waiting_queue = FCFSWaitingQueue()

    _fetch(stub, waiting_queue, total_num_active_requests=4, new_requests_hint=None)

    stub.executor_request_queue.get_from_request_queue.assert_called_once()
    assert stub.request_broadcaster.broadcast.call_args.kwargs["known_nonempty"] is False
    assert len(waiting_queue) == len(items)


def test_idle_ignores_hint_and_keeps_blocking_probe():
    stub = _make_stub_executor(queued_items=[])
    waiting_queue = FCFSWaitingQueue()

    _fetch(stub, waiting_queue, total_num_active_requests=0, new_requests_hint=0)

    # Idle path: the hint must not skip the (blocking) fetch/probe wake path.
    stub.executor_request_queue.get_from_request_queue.assert_called_once_with(None)
    stub.request_broadcaster.broadcast.assert_called_once()
    assert stub.request_broadcaster.broadcast.call_args.kwargs["known_nonempty"] is False


def test_pending_control_requests_block_fetch():
    stub = _make_stub_executor(queued_items=[Mock()])
    stub.control_requests = [Mock()]
    waiting_queue = FCFSWaitingQueue()

    _fetch(stub, waiting_queue, total_num_active_requests=4, new_requests_hint=3)

    stub.executor_request_queue.get_from_request_queue.assert_not_called()
    stub.request_broadcaster.broadcast.assert_not_called()


def _with_tracked_transfers(stub, local_inflight, poll_interval_ms=5000):
    stub.kv_cache_transceiver = SimpleNamespace(
        kv_transfer_poll_interval_ms=poll_interval_ms)
    stub.async_transfer_manager = Mock()
    stub.async_transfer_manager.has_any_inflight_requests = Mock(
        return_value=local_inflight)
    return stub


def test_idle_with_tracked_context_transfer_polls_instead_of_blocking():
    # A context-only request whose KV the generation side has not pulled yet
    # is no longer active, but the loop must keep polling the transceiver so
    # the transfer is reaped when it completes and the transfer timeout
    # measures transfer time, not idle time.
    stub = _with_tracked_transfers(_make_stub_executor(), local_inflight=True)
    waiting_queue = FCFSWaitingQueue()

    _fetch(stub, waiting_queue, total_num_active_requests=0, new_requests_hint=None)

    stub.executor_request_queue.get_from_request_queue.assert_called_once_with(
        datetime.timedelta(milliseconds=5000)
    )
    assert stub.request_broadcaster.broadcast.call_args.kwargs["prefer_cpu"] is True


def test_idle_with_tracked_transfer_bounds_the_ray_heartbeat():
    stub = _with_tracked_transfers(_make_stub_executor(), local_inflight=True)
    stub._disable_mpi = True
    waiting_queue = FCFSWaitingQueue()

    _fetch(stub, waiting_queue, total_num_active_requests=0, new_requests_hint=None)

    stub.executor_request_queue.get_from_request_queue.assert_called_once_with(
        datetime.timedelta(milliseconds=5000)
    )


def test_idle_without_tracked_transfer_keeps_blocking_probe():
    stub = _with_tracked_transfers(_make_stub_executor(), local_inflight=False)
    waiting_queue = FCFSWaitingQueue()

    _fetch(stub, waiting_queue, total_num_active_requests=0, new_requests_hint=None)

    stub.executor_request_queue.get_from_request_queue.assert_called_once_with(None)


def test_idle_transfer_poll_is_tp_wide_under_attention_dp():
    # Under attention DP the ranks own different requests: a transfer tracked
    # on a peer rank must bound rank 0's wait too, so the flag is combined
    # with a TP-wide max (called on every rank on the rank-consistent idle
    # path).
    stub = _with_tracked_transfers(_make_stub_executor(), local_inflight=False)
    stub.enable_attention_dp = True
    stub.dist = Mock(rank=0, tp_size=8)
    stub.dist.tp_allreduce = Mock(return_value=1)
    waiting_queue = FCFSWaitingQueue()

    _fetch(stub, waiting_queue, total_num_active_requests=0, new_requests_hint=None)

    stub.dist.tp_allreduce.assert_called_once()
    assert stub.dist.tp_allreduce.call_args.args[0] == 0
    stub.executor_request_queue.get_from_request_queue.assert_called_once_with(
        datetime.timedelta(milliseconds=5000)
    )


def test_busy_path_ignores_tracked_transfers():
    stub = _with_tracked_transfers(_make_stub_executor(queued_items=[Mock()]),
                                   local_inflight=True)
    stub.enable_attention_dp = True
    stub.dist = Mock(rank=0, tp_size=8)
    waiting_queue = FCFSWaitingQueue()

    _fetch(stub, waiting_queue, total_num_active_requests=3, new_requests_hint=None)

    stub.dist.tp_allreduce.assert_not_called()
    stub.executor_request_queue.get_from_request_queue.assert_called_once_with(
        datetime.timedelta(0)
    )


def test_idle_poll_falls_back_to_one_second_without_configured_interval():
    stub = _with_tracked_transfers(_make_stub_executor(), local_inflight=True,
                                   poll_interval_ms=None)
    waiting_queue = FCFSWaitingQueue()

    _fetch(stub, waiting_queue, total_num_active_requests=0, new_requests_hint=None)

    stub.executor_request_queue.get_from_request_queue.assert_called_once_with(
        datetime.timedelta(milliseconds=1000)
    )
