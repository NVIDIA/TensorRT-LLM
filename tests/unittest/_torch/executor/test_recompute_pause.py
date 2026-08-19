# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types
from unittest.mock import Mock

from tensorrt_llm._torch.pyexecutor.py_executor import _UNBOUNDED_PAUSE_MAX_INPUT_LEN, PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings.internal.batch_manager import ReqIdsSet


class _StubRequest:
    def __init__(self, request_id: int = 7) -> None:
        self.request_id = request_id
        self.py_request_id = request_id
        self.is_dummy_request = False
        self.is_finished = False
        self.reset_for_recompute = Mock()


def _make_executor(handler: Mock | None) -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor._disagg_pp_termination_handler = handler
    executor._pending_recompute_pause_ids = set()
    executor.inflight_req_ids = ReqIdsSet()
    executor.resource_manager = Mock()
    executor._prefetched_request_ids = set()
    executor._disagg_timed_out_ctx_cancelled_ids = set()
    executor._disagg_timed_out_gen_cancelled_ids = set()
    executor.gather_all_responses = False
    executor.dist = types.SimpleNamespace(rank=0)
    executor.result_wait_queues = {}
    executor.active_requests = []
    return executor


def test_recompute_pause_does_not_apply_executor_max_input_len() -> None:
    executor = types.SimpleNamespace(max_input_len=5)
    request = _StubRequest()

    PyExecutor._pause_recompute_request(executor, request)

    request.reset_for_recompute.assert_called_once_with(_UNBOUNDED_PAUSE_MAX_INPUT_LEN)


def test_recompute_pause_skips_request_finished_during_overlap() -> None:
    executor = _make_executor(None)
    request = _StubRequest()
    executor.active_requests = [request]
    scheduled_batch = ScheduledRequests()
    scheduled_batch.recompute_paused_requests = [request]

    executor._terminate_recompute_paused_requests(scheduled_batch)
    request.is_finished = True
    executor._pause_recompute_paused_requests(scheduled_batch)

    executor.resource_manager.free_resources.assert_called_once_with(request)
    request.reset_for_recompute.assert_not_called()


def test_recompute_pause_defers_reset_until_pp_consensus() -> None:
    handler = Mock()
    executor = _make_executor(handler)
    request = _StubRequest()
    executor.active_requests = [request]
    executor.result_wait_queues[request.py_request_id] = Mock()
    scheduled_batch = ScheduledRequests()
    scheduled_batch.recompute_paused_requests = [request]
    events = []
    executor.resource_manager.free_resources.side_effect = lambda _request: events.append("free")
    request.reset_for_recompute.side_effect = lambda _max_input_len: events.append("reset")

    executor._terminate_recompute_paused_requests(scheduled_batch)
    executor._pause_recompute_paused_requests(scheduled_batch)

    handler.terminate.assert_called_once_with(request)
    assert events == []
    assert request.py_request_id in executor._pending_recompute_pause_ids
    assert request.py_request_id in executor.inflight_req_ids
    assert request.py_request_id in executor.result_wait_queues

    executor._progress_recompute_pause_termination_if_idle(0)
    handler.terminate_pending_requests.assert_called_once_with()

    executor._on_disagg_pp_termination(request)

    assert events == ["free", "reset"]
    assert request.py_request_id not in executor._pending_recompute_pause_ids
    assert request.py_request_id not in executor.inflight_req_ids
    assert request.py_request_id in executor.result_wait_queues


def test_terminal_request_does_not_recompute_after_pp_consensus() -> None:
    executor = _make_executor(Mock())
    request = _StubRequest()
    request.is_finished = True
    executor.active_requests = [request]
    executor.result_wait_queues[request.py_request_id] = Mock()
    executor._pending_recompute_pause_ids.add(request.py_request_id)
    executor.inflight_req_ids.insert(request.py_request_id)

    executor._on_disagg_pp_termination(request)

    executor.resource_manager.free_resources.assert_called_once_with(request)
    request.reset_for_recompute.assert_not_called()
    assert request.py_request_id not in executor._pending_recompute_pause_ids
    assert request.py_request_id not in executor.inflight_req_ids
    assert request.py_request_id not in executor.result_wait_queues
