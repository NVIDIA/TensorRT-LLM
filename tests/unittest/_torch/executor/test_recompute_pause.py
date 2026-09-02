# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types
from unittest.mock import Mock

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
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
    executor._pending_pause_replay_ids = set()
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


def _make_multimodal_request(request_id: int = 7) -> _StubRequest:
    request = _StubRequest(request_id)
    request.py_multimodal_data = {
        "modality_type": "image",
        "image": {"pixel_values": object()},
    }
    request.py_mm_encoder_state = object()
    return request


def test_pause_terminate_keeps_multimodal_payload_for_replay() -> None:
    executor = _make_executor(None)
    request = _make_multimodal_request()

    executor._terminate_requests([request], for_pause=True)

    executor.resource_manager.free_resources.assert_called_once_with(request)
    assert request.py_multimodal_data["modality_type"] == "image"
    assert "image" in request.py_multimodal_data
    assert request.py_mm_encoder_state is not None


def test_deferred_pause_terminate_keeps_multimodal_payload() -> None:
    handler = Mock()
    executor = _make_executor(handler)
    request = _make_multimodal_request()
    executor.active_requests = [request]

    executor._terminate_requests([request], for_pause=True)

    handler.terminate.assert_called_once_with(request)
    assert request.py_request_id in executor._pending_pause_replay_ids
    assert request.py_multimodal_data["modality_type"] == "image"

    executor._on_disagg_pp_termination(request)

    executor.resource_manager.free_resources.assert_called_once_with(request)
    assert request.py_request_id not in executor._pending_pause_replay_ids
    assert request.py_multimodal_data["modality_type"] == "image"
    assert "image" in request.py_multimodal_data
    assert request.py_mm_encoder_state is not None


def test_plain_terminate_still_strips_multimodal_payload() -> None:
    executor = _make_executor(None)
    request = _make_multimodal_request()

    executor._terminate_requests([request])

    executor.resource_manager.free_resources.assert_called_once_with(request)
    assert request.py_multimodal_data == {}
    assert request.py_mm_encoder_state is None


def _make_final_chunk_context_request(request_id: int = 7) -> _StubRequest:
    request = _make_multimodal_request(request_id)
    request.state = LlmRequestState.CONTEXT_INIT
    request.context_current_position = 0
    request.context_chunk_size = 4
    request.context_remaining_length = 0
    request.move_to_next_context_chunk = Mock()
    return request


def _run_update_request_states_tp(executor: PyExecutor, request: _StubRequest) -> None:
    executor.disable_overlap_scheduler = True
    scheduled_batch = ScheduledRequests()
    # `context_requests` is a read-only property over the chunking /
    # last-chunk backing lists; a final-chunk request belongs to the latter.
    scheduled_batch.context_requests_last_chunk = [request]
    executor._update_request_states_tp(scheduled_batch)


def test_post_prefill_strip_skipped_when_pause_replay_possible() -> None:
    executor = _make_executor(None)
    executor._retain_mm_data_for_pause_replay = True
    request = _make_final_chunk_context_request()

    _run_update_request_states_tp(executor, request)

    assert request.state == LlmRequestState.GENERATION_IN_PROGRESS
    assert request.py_multimodal_data["modality_type"] == "image"
    assert request.py_mm_encoder_state is not None


def test_post_prefill_strip_runs_when_pause_replay_impossible() -> None:
    executor = _make_executor(None)
    executor._retain_mm_data_for_pause_replay = False
    request = _make_final_chunk_context_request()

    _run_update_request_states_tp(executor, request)

    assert request.state == LlmRequestState.GENERATION_IN_PROGRESS
    assert request.py_multimodal_data == {}
    assert request.py_mm_encoder_state is None
