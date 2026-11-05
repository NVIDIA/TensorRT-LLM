# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transitions of the encoder-to-decoder handoff described above PyExecutor's encoder methods."""

import types
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import EncoderStepResult, PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue, KVCacheV2Scheduler
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    split_encoder_from_decoder_context_requests,
)
from tensorrt_llm.bindings import executor as trtllm

_HIDDEN_SIZE = 4
_MAX_INPUT_LEN = 1024


class _InflightRequestIds:
    def __init__(self):
        self.ids = set()

    def insert(self, request_id):
        self.ids.add(request_id)

    def erase(self, request_id):
        self.ids.discard(request_id)


def _make_request(request_id, encoder_input, encoder_len):
    if encoder_input == "token":
        encoder_kwargs = {"encoder_input_tokens": list(range(1, encoder_len + 1))}
    else:
        # The encoder downsamples features, so the output length is explicit.
        encoder_kwargs = {
            "encoder_input_features": torch.ones(2 * encoder_len, 8),
            "encoder_output_len": encoder_len,
        }
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=[1, 2, 3],
        sampling_config=trtllm.SamplingConfig(),
        is_streaming=False,
        **encoder_kwargs,
    )


def _make_executor(tp_size=1):
    executor = object.__new__(PyExecutor)
    executor.dist = types.SimpleNamespace(tp_size=tp_size, rank=0)
    executor.encoder_launch_executor = Mock()
    executor.pending_encoder_steps = []
    executor.inflight_req_ids = _InflightRequestIds()
    executor.execution_stream = Mock()
    executor.encoder_stream = Mock()
    executor._handle_errors = Mock()
    executor.max_input_len = _MAX_INPUT_LEN
    executor.canceled_req_ids = []
    executor.waiting_queue = FCFSWaitingQueue()
    executor.active_requests = []
    executor.enable_attention_dp = False
    executor.kv_cache_transceiver = None
    return executor


def _run_encoder(executor, requests, *, sync=False, launch_error=None, hidden_states=True):
    """Run one encoder step whose ready event has completed, then poll it."""
    lengths = [request.encoder_output_len for request in requests]
    ready_event = Mock()
    ready_event.query.return_value = True
    result = EncoderStepResult(
        hidden_states=torch.randn(sum(lengths), _HIDDEN_SIZE) if hidden_states else None,
        sequence_lengths=lengths,
        ready_event=ready_event,
    )
    future = Mock()
    future.result.return_value = result
    future.result.side_effect = launch_error
    executor.encoder_launch_executor.submit.return_value = future
    if sync:
        executor._run_encoder_step(requests)
    else:
        executor._submit_encoder_step(requests)
        executor._poll_encoder_steps()
    executor.encoder_launch_executor.submit.reset_mock(return_value=True)
    return result


def _cancel(executor, requests):
    executor.active_requests = list(requests)
    executor.canceled_req_ids = [request.py_request_id for request in requests]
    executor._handle_canceled_requests()


def _cross_kv_reservation(request):
    """Return what V2 admission and the V1 cross manager reserve for a context step."""
    v1_manager = object.__new__(KVCacheManager)
    v1_manager._preprepared_dummy_request_ids = set()
    v1_manager.kv_connector_manager = None
    return (
        KVCacheV2Scheduler._needs_cross_context_allocation(request),
        v1_manager._context_seq_len(request, is_cross=True),
    )


def _forward_context(executor, requests):
    batch = types.SimpleNamespace(context_requests=list(requests))
    executor._attach_encoder_output_to_execution_stream(batch)
    executor._mark_cross_kv_projection_consumed(batch)


@pytest.fixture
def recorded_streams():
    """Record ``Tensor.record_stream`` calls, which CPU tensors do not support."""
    calls = []

    def record_stream(tensor, stream):
        calls.append((tensor.data_ptr(), stream))

    with patch.object(torch.Tensor, "record_stream", record_stream):
        yield calls


@pytest.mark.parametrize("encoder_input", ["token", "feature"])
@pytest.mark.parametrize(
    "tp_size, sync", [(1, False), (2, False), (1, True)], ids=["tp1_async", "tp2_inline", "sync"]
)
def test_published_output_is_consumed_once(encoder_input, tp_size, sync, recorded_streams):
    request = _make_request(1, encoder_input, 5)
    executor = _make_executor(tp_size)

    result = _run_encoder(executor, [request], sync=sync)

    assert request.state == LlmRequestState.CONTEXT_INIT
    assert torch.equal(request.py_encoder_output, result.hidden_states)
    assert executor.inflight_req_ids.ids == set()
    assert executor.pending_encoder_steps == []
    assert _cross_kv_reservation(request) == (True, 5)

    output_ptr = request.py_encoder_output.data_ptr()
    _forward_context(executor, [request])

    assert recorded_streams == [(output_ptr, executor.execution_stream)]
    assert request.py_encoder_output is None
    assert request.py_encoder_output_ready_event is None
    assert _cross_kv_reservation(request) == (False, None)

    # Later chunks and decode steps neither project nor reserve cross-KV again.
    _forward_context(executor, [request])
    assert len(recorded_streams) == 1
    assert _cross_kv_reservation(request) == (False, None)


@pytest.mark.parametrize("encoder_input", ["token", "feature"])
def test_cancel_while_step_pending_is_not_revived(encoder_input):
    requests = [_make_request(i, encoder_input, length) for i, length in ((1, 2), (2, 3), (3, 4))]
    first, cancelled, last = requests
    executor = _make_executor()
    future = Mock()
    executor.encoder_launch_executor.submit.return_value = future
    ready_event = Mock()
    ready_event.query.return_value = False
    hidden_states = torch.randn(9, _HIDDEN_SIZE)
    future.result.return_value = EncoderStepResult(hidden_states, [2, 3, 4], ready_event)

    executor._submit_encoder_step(requests)
    _cancel(executor, [cancelled])
    assert cancelled.is_finished
    executor._poll_encoder_steps()
    assert executor.inflight_req_ids.ids == {1, 2, 3}

    ready_event.query.return_value = True
    executor._poll_encoder_steps()

    assert executor.inflight_req_ids.ids == set()
    assert executor.pending_encoder_steps == []
    assert torch.equal(first.py_encoder_output, hidden_states[0:2])
    assert torch.equal(last.py_encoder_output, hidden_states[5:9])
    assert first.state == last.state == LlmRequestState.CONTEXT_INIT
    assert cancelled.state == LlmRequestState.GENERATION_COMPLETE
    assert cancelled.py_encoder_output is None
    assert cancelled.py_encoder_output_ready_event is None


@pytest.mark.parametrize(
    "tp_size, sync", [(1, False), (2, False), (1, True)], ids=["tp1_async", "tp2_inline", "sync"]
)
@pytest.mark.parametrize("failure", ["launch", "publish"])
def test_encoder_failure_fails_only_unfinished_requests(tp_size, sync, failure):
    unfinished, cancelled = _make_request(1, "token", 2), _make_request(2, "token", 3)
    executor = _make_executor(tp_size)
    _cancel(executor, [cancelled])

    _run_encoder(
        executor,
        [unfinished, cancelled],
        sync=sync,
        launch_error=RuntimeError("encoder failed") if failure == "launch" else None,
        hidden_states=failure != "publish",
    )

    executor._handle_errors.assert_called_once()
    failed = executor._handle_errors.call_args.kwargs["requests"]
    assert [request.py_request_id for request in failed] == [1]
    assert executor.inflight_req_ids.ids == set()
    assert executor.pending_encoder_steps == []
    for request in (unfinished, cancelled):
        assert request.py_encoder_output is None
        assert request.py_encoder_output_ready_event is None


@pytest.mark.parametrize("encoder_input", ["token", "feature"])
@pytest.mark.parametrize("pause", ["v1_pause", "v2_recompute"])
def test_recompute_projects_cross_kv_once_per_cycle(encoder_input, pause, recorded_streams):
    request = _make_request(1, encoder_input, 5)
    executor = _make_executor()

    for cycle in range(2):
        _run_encoder(executor, [request])
        assert request.state == LlmRequestState.CONTEXT_INIT
        assert _cross_kv_reservation(request) == (True, 5)

        _forward_context(executor, [request])
        assert len(recorded_streams) == cycle + 1
        assert _cross_kv_reservation(request) == (False, None)

        request.context_chunk_size = request.context_remaining_length
        request.move_to_next_context_chunk()
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        request.add_new_token(9, 0)
        if pause == "v1_pause":
            executor._pause_requests([request])
        else:
            executor._pause_recompute_request(request)

        # Back in ENCODER_INIT, the request reaches no reader of the skip
        # flag until the next publish resets it.
        assert request.state == LlmRequestState.ENCODER_INIT
        assert request.py_encoder_output is None
        assert split_encoder_from_decoder_context_requests([request]) == ([request], [])
