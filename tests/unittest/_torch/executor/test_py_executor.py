# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PyExecutor request handling functionality.

This module tests the request handling logic that was moved from ExecutorRequestQueue
to PyExecutor, including:
- _handle_special_queue_items method
- canceled_req_ids management
- waiting_queue management
- is_shutdown state management
- expected_num_active_requests tracking
- Event-loop crash propagation to await_responses callers (nvbug 6038228)
"""

import threading
import time
import types
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp

from tensorrt_llm._torch.disaggregation.executor.admission import DisaggTransferAdmissionController
from tensorrt_llm._torch.distributed.communicator import ReduceOp
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    SHUTDOWN_REQUEST_ID,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    LlmRequestState,
    LlmResponse,
    SamplingConfig,
)
from tensorrt_llm._torch.pyexecutor.py_executor import (
    ATTENTION_DP_DUMMY_REQUEST_ID,
    EncoderStepResult,
    PyExecutor,
    _ADPForwardIntent,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import NoFreeSlotsError, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import (
    FCFSWaitingQueue,
    RequestScheduler,
    ScheduledRequests,
    SerializableSchedulerOutput,
)
from tensorrt_llm.llmapi.llm_args import EncodeCudaGraphConfig, MTPDecodingConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import OutOfPagesError

pytestmark = pytest.mark.cpu_only


class _TorchCollectiveDist:
    """Minimal distributed adapter for idle-progress collective tests."""

    world_size = 2
    tp_size = 2
    cp_size = 1

    def allreduce(self, value: int, op: ReduceOp | None = None) -> int:
        tensor = torch.tensor(value)
        torch_dist.all_reduce(tensor)
        return int(tensor.item())


def _run_sync_idle_progress_rank(rank: int, world_size: int, rendezvous_file: str) -> None:
    torch_dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=1),
    )
    try:
        if rank == 0:
            # Model a rank blocked in request_and_receive_sync(). It cannot
            # participate in an idle-progress collective on the peer rank.
            time.sleep(2)
            return

        executor = object.__new__(PyExecutor)
        executor.dist = _TorchCollectiveDist()
        executor._check_disagg_gen_cache_transfer_status = Mock()
        executor._check_disagg_ctx_cache_transfer_status = Mock()
        PyExecutor._check_disagg_transfer_progress_when_idle(executor)
    finally:
        torch_dist.destroy_process_group()


class _InflightRequestIds:
    def __init__(self):
        self.ids = set()

    def insert(self, request_id):
        self.ids.add(request_id)

    def erase(self, request_id):
        self.ids.discard(request_id)

    def __contains__(self, request_id):
        return request_id in self.ids


class MockPyExecutor:
    """A mock PyExecutor class for testing request handling logic.

    This mock contains only the attributes and methods needed to test
    the _handle_special_queue_items functionality.
    """

    def __init__(self, dist):
        self.dist = dist
        self.canceled_req_ids = []
        self.control_requests = []
        self.request_accumulated = []
        self.is_shutdown = False
        self.expected_num_active_requests = 0
        self.new_active_requests_queue_latency_ms = 0.0
        self.waiting_queue = FCFSWaitingQueue()

    def _handle_special_queue_items(self, new_requests):
        """Handle special signals.

        This method mirrors PyExecutor._handle_special_queue_items.
        """
        accepted_new_requests = []
        for idx, req_item in enumerate(new_requests):
            if req_item.is_shutdown_request:
                self.is_shutdown = True
                break
            elif req_item.is_canceled_request:
                self.canceled_req_ids.append(req_item.id)
            elif req_item.is_control_request:
                self.control_requests.append(req_item)
                if self.dist.rank == 0:
                    self.request_accumulated.extend(new_requests[idx + 1 :])
                break
            else:
                accepted_new_requests.append(req_item)

        return accepted_new_requests

    def update_waiting_queue(self):
        """Update waiting queue to remove canceled requests.

        This method mirrors PyExecutor._handle_canceled_requests.
        """
        if self.canceled_req_ids:
            canceled_set = set(self.canceled_req_ids)
            self.waiting_queue.remove_by_ids(canceled_set)

    def clear_canceled_req_ids(self):
        """Clear the list of canceled request IDs."""
        self.canceled_req_ids.clear()

    def get_canceled_req_ids(self):
        """Get the list of canceled request IDs."""
        return self.canceled_req_ids

    def get_canceled_req_ids_size(self):
        """Get the number of canceled request IDs."""
        return len(self.canceled_req_ids)

    def get_expected_num_active_requests(self):
        """Get the expected number of active requests."""
        return self.expected_num_active_requests

    def get_waiting_queue_size(self):
        """Get the size of the waiting queue."""
        return len(self.waiting_queue)

    def _get_new_active_requests_queue_latency(self):
        """Get the queue latency for new active requests."""
        return self.new_active_requests_queue_latency_ms


@pytest.fixture
def mock_dist():
    """Create a mock Distributed instance for testing."""
    mock_dist = Mock()
    mock_dist.rank = 0
    mock_dist.tp_size = 1
    return mock_dist


def _make_async_encoder_executor(future):
    executor = object.__new__(PyExecutor)
    executor.dist = types.SimpleNamespace(tp_size=1)
    executor.encoder_launch_executor = Mock()
    executor.encoder_launch_executor.submit.return_value = future
    executor.pending_encoder_steps = []
    executor.inflight_req_ids = _InflightRequestIds()
    executor._run_encoder_step_unchecked = Mock()
    executor._publish_encoder_step = Mock()
    executor._handle_errors = Mock()
    return executor


def _make_encoder_batch_wait_executor(batch_sizes=None, encoder_max_batch_size=8):
    """Build a PyExecutor stub wired for token-path encoder batch-wait admission."""
    executor = object.__new__(PyExecutor)
    executor.max_batch_size = 32
    batch_sizes = batch_sizes or [1, 2, 4, 8]
    executor.llm_args = types.SimpleNamespace(
        encoder_cuda_graph_config=types.SimpleNamespace(
            batch_sizes=batch_sizes,
            enable_padding=True,
            num_tokens=[96],
            seq_lens=[512],
        ),
        encoder_max_batch_size=encoder_max_batch_size,
    )
    executor.model_engine = types.SimpleNamespace(
        encoder_cuda_graph_runner=types.SimpleNamespace(
            feature_mode=False, enabled=True, supported_batch_sizes=batch_sizes
        )
    )
    executor.batch_wait_timeout_iters = 48
    executor.encoder_batch_wait_iters_count = 0
    return executor


def _make_feature_encoder_batch_wait_executor(
    runner_batch_sizes, encoder_max_batch_size=8, runner_enabled=True
):
    """Batch-wait executor whose encoder graph config is the feature variant.

    A feature encoder leaves `num_tokens` / `seq_lens` unset and may have had
    its `batch_sizes` derived rather than configured, so the resolved sizes come
    from the engine's encoder graph runner rather than the config.
    """
    executor = object.__new__(PyExecutor)
    executor.max_batch_size = 32
    executor.llm_args = types.SimpleNamespace(
        encoder_cuda_graph_config=EncodeCudaGraphConfig(enable_padding=True),
        encoder_max_batch_size=encoder_max_batch_size,
    )
    executor.model_engine = types.SimpleNamespace(
        encoder_cuda_graph_runner=types.SimpleNamespace(
            supported_batch_sizes=runner_batch_sizes,
            enabled=runner_enabled,
            feature_mode=True,
        )
    )
    executor.batch_wait_timeout_iters = 48
    executor.encoder_batch_wait_iters_count = 0
    return executor


def _make_encoder_fallback_batch_wait_executor():
    """Build a PyExecutor stub with no encoder graph config, for the fallback path."""
    executor = object.__new__(PyExecutor)
    executor.llm_args = types.SimpleNamespace(
        encoder_cuda_graph_config=None,
        encoder_max_batch_size=None,
    )
    executor.model_engine = types.SimpleNamespace(encoder_cuda_graph_runner=None)
    executor.batch_wait_timeout_iters = 48
    executor.encoder_batch_wait_iters_count = 0
    executor.batch_wait_max_tokens_ratio = 0.5
    executor.max_num_tokens = 32
    executor.active_requests = []
    executor.inflight_req_ids = _InflightRequestIds()
    return executor


def _make_encoder_request(request_id):
    return types.SimpleNamespace(
        request_id=request_id,
        state=LlmRequestState.ENCODER_INIT,
        encoder_output_len=4,
    )


def test_encoder_graph_warmup_uses_runtime_encoder_stream():
    executor = object.__new__(PyExecutor)
    executor.device_id = 3
    executor.encoder_stream = Mock()
    executor.resource_manager = object()
    executor.model_engine = Mock()
    stream_context = MagicMock()

    with (
        patch("torch.cuda.set_device") as set_device,
        patch("torch.cuda.stream", return_value=stream_context) as cuda_stream,
    ):
        executor._warmup_encoder_cuda_graphs_enc_dec()

    set_device.assert_called_once_with(3)
    cuda_stream.assert_called_once_with(executor.encoder_stream)
    executor.model_engine._warmup_encoder_cuda_graphs_enc_dec.assert_called_once_with(
        executor.resource_manager
    )


@pytest.mark.parametrize(
    "runner_batch_sizes,config_batch_sizes,num_requests,expected",
    [
        # A feature encoder leaves num_tokens / seq_lens unset, so gating this
        # path on them would skip microbatch admission entirely.
        ([1, 2, 4, 8], None, 12, 8),
        # The runner's list is authoritative: the engine filters the configured
        # sizes by the scheduler's encoder-batch bound, so the config alone can
        # name sizes that were never captured.
        ([1, 2, 3, 4], [1, 2, 3, 4, 8], 6, 4),
    ],
)
def test_encoder_microbatch_admission_uses_resolved_feature_batch_sizes(
    runner_batch_sizes, config_batch_sizes, num_requests, expected
):
    """Feature admission targets the runner's resolved sizes, not the configured ones."""
    executor = _make_feature_encoder_batch_wait_executor(runner_batch_sizes)
    if config_batch_sizes is not None:
        executor.llm_args.encoder_cuda_graph_config.batch_sizes = config_batch_sizes
    encoder_requests = [object() for _ in range(num_requests)]

    scheduled = executor._waiting_encoder_requests(
        encoder_requests,
        [],
        [object()] * 20,
    )

    assert scheduled == encoder_requests[:expected]
    assert executor.encoder_batch_wait_iters_count == 0


def test_encoder_microbatch_admission_skips_uncaptured_padded_size():
    """Feature admission never targets a batch size the runner did not capture."""
    # An `encoder_max_batch_size` above `max_batch_size` leaves a captured
    # bucket beyond the admission limit. Padding admission up to the limit
    # itself is a token-path move: a feature batch of 8 would have to pad to
    # the captured 16, which `pad_batch` refuses, so it must target 4 instead.
    executor = _make_feature_encoder_batch_wait_executor([1, 2, 4, 16], encoder_max_batch_size=16)
    executor.max_batch_size = 8
    encoder_requests = [object() for _ in range(12)]

    scheduled = executor._waiting_encoder_requests(encoder_requests, [], [object()] * 2)

    assert scheduled == encoder_requests[:4]
    assert executor.encoder_batch_wait_iters_count == 0


def test_encoder_microbatch_admission_ignores_disabled_feature_runner():
    """A runner that declined capture must not make admission wait on unreplayable shapes."""
    # supported_batch_sizes stays populated from the config even when capture
    # was declined (TP > 1, or no bucket fits), so waiting on those shapes
    # would stall a batch that can only ever run eager. With no decoder work
    # the request must be released immediately instead.
    executor = _make_feature_encoder_batch_wait_executor([1, 2, 4, 8], runner_enabled=False)
    executor.batch_wait_max_tokens_ratio = 0.5
    executor.max_num_tokens = 32
    executor.active_requests = []
    executor.inflight_req_ids = _InflightRequestIds()
    encoder_requests = [_make_encoder_request(0)]

    scheduled = executor._waiting_encoder_requests(encoder_requests, [], [])

    assert scheduled == encoder_requests
    assert executor.encoder_batch_wait_iters_count == 0


def test_encoder_microbatch_graph_admission_boundaries():
    executor = _make_encoder_batch_wait_executor()
    encoder_requests = [object()] * 7
    scheduled = executor._waiting_encoder_requests(
        encoder_requests,
        [],
        [object()] * 24,
    )
    assert scheduled == []
    assert executor.encoder_batch_wait_iters_count == 1

    executor = _make_encoder_batch_wait_executor()
    encoder_requests = [object() for _ in range(12)]
    scheduled = executor._waiting_encoder_requests(
        encoder_requests,
        [],
        [object()] * 20,
    )
    assert scheduled == encoder_requests[:8]
    assert executor.encoder_batch_wait_iters_count == 0

    executor = _make_encoder_batch_wait_executor(
        batch_sizes=[1, 3, 6],
        encoder_max_batch_size=8,
    )
    executor.encoder_batch_wait_iters_count = executor.batch_wait_timeout_iters
    encoder_requests = [object() for _ in range(5)]
    scheduled = executor._waiting_encoder_requests(
        encoder_requests,
        [],
        [],
    )
    assert scheduled == encoder_requests[:3]
    assert executor.encoder_batch_wait_iters_count == 0

    executor = _make_encoder_batch_wait_executor()
    executor.encoder_batch_wait_iters_count = executor.batch_wait_timeout_iters
    encoder_requests = [object() for _ in range(8)]
    scheduled = executor._waiting_encoder_requests(
        encoder_requests,
        [],
        [object() for _ in range(25)],
    )
    assert scheduled == []
    assert executor.encoder_batch_wait_iters_count == executor.batch_wait_timeout_iters + 1

    scheduled = executor._waiting_encoder_requests(
        encoder_requests,
        [],
        [object() for _ in range(24)],
    )

    assert scheduled == encoder_requests
    assert executor.encoder_batch_wait_iters_count == 0


def test_encoder_fallback_distinguishes_inflight_encoder_and_decoder_work():
    executor = _make_encoder_fallback_batch_wait_executor()
    encoder_requests = [_make_encoder_request(1)]
    inflight_encoder_request = _make_encoder_request(2)
    executor.active_requests.append(inflight_encoder_request)
    executor.inflight_req_ids.insert(inflight_encoder_request.request_id)

    scheduled = executor._waiting_encoder_requests(encoder_requests, [], [])
    assert scheduled == encoder_requests
    assert executor.encoder_batch_wait_iters_count == 0

    decoder_request = types.SimpleNamespace(
        request_id=3,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
    )
    executor.active_requests = [decoder_request]
    executor.inflight_req_ids.erase(inflight_encoder_request.request_id)
    executor.inflight_req_ids.insert(decoder_request.request_id)

    scheduled = executor._waiting_encoder_requests(encoder_requests, [], [])
    assert scheduled == []
    assert executor.encoder_batch_wait_iters_count == 1


def test_async_encoder_step_lifecycle():
    ready_event = Mock()
    ready_event.query.side_effect = [False, True]
    result = EncoderStepResult(
        hidden_states=torch.arange(12).reshape(6, 2),
        sequence_lengths=[2, 4],
        ready_event=ready_event,
    )
    future = Mock()
    future.done.side_effect = [False, True]
    future.result.return_value = result
    executor = _make_async_encoder_executor(future)
    active_request = types.SimpleNamespace(
        request_id=11,
        state=LlmRequestState.ENCODER_INIT,
    )
    completed_request = types.SimpleNamespace(
        request_id=12,
        state=LlmRequestState.GENERATION_COMPLETE,
    )
    requests = [active_request, completed_request]
    executor._publish_encoder_step.side_effect = (
        lambda encoder_requests, encoder_result: PyExecutor._publish_encoder_step(
            executor,
            encoder_requests,
            encoder_result,
        )
    )

    executor._submit_encoder_step(requests)
    executor._poll_encoder_steps()

    future.result.assert_not_called()
    executor._publish_encoder_step.assert_not_called()
    assert executor.inflight_req_ids.ids == {11, 12}
    assert len(executor.pending_encoder_steps) == 1
    executor.encoder_launch_executor.submit.assert_called_once_with(
        executor._run_encoder_step_unchecked,
        requests,
    )

    executor._poll_encoder_steps()
    future.result.assert_called_once_with()
    ready_event.query.assert_called_once_with()
    executor._publish_encoder_step.assert_not_called()
    assert executor.inflight_req_ids.ids == {11, 12}
    assert len(executor.pending_encoder_steps) == 1

    executor._poll_encoder_steps()
    future.result.assert_called_once_with()
    assert ready_event.query.call_count == 2
    executor._publish_encoder_step.assert_called_once_with(requests, result)
    assert executor.inflight_req_ids.ids == set()
    assert executor.pending_encoder_steps == []
    assert active_request.state == LlmRequestState.CONTEXT_INIT
    assert active_request.py_encoder_output_ready_event is ready_event
    assert torch.equal(active_request.py_encoder_output, result.hidden_states[:2])
    assert completed_request.state == LlmRequestState.GENERATION_COMPLETE
    assert not hasattr(completed_request, "py_encoder_output")

    executor.execution_stream = Mock()
    encoder_output = Mock()
    active_request.py_encoder_output = encoder_output
    scheduled_requests = types.SimpleNamespace(context_requests=[active_request])
    executor._attach_encoder_output_to_execution_stream(scheduled_requests)

    executor.execution_stream.wait_event.assert_not_called()
    encoder_output.record_stream.assert_called_once_with(executor.execution_stream)
    assert active_request.py_encoder_output_ready_event is None


def test_tp_encoder_step_synchronizes_and_publishes_inline():
    call_order = []
    execution_stream = Mock()
    encoder_stream = Mock()
    encoder_stream.wait_stream.side_effect = lambda stream: call_order.append("wait_stream")
    ready_event = Mock()
    ready_event.synchronize.side_effect = lambda: call_order.append("synchronize")
    result = EncoderStepResult(
        hidden_states=torch.empty((1, 2)),
        sequence_lengths=[1],
        ready_event=ready_event,
    )
    future = Mock()
    future.result.side_effect = lambda: (call_order.append("result"), result)[1]
    executor = _make_async_encoder_executor(future)
    executor.dist.tp_size = 2
    executor.execution_stream = execution_stream
    executor.encoder_stream = encoder_stream
    executor._publish_encoder_step.side_effect = lambda requests, encoder_result: call_order.append(
        "publish"
    )
    request = types.SimpleNamespace(request_id=13, state=LlmRequestState.ENCODER_INIT)

    executor._submit_encoder_step([request])

    assert call_order == ["wait_stream", "result", "synchronize", "publish"]
    encoder_stream.wait_stream.assert_called_once_with(execution_stream)
    assert executor.inflight_req_ids.ids == set()
    assert executor.pending_encoder_steps == []


@pytest.fixture
def mock_executor(mock_dist):
    """Create a MockPyExecutor instance for testing."""
    return MockPyExecutor(dist=mock_dist)


def test_handle_special_queue_items(mock_executor):
    """Test special queue item handling."""
    # Create a mock request
    mock_request = Mock()
    if hasattr(mock_request, "sampling_config"):
        delattr(mock_request, "sampling_config")

    normal_req = RequestQueueItem(1, mock_request)
    cancel_req = RequestQueueItem(2, is_canceled_request=True)
    shutdown_req = RequestQueueItem(SHUTDOWN_REQUEST_ID)

    requests = [normal_req, cancel_req, shutdown_req]

    valid_requests = mock_executor._handle_special_queue_items(requests)

    assert len(valid_requests) == 1
    assert valid_requests[0] == normal_req
    assert mock_executor.is_shutdown
    assert 2 in mock_executor.canceled_req_ids


def test_clear_canceled_req_ids(mock_executor):
    """Test clearing canceled request IDs."""
    mock_executor.canceled_req_ids = [1, 2, 3]
    assert len(mock_executor.canceled_req_ids) == 3

    mock_executor.clear_canceled_req_ids()

    assert len(mock_executor.canceled_req_ids) == 0


def test_update_waiting_queue(mock_executor):
    """Test updating waiting queue to remove canceled requests."""
    items = [
        RequestQueueItem(1, Mock()),
        RequestQueueItem(2, Mock()),
        RequestQueueItem(3, Mock()),
    ]
    mock_executor.waiting_queue.extend(items)
    mock_executor.canceled_req_ids = [2]

    mock_executor.update_waiting_queue()

    assert len(mock_executor.waiting_queue) == 2
    remaining_ids = [item.id for item in mock_executor.waiting_queue]
    assert 1 in remaining_ids
    assert 3 in remaining_ids
    assert 2 not in remaining_ids


def test_getter_methods(mock_executor):
    """Test various getter methods."""
    # Test initial values
    assert mock_executor._get_new_active_requests_queue_latency() == 0
    assert mock_executor.get_expected_num_active_requests() == 0
    assert mock_executor.get_canceled_req_ids_size() == 0
    assert mock_executor.get_canceled_req_ids() == []
    assert mock_executor.get_waiting_queue_size() == 0

    # Add some data and test
    mock_executor.canceled_req_ids = [3, 4]
    mock_executor.expected_num_active_requests = 5
    mock_executor.new_active_requests_queue_latency_ms = 10.5
    mock_executor.waiting_queue.append(RequestQueueItem(1, Mock()))

    assert mock_executor.get_canceled_req_ids_size() == 2
    assert mock_executor.get_canceled_req_ids() == [3, 4]
    assert mock_executor.get_expected_num_active_requests() == 5
    assert mock_executor._get_new_active_requests_queue_latency() == 10.5
    assert mock_executor.get_waiting_queue_size() == 1


def _make_executor_with_kv_cache_manager(kv_cache_manager):
    executor = PyExecutor.__new__(PyExecutor)
    executor.resource_manager = Mock()
    executor.resource_manager.resource_managers = {
        ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager
    }
    return executor


def test_get_kv_cache_capacity_without_manager():
    executor = _make_executor_with_kv_cache_manager(None)

    assert executor.get_kv_cache_capacity() == {}


def test_get_kv_cache_capacity_from_stats():
    """KV capacity is available without consuming iteration stats."""
    kv_stats = Mock()
    kv_stats.max_num_blocks = 123
    kv_stats.tokens_per_block = 64

    kv_cache_manager = Mock()
    kv_cache_manager.get_kv_cache_stats.return_value = kv_stats

    executor = _make_executor_with_kv_cache_manager(kv_cache_manager)

    assert executor.get_kv_cache_capacity() == {
        "maxNumBlocks": 123,
        "tokensPerBlock": 64,
        "maxNumTokens": 7872,
    }


def test_get_kv_cache_capacity_falls_back_to_manager_pool_size():
    """KVCacheManagerV2 exposes capacity through pool attributes."""
    kv_stats = Mock()
    kv_stats.max_num_blocks = 0
    kv_stats.tokens_per_block = 0

    kv_cache_manager = Mock()
    kv_cache_manager.get_kv_cache_stats.return_value = kv_stats
    kv_cache_manager.get_max_resource_count.return_value = 0
    kv_cache_manager.blocks_in_primary_pool = 256
    kv_cache_manager.tokens_per_block = 32

    executor = _make_executor_with_kv_cache_manager(kv_cache_manager)

    assert executor.get_kv_cache_capacity() == {
        "maxNumBlocks": 256,
        "tokensPerBlock": 32,
        "maxNumTokens": 8192,
    }


def test_get_kv_cache_capacity_falls_back_to_max_resource_count():
    kv_stats = Mock()
    kv_stats.max_num_blocks = 0
    kv_stats.tokens_per_block = 0

    kv_cache_manager = Mock()
    kv_cache_manager.get_kv_cache_stats.return_value = kv_stats
    kv_cache_manager.blocks_in_primary_pool = 0
    kv_cache_manager.get_max_resource_count.return_value = 512
    kv_cache_manager.tokens_per_block = 16

    executor = _make_executor_with_kv_cache_manager(kv_cache_manager)

    assert executor.get_kv_cache_capacity() == {
        "maxNumBlocks": 512,
        "tokensPerBlock": 16,
        "maxNumTokens": 8192,
    }


def _classify_termination(
    request, enable_partial_reuse_for_disagg, is_vswa, is_kv_manager_v2, pp_size=1
):
    """Reproduce the termination logic from _handle_responses (py_executor.py).

    Mirrors ``force_terminate_for_partial_reuse = force_terminate_ctx_for_partial_reuse``:
    the early-termination path is enabled only for partial-reuse disagg on the
    V1 KVCacheManager at PP=1. It is disabled for VSWA, KVCacheManagerV2 (no
    store_blocks_for_reuse equivalent), and PP>1 — where termination is routed
    through the DisaggPPTerminationHandler ring consensus via the
    transfer-complete path. (Eager block store stays enabled for PP>1, but it
    is a separate, rank-local concern that does not affect this branch.)

    Returns:
        "terminate" | "stats_only" | "skip"
    """
    force_terminate_for_partial_reuse = (
        enable_partial_reuse_for_disagg and not is_vswa and not is_kv_manager_v2 and pp_size == 1
    )
    if request.is_disagg_context_complete_state:
        return "stats_only"
    elif force_terminate_for_partial_reuse:
        return "terminate"
    elif not request.is_disagg_context_transmission_state:
        return "terminate"
    return "skip"


def _make_request(complete_state, transmission_state):
    req = Mock()
    req.is_disagg_context_complete_state = complete_state
    req.is_disagg_context_transmission_state = transmission_state
    return req


class TestDisaggTerminationGuard:
    """Verify _handle_responses does not double-terminate DISAGG_CONTEXT_COMPLETE
    requests that were already cleaned up by _check_disagg_ctx_cache_transfer_status
    (nvbug/5961736)."""

    def test_normal_path_skips_context_complete(self):
        """Without partial reuse, CONTEXT_COMPLETE goes to stats only."""
        req = _make_request(complete_state=True, transmission_state=False)
        assert _classify_termination(req, False, False, False) == "stats_only"

    def test_normal_path_skips_transmission_in_progress(self):
        """Without partial reuse, TRANS_IN_PROGRESS is skipped (still in flight)."""
        req = _make_request(complete_state=False, transmission_state=True)
        assert _classify_termination(req, False, False, False) == "skip"

    def test_normal_path_terminates_regular_request(self):
        """Without partial reuse, a normal finished request is terminated."""
        req = _make_request(complete_state=False, transmission_state=False)
        assert _classify_termination(req, False, False, False) == "terminate"

    def test_partial_reuse_terminates_non_complete(self):
        """With partial reuse, non-CONTEXT_COMPLETE requests are terminated."""
        for complete, transmission in [(False, True), (False, False)]:
            req = _make_request(complete, transmission)
            assert _classify_termination(req, True, False, False) == "terminate"

    def test_partial_reuse_early_terminate_is_pp1_only(self):
        """Early termination of an in-transmission ctx request is a PP=1-only
        optimization. Under PP>1 it is skipped here and terminated later via
        the transfer-complete path (ring consensus); eager store still applies."""
        req = _make_request(complete_state=False, transmission_state=True)
        assert _classify_termination(req, True, False, False, pp_size=1) == "terminate"
        req = _make_request(complete_state=False, transmission_state=True)
        assert _classify_termination(req, True, False, False, pp_size=4) == "skip"

    def test_partial_reuse_skips_context_complete(self):
        """With partial reuse, CONTEXT_COMPLETE still goes to stats only."""
        req = _make_request(complete_state=True, transmission_state=False)
        assert _classify_termination(req, True, False, False) == "stats_only"

    def test_partial_reuse_disabled_by_vswa(self):
        """VSWA disables partial reuse path, falling back to normal logic."""
        req = _make_request(complete_state=False, transmission_state=True)
        assert _classify_termination(req, True, True, False) == "skip"

    def test_partial_reuse_disabled_by_kv_manager_v2(self):
        """KVCacheManagerV2 disables the eager-store path (no
        store_blocks_for_reuse), falling back to normal logic."""
        req = _make_request(complete_state=False, transmission_state=True)
        assert _classify_termination(req, True, False, True) == "skip"

    def test_pp_gt_1_terminates_on_transfer_complete(self):
        """PP>1: the early path leaves the request out of requests_to_terminate
        AND out of new_active_requests, so it is removed from active_requests
        but retained by AsyncTransferManager. The real
        _end_transfer_and_maybe_terminate must then terminate it exactly once
        when the transfer completes (force_terminate_ctx_for_partial_reuse=False)."""
        req = Mock()
        executor = types.SimpleNamespace(
            kv_cache_transceiver=Mock(),
            active_requests=[],  # already removed by _handle_responses
            async_transfer_manager=Mock(),
            force_terminate_ctx_for_partial_reuse=False,
            _terminate_request=Mock(),
        )
        executor.async_transfer_manager.end_transfer.return_value = True

        PyExecutor._end_transfer_and_maybe_terminate(executor, req)

        executor._terminate_request.assert_called_once_with(req)

    def test_pp1_does_not_double_terminate_on_transfer_complete(self):
        """PP=1: the early path already terminated the request (and removed it
        from active_requests). The real _end_transfer_and_maybe_terminate must
        skip re-terminating it (force_terminate_ctx_for_partial_reuse=True) to
        avoid a double free_resources (nvbug/5961736)."""
        req = Mock()
        executor = types.SimpleNamespace(
            kv_cache_transceiver=Mock(),
            active_requests=[],  # already removed + terminated by early path
            async_transfer_manager=Mock(),
            force_terminate_ctx_for_partial_reuse=True,
            _terminate_request=Mock(),
        )
        executor.async_transfer_manager.end_transfer.return_value = True

        PyExecutor._end_transfer_and_maybe_terminate(executor, req)

        executor._terminate_request.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for _compute_scheduled_tokens with KV cache reuse chunk-shift logic
# ---------------------------------------------------------------------------


def _make_ctx_request(
    context_chunk_size,
    context_remaining_length,
    estimated_reusable_tokens=0,
    is_first_context_chunk=True,
    context_current_position=0,
):
    """Helper to create a mock context request for token computation tests."""
    req = Mock()
    req.context_chunk_size = context_chunk_size
    req.context_remaining_length = context_remaining_length
    req.estimated_reusable_tokens = estimated_reusable_tokens
    req.is_first_context_chunk = is_first_context_chunk
    req.context_current_position = context_current_position
    return req


def _make_gen_request(num_draft_tokens=0):
    """Helper to create a mock generation request."""
    req = Mock()
    req.num_draft_tokens = num_draft_tokens
    return req


def _make_disagg_transfer_request(
    request_id, prompt_len, in_progress=False, total_input_len_cp=None
):
    """Helper to create a mock disaggregated generation transfer request."""
    req = Mock()
    req.request_id = request_id
    req.py_request_id = request_id
    req.py_prompt_len = prompt_len
    req.total_input_len_cp = prompt_len if total_input_len_cp is None else total_input_len_cp
    req.is_disagg_generation_transmission_in_progress = in_progress
    return req


@pytest.fixture
def _clear_disagg_transfer_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)


@pytest.mark.usefixtures("_clear_disagg_transfer_mode_env")
class TestDisaggTransferAdmissionController:
    def test_disabled_preserves_candidates(self):
        controller = DisaggTransferAdmissionController(
            max_tokens_in_buffer=None, tokens_per_block=32
        )
        candidate = _make_disagg_transfer_request(1, 64)

        result = controller.select(active_requests=[], candidates=[candidate])

        assert result.admitted_requests == [candidate]
        assert result.deferred_request_count == 0
        assert not result.is_blocked_by_active_transfers()

    def test_fcfs_budget_counts_active_transfers(self):
        controller = DisaggTransferAdmissionController(max_tokens_in_buffer=64, tokens_per_block=32)
        active = _make_disagg_transfer_request(1, 32, in_progress=True)
        admitted = _make_disagg_transfer_request(2, 32)
        deferred = _make_disagg_transfer_request(3, 32)

        result = controller.select(active_requests=[active], candidates=[admitted, deferred])

        assert result.admitted_requests == [admitted]
        assert result.active_transfer_blocks == 1
        assert result.admitted_transfer_blocks == 1
        assert result.deferred_request_count == 1
        assert result.limited_by_budget
        assert not result.is_blocked_by_active_transfers()

    def test_reports_active_transfer_budget_block(self):
        controller = DisaggTransferAdmissionController(max_tokens_in_buffer=32, tokens_per_block=32)
        active = _make_disagg_transfer_request(1, 32, in_progress=True)
        candidate = _make_disagg_transfer_request(2, 32)

        result = controller.select(active_requests=[active], candidates=[candidate])

        assert result.admitted_requests == []
        assert result.active_transfer_blocks == 1
        assert result.deferred_request_count == 1
        assert result.is_blocked_by_active_transfers()

    def test_admits_oversized_head_when_idle(self):
        controller = DisaggTransferAdmissionController(max_tokens_in_buffer=32, tokens_per_block=32)
        oversized = _make_disagg_transfer_request(1, 96)
        deferred = _make_disagg_transfer_request(2, 32)

        result = controller.select(active_requests=[], candidates=[oversized, deferred])

        assert result.admitted_requests == [oversized]
        assert result.admitted_transfer_blocks == 3
        assert result.deferred_request_count == 1
        assert result.limited_by_budget
        assert not result.is_blocked_by_active_transfers()

    def test_uses_global_cp_prompt_length_for_transfer_cost(self):
        controller = DisaggTransferAdmissionController(
            max_tokens_in_buffer=128, tokens_per_block=32
        )
        request = _make_disagg_transfer_request(1, 32, total_input_len_cp=96)

        result = controller.select(active_requests=[], candidates=[request])

        assert result.admitted_requests == [request]
        assert result.admitted_transfer_blocks == 3

    def test_apply_reverts_deferred_v2_allocations(self):
        executor = object.__new__(PyExecutor)
        executor.kv_cache_transceiver = Mock()
        executor._is_kv_manager_v2 = True
        executor._revert_ctx_alloc = Mock()
        executor.active_requests = [_make_disagg_transfer_request(1, 32, in_progress=True)]
        executor._disagg_transfer_admission_controller = DisaggTransferAdmissionController(
            max_tokens_in_buffer=32, tokens_per_block=32
        )
        candidate = _make_disagg_transfer_request(2, 32)

        admitted, wait_for_progress = PyExecutor._apply_disagg_transfer_admission(
            executor, [candidate]
        )

        assert admitted == []
        assert wait_for_progress
        executor._revert_ctx_alloc.assert_called_once_with([candidate])

    def test_apply_missing_controller_preserves_candidates(self):
        executor = object.__new__(PyExecutor)
        executor.kv_cache_transceiver = Mock()
        executor.active_requests = []
        candidate = _make_disagg_transfer_request(1, 32)

        admitted, wait_for_progress = PyExecutor._apply_disagg_transfer_admission(
            executor, [candidate]
        )

        assert admitted == [candidate]
        assert not wait_for_progress

    def test_apply_missing_v2_flag_defaults_to_non_v2(self):
        executor = object.__new__(PyExecutor)
        executor.kv_cache_transceiver = Mock()
        executor._revert_ctx_alloc = Mock()
        executor.active_requests = [_make_disagg_transfer_request(1, 32, in_progress=True)]
        executor._disagg_transfer_admission_controller = DisaggTransferAdmissionController(
            max_tokens_in_buffer=32, tokens_per_block=32
        )
        candidate = _make_disagg_transfer_request(2, 32)

        admitted, wait_for_progress = PyExecutor._apply_disagg_transfer_admission(
            executor, [candidate]
        )

        assert admitted == []
        assert wait_for_progress
        executor._revert_ctx_alloc.assert_not_called()

    def test_sync_mode_retains_transfer_budget(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
        executor = object.__new__(PyExecutor)
        executor.kv_cache_transceiver = Mock()
        executor._is_kv_manager_v2 = True
        executor._revert_ctx_alloc = Mock()
        executor.active_requests = []
        executor._disagg_transfer_admission_controller = DisaggTransferAdmissionController(
            max_tokens_in_buffer=32, tokens_per_block=32
        )
        candidates = [
            _make_disagg_transfer_request(2, 32),
            _make_disagg_transfer_request(3, 32),
        ]

        admitted, wait_for_progress = PyExecutor._apply_disagg_transfer_admission(
            executor, candidates
        )

        assert admitted == [candidates[0]]
        assert not wait_for_progress
        executor._revert_ctx_alloc.assert_called_once_with([candidates[1]])

    def test_gen_only_no_context_bypasses_transfer_budget(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", "1")
        executor = object.__new__(PyExecutor)
        executor.kv_cache_transceiver = Mock()
        executor._is_kv_manager_v2 = True
        executor._revert_ctx_alloc = Mock()
        executor.active_requests = [_make_disagg_transfer_request(1, 32, in_progress=True)]
        executor._disagg_transfer_admission_controller = DisaggTransferAdmissionController(
            max_tokens_in_buffer=32, tokens_per_block=32
        )
        candidates = [
            _make_disagg_transfer_request(2, 32),
            _make_disagg_transfer_request(3, 32),
        ]

        admitted, wait_for_progress = PyExecutor._apply_disagg_transfer_admission(
            executor, candidates
        )

        assert admitted == candidates
        assert not wait_for_progress
        executor._revert_ctx_alloc.assert_not_called()


@pytest.mark.usefixtures("_clear_disagg_transfer_mode_env")
class TestDisaggTransferIdleProgress:
    def test_gen_transfer_status_polls_active_transfers(self):
        executor = object.__new__(PyExecutor)
        executor.active_requests = [_make_disagg_transfer_request(1, 32, in_progress=True)]
        executor._check_disagg_gen_cache_transfer_status = Mock()

        PyExecutor._check_disagg_gen_transfer_status(executor)

        executor._check_disagg_gen_cache_transfer_status.assert_called_once_with(0)

    def test_gen_transfer_status_enters_without_local_active_transfers(self):
        executor = object.__new__(PyExecutor)
        executor.active_requests = []
        executor._check_disagg_gen_cache_transfer_status = Mock()

        PyExecutor._check_disagg_gen_transfer_status(executor)

        executor._check_disagg_gen_cache_transfer_status.assert_called_once_with(0)

    def test_gen_transfer_status_skips_sync_mode(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
        executor = object.__new__(PyExecutor)
        executor._check_disagg_gen_cache_transfer_status = Mock()

        PyExecutor._check_disagg_gen_transfer_status(executor)

        executor._check_disagg_gen_cache_transfer_status.assert_not_called()

    def test_polls_context_transfers_without_blocking(self):
        executor = object.__new__(PyExecutor)
        executor.dist = Mock(tp_size=1)
        executor._check_disagg_gen_cache_transfer_status = Mock()
        executor._check_disagg_ctx_cache_transfer_status = Mock()

        PyExecutor._check_disagg_transfer_progress_when_idle(executor)

        executor._check_disagg_ctx_cache_transfer_status.assert_called_once_with(0)

    def test_does_not_repeat_gen_status_polled_by_loop_head(self):
        """The loop head already polls GEN status every iteration."""
        executor = object.__new__(PyExecutor)
        executor.dist = Mock(tp_size=1)
        executor._check_disagg_gen_cache_transfer_status = Mock()
        executor._check_disagg_ctx_cache_transfer_status = Mock()

        PyExecutor._check_disagg_transfer_progress_when_idle(executor)

        executor._check_disagg_gen_cache_transfer_status.assert_not_called()

    def test_idle_poll_enters_no_extra_collective(self):
        """The context poll is rank-symmetric, so no gating collective is needed."""
        executor = object.__new__(PyExecutor)
        executor.dist = Mock(tp_size=4, cp_size=4, world_size=16)
        executor._check_disagg_gen_cache_transfer_status = Mock()
        executor._check_disagg_ctx_cache_transfer_status = Mock()

        PyExecutor._check_disagg_transfer_progress_when_idle(executor)

        executor.dist.allreduce.assert_not_called()
        executor.dist.tp_allreduce.assert_not_called()
        executor.dist.tp_cp_allgather.assert_not_called()
        executor._check_disagg_ctx_cache_transfer_status.assert_called_once_with(0)

    def test_gen_only_no_context_benchmark_polls_context_when_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", "1")
        executor = object.__new__(PyExecutor)
        executor.dist = Mock(tp_size=4, cp_size=1, world_size=4)
        executor._check_disagg_gen_cache_transfer_status = Mock()
        executor._check_disagg_ctx_cache_transfer_status = Mock()

        PyExecutor._check_disagg_transfer_progress_when_idle(executor)

        executor.dist.allreduce.assert_not_called()
        executor.dist.tp_allreduce.assert_not_called()
        executor._check_disagg_ctx_cache_transfer_status.assert_called_once_with(0)

    def test_sync_transfer_skips_idle_progress_collectives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
        executor = object.__new__(PyExecutor)
        executor.dist = Mock(tp_size=4, cp_size=1, world_size=4)
        executor._check_disagg_gen_cache_transfer_status = Mock()
        executor._check_disagg_ctx_cache_transfer_status = Mock()

        PyExecutor._check_disagg_transfer_progress_when_idle(executor)

        executor.dist.allreduce.assert_not_called()
        executor.dist.tp_allreduce.assert_not_called()
        executor.dist.tp_cp_allgather.assert_not_called()
        executor._check_disagg_gen_cache_transfer_status.assert_not_called()
        executor._check_disagg_ctx_cache_transfer_status.assert_not_called()

    def test_sync_single_rank_ctx_reaps_idle_transfer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
        executor = object.__new__(PyExecutor)
        executor.dist = Mock(tp_size=1, cp_size=1, world_size=1)
        executor.async_transfer_manager = Mock()
        executor.async_transfer_manager.has_any_inflight_requests.return_value = True
        executor._check_disagg_gen_cache_transfer_status = Mock()
        executor._check_disagg_ctx_cache_transfer_status = Mock()

        PyExecutor._check_disagg_transfer_progress_when_idle(executor)

        executor.dist.allreduce.assert_not_called()
        executor.dist.tp_allreduce.assert_not_called()
        executor.dist.tp_cp_allgather.assert_not_called()
        executor._check_disagg_gen_cache_transfer_status.assert_not_called()
        executor._check_disagg_ctx_cache_transfer_status.assert_called_once_with(0)

    def test_sync_multi_rank_does_not_wait_for_blocked_peer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
        rendezvous_file = tmp_path / "sync-idle-progress-rendezvous"

        torch_mp.spawn(
            _run_sync_idle_progress_rank,
            args=(2, str(rendezvous_file)),
            nprocs=2,
            join=True,
        )

    def test_sync_receive_does_not_poll_async_status(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
        executor = object.__new__(PyExecutor)
        executor.kv_cache_transceiver = Mock()
        executor._check_disagg_gen_cache_transfer_status = Mock()
        executor._check_cache_transfer_errors = Mock()
        requests = [
            Mock(state=LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE),
            Mock(state=LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE),
        ]

        PyExecutor._recv_disagg_gen_cache(executor, requests)

        assert [
            call.args[0]
            for call in executor.kv_cache_transceiver.request_and_receive_sync.call_args_list
        ] == requests
        executor.kv_cache_transceiver.request_and_receive_async.assert_not_called()
        executor._check_disagg_gen_cache_transfer_status.assert_not_called()
        executor._check_cache_transfer_errors.assert_called_once_with("generation requests")
        assert executor._sync_disagg_transfer_made_progress

    def test_sync_receive_drains_batch_before_rank_aligned_error_vote(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
        executor = object.__new__(PyExecutor)
        executor.kv_cache_transceiver = Mock()
        executor.enable_attention_dp = True
        executor.dist = Mock(world_size=4, rank=0)
        local_vote = {"error_ids": [1], "blocked_ids": []}
        executor.dist.tp_allgather.return_value = [
            local_vote,
            {"error_ids": [], "blocked_ids": []},
            {"error_ids": [], "blocked_ids": []},
            {"error_ids": [], "blocked_ids": []},
        ]
        executor._handle_errors = Mock()
        executor._check_cache_transfer_errors = Mock()
        executor._sync_disagg_transfer_made_progress = False
        error_request = Mock(
            py_request_id=1,
            state=LlmRequestState.DISAGG_GENERATION_INIT,
            is_child=False,
        )
        following_request = Mock(
            py_request_id=2,
            state=LlmRequestState.DISAGG_GENERATION_INIT,
            is_child=False,
        )
        executor.active_requests = [error_request, following_request]

        def complete_or_error(req):
            req.state = (
                LlmRequestState.DISAGG_TRANS_ERROR
                if req is error_request
                else LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            )

        executor.kv_cache_transceiver.request_and_receive_sync.side_effect = complete_or_error

        PyExecutor._recv_disagg_gen_cache(executor, [error_request, following_request])

        assert [
            call.args[0]
            for call in executor.kv_cache_transceiver.request_and_receive_sync.call_args_list
        ] == [error_request, following_request]
        assert error_request.state == LlmRequestState.DISAGG_TRANS_ERROR
        assert following_request.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        executor.kv_cache_transceiver.cancel_request.assert_not_called()
        executor._handle_errors.assert_not_called()
        executor._check_cache_transfer_errors.assert_called_once_with("generation requests")
        assert executor._sync_disagg_transfer_made_progress

        PyExecutor._handle_disagg_cache_errors_synced(executor)

        executor.dist.tp_allgather.assert_called_once_with(local_vote)
        executor._handle_errors.assert_called_once_with(
            "Disagg KV cache transfer error",
            requests=[error_request],
            charge_budget=False,
        )


class TestIdleDisaggLoopPacing:
    """The idle poll no longer blocks, so the executor loops pace themselves.

    Pacing must cost nothing when a transfer is not what is holding the loop
    back, and the PP loop must not pace while the ring still has work.
    """

    @staticmethod
    def _make_request(*, init_state: bool = False, transfer_in_progress: bool = False) -> Mock:
        req = Mock()
        req.is_disagg_generation_init_state = init_state
        req.is_disagg_generation_transmission_in_progress = transfer_in_progress
        return req

    @pytest.mark.parametrize(
        "has_transceiver, ctx_inflight, request_kwargs, expect_sleep",
        [
            pytest.param(False, True, {"init_state": True}, False, id="not_disagg"),
            pytest.param(True, False, {}, False, id="nothing_pending"),
            pytest.param(True, True, {}, True, id="context_send_inflight"),
            pytest.param(True, False, {"init_state": True}, True, id="gen_awaiting_transfer"),
            pytest.param(
                True, False, {"transfer_in_progress": True}, True, id="gen_receive_inflight"
            ),
        ],
    )
    def test_paces_only_when_a_transfer_can_unblock_the_loop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        has_transceiver: bool,
        ctx_inflight: bool,
        request_kwargs: dict,
        expect_sleep: bool,
    ) -> None:
        sleep = Mock()
        monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.time.sleep", sleep)
        executor = object.__new__(PyExecutor)
        executor.kv_cache_transceiver = Mock() if has_transceiver else None
        executor.async_transfer_manager = Mock()
        executor.async_transfer_manager.has_any_inflight_requests.return_value = ctx_inflight
        executor.active_requests = [self._make_request(**request_kwargs)]

        PyExecutor._pace_idle_disagg_loop(executor)

        assert sleep.called is expect_sleep

    @pytest.mark.parametrize(
        "unhandled_batches, micro_batches, expected",
        [
            pytest.param(0, [None, None], True, id="ring_empty"),
            pytest.param(1, [None, None], False, id="batch_awaiting_handling"),
            pytest.param(0, [None, "batch"], False, id="batch_still_queued"),
        ],
    )
    def test_pp_ring_drained_only_when_no_microbatch_is_outstanding(
        self, unhandled_batches: int, micro_batches: list, expected: bool
    ) -> None:
        executor = object.__new__(PyExecutor)
        executor.unhandled_batch_counter = unhandled_batches
        executor.micro_batches = micro_batches

        assert PyExecutor._pp_ring_is_drained(executor) is expected


@pytest.mark.usefixtures("_clear_disagg_transfer_mode_env")
class TestDisaggTransferAdmissionPP:
    def test_pp_schedule_applies_gate_before_serializing(self):
        executor = object.__new__(PyExecutor)
        executor.dist = Mock(
            rank=0, is_first_pp_rank=True, is_last_pp_rank=True, tp_size=1, cp_size=1
        )
        executor.enable_attention_dp = False
        executor.kv_cache_transceiver = Mock()
        executor.active_requests = [_make_disagg_transfer_request(1, 32, in_progress=True)]
        executor._disagg_transfer_admission_controller = DisaggTransferAdmissionController(
            max_tokens_in_buffer=32, tokens_per_block=32
        )
        scheduled_batch = ScheduledRequests()
        candidate = _make_disagg_transfer_request(2, 32)
        executor._schedule = Mock(return_value=(scheduled_batch, [candidate], 0))

        scheduled, fitting, num_fitting, wait_for_progress = PyExecutor._pp_schedule_and_propagate(
            executor, microbatch_id=0
        )

        assert scheduled is scheduled_batch
        assert fitting == []
        assert num_fitting == 0
        assert wait_for_progress

    def test_pp_schedule_restores_propagated_gate_decision(self):
        executor = object.__new__(PyExecutor)
        executor.dist = Mock(
            rank=1,
            is_first_pp_rank=False,
            is_last_pp_rank=True,
            prev_pp_rank=0,
            tp_size=1,
            cp_size=1,
        )
        executor.enable_attention_dp = False
        executor.active_requests = [
            _make_disagg_transfer_request(1, 32, in_progress=True),
            _make_disagg_transfer_request(2, 32),
        ]
        serializable_schedule = SerializableSchedulerOutput(
            encoder_requests=[],
            context_requests_chunking=[],
            context_requests_last_chunk=[],
            generation_requests=[],
            paused_requests=[],
            fitting_disagg_gen_init_requests=[2],
            num_fitting_requests=0,
            wait_for_disagg_gen_transfer_progress=True,
        )
        executor.dist.recv_object = Mock(return_value=serializable_schedule)

        _, fitting, _, wait_for_progress = PyExecutor._pp_schedule_and_propagate(
            executor, microbatch_id=0
        )

        assert [req.py_request_id for req in fitting] == [2]
        assert wait_for_progress


def test_nonzero_pp_rank_prepares_snapshot_points_before_local_schedule(
    monkeypatch,
):
    class StopLocalSchedule(RuntimeError):
        pass

    executor = object.__new__(PyExecutor)
    executor.dist = Mock(pp_rank=1, rank=1)
    executor.device_id = 0
    profiler = MagicMock()
    profiler.__enter__.return_value = Mock()
    executor._profiler = Mock(return_value=profiler)
    executor.hang_detector = MagicMock()
    executor.enable_iter_perf_stats = False
    executor._handle_disagg_cache_errors_synced = Mock()
    executor._fetch_and_activate_new_requests = Mock(return_value=[])
    executor.is_shutdown = False
    executor._handle_control_request = Mock()
    executor.kv_cache_transceiver = None
    executor._pad_attention_dp_dummy_request = Mock()
    scheduled_batch = Mock()
    executor._pp_schedule_and_propagate = Mock(return_value=(scheduled_batch, [], 0, False))
    executor._pp_retry_until_can_schedule = Mock()
    request = Mock()
    executor.active_requests = [request]
    executor.inflight_req_ids = set()
    executor.kv_cache_manager = Mock()
    executor.scheduler = Mock()

    calls = []
    executor.kv_cache_manager.prepare_expect_snapshot_points.side_effect = (
        lambda requests: calls.append(("prepare", requests))
    )

    def stop_after_schedule(requests, inflight_req_ids):
        calls.append(("schedule", requests, inflight_req_ids))
        raise StopLocalSchedule

    executor.scheduler.schedule_request.side_effect = stop_after_schedule

    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.set_device", Mock())
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.cudart.cudaSetDevice", Mock())
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.CUASSERT", Mock())

    with pytest.raises(StopLocalSchedule):
        PyExecutor._executor_loop_pp(executor)

    assert calls == [
        ("prepare", executor.active_requests),
        ("schedule", executor.active_requests, executor.inflight_req_ids),
    ]


def test_schedule_prepares_snapshot_points_before_scheduling():
    class StopSchedule(RuntimeError):
        pass

    executor = object.__new__(PyExecutor)
    request = Mock()
    executor.active_requests = [request]
    executor.inflight_req_ids = set()
    executor.kv_cache_manager = Mock()
    executor.scheduler = Mock()

    calls = []
    executor.kv_cache_manager.prepare_expect_snapshot_points.side_effect = (
        lambda requests: calls.append(("prepare", requests))
    )

    def stop_after_schedule(requests, inflight_req_ids):
        calls.append(("schedule", requests, inflight_req_ids))
        raise StopSchedule

    executor.scheduler.schedule_request.side_effect = stop_after_schedule

    with pytest.raises(StopSchedule):
        PyExecutor._schedule(executor)

    assert calls == [
        ("prepare", executor.active_requests),
        ("schedule", executor.active_requests, executor.inflight_req_ids),
    ]


class TestComputeScheduledTokens:
    """Tests for PyExecutor._compute_scheduled_tokens.

    Validates the chunk-shift aware token accounting: setPrepopulatedPromptLen
    shifts the chunk window right by the reused amount rather than shrinking it.
    Non-last chunks cost chunkSize; only last chunks cost remaining - reusable.
    """

    def test_no_reuse(self):
        """Without reuse, compute = chunk_size."""
        ctx = [_make_ctx_request(context_chunk_size=100, context_remaining_length=100)]
        assert PyExecutor._compute_scheduled_tokens(ctx, []) == 100

    def test_last_chunk_with_reuse(self):
        """Last chunk (reusable + chunk >= remaining): compute = chunk - reusable."""
        # promptLen=100, reusable=60, chunk=100 (full context)
        # 60 + 100 >= 100 → last chunk → compute = max(1, 100 - 60) = 40
        ctx = [
            _make_ctx_request(
                context_chunk_size=100, context_remaining_length=100, estimated_reusable_tokens=60
            )
        ]
        assert PyExecutor._compute_scheduled_tokens(ctx, []) == 40

    def test_non_last_chunk_with_reuse(self):
        """Non-last chunk (reusable + chunk < remaining): compute = chunk_size.

        This is the core chunk-shift scenario. The old formula would compute
        max(0, 25 - 30) = 0, but the correct cost is 25 because the chunk
        window shifts right rather than shrinking.
        """
        # promptLen=100, reusable=30, chunk=25
        # 30 + 25 = 55 < 100 → non-last chunk → compute = 25
        ctx = [
            _make_ctx_request(
                context_chunk_size=25, context_remaining_length=100, estimated_reusable_tokens=30
            )
        ]
        assert PyExecutor._compute_scheduled_tokens(ctx, []) == 25

    def test_non_first_chunk_ignores_reuse(self):
        """Reusable tokens only apply to the first context chunk."""
        ctx = [
            _make_ctx_request(
                context_chunk_size=50,
                context_remaining_length=50,
                estimated_reusable_tokens=30,
                is_first_context_chunk=False,
            )
        ]
        assert PyExecutor._compute_scheduled_tokens(ctx, []) == 50

    def test_v2_scheduler_position_advanced(self):
        """V2 scheduler: context_current_position already advanced past reuse.

        reusable_in_chunk = max(0, 30 - 30) = 0 → no credit → compute = chunk.
        """
        ctx = [
            _make_ctx_request(
                context_chunk_size=50,
                context_remaining_length=70,
                estimated_reusable_tokens=30,
                context_current_position=30,
            )
        ]
        assert PyExecutor._compute_scheduled_tokens(ctx, []) == 50

    def test_min_compute_is_one(self):
        """Compute cost is floored at 1 even when reusable >= chunk_size."""
        # chunk=10, remaining=10, reusable=15 → last chunk → max(1, 10-15) = 1
        ctx = [
            _make_ctx_request(
                context_chunk_size=10, context_remaining_length=10, estimated_reusable_tokens=15
            )
        ]
        assert PyExecutor._compute_scheduled_tokens(ctx, []) == 1

    def test_generation_tokens(self):
        """Generation requests contribute 1 + num_draft_tokens each."""
        gen = [_make_gen_request(3), _make_gen_request(0)]
        assert PyExecutor._compute_scheduled_tokens([], gen) == (1 + 3) + (1 + 0)

    def test_mixed_context_and_generation(self):
        """Combined context (with chunk-shift) and generation tokens."""
        # Non-last chunk: compute = 25
        ctx = [
            _make_ctx_request(
                context_chunk_size=25, context_remaining_length=100, estimated_reusable_tokens=30
            )
        ]
        gen = [_make_gen_request(2)]
        # 25 ctx + (1 + 2) gen = 28
        assert PyExecutor._compute_scheduled_tokens(ctx, gen) == 28

    def test_multiple_ctx_requests_mixed_chunks(self):
        """Multiple context requests: one non-last chunk, one last chunk."""
        # req0: non-last chunk → compute = 20
        req0 = _make_ctx_request(
            context_chunk_size=20, context_remaining_length=100, estimated_reusable_tokens=30
        )
        # req1: last chunk (reuse=10, chunk=50, remaining=50) → 10+50>=50
        # → compute = max(1, 50-10) = 40
        req1 = _make_ctx_request(
            context_chunk_size=50, context_remaining_length=50, estimated_reusable_tokens=10
        )
        assert PyExecutor._compute_scheduled_tokens([req0, req1], []) == 20 + 40


# ---------------------------------------------------------------------------
# Tests for event-loop crash propagation to _await_single_response callers.
#
# nvbug 6038228: when PyExecutor._event_loop_wrapper crashed (e.g. KV cache
# OOM), the main thread parked in _await_single_response would block forever
# because is_shutdown was never set / observed by the wait predicate. The fix
# stashes the original error in self._event_loop_error, sets is_shutdown +
# notifies in _executor_loop_cleanup, and re-raises the error from
# _await_single_response so callers exit promptly with a meaningful message.
#
# We exercise the actual PyExecutor methods by binding them to a lightweight
# stub that carries only the attributes those methods touch.
# ---------------------------------------------------------------------------


class _ResponseStub:
    """Minimal stub carrying only the state used by _await_single_response."""

    def __init__(self):
        self.response_lock = threading.Lock()
        self.response_cv = threading.Condition(self.response_lock)
        self.responses = {}
        self.is_shutdown = False
        self._event_loop_error = None
        # Set when the stashed error is handed to a caller: that is what tells
        # the rank-crash kill the crash was already reported and the world does
        # not need tearing down.
        self._event_loop_error_delivered = threading.Event()

    # Bind the real production method so the test exercises real code.
    _await_single_response = PyExecutor._await_single_response


class TestAwaitSingleResponseShutdown:
    """_await_single_response must not block forever when the event loop dies."""

    def test_returns_response_when_available(self):
        """Normal path: response exists, returned and consumed."""
        stub = _ResponseStub()
        stub.responses = {7: ["resp_a", "resp_b"]}

        result = stub._await_single_response(id=7, timeout=1.0)
        assert result == ["resp_a", "resp_b"]
        assert 7 not in stub.responses

    def test_returns_response_even_during_shutdown(self):
        """If a response was enqueued before shutdown it is still returned;
        the shutdown branch only fires when nothing is queued for this id."""
        stub = _ResponseStub()
        stub.is_shutdown = True
        stub._event_loop_error = RuntimeError("crash")
        stub.responses = {7: ["resp"]}

        result = stub._await_single_response(id=7, timeout=1.0)
        assert result == ["resp"]

    def test_raises_on_shutdown_with_event_loop_error(self):
        """When the event loop crashed, _await_single_response surfaces the
        original error as RuntimeError instead of hanging."""
        stub = _ResponseStub()
        stub.is_shutdown = True
        stub._event_loop_error = RuntimeError("KV cache OOM")

        with pytest.raises(RuntimeError, match="Event loop terminated"):
            stub._await_single_response(id=42, timeout=1.0)

        # The caller now holds the original error, so the rank-crash kill must
        # stand down: this is the signal it waits out its grace for.
        assert stub._event_loop_error_delivered.is_set()

    def test_raises_on_shutdown_without_event_loop_error(self):
        """Shutdown without a stored error still raises rather than blocking
        — distinguishes "shutdown" from "timed out without shutdown"."""
        stub = _ResponseStub()
        stub.is_shutdown = True

        with pytest.raises(RuntimeError, match="Event loop shut down"):
            stub._await_single_response(id=42, timeout=1.0)

        # Nothing was delivered -- there was no error to deliver. Leaving the
        # gate clear keeps the kill armed, which is correct here.
        assert not stub._event_loop_error_delivered.is_set()

    def test_returns_empty_on_timeout(self):
        """Pre-fix behaviour: a bare timeout (no shutdown, no response) used
        to KeyError. The fix returns an empty list to match the documented
        timeout contract used elsewhere in the executor API."""
        stub = _ResponseStub()
        result = stub._await_single_response(id=99, timeout=0.01)
        assert result == []

    def test_wakes_up_when_shutdown_set_from_another_thread(self):
        """Real-world scenario: main thread is parked in
        _await_single_response while the event-loop thread crashes and
        triggers _executor_loop_cleanup, which sets is_shutdown + notifies.
        The waiter must wake and re-raise."""
        stub = _ResponseStub()
        original_error = RuntimeError("simulated event-loop crash")

        def crash_after_delay():
            time.sleep(0.05)
            stub._event_loop_error = original_error
            with stub.response_cv:
                stub.is_shutdown = True
                stub.response_cv.notify_all()

        crash_thread = threading.Thread(target=crash_after_delay, daemon=True)
        crash_thread.start()

        with pytest.raises(RuntimeError, match="Event loop terminated"):
            stub._await_single_response(id=1, timeout=5.0)

        crash_thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Tests for _executor_loop_cleanup ordering (notify before PP wait).
# ---------------------------------------------------------------------------


class _CleanupStub:
    """Stub for _executor_loop_cleanup: records the order in which the
    shutdown notification and PP-handle wait happen."""

    def __init__(self, pp_handles_raise=False):
        self.response_lock = threading.Lock()
        self.response_cv = threading.Condition(self.response_lock)
        self.is_shutdown = False
        self.shutdown_event = threading.Event()
        self.num_micro_batches = 1
        self.send_handles = {}
        self.send_schedule_handles = {}
        self.send_expected_batch_num_handles = {}
        self._pp_handles_raise = pp_handles_raise
        self._events: list = []

        original_notify = self.response_cv.notify_all

        def record_notify():
            self._events.append("notify_all")
            original_notify()

        self.response_cv.notify_all = record_notify

    def wait_on_pp_send_handles(self, handles, idx):
        self._events.append(f"wait_pp_{idx}")
        if self._pp_handles_raise:
            raise RuntimeError("PP send handle in bad state")

    _executor_loop_cleanup = PyExecutor._executor_loop_cleanup


class TestExecutorLoopCleanup:
    """Cleanup must wake waiters BEFORE doing potentially-blocking PP work,
    and a PP-handle exception must not skip the shutdown notification."""

    def test_notify_happens_before_pp_wait(self):
        stub = _CleanupStub()
        stub._executor_loop_cleanup()

        assert stub._events[0] == "notify_all"
        assert "wait_pp_0" in stub._events
        assert stub._events.index("notify_all") < stub._events.index("wait_pp_0")
        assert stub.is_shutdown is True
        assert stub.shutdown_event.is_set()

    def test_pp_wait_exception_does_not_skip_notify(self):
        """If wait_on_pp_send_handles raises, the shutdown notification
        must still have happened (it ran first), and cleanup must not
        propagate the error so the executor thread terminates cleanly."""
        stub = _CleanupStub(pp_handles_raise=True)
        stub._executor_loop_cleanup()

        assert stub.is_shutdown is True
        assert "notify_all" in stub._events


_STATE_GENERATION_IN_PROGRESS = LlmRequestState.GENERATION_IN_PROGRESS
_STATE_GENERATION_TO_COMPLETE = LlmRequestState.GENERATION_TO_COMPLETE
_STATE_DISAGG_GENERATION_INIT = "_disagg_init_sentinel"
_STATE_DISAGG_GENERATION_TRANS_IN_PROGRESS = "_disagg_trans_sentinel"

# The sentinel mocks must expose the real state ints: the pad predicate
# compares state_value against the scheduler window bounds.
_SENTINEL_STATE_VALUES = {
    _STATE_DISAGG_GENERATION_INIT: LlmRequestState.DISAGG_GENERATION_INIT.value,
    _STATE_DISAGG_GENERATION_TRANS_IN_PROGRESS: LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS.value,
}


def _make_adp_request(
    state,
    *,
    request_id=1,
    llm_request_type=None,
    is_dummy_request=False,
    is_child=False,
    parent_request_id=None,
):
    req = Mock()
    req.state = state
    req.state_value = (
        state.value
        if isinstance(state, LlmRequestState)
        else _SENTINEL_STATE_VALUES.get(state, LlmRequestState.GENERATION_IN_PROGRESS.value)
    )
    req.py_request_id = request_id
    req.is_child = is_child
    req.parent_request_id = parent_request_id
    req.is_disagg_generation_init_state = state == _STATE_DISAGG_GENERATION_INIT
    req.is_disagg_generation_transmission_in_progress = (
        state == _STATE_DISAGG_GENERATION_TRANS_IN_PROGRESS
    )
    req.is_dummy_request = is_dummy_request
    req.is_attention_dp_dummy = False
    req.py_skip_gen_alloc_revert = False
    req.llm_request_type = llm_request_type
    req.py_seq_slot = None
    req.is_encoder_init_state = state == LlmRequestState.ENCODER_INIT
    req.is_context_init_state = state == LlmRequestState.CONTEXT_INIT
    req.is_generation_in_progress_state = state == _STATE_GENERATION_IN_PROGRESS
    req.py_encoder_output_ready_event = None
    return req


class _StubADPExecutor:
    def __init__(
        self,
        *,
        enable_attention_dp=True,
        kv_cache_transceiver=object(),
        max_num_tokens=8192,
        max_seq_len=8192,
        kv_manager_max_seq_len=None,
        is_warmup=False,
        benchmark_req_queues_size=0,
        enable_adp_dummy_fixes=True,
        enable_scheduler_aware_adp_dummy=None,
        enable_non_overlap_adp_forward_intent=None,
        peer_forward_intent=_ADPForwardIntent.GENERATION,
    ):
        self.enable_attention_dp = enable_attention_dp
        self.kv_cache_transceiver = kv_cache_transceiver
        self.is_warmup = is_warmup
        self.benchmark_req_queues_size = benchmark_req_queues_size
        self._benchmark_fill_phase_active = False
        self.num_fetch_requests = 0
        self.active_requests = []
        self.expected_num_active_requests = 1
        self.max_num_active_requests = 8
        self.max_total_draft_tokens = 0
        self.max_num_tokens = max_num_tokens
        self._adp_dummy_is_gen = True
        self._pending_adp_dummy_request = None
        self._enable_adp_dummy_fixes = enable_adp_dummy_fixes
        self._enable_scheduler_aware_adp_dummy = (
            enable_adp_dummy_fixes
            if enable_scheduler_aware_adp_dummy is None
            else enable_scheduler_aware_adp_dummy
        )
        self._enable_non_overlap_adp_forward_intent = (
            enable_adp_dummy_fixes
            if enable_non_overlap_adp_forward_intent is None
            else enable_non_overlap_adp_forward_intent
        )
        self.add_dummy_calls = []
        self.model_engine = Mock(max_num_tokens=max_num_tokens, max_seq_len=max_seq_len)

        self.dist = Mock()
        self.dist.tp_size = 1
        self.dist.tp_allgather.side_effect = lambda value: [value]
        self.dist.tp_allreduce.side_effect = lambda value, op: max(value, int(peer_forward_intent))

        self.scheduler = Mock()
        self.scheduler.scheduling_state_range = (
            LlmRequestState.CONTEXT_INIT,
            LlmRequestState.GENERATION_TO_COMPLETE,
        )
        self.scheduler.is_request_in_schedulable_state.side_effect = (
            lambda request: RequestScheduler.is_request_in_schedulable_state(
                self.scheduler, request
            )
        )

        kv_cache_manager = Mock()
        kv_cache_manager.mapping.has_cp_helix.return_value = False
        kv_cache_manager.get_num_available_tokens.return_value = 1 << 30
        kv_cache_manager.is_linear_attention = False
        kv_cache_manager.max_seq_len = (
            max_seq_len if kv_manager_max_seq_len is None else kv_manager_max_seq_len
        )
        kv_cache_manager.num_extra_kv_tokens = 0
        kv_cache_manager.tokens_per_block = 128

        def _add_dummy(**kwargs):
            self.add_dummy_calls.append(kwargs)
            state = (
                _STATE_GENERATION_IN_PROGRESS if kwargs["is_gen"] else LlmRequestState.CONTEXT_INIT
            )
            req = _make_adp_request(
                state,
                request_id=kwargs["request_ids"][0],
                is_dummy_request=True,
            )
            return [req]

        kv_cache_manager.add_dummy_requests.side_effect = _add_dummy
        self.kv_cache_manager = kv_cache_manager

        self.resource_manager = Mock()
        self.resource_manager.get_resource_manager.return_value = None


def _run_pad(stub):
    for helper in (
        "_count_schedulable_active_requests",
        "_get_non_overlap_adp_forward_intent",
        "_has_adp_dummy_kv_capacity",
        "_should_skip_dummy_for_benchmark_disagg",
    ):
        setattr(stub, helper, types.MethodType(getattr(PyExecutor, helper), stub))
    PyExecutor._pad_attention_dp_dummy_request(stub)


def _run_update_role(stub, candidates):
    PyExecutor._update_adp_dummy_role(stub, candidates)


def test_adp_dummy_role_set_to_ctx_on_context_only_request():
    from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

    stub = _StubADPExecutor()
    req = _make_adp_request(
        _STATE_GENERATION_IN_PROGRESS,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    _run_update_role(stub, [req])

    assert stub._adp_dummy_is_gen is False


def test_adp_dummy_role_set_to_gen_on_generation_only_request():
    from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

    stub = _StubADPExecutor()
    stub._adp_dummy_is_gen = False
    req = _make_adp_request(
        _STATE_GENERATION_IN_PROGRESS,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
    )
    _run_update_role(stub, [req])

    assert stub._adp_dummy_is_gen is True


def test_adp_dummy_role_flips_when_request_type_changes():
    from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

    stub = _StubADPExecutor()
    ctx_req = _make_adp_request(
        _STATE_GENERATION_IN_PROGRESS,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    _run_update_role(stub, [ctx_req])
    assert stub._adp_dummy_is_gen is False

    gen_req = _make_adp_request(
        _STATE_GENERATION_IN_PROGRESS,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
    )
    _run_update_role(stub, [gen_req])
    assert stub._adp_dummy_is_gen is True


def test_adp_dummy_role_unchanged_for_non_disagg_worker():
    from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

    stub = _StubADPExecutor(kv_cache_transceiver=None)
    req = _make_adp_request(
        _STATE_GENERATION_IN_PROGRESS,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    _run_update_role(stub, [req])

    assert stub._adp_dummy_is_gen is True


def test_adp_dummy_role_unchanged_when_attention_dp_disabled():
    from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

    stub = _StubADPExecutor(enable_attention_dp=False)
    req = _make_adp_request(
        _STATE_GENERATION_IN_PROGRESS,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    _run_update_role(stub, [req])

    assert stub._adp_dummy_is_gen is True


@pytest.mark.parametrize(
    "state",
    [
        _STATE_GENERATION_TO_COMPLETE,
        LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER,
    ],
)
def test_disabled_adp_dummy_fix_gate_preserves_pp_behavior(state):
    # PP configurations remain on the established dummy path.
    stub = _StubADPExecutor(enable_adp_dummy_fixes=False)
    stub.active_requests = [_make_adp_request(state)]
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert stub.add_dummy_calls == []
    assert len(stub.active_requests) == 1


def test_pad_dummy_added_when_only_to_complete_requests_disagg():
    # In disaggregated mode a GENERATION_TO_COMPLETE request is refused by
    # MicroBatchScheduler (no_schedule_after_state). When a peer has real
    # generation work, a rank holding only terminal requests must receive a
    # pad dummy or can_queue goes False fleet-wide.
    stub = _StubADPExecutor()
    stub.active_requests = [_make_adp_request(_STATE_GENERATION_TO_COMPLETE)]
    stub.expected_num_active_requests = 2

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    assert len(stub.active_requests) == 2


def test_pad_dummy_tolerates_active_request_overshoot():
    # A transient overshoot (len(active_requests) > expected_num_active_requests,
    # when disagg transfer-error requests linger a tick before cleanup) used to
    # trip a hard assert that crashed the gen loop on every ADP rank. It must now
    # warn and continue instead of raising.
    stub = _StubADPExecutor()
    stub.active_requests = [
        _make_adp_request(_STATE_GENERATION_IN_PROGRESS),
        _make_adp_request(_STATE_GENERATION_IN_PROGRESS),
    ]
    stub.expected_num_active_requests = 1  # < len(active_requests) == 2

    # Must not raise AssertionError (the pre-fix behavior on overshoot).
    _run_pad(stub)

    # Both requests are schedulable, so no dummy is added; pin it so the test
    # cannot pass on an early return or a stray pad.
    assert stub.add_dummy_calls == []
    assert len(stub.active_requests) == 2


def test_pad_dummy_added_when_overshoot_has_no_schedulable_requests():
    # The branch that matters: overshoot AND nothing schedulable (all at
    # GENERATION_TO_COMPLETE) must still pad, or can_queue goes False
    # fleet-wide.
    stub = _StubADPExecutor()
    stub.active_requests = [
        _make_adp_request(_STATE_GENERATION_TO_COMPLETE, request_id=1),
        _make_adp_request(_STATE_GENERATION_TO_COMPLETE, request_id=2),
    ]
    stub.expected_num_active_requests = 1  # < len(active_requests) == 2

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    assert len(stub.active_requests) == 3


def test_pad_dummy_added_when_only_wait_scheduler_requests_disagg():
    # Gen-first mode on the context server: DISAGG_CONTEXT_WAIT_SCHEDULER
    # sits BELOW the scheduler's window [CONTEXT_INIT, GENERATION_TO_COMPLETE)
    # (no_schedule_until_state), so a rank holding only such requests
    # schedules batch=0. A peer's generation intent therefore requires a pad
    # dummy — the left-boundary mirror of the TO_COMPLETE case above.
    stub = _StubADPExecutor()
    stub.active_requests = [_make_adp_request(LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER)]
    stub.expected_num_active_requests = 2

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    assert len(stub.active_requests) == 2


def test_pad_dummy_tolerates_surplus_over_expected_on_busy_rank() -> None:
    # expected_num_active_requests is capped at max_num_active_requests after
    # the per-rank-load floor is applied, so a rank can legitimately end up
    # holding more requests than the router expected -- e.g. when a pad dummy
    # survives an iteration that was skipped fleet-wide and the next
    # gather_all_rank_states counts it. This used to trip a bare assert and
    # kill the executor event loop; a busy rank needs no dummy, so it must be
    # tolerated instead.
    stub = _StubADPExecutor()
    stub.active_requests = [_make_adp_request(_STATE_GENERATION_IN_PROGRESS) for _ in range(3)]
    stub.expected_num_active_requests = 2

    _run_pad(stub)

    assert stub.add_dummy_calls == []
    assert len(stub.active_requests) == 3
    # Tolerating must not leak a mutated expectation to downstream consumers.
    assert stub.expected_num_active_requests == 2


def test_pad_dummy_still_added_when_surplus_requests_are_unschedulable() -> None:
    # Tolerating the surplus must not short-circuit padding. A rank can hold
    # more requests than expected AND have none of them schedulable (all parked
    # at GENERATION_TO_COMPLETE), in which case it still schedules batch=0 and
    # needs a dummy to stay in the forward-pass collectives.
    stub = _StubADPExecutor()
    stub.active_requests = [_make_adp_request(_STATE_GENERATION_TO_COMPLETE) for _ in range(3)]
    stub.expected_num_active_requests = 2

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    assert stub.expected_num_active_requests == 2


def test_encoder_init_uses_encoder_decoder_scheduler_state_window():
    stub = _StubADPExecutor()
    stub.scheduler.scheduling_state_range = (
        LlmRequestState.ENCODER_INIT,
        LlmRequestState.GENERATION_TO_COMPLETE,
    )
    stub.active_requests = [_make_adp_request(LlmRequestState.ENCODER_INIT)]
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert stub.add_dummy_calls == []
    assert len(stub.active_requests) == 1


@pytest.mark.parametrize(
    "state",
    [
        LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER,
        LlmRequestState.DISAGG_GENERATION_INIT,
        LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
    ],
)
def test_encoder_decoder_disagg_wait_and_transfer_states_are_not_schedulable(state):
    stub = _StubADPExecutor()
    stub.scheduler.scheduling_state_range = (
        LlmRequestState.ENCODER_INIT,
        LlmRequestState.GENERATION_TO_COMPLETE,
    )
    stub.active_requests = [_make_adp_request(state)]
    stub.expected_num_active_requests = 2

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    assert len(stub.active_requests) == 2


def test_decoder_context_waiting_for_encoder_output_is_not_counted():
    stub = _StubADPExecutor()
    request = _make_adp_request(LlmRequestState.CONTEXT_INIT)
    request.py_encoder_output_ready_event = Mock()
    request.py_encoder_output_ready_event.query.return_value = False
    stub.active_requests = [request]
    stub.expected_num_active_requests = 2

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    assert len(stub.active_requests) == 2


def test_generic_disagg_adp_mixed_rank_states_stay_queueable():
    # The generic non-PP path must give both ranks a non-empty scheduled batch:
    # one rank schedules its real request, while the terminal-only rank
    # schedules the dummy inserted for the scheduler-excluded request.
    busy_rank = _StubADPExecutor()
    busy_rank.active_requests = [_make_adp_request(_STATE_GENERATION_IN_PROGRESS)]
    busy_rank.expected_num_active_requests = 2
    terminal_rank = _StubADPExecutor()
    terminal_rank.active_requests = [_make_adp_request(_STATE_GENERATION_TO_COMPLETE)]
    terminal_rank.expected_num_active_requests = 2

    _run_pad(busy_rank)
    _run_pad(terminal_rank)

    assert busy_rank.add_dummy_calls == []
    assert len(terminal_rank.add_dummy_calls) == 1
    rank_batch_sizes = [
        busy_rank._count_schedulable_active_requests(),
        terminal_rank._count_schedulable_active_requests(),
    ]
    assert rank_batch_sizes == [1, 1]

    for stub, batch_size in zip((busy_rank, terminal_rank), rank_batch_sizes, strict=True):
        stub.dist.tp_allgather_int64 = Mock(
            return_value=np.array([[size] for size in rank_batch_sizes])
        )
        can_queue, can_queue_this_rank = PyExecutor._can_queue(
            stub, types.SimpleNamespace(batch_size=batch_size)
        )

        assert can_queue is True
        assert can_queue_this_rank is True
        PyExecutor._finalize_adp_dummy_allocation(stub, can_queue)

    assert terminal_rank._pending_adp_dummy_request is None


def test_pad_dummy_allocation_failure_skips_padding():
    # add_dummy_requests returns None when the rank has no free cache
    # resources for even a 1-token dummy (possible while non-schedulable
    # requests still hold theirs). Padding must degrade to a skipped
    # iteration, not crash on an unchecked [0].
    stub = _StubADPExecutor()
    stub.active_requests = [_make_adp_request(_STATE_GENERATION_TO_COMPLETE)]
    stub.expected_num_active_requests = 2
    stub.kv_cache_manager.add_dummy_requests.side_effect = None
    stub.kv_cache_manager.add_dummy_requests.return_value = None

    _run_pad(stub)

    assert len(stub.active_requests) == 1
    assert not any(r.is_attention_dp_dummy for r in stub.active_requests)


def test_adp_pad_dummy_checks_minimal_context_capacity():
    stub = _StubADPExecutor(
        max_num_tokens=4096,
        peer_forward_intent=_ADPForwardIntent.CONTEXT,
    )
    stub.kv_cache_manager.get_num_available_tokens.return_value = 0

    _run_pad(stub)

    stub.kv_cache_manager.get_num_available_tokens.assert_called_once_with(
        token_num_upper_bound=1, max_num_draft_tokens=0
    )
    stub.kv_cache_manager.add_dummy_requests.assert_not_called()
    assert stub._pending_adp_dummy_request is None


def test_adp_pad_dummy_ctx_preserves_helix_two_token_minimum():
    stub = _StubADPExecutor(
        max_num_tokens=4096,
        peer_forward_intent=_ADPForwardIntent.CONTEXT,
    )
    stub.kv_cache_manager.mapping.has_cp_helix.return_value = True
    stub.kv_cache_manager.get_num_available_tokens.return_value = 2

    _run_pad(stub)

    stub.kv_cache_manager.get_num_available_tokens.assert_called_once_with(
        token_num_upper_bound=2, max_num_draft_tokens=0
    )
    assert len(stub.add_dummy_calls) == 1
    # add_dummy_requests applies the same Helix minimum when constructing the
    # request; keep the caller's generic context minimum at one token.
    assert stub.add_dummy_calls[0]["token_nums"] == [1]


def test_adp_pad_dummy_checks_full_generation_capacity():
    stub = _StubADPExecutor()
    stub.kv_cache_manager.get_num_available_tokens.return_value = 0

    _run_pad(stub)

    stub.kv_cache_manager.get_num_available_tokens.assert_called_once_with(
        token_num_upper_bound=1, max_num_draft_tokens=0
    )
    stub.kv_cache_manager.add_dummy_requests.assert_not_called()
    assert stub._pending_adp_dummy_request is None


def test_adp_pad_dummy_capacity_includes_draft_reserve():
    stub = _StubADPExecutor()
    stub.max_total_draft_tokens = 3
    stub.kv_cache_manager.get_num_available_tokens.return_value = 3

    _run_pad(stub)

    stub.kv_cache_manager.get_num_available_tokens.assert_called_once_with(
        token_num_upper_bound=4, max_num_draft_tokens=3
    )
    stub.kv_cache_manager.add_dummy_requests.assert_not_called()
    assert stub._pending_adp_dummy_request is None


def test_pad_dummy_spec_allocation_failure_rolls_back_kv_candidate():
    stub = _StubADPExecutor()
    terminal_request = _make_adp_request(_STATE_GENERATION_TO_COMPLETE)
    stub.active_requests = [terminal_request]
    stub.expected_num_active_requests = 2
    spec_resource_manager = Mock()
    spec_resource_manager.add_dummy_requests.side_effect = NoFreeSlotsError("No free slots")
    stub.resource_manager.get_resource_manager.return_value = spec_resource_manager

    _run_pad(stub)

    assert stub.active_requests == [terminal_request]
    assert stub._pending_adp_dummy_request is None
    stub.kv_cache_manager.free_resources.assert_called_once()


def test_adp_dummy_peer_empty_rolls_back_and_retry_succeeds():
    stub = _StubADPExecutor()
    spec_resource_manager = Mock()
    stub.resource_manager.get_resource_manager.return_value = spec_resource_manager
    terminal_request = _make_adp_request(_STATE_GENERATION_TO_COMPLETE)
    stub.active_requests = [terminal_request]
    stub.expected_num_active_requests = 2

    _run_pad(stub)
    first_dummy = stub._pending_adp_dummy_request
    assert first_dummy is not None

    stub.dist.tp_allgather_int64 = Mock(return_value=np.array([[1], [0]]))
    can_queue, _ = PyExecutor._can_queue(stub, types.SimpleNamespace(batch_size=1))
    assert can_queue is False
    PyExecutor._finalize_adp_dummy_allocation(stub, can_queue)

    assert stub.active_requests == [terminal_request]
    spec_resource_manager.free_resources.assert_called_once_with(first_dummy)
    stub.kv_cache_manager.free_resources.assert_called_once_with(first_dummy)

    stub.dist.tp_allgather_int64 = Mock(return_value=np.array([[1], [1]]))
    _run_pad(stub)
    second_dummy = stub._pending_adp_dummy_request
    assert second_dummy is not None
    assert second_dummy is not first_dummy

    can_queue, _ = PyExecutor._can_queue(stub, types.SimpleNamespace(batch_size=1))
    assert can_queue is True
    PyExecutor._finalize_adp_dummy_allocation(stub, can_queue)

    assert stub._pending_adp_dummy_request is None
    assert stub.active_requests == [terminal_request, second_dummy]
    assert spec_resource_manager.add_dummy_requests.call_count == 2
    spec_resource_manager.free_resources.assert_called_once_with(first_dummy)
    stub.kv_cache_manager.free_resources.assert_called_once_with(first_dummy)


def test_adp_dummy_rollback_only_frees_pending_candidate():
    stub = _StubADPExecutor()
    prior_dummy = _make_adp_request(
        _STATE_GENERATION_IN_PROGRESS, request_id=10, is_dummy_request=True
    )
    prior_dummy.is_attention_dp_dummy = True
    current_dummy = _make_adp_request(
        _STATE_GENERATION_IN_PROGRESS, request_id=11, is_dummy_request=True
    )
    current_dummy.is_attention_dp_dummy = True
    stub.active_requests = [prior_dummy, current_dummy]
    stub._pending_adp_dummy_request = current_dummy

    PyExecutor._finalize_adp_dummy_allocation(stub, can_queue=False)

    assert stub.active_requests == [prior_dummy]
    stub.kv_cache_manager.free_resources.assert_called_once_with(current_dummy)


def test_adp_dummy_post_prepare_rollback_uses_normal_termination():
    stub = _StubADPExecutor()
    dummy_request = _make_adp_request(_STATE_GENERATION_IN_PROGRESS, is_dummy_request=True)
    dummy_request.is_attention_dp_dummy = True
    dummy_request.py_seq_slot = 3
    stub.active_requests = [dummy_request]
    stub._pending_adp_dummy_request = dummy_request
    stub._terminate_request = Mock()

    PyExecutor._finalize_adp_dummy_allocation(stub, can_queue=False)

    stub._terminate_request.assert_called_once_with(dummy_request)
    stub.kv_cache_manager.free_resources.assert_not_called()
    assert stub.active_requests == []


def test_pad_dummy_skips_when_active_request_present():
    stub = _StubADPExecutor()
    stub.active_requests = [_make_adp_request(_STATE_GENERATION_IN_PROGRESS)]
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert stub.add_dummy_calls == []
    assert len(stub.active_requests) == 1


@pytest.mark.parametrize(
    ("max_num_tokens", "model_max_seq_len", "manager_max_seq_len"),
    [
        (4096, 8232, 8233),
        (16384, 8232, 8233),
        (16384, 16384, 8192),
    ],
)
def test_pad_dummy_ctx_uses_minimal_prompt_independent_of_configured_limits(
    max_num_tokens,
    model_max_seq_len,
    manager_max_seq_len,
):
    # max_num_tokens is a batch-wide scheduling limit, not a target length for
    # the single synthetic context request.
    stub = _StubADPExecutor(
        max_num_tokens=max_num_tokens,
        max_seq_len=model_max_seq_len,
        kv_manager_max_seq_len=manager_max_seq_len,
        peer_forward_intent=_ADPForwardIntent.CONTEXT,
    )
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    call = stub.add_dummy_calls[0]
    assert call["token_nums"] == [1]
    assert call["is_gen"] is False
    assert stub.active_requests[-1].state == LlmRequestState.CONTEXT_INIT


def test_pad_dummy_gen_keeps_default_token_nums():
    stub = _StubADPExecutor(max_num_tokens=4096)
    stub._adp_dummy_is_gen = True
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    call = stub.add_dummy_calls[0]
    assert call["token_nums"] is None
    assert call["is_gen"] is True
    assert stub.active_requests[-1].state == _STATE_GENERATION_IN_PROGRESS


def test_overlap_adp_preserves_legacy_role_without_forward_intent_collective():
    stub = _StubADPExecutor(
        max_num_tokens=4096,
        enable_scheduler_aware_adp_dummy=False,
        enable_non_overlap_adp_forward_intent=False,
    )
    stub._adp_dummy_is_gen = False
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    assert stub.add_dummy_calls[0]["token_nums"] == [1]
    assert stub.add_dummy_calls[0]["is_gen"] is False
    stub.dist.tp_allreduce.assert_not_called()


def test_pad_dummy_ctx_uses_minimal_prompt_when_max_num_tokens_missing():
    stub = _StubADPExecutor(
        max_num_tokens=None,
        peer_forward_intent=_ADPForwardIntent.CONTEXT,
    )
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    assert stub.add_dummy_calls[0]["token_nums"] == [1]


def test_pad_dummy_not_added_when_all_ranks_only_await_kv_transfer():
    stub = _StubADPExecutor(
        max_num_tokens=4096,
        peer_forward_intent=_ADPForwardIntent.NONE,
    )
    stub.active_requests = [_make_adp_request(_STATE_DISAGG_GENERATION_INIT)]
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert stub.add_dummy_calls == []
    assert stub._adp_dummy_is_gen is True
    stub.dist.tp_allreduce.assert_called_once_with(int(_ADPForwardIntent.NONE), op=ReduceOp.MAX)


def test_pad_dummy_context_role_re_evaluated_while_local_rank_drains():
    stub = _StubADPExecutor(
        max_num_tokens=4096,
        peer_forward_intent=_ADPForwardIntent.CONTEXT,
    )
    stub._adp_dummy_is_gen = True
    stub.active_requests = [_make_adp_request(_STATE_DISAGG_GENERATION_INIT)]
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    assert stub.add_dummy_calls[0]["token_nums"] == [1]
    assert stub.add_dummy_calls[0]["is_gen"] is False


def test_compute_adp_dummy_tokens_splits_context_and_generation_work():
    scheduled_requests = types.SimpleNamespace(
        context_requests=[
            types.SimpleNamespace(
                is_attention_dp_dummy=True,
                context_chunk_size=1,
            ),
            types.SimpleNamespace(
                is_attention_dp_dummy=False,
                context_chunk_size=2048,
            ),
        ],
        generation_requests=[
            types.SimpleNamespace(
                is_attention_dp_dummy=True,
                py_draft_tokens=[1, 1, 1],
            ),
            types.SimpleNamespace(
                is_attention_dp_dummy=False,
                py_draft_tokens=[1, 1],
            ),
        ],
    )

    ctx_tokens, gen_tokens = PyExecutor._compute_adp_dummy_tokens(scheduled_requests)

    assert ctx_tokens == 1
    assert gen_tokens == 4


def test_pad_dummy_no_op_when_attention_dp_disabled():
    stub = _StubADPExecutor(enable_attention_dp=False)
    stub.active_requests = [_make_adp_request(_STATE_GENERATION_TO_COMPLETE)]
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert stub.add_dummy_calls == []


# ---------------------------------------------------------------------------
# Empty *scheduled* batch padding: a rank whose only active request does not fit
# the free KV cache is skipped by _pad_attention_dp_dummy_request, yet its empty
# scheduled batch vetoes the fleet-wide forward pass in _can_queue.
# ---------------------------------------------------------------------------
def _run_pad_empty(stub, scheduled_batch):
    for helper in (
        "_count_schedulable_active_requests",
        "_has_adp_dummy_kv_capacity",
        "_should_skip_dummy_for_benchmark_disagg",
    ):
        setattr(stub, helper, types.MethodType(getattr(PyExecutor, helper), stub))
    PyExecutor._pad_empty_attention_dp_batch(stub, scheduled_batch)


def _unfittable_rank(**kwargs):
    """A rank holding one active request the capacity scheduler could not fit."""
    stub = _StubADPExecutor(**kwargs)
    stub.active_requests = [_make_adp_request(_STATE_GENERATION_IN_PROGRESS)]
    stub.expected_num_active_requests = 1
    return stub, ScheduledRequests()


def test_pad_empty_batch_adds_generation_dummy_to_scheduled_batch():
    stub, scheduled_batch = _unfittable_rank()

    _run_pad_empty(stub, scheduled_batch)

    assert scheduled_batch.batch_size == 1
    assert len(scheduled_batch.generation_requests) == 1
    dummy = scheduled_batch.generation_requests[0]
    assert dummy.is_attention_dp_dummy is True
    assert dummy in stub.active_requests
    # A generation dummy, not the context dummy the pre-schedule path would
    # pick: this rank is empty precisely because it is short of KV cache.
    assert stub.add_dummy_calls[0]["is_gen"] is True
    assert stub.add_dummy_calls[0]["token_nums"] is None


def test_pad_empty_batch_no_op_when_batch_is_not_empty():
    stub, scheduled_batch = _unfittable_rank()
    scheduled_batch.context_requests_chunking = [
        _make_adp_request(_STATE_GENERATION_IN_PROGRESS, request_id=7)
    ]

    _run_pad_empty(stub, scheduled_batch)

    assert stub.add_dummy_calls == []
    assert scheduled_batch.generation_requests == []


def test_pad_empty_batch_no_op_when_attention_dp_disabled():
    stub, scheduled_batch = _unfittable_rank(enable_attention_dp=False)

    _run_pad_empty(stub, scheduled_batch)

    assert stub.add_dummy_calls == []
    assert scheduled_batch.batch_size == 0


def test_pad_empty_batch_no_op_when_rank_has_no_active_requests():
    # The pre-schedule path already covers this case.
    stub, scheduled_batch = _unfittable_rank()
    stub.active_requests = []

    _run_pad_empty(stub, scheduled_batch)

    assert stub.add_dummy_calls == []


def test_pad_empty_batch_respects_max_num_active_requests():
    # expected_num_active_requests is capped at max_num_active_requests and
    # _pad_attention_dp_dummy_request asserts it bounds len(active_requests),
    # so padding a rank already at the cap would trip that assert next
    # iteration.
    stub, scheduled_batch = _unfittable_rank()
    stub.max_num_active_requests = 1

    _run_pad_empty(stub, scheduled_batch)

    assert stub.add_dummy_calls == []
    assert scheduled_batch.batch_size == 0


def test_pad_empty_batch_skips_when_dummy_already_active():
    stub, scheduled_batch = _unfittable_rank()
    stub.active_requests.append(
        _make_adp_request(_STATE_GENERATION_IN_PROGRESS, request_id=ATTENTION_DP_DUMMY_REQUEST_ID)
    )

    _run_pad_empty(stub, scheduled_batch)

    assert stub.add_dummy_calls == []


def test_pad_empty_batch_degrades_when_kv_cache_cannot_afford_dummy():
    stub, scheduled_batch = _unfittable_rank()
    stub.kv_cache_manager.get_num_available_tokens.return_value = 0

    _run_pad_empty(stub, scheduled_batch)

    stub.kv_cache_manager.add_dummy_requests.assert_not_called()
    assert scheduled_batch.batch_size == 0


@pytest.mark.parametrize("error", [OutOfPagesError("no pages"), NoFreeSlotsError("no slots")])
def test_pad_empty_batch_degrades_on_allocation_error(error):
    # A rank-local raise would strand the peers in the collectives that follow.
    stub, scheduled_batch = _unfittable_rank()
    stub.kv_cache_manager.add_dummy_requests.side_effect = error

    _run_pad_empty(stub, scheduled_batch)

    assert scheduled_batch.batch_size == 0
    assert len(stub.active_requests) == 1


@pytest.mark.parametrize("enable_adp_dummy_fixes", [False, True])
def test_pad_empty_batch_dummy_rolled_back_when_fleet_still_cannot_queue(
    enable_adp_dummy_fixes,
):
    # can_queue=False skips the forward, so the usual dummy teardown in
    # _handle_responses is not reached. Rollback must not depend on whether
    # ADP dummy fixes are enabled.
    stub, scheduled_batch = _unfittable_rank(enable_adp_dummy_fixes=enable_adp_dummy_fixes)
    active_request = stub.active_requests[0]
    spec_resource_manager = Mock()
    stub.resource_manager.get_resource_manager.return_value = spec_resource_manager

    _run_pad_empty(stub, scheduled_batch)
    dummy = stub._pending_adp_dummy_request
    assert dummy is not None

    PyExecutor._finalize_adp_dummy_allocation(stub, False)

    assert stub.active_requests == [active_request]
    assert stub._pending_adp_dummy_request is None
    spec_resource_manager.free_resources.assert_called_once_with(dummy)
    stub.kv_cache_manager.free_resources.assert_called_once_with(dummy)


def test_pad_empty_batch_dummy_kept_when_fleet_can_queue():
    # Committed dummies are terminated after the forward by _handle_responses.
    stub, scheduled_batch = _unfittable_rank(enable_adp_dummy_fixes=False)

    _run_pad_empty(stub, scheduled_batch)
    dummy = stub._pending_adp_dummy_request

    PyExecutor._finalize_adp_dummy_allocation(stub, True)

    assert dummy in stub.active_requests
    assert stub._pending_adp_dummy_request is None
    stub.kv_cache_manager.free_resources.assert_not_called()


def test_pad_empty_batch_dummy_is_excluded_from_gen_alloc_revert():
    # The dummy joins after scheduling, so its capacity was never grown.
    stub, scheduled_batch = _unfittable_rank()

    _run_pad_empty(stub, scheduled_batch)

    # A request the scheduler did grow, for contrast.
    real_gen_request = _make_adp_request(_STATE_GENERATION_IN_PROGRESS, request_id=9)
    scheduled_batch.generation_requests.append(real_gen_request)

    stub._is_kv_manager_v2 = True
    PyExecutor._revert_gen_alloc(stub, scheduled_batch)

    reverted = [c.args[0] for c in stub.kv_cache_manager.revert_allocate_generation.call_args_list]
    assert reverted == [real_gen_request]


# ---------------------------------------------------------------------------
# ADP-safe disagg cache error handling (#13900): all TP ranks enter _handle_errors together.
# ---------------------------------------------------------------------------
def _err_req(request_id=1):
    return _make_adp_request(LlmRequestState.DISAGG_TRANS_ERROR, request_id=request_id)


def _disagg_error_vote(error_ids=(), blocked_ids=()):
    return {
        "error_ids": list(error_ids),
        "blocked_ids": list(blocked_ids),
    }


_DEFAULT_KV_CACHE_TRANSCEIVER = object()


def _make_disagg_err_stub(
    *,
    enable_attention_dp=True,
    kv_cache_transceiver=_DEFAULT_KV_CACHE_TRANSCEIVER,
    world_size=2,
    active_requests=None,
    tp_allgather_result=None,
):
    stub = types.SimpleNamespace()
    stub.enable_attention_dp = enable_attention_dp
    if kv_cache_transceiver is _DEFAULT_KV_CACHE_TRANSCEIVER:
        kv_cache_transceiver = Mock()
        kv_cache_transceiver.supports_inflight_request_cancellation.return_value = False
        kv_cache_transceiver.has_poisoned_transfer_buffer.return_value = False
    stub.kv_cache_transceiver = kv_cache_transceiver
    stub._disagg_inflight_cancel_unsupported_logged = False
    stub.active_requests = active_requests if active_requests is not None else []
    stub.dist = Mock()
    stub.dist.world_size = world_size
    stub.dist.rank = 0
    if tp_allgather_result is not None:
        stub.dist.tp_allgather = Mock(return_value=tp_allgather_result)
    else:
        stub.dist.tp_allgather = Mock(side_effect=lambda v: [v])
    stub.handle_errors_calls = []

    def _rec_handle_errors(error_msg, requests=None, charge_budget=True):
        stub.handle_errors_calls.append(
            {"error_msg": error_msg, "requests": requests, "charge_budget": charge_budget}
        )

    stub._handle_errors = _rec_handle_errors
    stub._request_vote_id = PyExecutor._request_vote_id
    for helper in (
        "_handle_disagg_cache_errors_synced",
        "_is_disagg_inflight_cancel_active",
        "_is_disagg_error_cleanup_blocked",
        "_get_disagg_reqs_in_error_state",
        "_check_cache_transfer_errors",
    ):
        setattr(stub, helper, types.MethodType(getattr(PyExecutor, helper), stub))
    return stub


class TestDisaggCacheErrorsSynced:
    def test_guard_short_circuits_without_transceiver(self):
        stub = _make_disagg_err_stub(kv_cache_transceiver=None, active_requests=[_err_req()])
        stub._handle_disagg_cache_errors_synced()
        stub.dist.tp_allgather.assert_not_called()
        assert stub.handle_errors_calls == []

    def test_guard_short_circuits_without_adp(self):
        stub = _make_disagg_err_stub(enable_attention_dp=False, active_requests=[_err_req()])
        stub._handle_disagg_cache_errors_synced()
        stub.dist.tp_allgather.assert_not_called()
        assert stub.handle_errors_calls == []

    def test_guard_short_circuits_single_rank(self):
        stub = _make_disagg_err_stub(world_size=1, active_requests=[_err_req()])
        stub._handle_disagg_cache_errors_synced()
        stub.dist.tp_allgather.assert_not_called()
        assert stub.handle_errors_calls == []

    def test_all_ranks_enter_when_a_peer_has_error(self):
        # A peer reports request 7. This rank must fail its matching replica,
        # while leaving an unrelated request active.
        matching = _make_adp_request(_STATE_GENERATION_IN_PROGRESS, request_id=7)
        unrelated = _make_adp_request(_STATE_GENERATION_IN_PROGRESS, request_id=8)
        stub = _make_disagg_err_stub(
            active_requests=[matching, unrelated],
            tp_allgather_result=[_disagg_error_vote(), _disagg_error_vote([7])],
        )
        stub._handle_disagg_cache_errors_synced()
        assert len(stub.handle_errors_calls) == 1
        assert stub.handle_errors_calls[0]["requests"] == [matching]
        assert stub.handle_errors_calls[0]["charge_budget"] is False

    def test_no_handle_when_no_rank_has_error(self):
        stub = _make_disagg_err_stub(
            active_requests=[],
            tp_allgather_result=[_disagg_error_vote(), _disagg_error_vote()],
        )
        stub._handle_disagg_cache_errors_synced()
        assert stub.handle_errors_calls == []

    def test_peer_error_without_local_replica_still_enters_handler(self):
        unrelated = _make_adp_request(_STATE_GENERATION_IN_PROGRESS, request_id=8)
        stub = _make_disagg_err_stub(
            active_requests=[unrelated],
            tp_allgather_result=[_disagg_error_vote(), _disagg_error_vote([7])],
        )

        stub._handle_disagg_cache_errors_synced()

        assert stub.handle_errors_calls[0]["requests"] == []

    def test_local_error_req_forwarded_request_scoped(self):
        err = _err_req()
        ok = _make_adp_request(_STATE_GENERATION_IN_PROGRESS, request_id=2)
        stub = _make_disagg_err_stub(
            active_requests=[ok, err], tp_allgather_result=[_disagg_error_vote([1])]
        )
        stub._handle_disagg_cache_errors_synced()
        assert len(stub.handle_errors_calls) == 1
        assert stub.handle_errors_calls[0]["requests"] == [err]
        assert stub.handle_errors_calls[0]["charge_budget"] is False

    def test_child_request_votes_by_parent_id(self):
        child = _make_adp_request(
            _STATE_GENERATION_IN_PROGRESS,
            request_id=101,
            is_child=True,
            parent_request_id=9,
        )
        stub = _make_disagg_err_stub(
            active_requests=[child],
            tp_allgather_result=[_disagg_error_vote(), _disagg_error_vote([9])],
        )

        stub._handle_disagg_cache_errors_synced()

        assert stub.handle_errors_calls[0]["requests"] == [child]

    def test_peer_vote_does_not_clean_up_locally_deferred_request(self):
        request = _err_req(request_id=7)
        request.is_context_only_request = True
        stub = _make_disagg_err_stub(
            active_requests=[request],
            tp_allgather_result=[
                _disagg_error_vote(blocked_ids=[7]),
                _disagg_error_vote([7]),
            ],
        )
        stub.canceled_req_ids = []
        stub.async_transfer_manager = Mock()
        stub.async_transfer_manager.requests_in_transfer.return_value = {
            request.py_request_id: request
        }

        stub._handle_disagg_cache_errors_synced()

        assert stub.handle_errors_calls == []


class TestCheckCacheTransferErrorsAdpNoop:
    def test_noop_under_adp_multirank(self):
        # Even with an error req present, ADP+world_size>1 defers to the synced handler.
        stub = _make_disagg_err_stub(active_requests=[_err_req()])
        stub._check_cache_transfer_errors("ctx")
        assert stub.handle_errors_calls == []

    def test_handles_error_when_not_adp(self):
        err = _err_req()
        stub = _make_disagg_err_stub(enable_attention_dp=False, active_requests=[err])
        stub._check_cache_transfer_errors("ctx")
        assert len(stub.handle_errors_calls) == 1
        assert stub.handle_errors_calls[0]["requests"] == [err]
        assert stub.handle_errors_calls[0]["charge_budget"] is False

    def test_handles_error_on_single_rank(self):
        err = _err_req()
        stub = _make_disagg_err_stub(world_size=1, active_requests=[err])
        stub._check_cache_transfer_errors("gen")
        assert len(stub.handle_errors_calls) == 1


class TestPendingTransferResponseFlush:
    def test_rank_local_fatal_error_does_not_issue_adp_response_gather(self):
        """A lone fatal rank must fail locally rather than desynchronize TP."""
        executor = object.__new__(PyExecutor)
        executor._error_budget = Mock()
        executor._error_budget.consume.return_value = True
        executor._error_budget.budget = 0.0
        executor._fatal_error = None
        executor.is_shutdown = False
        executor.enable_attention_dp = True
        executor.dist = Mock(world_size=2)
        executor.waiting_queue = []
        executor.executor_request_queue = Mock()
        executor.executor_request_queue.get_request_queue.return_value.empty.return_value = True
        executor.gather_all_responses = False
        executor.active_requests = []
        executor._pending_transfer_responses = []
        executor._enqueue_responses = Mock()
        executor._terminate_request = Mock()

        with pytest.raises(RuntimeError, match="Fatal error: local failure"):
            PyExecutor._handle_errors(executor, "local failure")

        executor._enqueue_responses.assert_not_called()
        executor.executor_request_queue.enqueue_shutdown_request.assert_called_once_with()

    def test_adp_flush_participates_with_an_empty_response_list(self):
        """Ranks without an error still join the synchronized response gather."""
        executor = object.__new__(PyExecutor)
        executor._pending_transfer_responses = []
        executor._pending_response_terminations = []
        executor.enable_attention_dp = True
        executor._enqueue_responses = Mock()

        PyExecutor._flush_pending_transfer_responses(executor)

        executor._enqueue_responses.assert_called_once_with([])

    def test_flush_delivers_and_clears_buffered_responses(self):
        executor = object.__new__(PyExecutor)
        responses = [(7, Mock())]
        executor._pending_transfer_responses = responses
        executor._pending_response_terminations = []
        executor.enable_attention_dp = False
        executor._enqueue_responses = Mock()

        PyExecutor._flush_pending_transfer_responses(executor)

        executor._enqueue_responses.assert_called_once_with(responses)
        assert executor._pending_transfer_responses == []

    def test_rank_zero_keeps_result_queue_until_buffered_error_flushes(self):
        """A queued client receives a rank-0 ADP error before cleanup."""
        executor = object.__new__(PyExecutor)
        request_id = 7
        response = LlmResponse(request_id=request_id)
        response.request_id = request_id
        response.client_id = 42
        response.error_msg = "transfer failed"
        result_queue = Mock()
        executor._pending_transfer_responses = [(request_id, response)]
        request = types.SimpleNamespace(py_request_id=request_id)
        executor._pending_response_terminations = [request]
        executor.enable_attention_dp = False
        executor.gather_all_responses = False
        executor.dist = Mock(rank=0)
        executor.dist.mapping.tp_group = [0]
        executor.responses = {}
        executor.response_cv = threading.Condition()
        executor.result_wait_queues = {request_id: result_queue}
        executor._terminate_request = Mock(
            side_effect=lambda _: executor.result_wait_queues.pop(request_id)
        )

        PyExecutor._flush_pending_transfer_responses(executor)

        result_queue.put_response.remote.assert_called_once_with(42, response)
        assert request_id not in executor.result_wait_queues
        executor._terminate_request.assert_called_once_with(request)

    @staticmethod
    def _make_executor_loop_stub():
        executor = object.__new__(PyExecutor)
        executor.device_id = 0
        profiler = MagicMock()
        profiler.__enter__.return_value = Mock()
        executor._profiler = Mock(return_value=profiler)
        executor.hang_detector = MagicMock()
        executor.enable_iter_perf_stats = False
        executor._resource_governor_enabled = False
        executor._is_kv_manager_v2 = False
        executor._mm_encoder_item_scheduling_enabled = False
        executor.is_benchmark_disagg = False
        executor._handle_disagg_cache_errors_synced = Mock()
        executor._flush_pending_transfer_responses = Mock()
        return executor

    @staticmethod
    def _patch_executor_loop_cuda(monkeypatch):
        monkeypatch.setattr(
            "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.set_device",
            Mock(),
        )
        monkeypatch.setattr(
            "tensorrt_llm._torch.pyexecutor.py_executor.cudart.cudaSetDevice",
            Mock(),
        )
        monkeypatch.setattr(
            "tensorrt_llm._torch.pyexecutor.py_executor.CUASSERT",
            Mock(),
        )

    def test_flushes_before_clean_scheduler_shutdown(self, monkeypatch):
        """A response buffered before scheduling must survive a clean exit."""
        executor = self._make_executor_loop_stub()
        executor._prepare_and_schedule_batch = Mock(return_value=(None, None))
        self._patch_executor_loop_cuda(monkeypatch)

        PyExecutor._executor_loop(executor)

        executor._flush_pending_transfer_responses.assert_called_once_with()

    def test_flushes_before_benchmark_retry(self, monkeypatch):
        """The synchronized benchmark retry path must not strand a response."""
        executor = self._make_executor_loop_stub()
        scheduled_batch = types.SimpleNamespace(generation_requests=[])
        executor._prepare_and_schedule_batch = Mock(
            side_effect=[(scheduled_batch, None), (None, None)]
        )
        executor._check_benchmark_disagg_gate = Mock(return_value=(False, True))
        executor._finalize_adp_dummy_allocation = Mock()
        self._patch_executor_loop_cuda(monkeypatch)

        PyExecutor._executor_loop(executor)

        # Once for the retry pass and once for the following clean exit.
        assert executor._flush_pending_transfer_responses.call_count == 2

    def test_idle_pass_has_one_flush(self, monkeypatch):
        """An idle pass must not pay an additional response gather."""
        executor = self._make_executor_loop_stub()
        scheduled_batch = types.SimpleNamespace(
            encoder_requests=[], paused_requests=[], generation_requests=[]
        )
        executor._prepare_and_schedule_batch = Mock(
            side_effect=[(scheduled_batch, None), (None, None)]
        )
        executor._check_benchmark_disagg_gate = Mock(return_value=(True, False))
        executor._terminate_requests = Mock()
        executor._pause_requests = Mock()
        executor._can_queue = Mock(return_value=(False, None))
        executor.kv_connector_manager = None
        executor._revert_gen_alloc = Mock()
        executor._finalize_adp_dummy_allocation = Mock()
        executor._handle_kv_transfer_timeouts_synced = Mock()
        executor.kv_cache_transceiver = None
        executor._kv_connector_terminate_requests = Mock()
        executor._flush_iter_stats_synced = Mock()
        executor.iter_counter = 0
        self._patch_executor_loop_cuda(monkeypatch)

        PyExecutor._executor_loop(executor)

        # One completed idle pass plus the clean-exit drain, not two flushes
        # during the idle pass itself.
        assert executor._flush_pending_transfer_responses.call_count == 2

    def test_overlap_flushes_before_clean_scheduler_shutdown(self, monkeypatch):
        """The overlap loop must not drop a buffered response on clean exit."""
        executor = self._make_executor_loop_stub()
        executor._can_pause_for_rebalance = Mock(return_value=False)
        executor._wait_for_model_engine_input_copy = Mock()
        executor._prepare_and_schedule_batch = Mock(return_value=(None, None))
        self._patch_executor_loop_cuda(monkeypatch)

        PyExecutor._executor_loop_overlap(executor)

        executor._flush_pending_transfer_responses.assert_called_once_with()

    def test_overlap_flushes_before_benchmark_retry(self, monkeypatch):
        """The overlap retry path must not strand a buffered response."""
        executor = self._make_executor_loop_stub()
        scheduled_batch = types.SimpleNamespace(generation_requests=[])
        executor._can_pause_for_rebalance = Mock(return_value=False)
        executor._wait_for_model_engine_input_copy = Mock()
        executor._prepare_and_schedule_batch = Mock(
            side_effect=[(scheduled_batch, None), (None, None)]
        )
        executor._check_benchmark_disagg_gate = Mock(return_value=(False, True))
        executor._finalize_adp_dummy_allocation = Mock()
        self._patch_executor_loop_cuda(monkeypatch)

        PyExecutor._executor_loop_overlap(executor)

        # Once for the retry pass and once for the following clean exit.
        assert executor._flush_pending_transfer_responses.call_count == 2


class TestOneModelMTPDraftTokenScheduling:
    """Regression tests for the one-model MTP over-scheduling bug (#16101).

    One-model MTP (``mtp_eagle_one_model``) has no separate drafter, so
    ``get_spec_drafter()`` returns None and the ``if self.drafter is not None``
    draft-token normalization block in ``_prepare_and_schedule_batch`` is
    skipped. Without the ``elif`` fallback that mirrors it, generation requests
    keep ``num_draft_tokens == 0`` and the C++ micro-batch scheduler
    under-reserves each gen request (it budgets ``beam_width +
    getNumDraftTokens()``). Under chunked prefill + overlap scheduler the
    forward then builds a uniform ``1 + runtime_draft_len`` per gen request and
    overshoots ``max_num_tokens`` (``total_num_tokens > max_num_tokens``).

    The fix populates both the Python and C++ draft-token representations on
    every in-progress generation request so both schedulers reserve the
    correct token budget. This test drives ``_prepare_and_schedule_batch`` for
    a one-model-MTP executor and asserts generation requests get the full
    draft-token budget while context requests are left untouched.

    The Python-side fill is placeholder-only: with the overlap scheduler
    disabled, ``_prepare_tp_inputs`` sources a generation request's draft
    tokens from ``py_draft_tokens``, which the one-model spec sampler wrote at
    the end of the previous iteration. Overwriting a populated list here would
    feed zeros to the target model and collapse the acceptance rate.

    NOTE: Like ``test_fetch_called_once_even_in_benchmark_disagg`` in
    ``test_benchmark_disagg.py``, this uses ``object.__new__(PyExecutor)`` to
    bypass ``__init__`` and sets internal attributes by hand. Real
    ``LlmRequest`` objects (not Mocks) are used so ``draft_tokens = [0] * N``
    actually updates the C++-backed ``num_draft_tokens`` count.
    """

    MAX_TOTAL_DRAFT_TOKENS = 3

    @staticmethod
    def _make_llm_request(request_id: int, state: LlmRequestState) -> LlmRequest:
        """Build a real LlmRequest in the given state (mirrors the helper in
        test_py_scheduler.py::make_generation_request)."""
        req = LlmRequest(
            request_id=request_id,
            max_new_tokens=10,
            input_tokens=list(range(10)),
            sampling_config=SamplingConfig(1),
            is_streaming=False,
            draft_tokens=None,
        )
        req.state = state
        return req

    @classmethod
    def _make_one_model_mtp_executor(
        cls,
        active_requests: list[LlmRequest],
        use_rejection_sampling: bool = False,
    ) -> PyExecutor:
        """Construct a partially-initialised one-model-MTP PyExecutor.

        drafter is None (one-model MTP has no separate drafter) and
        model_engine.is_spec_decode is True, so _prepare_and_schedule_batch
        takes the elif draft-token normalization branch. kv_cache_transceiver
        is None to keep the test hermetic (skips the disagg blocks).
        enable_attention_dp is False so _pad_empty_attention_dp_batch returns
        immediately instead of padding a batch this test does not exercise.
        """
        ex = object.__new__(PyExecutor)
        ex.drafter = None
        ex.max_total_draft_tokens = cls.MAX_TOTAL_DRAFT_TOKENS
        spec_config = MTPDecodingConfig(
            max_draft_len=cls.MAX_TOTAL_DRAFT_TOKENS,
            mtp_eagle_one_model=True,
            use_rejection_sampling=use_rejection_sampling,
            draft_len_schedule={1: cls.MAX_TOTAL_DRAFT_TOKENS},
        )
        ex.model_engine = Mock(
            is_spec_decode=True,
            spec_config=spec_config,
            max_draft_len=cls.MAX_TOTAL_DRAFT_TOKENS,
            max_total_draft_tokens=cls.MAX_TOTAL_DRAFT_TOKENS,
        )
        ex.kv_cache_transceiver = None
        ex.is_shutdown = False
        ex.enable_iter_perf_stats = False
        ex.enable_attention_dp = False
        ex.speculation_permanently_disabled = False
        ex.active_requests = active_requests
        ex.waiting_queue = []

        ex._fetch_and_activate_new_requests = Mock(return_value=[])
        ex._check_disagg_ctx_schedulable_status = Mock()
        ex._check_disagg_gen_transfer_status = Mock()
        ex._check_kv_transfer_timeout = Mock()
        ex._check_disagg_ctx_cache_transfer_status = Mock()
        ex._pad_attention_dp_dummy_request = Mock()
        ex._prefetch_for_context_requests = Mock()
        ex._prepare_disagg_gen_init = Mock()
        ex._schedule = Mock(return_value=(ScheduledRequests(), [], 0))
        return ex

    def test_one_model_mtp_populates_draft_tokens_for_scheduling(self):
        """The draft-token normalization must cover BOTH aggregated and
        disaggregated serving in one shot.

        The fix's state filter is {GENERATION_IN_PROGRESS,
        DISAGG_GENERATION_INIT}, mirroring the two-model normalization, so a
        single fix covers the aggregated decode path (GENERATION_IN_PROGRESS)
        and the disagg decode-worker path (DISAGG_GENERATION_INIT). Context
        requests (CONTEXT_INIT) are not generation requests and must be left
        untouched.
        """
        gen = self._make_llm_request(0, LlmRequestState.GENERATION_IN_PROGRESS)
        disagg_gen = self._make_llm_request(1, LlmRequestState.DISAGG_GENERATION_INIT)
        ctx = self._make_llm_request(2, LlmRequestState.CONTEXT_INIT)

        # Precondition: no draft tokens reserved yet on either gen request.
        assert gen.num_draft_tokens == 0
        assert disagg_gen.num_draft_tokens == 0
        assert gen.py_draft_tokens == []
        assert disagg_gen.py_draft_tokens == []

        ex = self._make_one_model_mtp_executor([gen, disagg_gen, ctx])
        scheduled_batch, _ = ex._prepare_and_schedule_batch()

        assert scheduled_batch is not None
        # Aggregated case: in-progress generation request is normalized to the
        # full draft-token budget so the micro-batch scheduler reserves
        # beam + max_total_draft_tokens.
        assert gen.num_draft_tokens == self.MAX_TOTAL_DRAFT_TOKENS
        assert gen.py_draft_tokens == [0] * self.MAX_TOTAL_DRAFT_TOKENS
        # Disaggregated case: decode-worker request awaiting KV also normalized.
        assert disagg_gen.num_draft_tokens == self.MAX_TOTAL_DRAFT_TOKENS
        assert disagg_gen.py_draft_tokens == [0] * self.MAX_TOTAL_DRAFT_TOKENS
        # Context requests are not generation requests and must be left alone.
        assert ctx.num_draft_tokens == 0
        assert ctx.py_draft_tokens == []

    def test_one_model_mtp_preserves_zero_proposal_signal_for_rejection(self) -> None:
        gen = self._make_llm_request(0, LlmRequestState.GENERATION_IN_PROGRESS)
        ex = self._make_one_model_mtp_executor([gen], use_rejection_sampling=True)

        ex._prepare_and_schedule_batch()

        assert gen.num_draft_tokens == self.MAX_TOTAL_DRAFT_TOKENS
        assert gen.py_draft_tokens == [0] * self.MAX_TOTAL_DRAFT_TOKENS
        assert gen.py_needs_onehot_draft_probs

        batch = ScheduledRequests()
        batch.append_generation_request(gen)
        ex._handle_dynamic_draft_len(batch)

        assert ex.model_engine.runtime_draft_len == self.MAX_TOTAL_DRAFT_TOKENS
        assert gen.py_needs_onehot_draft_probs
        assert gen.py_draft_tokens == [0] * self.MAX_TOTAL_DRAFT_TOKENS

    def test_one_model_mtp_preserves_sampler_draft_tokens(self) -> None:
        sampler_drafts = [7, 8]
        gen = self._make_llm_request(0, LlmRequestState.GENERATION_IN_PROGRESS)
        gen.py_draft_tokens = list(sampler_drafts)

        ex = self._make_one_model_mtp_executor([gen])
        ex._prepare_and_schedule_batch()

        assert gen.py_draft_tokens == sampler_drafts
        assert gen.num_draft_tokens == self.MAX_TOTAL_DRAFT_TOKENS


class TestAdpBalanceExcludesPadDummies:
    """The low-occupancy test must look at real decode work, not batch size.

    `_pad_attention_dp_dummy_request` gives an otherwise idle rank exactly one
    generation request so attention DP can make progress. Counting that dummy
    as decode work makes the rank look busy, which is precisely wrong for a
    check whose job is to notice idle ranks.
    """

    @staticmethod
    def _make_executor(per_rank, threshold):
        """per_rank: list of (num_ctx, num_gen, num_real) as allgathered."""
        executor = object.__new__(PyExecutor)
        executor.dist = Mock()
        executor.dist.tp_allgather_int64 = Mock(
            return_value=np.array([[ctx, gen, ctx + gen, real] for ctx, gen, real in per_rank])
        )
        executor.max_batch_size = 64
        executor.attention_dp_enable_balance = True
        executor.attention_dp_time_out_iters = 60
        executor.attention_dp_batching_wait_iters = 10
        executor.attention_dp_min_generation_requests = threshold
        executor.attention_dp_low_occupancy_timeout_iters = 0
        executor.adp_ctx_waiting_iters_count = 0
        executor.adp_ctx_batching_wait_iters_count = 0
        return executor

    @staticmethod
    def _make_generation_request(is_dummy):
        request = Mock()
        request.is_attention_dp_dummy = is_dummy
        return request

    @staticmethod
    def _make_context_request(num_tokens=4):
        request = Mock()
        request.get_tokens.return_value = [0] * num_tokens
        return request

    def test_padded_rank_counts_as_under_occupied(self):
        """A rank holding only a pad dummy has zero real decode work.

        Peer rank is padded: it reports one scheduled generation request and
        zero real ones. With a threshold of 1 the low-occupancy timeout of 0
        must be selected so the withheld context is released on this very
        iteration. Counting the dummy would leave the cross-rank minimum at 1,
        keep the 60-iteration timeout, and strand the context batch.
        """
        context_requests = [self._make_context_request()]
        # This rank has context and decode work; the peer is padded and has
        # no context request, so the batch is not aligned and the balancer
        # takes the timeout branch.
        executor = self._make_executor([(1, 4, 4), (0, 1, 0)], threshold=1)

        balanced = executor._balance_adp_requests(
            context_requests, [self._make_generation_request(False)]
        )

        assert balanced == context_requests
        assert executor.adp_ctx_waiting_iters_count == 0

    def test_busy_ranks_still_wait(self):
        """No rank is under-occupied, so the configured timeout still applies."""
        executor = self._make_executor([(1, 4, 4), (0, 4, 4)], threshold=1)

        balanced = executor._balance_adp_requests(
            [self._make_context_request()], [self._make_generation_request(False)]
        )

        assert balanced == []
        assert executor.adp_ctx_waiting_iters_count == 1

    def test_real_count_excludes_dummies(self):
        """The allgathered real count must not include the pad dummy."""
        executor = self._make_executor([(1, 2, 1)], threshold=0)

        executor._balance_adp_requests(
            [self._make_context_request()],
            [
                self._make_generation_request(True),
                self._make_generation_request(False),
            ],
        )

        gathered = executor.dist.tp_allgather_int64.call_args[0][0]
        assert gathered[1] == 2, "scheduled count keeps counting the dummy"
        assert gathered[3] == 1, "real count must exclude the dummy"
