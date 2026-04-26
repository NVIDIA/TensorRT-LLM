"""Tests for PyExecutor request handling functionality.

This module tests the request handling logic that was moved from ExecutorRequestQueue
to PyExecutor, including:
- _handle_special_queue_items method
- canceled_req_ids management
- waiting_queue management
- is_shutdown state management
- expected_num_active_requests tracking
"""

import types
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    SHUTDOWN_REQUEST_ID,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue


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


def _classify_termination(request, enable_partial_reuse_for_disagg, is_vswa, pp_size):
    """Reproduce the termination logic from _handle_responses (py_executor.py).

    Returns:
        "terminate" | "stats_only" | "skip"
    """
    force_terminate_for_partial_reuse = (
        enable_partial_reuse_for_disagg and not is_vswa and pp_size == 1
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
        assert _classify_termination(req, False, False, 1) == "stats_only"

    def test_normal_path_skips_transmission_in_progress(self):
        """Without partial reuse, TRANS_IN_PROGRESS is skipped (still in flight)."""
        req = _make_request(complete_state=False, transmission_state=True)
        assert _classify_termination(req, False, False, 1) == "skip"

    def test_normal_path_terminates_regular_request(self):
        """Without partial reuse, a normal finished request is terminated."""
        req = _make_request(complete_state=False, transmission_state=False)
        assert _classify_termination(req, False, False, 1) == "terminate"

    def test_partial_reuse_terminates_non_complete(self):
        """With partial reuse, non-CONTEXT_COMPLETE requests are terminated."""
        for complete, transmission in [(False, True), (False, False)]:
            req = _make_request(complete, transmission)
            assert _classify_termination(req, True, False, 1) == "terminate"

    def test_partial_reuse_skips_context_complete(self):
        """With partial reuse, CONTEXT_COMPLETE still goes to stats only."""
        req = _make_request(complete_state=True, transmission_state=False)
        assert _classify_termination(req, True, False, 1) == "stats_only"

    def test_partial_reuse_disabled_by_vswa(self):
        """VSWA disables partial reuse path, falling back to normal logic."""
        req = _make_request(complete_state=True, transmission_state=False)
        assert _classify_termination(req, True, True, 1) == "stats_only"

    def test_partial_reuse_disabled_by_pp(self):
        """PP > 1 disables partial reuse path, falling back to normal logic."""
        req = _make_request(complete_state=True, transmission_state=False)
        assert _classify_termination(req, True, False, 2) == "stats_only"


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


_STATE_GENERATION_IN_PROGRESS = LlmRequestState.GENERATION_IN_PROGRESS
_STATE_GENERATION_TO_COMPLETE = LlmRequestState.GENERATION_TO_COMPLETE
_STATE_DISAGG_GENERATION_INIT = "_disagg_init_sentinel"
_STATE_DISAGG_GENERATION_TRANS_IN_PROGRESS = "_disagg_trans_sentinel"


def _make_adp_request(state, *, llm_request_type=None, is_dummy_request=False):
    req = Mock()
    req.state = state
    req.is_disagg_generation_init_state = state == _STATE_DISAGG_GENERATION_INIT
    req.is_disagg_generation_transmission_in_progress = (
        state == _STATE_DISAGG_GENERATION_TRANS_IN_PROGRESS
    )
    req.is_dummy_request = is_dummy_request
    req.is_attention_dp_dummy = False
    req.llm_request_type = llm_request_type
    return req


class _StubADPExecutor:
    def __init__(
        self,
        *,
        enable_attention_dp=True,
        kv_cache_transceiver=object(),
        max_num_tokens=8192,
        is_warmup=False,
        benchmark_req_queues_size=0,
    ):
        self.enable_attention_dp = enable_attention_dp
        self.kv_cache_transceiver = kv_cache_transceiver
        self.is_warmup = is_warmup
        self.benchmark_req_queues_size = benchmark_req_queues_size
        self._benchmark_fill_phase_active = False
        self.num_fetch_requests = 0
        self.active_requests = []
        self.expected_num_active_requests = 1
        self.max_total_draft_tokens = 0
        self.max_num_tokens = max_num_tokens
        self._adp_dummy_is_gen = True
        self._adp_dummy_role_locked = False
        self.add_dummy_calls = []

        kv_cache_manager = Mock()

        def _add_dummy(**kwargs):
            self.add_dummy_calls.append(kwargs)
            req = _make_adp_request(_STATE_GENERATION_IN_PROGRESS, is_dummy_request=True)
            return [req]

        kv_cache_manager.add_dummy_requests.side_effect = _add_dummy
        self.kv_cache_manager = kv_cache_manager

        self.resource_manager = Mock()
        self.resource_manager.get_resource_manager.return_value = None


def _run_pad(stub):
    for helper in ("_count_schedulable_active_requests", "_should_skip_dummy_for_benchmark_disagg"):
        setattr(stub, helper, types.MethodType(getattr(PyExecutor, helper), stub))
    PyExecutor._pad_attention_dp_dummy_request(stub)


def _run_role_lock(stub, validated_requests):
    from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

    if (
        stub.enable_attention_dp
        and stub.kv_cache_transceiver is not None
        and not stub._adp_dummy_role_locked
    ):
        for request in validated_requests:
            rt = getattr(request, "llm_request_type", None)
            if rt == LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY:
                stub._adp_dummy_is_gen = False
                stub._adp_dummy_role_locked = True
                break
            if rt == LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY:
                stub._adp_dummy_is_gen = True
                stub._adp_dummy_role_locked = True
                break


def test_adp_dummy_role_locks_to_ctx_on_first_context_only_request():
    from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

    stub = _StubADPExecutor()
    req = _make_adp_request(
        _STATE_GENERATION_IN_PROGRESS,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    _run_role_lock(stub, [req])

    assert stub._adp_dummy_role_locked is True
    assert stub._adp_dummy_is_gen is False


def test_adp_dummy_role_locks_to_gen_on_first_generation_only_request():
    from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

    stub = _StubADPExecutor()
    stub._adp_dummy_is_gen = False
    req = _make_adp_request(
        _STATE_GENERATION_IN_PROGRESS,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
    )
    _run_role_lock(stub, [req])

    assert stub._adp_dummy_role_locked is True
    assert stub._adp_dummy_is_gen is True


def test_adp_dummy_role_does_not_lock_for_non_disagg_worker():
    from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

    stub = _StubADPExecutor(kv_cache_transceiver=None)
    req = _make_adp_request(
        _STATE_GENERATION_IN_PROGRESS,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    _run_role_lock(stub, [req])

    assert stub._adp_dummy_role_locked is False
    assert stub._adp_dummy_is_gen is True


def test_adp_dummy_role_does_not_lock_when_attention_dp_disabled():
    from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

    stub = _StubADPExecutor(enable_attention_dp=False)
    req = _make_adp_request(
        _STATE_GENERATION_IN_PROGRESS,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    _run_role_lock(stub, [req])

    assert stub._adp_dummy_role_locked is False


def test_pad_dummy_excludes_generation_to_complete_in_disagg():
    stub = _StubADPExecutor()
    stub.active_requests = [_make_adp_request(_STATE_GENERATION_TO_COMPLETE)]
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    assert len(stub.active_requests) == 2


def test_pad_dummy_excludes_generation_to_complete_in_non_disagg():
    stub = _StubADPExecutor(kv_cache_transceiver=None)
    stub.active_requests = [_make_adp_request(_STATE_GENERATION_TO_COMPLETE)]
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1


def test_pad_dummy_skips_when_active_request_present():
    stub = _StubADPExecutor()
    stub.active_requests = [_make_adp_request(_STATE_GENERATION_IN_PROGRESS)]
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert stub.add_dummy_calls == []
    assert len(stub.active_requests) == 1


def test_pad_dummy_ctx_pads_to_max_num_tokens():
    stub = _StubADPExecutor(max_num_tokens=4096)
    stub._adp_dummy_is_gen = False
    stub._adp_dummy_role_locked = True
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    call = stub.add_dummy_calls[0]
    assert call["token_nums"] == [4096]
    assert call["is_gen"] is False


def test_pad_dummy_gen_keeps_default_token_nums():
    stub = _StubADPExecutor(max_num_tokens=4096)
    stub._adp_dummy_is_gen = True
    stub._adp_dummy_role_locked = True
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    call = stub.add_dummy_calls[0]
    assert call["token_nums"] is None
    assert call["is_gen"] is True


def test_pad_dummy_ctx_skips_padding_when_max_num_tokens_missing():
    stub = _StubADPExecutor(max_num_tokens=None)
    stub._adp_dummy_is_gen = False
    stub._adp_dummy_role_locked = True
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert len(stub.add_dummy_calls) == 1
    assert stub.add_dummy_calls[0]["token_nums"] is None


def test_pad_dummy_no_op_when_attention_dp_disabled():
    stub = _StubADPExecutor(enable_attention_dp=False)
    stub.active_requests = [_make_adp_request(_STATE_GENERATION_TO_COMPLETE)]
    stub.expected_num_active_requests = 1

    _run_pad(stub)

    assert stub.add_dummy_calls == []
