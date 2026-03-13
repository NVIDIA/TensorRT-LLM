"""Tests for PyExecutor request handling functionality.

This module tests the request handling logic that was moved from ExecutorRequestQueue
to PyExecutor, including:
- _handle_special_queue_items method
- canceled_req_ids management
- waiting_queue management
- is_shutdown state management
- expected_num_active_requests tracking
- one-model MTP draft token normalization before scheduling
"""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    SHUTDOWN_REQUEST_ID,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
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


def _normalize_one_model_mtp_draft_tokens(drafter, model_engine, active_requests):
    """Replicate the one-model MTP draft token normalization from
    PyExecutor._prepare_and_schedule_batch so it can be unit-tested
    without standing up a full executor.
    """
    if drafter is None and model_engine.enable_spec_decode:
        max_draft = model_engine.max_total_draft_tokens
        if max_draft > 0:
            for request in active_requests:
                if request.state not in (
                    LlmRequestState.GENERATION_IN_PROGRESS,
                    LlmRequestState.DISAGG_GENERATION_INIT,
                ):
                    continue
                request.draft_tokens = [0] * max_draft


def test_one_model_mtp_draft_token_normalization():
    """Verify that one-model MTP normalizes C++ draft_tokens for all
    generation requests before the scheduler runs.

    Without this normalization the C++ scheduler can under-budget
    generation requests (seeing fewer draft tokens than the runtime
    will actually materialize), causing a total_num_tokens overflow.
    """
    max_total_draft_tokens = 2  # e.g. MTP nextn=2

    model_engine = Mock()
    model_engine.enable_spec_decode = True
    model_engine.max_total_draft_tokens = max_total_draft_tokens

    # Generation request with stale (empty) draft_tokens — simulates
    # a request that just transitioned from context to generation.
    gen_req = Mock()
    gen_req.state = LlmRequestState.GENERATION_IN_PROGRESS
    gen_req.draft_tokens = []

    # Context request — should NOT be touched.
    ctx_req = Mock()
    ctx_req.state = LlmRequestState.CONTEXT_INIT
    ctx_req.draft_tokens = []

    # Disagg generation init — should be normalized.
    disagg_req = Mock()
    disagg_req.state = LlmRequestState.DISAGG_GENERATION_INIT
    disagg_req.draft_tokens = [99]  # stale value

    active_requests = [gen_req, ctx_req, disagg_req]

    _normalize_one_model_mtp_draft_tokens(
        drafter=None,
        model_engine=model_engine,
        active_requests=active_requests,
    )

    assert gen_req.draft_tokens == [0] * max_total_draft_tokens, (
        "GENERATION_IN_PROGRESS request should have normalized draft_tokens"
    )
    assert ctx_req.draft_tokens == [], "CONTEXT_INIT request should not be modified"
    assert disagg_req.draft_tokens == [0] * max_total_draft_tokens, (
        "DISAGG_GENERATION_INIT request should have normalized draft_tokens"
    )


def test_one_model_mtp_normalization_skipped_with_drafter():
    """When a two-model drafter is present, one-model normalization
    should be skipped (the drafter path handles it separately)."""
    model_engine = Mock()
    model_engine.enable_spec_decode = True
    model_engine.max_total_draft_tokens = 3

    drafter = Mock()  # non-None drafter

    gen_req = Mock()
    gen_req.state = LlmRequestState.GENERATION_IN_PROGRESS
    gen_req.draft_tokens = []

    _normalize_one_model_mtp_draft_tokens(
        drafter=drafter,
        model_engine=model_engine,
        active_requests=[gen_req],
    )

    assert gen_req.draft_tokens == [], (
        "With a two-model drafter, one-model normalization should not run"
    )


def test_one_model_mtp_normalization_skipped_when_spec_decode_off():
    """When spec decode is disabled, normalization should be skipped."""
    model_engine = Mock()
    model_engine.enable_spec_decode = False
    model_engine.max_total_draft_tokens = 2

    gen_req = Mock()
    gen_req.state = LlmRequestState.GENERATION_IN_PROGRESS
    gen_req.draft_tokens = []

    _normalize_one_model_mtp_draft_tokens(
        drafter=None,
        model_engine=model_engine,
        active_requests=[gen_req],
    )

    assert gen_req.draft_tokens == [], "With spec decode disabled, normalization should not run"
