"""Tests for PyExecutor request handling functionality.

This module tests the request handling logic that was moved from ExecutorRequestQueue
to PyExecutor, including:
- _handle_special_queue_items method
- canceled_req_ids management
- waiting_queue management
- is_shutdown state management
- expected_num_active_requests tracking
"""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    SHUTDOWN_REQUEST_ID,
    RequestQueueItem,
)
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


# ---------------------------------------------------------------------------
# Tests for _waiting_requests with KV cache reuse awareness
# ---------------------------------------------------------------------------


def _make_ctx_request(
    num_tokens, estimated_reusable_tokens=0, is_first_context_chunk=True, context_current_position=0
):
    """Helper to create a mock context request."""
    req = Mock()
    req.get_tokens = Mock(return_value=list(range(num_tokens)))
    req.estimated_reusable_tokens = estimated_reusable_tokens
    req.is_first_context_chunk = is_first_context_chunk
    req.context_current_position = context_current_position
    req.context_chunk_size = num_tokens
    return req


class MockPyExecutorForWaiting:
    """Mock for testing _waiting_requests.

    Calls PyExecutor._waiting_requests directly to avoid mirroring logic.
    """

    # Expose the static method so self._compute_scheduled_tokens works
    # when PyExecutor._waiting_requests is called with this mock as self.
    _compute_scheduled_tokens = staticmethod(PyExecutor._compute_scheduled_tokens)

    def __init__(
        self, max_num_tokens=1000, batch_wait_max_tokens_ratio=0.5, batch_wait_timeout_iters=3
    ):
        self.max_num_tokens = max_num_tokens
        self.batch_wait_max_tokens_ratio = batch_wait_max_tokens_ratio
        self.batch_wait_timeout_iters = batch_wait_timeout_iters
        self.batch_wait_iters_count = 0

    def _waiting_requests(self, context_requests, generation_requests):
        return PyExecutor._waiting_requests(self, context_requests, generation_requests)


class TestWaitingRequests:
    def test_no_reuse_counts_all_tokens(self):
        """Without KV cache reuse, all context tokens are counted."""
        executor = MockPyExecutorForWaiting(max_num_tokens=1000, batch_wait_max_tokens_ratio=0.5)
        # 100 tokens < 500 threshold => should wait
        ctx_reqs = [_make_ctx_request(100, estimated_reusable_tokens=0)]
        result = executor._waiting_requests(ctx_reqs, [])
        assert result == []  # waiting

    def test_reuse_reduces_token_count(self):
        """With KV cache reuse, only compute tokens are counted."""
        executor = MockPyExecutorForWaiting(max_num_tokens=1000, batch_wait_max_tokens_ratio=0.5)
        # 600 total tokens, 500 reusable => 100 compute tokens < 500 threshold
        ctx_reqs = [
            _make_ctx_request(600, estimated_reusable_tokens=500, is_first_context_chunk=True)
        ]
        result = executor._waiting_requests(ctx_reqs, [])
        assert result == []  # waiting because compute tokens = 100

    def test_reuse_not_applied_for_non_first_chunk(self):
        """Reusable tokens are ignored for non-first context chunks."""
        executor = MockPyExecutorForWaiting(max_num_tokens=1000, batch_wait_max_tokens_ratio=0.5)
        # 600 tokens, reusable=500 but is_first_context_chunk=False => counts all 600
        ctx_reqs = [
            _make_ctx_request(600, estimated_reusable_tokens=500, is_first_context_chunk=False)
        ]
        result = executor._waiting_requests(ctx_reqs, [])
        assert result == ctx_reqs  # not waiting, 600 >= 500

    def test_compute_tokens_at_least_one(self):
        """Each request contributes at least 1 compute token."""
        executor = MockPyExecutorForWaiting(max_num_tokens=1000, batch_wait_max_tokens_ratio=0.5)
        # 100 tokens, 100 reusable => max(1, 0) = 1 compute token
        ctx_reqs = [
            _make_ctx_request(100, estimated_reusable_tokens=100, is_first_context_chunk=True)
        ]
        result = executor._waiting_requests(ctx_reqs, [])
        assert result == []  # 1 token < 500, should wait
