"""Tests for PyExecutor request handling functionality.

This module tests the request handling logic that was moved from ExecutorRequestQueue
to PyExecutor, including:
- _handle_special_queue_items method
- canceled_req_ids management
- waiting_queue management
- is_shutdown state management
- expected_num_active_requests tracking
- Event loop error propagation to await_responses callers
"""

import threading
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
# Tests for event loop error propagation (_await_single_response /
# _await_any_response).
#
# We exercise the actual PyExecutor methods by importing them and binding to
# a lightweight stub that carries only the attributes those methods touch.
# ---------------------------------------------------------------------------


class _ResponseStub:
    """Minimal stub that carries only the state used by _await_*_response."""

    def __init__(self):
        self.response_lock = threading.Lock()
        self.response_cv = threading.Condition(self.response_lock)
        self.responses = {}
        self.is_shutdown = False
        self._event_loop_error = None

    # Bind the real methods from PyExecutor.
    _await_single_response = PyExecutor._await_single_response
    _await_any_response = PyExecutor._await_any_response


class TestAwaitSingleResponseShutdown:
    """_await_single_response must not block forever when the event loop dies."""

    def test_raises_on_shutdown_with_error(self):
        """After event loop crash, _await_single_response raises RuntimeError
        containing the original error instead of hanging."""
        stub = _ResponseStub()
        stub.is_shutdown = True
        stub._event_loop_error = RuntimeError("KV cache OOM")

        with pytest.raises(RuntimeError, match="Event loop terminated"):
            stub._await_single_response(id=42, timeout=1.0)

    def test_raises_on_shutdown_without_error(self):
        """Shutdown without a stored error still raises instead of blocking."""
        stub = _ResponseStub()
        stub.is_shutdown = True

        with pytest.raises(RuntimeError, match="Event loop shut down"):
            stub._await_single_response(id=42, timeout=1.0)

    def test_returns_response_when_available(self):
        """Normal path: response exists, returned immediately."""
        stub = _ResponseStub()
        stub.responses = {7: ["resp_a", "resp_b"]}

        result = stub._await_single_response(id=7, timeout=1.0)
        assert result == ["resp_a", "resp_b"]
        assert 7 not in stub.responses  # consumed

    def test_returns_response_even_during_shutdown(self):
        """If a response was enqueued before shutdown, it should be returned
        (not discarded in favour of an error)."""
        stub = _ResponseStub()
        stub.is_shutdown = True
        stub._event_loop_error = RuntimeError("crash")
        stub.responses = {7: ["resp"]}

        result = stub._await_single_response(id=7, timeout=1.0)
        assert result == ["resp"]

    def test_returns_empty_on_timeout(self):
        """If neither response nor shutdown, timeout returns empty list."""
        stub = _ResponseStub()

        result = stub._await_single_response(id=99, timeout=0.01)
        assert result == []

    def test_wakes_up_when_shutdown_set_from_another_thread(self):
        """Simulates the real scenario: main thread blocks in
        _await_single_response while the event loop thread crashes and sets
        is_shutdown."""
        stub = _ResponseStub()

        error = RuntimeError("simulated crash")

        def crash_after_delay():
            import time

            time.sleep(0.05)
            stub._event_loop_error = error
            with stub.response_cv:
                stub.is_shutdown = True
                stub.response_cv.notify_all()

        t = threading.Thread(target=crash_after_delay, daemon=True)
        t.start()

        with pytest.raises(RuntimeError, match="Event loop terminated"):
            stub._await_single_response(id=1, timeout=5.0)

        t.join(timeout=1.0)


class TestAwaitAnyResponseShutdown:
    """_await_any_response should wake up on shutdown and return empty."""

    def test_returns_empty_on_shutdown(self):
        stub = _ResponseStub()
        stub.is_shutdown = True

        result = stub._await_any_response(timeout=1.0)
        assert result == []

    def test_returns_responses_on_shutdown(self):
        """If error responses were enqueued before shutdown, they are returned."""
        stub = _ResponseStub()
        stub.is_shutdown = True
        stub.responses = {1: ["err_resp"]}

        result = stub._await_any_response(timeout=1.0)
        assert result == ["err_resp"]
