"""Tests for ExecutorRequestQueue class.

This module tests the ExecutorRequestQueue class functionality including:
- Queue initialization
- Request enqueuing (single, multiple, cancel, shutdown)
- Queue operations (get from request queue, timeout behavior)
- RequestQueueItem special types
"""

import datetime
import queue
import threading
import time
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    SHUTDOWN_REQUEST_ID, ExecutorRequestQueue, RequestQueueItem)


@pytest.fixture
def mock_dist():
    """Create a mock Distributed instance for testing."""
    mock_dist = Mock()
    mock_dist.rank = 0
    mock_dist.tp_size = 1
    mock_dist.pp_size = 1
    mock_dist.has_pp = False
    mock_dist.tp_rank = 0
    mock_dist.cp_rank = 0
    mock_dist.cp_size = 1
    mock_dist.cp_config = {}
    mock_dist.is_first_pp_rank = True
    mock_dist.is_last_pp_rank = True
    mock_dist.next_pp_rank = 1
    mock_dist.prev_pp_rank = 0
    mock_dist.broadcast = Mock(return_value=([], None))
    return mock_dist


@pytest.fixture
def executor_queue(mock_dist):
    """Create an ExecutorRequestQueue instance for testing."""
    return ExecutorRequestQueue(dist=mock_dist,
                                max_batch_size=8,
                                enable_iter_perf_stats=True,
                                batch_wait_timeout_ms=0.0)


@pytest.fixture
def integration_queue(mock_dist):
    """Create an ExecutorRequestQueue instance for integration testing."""
    return ExecutorRequestQueue(dist=mock_dist,
                                max_batch_size=4,
                                enable_iter_perf_stats=True,
                                batch_wait_timeout_ms=0.0)


def test_executor_queue_init(executor_queue, mock_dist):
    """Test ExecutorRequestQueue initialization."""
    assert executor_queue.dist == mock_dist
    assert executor_queue.next_request_id == 8
    assert executor_queue.enable_iter_perf_stats
    assert executor_queue.active
    assert isinstance(executor_queue.request_queue, queue.Queue)
    assert isinstance(executor_queue.enqueue_lock, type(threading.Lock()))


def test_enqueue_requests(executor_queue):
    """Test enqueuing multiple requests."""
    mock_requests = [Mock(), Mock(), Mock()]

    with (patch('time.time', return_value=1234.5),
          patch.object(executor_queue, '_generate_child_request_ids')):
        req_ids = executor_queue.enqueue_requests(mock_requests)  # type: ignore

    assert len(req_ids) == 3
    assert req_ids == [8, 9, 10]
    assert executor_queue.next_request_id == 11

    # Check start times were recorded
    for req_id in req_ids:
        assert req_id in executor_queue.start_times
        assert executor_queue.start_times[req_id] == 1234.5


def test_enqueue_request_single(executor_queue):
    """Test enqueuing a single request."""
    mock_request = Mock()

    with (patch('time.time', return_value=1234.5),
          patch.object(executor_queue, '_generate_child_request_ids')):
        req_id = executor_queue.enqueue_request(mock_request)

    assert req_id == 8
    assert executor_queue.next_request_id == 9
    assert req_id in executor_queue.start_times


def test_enqueue_request_with_query(executor_queue):
    """Test enqueuing a request with query data."""
    mock_request = Mock()
    query_data = [1, 2, 3, 4]
    with patch.object(executor_queue, '_generate_child_request_ids'):
        req_id = executor_queue.enqueue_request(mock_request, query=query_data)

    assert req_id == 8

    # Verify the item was enqueued with query
    item = executor_queue.request_queue.get_nowait()
    assert item.id == req_id
    assert item.request == mock_request


@pytest.mark.parametrize("n_children", [0, 1, 2])
def test_enqueue_request_with_child_ids(executor_queue, n_children):
    """Test enqueuing a request with query data."""
    mock_request = Mock()
    query_data = [1, 2, 3, 4]
    with patch(
            'tensorrt_llm._torch.pyexecutor.executor_request_queue.get_num_child_requests'
    ) as mock_children:
        mock_children.return_value = n_children
        req_id = executor_queue.enqueue_request(mock_request, query=query_data)

    assert req_id == 8

    # Verify the item was enqueued with child ids
    item = executor_queue.request_queue.get_nowait()
    assert item.id == req_id
    assert item.request == mock_request
    if n_children == 0:
        assert item.child_req_ids is None
    else:
        assert item.child_req_ids is not None
        assert len(item.child_req_ids) == n_children
        assert item.child_req_ids == list(
            range(1 + req_id, 1 + req_id + n_children))


def test_enqueue_cancel_request(executor_queue):
    """Test enqueuing a cancel request."""
    req_id = 42
    executor_queue.enqueue_cancel_request(req_id)

    item = executor_queue.request_queue.get_nowait()
    assert item.id == req_id
    assert item.request is None
    assert item.is_canceled_request


def test_enqueue_shutdown_request(executor_queue):
    """Test enqueuing a shutdown request."""
    assert executor_queue.active

    executor_queue.enqueue_shutdown_request()

    assert not executor_queue.active
    item = executor_queue.request_queue.get_nowait()
    assert item.is_shutdown_request


def test_enqueue_request_after_shutdown(executor_queue):
    """Test that enqueuing fails after shutdown."""
    executor_queue.enqueue_shutdown_request()

    with pytest.raises(AssertionError):
        executor_queue.enqueue_request(Mock())


@pytest.mark.parametrize(
    "rank,active,expected",
    [
        (0, True, True),  # rank 0 and active
        (0, False, False),  # rank 0 but not active
        (1, True, False),  # not rank 0
    ])
def test_can_enqueue_request(executor_queue, mock_dist, rank, active, expected):
    """Test can_enqueue_request method."""
    mock_dist.rank = rank
    executor_queue.active = active

    assert executor_queue.can_enqueue_request() == expected


def test_get_from_request_queue_no_timeout(executor_queue):
    """Test getting items from request queue without timeout."""
    # Add some items
    item1 = RequestQueueItem(1, Mock())
    item2 = RequestQueueItem(2, Mock())
    executor_queue.request_queue.put(item1)
    executor_queue.request_queue.put(item2)

    items = executor_queue.get_from_request_queue(None)

    assert len(items) == 2
    assert items[0] == item1
    assert items[1] == item2


def test_get_from_request_queue_with_timeout(executor_queue):
    """Test getting items from request queue with timeout."""
    timeout = datetime.timedelta(seconds=0.1)

    # Empty queue should return empty list quickly
    start_time = time.time()
    items = executor_queue.get_from_request_queue(timeout)
    elapsed = time.time() - start_time

    assert len(items) == 0
    assert elapsed < 0.2  # Should finish within timeout


def test_get_from_request_queue_async_behavior(executor_queue):
    """Test asynchronous behavior where requests arrive over time."""
    import threading

    def add_requests_after_delay(delay, num_requests):
        """Helper function to add requests after a delay."""
        time.sleep(delay)
        for i in range(num_requests):
            item = RequestQueueItem(i + 10, Mock())
            executor_queue.request_queue.put(item)

    # Test 1: Without batch_wait_timeout_ms (should only get initial requests)
    executor_queue.batch_wait_timeout_ms = 0.0

    initial_requests = 3
    for i in range(initial_requests):
        item = RequestQueueItem(i, Mock())
        executor_queue.request_queue.put(item)

    thread = threading.Thread(target=add_requests_after_delay, args=(0.05, 2))
    thread.start()

    # Get requests immediately - should only get the initial ones
    start_time = time.time()
    items = executor_queue.get_from_request_queue(None)
    elapsed = time.time() - start_time

    assert len(items) == initial_requests
    assert elapsed < 0.1
    assert all(item.id < 10 for item in items)

    thread.join()

    # Test 2: With batch_wait_timeout_ms (should wait and get all requests)
    executor_queue.batch_wait_timeout_ms = 200.0

    # Clear the queue and add initial requests again
    while not executor_queue.request_queue.empty():
        try:
            executor_queue.request_queue.get_nowait()
        except queue.Empty:
            break

    initial_requests = 2
    for i in range(initial_requests):
        item = RequestQueueItem(i + 20, Mock())
        executor_queue.request_queue.put(item)

    thread = threading.Thread(target=add_requests_after_delay, args=(0.05, 3))
    thread.start()

    # Get requests with batch_wait_timeout_ms - should wait and get all
    start_time = time.time()
    items = executor_queue.get_from_request_queue(None)
    elapsed = time.time() - start_time

    # Should wait and return all requests
    assert len(items) == initial_requests + 3
    assert elapsed >= 0.05
    assert elapsed < 0.3

    initial_ids = {item.id for item in items if 20 <= item.id < 30}
    delayed_ids = {item.id for item in items if 10 <= item.id < 20}
    assert len(initial_ids) == initial_requests
    assert len(delayed_ids) == 3

    thread.join()


def test_request_queue_item_special_types():
    """Test RequestQueueItem special type detection."""
    # Create a mock request without sampling_config to avoid beam validation
    mock_request = Mock()
    delattr(mock_request, 'sampling_config') if hasattr(
        mock_request, 'sampling_config') else None

    normal_req = RequestQueueItem(1, mock_request)
    cancel_req = RequestQueueItem(2, is_canceled_request=True)
    shutdown_req = RequestQueueItem(SHUTDOWN_REQUEST_ID)

    # Test normal request
    assert normal_req.is_normal_request
    assert not normal_req.is_shutdown_request
    assert not normal_req.is_canceled_request

    # Test cancel request
    assert cancel_req.is_canceled_request
    assert not cancel_req.is_shutdown_request
    assert not cancel_req.is_normal_request

    # Test shutdown request
    assert shutdown_req.is_shutdown_request
    assert not shutdown_req.is_canceled_request
    assert not shutdown_req.is_normal_request


def test_queue_size_methods(executor_queue):
    """Test queue size getter methods."""
    # Test initial values
    assert executor_queue.get_request_queue_size() == 0

    # Add some data and test
    executor_queue.request_queue.put(RequestQueueItem(1, Mock()))
    assert executor_queue.get_request_queue_size() == 1
