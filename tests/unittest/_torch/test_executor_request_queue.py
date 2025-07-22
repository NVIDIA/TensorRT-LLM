import datetime
import queue
import threading
import time
from collections import deque
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
                                enable_attention_dp=False,
                                max_batch_size=8,
                                max_beam_width=1,
                                max_num_active_requests=16,
                                enable_iter_perf_stats=True,
                                is_disaggregated=False)


@pytest.fixture
def integration_queue(mock_dist):
    """Create an ExecutorRequestQueue instance for integration testing."""
    return ExecutorRequestQueue(dist=mock_dist,
                                enable_attention_dp=True,
                                max_batch_size=4,
                                max_beam_width=2,
                                max_num_active_requests=8,
                                enable_iter_perf_stats=True,
                                is_disaggregated=False)


def test_executor_queue_init(executor_queue, mock_dist):
    """Test ExecutorRequestQueue initialization."""
    assert executor_queue.dist == mock_dist
    assert not executor_queue.enable_attention_dp
    assert executor_queue.max_beam_width == 1
    assert executor_queue.max_num_active_requests == 16
    assert not executor_queue.is_disaggregated
    assert executor_queue.next_request_id == 8
    assert executor_queue.enable_iter_perf_stats
    assert executor_queue.active
    assert isinstance(executor_queue.request_queue, queue.Queue)
    assert isinstance(executor_queue.waiting_queue, deque)
    assert len(executor_queue.canceled_req_ids) == 0
    assert isinstance(executor_queue.enqueue_lock, type(threading.Lock()))


def test_enqueue_requests(executor_queue):
    """Test enqueuing multiple requests."""
    mock_requests = [Mock(), Mock(), Mock()]

    with patch('time.time', return_value=1234.5):
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

    with patch('time.time', return_value=1234.5):
        req_id = executor_queue.enqueue_request(mock_request)

    assert req_id == 8
    assert executor_queue.next_request_id == 9
    assert req_id in executor_queue.start_times


def test_enqueue_request_with_query(executor_queue):
    """Test enqueuing a request with query data."""
    mock_request = Mock()
    query_data = [1, 2, 3, 4]

    req_id = executor_queue.enqueue_request(mock_request, query=query_data)

    assert req_id == 8

    # Verify the item was enqueued with query
    item = executor_queue.request_queue.get_nowait()
    assert item.id == req_id
    assert item.request == mock_request


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

    items = executor_queue._get_from_request_queue(None)

    assert len(items) == 2
    assert items[0] == item1
    assert items[1] == item2


def test_get_from_request_queue_with_timeout(executor_queue):
    """Test getting items from request queue with timeout."""
    timeout = datetime.timedelta(seconds=0.1)

    # Empty queue should return empty list quickly
    start_time = time.time()
    items = executor_queue._get_from_request_queue(timeout)
    elapsed = time.time() - start_time

    assert len(items) == 0
    assert elapsed < 0.2  # Should finish within timeout


def test_get_from_waiting_queue(executor_queue):
    """Test getting items from waiting queue."""
    # Add items to waiting queue
    items = [RequestQueueItem(i, Mock()) for i in range(5)]
    executor_queue.waiting_queue.extend(items)

    # Get 3 items
    result = executor_queue._get_from_waiting_queue(
        executor_queue.waiting_queue, 3)

    assert len(result) == 3
    assert result == items[:3]
    assert len(executor_queue.waiting_queue) == 2


@pytest.mark.parametrize(
    "queue_size,request_count,expected_result,expected_remaining",
    [
        (0, 5, 0, 0),  # Empty queue
        (3, -1, 0, 3),  # Negative count
        (3, 0, 0, 3),  # Zero count
        (3, 10, 3, 0),  # Request more than available
    ])
def test_get_from_waiting_queue_edge_cases(executor_queue, queue_size,
                                           request_count, expected_result,
                                           expected_remaining):
    """Test edge cases for getting items from waiting queue."""
    # Setup queue
    if queue_size > 0:
        items = [RequestQueueItem(i, Mock()) for i in range(queue_size)]
        executor_queue.waiting_queue.extend(items)

    result = executor_queue._get_from_waiting_queue(
        executor_queue.waiting_queue, request_count)

    assert len(result) == expected_result
    assert len(executor_queue.waiting_queue) == expected_remaining


def test_validate_and_filter_requests(executor_queue):
    """Test request validation and filtering."""
    # Create a mock request without sampling_config to avoid beam validation
    mock_request = Mock()
    delattr(mock_request, 'sampling_config') if hasattr(
        mock_request, 'sampling_config') else None

    normal_req = RequestQueueItem(1, mock_request)
    cancel_req = RequestQueueItem(2, is_canceled_request=True)
    shutdown_req = RequestQueueItem(SHUTDOWN_REQUEST_ID)

    requests = [normal_req, cancel_req, shutdown_req]

    valid_requests = executor_queue._validate_and_filter_requests(requests)

    assert len(valid_requests) == 1
    assert valid_requests[0] == normal_req
    assert executor_queue.is_shutdown
    assert 2 in executor_queue.canceled_req_ids


@patch(
    'tensorrt_llm._torch.pyexecutor.executor_request_queue.executor_request_to_llm_request'
)
def test_merge_requests_default(mock_convert, executor_queue):
    """Test merging requests with default configuration."""
    mock_llm_request = Mock()
    mock_convert.return_value = mock_llm_request

    requests = [RequestQueueItem(1, Mock()), RequestQueueItem(2, Mock())]

    result = executor_queue._merge_requests(requests)

    assert len(result) == 2
    assert mock_convert.call_count == 2


def test_update_waiting_queue(executor_queue):
    """Test updating waiting queue to remove canceled requests."""
    items = [
        RequestQueueItem(1, Mock()),
        RequestQueueItem(2, Mock()),
        RequestQueueItem(3, Mock()),
    ]
    executor_queue.waiting_queue.extend(items)
    executor_queue.canceled_req_ids = [2]

    executor_queue.update_waiting_queue()

    assert len(executor_queue.waiting_queue) == 2
    remaining_ids = [item.id for item in executor_queue.waiting_queue]
    assert 1 in remaining_ids
    assert 3 in remaining_ids
    assert 2 not in remaining_ids


def test_performance_metrics_methods(executor_queue):
    """Test various performance metrics getter methods."""
    # Test initial values
    assert executor_queue.get_new_active_requests_queue_latency() == 0
    assert executor_queue.get_expected_num_active_requests() == 0
    assert executor_queue.get_request_queue_size() == 0
    assert executor_queue.get_waiting_queue_size() == 0
    assert executor_queue.get_canceled_req_ids_size() == 0
    assert executor_queue.get_canceled_req_ids() == []

    # Add some data and test
    executor_queue.request_queue.put(RequestQueueItem(1, Mock()))
    executor_queue.waiting_queue.append(RequestQueueItem(2, Mock()))
    executor_queue.canceled_req_ids = [3, 4]
    executor_queue.expected_num_active_requests = 5

    assert executor_queue.get_request_queue_size() == 1
    assert executor_queue.get_waiting_queue_size() == 1
    assert executor_queue.get_canceled_req_ids_size() == 2
    assert executor_queue.get_canceled_req_ids() == [3, 4]
    assert executor_queue.get_expected_num_active_requests() == 5


def test_clear_canceled_req_ids(executor_queue):
    """Test clearing canceled request IDs."""
    executor_queue.canceled_req_ids = [1, 2, 3]
    assert len(executor_queue.canceled_req_ids) == 3

    executor_queue.clear_canceled_req_ids()

    assert len(executor_queue.canceled_req_ids) == 0


def test_thread_safety(executor_queue):
    """Test thread safety of enqueue operations."""
    results = []
    errors = []

    def enqueue_worker():
        try:
            for i in range(10):
                req_id = executor_queue.enqueue_request(Mock())
                results.append(req_id)
        except Exception as e:
            errors.append(e)

    # Create multiple threads
    threads = []
    for _ in range(3):
        thread = threading.Thread(target=enqueue_worker)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Check results
    assert len(errors) == 0
    assert len(results) == 30
    assert len(set(results)) == 30  # All IDs should be unique


@patch('tensorrt_llm._torch.pyexecutor.executor_request_queue.time.time')
def test_update_new_active_requests_queue_latency(mock_time, executor_queue):
    """Test updating queue latency metrics."""
    mock_time.return_value = 1000.0

    # Set up start times
    executor_queue.start_times = {1: 998.0, 2: 999.0}

    requests = [RequestQueueItem(1, Mock()), RequestQueueItem(2, Mock())]

    executor_queue._update_new_active_requests_queue_latency(requests)

    # Check latency was updated (1000.0 - 998.0) + (1000.0 - 999.0) = 3.0
    assert executor_queue.new_active_requests_queue_latency_ms == 3.0

    # Check start times were removed
    assert len(executor_queue.start_times) == 0


@pytest.mark.parametrize("enable_attention_dp", [False, True])
def test_fetch_new_requests_routing(executor_queue, enable_attention_dp):
    """Test that fetch_new_requests routes correctly based on attention_dp setting."""
    mock_active_requests = []
    executor_queue.enable_attention_dp = enable_attention_dp

    if enable_attention_dp:
        with patch.object(executor_queue,
                          '_fetch_new_requests_attention_dp') as mock_dp:
            mock_dp.return_value = []
            executor_queue.fetch_new_requests(len(mock_active_requests))
            mock_dp.assert_called_once_with(len(mock_active_requests))
    else:
        with patch.object(executor_queue,
                          '_fetch_new_requests_attention_tp') as mock_tp:
            mock_tp.return_value = []
            executor_queue.fetch_new_requests(len(mock_active_requests))
            mock_tp.assert_called_once_with(len(mock_active_requests))


# Integration tests
def test_full_workflow(integration_queue):
    """Test a complete workflow from enqueue to processing."""
    # Enqueue some requests - create mocks without sampling_config to avoid beam validation
    mock_requests = []
    for _ in range(3):
        mock_req = Mock()
        delattr(mock_req, 'sampling_config') if hasattr(
            mock_req, 'sampling_config') else None
        mock_requests.append(mock_req)
    req_ids = integration_queue.enqueue_requests(mock_requests)  # type: ignore

    # Enqueue a cancel request
    integration_queue.enqueue_cancel_request(req_ids[1])

    # Simulate fetching from request queue
    items = []
    while not integration_queue.request_queue.empty():
        try:
            items.append(integration_queue.request_queue.get_nowait())
        except queue.Empty:
            break

    assert len(items) == 4  # 3 requests + 1 cancel

    # Filter and validate
    valid_items = integration_queue._validate_and_filter_requests(items)

    assert len(valid_items) == 3
    assert req_ids[1] in integration_queue.canceled_req_ids


@patch(
    'tensorrt_llm._torch.pyexecutor.executor_request_queue.executor_request_to_llm_request'
)
def test_merge_requests_with_beam_validation(mock_convert, integration_queue):
    """Test request merging with beam width validation."""
    # Create mock requests with different beam widths
    mock_req1 = Mock()
    mock_req1.sampling_config = Mock()
    mock_req1.sampling_config.beam_width = 2  # Matches max_beam_width

    mock_req2 = Mock()
    mock_req2.sampling_config = Mock()
    mock_req2.sampling_config.beam_width = 3  # Doesn't match max_beam_width

    requests = [RequestQueueItem(1, mock_req1), RequestQueueItem(2, mock_req2)]

    # First request should pass validation
    valid_requests = integration_queue._validate_and_filter_requests(
        [requests[0]])
    assert len(valid_requests) == 1

    # Second request should fail validation
    with pytest.raises(AssertionError):
        integration_queue._validate_and_filter_requests([requests[1]])


def test_beam_width_validation_success(integration_queue):
    """Test that beam width validation passes for correct beam width."""
    mock_req = Mock()
    mock_req.sampling_config = Mock()
    mock_req.sampling_config.beam_width = 2  # Matches integration test max_beam_width

    request = RequestQueueItem(1, mock_req)
    valid_requests = integration_queue._validate_and_filter_requests([request])

    assert len(valid_requests) == 1
    assert valid_requests[0] == request
