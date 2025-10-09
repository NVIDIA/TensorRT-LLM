import datetime
import queue
import threading
import time
from collections import deque
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    SHUTDOWN_REQUEST_ID, ExecutorRequestQueue, RequestQueueItem)
from tensorrt_llm.bindings import executor as trtllm
from tensorrt_llm.mapping import CpType


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
                                batch_wait_timeout_ms=0.0,
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
                                batch_wait_timeout_ms=0.0,
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


def test_merge_helix_requests_with_padding(mock_dist):
    """Test _merge_helix_requests with basic valid input."""

    tokens_per_block = 2

    # Create request item with 13 tokens to get exactly 7 blocks for 4 CP ranks.
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    executor_request = trtllm.Request(input_token_ids=input_tokens,
                                      max_tokens=5,
                                      streaming=False,
                                      sampling_config=trtllm.SamplingConfig(),
                                      output_config=trtllm.OutputConfig())
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    for rank in [0, 1, 2, 3]:
        # Create executor queue for helix with 4 CP ranks.
        mock_dist.cp_size = 4
        mock_dist.cp_rank = rank
        mock_dist.cp_config = {
            'cp_type': CpType.HELIX,
            'tokens_per_block': tokens_per_block,
        }
        executor_queue = ExecutorRequestQueue(dist=mock_dist,
                                              enable_attention_dp=False,
                                              max_batch_size=8,
                                              max_beam_width=1,
                                              max_num_active_requests=16,
                                              enable_iter_perf_stats=True,
                                              batch_wait_timeout_ms=0.0,
                                              is_disaggregated=True)

        # Mock _should_exclude_last_generation_logits.
        with patch.object(executor_queue,
                          '_should_exclude_last_generation_logits',
                          return_value=False):
            result = executor_queue._merge_helix_requests([request_item],
                                                          tokens_per_block)

        # Verify the result.
        assert len(result) == 1
        llm_request = result[0]
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
        assert isinstance(llm_request, LlmRequest)
        assert llm_request.request_id == 1
        if rank == 0:
            assert llm_request.get_tokens(0) == [1, 2, 3, 4]
        elif rank == 1:
            assert llm_request.get_tokens(0) == [5, 6, 7, 8]
        elif rank == 2:
            assert llm_request.get_tokens(0) == [9, 10, 11, 12]
        else:
            assert llm_request.get_tokens(0) == [13]


def test_merge_helix_requests_without_padding(mock_dist):
    """Test _merge_helix_requests with evenly divisible tokens (no padding)."""

    tokens_per_block = 4

    # Create request item with 12 tokens to get exactly 3 blocks for 2 CP ranks.
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    executor_request = trtllm.Request(input_token_ids=input_tokens,
                                      max_tokens=5,
                                      streaming=False,
                                      sampling_config=trtllm.SamplingConfig(),
                                      output_config=trtllm.OutputConfig())
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    for rank in [0, 1]:
        # Create executor queue for helix with 2 CP ranks.
        mock_dist.cp_size = 2
        mock_dist.cp_rank = rank
        mock_dist.cp_config = {
            'cp_type': CpType.HELIX,
            'tokens_per_block': tokens_per_block,
        }
        executor_queue = ExecutorRequestQueue(dist=mock_dist,
                                              enable_attention_dp=False,
                                              max_batch_size=8,
                                              max_beam_width=1,
                                              max_num_active_requests=16,
                                              enable_iter_perf_stats=True,
                                              batch_wait_timeout_ms=0.0,
                                              is_disaggregated=True)

        # Mock _should_exclude_last_generation_logits.
        with patch.object(executor_queue,
                          '_should_exclude_last_generation_logits',
                          return_value=False):
            result = executor_queue._merge_helix_requests([request_item],
                                                          tokens_per_block)

        # Verify the result.
        assert len(result) == 1
        llm_request = result[0]
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
        assert isinstance(llm_request, LlmRequest)
        assert llm_request.request_id == 1
        if rank == 0:
            assert llm_request.get_tokens(0) == [1, 2, 3, 4, 5, 6, 7, 8]
        else:
            assert llm_request.get_tokens(0) == [9, 10, 11, 12]


def test_merge_helix_requests_insufficient_blocks_error(mock_dist):
    """Test _merge_helix_requests raises error when insufficient blocks."""
    mock_dist.cp_size = 4

    tokens_per_block = 4
    mock_dist.cp_config = {
        'cp_type': CpType.HELIX,
        'tokens_per_block': tokens_per_block,
    }

    # Create input with only 12 tokens. This creates 3 blocks which is fewer than 4 CP ranks.
    executor_request = trtllm.Request(
        input_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        max_tokens=12,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig())
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    # Loop over ranks 0, 1, 2, 3 and verify that all ranks throw assertion.
    for rank in range(4):
        mock_dist.cp_rank = rank

        executor_queue = ExecutorRequestQueue(dist=mock_dist,
                                              enable_attention_dp=False,
                                              max_batch_size=8,
                                              max_beam_width=1,
                                              max_num_active_requests=16,
                                              enable_iter_perf_stats=True,
                                              batch_wait_timeout_ms=0.0,
                                              is_disaggregated=True)

        with pytest.raises(
                ValueError,
                match=
                "There aren't enough tokens to get at least one block per CP rank"
        ):
            executor_queue._merge_helix_requests([request_item],
                                                 tokens_per_block)


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
    with patch.object(executor_queue,
                      '_get_num_child_requests') as mock_children:
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
    items = executor_queue._get_from_request_queue(None)
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
    items = executor_queue._get_from_request_queue(None)
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


def test_get_from_waiting_queue(executor_queue):
    """Test getting items from waiting queue."""
    # Add items to waiting queue
    items = [RequestQueueItem(i, Mock()) for i in range(5)]
    executor_queue.waiting_queue.extend(items)

    # Get 3 items
    result = executor_queue._get_from_waiting_queue(
        executor_queue.waiting_queue, 3, enable_attention_dp=False)

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
        executor_queue.waiting_queue, request_count, enable_attention_dp=False)

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
    mock_llm_request = Mock(child_requests=[])
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


@pytest.fixture
def mock_dist_attention_dp():
    """Create a mock Distributed instance for testing."""
    mock_dist = Mock()
    mock_dist.rank = 0
    mock_dist.tp_size = 4
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
def attention_dp_queue(mock_dist_attention_dp):
    """Create an ExecutorRequestQueue instance for attention DP testing."""
    queue = ExecutorRequestQueue(dist=mock_dist_attention_dp,
                                 enable_attention_dp=True,
                                 max_batch_size=4,
                                 max_beam_width=2,
                                 max_num_active_requests=8,
                                 enable_iter_perf_stats=True,
                                 batch_wait_timeout_ms=0.0,
                                 is_disaggregated=False)
    # Initialize all_ranks_num_active_requests
    return queue


@pytest.fixture
def all_ranks_num_active_requests():
    return [2, 1, 3, 0]  # 4 ranks


@pytest.fixture
def all_ranks_num_active_tokens():
    return [10, 5, 15, 8]  # 4 ranks


def create_mock_request_with_py_schedule_params(attention_dp_rank=None,
                                                attention_dp_relax=False):
    mock_request = Mock()

    if attention_dp_rank is not None:
        mock_schedule_params = Mock()
        mock_schedule_params.attention_dp_rank = attention_dp_rank
        mock_schedule_params.attention_dp_relax = attention_dp_relax

        mock_schedule_params.configure_mock(
            attention_dp_rank=attention_dp_rank,
            attention_dp_relax=attention_dp_relax)

        mock_request.py_scheduling_params = mock_schedule_params
    else:
        mock_request.py_scheduling_params = None

    mock_request.input_token_ids = [1, 2, 3]

    return mock_request


# Unit tests for _schedule_attention_dp_requests
def test_schedule_attention_dp_requests_scheduled_requests(
        attention_dp_queue, all_ranks_num_active_requests,
        all_ranks_num_active_tokens):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))

    new_requests = [req1, req2]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)
    result = all_ranks_new_requests[0]

    assert len(result) == 2
    assert req1 in result
    assert req2 in result

    assert all_ranks_num_active_requests[0] == 4


def test_schedule_attention_dp_requests_scheduled_requests_other_ranks(
        attention_dp_queue, all_ranks_num_active_requests,
        all_ranks_num_active_tokens):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1,
                                                    attention_dp_relax=False))
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=2,
                                                    attention_dp_relax=False))

    new_requests = [req1, req2]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)

    result = all_ranks_new_requests[0]
    assert len(result) == 0

    assert all_ranks_num_active_requests[1] == 2
    assert all_ranks_num_active_requests[2] == 4


def test_schedule_attention_dp_requests_unscheduled_requests(
        attention_dp_queue, all_ranks_num_active_requests,
        all_ranks_num_active_tokens):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1,
                                                    attention_dp_relax=True))

    new_requests = [req1, req2]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)
    result = all_ranks_new_requests[0]

    assert len(result) == 1  # Only req1 for current rank
    assert req1 in result


def test_schedule_attention_dp_requests_unscheduled_no_capacity(
        attention_dp_queue, all_ranks_num_active_requests,
        all_ranks_num_active_tokens):
    all_ranks_num_active_requests[0] = 8

    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))

    new_requests = [req1]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)
    result = all_ranks_new_requests[0]

    assert len(result) == 0  # No capacity


def test_schedule_attention_dp_requests_mixed_scenarios(
        attention_dp_queue, all_ranks_num_active_requests,
        all_ranks_num_active_tokens):
    req_scheduled_current = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    req_scheduled_other = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1,
                                                    attention_dp_relax=False))
    req_unscheduled_current = RequestQueueItem(
        3,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))
    req_unscheduled_other = RequestQueueItem(
        4,
        create_mock_request_with_py_schedule_params(attention_dp_rank=2,
                                                    attention_dp_relax=True))

    new_requests = [
        req_scheduled_current, req_scheduled_other, req_unscheduled_current,
        req_unscheduled_other
    ]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)
    result = all_ranks_new_requests[0]

    assert len(result) == 2
    assert req_scheduled_current in result
    assert req_unscheduled_current in result


def test_schedule_attention_dp_requests_empty_lists(
        attention_dp_queue, all_ranks_num_active_requests,
        all_ranks_num_active_tokens):
    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        [], all_ranks_num_active_requests, all_ranks_num_active_tokens)
    result = all_ranks_new_requests[0]

    assert len(result) == 0


def test_schedule_attention_dp_requests_expected_num_active_calculation(
        attention_dp_queue, all_ranks_num_active_requests,
        all_ranks_num_active_tokens):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1,
                                                    attention_dp_relax=True))

    new_requests = [req1, req2]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)
    all_ranks_new_requests[0]

    # 2 + 1 + 3 + 0 = 6, 6 + 2 = 8, (8 + 3) // 4 = 2, max(2, 2, 1, 3, 0) = 3
    # expected_num_active_requests = max((6 + 2 + 3) // 4, 3) = max(2, 3) = 3
    assert attention_dp_queue.expected_num_active_requests == 3


def test_schedule_attention_dp_requests_balance_requests_called(
        attention_dp_queue, all_ranks_num_active_requests,
        all_ranks_num_active_tokens):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))

    new_requests = [req1]

    with patch.object(attention_dp_queue,
                      '_balance_requests_across_ranks') as mock_balance:
        mock_balance.return_value = {0: req1}

        all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
            new_requests, all_ranks_num_active_requests,
            all_ranks_num_active_tokens)
        all_ranks_new_requests[0]

    # Check that _balance_requests_across_ranks was called
    mock_balance.assert_called_once()
    call_args = mock_balance.call_args[0]
    assert isinstance(call_args[0], list)
    assert isinstance(call_args[1], dict)
    assert call_args[2] == all_ranks_num_active_requests  # Third arg
    assert call_args[3] == all_ranks_num_active_tokens  # Fourth arg


def test_schedule_attention_dp_requests_no_scheduling_when_capacity_exceeded(
        attention_dp_queue, all_ranks_num_active_requests,
        all_ranks_num_active_tokens):
    all_ranks_num_active_requests[0] = 8

    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))

    new_requests = [req1]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)
    result = all_ranks_new_requests[0]

    assert len(result) == 0  # No requests scheduled
    assert all_ranks_num_active_requests[0] == 8  # Capacity unchanged


# Integration tests combining both methods
def test_filter_and_schedule_integration(attention_dp_queue,
                                         all_ranks_num_active_requests,
                                         all_ranks_num_active_tokens):
    req_schedulable = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    req_schedulable.request.input_token_ids = [1, 2, 3, 4]
    req_relax = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))
    req_relax.request.input_token_ids = [1, 2]

    req_no_params = RequestQueueItem(
        3, create_mock_request_with_py_schedule_params(attention_dp_rank=None))

    new_requests = [req_schedulable, req_relax, req_no_params]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)
    result = all_ranks_new_requests[0]

    assert len(result) == 2
    assert req_schedulable in result
    assert req_relax in result


def test_filter_and_schedule_with_capacity_limits(attention_dp_queue,
                                                  all_ranks_num_active_requests,
                                                  all_ranks_num_active_tokens):
    all_ranks_num_active_requests[0] = 7

    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    req1.request.input_token_ids = [1, 2, 3, 4]
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    req2.request.input_token_ids = [1, 2, 3]

    new_requests = [req1, req2]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)
    result = all_ranks_new_requests[0]

    assert len(result) == 1
    assert req1 in result


def test_get_from_waiting_queue_with_attention_dp(
        attention_dp_queue, all_ranks_num_active_requests):
    items = [RequestQueueItem(i, Mock()) for i in range(5)]
    attention_dp_queue.waiting_queue.extend(items)

    result = attention_dp_queue._get_from_waiting_queue(
        attention_dp_queue.waiting_queue, 3, True,
        all_ranks_num_active_requests)

    assert len(result) == 3
    assert result == items[:3]
    assert len(attention_dp_queue.waiting_queue) == 2


def test_get_from_waiting_queue_with_attention_dp_filtering(
        attention_dp_queue, all_ranks_num_active_requests):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1,
                                                    attention_dp_relax=True))
    req3 = RequestQueueItem(3,
                            create_mock_request_with_py_schedule_params(
                                attention_dp_rank=None))  # No scheduling params

    attention_dp_queue.waiting_queue.extend([req1, req2, req3])

    # Set rank 0 to full capacity to test filtering
    all_ranks_num_active_requests[0] = 8

    result = attention_dp_queue._get_from_waiting_queue(
        attention_dp_queue.waiting_queue, 3, True,
        all_ranks_num_active_requests)

    assert len(result) == 2
    assert req2 in result
    assert req3 in result
    assert req1 not in result


def test_can_process_attention_dp_request(attention_dp_queue):
    req_no_params = RequestQueueItem(1, Mock())
    assert attention_dp_queue._can_process_attention_dp_request(
        req_no_params, [0, 0, 0, 0]) == True

    req_relax = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))
    assert attention_dp_queue._can_process_attention_dp_request(
        req_relax, [0, 0, 0, 0]) == True

    req_target = RequestQueueItem(
        3,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1,
                                                    attention_dp_relax=False))
    all_ranks = [0, 0, 0, 0]
    assert attention_dp_queue._can_process_attention_dp_request(
        req_target, all_ranks) == True
    assert all_ranks[1] == 1

    req_no_capacity = RequestQueueItem(
        4,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    all_ranks_full = [8, 0, 0, 0]  # Rank 0 is at capacity
    assert attention_dp_queue._can_process_attention_dp_request(
        req_no_capacity, all_ranks_full) == False


def test_achieve_max_num_active_requests(attention_dp_queue):
    req_list = []
    req_id = 0
    for rank in range(4):
        for i in range(5):
            req_list.append(
                RequestQueueItem(
                    req_id,
                    create_mock_request_with_py_schedule_params(
                        attention_dp_rank=rank, attention_dp_relax=False)))
            req_id += 1
            req_list.append(
                RequestQueueItem(
                    req_id,
                    create_mock_request_with_py_schedule_params(
                        attention_dp_rank=rank, attention_dp_relax=True)))
            req_id += 1

    all_ranks_num_active_requests = [5, 6, 3, 7]
    attention_dp_queue.waiting_queue.extend(req_list)
    available_active_requests = attention_dp_queue.max_num_active_requests * 4 - sum(
        all_ranks_num_active_requests)

    result = attention_dp_queue._get_from_waiting_queue(
        attention_dp_queue.waiting_queue, available_active_requests, True,
        all_ranks_num_active_requests)

    assert len(result) == available_active_requests


def append_to_waiting_queue(waiting_queue, rank, attention_dp_relax):
    req_id = len(waiting_queue)
    waiting_queue.append(
        RequestQueueItem(
            req_id,
            create_mock_request_with_py_schedule_params(
                attention_dp_rank=rank, attention_dp_relax=attention_dp_relax)))


@pytest.mark.parametrize(
    "max_num_active_requests,all_ranks_num_active_requests,request_configs,all_ranks_expected_req_ids",
    [
        # Case: Balanced distribution of relaxed requests
        (
            3,
            [0, 0, 0, 0],
            [(None, True)] * 7,
            {
                0: [0, 1],  # First 2 requests go to rank 0
                1: [2, 3],  # Next 2 requests go to rank 1
                2: [4, 5],  # Next 2 requests go to rank 2
                3: [6]  # Last request goes to rank 3
            }),
        # Case: Balanced distribution of relaxed requests with existing load
        (
            3,
            [1, 2, 3, 0],
            [(None, True)] * 13,
            {
                0: [0, 1],  # Rank 0 gets first 2 requests
                1: [2],  # Rank 1 gets 1 request (already has 2)
                2: [],  # Rank 2 is at capacity (3)
                3: [3, 4, 5]  # Rank 3 gets 3 requests (starts with 0)
            }),
        # Case: Limited by max active
        (
            3,
            [0, 0, 0, 0],
            [(None, True)] * 13,
            {
                0: [0, 1, 3],  # First 3 requests (0, 1, 3)
                1: [2, 4, 6],  # Next 3 requests (2, 4, 6)
                2: [5, 7, 9],  # Next 3 requests (5, 7, 9)
                3: [8, 10, 11]  # Last 3 requests (8, 10, 11)
            }),
        # Case: Empty new requests
        (3, [3, 3, 3, 0], [], {
            0: [],
            1: [],
            2: [],
            3: []
        }),
        # Case: Rank 0 is full and cannot schedule attention_dp rank request
        (
            3,
            [3, 1, 3, 0],
            [(0, False), (0, True)],
            {
                0: [],  # Rank 0 is full
                1: [1],  # Rank 1 gets the relaxed request (req1)
                2: [],  # No relaxed requests assigned here
                3: []  # No relaxed requests assigned here
            }),
        # Case: Only room for 1 request, need to skip req0 with attention dp rank
        (
            3,
            [3, 2, 3, 3],
            [(0, False), (0, True)],
            {
                0: [],  # Rank 0 is full
                1: [1],  # Rank 1 gets the relaxed request
                2: [],  # Rank 2 is at capacity
                3: []  # Rank 3 is at capacity
            }),
        # Case: Targeting ranks 1 and 3 that have room
        (
            3,
            [2, 1, 3, 0],
            [(1, False), (3, False)],
            {
                0: [],  # No requests assigned to rank 0
                1: [0],  # Request 0 targets rank 1
                2: [],  # No requests assigned to rank 2
                3: [1]  # Request 1 targets rank 3
            }),
        # Case: Target dp rank specified, but relax is True
        (
            3,
            [3, 3, 3, 1],
            [(0, True), (1, True), (2, True)],
            {
                0: [],  # Rank 0 is at capacity
                1: [],  # Rank 1 is at capacity
                2: [],  # Rank 2 is at capacity
                3: [0, 1]  # Rank 3 gets both relaxed requests
            }),
        # Case: Mixed targeting and relaxed
        (
            3,
            [3, 3, 3, 0],
            [(0, False), (1, True), (3, False)],
            {
                0: [],  # Rank 0 is at capacity
                1: [],  # Rank 1 is at capacity
                2: [],  # Rank 2 is at capacity
                3: [2, 1]  # Rank 3 gets both requests (targeted + relaxed)
            }),
    ])
def test_attention_dp_scheduling_cases(attention_dp_queue,
                                       max_num_active_requests,
                                       all_ranks_num_active_requests,
                                       request_configs,
                                       all_ranks_expected_req_ids):
    """Test attention DP scheduling with various scenarios."""
    attention_dp_queue.max_num_active_requests = max_num_active_requests

    waiting_queue = deque()
    for rank, relax in request_configs:
        append_to_waiting_queue(waiting_queue, rank, relax)

    run_test_attention_dp_scheduling(attention_dp_queue, waiting_queue,
                                     all_ranks_num_active_requests,
                                     all_ranks_expected_req_ids)


def run_test_attention_dp_scheduling(attention_dp_queue, waiting_queue,
                                     all_ranks_num_active_requests,
                                     all_ranks_expected_req_ids):

    num_ranks = len(all_ranks_num_active_requests)
    total_num_active_requests = sum(all_ranks_num_active_requests)
    total_max_num_active_requests = attention_dp_queue.max_num_active_requests * num_ranks
    enable_attention_dp = True

    new_requests = attention_dp_queue._get_from_waiting_queue(
        waiting_queue,
        total_max_num_active_requests - total_num_active_requests,
        enable_attention_dp, all_ranks_num_active_requests)

    # Create mock token counts for testing
    all_ranks_num_active_tokens = [10 + i * 5 for i in range(num_ranks)]

    # Schedule attention dp requests
    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)

    assert len(all_ranks_new_requests) == num_ranks
    print("all_ranks_new_requests:", all_ranks_new_requests)
    for rank, reqs in all_ranks_new_requests.items():
        req_ids = [req.id for req in reqs]
        assert req_ids == all_ranks_expected_req_ids[rank]


# New tests for _balance_requests_across_ranks method
def test_balance_requests_across_ranks_empty_requests(attention_dp_queue):
    """Test _balance_requests_across_ranks with empty requests list."""
    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [2, 1, 3, 0]
    all_ranks_num_active_tokens = [20, 10, 30, 5]

    # Set expected_num_active_requests for testing
    attention_dp_queue.expected_num_active_requests = 3

    result = attention_dp_queue._balance_requests_across_ranks(
        [], all_ranks_new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)

    # Should return the original structure unchanged
    assert result == all_ranks_new_requests
    for rank in range(4):
        assert len(result[rank]) == 0


def test_balance_requests_across_ranks_single_request(attention_dp_queue):
    """Test _balance_requests_across_ranks with a single request."""
    req = RequestQueueItem(
        1, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req.request.input_token_ids = [1, 2, 3, 4, 5]  # 5 tokens

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [1, 2, 0, 1]  # Rank 2 has lowest count
    all_ranks_num_active_tokens = [10, 20, 5, 15]

    # Set expected_num_active_requests for testing
    attention_dp_queue.expected_num_active_requests = 2

    result = attention_dp_queue._balance_requests_across_ranks(
        [req], all_ranks_new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)

    # Request should be assigned to rank 2 (lowest active count)
    assert len(result[0]) == 0
    assert len(result[1]) == 0
    assert len(result[2]) == 1
    assert len(result[3]) == 0
    assert result[2][0] == req


def test_balance_requests_across_ranks_multiple_requests(attention_dp_queue):
    """Test _balance_requests_across_ranks with multiple requests."""
    # Create requests with different token counts
    req1 = RequestQueueItem(
        1, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req1.request.input_token_ids = [1, 2, 3]  # 3 tokens

    req2 = RequestQueueItem(
        2, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req2.request.input_token_ids = [1, 2, 3, 4, 5, 6]  # 6 tokens

    req3 = RequestQueueItem(
        3, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req3.request.input_token_ids = [1, 2]  # 2 tokens

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [0, 1, 2, 1]
    all_ranks_num_active_tokens = [5, 15, 25, 10]

    # Set expected_num_active_requests for testing
    attention_dp_queue.expected_num_active_requests = 2

    result = attention_dp_queue._balance_requests_across_ranks(
        [req1, req2, req3], all_ranks_new_requests,
        all_ranks_num_active_requests, all_ranks_num_active_tokens)

    # Requests should be distributed based on heap (lowest active count first)
    # Requests are sorted by token count (descending) first, then assigned to ranks with lowest active count
    # req2 (6 tokens) -> rank 0 (0 active) -> total: 1 active, 11 tokens
    # req3 (2 tokens) -> rank 0 (1 active) -> total: 2 active, 13 tokens (rank 0 still has capacity)
    # req1 (3 tokens) -> rank 3 (1 active) -> total: 2 active, 13 tokens
    # Rank 1: 1 active, gets nothing (rank 0 took 2 requests)
    # Rank 2: 2 active, gets nothing (at capacity)

    assert len(result[0]) == 2  # req2 and req3 (rank 0 has capacity for 2)
    assert len(result[1]) == 0  # no requests (rank 0 took 2 requests)
    assert len(result[2]) == 0  # at capacity
    assert len(result[3]) == 1  # req1

    # Verify the requests are assigned correctly
    assert result[0][0] == req2  # First request (highest token count)
    assert result[0][1] == req3  # Second request
    assert result[3][0] == req1


def test_balance_requests_across_ranks_capacity_limits(attention_dp_queue):
    """Test _balance_requests_across_ranks respects capacity limits."""
    # Create multiple requests
    requests = []
    for i in range(4):
        req = RequestQueueItem(
            i,
            create_mock_request_with_py_schedule_params(attention_dp_rank=None))
        req.request.input_token_ids = [1] * (i + 1)  # Variable token counts
        requests.append(req)

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [1, 1, 1, 1]  # All ranks start with 1
    all_ranks_num_active_tokens = [10, 10, 10, 10]

    # Set expected_num_active_requests to limit capacity
    attention_dp_queue.expected_num_active_requests = 2

    result = attention_dp_queue._balance_requests_across_ranks(
        requests, all_ranks_new_requests, all_ranks_num_active_requests,
        all_ranks_num_active_tokens)

    # Each rank can only take 1 more request (1 + 1 = 2, which equals expected_num_active_requests)
    total_assigned = sum(
        len(rank_requests) for rank_requests in result.values())
    assert total_assigned == 4  # 4 ranks × 1 additional request each

    # Verify no rank exceeds capacity
    for rank in range(4):
        assert len(result[rank]) <= 1


def test_balance_requests_across_ranks_heap_ordering(attention_dp_queue):
    """Test that _balance_requests_across_ranks uses heap ordering correctly."""
    # Create requests with same token count to test heap ordering
    req1 = RequestQueueItem(
        1, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req1.request.input_token_ids = [1, 2, 3]  # 3 tokens

    req2 = RequestQueueItem(
        2, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req2.request.input_token_ids = [1, 2, 3]  # 3 tokens

    req3 = RequestQueueItem(
        3, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req3.request.input_token_ids = [1, 2, 3]  # 3 tokens

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    # Rank 0 has highest active count, should get requests last
    all_ranks_num_active_requests = [3, 1, 0, 2]
    all_ranks_num_active_tokens = [30, 10, 5, 20]

    # Set expected_num_active_requests for testing
    attention_dp_queue.expected_num_active_requests = 4

    result = attention_dp_queue._balance_requests_across_ranks(
        [req1, req2, req3], all_ranks_new_requests,
        all_ranks_num_active_requests, all_ranks_num_active_tokens)

    # Requests should be assigned in order of lowest active count first
    # Since all requests have same token count, they're assigned based on active count order
    # Rank 2: 0 active -> gets req1 and req2 (has capacity for 2)
    # Rank 1: 1 active -> gets req3 (after rank 2 takes 2)
    # Rank 3: 2 active -> gets nothing (rank 1 took req3)
    # Rank 0: 3 active -> gets nothing (at capacity)

    assert len(result[0]) == 0  # at capacity
    assert len(result[1]) == 1  # req3
    assert len(result[2]) == 2  # req1 and req2
    assert len(result[3]) == 0  # no requests

    # Verify the requests are assigned correctly
    assert result[1][0] == req3  # Third request
    assert result[2][0] == req1  # First request
    assert result[2][1] == req2  # Second request


def test_balance_requests_across_ranks_token_count_sorting(attention_dp_queue):
    """Test that requests are sorted by token count before distribution."""
    # Create requests with different token counts
    req1 = RequestQueueItem(
        1, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req1.request.input_token_ids = [1]  # 1 token (smallest)

    req2 = RequestQueueItem(
        2, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req2.request.input_token_ids = [1, 2, 3, 4, 5]  # 5 tokens (largest)

    req3 = RequestQueueItem(
        3, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req3.request.input_token_ids = [1, 2, 3]  # 3 tokens (medium)

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [0, 0, 0, 0]  # All ranks start empty
    all_ranks_num_active_tokens = [5, 5, 5, 5]

    # Set expected_num_active_requests for testing
    attention_dp_queue.expected_num_active_requests = 2

    result = attention_dp_queue._balance_requests_across_ranks(
        [req1, req2, req3], all_ranks_new_requests,
        all_ranks_num_active_requests, all_ranks_num_active_tokens)

    # Requests should be sorted by token count (descending) before distribution
    # Then assigned to ranks with lowest active count first
    # req2 (5 tokens) -> rank 0 (0 active)
    # req3 (3 tokens) -> rank 1 (0 active)
    # req1 (1 token) -> rank 2 (0 active)

    assert len(result[0]) == 1  # req2 (highest token count)
    assert len(result[1]) == 1  # req3
    assert len(result[2]) == 1  # req1 (lowest token count)
    assert len(result[3]) == 0

    # Verify the requests are assigned correctly
    assert result[0][0] == req2
    assert result[1][0] == req3
    assert result[2][0] == req1
