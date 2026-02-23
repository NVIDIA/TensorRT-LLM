"""Tests for request_utils.py functions.

This module tests:
- Request merging functions (merge_requests, merge_helix_requests)
- Waiting queue functions (get_from_waiting_queue, can_process_attention_dp_request)

ADP-specific tests (DefaultADPRequestAssigner, _balance_requests_across_ranks)
have been moved to test_adp.py.
"""

from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.request_utils import (
    can_process_attention_dp_request,
    get_from_waiting_queue,
    merge_helix_requests,
    merge_requests,
)
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm.bindings import executor as trtllm
from tensorrt_llm.mapping import CpType


@pytest.fixture
def attention_dp_config():
    """Create a config dict for attention DP testing."""
    return {
        "tp_size": 4,
        "max_num_active_requests": 8,
    }


@pytest.fixture
def all_ranks_num_active_requests():
    return [2, 1, 3, 0]  # 4 ranks


def create_mock_request_with_py_schedule_params(attention_dp_rank=None, attention_dp_relax=False):
    mock_request = Mock()

    if attention_dp_rank is not None:
        mock_schedule_params = Mock()
        mock_schedule_params.attention_dp_rank = attention_dp_rank
        mock_schedule_params.attention_dp_relax = attention_dp_relax

        mock_schedule_params.configure_mock(
            attention_dp_rank=attention_dp_rank, attention_dp_relax=attention_dp_relax
        )

        mock_request.py_scheduling_params = mock_schedule_params
    else:
        mock_request.py_scheduling_params = None

    mock_request.input_token_ids = [1, 2, 3]

    return mock_request


def test_merge_helix_requests_with_padding():
    """Test merge_helix_requests with basic valid input."""

    tokens_per_block = 2

    # Create request item with 13 tokens to get exactly 7 blocks for 4 CP ranks.
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    executor_request = trtllm.Request(
        input_token_ids=input_tokens,
        max_tokens=5,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    for rank in [0, 1, 2, 3]:
        # Test merge_helix_requests with 4 CP ranks.
        result = merge_helix_requests(
            [request_item],
            cp_rank=rank,
            cp_size=4,
            tokens_per_block=tokens_per_block,
            exclude_last_generation_logits=False,
        )

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


def test_merge_helix_requests_without_padding():
    """Test merge_helix_requests with evenly divisible tokens (no padding)."""

    tokens_per_block = 4

    # Create request item with 12 tokens to get exactly 3 blocks for 2 CP ranks.
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    executor_request = trtllm.Request(
        input_token_ids=input_tokens,
        max_tokens=5,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    for rank in [0, 1]:
        # Test merge_helix_requests with 2 CP ranks.
        result = merge_helix_requests(
            [request_item],
            cp_rank=rank,
            cp_size=2,
            tokens_per_block=tokens_per_block,
            exclude_last_generation_logits=False,
        )

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


def test_merge_helix_requests_insufficient_blocks_error():
    """Test merge_helix_requests raises error when insufficient blocks."""
    tokens_per_block = 4

    # Create input with only 12 tokens. This creates 3 blocks which is fewer than 4 CP ranks.
    executor_request = trtllm.Request(
        input_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        max_tokens=12,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    # Loop over ranks 0, 1, 2, 3 and verify that all ranks throw assertion.
    for rank in range(4):
        with pytest.raises(
            ValueError, match="There aren't enough tokens to get at least one block per CP rank"
        ):
            merge_helix_requests(
                [request_item],
                cp_rank=rank,
                cp_size=4,
                tokens_per_block=tokens_per_block,
                exclude_last_generation_logits=False,
            )


@patch("tensorrt_llm._torch.pyexecutor.request_utils.executor_request_to_llm_request")
def test_merge_requests_default(mock_convert):
    """Test merging requests with default configuration."""
    mock_llm_request = Mock(child_requests=[])
    mock_convert.return_value = mock_llm_request

    requests = [RequestQueueItem(1, Mock()), RequestQueueItem(2, Mock())]
    result = merge_requests(
        requests, cp_config={}, cp_rank=0, cp_size=1, exclude_last_generation_logits=False
    )

    assert len(result) == 2
    assert mock_convert.call_count == 2


def test_merge_requests_with_helix_cp_config():
    """Test merge_requests routes to merge_helix_requests with HELIX cp_config."""
    tokens_per_block = 2

    # Create request item with 13 tokens to get exactly 7 blocks for 4 CP ranks.
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    executor_request = trtllm.Request(
        input_token_ids=input_tokens,
        max_tokens=5,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    cp_config = {
        "cp_type": CpType.HELIX,
        "tokens_per_block": tokens_per_block,
    }

    for rank in [0, 1, 2, 3]:
        # Test merge_requests with HELIX cp_config and 4 CP ranks.
        result = merge_requests(
            [request_item],
            cp_config=cp_config,
            cp_rank=rank,
            cp_size=4,
            exclude_last_generation_logits=False,
        )

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


def test_get_from_waiting_queue():
    """Test getting items from waiting queue."""
    # Add items to waiting queue
    waiting_queue = FCFSWaitingQueue()
    items = [RequestQueueItem(i, Mock()) for i in range(5)]
    waiting_queue.extend(items)

    # Get 3 items
    result = get_from_waiting_queue(
        waiting_queue, 3, enable_attention_dp=False, max_num_active_requests=16
    )

    assert len(result) == 3
    assert result == items[:3]
    assert len(waiting_queue) == 2


@pytest.mark.parametrize(
    "queue_size,request_count,expected_result,expected_remaining",
    [
        (0, 5, 0, 0),  # Empty queue
        (3, -1, 0, 3),  # Negative count
        (3, 0, 0, 3),  # Zero count
        (3, 10, 3, 0),  # Request more than available
    ],
)
def test_get_from_waiting_queue_edge_cases(
    queue_size, request_count, expected_result, expected_remaining
):
    """Test edge cases for getting items from waiting queue."""
    # Setup queue
    waiting_queue = FCFSWaitingQueue()
    if queue_size > 0:
        items = [RequestQueueItem(i, Mock()) for i in range(queue_size)]
        waiting_queue.extend(items)

    result = get_from_waiting_queue(
        waiting_queue, request_count, enable_attention_dp=False, max_num_active_requests=16
    )

    assert len(result) == expected_result
    assert len(waiting_queue) == expected_remaining


def test_get_from_waiting_queue_with_attention_dp(
    attention_dp_config, all_ranks_num_active_requests
):
    waiting_queue = FCFSWaitingQueue()
    items = [RequestQueueItem(i, Mock()) for i in range(5)]
    waiting_queue.extend(items)

    result = get_from_waiting_queue(
        waiting_queue,
        3,
        True,
        attention_dp_config["max_num_active_requests"],
        all_ranks_num_active_requests,
    )

    assert len(result) == 3
    assert result == items[:3]
    assert len(waiting_queue) == 2


def test_get_from_waiting_queue_with_attention_dp_filtering(
    attention_dp_config, all_ranks_num_active_requests
):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )
    req2 = RequestQueueItem(
        2, create_mock_request_with_py_schedule_params(attention_dp_rank=1, attention_dp_relax=True)
    )
    req3 = RequestQueueItem(
        3, create_mock_request_with_py_schedule_params(attention_dp_rank=None)
    )  # No scheduling params

    waiting_queue = FCFSWaitingQueue()
    waiting_queue.extend([req1, req2, req3])

    # Set rank 0 to full capacity to test filtering
    all_ranks_num_active_requests[0] = 8

    result = get_from_waiting_queue(
        waiting_queue,
        3,
        True,
        attention_dp_config["max_num_active_requests"],
        all_ranks_num_active_requests,
    )

    assert len(result) == 2
    assert req2 in result
    assert req3 in result
    assert req1 not in result


def test_can_process_attention_dp_request(attention_dp_config):
    max_num_active_requests = attention_dp_config["max_num_active_requests"]

    req_no_params = RequestQueueItem(1, Mock())
    assert can_process_attention_dp_request(req_no_params, [0, 0, 0, 0], max_num_active_requests)

    req_relax = RequestQueueItem(
        2, create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True)
    )
    assert can_process_attention_dp_request(req_relax, [0, 0, 0, 0], max_num_active_requests)

    req_target = RequestQueueItem(
        3,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1, attention_dp_relax=False),
    )
    all_ranks = [0, 0, 0, 0]
    assert can_process_attention_dp_request(req_target, all_ranks, max_num_active_requests)
    assert all_ranks[1] == 1

    req_no_capacity = RequestQueueItem(
        4,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )
    all_ranks_full = [8, 0, 0, 0]  # Rank 0 is at capacity
    assert not can_process_attention_dp_request(
        req_no_capacity, all_ranks_full, max_num_active_requests
    )
