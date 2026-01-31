"""Tests for request_utils.py functions.

This module tests:
- Request merging functions (merge_requests, merge_helix_requests)
- Attention DP scheduling functions (schedule_attention_dp_requests, balance_requests_across_ranks)
- Waiting queue functions (get_from_waiting_queue, can_process_attention_dp_request)
"""

from collections import deque
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.request_utils import (
    balance_requests_across_ranks,
    can_process_attention_dp_request,
    get_from_waiting_queue,
    merge_helix_requests,
    merge_requests,
    schedule_attention_dp_requests,
)
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


@pytest.fixture
def all_ranks_num_active_tokens():
    return [10, 5, 15, 8]  # 4 ranks


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


def append_to_waiting_queue(waiting_queue, rank, attention_dp_relax):
    req_id = len(waiting_queue)
    waiting_queue.append(
        RequestQueueItem(
            req_id,
            create_mock_request_with_py_schedule_params(
                attention_dp_rank=rank, attention_dp_relax=attention_dp_relax
            ),
        )
    )


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
    waiting_queue = deque()
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
    waiting_queue = deque()
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
    waiting_queue = deque()
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

    waiting_queue = deque([req1, req2, req3])

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


def test_schedule_attention_dp_requests_scheduled_requests(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )

    new_requests = [req1, req2]

    all_ranks_new_requests, _ = schedule_attention_dp_requests(
        new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        attention_dp_config["tp_size"],
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]

    assert len(result) == 2
    assert req1 in result
    assert req2 in result

    assert all_ranks_num_active_requests[0] == 4


def test_schedule_attention_dp_requests_scheduled_requests_other_ranks(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1, attention_dp_relax=False),
    )
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=2, attention_dp_relax=False),
    )

    new_requests = [req1, req2]

    all_ranks_new_requests, _ = schedule_attention_dp_requests(
        new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        attention_dp_config["tp_size"],
        attention_dp_config["max_num_active_requests"],
    )

    result = all_ranks_new_requests[0]
    assert len(result) == 0

    assert all_ranks_num_active_requests[1] == 2
    assert all_ranks_num_active_requests[2] == 4


def test_schedule_attention_dp_requests_unscheduled_requests(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    req1 = RequestQueueItem(
        1, create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True)
    )
    req2 = RequestQueueItem(
        2, create_mock_request_with_py_schedule_params(attention_dp_rank=1, attention_dp_relax=True)
    )

    new_requests = [req1, req2]

    all_ranks_new_requests, _ = schedule_attention_dp_requests(
        new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        attention_dp_config["tp_size"],
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]

    assert len(result) == 1  # Only req1 for current rank
    assert req1 in result


def test_schedule_attention_dp_requests_unscheduled_no_capacity(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    all_ranks_num_active_requests[0] = 8

    req1 = RequestQueueItem(
        1, create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True)
    )

    new_requests = [req1]

    all_ranks_new_requests, _ = schedule_attention_dp_requests(
        new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        attention_dp_config["tp_size"],
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]

    assert len(result) == 0  # No capacity


def test_schedule_attention_dp_requests_mixed_scenarios(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    req_scheduled_current = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )
    req_scheduled_other = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1, attention_dp_relax=False),
    )
    req_unscheduled_current = RequestQueueItem(
        3, create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True)
    )
    req_unscheduled_other = RequestQueueItem(
        4, create_mock_request_with_py_schedule_params(attention_dp_rank=2, attention_dp_relax=True)
    )

    new_requests = [
        req_scheduled_current,
        req_scheduled_other,
        req_unscheduled_current,
        req_unscheduled_other,
    ]

    all_ranks_new_requests, _ = schedule_attention_dp_requests(
        new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        attention_dp_config["tp_size"],
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]

    assert len(result) == 2
    assert req_scheduled_current in result
    assert req_unscheduled_current in result


def test_schedule_attention_dp_requests_empty_lists(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    all_ranks_new_requests, _ = schedule_attention_dp_requests(
        [],
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        attention_dp_config["tp_size"],
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]

    assert len(result) == 0


def test_schedule_attention_dp_requests_expected_num_active_calculation(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    req1 = RequestQueueItem(
        1, create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True)
    )
    req2 = RequestQueueItem(
        2, create_mock_request_with_py_schedule_params(attention_dp_rank=1, attention_dp_relax=True)
    )

    new_requests = [req1, req2]

    _, expected_num_active_requests = schedule_attention_dp_requests(
        new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        attention_dp_config["tp_size"],
        attention_dp_config["max_num_active_requests"],
    )

    # 2 + 1 + 3 + 0 = 6, 6 + 2 = 8, (8 + 3) // 4 = 2, max(2, 2, 1, 3, 0) = 3
    # expected_num_active_requests = max((6 + 2 + 3) // 4, 3) = max(2, 3) = 3
    assert expected_num_active_requests == 3


def test_schedule_attention_dp_requests_balance_requests_called(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    """Test that balance_requests_across_ranks is called with correct arguments."""
    req1 = RequestQueueItem(
        1, create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True)
    )

    new_requests = [req1]

    with patch(
        "tensorrt_llm._torch.pyexecutor.request_utils.balance_requests_across_ranks"
    ) as mock_balance:
        mock_balance.return_value = {0: [req1], 1: [], 2: [], 3: []}

        schedule_attention_dp_requests(
            new_requests,
            all_ranks_num_active_requests,
            all_ranks_num_active_tokens,
            attention_dp_config["tp_size"],
            attention_dp_config["max_num_active_requests"],
        )

    # Check that balance_requests_across_ranks was called
    mock_balance.assert_called_once()
    call_args = mock_balance.call_args[0]
    assert isinstance(call_args[0], list)
    assert isinstance(call_args[1], dict)
    assert call_args[2] == all_ranks_num_active_requests  # Third arg
    assert call_args[3] == all_ranks_num_active_tokens  # Fourth arg


def test_schedule_attention_dp_requests_no_scheduling_when_capacity_exceeded(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    all_ranks_num_active_requests[0] = 8

    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )

    new_requests = [req1]

    all_ranks_new_requests, _ = schedule_attention_dp_requests(
        new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        attention_dp_config["tp_size"],
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]

    assert len(result) == 0  # No requests scheduled
    assert all_ranks_num_active_requests[0] == 8  # Capacity unchanged


def test_filter_and_schedule_integration(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    req_schedulable = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )
    req_schedulable.request.input_token_ids = [1, 2, 3, 4]
    req_relax = RequestQueueItem(
        2, create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True)
    )
    req_relax.request.input_token_ids = [1, 2]

    req_no_params = RequestQueueItem(
        3, create_mock_request_with_py_schedule_params(attention_dp_rank=None)
    )

    new_requests = [req_schedulable, req_relax, req_no_params]

    all_ranks_new_requests, _ = schedule_attention_dp_requests(
        new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        attention_dp_config["tp_size"],
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]

    assert len(result) == 2
    assert req_schedulable in result
    assert req_relax in result


def test_filter_and_schedule_with_capacity_limits(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    all_ranks_num_active_requests[0] = 7

    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )
    req1.request.input_token_ids = [1, 2, 3, 4]
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )
    req2.request.input_token_ids = [1, 2, 3]

    new_requests = [req1, req2]

    all_ranks_new_requests, _ = schedule_attention_dp_requests(
        new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        attention_dp_config["tp_size"],
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]

    assert len(result) == 1
    assert req1 in result


def test_achieve_max_num_active_requests(attention_dp_config):
    max_num_active_requests = attention_dp_config["max_num_active_requests"]
    req_list = []
    req_id = 0
    for rank in range(4):
        for _ in range(5):
            req_list.append(
                RequestQueueItem(
                    req_id,
                    create_mock_request_with_py_schedule_params(
                        attention_dp_rank=rank, attention_dp_relax=False
                    ),
                )
            )
            req_id += 1
            req_list.append(
                RequestQueueItem(
                    req_id,
                    create_mock_request_with_py_schedule_params(
                        attention_dp_rank=rank, attention_dp_relax=True
                    ),
                )
            )
            req_id += 1

    all_ranks_num_active_requests = [5, 6, 3, 7]
    waiting_queue = deque(req_list)
    available_active_requests = max_num_active_requests * 4 - sum(all_ranks_num_active_requests)

    result = get_from_waiting_queue(
        waiting_queue,
        available_active_requests,
        True,
        max_num_active_requests,
        all_ranks_num_active_requests,
    )

    assert len(result) == available_active_requests


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
                3: [6],  # Last request goes to rank 3
            },
        ),
        # Case: Balanced distribution of relaxed requests with existing load
        (
            3,
            [1, 2, 3, 0],
            [(None, True)] * 13,
            {
                0: [0, 1],  # Rank 0 gets first 2 requests
                1: [2],  # Rank 1 gets 1 request (already has 2)
                2: [],  # Rank 2 is at capacity (3)
                3: [3, 4, 5],  # Rank 3 gets 3 requests (starts with 0)
            },
        ),
        # Case: Limited by max active
        (
            3,
            [0, 0, 0, 0],
            [(None, True)] * 13,
            {
                0: [0, 1, 3],  # First 3 requests (0, 1, 3)
                1: [2, 4, 6],  # Next 3 requests (2, 4, 6)
                2: [5, 7, 9],  # Next 3 requests (5, 7, 9)
                3: [8, 10, 11],  # Last 3 requests (8, 10, 11)
            },
        ),
        # Case: Empty new requests
        (3, [3, 3, 3, 0], [], {0: [], 1: [], 2: [], 3: []}),
        # Case: Rank 0 is full and cannot schedule attention_dp rank request
        (
            3,
            [3, 1, 3, 0],
            [(0, False), (0, True)],
            {
                0: [],  # Rank 0 is full
                1: [1],  # Rank 1 gets the relaxed request (req1)
                2: [],  # No relaxed requests assigned here
                3: [],  # No relaxed requests assigned here
            },
        ),
        # Case: Only room for 1 request, need to skip req0 with attention dp rank
        (
            3,
            [3, 2, 3, 3],
            [(0, False), (0, True)],
            {
                0: [],  # Rank 0 is full
                1: [1],  # Rank 1 gets the relaxed request
                2: [],  # Rank 2 is at capacity
                3: [],  # Rank 3 is at capacity
            },
        ),
        # Case: Targeting ranks 1 and 3 that have room
        (
            3,
            [2, 1, 3, 0],
            [(1, False), (3, False)],
            {
                0: [],  # No requests assigned to rank 0
                1: [0],  # Request 0 targets rank 1
                2: [],  # No requests assigned to rank 2
                3: [1],  # Request 1 targets rank 3
            },
        ),
        # Case: Target dp rank specified, but relax is True
        (
            3,
            [3, 3, 3, 1],
            [(0, True), (1, True), (2, True)],
            {
                0: [],  # Rank 0 is at capacity
                1: [],  # Rank 1 is at capacity
                2: [],  # Rank 2 is at capacity
                3: [0, 1],  # Rank 3 gets both relaxed requests
            },
        ),
        # Case: Mixed targeting and relaxed
        (
            3,
            [3, 3, 3, 0],
            [(0, False), (1, True), (3, False)],
            {
                0: [],  # Rank 0 is at capacity
                1: [],  # Rank 1 is at capacity
                2: [],  # Rank 2 is at capacity
                3: [2, 1],  # Rank 3 gets both requests (targeted + relaxed)
            },
        ),
    ],
)
def test_attention_dp_scheduling_cases(
    max_num_active_requests,
    all_ranks_num_active_requests,
    request_configs,
    all_ranks_expected_req_ids,
):
    """Test attention DP scheduling with various scenarios."""
    waiting_queue = deque()
    for rank, relax in request_configs:
        append_to_waiting_queue(waiting_queue, rank, relax)

    run_test_attention_dp_scheduling(
        max_num_active_requests,
        waiting_queue,
        all_ranks_num_active_requests,
        all_ranks_expected_req_ids,
    )


def run_test_attention_dp_scheduling(
    max_num_active_requests,
    waiting_queue,
    all_ranks_num_active_requests,
    all_ranks_expected_req_ids,
):
    num_ranks = len(all_ranks_num_active_requests)
    total_num_active_requests = sum(all_ranks_num_active_requests)
    total_max_num_active_requests = max_num_active_requests * num_ranks
    enable_attention_dp = True

    new_requests = get_from_waiting_queue(
        waiting_queue,
        total_max_num_active_requests - total_num_active_requests,
        enable_attention_dp,
        max_num_active_requests,
        all_ranks_num_active_requests,
    )

    # Create mock token counts for testing
    all_ranks_num_active_tokens = [10 + i * 5 for i in range(num_ranks)]

    # Schedule attention dp requests
    all_ranks_new_requests, _ = schedule_attention_dp_requests(
        new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        num_ranks,
        max_num_active_requests,
    )

    assert len(all_ranks_new_requests) == num_ranks
    print("all_ranks_new_requests:", all_ranks_new_requests)
    for rank, reqs in all_ranks_new_requests.items():
        req_ids = [req.id for req in reqs]
        assert req_ids == all_ranks_expected_req_ids[rank]


def test_balance_requests_across_ranks_empty_requests():
    """Test balance_requests_across_ranks with empty requests list."""
    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [2, 1, 3, 0]
    all_ranks_num_active_tokens = [20, 10, 30, 5]
    expected_num_active_requests = 3

    result = balance_requests_across_ranks(
        [],
        all_ranks_new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        expected_num_active_requests,
    )

    # Should return the original structure unchanged
    assert result == all_ranks_new_requests
    for rank in range(4):
        assert len(result[rank]) == 0


def test_balance_requests_across_ranks_single_request():
    """Test balance_requests_across_ranks with a single request."""
    req = RequestQueueItem(1, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req.request.input_token_ids = [1, 2, 3, 4, 5]  # 5 tokens

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [1, 2, 0, 1]  # Rank 2 has lowest count
    all_ranks_num_active_tokens = [10, 20, 5, 15]
    expected_num_active_requests = 2

    result = balance_requests_across_ranks(
        [req],
        all_ranks_new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        expected_num_active_requests,
    )

    # Request should be assigned to rank 2 (lowest active count)
    assert len(result[0]) == 0
    assert len(result[1]) == 0
    assert len(result[2]) == 1
    assert len(result[3]) == 0
    assert result[2][0] == req


def test_balance_requests_across_ranks_multiple_requests():
    """Test balance_requests_across_ranks with multiple requests."""
    # Create requests with different token counts
    req1 = RequestQueueItem(1, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req1.request.input_token_ids = [1, 2, 3]  # 3 tokens

    req2 = RequestQueueItem(2, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req2.request.input_token_ids = [1, 2, 3, 4, 5, 6]  # 6 tokens

    req3 = RequestQueueItem(3, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req3.request.input_token_ids = [1, 2]  # 2 tokens

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [0, 1, 2, 1]
    all_ranks_num_active_tokens = [5, 15, 25, 10]
    expected_num_active_requests = 2

    result = balance_requests_across_ranks(
        [req1, req2, req3],
        all_ranks_new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        expected_num_active_requests,
    )

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


def test_balance_requests_across_ranks_capacity_limits():
    """Test balance_requests_across_ranks respects capacity limits."""
    # Create multiple requests
    requests = []
    for i in range(4):
        req = RequestQueueItem(
            i, create_mock_request_with_py_schedule_params(attention_dp_rank=None)
        )
        req.request.input_token_ids = [1] * (i + 1)  # Variable token counts
        requests.append(req)

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [1, 1, 1, 1]  # All ranks start with 1
    all_ranks_num_active_tokens = [10, 10, 10, 10]
    expected_num_active_requests = 2

    result = balance_requests_across_ranks(
        requests,
        all_ranks_new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        expected_num_active_requests,
    )

    # Each rank can only take 1 more request (1 + 1 = 2, which equals expected_num_active_requests)
    total_assigned = sum(len(rank_requests) for rank_requests in result.values())
    assert total_assigned == 4  # 4 ranks with 1 additional request each

    # Verify no rank exceeds capacity
    for rank in range(4):
        assert len(result[rank]) <= 1


def test_balance_requests_across_ranks_heap_ordering():
    """Test that balance_requests_across_ranks uses heap ordering correctly."""
    # Create requests with same token count to test heap ordering
    req1 = RequestQueueItem(1, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req1.request.input_token_ids = [1, 2, 3]  # 3 tokens

    req2 = RequestQueueItem(2, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req2.request.input_token_ids = [1, 2, 3]  # 3 tokens

    req3 = RequestQueueItem(3, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req3.request.input_token_ids = [1, 2, 3]  # 3 tokens

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    # Rank 0 has highest active count, should get requests last
    all_ranks_num_active_requests = [3, 1, 0, 2]
    all_ranks_num_active_tokens = [30, 10, 5, 20]
    expected_num_active_requests = 4

    result = balance_requests_across_ranks(
        [req1, req2, req3],
        all_ranks_new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        expected_num_active_requests,
    )

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


def test_balance_requests_across_ranks_token_count_sorting():
    """Test that requests are sorted by token count before distribution."""
    # Create requests with different token counts
    req1 = RequestQueueItem(1, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req1.request.input_token_ids = [1]  # 1 token (smallest)

    req2 = RequestQueueItem(2, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req2.request.input_token_ids = [1, 2, 3, 4, 5]  # 5 tokens (largest)

    req3 = RequestQueueItem(3, create_mock_request_with_py_schedule_params(attention_dp_rank=None))
    req3.request.input_token_ids = [1, 2, 3]  # 3 tokens (medium)

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [0, 0, 0, 0]  # All ranks start empty
    all_ranks_num_active_tokens = [5, 5, 5, 5]
    expected_num_active_requests = 2

    result = balance_requests_across_ranks(
        [req1, req2, req3],
        all_ranks_new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        expected_num_active_requests,
    )

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
