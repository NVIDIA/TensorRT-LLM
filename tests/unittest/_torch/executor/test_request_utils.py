"""Tests for request_utils.py functions.

This module tests:
- Request merging functions (merge_requests, merge_helix_requests)
- Waiting queue functions (get_from_waiting_queue, can_process_attention_dp_request)

"""

from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.request_utils import (
    can_process_attention_dp_request,
    derive_attention_dp_per_rank_request_cap,
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
        # Round-robin block distribution across 4 CP ranks (7 blocks total, 2 tokens/block):
        #   rank 0 owns blocks {0, 4} -> tokens [1,2, 9,10]
        #   rank 1 owns blocks {1, 5} -> tokens [3,4, 11,12]
        #   rank 2 owns blocks {2, 6} -> tokens [5,6, 13]  (block 6 is the last block; padding stripped)
        #   rank 3 owns block  {3}    -> tokens [7,8]
        if rank == 0:
            assert llm_request.get_tokens(0) == [1, 2, 9, 10]
        elif rank == 1:
            assert llm_request.get_tokens(0) == [3, 4, 11, 12]
        elif rank == 2:
            assert llm_request.get_tokens(0) == [5, 6, 13]
        else:
            assert llm_request.get_tokens(0) == [7, 8]


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
        # Round-robin block distribution across 2 CP ranks (3 blocks total, 4 tokens/block):
        #   rank 0 owns blocks {0, 2} -> tokens [1,2,3,4, 9,10,11,12]
        #   rank 1 owns block  {1}    -> tokens [5,6,7,8]
        if rank == 0:
            assert llm_request.get_tokens(0) == [1, 2, 3, 4, 9, 10, 11, 12]
        else:
            assert llm_request.get_tokens(0) == [5, 6, 7, 8]


def test_merge_helix_requests_empty_ranks():
    """When num_total_blocks < cp_size, the highest CP ranks own no blocks.

    Such "empty" ranks must produce an empty token list (and seqlen_this_rank_cp
    == 0), while total_input_len_cp still reflects the full prompt length so the
    global position ids stay correct. They are no longer rejected.
    """
    tokens_per_block = 4

    # 12 tokens -> 3 blocks, which is fewer than 4 CP ranks, so rank 3 is empty.
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    executor_request = trtllm.Request(
        input_token_ids=input_tokens,
        max_tokens=12,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

    # Round-robin block distribution across 4 CP ranks (3 blocks total, 4 tokens/block):
    #   rank 0 owns block {0} -> tokens [1,2,3,4]
    #   rank 1 owns block {1} -> tokens [5,6,7,8]
    #   rank 2 owns block {2} -> tokens [9,10,11,12]
    #   rank 3 owns no blocks -> [] (empty rank)
    expected_tokens = {
        0: [1, 2, 3, 4],
        1: [5, 6, 7, 8],
        2: [9, 10, 11, 12],
        3: [],
    }
    for rank in range(4):
        result = merge_helix_requests(
            [request_item],
            cp_rank=rank,
            cp_size=4,
            tokens_per_block=tokens_per_block,
            exclude_last_generation_logits=False,
        )

        assert len(result) == 1
        llm_request = result[0]
        assert isinstance(llm_request, LlmRequest)
        assert llm_request.request_id == 1
        assert llm_request.get_tokens(0) == expected_tokens[rank]
        # total_input_len_cp is always the full prompt length.
        assert llm_request.total_input_len_cp == len(input_tokens)
        assert llm_request.seqlen_this_rank_cp == len(expected_tokens[rank])


def test_merge_helix_requests_empty_ranks_with_padding():
    """Exercise padding-strip-on-last-owner and empty ranks together.

    With 10 tokens and tokens_per_block=4 there are 3 blocks, the last of which
    is partially filled (tokens [9, 10]). Distributed round-robin over 4 CP
    ranks, the last block owner (rank 2) must strip the block padding while the
    block-less rank (rank 3) must be an empty rank.
    """
    tokens_per_block = 4

    # 10 tokens -> 3 blocks (last block half-full), fewer than 4 CP ranks.
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    executor_request = trtllm.Request(
        input_token_ids=input_tokens,
        max_tokens=12,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

    # Round-robin block distribution across 4 CP ranks (3 blocks, 4 tokens/block):
    #   rank 0 owns block {0} -> tokens [1,2,3,4]
    #   rank 1 owns block {1} -> tokens [5,6,7,8]
    #   rank 2 owns block {2} -> tokens [9,10]  (last block; padding stripped)
    #   rank 3 owns no blocks -> [] (empty rank)
    expected_tokens = {
        0: [1, 2, 3, 4],
        1: [5, 6, 7, 8],
        2: [9, 10],
        3: [],
    }
    for rank in range(4):
        result = merge_helix_requests(
            [request_item],
            cp_rank=rank,
            cp_size=4,
            tokens_per_block=tokens_per_block,
            exclude_last_generation_logits=False,
        )

        assert len(result) == 1
        llm_request = result[0]
        assert isinstance(llm_request, LlmRequest)
        assert llm_request.request_id == 1
        assert llm_request.get_tokens(0) == expected_tokens[rank]
        # total_input_len_cp is always the full prompt length.
        assert llm_request.total_input_len_cp == len(input_tokens)
        assert llm_request.seqlen_this_rank_cp == len(expected_tokens[rank])


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
        # Round-robin block distribution across 4 CP ranks (7 blocks total, 2 tokens/block):
        #   rank 0 owns blocks {0, 4} -> tokens [1,2, 9,10]
        #   rank 1 owns blocks {1, 5} -> tokens [3,4, 11,12]
        #   rank 2 owns blocks {2, 6} -> tokens [5,6, 13]  (block 6 is the last block; padding stripped)
        #   rank 3 owns block  {3}    -> tokens [7,8]
        if rank == 0:
            assert llm_request.get_tokens(0) == [1, 2, 9, 10]
        elif rank == 1:
            assert llm_request.get_tokens(0) == [3, 4, 11, 12]
        elif rank == 2:
            assert llm_request.get_tokens(0) == [5, 6, 13]
        else:
            assert llm_request.get_tokens(0) == [7, 8]


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


# --------------------------------------------------------------------------
# nvbug-6133201: per-rank gen-phase step-token cap via tightened
# per-rank request cap.
# --------------------------------------------------------------------------
# Under enable_attention_dp the global Python scheduler caps tokens
# cluster-wide and the ADP router caps per-rank requests, but no
# component caps per-rank gen-phase step-tokens.  PyExecutor tightens
# the per-rank request cap to
# ``max_num_tokens // (1 + max_total_draft_tokens)`` so per-rank step-
# token load cannot exceed max_num_tokens by construction.


class TestDeriveAttentionDpPerRankRequestCap:
    """Unit tests for ``derive_attention_dp_per_rank_request_cap``.

    The fix for nvbug-6133201 is the cap arithmetic implemented in this
    helper; PyExecutor calls it once at ``__init__`` and the result
    flows through the existing ``max_num_active_requests`` plumbing.
    """

    def test_no_tightening_when_max_num_tokens_is_none(self):
        # LlmArgs.max_num_tokens == None -> helper is a no-op.
        assert (
            derive_attention_dp_per_rank_request_cap(
                base_cap=128, max_num_tokens=None, max_total_draft_tokens=3
            )
            == 128
        )

    def test_nvbug_6133201_failing_config(self):
        # nvbug-6133201 numbers: max_batch_size=128,
        # max_total_draft_tokens=3 (MTP3), max_num_tokens=256.
        # Per-rank step-token cost per req = 1 + 3 = 4.
        # Effective cap = 256 // 4 = 64; per-rank load at saturation
        # 64 * 4 = 256 = max_num_tokens, so the per-rank assert in
        # model_engine.py cannot trip on gen-phase accumulation.
        assert (
            derive_attention_dp_per_rank_request_cap(
                base_cap=128, max_num_tokens=256, max_total_draft_tokens=3
            )
            == 64
        )

    def test_no_tightening_when_arithmetic_already_fits(self):
        # Correctly-sized LlmArgs (max_batch_size * (1+max_total_draft_tokens)
        # <= max_num_tokens): cap == base_cap, no behavioral change.
        assert (
            derive_attention_dp_per_rank_request_cap(
                base_cap=128, max_num_tokens=512, max_total_draft_tokens=3
            )
            == 128
        )
        assert (
            derive_attention_dp_per_rank_request_cap(
                base_cap=128, max_num_tokens=4096, max_total_draft_tokens=3
            )
            == 128
        )

    def test_no_spec_decoding(self):
        # max_total_draft_tokens == 0: step-token cost per req == 1,
        # effective cap == max_num_tokens.
        assert (
            derive_attention_dp_per_rank_request_cap(
                base_cap=128, max_num_tokens=64, max_total_draft_tokens=0
            )
            == 64
        )

    def test_negative_max_total_draft_tokens_clamped(self):
        # Defensive: a stray negative value must not yield div-by-zero
        # or a negative cap.
        assert (
            derive_attention_dp_per_rank_request_cap(
                base_cap=128, max_num_tokens=256, max_total_draft_tokens=-5
            )
            == 128
        )
