# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for request_utils.py functions.

This module tests:
- Request merging functions (merge_requests, merge_helix_requests)
- Waiting queue functions (get_from_waiting_queue, can_process_attention_dp_request)

"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.request_utils import (
    RequestBroadcaster,
    attach_py_objects_to_requests,
    can_process_attention_dp_request,
    derive_attention_dp_per_rank_request_cap,
    executor_request_to_llm_request,
    get_from_waiting_queue,
    merge_helix_requests,
    merge_requests,
)
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm.bindings import executor as trtllm
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.mapping import CpType

pytestmark = pytest.mark.cpu_only


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


def test_request_broadcaster_collects_conversation_params_with_none():
    conversation_params = ConversationParams(conversation_id="conv")
    source_items = [
        RequestQueueItem(1, SimpleNamespace(py_conversation_params=conversation_params)),
        RequestQueueItem(2, SimpleNamespace(py_conversation_params=None)),
    ]

    py_request_objects = RequestBroadcaster._collect_py_objects(None, source_items)

    py_objects = dict(py_request_objects)
    assert py_objects["py_conversation_params"] == {
        1: conversation_params,
        2: None,
    }

    target_items = [
        RequestQueueItem(1, SimpleNamespace()),
        RequestQueueItem(2, SimpleNamespace()),
    ]
    attach_py_objects_to_requests(target_items, py_request_objects)

    assert target_items[0].request.py_conversation_params is conversation_params
    assert hasattr(target_items[1].request, "py_conversation_params")
    assert target_items[1].request.py_conversation_params is None


def test_request_broadcaster_requires_conversation_params_attr():
    source_items = [RequestQueueItem(1, SimpleNamespace())]

    with pytest.raises(AttributeError):
        RequestBroadcaster._collect_py_objects(None, source_items)


def test_executor_request_to_llm_request_adopts_context_phase_draft_tokens() -> None:
    request_id = 42
    first_gen_tokens = [100]
    draft_tokens = [101, 102, 103]
    context_phase_params = trtllm.ContextPhaseParams(
        first_gen_tokens,
        request_id,
        None,
        draft_tokens,
        None,
        None,
    )
    executor_request = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=10,
        type=trtllm.RequestType.REQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=context_phase_params,
    )

    llm_request = executor_request_to_llm_request(
        request_id,
        executor_request,
        child_req_ids=[],
        exclude_last_generation_logits=False,
    )

    assert llm_request.is_generation_only_request
    assert llm_request.has_draft_tokens()
    assert llm_request.num_draft_tokens == len(draft_tokens)
    assert llm_request.draft_tokens == draft_tokens
    assert llm_request.py_draft_tokens == draft_tokens
    assert llm_request.context_phase_params.draft_tokens == draft_tokens


def _make_broadcast_item(req_id):
    return RequestQueueItem(
        req_id,
        SimpleNamespace(
            py_logits_post_processors=None,
            py_multimodal_data=None,
            py_scheduling_params=None,
            py_num_logprobs=None,
            py_disaggregated_params=None,
            py_conversation_params=None,
            py_lora_path=None,
        ),
    )


def test_request_broadcaster_known_nonempty_skips_count_probe():
    """With known_nonempty (all ranks agree via the ADP allgather hint that
    rank 0 holds requests), the count probe is skipped and the result is
    identical to the probing path."""
    dist = Mock()
    dist.rank = 0
    dist.world_size = 1
    hang_detector = MagicMock()
    broadcaster = RequestBroadcaster(dist, hang_detector)
    probe = Mock(wraps=broadcaster._broadcast_request_count)
    broadcaster._broadcast_request_count = probe
    items = [_make_broadcast_item(1), _make_broadcast_item(2)]

    probed_requests, probed_py_objects = broadcaster.broadcast(items)
    assert probe.call_count == 1
    assert probed_requests == items

    hinted_requests, hinted_py_objects = broadcaster.broadcast(items, known_nonempty=True)
    assert probe.call_count == 1  # probe skipped
    assert hinted_requests == probed_requests
    assert hinted_py_objects == probed_py_objects


def test_request_broadcaster_empty_probing_path_unchanged():
    dist = Mock()
    dist.rank = 0
    dist.world_size = 1
    broadcaster = RequestBroadcaster(dist, MagicMock())

    new_requests, py_request_objects = broadcaster.broadcast([])

    assert new_requests == []
    assert py_request_objects is None


class _SingleRoundDist:
    """Fake transport recording the collectives RequestBroadcaster issues.

    ``broadcast_or_none`` behaves like the real one: the root's object comes
    back on the root, ``received`` stands for what a non-root rank gets.
    """

    def __init__(self, rank, *, world_size=2, has_pp=False, supports=True, received=None):
        self.rank = rank
        self.world_size = world_size
        self.tp_size = world_size
        self.cp_size = 1
        self.has_pp = has_pp
        # Single PP stage that is both first and last: the PP route then only
        # runs its intra-stage tp_cp_broadcast (no send/recv chain).
        self.pp_size = 2 if has_pp else 1
        self.is_first_pp_rank = True
        self.is_last_pp_rank = True
        self.supports_single_round_broadcast = supports
        self._received = received
        self.calls = []

    def tp_cp_broadcast(self, obj, root=0, **kwargs):
        self.calls.append(("tp_cp_broadcast", obj))
        return obj

    # Resolved (not called) by the PP route on a first-and-last stage.
    def send_object(self, *args, **kwargs):
        raise AssertionError("no PP send expected on a first-and-last stage")

    isend_object = recv_object = send_object

    def broadcast_or_none(self, obj, root=0):
        self.calls.append(("broadcast_or_none", obj))
        return obj if self.rank == root else self._received

    def broadcast(self, obj, root=0, prefer_cpu=False):
        self.calls.append(("broadcast", obj, prefer_cpu))
        return obj

    # Scalar request-count collectives (main routes them through the int64
    # helpers instead of the object broadcasts).
    def broadcast_int64(self, values, root=0, prefer_cpu=False):
        values = list(values)
        self.calls.append(("broadcast_int64", values, prefer_cpu))
        return values

    def tp_cp_broadcast_int64(self, values, root=0, **kwargs):
        values = list(values)
        self.calls.append(("tp_cp_broadcast_int64", values))
        return values


def test_single_round_empty_iteration_is_one_collective():
    dist = _SingleRoundDist(rank=0)
    hang_detector = MagicMock()
    broadcaster = RequestBroadcaster(dist, hang_detector)

    new_requests, py_request_objects = broadcaster.broadcast([])

    assert (new_requests, py_request_objects) == ([], None)
    assert dist.calls == [("broadcast_or_none", None)]
    hang_detector.pause.assert_called_once()


def test_single_round_busy_iteration_sends_requests_and_py_objects_once():
    dist = _SingleRoundDist(rank=0)
    broadcaster = RequestBroadcaster(dist, MagicMock())
    items = [_make_broadcast_item(1), _make_broadcast_item(2)]

    new_requests, py_request_objects = broadcaster.broadcast(items)

    assert new_requests == items
    assert py_request_objects == broadcaster._collect_py_objects(items)
    assert len(dist.calls) == 1
    kind, payload = dist.calls[0]
    assert kind == "broadcast_or_none"
    assert payload == (items, py_request_objects)


def test_single_round_non_root_receives_payload_or_nothing():
    items = [_make_broadcast_item(7)]
    py_objects = RequestBroadcaster(_SingleRoundDist(rank=0), MagicMock())._collect_py_objects(
        items
    )

    dist = _SingleRoundDist(rank=1, received=(items, py_objects))
    assert RequestBroadcaster(dist, MagicMock()).broadcast([]) == (items, py_objects)
    assert dist.calls == [("broadcast_or_none", None)]

    dist = _SingleRoundDist(rank=1, received=None)
    assert RequestBroadcaster(dist, MagicMock()).broadcast([]) == ([], None)


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"supports": False}, "transport without the single-round path"),
        ({"has_pp": True}, "pipeline route keeps probe + send/recv chain"),
        ({"world_size": 1}, "single rank never broadcasts"),
    ],
)
def test_single_round_falls_back_to_probe_when_not_applicable(kwargs, reason):
    dist = _SingleRoundDist(rank=0, **kwargs)
    broadcaster = RequestBroadcaster(dist, MagicMock())
    items = [_make_broadcast_item(1)]

    new_requests, _ = broadcaster.broadcast(items)

    assert new_requests == items, reason
    assert all(kind != "broadcast_or_none" for kind, *_ in dist.calls), reason


def test_single_round_idle_path_keeps_cpu_probe():
    """prefer_cpu marks the idle path: non-root ranks may park in the probe for
    minutes, so it must stay on the CPU transport (probe first, not the
    GPU-staged single round)."""
    dist = _SingleRoundDist(rank=0)
    broadcaster = RequestBroadcaster(dist, MagicMock())

    new_requests, py_request_objects = broadcaster.broadcast([], prefer_cpu=True)

    assert (new_requests, py_request_objects) == ([], None)
    assert dist.calls == [("broadcast_int64", [0], True)]


def test_single_round_known_nonempty_still_skips_straight_to_payload():
    dist = _SingleRoundDist(rank=0)
    broadcaster = RequestBroadcaster(dist, MagicMock())
    items = [_make_broadcast_item(1)]

    new_requests, py_request_objects = broadcaster.broadcast(items, known_nonempty=True)

    assert new_requests == items
    assert [kind for kind, *_ in dist.calls] == ["broadcast"]
    assert dist.calls[0][1] == (items, py_request_objects)


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
