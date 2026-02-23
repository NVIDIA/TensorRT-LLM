"""Tests for ADP (Attention Data Parallelism) abstractions.

Tests for:
- RankState serialization/deserialization
- ADPRequestAssigner interface and DefaultADPRequestAssigner
- _balance_requests_across_ranks
"""

from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.request_utils import get_from_waiting_queue
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_request_assigner import (
    ADPRequestAssigner,
    DefaultADPRequestAssigner,
    RankState,
    _balance_requests_across_ranks,
)


def create_mock_request_with_py_schedule_params(attention_dp_rank=None, attention_dp_relax=False):
    mock_request = Mock()
    if attention_dp_rank is not None:
        mock_schedule_params = Mock()
        mock_schedule_params.attention_dp_rank = attention_dp_rank
        mock_schedule_params.attention_dp_relax = attention_dp_relax
        mock_schedule_params.configure_mock(
            attention_dp_rank=attention_dp_rank,
            attention_dp_relax=attention_dp_relax,
        )
        mock_request.py_scheduling_params = mock_schedule_params
    else:
        mock_request.py_scheduling_params = None
    mock_request.input_token_ids = [1, 2, 3]
    return mock_request


def _make_request_item(req_id, num_tokens=10, target_dp_rank=None, attention_dp_relax=True):
    """Create a mock RequestQueueItem for testing."""
    item = MagicMock()
    item.id = req_id
    item.child_req_ids = None
    scheduling_params = MagicMock()
    scheduling_params.attention_dp_rank = target_dp_rank
    scheduling_params.attention_dp_relax = attention_dp_relax
    item.request = MagicMock()
    item.request.py_scheduling_params = scheduling_params
    item.request.input_token_ids = list(range(num_tokens))
    return item


def append_to_waiting_queue(waiting_queue, rank, attention_dp_relax):
    req_id = len(waiting_queue)
    waiting_queue.append(
        RequestQueueItem(
            req_id,
            create_mock_request_with_py_schedule_params(
                attention_dp_rank=rank,
                attention_dp_relax=attention_dp_relax,
            ),
        )
    )


def _assign(
    all_ranks_num_active_requests,
    all_ranks_num_active_tokens,
    new_requests,
    max_num_active_requests,
):
    """Helper: call DefaultADPRequestAssigner with flat-list args."""
    tp_size = len(all_ranks_num_active_requests)
    states = [
        RankState(
            rank=i,
            num_active_requests=all_ranks_num_active_requests[i],
            num_active_tokens=all_ranks_num_active_tokens[i],
        )
        for i in range(tp_size)
    ]
    return DefaultADPRequestAssigner().assign_requests(
        states, new_requests, max_num_active_requests
    )


@pytest.fixture
def attention_dp_config():
    return {"tp_size": 4, "max_num_active_requests": 8}


@pytest.fixture
def all_ranks_num_active_requests():
    return [2, 1, 3, 0]


@pytest.fixture
def all_ranks_num_active_tokens():
    return [10, 5, 15, 8]


class TestRankState:
    def test_creation(self):
        state = RankState(rank=0, num_active_requests=5, num_active_tokens=100)
        assert state.rank == 0
        assert state.num_active_requests == 5
        assert state.num_active_tokens == 100

    def test_to_list(self):
        state = RankState(rank=0, num_active_requests=5, num_active_tokens=100)
        assert state.to_list() == [5, 100]

    def test_from_list(self):
        state = RankState.from_list(rank=2, data=[3, 50])
        assert state.rank == 2
        assert state.num_active_requests == 3
        assert state.num_active_tokens == 50

    def test_roundtrip(self):
        original = RankState(rank=1, num_active_requests=10, num_active_tokens=200)
        restored = RankState.from_list(rank=1, data=original.to_list())
        assert original == restored

    def test_defaults(self):
        state = RankState(rank=0)
        assert state.num_active_requests == 0
        assert state.num_active_tokens == 0


class TestDefaultADPRequestAssigner:
    def test_interface_compliance(self):
        assigner = DefaultADPRequestAssigner()
        assert isinstance(assigner, ADPRequestAssigner)

    def test_empty_requests(self):
        assigner = DefaultADPRequestAssigner()
        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        result, expected = assigner.assign_requests(states, [], max_num_active_requests=10)
        assert result == {0: [], 1: []}
        assert expected >= 0

    def test_balanced_distribution(self):
        assigner = DefaultADPRequestAssigner()
        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        reqs = [_make_request_item(i, num_tokens=10) for i in range(4)]
        result, _ = assigner.assign_requests(states, reqs, max_num_active_requests=10)
        total_assigned = sum(len(v) for v in result.values())
        assert total_assigned == 4
        assert abs(len(result[0]) - len(result[1])) <= 1

    def test_target_dp_rank_respected(self):
        assigner = DefaultADPRequestAssigner()
        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        req = _make_request_item(1, target_dp_rank=1, attention_dp_relax=False)
        result, _ = assigner.assign_requests(states, [req], max_num_active_requests=10)
        assert len(result[1]) == 1
        assert result[1][0].id == 1

    def test_target_dp_rank_at_capacity_falls_through(self):
        assigner = DefaultADPRequestAssigner()
        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=2, num_active_tokens=20),
        ]
        req = _make_request_item(1, target_dp_rank=1, attention_dp_relax=False)
        result, _ = assigner.assign_requests(states, [req], max_num_active_requests=2)
        assert len(result[0]) == 1
        assert len(result[1]) == 0

    def test_favors_less_loaded_rank(self):
        assigner = DefaultADPRequestAssigner()
        states = [
            RankState(rank=0, num_active_requests=3, num_active_tokens=300),
            RankState(rank=1, num_active_requests=1, num_active_tokens=100),
        ]
        reqs = [_make_request_item(i, num_tokens=50) for i in range(4)]
        result, _ = assigner.assign_requests(states, reqs, max_num_active_requests=10)
        assert len(result[1]) >= len(result[0])
        assert sum(len(v) for v in result.values()) == 4

    def test_single_rank(self):
        assigner = DefaultADPRequestAssigner()
        states = [RankState(rank=0, num_active_requests=0, num_active_tokens=0)]
        reqs = [_make_request_item(i) for i in range(5)]
        result, expected = assigner.assign_requests(states, reqs, max_num_active_requests=10)
        assert len(result[0]) == 5
        assert expected == 5

    def test_four_ranks(self):
        assigner = DefaultADPRequestAssigner()
        states = [RankState(rank=i, num_active_requests=0, num_active_tokens=0) for i in range(4)]
        reqs = [_make_request_item(i, num_tokens=10) for i in range(8)]
        result, _ = assigner.assign_requests(states, reqs, max_num_active_requests=10)
        total = sum(len(v) for v in result.values())
        assert total == 8
        for rank_reqs in result.values():
            assert len(rank_reqs) == 2


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

    all_ranks_new_requests, _ = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        new_requests,
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]
    assert len(result) == 2
    assert req1 in result
    assert req2 in result


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

    all_ranks_new_requests, _ = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        new_requests,
        attention_dp_config["max_num_active_requests"],
    )

    result = all_ranks_new_requests[0]
    assert len(result) == 0

    assert len(all_ranks_new_requests[1]) == 1
    assert req1 in all_ranks_new_requests[1]
    assert len(all_ranks_new_requests[2]) == 1
    assert req2 in all_ranks_new_requests[2]


def test_schedule_attention_dp_requests_unscheduled_requests(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True),
    )
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1, attention_dp_relax=True),
    )
    new_requests = [req1, req2]

    all_ranks_new_requests, _ = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        new_requests,
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]
    assert len(result) == 1
    assert req1 in result


def test_schedule_attention_dp_requests_unscheduled_no_capacity(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    all_ranks_num_active_requests[0] = 8
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True),
    )
    new_requests = [req1]

    all_ranks_new_requests, _ = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        new_requests,
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]
    assert len(result) == 0


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
        3,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True),
    )
    req_unscheduled_other = RequestQueueItem(
        4,
        create_mock_request_with_py_schedule_params(attention_dp_rank=2, attention_dp_relax=True),
    )
    new_requests = [
        req_scheduled_current,
        req_scheduled_other,
        req_unscheduled_current,
        req_unscheduled_other,
    ]

    all_ranks_new_requests, _ = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        new_requests,
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]
    assert len(result) == 2
    assert req_scheduled_current in result
    assert req_unscheduled_current in result


def test_schedule_attention_dp_requests_empty_lists(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    all_ranks_new_requests, _ = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        [],
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]
    assert len(result) == 0


def test_schedule_attention_dp_requests_expected_num_active_calculation(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True),
    )
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1, attention_dp_relax=True),
    )
    new_requests = [req1, req2]

    _, expected_num_active_requests = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        new_requests,
        attention_dp_config["max_num_active_requests"],
    )
    assert expected_num_active_requests == 3


def test_schedule_attention_dp_requests_relaxed_requests_distributed(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    """Test that relaxed requests are distributed via balancing."""
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True),
    )
    new_requests = [req1]

    all_ranks_new_requests, _ = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        new_requests,
        attention_dp_config["max_num_active_requests"],
    )
    total_assigned = sum(len(v) for v in all_ranks_new_requests.values())
    assert total_assigned == 1
    assert req1 in [r for reqs in all_ranks_new_requests.values() for r in reqs]


def test_schedule_attention_dp_requests_no_scheduling_when_capacity_exceeded(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    all_ranks_num_active_requests[0] = 8
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )
    new_requests = [req1]

    all_ranks_new_requests, _ = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        new_requests,
        attention_dp_config["max_num_active_requests"],
    )
    result = all_ranks_new_requests[0]
    assert len(result) == 0


def test_filter_and_schedule_integration(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    req_schedulable = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )
    req_schedulable.request.input_token_ids = [1, 2, 3, 4]
    req_relax = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True),
    )
    req_relax.request.input_token_ids = [1, 2]
    req_no_params = RequestQueueItem(
        3,
        create_mock_request_with_py_schedule_params(attention_dp_rank=None),
    )
    new_requests = [req_schedulable, req_relax, req_no_params]

    all_ranks_new_requests, _ = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        new_requests,
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

    all_ranks_new_requests, _ = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        new_requests,
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
    waiting_queue = FCFSWaitingQueue()
    waiting_queue.extend(req_list)
    available = max_num_active_requests * 4 - sum(all_ranks_num_active_requests)

    result = get_from_waiting_queue(
        waiting_queue,
        available,
        True,
        max_num_active_requests,
        all_ranks_num_active_requests,
    )
    assert len(result) == available


@pytest.mark.parametrize(
    "max_num_active_requests,all_ranks_num_active_requests,request_configs,all_ranks_expected_req_ids",
    [
        (3, [0, 0, 0, 0], [(None, True)] * 7, {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6]}),
        (3, [1, 2, 3, 0], [(None, True)] * 13, {0: [0, 1], 1: [2], 2: [], 3: [3, 4, 5]}),
        (
            3,
            [0, 0, 0, 0],
            [(None, True)] * 13,
            {0: [0, 1, 3], 1: [2, 4, 6], 2: [5, 7, 9], 3: [8, 10, 11]},
        ),
        (3, [3, 3, 3, 0], [], {0: [], 1: [], 2: [], 3: []}),
        (3, [3, 1, 3, 0], [(0, False), (0, True)], {0: [], 1: [1], 2: [], 3: []}),
        (3, [3, 2, 3, 3], [(0, False), (0, True)], {0: [], 1: [1], 2: [], 3: []}),
        (3, [2, 1, 3, 0], [(1, False), (3, False)], {0: [], 1: [0], 2: [], 3: [1]}),
        (3, [3, 3, 3, 1], [(0, True), (1, True), (2, True)], {0: [], 1: [], 2: [], 3: [0, 1]}),
        (3, [3, 3, 3, 0], [(0, False), (1, True), (3, False)], {0: [], 1: [], 2: [], 3: [2, 1]}),
    ],
)
def test_attention_dp_scheduling_cases(
    max_num_active_requests,
    all_ranks_num_active_requests,
    request_configs,
    all_ranks_expected_req_ids,
):
    """Test attention DP scheduling with various scenarios."""
    waiting_queue = FCFSWaitingQueue()
    for rank, relax in request_configs:
        append_to_waiting_queue(waiting_queue, rank, relax)

    num_ranks = len(all_ranks_num_active_requests)
    total_num_active_requests = sum(all_ranks_num_active_requests)
    total_max = max_num_active_requests * num_ranks

    new_requests = get_from_waiting_queue(
        waiting_queue,
        total_max - total_num_active_requests,
        True,
        max_num_active_requests,
        all_ranks_num_active_requests,
    )

    all_ranks_num_active_tokens = [10 + i * 5 for i in range(num_ranks)]
    all_ranks_new_requests, _ = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        new_requests,
        max_num_active_requests,
    )

    assert len(all_ranks_new_requests) == num_ranks
    for rank, reqs in all_ranks_new_requests.items():
        req_ids = [req.id for req in reqs]
        assert req_ids == all_ranks_expected_req_ids[rank]


def test_balance_requests_across_ranks_empty_requests():
    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    result = _balance_requests_across_ranks(
        [],
        all_ranks_new_requests,
        [2, 1, 3, 0],
        [20, 10, 30, 5],
        3,
    )
    for rank in range(4):
        assert len(result[rank]) == 0


def test_balance_requests_across_ranks_single_request():
    """Test balance_requests_across_ranks with a single request."""
    req = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=None),
    )
    req.request.input_token_ids = [1, 2, 3, 4, 5]  # 5 tokens
    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [1, 2, 0, 1]  # Rank 2 has lowest count
    all_ranks_num_active_tokens = [10, 20, 5, 15]
    expected_num_active_requests = 2

    result = _balance_requests_across_ranks(
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
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=None),
    )
    req1.request.input_token_ids = [1, 2, 3]  # 3 tokens

    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=None),
    )
    req2.request.input_token_ids = [1, 2, 3, 4, 5, 6]  # 6 tokens

    req3 = RequestQueueItem(
        3,
        create_mock_request_with_py_schedule_params(attention_dp_rank=None),
    )
    req3.request.input_token_ids = [1, 2]  # 2 tokens

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [0, 1, 2, 1]
    all_ranks_num_active_tokens = [5, 15, 25, 10]
    expected_num_active_requests = 2

    result = _balance_requests_across_ranks(
        [req1, req2, req3],
        all_ranks_new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        expected_num_active_requests,
    )

    # Requests sorted by token count (descending), assigned to ranks with lowest active count
    # req2 (6 tokens) -> rank 0 (0 active)
    # req3 (2 tokens) -> rank 0 (1 active, still has capacity)
    # req1 (3 tokens) -> rank 3 (1 active)
    assert len(result[0]) == 2
    assert len(result[1]) == 0
    assert len(result[2]) == 0
    assert len(result[3]) == 1

    assert result[0][0] == req2
    assert result[0][1] == req3
    assert result[3][0] == req1


def test_balance_requests_across_ranks_capacity_limits():
    """Test balance_requests_across_ranks respects capacity limits."""
    requests = []
    for i in range(4):
        req = RequestQueueItem(
            i,
            create_mock_request_with_py_schedule_params(attention_dp_rank=None),
        )
        req.request.input_token_ids = [1] * (i + 1)  # Variable token counts
        requests.append(req)

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    all_ranks_num_active_requests = [1, 1, 1, 1]  # All ranks start with 1
    all_ranks_num_active_tokens = [10, 10, 10, 10]
    expected_num_active_requests = 2

    result = _balance_requests_across_ranks(
        requests,
        all_ranks_new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        expected_num_active_requests,
    )

    # Each rank can only take 1 more request (1 + 1 = 2 = expected_num_active_requests)
    total_assigned = sum(len(rank_requests) for rank_requests in result.values())
    assert total_assigned == 4

    for rank in range(4):
        assert len(result[rank]) <= 1


def test_balance_requests_across_ranks_heap_ordering():
    """Test that balance_requests_across_ranks uses heap ordering correctly."""
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=None),
    )
    req1.request.input_token_ids = [1, 2, 3]  # 3 tokens

    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=None),
    )
    req2.request.input_token_ids = [1, 2, 3]  # 3 tokens

    req3 = RequestQueueItem(
        3,
        create_mock_request_with_py_schedule_params(attention_dp_rank=None),
    )
    req3.request.input_token_ids = [1, 2, 3]  # 3 tokens

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}
    # Rank 0 has highest active count, should get requests last
    all_ranks_num_active_requests = [3, 1, 0, 2]
    all_ranks_num_active_tokens = [30, 10, 5, 20]
    expected_num_active_requests = 4

    result = _balance_requests_across_ranks(
        [req1, req2, req3],
        all_ranks_new_requests,
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        expected_num_active_requests,
    )

    # Requests assigned in order of lowest active count first
    # Rank 2: 0 active -> gets req1 and req2 (has capacity for 4)
    # Rank 1: 1 active -> gets req3
    # Rank 3: 2 active -> gets nothing
    # Rank 0: 3 active -> gets nothing
    assert len(result[0]) == 0
    assert len(result[1]) == 1
    assert len(result[2]) == 2
    assert len(result[3]) == 0

    assert result[1][0] == req3
    assert result[2][0] == req1
    assert result[2][1] == req2


def test_balance_requests_across_ranks_token_count_sorting():
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=None),
    )
    req1.request.input_token_ids = [1]

    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=None),
    )
    req2.request.input_token_ids = [1, 2, 3, 4, 5]

    req3 = RequestQueueItem(
        3,
        create_mock_request_with_py_schedule_params(attention_dp_rank=None),
    )
    req3.request.input_token_ids = [1, 2, 3]

    all_ranks_new_requests = {0: [], 1: [], 2: [], 3: []}

    result = _balance_requests_across_ranks(
        [req1, req2, req3],
        all_ranks_new_requests,
        [0, 0, 0, 0],
        [5, 5, 5, 5],
        2,
    )

    assert len(result[0]) == 1
    assert len(result[1]) == 1
    assert len(result[2]) == 1
    assert len(result[3]) == 0
    assert result[0][0] == req2
    assert result[1][0] == req3
    assert result[2][0] == req1
