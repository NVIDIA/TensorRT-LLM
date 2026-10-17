# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ADP (Attention Data Parallelism) abstractions.

Tests for:
- RankState serialization/deserialization
- ADPRouter interface and DefaultADPRouter
- Piggybacking per-rank iter-stats payloads on the existing ADP allgather
- Strict/relaxed attention-DP request routing while respecting rank capacity
"""

import inspect
import random
from collections import OrderedDict
from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.request_utils import get_from_waiting_queue
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
    ADPRouter,
    ConversationAwareADPRouter,
    DefaultADPRouter,
    KVCacheAwareADPRouter,
    RankIterStatsPayload,
    RankState,
    _num_input_tokens,
    build_active_requests_for_overlap,
    count_retiring_requests,
    is_retiring_request,
)
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.llmapi.llm_args import AttentionDpConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.scheduling_params import SchedulingParams

pytestmark = pytest.mark.cpu_only


class _MockRequest(MagicMock):
    """Mock executor Request whose ``num_input_tokens`` mirrors the real binding
    (it equals ``len(input_token_ids)``) instead of an auto-generated child mock,
    so the routers' cheap token-count reads work without setting it per call site.
    """

    @property
    def num_input_tokens(self):
        return len(self.input_token_ids)


def _mock_dist(tp_rank=0, tp_size=1, has_cp_helix=False, has_pp=False):
    """Create a mock Distributed object for testing."""
    dist = MagicMock()
    dist.tp_rank = tp_rank
    dist.tp_size = tp_size
    dist.has_cp_helix = has_cp_helix
    dist.mapping.has_pp.return_value = has_pp
    return dist


def create_mock_request_with_py_schedule_params(attention_dp_rank=None, attention_dp_relax=False):
    mock_request = _MockRequest()
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
    mock_request.py_conversation_params = None
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
    item.request = _MockRequest()
    item.request.py_scheduling_params = scheduling_params
    item.request.py_conversation_params = None
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
    """Helper: call DefaultADPRouter.route_requests with flat-list args."""
    tp_size = len(all_ranks_num_active_requests)
    states = [
        RankState(
            rank=i,
            num_active_requests=all_ranks_num_active_requests[i],
            num_active_tokens=all_ranks_num_active_tokens[i],
        )
        for i in range(tp_size)
    ]
    dist = _mock_dist(tp_size=tp_size)
    return DefaultADPRouter(dist=dist).route_requests(states, new_requests, max_num_active_requests)


@pytest.fixture
def attention_dp_config():
    return {"tp_size": 4, "max_num_active_requests": 8}


@pytest.fixture
def all_ranks_num_active_requests():
    return [2, 1, 3, 0]


@pytest.fixture
def all_ranks_num_active_tokens():
    return [10, 5, 15, 8]


def _retiring_request(prompt_len=100):
    """Active request that has produced its final token (state 14)."""
    return Mock(
        py_orig_prompt_len=prompt_len,
        cached_tokens=0,
        state=LlmRequestState.GENERATION_TO_COMPLETE,
    )


class TestBuildActiveRequestsForOverlap:
    def test_empty(self):
        assert build_active_requests_for_overlap([]) == []
        assert count_retiring_requests([]) == 0

    def test_drops_generation_to_complete(self):
        reqs = [_retiring_request(), _retiring_request(), _retiring_request()]
        assert build_active_requests_for_overlap(reqs) == []
        assert count_retiring_requests(reqs) == 3

    @pytest.mark.parametrize(
        "state",
        [
            LlmRequestState.CONTEXT_INIT,
            LlmRequestState.GENERATION_IN_PROGRESS,
            LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER,
            LlmRequestState.DISAGG_GENERATION_INIT,
            LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
            LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
            LlmRequestState.DISAGG_TRANS_ERROR,
        ],
    )
    def test_other_states_still_count_as_load(self, state):
        req = Mock(state=state)
        assert is_retiring_request(req) is False
        assert build_active_requests_for_overlap([req]) == [req]
        assert count_retiring_requests([req]) == 0

    def test_mixed_preserves_order_of_survivors(self):
        keep_a = Mock(state=LlmRequestState.GENERATION_IN_PROGRESS)
        keep_b = Mock(state=LlmRequestState.DISAGG_GENERATION_INIT)
        reqs = [keep_a, _retiring_request(), keep_b, _retiring_request()]
        assert build_active_requests_for_overlap(reqs) == [keep_a, keep_b]
        assert count_retiring_requests(reqs) == 2

    def test_returns_a_new_list(self):
        reqs = [Mock(state=LlmRequestState.GENERATION_IN_PROGRESS)]
        filtered = build_active_requests_for_overlap(reqs)
        assert filtered is not reqs
        filtered.clear()
        assert len(reqs) == 1

    def test_bare_mock_is_not_retiring(self):
        req = Mock(py_orig_prompt_len=10)
        assert build_active_requests_for_overlap([req]) == [req]
        assert count_retiring_requests([req]) == 0


class TestRankState:
    # RankState is the wire payload shared across attention-DP ranks. Keep its
    # serialization stable because iter-stats now ride on the same allgather.
    def test_creation(self):
        state = RankState(rank=0, num_active_requests=5, num_active_tokens=100)
        assert state.rank == 0
        assert state.num_active_requests == 5
        assert state.num_active_tokens == 100

    def test_serialize(self):
        state = RankState(rank=0, num_active_requests=5, num_active_tokens=100)
        assert state.serialize() == [0, 5, 100, 0, *RankIterStatsPayload().serialize(), 0, 0]

    def test_deserialize(self):
        state = RankState.deserialize(data=[2, 3, 50])
        assert state.rank == 2
        assert state.num_active_requests == 3
        assert state.num_active_tokens == 50

    def test_roundtrip(self):
        original = RankState(rank=1, num_active_requests=10, num_active_tokens=200)
        restored = RankState.deserialize(data=original.serialize())
        assert original == restored

    def test_roundtrip_with_retiring_requests(self):
        original = RankState(
            rank=1,
            num_active_requests=10,
            num_active_tokens=200,
            num_retiring_requests=3,
        )
        restored = RankState.deserialize(data=original.serialize())
        assert original == restored
        assert restored.num_retiring_requests == 3

    def test_defaults(self):
        state = RankState(rank=0)
        assert state.num_active_requests == 0
        assert state.num_active_tokens == 0
        assert state.num_retiring_requests == 0
        assert state.iter_stats.has_iter_stats == 0
        assert state.iter_stats.iter_stats_iter == -1

    def test_copy_iter_stats_from_clones_payload(self):
        state = RankState(rank=0)
        payload = RankIterStatsPayload(has_iter_stats=1, iter_stats_iter=7)

        state.copy_iter_stats_from(payload)

        assert state.iter_stats == payload
        assert state.iter_stats is not payload


class TestDefaultADPRouter:
    # Router tests model strict placement, relaxed placement, and capacity
    # handling so ADP ranks report consistent load and iter-stats state.
    def test_interface_compliance(self):
        router = DefaultADPRouter(dist=_mock_dist())
        assert isinstance(router, ADPRouter)

    def test_empty_requests(self):
        router = DefaultADPRouter(dist=_mock_dist())
        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        result, expected = router.route_requests(states, [], max_num_active_requests=10)
        assert result == {0: [], 1: []}
        assert expected >= 0

    def test_balanced_distribution(self):
        router = DefaultADPRouter(dist=_mock_dist())
        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        reqs = [_make_request_item(i, num_tokens=10) for i in range(4)]
        result, _ = router.route_requests(states, reqs, max_num_active_requests=10)
        total_assigned = sum(len(v) for v in result.values())
        assert total_assigned == 4
        assert abs(len(result[0]) - len(result[1])) <= 1

    def test_target_dp_rank_respected(self):
        router = DefaultADPRouter(dist=_mock_dist())
        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        req = _make_request_item(1, target_dp_rank=1, attention_dp_relax=False)
        result, _ = router.route_requests(states, [req], max_num_active_requests=10)
        assert len(result[1]) == 1
        assert result[1][0].id == 1

    def test_target_dp_rank_at_capacity_falls_through(self):
        router = DefaultADPRouter(dist=_mock_dist())
        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=2, num_active_tokens=20),
        ]
        req = _make_request_item(1, target_dp_rank=1, attention_dp_relax=False)
        result, _ = router.route_requests(states, [req], max_num_active_requests=2)
        assert len(result[0]) == 1
        assert len(result[1]) == 0

    def test_default_attention_dp_relax_is_relaxed(self):
        router = DefaultADPRouter(dist=_mock_dist())
        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        req_relax = _make_request_item(1, target_dp_rank=0)
        req_relax.request.py_scheduling_params = SchedulingParams(attention_dp_rank=0)
        req_strict = _make_request_item(2, target_dp_rank=0, attention_dp_relax=False)

        result, _ = router.route_requests(
            states, [req_relax, req_strict], max_num_active_requests=1
        )

        assert result[0] == [req_strict]
        assert req_relax in result[1]

    def test_favors_less_loaded_rank(self):
        router = DefaultADPRouter(dist=_mock_dist())
        states = [
            RankState(rank=0, num_active_requests=3, num_active_tokens=300),
            RankState(rank=1, num_active_requests=1, num_active_tokens=100),
        ]
        reqs = [_make_request_item(i, num_tokens=50) for i in range(4)]
        result, _ = router.route_requests(states, reqs, max_num_active_requests=10)
        assert len(result[1]) >= len(result[0])
        assert sum(len(v) for v in result.values()) == 4

    def test_single_rank(self):
        router = DefaultADPRouter(dist=_mock_dist())
        states = [RankState(rank=0, num_active_requests=0, num_active_tokens=0)]
        reqs = [_make_request_item(i) for i in range(5)]
        result, expected = router.route_requests(states, reqs, max_num_active_requests=10)
        assert len(result[0]) == 5
        assert expected == 5

    def test_four_ranks(self):
        router = DefaultADPRouter(dist=_mock_dist())
        states = [RankState(rank=i, num_active_requests=0, num_active_tokens=0) for i in range(4)]
        reqs = [_make_request_item(i, num_tokens=10) for i in range(8)]
        result, _ = router.route_requests(states, reqs, max_num_active_requests=10)
        total = sum(len(v) for v in result.values())
        assert total == 8
        for rank_reqs in result.values():
            assert len(rank_reqs) == 2

    def test_create_rank_state_default(self):
        dist = _mock_dist(tp_rank=0, has_cp_helix=False)
        router = DefaultADPRouter(dist=dist)
        req1 = Mock(py_orig_prompt_len=100)
        req2 = Mock(py_orig_prompt_len=200)
        state = router.create_rank_state(active_requests=[req1, req2], new_requests=[])
        assert state.rank == 0
        assert state.num_active_requests == 2
        assert state.num_active_tokens == 300

    def test_create_rank_state_does_not_filter_retiring_itself(self):
        dist = _mock_dist(tp_rank=0, has_cp_helix=False)
        router = DefaultADPRouter(dist=dist)
        active = [
            Mock(py_orig_prompt_len=100, state=LlmRequestState.GENERATION_IN_PROGRESS),
            _retiring_request(prompt_len=200),
        ]
        state = router.create_rank_state(active_requests=active, new_requests=[])
        assert state.num_active_requests == 2
        assert state.num_active_tokens == 300

    def test_gather_all_rank_states_excludes_retiring(self):
        dist = _mock_dist(tp_rank=0, has_cp_helix=False)
        router = DefaultADPRouter(dist=dist, has_seq_slot_headroom=True)
        active = [
            Mock(py_orig_prompt_len=100, state=LlmRequestState.GENERATION_IN_PROGRESS),
            _retiring_request(prompt_len=200),
            _retiring_request(prompt_len=300),
        ]
        dist.tp_allgather.side_effect = lambda payload: [payload]

        states = router.gather_all_rank_states(active_requests=active)

        assert len(states) == 1
        assert states[0].num_active_requests == 1
        assert states[0].num_retiring_requests == 2
        assert states[0].num_active_tokens == 100
        assert len(active) == 3

    def test_gather_all_rank_states_reports_zero_when_all_retiring(self):
        dist = _mock_dist(tp_rank=0, has_cp_helix=False)
        router = DefaultADPRouter(dist=dist, has_seq_slot_headroom=True)
        active = [_retiring_request(), _retiring_request()]
        dist.tp_allgather.side_effect = lambda payload: [payload]

        states = router.gather_all_rank_states(active_requests=active)

        assert states[0].num_active_requests == 0
        assert states[0].num_active_tokens == 0
        assert states[0].num_retiring_requests == 2

    def test_gather_all_rank_states_keeps_retiring_without_headroom(self):
        dist = _mock_dist(tp_rank=0, has_cp_helix=False, has_pp=True)
        router = DefaultADPRouter(dist=dist, has_seq_slot_headroom=False)
        assert router.exclude_retiring_requests is False
        active = [
            Mock(py_orig_prompt_len=100, state=LlmRequestState.GENERATION_IN_PROGRESS),
            _retiring_request(prompt_len=200),
            _retiring_request(prompt_len=300),
        ]
        dist.tp_allgather.side_effect = lambda payload: [payload]

        states = router.gather_all_rank_states(active_requests=active)

        assert states[0].num_active_requests == 3
        assert states[0].num_retiring_requests == 0
        assert states[0].num_active_tokens == 600

    def test_exclude_retiring_requests_follows_the_seat_pool_headroom(self):
        for has_pp in (False, True):
            dist = _mock_dist(has_pp=has_pp)
            assert (
                DefaultADPRouter(dist=dist, has_seq_slot_headroom=True).exclude_retiring_requests
                is True
            )
            assert (
                DefaultADPRouter(dist=dist, has_seq_slot_headroom=False).exclude_retiring_requests
                is False
            )

    def test_router_factory_requires_the_headroom_flag(self):
        params = inspect.signature(ADPRouter.create).parameters
        assert params["has_seq_slot_headroom"].default is inspect.Parameter.empty

    def test_router_constructors_default_to_no_headroom(self):
        for cls in (DefaultADPRouter, ConversationAwareADPRouter, KVCacheAwareADPRouter):
            default = inspect.signature(cls.__init__).parameters["has_seq_slot_headroom"].default
            assert default is False, cls.__name__

    @pytest.mark.parametrize("has_seq_slot_headroom", [True, False])
    def test_router_factory_propagates_the_headroom_flag(self, has_seq_slot_headroom):
        dist = _mock_dist(tp_size=2)
        mgr = Mock(enable_block_reuse=True)
        configs = [
            None,
            Mock(
                kv_cache_routing_conversation_affinity=False,
                enable_kv_cache_aware_routing=True,
                kv_cache_routing_load_balance_weight=1.0,
                kv_cache_routing_match_rate_threshold=0.1,
                kv_cache_routing_fair_share_multiplier=2.0,
                kv_cache_routing_cold_start_warmup=False,
                kv_cache_routing_account_for_in_transfer=False,
            ),
            Mock(
                kv_cache_routing_conversation_affinity=True,
                kv_cache_routing_max_sessions=1 << 16,
                kv_cache_routing_fair_share_multiplier=2.0,
                kv_cache_routing_new_conv_placement="round_robin",
            ),
        ]
        built = set()
        for attention_dp_config in configs:
            router = ADPRouter.create(
                dist=dist,
                has_seq_slot_headroom=has_seq_slot_headroom,
                kv_cache_manager=mgr,
                attention_dp_config=attention_dp_config,
            )
            built.add(type(router).__name__)
            assert router.exclude_retiring_requests is has_seq_slot_headroom
        # Anti-vacuity: the three configs must actually reach three branches.
        assert built == {
            "DefaultADPRouter",
            "KVCacheAwareADPRouter",
            "ConversationAwareADPRouter",
        }

    def test_create_rank_state_cp_helix(self):
        dist = _mock_dist(tp_rank=1, has_cp_helix=True)
        router = DefaultADPRouter(dist=dist)
        req1 = Mock(total_input_len_cp=150)
        req2 = Mock(total_input_len_cp=250)
        state = router.create_rank_state(active_requests=[req1, req2], new_requests=[])
        assert state.rank == 1
        assert state.num_active_requests == 2
        assert state.num_active_tokens == 400

    def test_gather_all_rank_states(self):
        dist = _mock_dist(tp_rank=0, tp_size=2, has_cp_helix=False)
        dist.tp_allgather.return_value = [[0, 1, 10], [1, 2, 20]]
        router = DefaultADPRouter(dist=dist)
        req = Mock(py_orig_prompt_len=10)
        states = router.gather_all_rank_states([req])
        assert len(states) == 2
        assert states[0] == RankState(rank=0, num_active_requests=1, num_active_tokens=10)
        assert states[1] == RankState(rank=1, num_active_requests=2, num_active_tokens=20)
        dist.tp_allgather.assert_called_once_with(
            RankState(rank=0, num_active_requests=1, num_active_tokens=10).serialize()
        )

    def test_gather_all_rank_states_piggybacks_iter_stats(self):
        dist = _mock_dist(tp_rank=0, tp_size=2, has_cp_helix=False)
        pending = RankIterStatsPayload(
            has_iter_stats=1,
            iter_stats_iter=7,
            num_context_requests=2,
            num_ctx_tokens=128,
            num_ctx_kv_tokens=16,
            num_gen_requests=3,
            num_gen_kv_tokens=1024,
            num_paused_requests=1,
            num_paused_kv_tokens=256,
        )
        expected_local = RankState(
            rank=0,
            num_active_requests=1,
            num_active_tokens=10,
            iter_stats=pending,
        )
        rank1 = RankState(rank=1, num_active_requests=2, num_active_tokens=20)
        dist.tp_allgather.return_value = [expected_local.serialize(), rank1.serialize()]

        router = DefaultADPRouter(dist=dist)
        req = Mock(py_orig_prompt_len=10)
        states = router.gather_all_rank_states([req], iter_stats_payload=pending)

        assert states[0] == expected_local
        assert states[1] == rank1
        dist.tp_allgather.assert_called_once_with(expected_local.serialize())

    def test_kv_load_payload_only_changes_wire_shape_when_enabled(self):
        payload = RankIterStatsPayload(
            kv_used_blocks=4,
            kv_total_blocks=10,
            kv_load_timestamp_ns=123,
        )
        state = RankState(
            rank=0,
            iter_stats=payload,
            num_context_requests=2,
            num_generation_requests=3,
        )

        disabled = state.serialize()
        assert len(disabled) == 15
        assert RankState.deserialize(disabled).num_context_requests == 2
        enabled = state.serialize(include_kv_cache_load=True)
        assert len(enabled) == 18
        restored = RankState.deserialize(enabled)
        assert restored.iter_stats.kv_used_blocks == 4
        assert restored.num_context_requests == 2
        assert restored.num_generation_requests == 3


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


def test_schedule_attention_dp_requests_serve_default_relax_None_does_not_crash(
    attention_dp_config, all_ranks_num_active_requests, all_ranks_num_active_tokens
):
    """Regression: trtllm-serve produces SchedulingParams with attention_dp_relax=None.

    openai_server.py builds ``SchedulingParams(agent_hierarchy=...)`` on every
    chat request, leaving ``attention_dp_rank`` and ``attention_dp_relax`` at
    their dataclass default of None. With >=2 such requests in one scheduling
    iteration, ``sorted(new_requests, key=get_relax_value)`` previously raised
    ``TypeError: '<' not supported between instances of 'NoneType' and
    'NoneType'`` because the key returned None for every item.

    Why prior coverage missed it:
      * Every ADP CI E2E test calls ``LLM(...).generate`` where
        ``request.scheduling_params`` is None, so ``base_worker`` leaves
        ``py_scheduling_params=None`` and ``get_relax_value`` short-circuits
        on the ``if scheduling_params is None`` branch.
      * The mock factory in this file defaults ``attention_dp_relax=False``,
        not None, so existing unit tests never construct the buggy shape.

    Use the real ``SchedulingParams`` dataclass so the test tracks the actual
    production default — if it ever flips back to None this stays accurate.
    """
    from tensorrt_llm.scheduling_params import SchedulingParams

    def _make_serve_shaped_request(req_id):
        mock_request = Mock()
        mock_request.py_scheduling_params = SchedulingParams()
        mock_request.input_token_ids = [1, 2, 3]
        return RequestQueueItem(req_id, mock_request)

    new_requests = [_make_serve_shaped_request(i) for i in range(4)]

    # No raise = sort key is None-safe. All four items have
    # attention_dp_rank=None so they fall into remaining_unscheduled and
    # then balance across ranks; with capacity 8 per rank they all land.
    all_ranks_new_requests, _ = _assign(
        all_ranks_num_active_requests,
        all_ranks_num_active_tokens,
        new_requests,
        attention_dp_config["max_num_active_requests"],
    )

    total_assigned = sum(len(reqs) for reqs in all_ranks_new_requests.values())
    assert total_assigned == 4


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
    dist = _mock_dist()
    result = DefaultADPRouter(dist=dist)._balance_requests_across_ranks(
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

    dist = _mock_dist()
    result = DefaultADPRouter(dist=dist)._balance_requests_across_ranks(
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

    dist = _mock_dist()
    result = DefaultADPRouter(dist=dist)._balance_requests_across_ranks(
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

    dist = _mock_dist()
    result = DefaultADPRouter(dist=dist)._balance_requests_across_ranks(
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

    dist = _mock_dist()
    result = DefaultADPRouter(dist=dist)._balance_requests_across_ranks(
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

    dist = _mock_dist()
    result = DefaultADPRouter(dist=dist)._balance_requests_across_ranks(
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


class TestKVCacheAwareADPRouterRouting:
    """Routing behavior of KVCacheAwareADPRouter."""

    def _make_router(
        self,
        tp_size=2,
        tp_rank=0,
        lbw=1.0,
        match_rate_threshold=0.1,
        fair_share_multiplier=2.0,
        prefix_matches=None,
    ):
        dist = _mock_dist(tp_rank=tp_rank, tp_size=tp_size)
        dist.tp_allgather = lambda data: [list(data) for _ in range(tp_size)]
        kv_cache_manager = MagicMock()
        kv_cache_manager.enable_block_reuse = True
        router = KVCacheAwareADPRouter(
            dist=dist,
            kv_cache_manager=kv_cache_manager,
            load_balance_weight=lbw,
            match_rate_threshold=match_rate_threshold,
            fair_share_multiplier=fair_share_multiplier,
        )
        if prefix_matches is None:
            # Default: rank 0 has 0 match, rank 1 has 50 match on req 1.
            prefix_matches = [
                {0: 0, 1: 0},
                {0: 0, 1: 50},
            ]
        router._all_ranks_prefix_matches = prefix_matches
        return router

    def _rank_states(self, tp_size, active_reqs=None, active_tokens=None):
        active_reqs = active_reqs or [0] * tp_size
        active_tokens = active_tokens or [0] * tp_size
        return [
            RankState(
                rank=r, num_active_requests=active_reqs[r], num_active_tokens=active_tokens[r]
            )
            for r in range(tp_size)
        ]

    def test_cache_affinity_wins(self):
        """50/100 = 50% match is well above the 0.1 default gate, so the
        request must land on the rank holding the cache (rank 1)."""
        router = self._make_router(tp_size=2, lbw=0.5)
        req = _make_request_item(req_id=1, num_tokens=100)
        result, _ = router.route_requests(self._rank_states(2), [req], max_num_active_requests=10)
        assert result[0] == []
        assert result[1] == [req]

    def test_default_attention_dp_relax_is_relaxed(self):
        router = self._make_router(tp_size=2)
        req_relax = _make_request_item(1, target_dp_rank=0)
        req_relax.request.py_scheduling_params = SchedulingParams(attention_dp_rank=0)
        req_strict = _make_request_item(2, target_dp_rank=0, attention_dp_relax=False)

        result, _ = router.route_requests(
            self._rank_states(2), [req_relax, req_strict], max_num_active_requests=1
        )

        assert result[0] == [req_strict]
        assert req_relax in result[1]

    def test_match_rate_threshold_gates_cache_affinity(self):
        """With rank 0 loaded but holding cache, and rank 1 idle with no
        cache:
        - threshold below the hit rate (0.5 < 0.6) keeps cache affinity
          active and routes to rank 0.
        - threshold above the hit rate (0.7 >= 0.6) suppresses affinity
          and the idle rank 1 wins on load.
        """
        prefix_matches = [{1: 60}, {1: 0}]
        states = [
            RankState(rank=0, num_active_requests=1, num_active_tokens=100),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]

        router_active = self._make_router(
            tp_size=2,
            lbw=0.1,
            match_rate_threshold=0.5,
            prefix_matches=prefix_matches,
        )
        req = _make_request_item(req_id=1, num_tokens=100)
        result_active, _ = router_active.route_requests(states, [req], max_num_active_requests=10)
        assert result_active[0] == [req]
        assert result_active[1] == []

        router_gated = self._make_router(
            tp_size=2,
            lbw=0.1,
            match_rate_threshold=0.7,
            prefix_matches=prefix_matches,
        )
        req = _make_request_item(req_id=1, num_tokens=100)
        result_gated, _ = router_gated.route_requests(states, [req], max_num_active_requests=10)
        assert result_gated[0] == []
        assert result_gated[1] == [req]

    def test_fair_share_multiplier_caps_per_rank(self):
        """With 8 requests all preferring rank 0 (cache hit), a loose
        multiplier lets rank 0 absorb them all; a strict multiplier=1.0
        evicts rank 0 at the fair share and spreads the remainder across
        the other ranks."""
        tp_size = 4
        # rank 0 has an 80-token match on every req; other ranks have none.
        prefix_matches = [{i: 80 for i in range(8)}] + [
            {i: 0 for i in range(8)} for _ in range(tp_size - 1)
        ]
        states = [
            RankState(rank=r, num_active_requests=0, num_active_tokens=0) for r in range(tp_size)
        ]

        # Loose (multiplier=4.0 -> cap = 4 * ceil(8/4) = 8): rank 0 takes all.
        router_loose = self._make_router(
            tp_size=tp_size,
            lbw=0.1,
            match_rate_threshold=0.0,
            fair_share_multiplier=4.0,
            prefix_matches=prefix_matches,
        )
        reqs_loose = [_make_request_item(req_id=i, num_tokens=100) for i in range(8)]
        result_loose, _ = router_loose.route_requests(
            states, reqs_loose, max_num_active_requests=100
        )
        assert len(result_loose[0]) == 8
        for r in range(1, tp_size):
            assert result_loose[r] == []

        # Strict (multiplier=1.0 -> cap = ceil(8/4) = 2): rank 0 evicted
        # after 2; remaining 6 spread across ranks 1..3.
        router_strict = self._make_router(
            tp_size=tp_size,
            lbw=0.1,
            match_rate_threshold=0.0,
            fair_share_multiplier=1.0,
            prefix_matches=prefix_matches,
        )
        reqs_strict = [_make_request_item(req_id=i, num_tokens=100) for i in range(8)]
        result_strict, _ = router_strict.route_requests(
            states, reqs_strict, max_num_active_requests=100
        )
        assert len(result_strict[0]) == 2
        assert sum(len(result_strict[r]) for r in range(1, tp_size)) == 6


def _make_conv_request_item(
    req_id,
    conversation_id,
    target_dp_rank=None,
    attention_dp_relax=True,
    num_tokens=10,
    subagent_affinity_id=None,
):
    """Mock RequestQueueItem carrying conversation params."""
    item = MagicMock()
    item.id = req_id
    item.child_req_ids = None
    scheduling_params = MagicMock()
    scheduling_params.attention_dp_rank = target_dp_rank
    scheduling_params.attention_dp_relax = attention_dp_relax
    # Default: no sub-agent affinity override (else a bare MagicMock attribute
    # would be truthy and shadow conversation_id in _conversation_id).
    scheduling_params.subagent_affinity_id = subagent_affinity_id
    item.request = _MockRequest()
    item.request.py_scheduling_params = scheduling_params
    item.request.py_conversation_params = None
    if conversation_id is not None:
        item.request.py_conversation_params = ConversationParams(conversation_id=conversation_id)
    item.request.input_token_ids = list(range(num_tokens))
    item.request.py_orig_prompt_len = num_tokens
    item.request.py_disaggregated_params = None
    return item


class TestConversationAwareADPRouterConversationId:
    """_conversation_id is the rank-affinity key: the sub-agent parent-affinity id
    (from scheduling_params, set at the disagg edge from SUBAGENT_AFFINITY_HEADER)
    when present, else the request's own conversation_id -- so sibling sub-agents
    co-locate on the parent's rank without their conversation_id being rewritten."""

    def test_prefers_subagent_affinity_id(self):
        item = _make_conv_request_item(1, "own-id", subagent_affinity_id="parent-id")
        assert ConversationAwareADPRouter._conversation_id(item) == "parent-id"

    def test_falls_back_to_conversation_id(self):
        item = _make_conv_request_item(1, "own-id")
        assert ConversationAwareADPRouter._conversation_id(item) == "own-id"

    def test_affinity_id_used_without_own_conversation_id(self):
        # Parent-only request: no own conversation_id, but the affinity id pins it.
        item = _make_conv_request_item(1, None, subagent_affinity_id="parent-id")
        assert ConversationAwareADPRouter._conversation_id(item) == "parent-id"

    def test_none_when_neither_present(self):
        item = _make_conv_request_item(1, None)
        assert ConversationAwareADPRouter._conversation_id(item) is None

    def test_siblings_collapse_to_parent_rank_key(self):
        # Two siblings with distinct own ids but the same parent affinity id must
        # yield the same routing key, so the router pins them to one rank.
        a = _make_conv_request_item(1, "own-a", subagent_affinity_id="parent-id")
        b = _make_conv_request_item(2, "own-b", subagent_affinity_id="parent-id")
        assert (
            ConversationAwareADPRouter._conversation_id(a)
            == ConversationAwareADPRouter._conversation_id(b)
            == "parent-id"
        )


class TestConversationAwareADPRouter:
    """Routing behavior of ConversationAwareADPRouter (explicit conv->rank)."""

    @staticmethod
    def _router(tp_size=4, max_sessions=1 << 16, placement="round_robin"):
        return ConversationAwareADPRouter(
            dist=_mock_dist(tp_size=tp_size),
            max_sessions=max_sessions,
            new_conv_placement=placement,
        )

    @staticmethod
    def _states(tp_size, active=None):
        active = active or [0] * tp_size
        return [
            RankState(rank=r, num_active_requests=active[r], num_active_tokens=active[r] * 10)
            for r in range(tp_size)
        ]

    @staticmethod
    def _route(router, states, items, cap=1000):
        assign, _ = router.route_requests(states, items, max_num_active_requests=cap)
        pos = {}
        for rank, lst in assign.items():
            for it in lst:
                pos[it.id] = rank
        return pos

    def test_interface_compliance(self):
        assert isinstance(self._router(), ADPRouter)

    def test_first_turn_round_robin(self):
        """First request of each conversation spreads one-per-rank (RR)."""
        router = self._router(tp_size=4)
        items = [_make_conv_request_item(i, f"conv{i}") for i in range(4)]
        pos = self._route(router, self._states(4), items)
        assert sorted(pos.values()) == [0, 1, 2, 3]

    @pytest.mark.parametrize("placement", ["round_robin", "least_queued", "least_tokens"])
    def test_subsequent_turns_are_sticky(self, placement: str) -> None:
        """Later turns of a conversation return to its first-turn rank."""
        router = self._router(tp_size=4, placement=placement)
        convs = ["A", "B", "C", "D"]
        first = [_make_conv_request_item(i, convs[i]) for i in range(4)]
        pos1 = self._route(router, self._states(4), first)
        home = {convs[i]: pos1[i] for i in range(4)}
        # A new batch of second turns (fresh ids) for the same conversations.
        second = [_make_conv_request_item(100 + i, convs[i]) for i in range(4)]
        pos2 = self._route(router, self._states(4), second)
        for i in range(4):
            assert pos2[100 + i] == home[convs[i]]

    def test_parent_and_child_turns_share_rank(self):
        router = self._router(tp_size=4)
        states = self._states(4)
        parent = _make_conv_request_item(1, "parent", target_dp_rank=2)
        assert self._route(router, states, [parent])[1] == 2

        turns = [
            _make_conv_request_item(2, "child-a", subagent_affinity_id="parent"),
            _make_conv_request_item(3, "child-b", subagent_affinity_id="parent"),
            _make_conv_request_item(4, "child-a", subagent_affinity_id="parent"),
        ]
        for turn in turns:
            assert self._route(router, states, [turn])[turn.id] == 2

    def test_no_conversation_id_falls_back_and_is_not_recorded(self):
        router = self._router(tp_size=4)
        items = [_make_conv_request_item(i, None) for i in range(8)]
        pos = self._route(router, self._states(4), items)
        assert len(pos) == 8
        # Nothing recorded for conversation-less requests.
        assert dict(router._conv_to_rank) == {}
        # Round-robin spread keeps ranks within one of each other.
        counts = [sum(1 for v in pos.values() if v == r) for r in range(4)]
        assert max(counts) - min(counts) <= 1

    @pytest.mark.parametrize("placement", ["round_robin", "least_queued", "least_tokens"])
    def test_routing_is_deterministic_across_ranks(self, placement: str) -> None:
        """Two independent instances (= two TP ranks) must agree exactly, or
        the no-broadcast allgather protocol would deadlock."""
        convs = ["A", "B", "C", None, "A", "B", "E", None, "C"]

        def run():
            router = self._router(tp_size=4, placement=placement)
            items = [
                _make_conv_request_item(i, convs[i], num_tokens=(i + 1) * 13)
                for i in range(len(convs))
            ]
            return self._route(router, self._states(4), items)

        assert run() == run()

    def test_lru_eviction_bounds_map(self):
        router = self._router(tp_size=2, max_sessions=2)
        items = [_make_conv_request_item(i, f"c{i}") for i in range(5)]
        self._route(router, self._states(2), items)
        assert len(router._conv_to_rank) == 2
        assert set(router._conv_to_rank) == {"c3", "c4"}

    def test_explicit_target_dp_rank_establishes_binding(self):
        router = self._router(tp_size=4)
        first = _make_conv_request_item(1, "A", target_dp_rank=2, attention_dp_relax=False)
        assert self._route(router, self._states(4), [first])[1] == 2
        assert router._conv_to_rank["A"] == 2

        second = _make_conv_request_item(2, "A")
        assert self._route(router, self._states(4), [second])[2] == 2

    def test_existing_binding_wins_over_later_explicit_target(self):
        router = self._router(tp_size=4)
        first = _make_conv_request_item(1, "A", target_dp_rank=2, attention_dp_relax=False)
        self._route(router, self._states(4), [first])

        second = _make_conv_request_item(2, "A", target_dp_rank=1, attention_dp_relax=False)
        assert self._route(router, self._states(4), [second])[2] == 2
        assert router._conv_to_rank["A"] == 2

    @pytest.mark.parametrize("placement", ["round_robin", "least_queued", "least_tokens"])
    def test_sticky_overflow_keeps_mapping(self, placement: str) -> None:
        """When the home rank is saturated at the HARD cap (max_num_active_requests),
        the turn overflows off home but the conversation stays mapped home (so it
        returns home next batch once that rank has capacity). Stickiness uses the
        hard cap -- not the soft fair-share -- so a conversation's KV prefix stays
        resident on one rank in the common case."""
        router = self._router(tp_size=2, placement=placement)
        # Seed conv A -> rank 0.
        self._route(router, self._states(2), [_make_conv_request_item(1, "A")])
        home = router._conv_to_rank["A"]
        # Fill the home rank to the hard cap, route A again -> must overflow off home...
        states = self._states(2)
        states[home] = RankState(rank=home, num_active_requests=5, num_active_tokens=50)
        pos = self._route(router, states, [_make_conv_request_item(2, "A")], cap=5)
        assert pos[2] != home
        # ...but the mapping is unchanged so it can return home later.
        assert router._conv_to_rank["A"] == home

    def test_create_rank_state(self):
        router = ConversationAwareADPRouter(dist=_mock_dist(tp_rank=2))
        req1 = Mock(py_orig_prompt_len=100)
        req2 = Mock(py_orig_prompt_len=50)
        state = router.create_rank_state(active_requests=[req1, req2], new_requests=[])
        assert state.rank == 2
        assert state.num_active_requests == 2
        assert state.num_active_tokens == 150

    def test_gather_all_rank_states_excludes_retiring(self):
        dist = _mock_dist(tp_rank=2)
        router = ConversationAwareADPRouter(dist=dist, has_seq_slot_headroom=True)
        active = [
            Mock(py_orig_prompt_len=100, state=LlmRequestState.GENERATION_IN_PROGRESS),
            _retiring_request(prompt_len=50),
        ]
        dist.tp_allgather.side_effect = lambda payload: [payload]

        states = router.gather_all_rank_states(active_requests=active)

        assert states[0].num_active_requests == 1
        assert states[0].num_retiring_requests == 1
        assert states[0].num_active_tokens == 100

    def test_factory_selects_conversation_router(self):
        cfg = MagicMock()
        cfg.kv_cache_routing_conversation_affinity = True
        cfg.kv_cache_routing_max_sessions = 8
        router = ADPRouter.create(
            dist=_mock_dist(),
            has_seq_slot_headroom=True,
            kv_cache_manager=None,
            attention_dp_config=cfg,
        )
        assert isinstance(router, ConversationAwareADPRouter)
        assert router._max_sessions == 8
        # A mocked (non-string) placement value must fall back to round_robin.
        assert router._new_conv_placement == "round_robin"

    @staticmethod
    def _lq_router(tp_size=4):
        return ConversationAwareADPRouter(
            dist=_mock_dist(tp_size=tp_size), new_conv_placement="least_queued"
        )

    def test_least_queued_places_new_conversation_on_least_loaded_rank(self):
        router = self._lq_router()
        pos = self._route(
            router, self._states(4, active=[5, 1, 3, 4]), [_make_conv_request_item(1, "A")]
        )
        assert pos[1] == 1
        assert router._conv_to_rank["A"] == 1

    def test_least_queued_fills_valleys_within_one_batch(self):
        """The shared count accumulator spreads a burst valley-first:
        [3, 0, 2, 3] + 4 new conversations ends level at [3, 3, 3, 3]."""
        items = [_make_conv_request_item(i, f"c{i}") for i in range(4)]
        pos = self._route(self._lq_router(), self._states(4, active=[3, 0, 2, 3]), items)
        placed = [sum(1 for v in pos.values() if v == r) for r in range(4)]
        assert placed == [0, 3, 1, 0]

    def test_least_queued_sticky_returns_unaffected(self):
        """A later turn returns to its pinned rank even when it is the busiest."""
        router = self._lq_router()
        home = self._route(
            router, self._states(4, active=[2, 0, 1, 1]), [_make_conv_request_item(1, "A")]
        )[1]
        assert home == 1
        pos = self._route(
            router, self._states(4, active=[0, 9, 0, 0]), [_make_conv_request_item(2, "A")], cap=100
        )
        assert pos[2] == home

    def test_least_queued_routing_is_deterministic_across_ranks(self):
        convs = ["A", "B", None, "A", "C", None, "B"]

        def run():
            items = [_make_conv_request_item(i, convs[i]) for i in range(len(convs))]
            return self._route(self._lq_router(), self._states(4, active=[2, 5, 0, 1]), items)

        assert run() == run()

    @pytest.mark.parametrize(
        "placement, expected_rank",
        [("round_robin", 0), ("least_queued", 1), ("least_tokens", 2)],
    )
    def test_new_conv_placement_config(self, placement: str, expected_rank: int) -> None:
        """The serialized public config selects each strategy."""
        cfg = AttentionDpConfig(
            kv_cache_routing_conversation_affinity=True,
            kv_cache_routing_max_sessions=8,
            kv_cache_routing_new_conv_placement=placement,
        )
        cfg = AttentionDpConfig.model_validate(cfg.model_dump())
        router = ADPRouter.create(
            dist=_mock_dist(tp_size=3),
            has_seq_slot_headroom=True,
            kv_cache_manager=None,
            attention_dp_config=cfg,
        )
        assert isinstance(router, ConversationAwareADPRouter)
        assert router._max_sessions == 8
        assert router._new_conv_placement == placement
        states = [
            RankState(rank=0, num_active_requests=3, num_active_tokens=1000),
            RankState(rank=1, num_active_requests=1, num_active_tokens=500),
            RankState(rank=2, num_active_requests=2, num_active_tokens=10),
        ]
        assert self._route(router, states, [_make_conv_request_item(1, "A")])[1] == expected_rank

    def test_unknown_new_conv_placement(self) -> None:
        with pytest.raises(ValueError, match="kv_cache_routing_new_conv_placement"):
            AttentionDpConfig(kv_cache_routing_new_conv_placement="banana")
        # Keep the direct-constructor fallback for callers without validated config.
        bad = ConversationAwareADPRouter(dist=_mock_dist(tp_size=4), new_conv_placement="banana")
        assert bad._new_conv_placement == "round_robin"

    @pytest.mark.parametrize("has_conversation_id", [True, False])
    def test_least_tokens_accumulates_same_batch_load(self, has_conversation_id: bool) -> None:
        router = self._router(tp_size=4, placement="least_tokens")
        items = [
            _make_conv_request_item(i, f"c{i}" if has_conversation_id else None, num_tokens=tokens)
            for i, tokens in enumerate([500, 500, 100, 100, 100])
        ]
        pos = self._route(router, self._states(4), items)
        assert [pos[i] for i in range(5)] == [0, 1, 2, 3, 2]
        if has_conversation_id:
            assert dict(router._conv_to_rank) == {f"c{i}": pos[i] for i in range(5)}
        else:
            assert dict(router._conv_to_rank) == {}

    def test_least_tokens_ties_use_request_count_then_rotating_cursor(self) -> None:
        router = self._router(tp_size=3, placement="least_tokens")
        states = [
            RankState(rank=r, num_active_requests=count, num_active_tokens=100)
            for r, count in enumerate([2, 1, 1])
        ]
        assert self._route(router, states, [_make_conv_request_item(1, "A")])[1] == 1
        # Fresh rank snapshots replace prior load; only the tie cursor persists.
        states = [RankState(rank=r, num_active_requests=1, num_active_tokens=100) for r in range(3)]
        assert self._route(router, states, [_make_conv_request_item(2, "B")])[2] == 2
        assert self._route(router, states, [_make_conv_request_item(3, "C")])[3] == 0

    @pytest.mark.parametrize("placement_source", ["explicit", "sticky"])
    def test_least_tokens_counts_explicit_and_sticky_batch_load(
        self, placement_source: str
    ) -> None:
        router = self._router(tp_size=2, placement="least_tokens")
        if placement_source == "sticky":
            self._route(router, self._states(2), [_make_conv_request_item(0, "A")])
            first = _make_conv_request_item(1, "A", num_tokens=500)
        else:
            first = _make_conv_request_item(
                1, "A", target_dp_rank=0, attention_dp_relax=False, num_tokens=500
            )
        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=1, num_active_tokens=100),
        ]
        pos = self._route(router, states, [first, _make_conv_request_item(2, "B")])
        assert pos == {1: 0, 2: 1}
        assert router._conv_to_rank["A"] == 0

    def test_factory_default_when_disabled(self):
        cfg = MagicMock()
        cfg.kv_cache_routing_conversation_affinity = False
        cfg.enable_kv_cache_aware_routing = False
        router = ADPRouter.create(
            dist=_mock_dist(),
            has_seq_slot_headroom=True,
            kv_cache_manager=None,
            attention_dp_config=cfg,
        )
        assert isinstance(router, DefaultADPRouter)

    @pytest.mark.parametrize("placement", ["round_robin", "least_queued", "least_tokens"])
    def test_returned_expected_covers_every_rank(self, placement: str) -> None:
        """Regression: returned expected_num_active_requests must be >= every
        rank's post-assignment active count, else
        py_executor._pad_attention_dp_dummy_request asserts and the executor
        hangs. A sticky storm (one conversation hammered) must not push its home
        rank past expected."""
        router = self._router(tp_size=8, placement=placement)
        states = self._states(8)
        # 40 requests for the SAME conversation: hard-cap stickiness concentrates
        # all of them on the home rank, pushing it well past the pre-loop soft
        # `expected`. The post-loop re-bump must lift `expected` to cover the home
        # rank's actual count, else _pad_attention_dp_dummy_request asserts.
        items = [_make_conv_request_item(i, "A") for i in range(40)]
        assign, expected = router.route_requests(states, items, max_num_active_requests=256)
        final = [states[r].num_active_requests + len(assign[r]) for r in range(8)]
        assert all(expected >= f for f in final), (expected, final)
        assert sum(len(v) for v in assign.values()) == 40  # nothing dropped

    @pytest.mark.parametrize("placement", ["round_robin", "least_queued", "least_tokens"])
    def test_new_conversations_never_exceed_slot_capacity(self, placement: str) -> None:
        """Regression: the spreading target must be clamped to
        max_num_active_requests. Unclamped it reaches 2 * fair_share, so a rank
        already holding every sequence slot still passes the `< soft_cap` gate
        and add_slot later raises NoFreeSlotsError mid-collective."""
        router = self._router(tp_size=4, placement=placement)
        cap = 64
        # 2 * ceil((184 + 8) / 4) = 96 > cap, so an unclamped target would
        # incorrectly admit the full rank 3.
        states = self._states(4, active=[40, 40, 40, cap])
        # Make the full rank token-light so least_tokens must check capacity.
        states[3].num_active_tokens = cap
        items = [_make_conv_request_item(i, None) for i in range(8)]
        assign, expected = router.route_requests(states, items, max_num_active_requests=cap)
        final = [states[r].num_active_requests + len(assign[r]) for r in range(4)]
        assert all(f <= cap for f in final), final
        assert expected <= cap
        assert sum(len(v) for v in assign.values()) == 8  # nothing dropped


class TestConversationAwareTokenAwarePlacement:
    """TLLM_ADP_TOKEN_AWARE_PLACEMENT=1: the ``rank is None`` branch of
    ConversationAwareADPRouter places by pinned prompt tokens (last-seen prompt
    length per live conversation), ties round-robin; load expires by write volume."""

    ENV = ConversationAwareADPRouter.TOKEN_AWARE_ENV
    WINDOW_ENV = ConversationAwareADPRouter.TOKEN_AWARE_WINDOW_ENV

    @staticmethod
    def _router(tp_size=4, window=0, max_sessions=1 << 16, placement="round_robin"):
        return ConversationAwareADPRouter(
            dist=_mock_dist(tp_size=tp_size),
            max_sessions=max_sessions,
            new_conv_placement=placement,
            token_aware_placement=True,
            token_window_tokens=window,
        )

    @staticmethod
    def _states(tp_size, active=None):
        active = active or [0] * tp_size
        return [
            RankState(rank=r, num_active_requests=active[r], num_active_tokens=0)
            for r in range(tp_size)
        ]

    @classmethod
    def _turns(cls, router, specs, tp_size=4, active=None, cap=256):
        """specs: (req_id, conversation_id or None, isl). Returns {req_id: rank}."""
        items = [_make_conv_request_item(rid, conv, num_tokens=isl) for rid, conv, isl in specs]
        assign, _ = router.route_requests(
            cls._states(tp_size, active), items, max_num_active_requests=cap
        )
        return {item.id: r for r, lst in assign.items() for item in lst}

    @staticmethod
    def _cfg(placement="round_robin", affinity=True, max_sessions=1 << 16):
        cfg = MagicMock()
        cfg.kv_cache_routing_conversation_affinity = affinity
        cfg.enable_kv_cache_aware_routing = False
        cfg.kv_cache_routing_max_sessions = max_sessions
        cfg.kv_cache_routing_fair_share_multiplier = 2.0
        cfg.kv_cache_routing_new_conv_placement = placement
        return cfg

    @staticmethod
    def _kvm(primary=32578, secondary=44362, tokens_per_block=64):
        kvm = MagicMock(spec=[])
        kvm.blocks_in_primary_pool = primary
        kvm.blocks_in_secondary_pool = secondary
        kvm.tokens_per_block = tokens_per_block
        return kvm

    @staticmethod
    def _check_invariants(router, tp_size):
        assert sum(router._pinned_tokens[:tp_size]) == sum(router._last_isl.values())
        assert all(v >= 0 for v in router._pinned_tokens)
        for r in range(tp_size):
            expected = {c for c in router._last_isl if router._conv_to_rank[c] == r}
            assert set(router._live[r]) == expected
        assert set(router._last_isl) <= set(router._conv_to_rank)

    # --- gating -----------------------------------------------------------------------------------------

    def test_default_off_is_bit_identical(self, monkeypatch):
        monkeypatch.delenv(self.ENV, raising=False)
        created = ADPRouter.create(
            dist=_mock_dist(tp_size=4), kv_cache_manager=None, attention_dp_config=self._cfg()
        )
        plain = ConversationAwareADPRouter(dist=_mock_dist(tp_size=4))
        assert created._token_aware is False and plain._token_aware is False
        assert not hasattr(created, "_pinned_tokens")
        assert created.pinned_tokens_per_rank() == []

        messages = []
        for name in ("debug", "info", "warning"):
            monkeypatch.setattr(
                logger, name, lambda *a, **k: messages.append(" ".join(str(x) for x in a))
            )
        batches = [
            [(1, "A", 1000), (2, "B", 10), (3, "C", 10), (4, "D", 10)],
            [(5, "E", 10), (6, "A", 1200), (7, None, 50)],
            [(8, "F", 300), (9, "B", 40)],
        ]
        positions = [
            [self._turns(router, batch, active=[i, 0, 1, 0]) for i, batch in enumerate(batches)]
            for router in (created, plain)
        ]
        assert positions[0] == positions[1]
        assert created._conv_to_rank == plain._conv_to_rank
        assert created._round_robin_cursor == plain._round_robin_cursor
        # The off path must not leak any token-aware log line (h00 runs at DEBUG).
        assert messages, "the pre-existing conversation debug line is expected"
        assert not any("token_aware" in m or "token-aware" in m for m in messages)

    def test_env_var_enables_via_factory(self, monkeypatch):
        monkeypatch.delenv(self.WINDOW_ENV, raising=False)
        monkeypatch.setenv(self.ENV, "1")
        router = ADPRouter.create(
            dist=_mock_dist(tp_size=4),
            kv_cache_manager=self._kvm(100, 50, 64),
            attention_dp_config=self._cfg(),
        )
        assert isinstance(router, ConversationAwareADPRouter)
        assert router._token_aware is True
        assert router._window_tokens == 150 * 64

        monkeypatch.setenv(self.ENV, "0")
        assert ADPRouter.create(dist=_mock_dist(), attention_dp_config=self._cfg())._token_aware is False
        monkeypatch.delenv(self.ENV)
        assert ADPRouter.create(dist=_mock_dist(), attention_dp_config=self._cfg())._token_aware is False

        # Set, but the conversation router is not selected -> default router and one warning.
        monkeypatch.setenv(self.ENV, "1")
        warn = MagicMock()
        monkeypatch.setattr(logger, "warning_once", warn)
        router = ADPRouter.create(
            dist=_mock_dist(), kv_cache_manager=None, attention_dp_config=self._cfg(affinity=False)
        )
        assert isinstance(router, DefaultADPRouter)
        warn.assert_called_once()
        assert warn.call_args.kwargs["key"] == "adp_token_aware_ignored"

    def test_env_overrides_least_queued(self):
        router = self._router(placement="least_queued")
        self._turns(router, [(1, "A", 100), (2, "B", 100), (3, "C", 100)])
        assert router.pinned_tokens_per_rank() == [100, 100, 100, 0]
        # least_queued would pick rank 1 (least active); token-aware picks the un-pinned rank 3.
        pos = self._turns(router, [(4, "D", 10)], active=[5, 1, 5, 5])
        assert pos[4] == 3

    # --- placement ------------------------------------------------------------------------------------

    def test_new_conversation_goes_to_least_pinned_rank(self):
        router = self._router()
        pos = self._turns(router, [(1, "A", 1000), (2, "B", 10), (3, "C", 10), (4, "D", 10)])
        assert pos == {1: 0, 2: 1, 3: 2, 4: 3}
        pos = self._turns(router, [(5, "E", 20)])
        assert pos[5] == 1  # ranks 1..3 tie at 10; the cursor (0 after four placements) scans 0,1,...
        assert router.pinned_tokens_per_rank() == [1000, 30, 10, 10]
        assert router._conv_to_rank["E"] == 1

    def test_ties_round_robin_not_lowest_index(self):
        router = self._router()
        pos = self._turns(router, [(i, f"c{i}", 10) for i in range(8)])
        assert [pos[i] for i in range(4)] == [0, 1, 2, 3]
        assert [sum(1 for r in pos.values() if r == k) for k in range(4)] == [2, 2, 2, 2]

    def test_sticky_return_replaces_last_seen_isl(self):
        router = self._router()
        self._turns(router, [(1, "A", 100)])
        assert router.pinned_tokens_per_rank() == [100, 0, 0, 0]
        self._turns(router, [(2, "A", 700)], active=[9, 0, 0, 0])
        assert router.pinned_tokens_per_rank() == [700, 0, 0, 0]
        assert router._written_tokens[0] == 700
        self._turns(router, [(3, "A", 650)])
        assert router.pinned_tokens_per_rank() == [650, 0, 0, 0]
        assert router._written_tokens[0] == 700  # shrink writes nothing new
        assert router._conv_to_rank["A"] == 0

    def test_within_batch_burst_spreads_by_tokens(self):
        router = self._router()
        pos = self._turns(
            router, [(1, "A", 500), (2, "B", 500), (3, "C", 100), (4, "D", 100), (5, "E", 100)]
        )
        assert [pos[i] for i in range(1, 6)] == [0, 1, 2, 3, 2]
        assert router.pinned_tokens_per_rank() == [500, 500, 200, 100]

    def test_sub_agent_children_are_independent_conversations(self):
        router = self._router()
        self._turns(router, [(1, "P", 500_000)])
        children = ["P::sa:subagent_001", "P::sa:subagent_002", "P::fa:003"]
        pos = self._turns(router, [(10 + i, c, 1000) for i, c in enumerate(children)])
        assert all(r != 0 for r in pos.values())
        assert len(set(pos.values())) == 3
        for c in children:
            assert router._conv_to_rank[c] == pos[10 + children.index(c)]
        assert router.pinned_tokens_per_rank()[0] == 500_000
        assert router._last_isl["P"] == 500_000

    # --- expiry and eviction --------------------------------------------------------------------------

    def test_expiry_by_write_volume(self):
        router = self._router(tp_size=1, window=1000)
        self._turns(router, [(1, "A", 600)], tp_size=1)
        assert router._live[0]["A"] == 600
        self._turns(router, [(2, None, 700)], tp_size=1)  # written 1300; 1300-600 <= 1000 -> stays
        self._turns(router, [(3, None, 400)], tp_size=1)  # written 1700
        assert "A" in router._last_isl
        self._turns(router, [], tp_size=1)  # next call: 1700-600 > 1000 -> A's load expires
        assert "A" not in router._last_isl
        assert "A" not in router._live[0]
        assert router.pinned_tokens_per_rank() == [0]
        assert router._conv_to_rank["A"] == 0  # stickiness survives
        self._turns(router, [(4, "A", 650)], tp_size=1)
        assert router.pinned_tokens_per_rank() == [650]
        assert router._live[0]["A"] == router._written_tokens[0]

    def test_expiry_disabled_when_window_zero(self):
        router = self._router(tp_size=1, window=0)
        self._turns(router, [(1, "A", 600)], tp_size=1)
        for i in range(10):
            self._turns(router, [(10 + i, None, 5000)], tp_size=1)
        self._turns(router, [], tp_size=1)
        assert router.pinned_tokens_per_rank() == [600]
        assert "A" in router._live[0]

    def test_lru_eviction_subtracts_pinned(self):
        router = self._router(tp_size=1, max_sessions=2)
        self._turns(router, [(1, "A", 100), (2, "B", 200), (3, "C", 300)], tp_size=1)
        assert list(router._conv_to_rank) == ["B", "C"]
        assert "A" not in router._last_isl and "A" not in router._live[0]
        assert router.pinned_tokens_per_rank() == [500]
        self._turns(router, [(4, "D", 400), (5, "E", 50)], tp_size=1)
        assert list(router._conv_to_rank) == ["D", "E"]
        assert router.pinned_tokens_per_rank() == [450]
        self._check_invariants(router, 1)

    def test_no_conversation_id_counts_writes_not_pinned(self):
        router = self._router()
        pos = self._turns(router, [(1, None, 50), (2, None, 70)])
        assert pos == {1: 0, 2: 1}  # ties -> round robin
        assert router.pinned_tokens_per_rank() == [0, 0, 0, 0]
        assert router._written_tokens[:4] == [50, 70, 0, 0]
        assert router._last_isl == {} and router._conv_to_rank == OrderedDict()

    def test_sticky_overflow_accounting(self):
        router = self._router(tp_size=2)
        self._turns(router, [(1, "A", 100)], tp_size=2, cap=1)
        assert router._conv_to_rank["A"] == 0
        # Home rank 0 is at the hard cap: A overflows to the least pinned rank without rebinding.
        pos = self._turns(router, [(2, "A", 300)], tp_size=2, active=[1, 0], cap=1)
        assert pos[2] == 1
        assert router._conv_to_rank["A"] == 0
        assert router.pinned_tokens_per_rank() == [100, 0]
        assert router._last_isl["A"] == 100
        assert router._written_tokens[:2] == [100, 300]

    # --- determinism, capacity, window ----------------------------------------------------------------

    def test_deterministic_across_ranks_with_expiry(self):
        routers = [self._router(window=3000) for _ in range(2)]
        rng = random.Random(7)
        req_id = 0
        for batch in range(40):
            specs = []
            for _ in range(rng.randint(0, 6)):
                req_id += 1
                conv = rng.choice([f"c{rng.randint(0, 15)}", None])
                specs.append((req_id, conv, rng.randint(1, 900)))
            active = [rng.randint(0, 3) for _ in range(4)]
            results = [
                self._turns(r, list(specs), active=list(active), cap=8) for r in routers
            ]
            assert results[0] == results[1]
            a, b = routers
            assert a._pinned_tokens == b._pinned_tokens
            assert a._written_tokens == b._written_tokens
            assert a._last_isl == b._last_isl
            assert a._conv_to_rank == b._conv_to_rank
            assert [list(x.items()) for x in a._live] == [list(x.items()) for x in b._live]
            assert a._round_robin_cursor == b._round_robin_cursor

    def test_returned_expected_covers_every_rank_token_aware(self):
        router = self._router(tp_size=8)
        states = self._states(8, active=[3, 0, 5, 1, 2, 0, 4, 1])
        items = [_make_conv_request_item(i, "A", num_tokens=100) for i in range(40)]
        assign, expected = router.route_requests(states, items, max_num_active_requests=256)
        final = [states[r].num_active_requests + len(assign[r]) for r in range(8)]
        assert all(expected >= f for f in final), (expected, final)
        assert sum(len(v) for v in assign.values()) == 40

    def test_window_derived_from_kv_manager_and_override(self, monkeypatch):
        monkeypatch.delenv(self.WINDOW_ENV, raising=False)
        warn = MagicMock()
        monkeypatch.setattr(logger, "warning", warn)
        make = lambda kvm, override=None: ConversationAwareADPRouter(
            dist=_mock_dist(tp_size=4),
            token_aware_placement=True,
            kv_cache_manager=kvm,
            token_window_tokens=override,
        )
        assert make(self._kvm())._window_tokens == (32578 + 44362) * 64 == 4_924_160
        warn.assert_not_called()
        # KVCacheManagerV2 shape: no secondary pool attribute, but a summed all-tier capacity.
        v2 = MagicMock(spec=[])
        v2.blocks_in_primary_pool = 100
        v2.tokens_per_block = 64
        v2.kv_capacity_tokens_all_tiers = 6_343_744
        assert make(v2)._window_tokens == 6_343_744
        warn.assert_not_called()
        # Neither shape (primary only) -> 0 + one warning.
        partial = MagicMock(spec=[])
        partial.blocks_in_primary_pool = 100
        partial.tokens_per_block = 64
        assert make(partial)._window_tokens == 0
        assert warn.call_count == 1
        assert make(None)._window_tokens == 0
        assert warn.call_count == 2
        # Env override wins over the manager; constructor override wins over env.
        monkeypatch.setenv(self.WINDOW_ENV, "123")
        assert make(self._kvm())._window_tokens == 123
        assert make(self._kvm(), override=77)._window_tokens == 77
        monkeypatch.setenv(self.WINDOW_ENV, "-5")
        assert make(self._kvm())._window_tokens == 0
        monkeypatch.setenv(self.WINDOW_ENV, "abc")
        assert make(self._kvm())._window_tokens == 4_924_160  # non-integer env ignored with a warning
        assert warn.call_count == 3

    def test_invariants_random_sequence(self):
        tp_size = 6
        router = self._router(tp_size=tp_size, window=5000, max_sessions=64)
        rng = random.Random(2026)
        req_id = 0
        for _ in range(250):
            specs = []
            for _ in range(rng.randint(0, 8)):
                req_id += 1
                conv = rng.choice([f"c{rng.randint(0, 120)}", None, None])
                specs.append((req_id, conv, rng.randint(1, 4000)))
            active = [rng.randint(0, 4) for _ in range(tp_size)]
            self._turns(router, specs, tp_size=tp_size, active=active, cap=6)
            self._check_invariants(router, tp_size)
        assert req_id > 800

class TestNumInputTokens:
    """The cheap token-count accessor used by the routers (avoids copying the
    full input_token_ids list just to count it)."""

    def test_none_request(self):
        assert _num_input_tokens(None) == 0

    def test_prefers_num_input_tokens_without_touching_token_list(self):
        """When num_input_tokens is available it must be used, and the heavy
        input_token_ids getter must NOT be accessed."""

        class _Req:
            num_input_tokens = 1234

            @property
            def input_token_ids(self):  # materializing it would be a regression
                raise AssertionError("input_token_ids must not be materialized")

        assert _num_input_tokens(_Req()) == 1234
