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
"""Tests for KV cache-aware ADP router and probe_prefix_match_length.

These tests use mock objects and do NOT require GPU.
"""

from unittest.mock import MagicMock, Mock

from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
    ADPRouter,
    KVCacheAwareADPRouter,
    RankState,
)

# ---- Helpers ----


def _mock_dist(tp_rank=0, tp_size=1, has_cp_helix=False):
    """Create a mock Distributed object for testing."""
    dist = MagicMock()
    dist.tp_rank = tp_rank
    dist.tp_size = tp_size
    dist.has_cp_helix = has_cp_helix
    return dist


def _make_request_item(
    req_id, num_tokens=10, target_dp_rank=None, attention_dp_relax=True, lora_task_id=None
):
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
    if lora_task_id is not None:
        lora_config = MagicMock()
        lora_config.task_id = lora_task_id
        item.request.lora_config = lora_config
    else:
        item.request.lora_config = None
    return item


def _mock_kv_cache_manager(probe_results=None):
    """Create mock KV cache manager with configurable probe results.

    Args:
        probe_results: dict mapping (tuple(input_tokens), lora_task_id) -> match_length.
            If None, all probes return 0.
    """
    mgr = MagicMock()
    probe_results = probe_results or {}

    def mock_probe(input_tokens, lora_task_id=None):
        key = (tuple(input_tokens), lora_task_id)
        return probe_results.get(key, 0)

    mgr.probe_prefix_match_length = Mock(side_effect=mock_probe)
    return mgr


# ---- Tests for KVCacheAwareADPRouter ----


class TestKVCacheAwareADPRouter:
    """Tests for the KV cache-aware ADP router."""

    def test_is_adp_router(self):
        dist = _mock_dist()
        mgr = _mock_kv_cache_manager()
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)
        assert isinstance(router, ADPRouter)

    def test_create_rank_state(self):
        dist = _mock_dist(tp_rank=0)
        mgr = _mock_kv_cache_manager()
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)

        req1 = Mock(py_orig_prompt_len=100, cached_tokens=0)
        req2 = Mock(py_orig_prompt_len=200, cached_tokens=0)
        state = router.create_rank_state([req1, req2], [])
        assert state.rank == 0
        assert state.num_active_requests == 2
        assert state.num_active_tokens == 300

    def test_create_rank_state_cp_helix(self):
        dist = _mock_dist(tp_rank=1, has_cp_helix=True)
        mgr = _mock_kv_cache_manager()
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)

        req1 = Mock(total_input_len_cp=150, cached_tokens=0)
        state = router.create_rank_state([req1], [])
        assert state.rank == 1
        assert state.num_active_tokens == 150

    # -- gather_prefix_matches tests --

    def test_gather_prefix_matches_single_rank(self):
        tokens_a = list(range(100))
        tokens_b = list(range(50))
        probe_results = {
            (tuple(tokens_a[:-1]), None): 64,
            (tuple(tokens_b[:-1]), None): 0,
        }

        dist = _mock_dist(tp_rank=0, tp_size=1)
        dist.tp_allgather = Mock(side_effect=lambda x: [x])
        mgr = _mock_kv_cache_manager(probe_results)
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)

        req_a = _make_request_item(1, num_tokens=100)
        req_b = _make_request_item(2, num_tokens=50)

        router.gather_prefix_matches([req_a, req_b])
        assert len(router._all_ranks_prefix_matches) == 1
        assert router._all_ranks_prefix_matches[0] == {1: 64, 2: 0}

    def test_gather_prefix_matches_two_ranks(self):
        tokens_a = list(range(100))
        probe_results = {
            (tuple(tokens_a[:-1]), None): 64,
        }

        dist = _mock_dist(tp_rank=0, tp_size=2)
        # Simulate allgather: rank 0 sends [1, 64, 2, 0], rank 1 sends [1, 32, 2, 0]
        dist.tp_allgather = Mock(return_value=[[1, 64, 2, 0], [1, 32, 2, 0]])
        mgr = _mock_kv_cache_manager(probe_results)
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)

        req_a = _make_request_item(1, num_tokens=100)
        req_b = _make_request_item(2, num_tokens=50)

        router.gather_prefix_matches([req_a, req_b])
        assert len(router._all_ranks_prefix_matches) == 2
        assert router._all_ranks_prefix_matches[0] == {1: 64, 2: 0}
        assert router._all_ranks_prefix_matches[1] == {1: 32, 2: 0}

    def test_gather_prefix_matches_with_lora(self):
        tokens_a = list(range(100))
        probe_results = {
            (tuple(tokens_a[:-1]), 42): 64,
        }

        dist = _mock_dist(tp_rank=0, tp_size=1)
        dist.tp_allgather = Mock(side_effect=lambda x: [x])
        mgr = _mock_kv_cache_manager(probe_results)
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)

        req_a = _make_request_item(1, num_tokens=100, lora_task_id=42)
        router.gather_prefix_matches([req_a])
        assert router._all_ranks_prefix_matches[0] == {1: 64}

    def test_gather_prefix_matches_empty(self):
        dist = _mock_dist(tp_rank=0, tp_size=1)
        dist.tp_allgather = Mock(side_effect=lambda x: [x])
        mgr = _mock_kv_cache_manager()
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)

        router.gather_prefix_matches([])
        assert len(router._all_ranks_prefix_matches) == 1
        assert router._all_ranks_prefix_matches[0] == {}

    # -- route_requests tests --

    def test_route_prefers_cached_rank(self):
        """Request with cache hit on rank 0 → routed to rank 0."""
        dist = _mock_dist(tp_rank=0, tp_size=2)
        mgr = _mock_kv_cache_manager()
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)

        router._all_ranks_prefix_matches = [
            {1: 80},  # rank 0: 80 tokens cached
            {1: 0},  # rank 1: no cache
        ]

        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        req = _make_request_item(1, num_tokens=100)

        result, _ = router.route_requests(states, [req], max_num_active_requests=10)
        assert len(result[0]) == 1  # routed to rank 0
        assert len(result[1]) == 0

    def test_route_degenerates_to_load_balance_no_cache(self):
        """No cache hits → routes to least loaded rank."""
        dist = _mock_dist(tp_rank=0, tp_size=2)
        mgr = _mock_kv_cache_manager()
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)

        router._all_ranks_prefix_matches = [{1: 0}, {1: 0}]

        states = [
            RankState(rank=0, num_active_requests=5, num_active_tokens=500),
            RankState(rank=1, num_active_requests=1, num_active_tokens=100),
        ]
        req = _make_request_item(1, num_tokens=100)

        result, _ = router.route_requests(states, [req], max_num_active_requests=10)
        assert len(result[1]) == 1  # rank 1 is less loaded
        assert len(result[0]) == 0

    def test_route_load_overcomes_cache(self):
        """Heavy load on cached rank → routes to idle rank despite no cache."""
        dist = _mock_dist(tp_rank=0, tp_size=2)
        mgr = _mock_kv_cache_manager()
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr, load_balance_weight=10.0)

        router._all_ranks_prefix_matches = [{1: 80}, {1: 0}]

        states = [
            RankState(rank=0, num_active_requests=5, num_active_tokens=5000),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        req = _make_request_item(1, num_tokens=100)

        result, _ = router.route_requests(states, [req], max_num_active_requests=10)
        # total_load=5000, load_denom=max(5000, 100)=5000
        # score(rank0) = (100-80) + 10 * (5000/5000 * 100) = 20 + 1000 = 1020
        # score(rank1) = (100-0)  + 10 * (0/5000 * 100)    = 100 + 0   = 100
        assert len(result[1]) == 1
        assert len(result[0]) == 0

    def test_route_respects_explicit_dp_rank(self):
        """Non-relaxed request with explicit dp_rank → honored."""
        dist = _mock_dist(tp_rank=0, tp_size=2)
        mgr = _mock_kv_cache_manager()
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)

        router._all_ranks_prefix_matches = [{1: 80}, {1: 0}]

        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        req = _make_request_item(1, num_tokens=100, target_dp_rank=1, attention_dp_relax=False)

        result, _ = router.route_requests(states, [req], max_num_active_requests=10)
        assert len(result[1]) == 1  # forced to rank 1
        assert len(result[0]) == 0

    def test_route_multiple_requests_effective_token_update(self):
        """After routing, rank load increases by effective (not full) tokens.

        Uses 4 requests so expected_num_active_requests=2, allowing 2 cached
        requests to land on rank 0 before the capacity ceiling kicks in.
        If load were updated with full tokens (100) instead of effective (20),
        the 2nd cached request would score 120 on rank 0 vs 100 on rank 1
        and go to rank 1 instead.
        """
        dist = _mock_dist(tp_rank=0, tp_size=2)
        mgr = _mock_kv_cache_manager()
        # Pin fair_share_multiplier=1.0 so the cap equals fair_share (2).
        # This test isolates the effective-token bookkeeping from the
        # cap-relaxation behavior covered separately in
        # test_fair_share_multiplier_caps_per_rank.
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr, fair_share_multiplier=1.0)

        # Requests 1,2 have cache on rank 0; requests 3,4 have no cache
        router._all_ranks_prefix_matches = [
            {1: 80, 2: 80, 3: 0, 4: 0},  # rank 0
            {1: 0, 2: 0, 3: 0, 4: 0},  # rank 1
        ]

        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        req_a = _make_request_item(1, num_tokens=100)
        req_b = _make_request_item(2, num_tokens=100)
        req_c = _make_request_item(3, num_tokens=100)
        req_d = _make_request_item(4, num_tokens=100)

        result, _ = router.route_requests(
            states, [req_a, req_b, req_c, req_d], max_num_active_requests=10
        )
        # expected_num_active_requests = max((0+4+1)//2, 0) = 2
        # req_a → rank 0: total_load=0, load_denom=max(0,100)=100
        #   score(0)=(100-80)+1.0*(0/100*100)=20, score(1)=100+0=100 → rank0
        #   active_tokens[0] += 20 → [20, 0]
        # req_b → rank 0: total_load=20, load_denom=max(20,100)=100
        #   score(0)=(100-80)+1.0*(20/100*100)=40, score(1)=100+0=100 → rank0
        #   active_tokens[0] += 20 → [40, 0]; rank0 at capacity (2), removed
        # req_c → rank 1 (only eligible): active_tokens[1] += 100
        # req_d → rank 1: active_tokens[1] += 100
        assert len(result[0]) == 2  # cached requests on rank 0
        assert result[0][0].id == 1
        assert result[0][1].id == 2
        assert len(result[1]) == 2  # uncached requests on rank 1
        assert result[1][0].id == 3
        assert result[1][1].id == 4

    def test_route_empty_requests(self):
        dist = _mock_dist(tp_rank=0, tp_size=2)
        mgr = _mock_kv_cache_manager()
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)
        router._all_ranks_prefix_matches = [{}, {}]

        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]

        result, expected = router.route_requests(states, [], max_num_active_requests=10)
        assert result == {0: [], 1: []}
        assert expected >= 0

    def test_route_four_ranks_balanced(self):
        """4 ranks, no cache hits → balanced distribution."""
        dist = _mock_dist(tp_rank=0, tp_size=4)
        mgr = _mock_kv_cache_manager()
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)
        router._all_ranks_prefix_matches = [{}, {}, {}, {}]

        states = [RankState(rank=i, num_active_requests=0, num_active_tokens=0) for i in range(4)]
        reqs = [_make_request_item(i, num_tokens=10) for i in range(8)]

        result, _ = router.route_requests(states, reqs, max_num_active_requests=10)
        total = sum(len(v) for v in result.values())
        assert total == 8
        for rank_reqs in result.values():
            assert len(rank_reqs) == 2

    def test_route_capacity_limit_respected(self):
        """Requests beyond max capacity are not assigned."""
        dist = _mock_dist(tp_rank=0, tp_size=2)
        mgr = _mock_kv_cache_manager()
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)
        router._all_ranks_prefix_matches = [{1: 0, 2: 0, 3: 0}, {1: 0, 2: 0, 3: 0}]

        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        reqs = [_make_request_item(i, num_tokens=10) for i in range(1, 4)]

        result, _ = router.route_requests(states, reqs, max_num_active_requests=2)
        total = sum(len(v) for v in result.values())
        assert total == 3  # all assigned (expected = max(ceil(3/2), 0) = 2 per rank)

    def test_route_mixed_cache_and_no_cache(self):
        """Some requests have cache hits, some don't → smart routing."""
        dist = _mock_dist(tp_rank=0, tp_size=2)
        mgr = _mock_kv_cache_manager()
        router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=mgr)

        # req 1 has cache on rank 0, req 2 has no cache anywhere
        router._all_ranks_prefix_matches = [
            {1: 80, 2: 0},  # rank 0
            {1: 0, 2: 0},  # rank 1
        ]

        states = [
            RankState(rank=0, num_active_requests=0, num_active_tokens=0),
            RankState(rank=1, num_active_requests=0, num_active_tokens=0),
        ]
        req_a = _make_request_item(1, num_tokens=100)  # cache hit on rank 0
        req_b = _make_request_item(2, num_tokens=100)  # no cache

        result, _ = router.route_requests(states, [req_a, req_b], max_num_active_requests=10)
        # req_a → rank 0 (cache hit: effective=20 vs 100)
        #   active_tokens → [20, 0]
        # req_b → rank 1 (both ranks have 0 cache; rank 0 carries 20 tokens)
        #   total_load=20, load_denom=max(20,100)=100
        #   score(rank0) = 100 + 1.0*(20/100*100) = 120
        #   score(rank1) = 100 + 1.0*(0/100*100)  = 100
        assert len(result[0]) == 1
        assert result[0][0].id == 1
        assert len(result[1]) == 1
        assert result[1][0].id == 2


# ---- Tests for V1 KVCacheManager.probe_prefix_match_length ----


class TestProbeOnV1KVCacheManager:
    """Test probe_prefix_match_length on v1 (C++) KVCacheManager using mocks."""

    def test_block_reuse_disabled_returns_zero(self):
        from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

        mgr = Mock(spec=KVCacheManager)
        mgr.enable_block_reuse = False

        result = KVCacheManager.probe_prefix_match_length(mgr, input_tokens=[1, 2, 3])
        assert result == 0

    def test_variable_window_returns_zero(self):
        from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

        mgr = Mock(spec=KVCacheManager)
        mgr.enable_block_reuse = True
        mgr.impl = Mock()
        mgr.impl.is_variable_window = True

        result = KVCacheManager.probe_prefix_match_length(mgr, input_tokens=[1, 2, 3])
        assert result == 0
        # analyze_prefix_reuse should NOT be called (would crash)
        mgr.impl.analyze_prefix_reuse.assert_not_called()

    def test_empty_tokens_returns_zero(self):
        from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

        mgr = Mock(spec=KVCacheManager)
        mgr.enable_block_reuse = True
        mgr.impl = Mock()
        mgr.impl.is_variable_window = False

        result = KVCacheManager.probe_prefix_match_length(mgr, input_tokens=[])
        assert result == 0

    def test_block_to_token_conversion(self):
        """Verify num_blocks * tokens_per_block conversion."""
        from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

        mgr = Mock(spec=KVCacheManager)
        mgr.enable_block_reuse = True
        mgr.impl = Mock()
        mgr.impl.is_variable_window = False
        mock_summary = Mock()
        mock_summary.reusable_blocks_all = 3
        mgr.impl.analyze_prefix_reuse = Mock(return_value=mock_summary)
        mgr.tokens_per_block = 64

        result = KVCacheManager.probe_prefix_match_length(mgr, input_tokens=list(range(200)))
        assert result == 192  # 3 blocks * 64 tokens/block
