# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-DP groups (``TRTLLM_ADPTP_ATTN_TP``) in the executor and the ADP routers. CPU only: the real
``PyExecutor`` methods are called on a minimal stand-in ``self`` (mock dist), and the real routers on mock
dist / KV managers.

Covers: the lockstep signature on the ADP batch-size all-gather (and its stock shape with the switch off), group
admission capacity, leader-only responses, and the group-safe KV-cache-aware / conversation-aware routers.
"""

import os
import threading
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from tensorrt_llm._torch import adp_groups
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
    ConversationAwareADPRouter,
    DefaultADPRouter,
    KVCacheAwareADPRouter,
    RankState,
)
from tensorrt_llm.conversation_params import ConversationParams

TP, GT = 8, 4  # DP2TP4: 8 ranks, 2 groups of 4


@pytest.fixture
def clean_env():
    with mock.patch.dict(os.environ, {}, clear=False):
        for name in (adp_groups.ENV, adp_groups.LEGACY_ENV, adp_groups.CHECK_ENV, adp_groups.LEGACY_CHECK_ENV):
            os.environ.pop(name, None)
        yield


@pytest.fixture
def grouped(clean_env):
    os.environ[adp_groups.ENV] = str(GT)
    yield


def _mapping(tp_rank=0):
    return SimpleNamespace(enable_attention_dp=True, tp_size=TP, tp_rank=tp_rank, pp_size=1, cp_size=1,
                           enable_lm_head_tp_in_adp=False, dp_size=TP, tp_group=list(range(TP)))


# ---------------------------------------------------------------------------------------------------------------
# PyExecutor methods on a stand-in self
# ---------------------------------------------------------------------------------------------------------------


def _req(rid, ntok, last):
    return SimpleNamespace(py_request_id=rid, get_num_tokens=lambda beam: ntok, get_last_tokens=lambda beam: last)


def _batch(reqs):
    return SimpleNamespace(batch_size=len(reqs), all_requests=lambda: list(reqs))


class _GatherDist:
    """tp_allgather_int64 that returns this rank's row for every rank, optionally with one rank's row replaced."""

    def __init__(self, tp_rank=0, override=None):
        self.tp_rank = tp_rank
        self.tp_size = TP
        self.rank = tp_rank
        self.world_size = TP
        self.mapping = _mapping(tp_rank)
        self.sent = []
        self.override = override or {}

    def tp_allgather_int64(self, values):
        values = [int(v) for v in values]
        self.sent.append(values)
        rows = [list(values) for _ in range(TP)]
        for r, row in self.override.items():
            rows[r] = row(values)
        return np.asarray(rows, dtype=np.int64)


def _executor_self(dist, group_tp, check):
    return SimpleNamespace(enable_attention_dp=True, dist=dist, _adp_group_tp=group_tp,
                           _adp_dp_rank=dist.tp_rank // group_tp, _adp_group_check=check)


class TestCanQueueLockstep:
    REQS = [_req(11, 100, 7), _req(12, 40, 3)]

    def test_stock_shape_when_off(self, clean_env):
        dist = _GatherDist()
        out = PyExecutor._can_queue(_executor_self(dist, 1, False), _batch(self.REQS))
        assert out == (True, True)
        assert dist.sent == [[2]]  # the stock one-value gather

    def test_grouped_without_check_is_stock_gather(self, grouped):
        dist = _GatherDist()
        PyExecutor._can_queue(_executor_self(dist, GT, False), _batch(self.REQS))
        assert dist.sent == [[2]]

    def test_signature_matches(self, grouped):
        dist = _GatherDist()
        assert PyExecutor._can_queue(_executor_self(dist, GT, True), _batch(self.REQS)) == (True, True)
        sig = adp_groups.batch_signature([(11, 100, 7), (12, 40, 3)])
        assert dist.sent == [[2, sig]]

    def test_divergent_rank_raises(self, grouped):
        # rank 6 (group 1) sampled a different last token
        other = adp_groups.batch_signature([(11, 100, 7), (12, 40, 4)])
        dist = _GatherDist(override={6: lambda v: [v[0], other]})
        with pytest.raises(RuntimeError, match=r"attention-DP groups \[1\]"):
            PyExecutor._can_queue(_executor_self(dist, GT, True), _batch(self.REQS))

    def test_groups_may_differ(self, grouped):
        """Different groups schedule different batches; only intra-group divergence is an error."""
        other = adp_groups.batch_signature([(99, 5, 1)])
        dist = _GatherDist(override={r: (lambda v: [1, other]) for r in range(4, 8)})
        assert PyExecutor._can_queue(_executor_self(dist, GT, True), _batch(self.REQS)) == (True, True)

    def test_empty_group_blocks_queue_like_stock(self, grouped):
        dist = _GatherDist(override={r: (lambda v: [0, adp_groups.batch_signature([])]) for r in range(4, 8)})
        assert PyExecutor._can_queue(_executor_self(dist, GT, True), _batch(self.REQS)) == (False, True)


class TestAdmissionCapacity:
    def test_capacity(self, clean_env):
        for group_tp, expect in ((1, TP * 16), (4, 2 * 16), (8, 16)):
            fake = SimpleNamespace(enable_attention_dp=True, dist=SimpleNamespace(tp_size=TP),
                                   _adp_group_tp=group_tp, max_num_active_requests=16)
            assert PyExecutor._get_request_admission_capacity(fake) == expect


class _RespDist(_GatherDist):
    def tp_gather(self, obj):
        self.gathered = obj
        return [obj] if self.rank == 0 else None

    def allgather(self, obj):
        self.gathered = obj
        return [obj]


def _resp_self(dist, group_tp):
    fake = _executor_self(dist, group_tp, False)
    fake.gather_all_responses = False
    fake.response_cv = threading.Condition()
    fake.responses = {}
    fake.result_wait_queues = {}
    fake._maybe_record_hang_diagnostic_phase = lambda *a, **k: None
    return fake


def test_class_defaults_are_stock():
    """Executors built without __init__ (test doubles, subclasses) see no groups."""
    ex = object.__new__(PyExecutor)
    assert (ex._adp_group_tp, ex._adp_dp_rank, ex._adp_group_check) == (1, 0, False)


class TestLeaderOnlyResponses:
    RESP = [(1, "r1"), (2, "r2")]

    def test_grouped_non_leaders_send_nothing(self, grouped):
        for rank in range(TP):
            dist = _RespDist(rank)
            PyExecutor._enqueue_responses(_resp_self(dist, GT), list(self.RESP))
            assert dist.gathered == (list(self.RESP) if rank % GT == 0 else []), rank

    def test_stock_every_rank_sends(self, clean_env):
        for rank in range(TP):
            dist = _RespDist(rank)
            PyExecutor._enqueue_responses(_resp_self(dist, 1), list(self.RESP))
            assert dist.gathered == list(self.RESP)

    def test_rank0_enqueues(self, grouped):
        dist = _RespDist(0)
        fake = _resp_self(dist, GT)
        PyExecutor._enqueue_responses(fake, list(self.RESP))
        assert fake.responses == {1: ["r1"], 2: ["r2"]}


# ---------------------------------------------------------------------------------------------------------------
# Group-safe routers (ported from M3.1 m31/adp2tp4-serve, minus waitq-srpt)
# ---------------------------------------------------------------------------------------------------------------


def _dist(tp_rank=0, gathered=None):
    dist = MagicMock()
    dist.tp_rank = tp_rank
    dist.tp_size = TP
    dist.mapping = _mapping(tp_rank)
    dist.has_cp_helix = False
    if gathered is not None:
        dist.tp_allgather = Mock(return_value=gathered)
    return dist


def _item(req_id, n=100, conv=None):
    item = MagicMock()
    item.id = req_id
    item.child_req_ids = None
    sp = MagicMock()
    sp.attention_dp_rank = None
    sp.attention_dp_relax = True
    sp.subagent_affinity_id = None
    item.request = MagicMock()
    item.request.py_scheduling_params = sp
    item.request.input_token_ids = list(range(n))
    item.request.num_input_tokens = n
    item.request.cache_salt = None
    item.request.lora_config = None
    item.request.py_conversation_params = ConversationParams(conversation_id=conv) if conv else None
    return item


def _group_states(per_group_active):
    per_rank = [RankState(rank=r, num_active_requests=per_group_active[r // GT], num_active_tokens=0)
                for r in range(TP)]
    return adp_groups.route_states(per_rank, GT)


def test_group_safe_routers():
    for cls in (DefaultADPRouter, ConversationAwareADPRouter, KVCacheAwareADPRouter):
        assert adp_groups.router_mismatch(cls.__name__) is None
    assert adp_groups.router_mismatch("SomeOtherRouter") is not None


def test_kv_router_collapses_prefix_matches_to_groups(grouped):
    gathered = [[7, 0]] * 4 + [[7, 90]] * 4  # group 1 (ranks 4..7) holds 90 tokens of request 7
    for tp_rank in range(TP):
        router = KVCacheAwareADPRouter(dist=_dist(tp_rank, gathered), kv_cache_manager=MagicMock())
        assert router._group_tp == GT
        router.gather_prefix_matches([_item(7)])
        assert router._all_ranks_prefix_matches == [{7: 0}, {7: 90}]
        out, _ = router.route_requests(_group_states([0, 0]), [_item(7)], max_num_active_requests=48)
        assert set(out) == {0, 1}
        assert [len(out[0]), len(out[1])] == [0, 1]


def test_kv_router_match_len_reads_group_tables(grouped):
    gathered = [[7, 0]] * 4 + [[7, 90]] * 4
    router = KVCacheAwareADPRouter(dist=_dist(0, gathered), kv_cache_manager=MagicMock())
    router.gather_prefix_matches([_item(7)])
    assert router._match_len(1, 7) == 90 and router._match_len(0, 7) == 0


def test_kv_router_warmup_spans_groups(grouped):
    router = KVCacheAwareADPRouter(dist=_dist(0), kv_cache_manager=MagicMock(), cold_start_warmup=True)
    assert router._pending_warmup_ranks == {0, 1}
    router._all_ranks_prefix_matches = [{}, {}]
    out, _ = router.route_requests(_group_states([0, 0]), [_item(1), _item(2)], max_num_active_requests=48)
    assert [len(out[0]), len(out[1])] == [1, 1]
    assert router._pending_warmup_ranks == set()


def test_kv_router_load_cap_overrides_affinity(grouped):
    router = KVCacheAwareADPRouter(dist=_dist(0), kv_cache_manager=MagicMock())
    router._all_ranks_prefix_matches = [{3: 0}, {3: 90}]
    out, _ = router.route_requests(_group_states([0, 48]), [_item(3)], max_num_active_requests=48)
    assert [len(out[0]), len(out[1])] == [1, 0]


def test_kv_router_stock_unchanged_without_groups(clean_env):
    gathered = [[7, r * 10] for r in range(TP)]
    router = KVCacheAwareADPRouter(dist=_dist(0, gathered), kv_cache_manager=MagicMock(), cold_start_warmup=True)
    assert router._group_tp == 1
    assert router._pending_warmup_ranks == set(range(TP))
    router.gather_prefix_matches([_item(7)])
    assert len(router._all_ranks_prefix_matches) == TP
    assert router._match_len(5, 7) == 50


def test_conversation_router_sticks_to_group(grouped):
    routers = [ConversationAwareADPRouter(dist=_dist(r), new_conv_placement="least_tokens") for r in range(TP)]
    first = [rt.route_requests(_group_states([0, 0]), [_item(1, conv="a"), _item(2, conv="b")], 48)[0]
             for rt in routers]
    assert all(set(o) == {0, 1} for o in first)
    where = {it.request.py_conversation_params.conversation_id: g for g, its in first[0].items() for it in its}
    assert sorted(where.values()) == [0, 1]
    for o in first:  # every rank made the same choice
        assert {it.request.py_conversation_params.conversation_id: g for g, its in o.items() for it in its} == where
    busy = [0, 0]
    busy[where["a"]] = 20
    nxt = routers[0].route_requests(_group_states(busy), [_item(3, conv="a")], 48)[0]
    assert len(nxt[where["a"]]) == 1
