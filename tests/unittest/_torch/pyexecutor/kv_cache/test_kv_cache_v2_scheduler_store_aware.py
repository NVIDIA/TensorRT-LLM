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
"""Shortest-uncached-first ranking that counts the prefix a KV connector can load,
and the lease touch that keeps a queued request's stored prefix from expiring.

Both are opt-in (TLLM_CTX_SHORTEST_FIRST_STORE_AWARE=1, TLLM_STORE_LEASE_TOUCH=1); the tests
below set them, except the one that checks the defaults."""

from unittest.mock import Mock

import pytest
from test_kv_cache_v2_scheduler import ids, make_ctx_request, make_kv_cache_manager, make_scheduler

from tensorrt_llm.llmapi.llm_args import ContextSchedulingPolicy

pytestmark = pytest.mark.cpu_only

SRPT = ContextSchedulingPolicy.SHORTEST_UNCACHED_FIRST
STEP = 16384
UNIT = 128
ENV = "TLLM_CTX_SHORTEST_FIRST_STORE_AWARE"
LEASE_ENV = "TLLM_STORE_LEASE_TOUCH"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "TLLM_CTX_SRPT_STORE_AWARE",
        "TLLM_DISABLE_STORE_LEASE_TOUCH",
        "TRTLLM_M31_DISABLE_LEASE_TOUCH",
        "TRTLLM_M31_LEASE_TOUCH_S",
        "TRTLLM_M31_LEASE_TOUCH_FIRST_S",
        "TRTLLM_M31_LEASE_TOUCH_MAX",
        "TLLM_STORE_LEASE_TOUCH_S",
        "TLLM_STORE_LEASE_TOUCH_FIRST_S",
        "TLLM_STORE_LEASE_TOUCH_MAX_PER_PASS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(LEASE_ENV, "1")


def _store_mgr(store_tokens, local=None):
    """A manager whose connector can load ``store_tokens[request_id]`` past the local match."""
    mgr = make_kv_cache_manager(enable_block_reuse=True)
    local = local or {}
    mgr.probe_context_reuse.side_effect = lambda req: local.get(req.py_request_id, 0)
    mgr.probe_connector_prefix.side_effect = lambda req, n: store_tokens.get(req.py_request_id, 0)
    return mgr


def test_both_are_off_by_default(monkeypatch):
    monkeypatch.delenv(ENV)
    monkeypatch.delenv(LEASE_ENV)
    mgr = _store_mgr({1: 399_000})
    sched = make_scheduler(mgr, max_num_tokens=STEP, context_scheduling_policy=SRPT)
    assert sched._ctx_store_aware is False and sched._lease_touch is None
    cold, warm = make_ctx_request(0, 100_000), make_ctx_request(1, 400_000)
    assert ids(sched._order_ctx_shortest_first([cold, warm])) == [0, 1]
    mgr.probe_connector_prefix.assert_not_called()


@pytest.mark.parametrize(
    "env, opted_in, expected",
    [
        ({}, True, True),  # a model opt-in turns both on
        ({ENV: "0", LEASE_ENV: "0"}, True, False),  # the knobs turn an opt-in off
        ({"TLLM_CTX_SRPT_STORE_AWARE": "1", LEASE_ENV: "1"}, False, True),  # older spelling
        ({"TLLM_CTX_SRPT_STORE_AWARE": "0", "TRTLLM_M31_DISABLE_LEASE_TOUCH": "1"}, True, False),
    ],
)
def test_model_opt_in_and_knob_overrides(monkeypatch, env, opted_in, expected):
    monkeypatch.delenv(ENV)
    monkeypatch.delenv(LEASE_ENV)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    mgr = _store_mgr({1: 399_000})
    sched = make_scheduler(
        mgr,
        max_num_tokens=STEP,
        context_scheduling_policy=SRPT,
        store_aware_ranking=opted_in,
        store_lease_touch=opted_in,
    )
    assert sched._ctx_store_aware is expected
    assert (sched._lease_touch is not None) is expected
    assert mgr.ctx_fill_v2 is expected


def test_a_stored_prefix_ranks_ahead_of_a_shorter_cold_prompt():
    mgr = _store_mgr({1: 399_000})
    sched = make_scheduler(mgr, max_num_tokens=STEP, context_scheduling_policy=SRPT)
    cold, warm = make_ctx_request(0, 100_000), make_ctx_request(1, 400_000)
    assert ids(sched._order_ctx_shortest_first([cold, warm])) == [1, 0]
    assert sched._uncached_tokens(warm) == 1000


def test_fcfs_never_probes_the_connector():
    mgr = _store_mgr({1: 399_000})
    sched = make_scheduler(mgr, max_num_tokens=STEP, ctx_chunk_config=(None, UNIT))
    out = sched.schedule_request(
        [make_ctx_request(0, 100_000), make_ctx_request(1, 400_000)], set()
    )
    assert ids(out.context_requests)[0] == 0
    mgr.probe_connector_prefix.assert_not_called()


def test_each_request_is_probed_once_across_passes():
    mgr = _store_mgr({1: 399_000})
    sched = make_scheduler(mgr, max_num_tokens=STEP, context_scheduling_policy=SRPT)
    reqs = [make_ctx_request(0, 100_000), make_ctx_request(1, 400_000)]
    for _ in range(3):
        sched._reuse_probes.clear()
        sched._order_ctx_shortest_first(reqs)
    assert sorted(c.args[0].py_request_id for c in mgr.probe_connector_prefix.call_args_list) == [
        0,
        1,
    ]
    # The local reuse probe still runs once per request per pass.
    assert mgr.probe_context_reuse.call_count == 6


@pytest.mark.parametrize(
    "store, local, expected",
    [
        # The probe starts at the local match.
        pytest.param({1: 1000}, {1: 2048}, 10_000 - 3048, id="past_local_match"),
        # A stored prefix never covers the last prompt token.
        pytest.param({1: 10_000}, {}, 1, id="whole_prompt_stored"),
        # No answer falls back to the local cache.
        pytest.param(None, {1: 2048}, 10_000 - 2048, id="no_answer"),
    ],
)
def test_uncached_tokens_count_the_stored_prefix(store, local, expected):
    mgr = _store_mgr(store or {}, local=local)
    if store is None:
        mgr.probe_connector_prefix.side_effect = lambda req, n: None
    sched = make_scheduler(mgr, context_scheduling_policy=SRPT)
    req = make_ctx_request(1, 10_000)
    assert sched._uncached_tokens(req) == expected
    assert [c.args[1] for c in mgr.probe_connector_prefix.call_args_list] == [local.get(1, 0)]


def test_probes_of_requests_that_left_the_queue_are_dropped():
    mgr = _store_mgr({0: 512, 1: 512})
    sched = make_scheduler(mgr, max_num_tokens=STEP, context_scheduling_policy=SRPT)
    first, second = make_ctx_request(0, 4096), make_ctx_request(1, 4096)
    sched._order_ctx_shortest_first([first, second])
    assert set(sched._store_probes) == {0, 1}
    sched.schedule_request([second], set())
    assert 0 not in sched._store_probes


# ---- store lease touch ----


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _lease_sched(monkeypatch, store_tokens=None, *, adp=True, policy=SRPT):
    mgr = _store_mgr(store_tokens or {})
    mgr.kv_connector_manager = Mock(capacity_only=False, enable_attention_dp=adp)
    sched = make_scheduler(mgr, max_num_tokens=STEP, context_scheduling_policy=policy)
    clock = _Clock()
    sched._lease_touch.clock = clock
    return sched, mgr, clock


def _waiting(*request_ids):
    reqs = [make_ctx_request(rid, 4096) for rid in request_ids]
    for req in reqs:
        req.py_connector_allocation_reported = False
    return reqs


def _touched(mgr):
    return [c.args[0].py_request_id for c in mgr.probe_connector_prefix.call_args_list]


def test_a_waiting_request_is_first_touched_after_first_s(monkeypatch):
    sched, mgr, clock = _lease_sched(monkeypatch, policy=ContextSchedulingPolicy.FCFS)
    reqs = _waiting(1)
    sched._lease_touch_pass(reqs, [])
    assert _touched(mgr) == []
    clock.t += sched._lease_touch.first_s
    sched._lease_touch_pass(reqs, [])
    assert _touched(mgr) == [1]
    clock.t += sched._lease_touch.interval_s - 0.5
    sched._lease_touch_pass(reqs, [])
    assert _touched(mgr) == [1]
    clock.t += 0.5
    sched._lease_touch_pass(reqs, [])
    assert _touched(mgr) == [1, 1]


def test_an_admitted_request_is_not_touched(monkeypatch):
    sched, mgr, clock = _lease_sched(monkeypatch, policy=ContextSchedulingPolicy.FCFS)
    reqs = _waiting(1, 2)
    clock.t += sched._lease_touch.interval_s
    sched._lease_touch.last.update({1: 0.0, 2: 0.0})
    sched._lease_touch_pass(reqs, reqs[:1])
    assert _touched(mgr) == [2]


def test_a_ranking_probe_counts_as_a_touch(monkeypatch):
    sched, mgr, clock = _lease_sched(monkeypatch, {1: 2048})
    reqs = _waiting(1, 2)
    sched._order_ctx_shortest_first(reqs)
    assert sorted(sched._lease_touch.last) == [1, 2]
    clock.t += sched._lease_touch.interval_s - 0.5
    sched._lease_touch_pass(reqs, [])
    assert sorted(_touched(mgr)) == [1, 2]  # only the ranking probes


def test_without_attention_dp_nothing_is_touched(monkeypatch):
    sched, mgr, clock = _lease_sched(monkeypatch, adp=False, policy=ContextSchedulingPolicy.FCFS)
    reqs = _waiting(1)
    for _ in range(3):
        clock.t += sched._lease_touch.interval_s
        sched._lease_touch_pass(reqs, [])
    assert _touched(mgr) == []


def test_an_error_turns_the_touch_off(monkeypatch):
    sched, mgr, clock = _lease_sched(monkeypatch, policy=ContextSchedulingPolicy.FCFS)
    mgr.probe_connector_prefix.side_effect = RuntimeError("store down")
    sched._lease_touch.last[1] = 0.0
    sched._lease_touch_pass(_waiting(1), [])
    assert sched._lease_touch is None


@pytest.mark.parametrize("renewed, expected", [(1024, 1024), (0, None)])
def test_a_shrunk_renewal_replaces_the_ranking_probe(monkeypatch, renewed, expected):
    store = {1: 2048}
    sched, mgr, clock = _lease_sched(monkeypatch, store)
    reqs = _waiting(1)
    sched._order_ctx_shortest_first(reqs)
    assert sched._store_probes[1] == 2048
    sched._ctx_fill_defers[1] = 2
    store[1] = renewed
    clock.t += sched._lease_touch.interval_s
    sched._lease_touch_pass(reqs, [])
    assert sched._store_probes[1] == expected
    assert 1 not in sched._ctx_fill_defers
