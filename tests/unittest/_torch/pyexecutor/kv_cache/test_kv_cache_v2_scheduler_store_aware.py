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
and the lease touch that keeps a queued request's stored prefix from expiring."""

from unittest.mock import Mock

import pytest
from test_kv_cache_v2_scheduler import ids, make_ctx_request, make_kv_cache_manager, make_scheduler

from tensorrt_llm.llmapi.llm_args import ContextSchedulingPolicy

pytestmark = pytest.mark.cpu_only

SRPT = ContextSchedulingPolicy.SHORTEST_UNCACHED_FIRST
STEP = 16384
UNIT = 128
ENV = "TLLM_CTX_SHORTEST_FIRST_STORE_AWARE"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        ENV,
        "TLLM_DISABLE_STORE_LEASE_TOUCH",
        "TLLM_STORE_LEASE_TOUCH_S",
        "TLLM_STORE_LEASE_TOUCH_FIRST_S",
        "TLLM_STORE_LEASE_TOUCH_MAX_PER_PASS",
    ):
        monkeypatch.delenv(name, raising=False)


def _store_mgr(store_tokens, local=None):
    """A manager whose connector can load ``store_tokens[request_id]`` past the local match."""
    mgr = make_kv_cache_manager(enable_block_reuse=True)
    local = local or {}
    mgr.probe_context_reuse.side_effect = lambda req: local.get(req.py_request_id, 0)
    mgr.probe_connector_prefix.side_effect = lambda req, n: store_tokens.get(req.py_request_id, 0)
    return mgr


def test_a_stored_prefix_ranks_ahead_of_a_shorter_cold_prompt():
    mgr = _store_mgr({1: 399_000})
    sched = make_scheduler(mgr, max_num_tokens=STEP, context_scheduling_policy=SRPT)
    cold, warm = make_ctx_request(0, 100_000), make_ctx_request(1, 400_000)
    assert ids(sched._order_ctx_shortest_first([cold, warm])) == [1, 0]
    assert sched._uncached_tokens(warm) == 1000


def test_the_kill_switch_ranks_on_the_local_cache_only(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    mgr = _store_mgr({1: 399_000})
    sched = make_scheduler(mgr, max_num_tokens=STEP, context_scheduling_policy=SRPT)
    cold, warm = make_ctx_request(0, 100_000), make_ctx_request(1, 400_000)
    assert ids(sched._order_ctx_shortest_first([cold, warm])) == [0, 1]
    mgr.probe_connector_prefix.assert_not_called()


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


def test_the_probe_starts_at_the_local_match():
    mgr = _store_mgr({1: 1000}, local={1: 2048})
    sched = make_scheduler(mgr, context_scheduling_policy=SRPT)
    req = make_ctx_request(1, 10_000)
    assert sched._uncached_tokens(req) == 10_000 - 3048
    assert [(c.args[0], c.args[1]) for c in mgr.probe_connector_prefix.call_args_list] == [
        (req, 2048)
    ]


def test_a_stored_prefix_never_covers_the_last_prompt_token():
    mgr = _store_mgr({1: 10_000})
    sched = make_scheduler(mgr, context_scheduling_policy=SRPT)
    assert sched._uncached_tokens(make_ctx_request(1, 10_000)) == 1


def test_no_answer_falls_back_to_the_local_cache():
    mgr = _store_mgr({}, local={1: 2048})
    mgr.probe_connector_prefix.side_effect = lambda req, n: None
    sched = make_scheduler(mgr, context_scheduling_policy=SRPT)
    assert sched._uncached_tokens(make_ctx_request(1, 10_000)) == 10_000 - 2048


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


def test_the_oldest_due_requests_are_touched_first_up_to_the_cap(monkeypatch):
    monkeypatch.setenv("TLLM_STORE_LEASE_TOUCH_MAX_PER_PASS", "2")
    sched, mgr, clock = _lease_sched(monkeypatch, policy=ContextSchedulingPolicy.FCFS)
    lease = sched._lease_touch
    reqs = _waiting(1, 2, 3)
    for offset, rid in zip((3.0, 1.0, 2.0), (1, 2, 3)):
        lease.last[rid] = clock.t - lease.interval_s - offset
    sched._lease_touch_pass(reqs, [])
    assert _touched(mgr) == [1, 3]


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


def test_a_prefix_lost_while_queued_is_counted(monkeypatch):
    store = {1: 2048}
    sched, mgr, clock = _lease_sched(monkeypatch, store)
    reqs = _waiting(1)
    sched._order_ctx_shortest_first(reqs)
    store[1] = 1024
    clock.t += sched._lease_touch.interval_s
    sched._lease_touch_pass(reqs, [])
    assert (sched._lease_touch.shrunk, sched._lease_touch.lost_tokens) == (1, 1024)


def test_without_attention_dp_nothing_is_touched(monkeypatch):
    sched, mgr, clock = _lease_sched(monkeypatch, adp=False, policy=ContextSchedulingPolicy.FCFS)
    reqs = _waiting(1)
    for _ in range(3):
        clock.t += sched._lease_touch.interval_s
        sched._lease_touch_pass(reqs, [])
    assert _touched(mgr) == []


def test_requests_that_left_are_forgotten(monkeypatch):
    sched, mgr, clock = _lease_sched(monkeypatch, policy=ContextSchedulingPolicy.FCFS)
    sched._lease_touch_pass(_waiting(1, 2), [])
    sched._lease_touch_pass(_waiting(2), [])
    assert list(sched._lease_touch.last) == [2]


def test_an_error_turns_the_touch_off(monkeypatch):
    sched, mgr, clock = _lease_sched(monkeypatch, policy=ContextSchedulingPolicy.FCFS)
    mgr.probe_connector_prefix.side_effect = RuntimeError("store down")
    sched._lease_touch.last[1] = 0.0
    sched._lease_touch_pass(_waiting(1), [])
    assert sched._lease_touch is None


def test_the_kill_switch_disables_the_touch(monkeypatch):
    monkeypatch.setenv("TLLM_DISABLE_STORE_LEASE_TOUCH", "1")
    sched = make_scheduler(_store_mgr({}), context_scheduling_policy=SRPT)
    assert sched._lease_touch is None
