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
"""KVCacheV2Scheduler context fill: charge a store-served first chunk only what it computes.

- v1 sizes the chunk (``TRTLLM_DISABLE_CTX_FILL=1`` charges the full chunk);
- v2 reserves the probed served prefix's pages for a shrunk chunk, defers it a pass when they
  do not fit (``TRTLLM_CTX_FILL_MAX_DEFER`` passes), and needs a backlog
  (``TRTLLM_CTX_FILL_MIN_BACKLOG``); ``TRTLLM_DISABLE_CTX_FILL_V2=1`` restores v1;
- the connector side honours a reserved prefix behind a one-block chunk.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _torch.pyexecutor.kv_cache.test_kv_cache_v2_scheduler import (
    ids,
    make_ctx_request,
    make_kv_cache_manager,
    make_scheduler,
)
from _torch.pyexecutor.kv_cache.test_kv_connector_v2_prefix import (
    FakeConnectorManager,
    FakeKvCache,
    FakeRequest,
    make_manager,
)

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import ScheduleAction
from tensorrt_llm.llmapi.llm_args import ContextSchedulingPolicy

pytestmark = pytest.mark.cpu_only

SRPT = ContextSchedulingPolicy.SHORTEST_UNCACHED_FIRST
STEP = 16384
UNIT = 128
CAP = 8192
LONG = 300_000

ENV_NAMES = (
    "TLLM_CTX_SHORTEST_FIRST_STORE_AWARE",
    "TLLM_CTX_CHUNK_CAP_ELIGIBLE_SHARE",
    "TLLM_CTX_REQUEST_CHUNK_CAP",
    "TLLM_CTX_SRPT_MAX_WAIT_PASSES",
    "TLLM_CTX_SRPT_AGED_SLOTS",
    "TRTLLM_DISABLE_CTX_FILL",
    "TRTLLM_DISABLE_CTX_FILL_V2",
    "TRTLLM_CTX_FILL_MIN_BACKLOG",
    "TRTLLM_CTX_FILL_MAX_DEFER",
    "TLLM_STORE_LEASE_TOUCH",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TLLM_CTX_REQUEST_CHUNK_CAP", str(CAP))
    # Context fill rides store-aware ranking, which is opt-in. The lease touch (also opt-in)
    # stays off: it only re-probes, and would add to the probe counts read here.
    monkeypatch.setenv("TLLM_CTX_SHORTEST_FIRST_STORE_AWARE", "1")


@pytest.fixture
def no_backlog_gate(monkeypatch):
    monkeypatch.setenv("TRTLLM_CTX_FILL_MIN_BACKLOG", "0")


def _mgr(store_tokens, loads_async=True, tokens_per_block=128):
    mgr = make_kv_cache_manager(enable_block_reuse=True, tokens_per_block=tokens_per_block)
    mgr.probe_context_reuse.side_effect = lambda req: 0
    mgr.probe_connector_prefix.side_effect = lambda req, n: store_tokens.get(req.py_request_id, 0)
    mgr.connector_loads_prefix_async.return_value = loads_async
    return mgr


def _sched(mgr):
    return make_scheduler(
        mgr, max_num_tokens=STEP, ctx_chunk_config=(None, UNIT), context_scheduling_policy=SRPT
    )


def _backlogged(sched, backlog=8):
    sched._ctx_fill_backlog = backlog
    return sched


def _req(request_id, length, **kwargs):
    req = make_ctx_request(request_id, length, **kwargs)
    req.py_connector_allocation_reported = False
    req.py_kv_gpu_hold = 0
    return req


def _forward_tokens(reqs, served_async):
    """Tokens the forward pass computes once async-served requests leave the batch."""
    return sum(r.context_chunk_size for r in reqs if r.py_request_id not in served_async)


# ---- v1 sizing ----


def test_async_store_hit_no_longer_strands_its_cap_share(no_backlog_gate):
    """Warm prompt (async store hit) + two cold long prompts in a 16K step."""
    sched = _sched(_mgr({0: 399_000}))
    warm, a, b = _req(0, 400_000), _req(1, LONG), _req(2, LONG)
    out = sched.schedule_request([a, b, warm], set())
    assert ids(out.context_requests) == [0, 1, 2]
    assert warm.context_chunk_size == UNIT
    assert a.context_chunk_size == CAP
    assert b.context_chunk_size == STEP - UNIT - CAP
    assert _forward_tokens([warm, a, b], {0}) == STEP - UNIT


def test_kill_switch_charges_the_full_chunk(monkeypatch, no_backlog_gate):
    monkeypatch.setenv("TRTLLM_DISABLE_CTX_FILL", "1")
    mgr = _mgr({0: 399_000})
    sched = _sched(mgr)
    assert sched._ctx_fill is False and sched._ctx_fill_v2 is False
    warm, a, b = _req(0, 400_000), _req(1, LONG), _req(2, LONG)
    out = sched.schedule_request([a, b, warm], set())
    # The warm request takes a full cap share and leaves the batch: 8K of 16K computed.
    assert ids(out.context_requests) == [0, 1]
    assert warm.context_chunk_size == CAP
    assert _forward_tokens([warm, a], {0}) == CAP
    mgr.connector_loads_prefix_async.assert_not_called()


def test_several_async_hits_and_one_cold_prompt_fill_the_step(no_backlog_gate):
    sched = _sched(_mgr({0: 99_000, 1: 199_000, 2: 49_000}))
    reqs = [_req(0, 100_000), _req(1, 200_000), _req(2, 50_000), _req(3, LONG)]
    out = sched.schedule_request(reqs, set())
    assert ids(out.context_requests) == [0, 1, 2, 3]
    assert [r.context_chunk_size for r in reqs[:3]] == [UNIT] * 3
    assert reqs[3].context_chunk_size == STEP - 3 * UNIT
    assert _forward_tokens(reqs, {0, 1, 2}) == STEP - 3 * UNIT


@pytest.mark.parametrize(
    "wait_lens",
    [
        pytest.param((1_000, 2_000), id="waits_visited_first"),
        pytest.param((2 * LONG, 3 * LONG), id="runnable_visited_first"),
    ],
)
def test_prefix_waits_take_no_cap_share(no_backlog_gate, wait_lens):
    """Two requests waiting for a prefix with one runnable long prompt: the waits run nothing
    this pass, so the runnable prompt's share is the whole step, in either visiting order."""
    sched = _sched(_mgr({}))
    sched._prefix_wait_planner = Mock()
    sched._prefix_wait_planner.plan.side_effect = lambda reqs: {0: None, 1: None}
    waiting = [_req(0, wait_lens[0]), _req(1, wait_lens[1])]
    runnable = _req(2, LONG)
    out = sched.schedule_request([runnable, *waiting], set())
    assert ids(out.context_requests) == [2]
    assert runnable.context_chunk_size == STEP


@pytest.mark.parametrize(
    "served, expected",
    [
        # 3000 uncached tokens round up to the chunk unit.
        (397_000, 3072),
        # A tail at or past the cap keeps the cap.
        (390_000, CAP),
    ],
)
def test_sync_store_hit_is_charged_its_uncached_tail(no_backlog_gate, served, expected):
    sched = _sched(_mgr({0: served}, loads_async=False))
    warm, a = _req(0, 400_000), _req(1, LONG)
    sched.schedule_request([a, warm], set())
    assert warm.context_chunk_size == expected
    if expected != CAP:
        assert a.context_chunk_size == STEP - expected


@pytest.mark.parametrize(
    "store, loads_async, warm_len, expected",
    [
        # A store prefix shorter than the uncached tail would leave the shift unserved.
        pytest.param({0: 1_000}, False, 5_000, 5_000, id="served_end_inside_chunk"),
        pytest.param({0: 399_000}, None, 400_000, CAP, id="unknown_loading_mode"),
        pytest.param({0: 399_000}, Mock(), 400_000, CAP, id="non_bool_loading_mode"),
        pytest.param({}, True, LONG, CAP, id="store_miss"),
    ],
)
def test_the_chunk_is_kept_unless_the_shrink_is_served(
    no_backlog_gate, store, loads_async, warm_len, expected
):
    mgr = _mgr(store, loads_async=loads_async)
    sched = _sched(mgr)
    warm, a = _req(0, warm_len), _req(1, LONG)
    sched.schedule_request([a, warm], set())
    assert warm.context_chunk_size == expected
    if not store:
        assert a.context_chunk_size == STEP - CAP
        mgr.connector_loads_prefix_async.assert_not_called()


def test_continuation_chunk_is_not_shrunk(no_backlog_gate):
    sched = _sched(_mgr({0: 9_000}))
    sched._store_probes[0] = 9_000
    req = _req(0, 6_000, prompt_len=10_000, is_first_context_chunk=False)
    req.context_current_position = 4_000
    sched.schedule_request([req, _req(1, LONG)], set())
    assert req.context_chunk_size == 6_000


def _connector(adp=True, capacity_only=False, scheduler=None, **extra):
    return SimpleNamespace(
        enable_attention_dp=adp, capacity_only=capacity_only, scheduler=scheduler, **extra
    )


@pytest.mark.parametrize(
    "connector, is_draft, expected",
    [
        (_connector(scheduler=SimpleNamespace(loads_async=True)), False, True),
        (_connector(scheduler=SimpleNamespace(loads_async=False)), False, False),
        (_connector(capacity_only=True, scheduler=SimpleNamespace(loads_async=True)), False, None),
        (_connector(adp=False, scheduler=SimpleNamespace(loads_async=True)), False, None),
        # The connector's own answer wins.
        (_connector(adp=False, loads_prefix_async=lambda: True), False, True),
        (None, False, None),
        (_connector(scheduler=SimpleNamespace(loads_async=True)), True, None),
    ],
)
def test_manager_reads_the_connectors_loading_mode(connector, is_draft, expected):
    manager = SimpleNamespace(kv_connector_manager=connector, is_draft=is_draft)
    assert KVCacheManagerV2.connector_loads_prefix_async(manager) is expected


# ---- v2: reservation, deferral and the backlog gate ----


def test_shrunk_async_chunk_reserves_its_probed_prefix():
    mgr = _mgr({0: 399_000})
    mgr.reserve_served_prefix_capacity.return_value = True
    sched = _backlogged(_sched(mgr))
    assert sched._ctx_fill_v2 is True
    warm, a, b = _req(0, 400_000), _req(1, LONG), _req(2, LONG)
    out = sched.schedule_request([a, b, warm], set())
    assert ids(out.context_requests) == [0, 1, 2]
    assert warm.context_chunk_size == UNIT
    mgr.reserve_served_prefix_capacity.assert_called_once_with(warm, 399_000 + UNIT)
    assert sched._ctx_fill_counts[4] == 1


def test_prefix_that_does_not_fit_defers_instead_of_losing_the_hit():
    """The request leaves the pass and its first-chunk allocation is freed."""
    mgr = _mgr({0: 399_000})
    mgr.reserve_served_prefix_capacity.return_value = False
    sched = _backlogged(_sched(mgr))
    warm, a, b = _req(0, 400_000), _req(1, LONG), _req(2, LONG)
    mgr.kv_cache_map[0] = object()
    out = sched.schedule_request([a, b, warm], set())
    assert ids(out.context_requests) == [1, 2]
    assert [a.context_chunk_size, b.context_chunk_size] == [CAP, STEP - CAP]
    assert sched._ctx_fill_defers == {0: 1}
    assert sched._ctx_fill_counts[5] == 1
    mgr.free_resources.assert_any_call(warm)


def test_deferral_limit_stops_shrinking(monkeypatch):
    monkeypatch.setenv("TRTLLM_CTX_FILL_MAX_DEFER", "1")
    mgr = _mgr({0: 399_000})
    mgr.reserve_served_prefix_capacity.return_value = False
    sched = _backlogged(_sched(mgr))
    warm, a = _req(0, 400_000), _req(1, LONG)
    out = sched.schedule_request([a, warm], set())
    assert ids(out.context_requests) == [1]
    _backlogged(sched)
    out = sched.schedule_request([a, warm], set())
    # Not shrunk any more: the full chunk, whose in-page fallback keeps part of the hit.
    assert ids(out.context_requests) == [0, 1]
    assert warm.context_chunk_size == CAP
    assert mgr.reserve_served_prefix_capacity.call_count == 1
    assert sched._ctx_fill_counts[6] == 1


def test_no_backlog_no_shrink():
    mgr = _mgr({0: 399_000})
    sched = _sched(mgr)
    warm, a = _req(0, 400_000), _req(1, LONG)
    out = sched.schedule_request([a, warm], set())
    assert ids(out.context_requests) == [0, 1]
    assert warm.context_chunk_size == CAP
    assert sched._ctx_fill_counts[3] == 1
    mgr.reserve_served_prefix_capacity.assert_not_called()
    # Both pending requests were scheduled: no backlog for the next pass either.
    assert sched._ctx_fill_backlog == 0


def test_backlog_from_the_previous_pass_turns_the_shrink_on():
    mgr = _mgr({0: 399_000})
    mgr.reserve_served_prefix_capacity.return_value = True
    sched = _sched(mgr)
    warm = _req(0, 400_000)
    cold = [_req(i, LONG) for i in range(1, 11)]
    # Pass 1: no backlog seen yet -> the warm prompt keeps its cap share; two of eleven run.
    out = sched.schedule_request(cold + [warm], set())
    assert warm.context_chunk_size == CAP
    assert len(out.context_requests) == 2
    assert sched._ctx_fill_backlog == 9
    # Pass 2: the backlog is >= 8, so the warm prompt is shrunk and three requests run.
    out = sched.schedule_request(cold + [warm], set())
    assert warm.context_chunk_size == UNIT
    assert len(out.context_requests) == 3


# ---- v2, connector side: a large served prefix behind a one-block chunk ----

TPB = 32  # test_kv_connector_v2_prefix.TOKENS_PER_BLOCK
PROMPT = 8192
PROBED = 93 * TPB  # a whole-block store prefix
CHUNK = TPB  # the async shrink: one block


def _serve(manager, req, kv_cache, chunk):
    manager.kv_cache_map[req.py_request_id] = kv_cache
    req.context_chunk_size = chunk
    return manager._apply_connector_matched_prefix(req)


@pytest.mark.parametrize("chunk", [CHUNK, 1024])
def test_unreserved_chunk_keeps_only_its_in_page_fallback(chunk):
    """With no page to grow into, a one-block chunk honours nothing (the hit is awaited empty
    and recomputed); a full chunk honours the whole blocks inside its own pages."""
    connector = FakeConnectorManager(num_matched=PROBED, load_async=True)
    manager = make_manager(connector)
    req = FakeRequest(prompt_len=PROMPT)
    kv_cache = FakeKvCache(committed=0, capacity=chunk, grow_ok=False)

    assert _serve(manager, req, kv_cache, chunk) is (chunk != CHUNK)

    assert connector.commits == [(0, 0 if chunk == CHUNK else chunk - TPB, True)]


def test_reserved_prefix_is_honoured_without_a_new_page():
    connector = FakeConnectorManager(num_matched=PROBED, load_async=True)
    manager = make_manager(connector)
    req = FakeRequest(prompt_len=PROMPT)
    kv_cache = FakeKvCache(committed=0, capacity=CHUNK)
    manager.kv_cache_map[0] = kv_cache
    # Scheduler side: the chunk is allocated, then the probed prefix is reserved.
    assert manager.reserve_served_prefix_capacity(req, PROBED + CHUNK)
    assert kv_cache.capacity == PROBED + CHUNK
    assert req.py_ctx_pre_resize_cap == CHUNK
    kv_cache.grow_ok = False  # the pool fills up before the hook runs

    assert _serve(manager, req, kv_cache, CHUNK)

    assert connector.commits == [(0, PROBED, True)]
    assert req.context_current_position == PROBED
    assert req.context_chunk_size == CHUNK
    assert kv_cache.capacity == PROBED + CHUNK


@pytest.mark.parametrize(
    "num_matched, v2, honoured",
    [
        # The store grew between the probe and the query; the grow for the rest fails.
        pytest.param(PROBED + 31 * TPB, True, PROBED, id="offer_past_probe"),
        pytest.param(PROBED + 31 * TPB, False, 0, id="offer_past_probe_v1"),
        pytest.param(PROBED - 10 * TPB, True, PROBED - 10 * TPB, id="offer_inside_reservation"),
    ],
)
def test_an_offer_keeps_the_reserved_part_without_growing(monkeypatch, num_matched, v2, honoured):
    if not v2:
        monkeypatch.setenv("TRTLLM_DISABLE_CTX_FILL_V2", "1")
    connector = FakeConnectorManager(num_matched=num_matched, load_async=True)
    manager = make_manager(connector)
    req = FakeRequest(prompt_len=PROMPT)
    kv_cache = FakeKvCache(committed=0, capacity=PROBED + CHUNK, grow_ok=False)

    assert _serve(manager, req, kv_cache, CHUNK) is bool(honoured)

    assert connector.commits == [(0, honoured, True)]
    if honoured == PROBED:
        assert req.context_current_position == PROBED
        assert req.context_chunk_size == CHUNK
    elif honoured:
        assert kv_cache.capacity == PROBED + CHUNK


def test_reserve_failure_leaves_the_cache_unchanged():
    manager = make_manager(FakeConnectorManager())
    req = FakeRequest(prompt_len=PROMPT)
    kv_cache = FakeKvCache(committed=0, capacity=CHUNK, grow_ok=False)
    manager.kv_cache_map[0] = kv_cache

    assert not manager.reserve_served_prefix_capacity(req, PROBED + CHUNK)

    assert kv_cache.capacity == CHUNK and kv_cache.history_length == 0
    assert req.py_ctx_pre_resize_cap is None


def test_reserve_counts_extra_and_draft_tokens():
    manager = make_manager(FakeConnectorManager(), num_extra_kv_tokens=3)
    req = FakeRequest(prompt_len=PROMPT)
    req.py_draft_tokens = [0, 0]
    kv_cache = FakeKvCache(committed=0, capacity=CHUNK)
    manager.kv_cache_map[0] = kv_cache

    assert manager.reserve_served_prefix_capacity(req, PROBED + CHUNK)
    assert kv_cache.capacity == PROBED + CHUNK + 3
    # Through the prompt end the draft tokens are included too.
    assert manager.reserve_served_prefix_capacity(req, PROMPT)
    assert kv_cache.capacity == PROMPT + 3 + 2


def test_the_reserved_prefix_wins_over_the_legacy_cap_for_a_multi_block_chunk():
    """A two-block chunk's legacy cap honours one block; the reserved prefix honours more."""
    chunk = 2 * TPB
    connector = FakeConnectorManager(num_matched=PROBED + 31 * TPB, load_async=True)
    manager = make_manager(connector)
    req = FakeRequest(prompt_len=PROMPT)
    kv_cache = FakeKvCache(committed=0, capacity=PROBED + chunk, grow_ok=False)

    assert _serve(manager, req, kv_cache, chunk)

    assert connector.commits == [(0, PROBED, True)]
    assert req.context_current_position == PROBED
    assert req.context_chunk_size == chunk


# ---- eligible-share accounting ----


def test_a_deferred_request_is_counted_again_on_its_revisit(monkeypatch):
    """A DEFER re-appends its request; its revisit shares the budget like the first visit."""
    monkeypatch.setenv("TLLM_CTX_REQUEST_CHUNK_CAP", "2048")
    monkeypatch.setenv("TLLM_CTX_SHORTEST_FIRST_STORE_AWARE", "0")
    sched = _sched(_mgr({}))
    original = sched._try_schedule_context
    deferred = []

    def defer_once(req, *args, **kwargs):
        if req.py_request_id == 0 and not deferred:
            deferred.append(req)
            return ScheduleAction.DEFER, 0, False
        return original(req, *args, **kwargs)

    sched._try_schedule_context = defer_once
    reqs = [_req(0, LONG), _req(1, LONG), _req(2, LONG)]
    out = sched.schedule_request(reqs, set())
    assert sorted(ids(out.context_requests)) == [0, 1, 2]
    # 16384 // 3 rounded to the unit, then the rest split between the two left.
    assert reqs[1].context_chunk_size == 5376
    assert reqs[2].context_chunk_size == 5504
    assert reqs[0].context_chunk_size == 5504


def test_a_request_waiting_for_budget_leaves_its_share_to_the_others(monkeypatch):
    monkeypatch.setenv("TLLM_CTX_REQUEST_CHUNK_CAP", "2048")
    monkeypatch.setenv("TLLM_CTX_SHORTEST_FIRST_STORE_AWARE", "0")
    sched = _sched(_mgr({}))
    sched._first_chunk_waits_for_budget = lambda req, budget: req.py_request_id == 1
    reqs = [_req(0, LONG), _req(1, LONG), _req(2, LONG)]
    out = sched.schedule_request(reqs, set())
    assert ids(out.context_requests) == [0, 2]
    assert reqs[0].context_chunk_size == 5376
    # The waiting request takes no share: the last eligible request takes the rest.
    assert reqs[2].context_chunk_size == STEP - 5376
