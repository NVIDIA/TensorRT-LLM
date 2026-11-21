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
"""KVCacheV2Scheduler context preemption and shortest-uncached-first aging.

- Victim order: ``TLLM_V2_CTX_PREEMPT_LEAST_PROGRESS=1`` takes the started request with the
  least prefill progress; the default takes the newest one.
- ``TLLM_V2_CTX_ONLY_PREEMPTION=1`` keeps a started context-only request a victim on a
  disaggregated context worker, which runs with recompute pause off.
- Shortest-uncached-first aging counted from arrival (``TLLM_CTX_SRPT_*``).
"""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _torch.pyexecutor.kv_cache.test_kv_cache_v2_scheduler import (
    CONTEXT_INIT,
    DISAGG_CTX_TRANS_IN_PROGRESS,
    defer_preemption,
    ids,
    make_ctx_request,
    make_gen_request,
    make_kv_cache_manager,
    make_scheduler,
)

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2
from tensorrt_llm.llmapi.llm_args import ContextSchedulingPolicy

pytestmark = pytest.mark.cpu_only

LEAST_PROGRESS = "TLLM_V2_CTX_PREEMPT_LEAST_PROGRESS"
CTX_ONLY = "TLLM_V2_CTX_ONLY_PREEMPTION"
ENV = (
    LEAST_PROGRESS,
    CTX_ONLY,
    "TLLM_CTX_SRPT_AGE_FROM_ARRIVAL",
    "TLLM_CTX_SRPT_MAX_WAIT_S",
    "TLLM_CTX_SRPT_AGED_SLOTS",
    "TLLM_CTX_SRPT_MAX_WAIT_PASSES",
    "TLLM_CTX_SHORTEST_FIRST_STORE_AWARE",
    "TLLM_CTX_REQUEST_CHUNK_CAP",
    "TLLM_CTX_CHUNK_CAP_ELIGIBLE_SHARE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def least_progress(monkeypatch):
    monkeypatch.setenv(LEAST_PROGRESS, "1")


@pytest.fixture
def ctx_only(monkeypatch):
    monkeypatch.setenv(CTX_ONLY, "1")
    monkeypatch.setenv(LEAST_PROGRESS, "1")


def _out_of_pages_for(request_id):
    """resize_context that only fails for *request_id*."""
    return lambda req, n: req.py_request_id != request_id


def _started(request_id, position, reused=0, context_only=None, length=100):
    """A started context request that has computed ``position - reused`` prompt tokens."""
    req = make_ctx_request(request_id, length, is_first_context_chunk=False)
    req.context_current_position = position
    req.prepopulated_prompt_len = reused
    if context_only is not None:
        req.is_context_only_request = context_only
    return req


def _preemptable(**manager):
    """A manager with GPU as the last cache level whose request 0 is out of pages."""
    return make_kv_cache_manager(
        **{"resize_context_fn": _out_of_pages_for(0), "has_cache_tier_below_gpu": False, **manager}
    )


def _victims(mgr):
    return [c.args[0].py_request_id for c in mgr.preempt_request.call_args_list]


# ---- victim order ----


@pytest.mark.parametrize(("raw", "expected"), [("1", 2), ("", 3)])
def test_least_progress_victim_only_when_opted_in(monkeypatch, raw, expected):
    """Opted in: the least progressed started request; default: the newest one."""
    monkeypatch.setenv(LEAST_PROGRESS, raw)
    mgr = _preemptable()
    sched = make_scheduler(mgr, max_num_tokens=1000)
    assert sched._ctx_preempt_least_progress is (expected == 2)
    reqs = [make_ctx_request(0, 100), _started(1, 4000), _started(2, 512), _started(3, 2048)]

    out = sched.schedule_request(reqs, set())

    assert _victims(mgr) == [expected]
    assert ids(out.recompute_paused_requests) == [expected]
    assert ids(out.context_requests) == []


def test_reused_prefix_is_not_progress(least_progress):
    """Progress counts the prompt computed past the reused or loaded prefix."""
    mgr = _preemptable()
    sched = make_scheduler(mgr, max_num_tokens=1000)
    deep_reuse = _started(1, 5000, reused=4864)  # 136 tokens computed

    sched.schedule_request([make_ctx_request(0, 100), deep_reuse, _started(2, 1024)], set())

    assert _victims(mgr) == [1]
    stats = sched._ctx_preempt_stats
    assert (stats.passes, stats.preempted, stats.preempted_tokens) == (1, 1, 136)


@pytest.mark.parametrize("disqualify", ["inflight", "suspended", "first_chunk"])
def test_an_ineligible_least_progressed_request_is_passed_over(least_progress, disqualify):
    mgr = _preemptable()
    sched = make_scheduler(mgr, max_num_tokens=1000)
    low = _started(1, 128)
    inflight = set()
    if disqualify == "inflight":
        inflight.add(low.request_id)
    elif disqualify == "suspended":
        mgr.kv_cache_map[low.py_request_id].is_active = False
    else:
        low.is_first_context_chunk = True
    reqs = [make_ctx_request(0, 100), low, _started(2, 2048), _started(3, 4096)]

    sched.schedule_request(reqs, inflight)

    assert _victims(mgr) == [2]


def test_a_deferred_victim_ends_the_search_and_is_counted(least_progress):
    mgr = _preemptable()
    defer_preemption(mgr, 2)
    sched = make_scheduler(mgr, max_num_tokens=1000)
    reqs = [make_ctx_request(0, 100), _started(1, 4000), _started(2, 512), _started(3, 2048)]

    out = sched.schedule_request(reqs, set())

    assert _victims(mgr) == [2]
    assert ids(out.recompute_paused_requests) == []
    assert (sched._ctx_preempt_stats.deferred, sched._ctx_preempt_stats.preempted) == (1, 0)


@pytest.mark.parametrize(
    ("manager", "scheduler"),
    [({"has_cache_tier_below_gpu": True}, {}), ({}, {"enable_recompute_pause": False})],
    ids=["cache_tier_below_gpu", "recompute_pause_off"],
)
def test_preemption_off_preempts_nothing_and_is_counted(least_progress, manager, scheduler):
    mgr = _preemptable(**manager)
    sched = make_scheduler(mgr, max_num_tokens=1000, **scheduler)

    sched.schedule_request([make_ctx_request(0, 100), _started(1, 512)], set())

    mgr.preempt_request.assert_not_called()
    assert sched._ctx_preempt_stats.preempt_off == 1


def test_srpt_aging_still_protects_started_requests_from_a_first_chunk(monkeypatch, least_progress):
    monkeypatch.setenv("TLLM_CTX_SRPT_MAX_WAIT_PASSES", "128")
    mgr = _preemptable()
    sched = make_scheduler(
        mgr,
        max_num_tokens=1000,
        context_scheduling_policy=scheduler_v2.ContextSchedulingPolicy.SHORTEST_UNCACHED_FIRST,
    )
    reqs = [make_ctx_request(0, 100), _started(1, 512), _started(2, 4096)]

    sched.schedule_request(reqs, set())

    mgr.preempt_request.assert_not_called()
    assert sched._ctx_preempt_stats.no_victim == 1


# ---- context-only preemption on a disaggregated context worker ----


def _requester():
    req = make_ctx_request(0, 100)
    req.is_context_only_request = True
    return req


def _ctx_worker():
    """A disaggregated context worker whose request 0 is out of pages."""
    mgr = _preemptable()
    return mgr, make_scheduler(mgr, max_num_tokens=1000, enable_recompute_pause=False)


def _idle_ctx_worker():
    mgr = make_kv_cache_manager(has_cache_tier_below_gpu=False)
    return mgr, make_scheduler(mgr, max_num_tokens=1000, enable_recompute_pause=False)


def test_a_started_context_only_request_is_preempted(ctx_only):
    mgr, sched = _ctx_worker()
    assert sched._ctx_only_preempt is True
    reqs = [_requester()] + [_started(i, p, context_only=True) for i, p in ((1, 4000), (2, 512))]

    out = sched.schedule_request(reqs, set())

    assert _victims(mgr) == [2]
    assert ids(out.recompute_paused_requests) == [2]
    stats = sched._ctx_preempt_stats
    assert (stats.preempted, stats.preempted_tokens, stats.preempt_off) == (1, 512, 0)


@pytest.mark.parametrize(
    "raw, pp_size", [("", 1), ("yes", 1), ("1", 2)], ids=["unset", "not_one", "pipeline_parallel"]
)
def test_context_only_preemption_off_unless_opted_in(monkeypatch, ctx_only, raw, pp_size):
    """Under PP a chunk stays in the microbatch ring longer than the in-flight window covers."""
    monkeypatch.setenv(CTX_ONLY, raw)
    mgr = _preemptable()
    mgr.mapping.pp_size = pp_size
    sched = make_scheduler(mgr, max_num_tokens=1000, enable_recompute_pause=False)
    assert sched._ctx_only_preempt is False

    sched.schedule_request([_requester(), _started(1, 512, context_only=True)], set())

    mgr.preempt_request.assert_not_called()
    if pp_size == 1:
        assert sched._ctx_preempt_stats.preempt_off == 1


@pytest.mark.parametrize("disqualify", ["not_context_only", "first_chunk", "inflight", "suspended"])
def test_ineligible_context_only_requests_are_never_taken(ctx_only, disqualify):
    mgr, sched = _ctx_worker()
    low = _started(1, 128, context_only=True)
    inflight = set()
    if disqualify == "not_context_only":
        low.is_context_only_request = False
    elif disqualify == "first_chunk":
        low.is_first_context_chunk = True
    elif disqualify == "inflight":
        inflight.add(low.request_id)
    else:
        mgr.kv_cache_map[low.py_request_id].is_active = False

    sched.schedule_request([_requester(), low, _started(2, 2048, context_only=True)], inflight)

    assert _victims(mgr) == [2]


@pytest.mark.parametrize("kind", ["kv_being_sent", "generation_only"])
def test_sending_and_generation_only_requests_are_never_taken(ctx_only, kind):
    mgr, sched = _ctx_worker()
    if kind == "kv_being_sent":
        other = _started(1, 128, context_only=True)
        other.state_value = DISAGG_CTX_TRANS_IN_PROGRESS
        other.is_context_init_state = False
    else:
        other = make_gen_request(1)
        other.is_context_only_request = False
        other.is_generation_only_request = True

    sched.schedule_request([_requester(), other], set())

    mgr.preempt_request.assert_not_called()


@pytest.mark.parametrize("window", ["last", "prev"])
def test_a_request_scheduled_in_the_last_two_passes_is_not_taken(ctx_only, window):
    """Under the overlap executor the previous pass's chunk may still be on the GPU."""
    mgr, sched = _ctx_worker()
    recent, idle = _started(1, 128, context_only=True), _started(2, 4096, context_only=True)
    setattr(sched, f"_ctx_scheduled_{window}", frozenset({recent.py_request_id}))

    sched.schedule_request([_requester(), recent, idle], set())

    assert _victims(mgr) == [2]


def test_a_launched_batch_stays_protected_through_passes_that_launch_nothing(ctx_only):
    """Under attention DP a launched batch may still run through passes whose own context
    batch does not run (no position moves); those passes must not age it out."""
    mgr, sched = _idle_ctx_worker()
    a, b = _started(1, 128, context_only=True), _started(2, 128, context_only=True)

    mgr.prepare_context.side_effect = lambda req, reuse_limit=None: req.py_request_id == 1
    sched.schedule_request([a, b], set())  # pass N schedules a chunk of request 1 ...
    a.context_current_position += 64  # ... and the executor ran that batch
    mgr.prepare_context.side_effect = lambda req, reuse_limit=None: req.py_request_id == 2
    for _ in range(3):  # passes that schedule request 2, whose batch never runs
        sched.schedule_request([b, a], set())
    assert sched._ctx_scheduled_last == sched._ctx_scheduled_prev == {2}
    assert sched._ctx_launched == {1}
    assert not sched._is_ctx_only_victim(a)

    b.context_current_position += 64  # request 2's batch finally launches
    sched.schedule_request([b, a], set())
    assert sched._ctx_launched == {2}
    assert sched._is_ctx_only_victim(a)


def test_a_pass_with_no_context_work_releases_the_launched_batch(ctx_only):
    mgr, sched = _idle_ctx_worker()
    stuck = _started(1, 128, context_only=True)

    sched.schedule_request([stuck], set())
    stuck.context_current_position += 64  # launched
    mgr.prepare_context.side_effect = lambda req, reuse_limit=None: False
    sched.schedule_request([stuck], set())  # detects the launch; schedules nothing
    assert sched._ctx_launched == {1}
    sched.schedule_request([stuck], set())  # the previous pass had no context work

    assert sched._ctx_launched == frozenset()
    sched.schedule_request([stuck], set())  # ages out the two-pass window too
    assert sched._is_ctx_only_victim(stuck)


def test_a_batch_the_executor_keeps_unprocessed_stays_protected(ctx_only):
    mgr, sched = _idle_ctx_worker()
    held = _started(1, 128, context_only=True)

    sched.schedule_request([held], set())
    held.context_current_position += 64  # launched, then kept unprocessed
    sched.set_unprocessed_context(frozenset({1}))
    mgr.prepare_context.side_effect = lambda req, reuse_limit=None: False
    for _ in range(4):  # generation-only passes that cannot queue
        sched.schedule_request([held], set())
    assert sched._ctx_launched == frozenset()
    assert not sched._is_ctx_only_victim(held)

    sched.set_unprocessed_context(frozenset())  # the executor processed it
    assert sched._is_ctx_only_victim(held)


@pytest.mark.parametrize("held", [True, False])
def test_the_executor_reports_the_context_requests_of_a_held_batch(held):
    ex = PyExecutor.__new__(PyExecutor)
    ex.scheduler = Mock()
    batch = SimpleNamespace(
        scheduled_requests=SimpleNamespace(
            context_requests=[SimpleNamespace(py_request_id=1), SimpleNamespace(py_request_id=3)]
        )
    )

    ex._report_unprocessed_context(batch if held else None)

    ex.scheduler.set_unprocessed_context.assert_called_once_with(
        frozenset({1, 3}) if held else frozenset()
    )


def test_a_deferred_victim_is_not_scheduled_in_the_same_pass(ctx_only):
    """Regression: a connector save parks the victim in DISAGG_CONTEXT_TRANS_IN_PROGRESS while it
    still sits on the pass's pending list; scheduling it there would set a chunk size outside the
    context phase."""
    mgr, sched = _ctx_worker()
    victim = _started(1, 128, context_only=True)
    parked = defer_preemption(mgr, victim.py_request_id)
    real_preempt = mgr.preempt_request.side_effect

    def park(req):
        released = real_preempt(req)
        if not released:
            req.state_value = DISAGG_CTX_TRANS_IN_PROGRESS
            req.is_context_init_state = False
        return released

    mgr.preempt_request.side_effect = park

    out = sched.schedule_request([_requester(), victim], set())

    assert _victims(mgr) == [1]
    assert parked == {1}
    assert ids(out.context_requests) == []
    assert ids(out.recompute_paused_requests) == []
    assert all(c.args[0].py_request_id != 1 for c in mgr.resize_context.call_args_list)


# ---- shortest-uncached-first aging counted from arrival ----

SRPT = ContextSchedulingPolicy.SHORTEST_UNCACHED_FIRST
STEP = 16384
UNIT = 128
CAP = 8192
MAX_WAIT = 128
LONG = 800_000
LOADING = scheduler_v2.LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS.value


def _sched(monkeypatch, **env):
    monkeypatch.setenv("TLLM_CTX_REQUEST_CHUNK_CAP", str(CAP))
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    return make_scheduler(
        make_kv_cache_manager(),
        max_num_tokens=STEP,
        ctx_chunk_config=(None, UNIT),
        context_scheduling_policy=SRPT,
    )


def _req(request_id, length):
    req = make_ctx_request(request_id, length)
    req.py_kv_gpu_hold = 0
    return req


def _advance(reqs, scheduled):
    """Apply one pass's chunks the way the executor does; return the requests still prefilling."""
    for req in scheduled:
        req.context_current_position += req.context_chunk_size
        req.context_remaining_length -= req.context_chunk_size
        req.is_first_context_chunk = False
        req.context_chunk_size = 0
        if req.context_remaining_length <= 0:
            req.state_value = CONTEXT_INIT + 1
    return [r for r in reqs if r.state_value == CONTEXT_INIT]


def _passes_to_finish_long(sched, max_passes, short_len=6144, pool=4):
    """Run one cold LONG prompt against a short-prompt stream that always fills the step.

    Returns the pass at which the long prompt finished, or None.
    """
    long_req = _req(0, LONG)
    reqs = [long_req]
    next_id = 1
    for n in range(1, max_passes + 1):
        while sum(1 for r in reqs if r is not long_req) < pool:
            reqs.append(_req(next_id, short_len))
            next_id += 1
        out = sched.schedule_request(reqs, set())
        reqs = _advance(reqs, out.context_requests)
        if long_req.state_value != CONTEXT_INIT:
            return n
    return None


@pytest.mark.parametrize("from_arrival", [True, False])
def test_long_cold_prompt_finishes_within_its_bound_only_with_arrival_aging(
    monkeypatch, from_arrival
):
    """Per-chunk aging gives the long prompt one chunk per MAX_WAIT passes and starves it."""
    sched = _sched(
        monkeypatch,
        TLLM_CTX_SRPT_AGE_FROM_ARRIVAL=int(from_arrival),
        TLLM_CTX_SRPT_MAX_WAIT_PASSES=MAX_WAIT,
        TLLM_CTX_SRPT_MAX_WAIT_S=0,
    )
    if from_arrival:
        bound = MAX_WAIT + math.ceil(LONG / CAP) + 8
        done = _passes_to_finish_long(sched, bound)
        assert done is not None and done <= bound, done
    else:
        assert _passes_to_finish_long(sched, 20 * MAX_WAIT) is None


def test_short_prompts_keep_half_the_step_while_a_long_prompt_is_aged(monkeypatch):
    sched = _sched(monkeypatch, TLLM_CTX_SRPT_MAX_WAIT_PASSES=1, TLLM_CTX_SRPT_MAX_WAIT_S=0)
    long_req = _req(0, LONG)
    reqs = [long_req] + [_req(i, 6144) for i in range(1, 5)]
    sched.schedule_request(reqs, set())
    reqs = _advance(reqs, [])
    out = sched.schedule_request(reqs, set())
    assert ids(out.context_requests)[0] == 0
    assert long_req.context_chunk_size == CAP
    assert (
        sum(r.context_chunk_size for r in out.context_requests if r is not long_req) == STEP - CAP
    )


def test_aged_request_stays_aged_until_its_prefill_ends(monkeypatch):
    sched = _sched(monkeypatch, TLLM_CTX_SRPT_MAX_WAIT_PASSES=2, TLLM_CTX_SRPT_MAX_WAIT_S=0)
    long_req = _req(0, LONG)
    for _ in range(3):
        sched._advance_ctx_srpt_clock([long_req])
    assert sched._is_ctx_srpt_aged(long_req)
    long_req.is_first_context_chunk = False
    for _ in range(5):
        sched._advance_ctx_srpt_clock([long_req])
        assert sched._is_ctx_srpt_aged(long_req)
    long_req.state_value = CONTEXT_INIT + 1
    sched._advance_ctx_srpt_clock([long_req])
    assert sched._ctx_srpt_waiting_since == {}


def test_age_survives_an_async_connector_load(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(scheduler_v2.time, "monotonic", lambda: now[0])
    sched = _sched(monkeypatch, TLLM_CTX_SRPT_MAX_WAIT_PASSES=3, TLLM_CTX_SRPT_MAX_WAIT_S=30)
    req = _req(0, LONG)
    sched._advance_ctx_srpt_clock([req])
    now[0] += 31.0
    sched._advance_ctx_srpt_clock([req])
    assert sched._is_ctx_srpt_aged(req)
    req.state_value = LOADING
    for _ in range(4):
        sched._advance_ctx_srpt_clock([req])
    assert sched._ctx_srpt_waiting_since == {0: 1}
    assert sched._ctx_srpt_arrival_s == {0: 100.0}
    req.state_value = CONTEXT_INIT
    req.is_first_context_chunk = False
    sched._advance_ctx_srpt_clock([req])
    assert sched._is_ctx_srpt_aged(req)
    sched._advance_ctx_srpt_clock([])
    assert sched._ctx_srpt_waiting_since == {}
    assert sched._ctx_srpt_arrival_s == {}


@pytest.mark.parametrize(
    ("attention_dp", "tp_size", "pp_size", "expected"),
    [(True, 2, 1, 120.0), (False, 2, 1, 0.0), (True, 2, 2, 0.0)],
)
def test_wall_clock_bound_only_where_ranks_schedule_alone(
    monkeypatch, attention_dp, tp_size, pp_size, expected
):
    monkeypatch.setenv("TLLM_CTX_REQUEST_CHUNK_CAP", str(CAP))
    mgr = make_kv_cache_manager()
    mgr.mapping.enable_attention_dp = attention_dp
    mgr.mapping.tp_size = tp_size
    mgr.mapping.pp_size = pp_size
    mgr.mapping.cp_size = 1
    sched = make_scheduler(
        mgr, max_num_tokens=STEP, ctx_chunk_config=(None, UNIT), context_scheduling_policy=SRPT
    )
    assert sched._ctx_srpt_max_wait_s == expected


def _order(sched, reqs, passes):
    for _ in range(passes):
        sched._advance_ctx_srpt_clock(reqs)
    return [r.py_request_id for r in sched._order_ctx_shortest_first(reqs)]


@pytest.mark.parametrize(
    "env, expected",
    [
        pytest.param({}, [0, 3, 2, 1], id="one_aged_slot"),
        pytest.param({"TLLM_CTX_SRPT_AGED_SLOTS": 0}, [0, 1, 3, 2], id="zero_slots"),
        pytest.param({"TLLM_CTX_SRPT_AGE_FROM_ARRIVAL": 0}, [0, 1, 3, 2], id="kill_switch"),
    ],
)
def test_aged_slots_then_shortest_first(monkeypatch, env, expected):
    sched = _sched(monkeypatch, TLLM_CTX_SRPT_MAX_WAIT_PASSES=2, TLLM_CTX_SRPT_MAX_WAIT_S=0, **env)
    old = [_req(0, 300_000), _req(1, 200_000)]
    for _ in range(3):
        sched._advance_ctx_srpt_clock(old)
    assert sched._ctx_srpt_aged_slots == (0 if env else 1)
    assert _order(sched, old + [_req(2, 9000), _req(3, 3000)], 1) == expected
