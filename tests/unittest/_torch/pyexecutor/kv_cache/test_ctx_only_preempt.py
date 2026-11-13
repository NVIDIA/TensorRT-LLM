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
"""KVCacheV2Scheduler context preemption on a disaggregated context worker.

A disaggregated worker runs with recompute pause off. With
``enable_context_only_preemption`` (or ``TLLM_V2_CTX_ONLY_PREEMPTION=1``) a
context-only request still in its prefill stays a victim there.
"""

import pytest
from _torch.pyexecutor.kv_cache.test_kv_cache_v2_scheduler import (
    DISAGG_CTX_TRANS_IN_PROGRESS,
    defer_preemption,
    ids,
    make_ctx_request,
    make_gen_request,
    make_kv_cache_manager,
    make_scheduler,
)

pytestmark = pytest.mark.cpu_only

OPT_IN = "TLLM_V2_CTX_ONLY_PREEMPTION"


@pytest.fixture(autouse=True)
def _ctx_only_preemption(monkeypatch):
    monkeypatch.setenv(OPT_IN, "1")
    # The victim order these tests read is the least-progress one.
    monkeypatch.setenv("TLLM_V2_CTX_PREEMPT_LEAST_PROGRESS", "1")


def _out_of_pages_for(request_id):
    """resize_context that only fails for *request_id*."""
    return lambda req, n: req.py_request_id != request_id


def _started(request_id, position, context_only=True, length=100):
    """A started context request that has computed ``position`` prompt tokens."""
    req = make_ctx_request(request_id, length, is_first_context_chunk=False)
    req.context_current_position = position
    req.is_context_only_request = context_only
    return req


def _requester():
    req = make_ctx_request(0, 100)
    req.is_context_only_request = True
    return req


def _ctx_worker(**manager):
    """A disaggregated context worker whose request 0 is out of pages."""
    mgr = make_kv_cache_manager(
        **{"resize_context_fn": _out_of_pages_for(0), "has_cache_tier_below_gpu": False, **manager}
    )
    return mgr, make_scheduler(mgr, max_num_tokens=1000, enable_recompute_pause=False)


def _victims(mgr):
    return [c.args[0].py_request_id for c in mgr.preempt_request.call_args_list]


def test_a_started_context_only_request_is_preempted():
    mgr, sched = _ctx_worker()
    assert sched._ctx_only_preempt is True
    reqs = [_requester(), _started(1, 4000), _started(2, 512), _started(3, 2048)]

    out = sched.schedule_request(reqs, set())

    assert _victims(mgr) == [2]
    assert ids(out.recompute_paused_requests) == [2]


@pytest.mark.parametrize("raw", ["", "0", "yes"])
def test_off_unless_opted_in(monkeypatch, raw):
    monkeypatch.setenv(OPT_IN, raw)
    mgr, sched = _ctx_worker()
    assert sched._ctx_only_preempt is False

    sched.schedule_request([_requester(), _started(1, 512)], set())

    mgr.preempt_request.assert_not_called()


def test_off_under_pipeline_parallelism():
    """A chunk stays in the PP microbatch ring longer than the in-flight window covers."""
    mgr = make_kv_cache_manager(
        resize_context_fn=_out_of_pages_for(0), has_cache_tier_below_gpu=False
    )
    mgr.mapping.pp_size = 2
    sched = make_scheduler(mgr, max_num_tokens=1000, enable_recompute_pause=False)
    assert sched._ctx_only_preempt is False

    sched.schedule_request([_requester(), _started(1, 512)], set())

    mgr.preempt_request.assert_not_called()


def test_aggregated_worker_is_unchanged():
    mgr = make_kv_cache_manager(
        resize_context_fn=_out_of_pages_for(0), has_cache_tier_below_gpu=False
    )
    sched = make_scheduler(mgr, max_num_tokens=1000)
    assert sched._ctx_only_preempt is False
    # Not context-only: an aggregated request is still a victim with recompute pause on.
    reqs = [make_ctx_request(0, 100), _started(1, 512, context_only=False)]

    sched.schedule_request(reqs, set())

    assert _victims(mgr) == [1]


@pytest.mark.parametrize("disqualify", ["not_context_only", "first_chunk", "inflight", "suspended"])
def test_ineligible_requests_are_never_taken(disqualify):
    mgr, sched = _ctx_worker()
    low = _started(1, 128)
    inflight = set()
    if disqualify == "not_context_only":
        low.is_context_only_request = False
    elif disqualify == "first_chunk":
        low.is_first_context_chunk = True
    elif disqualify == "inflight":
        inflight.add(low.request_id)
    else:
        mgr.kv_cache_map[low.py_request_id].is_active = False

    sched.schedule_request([_requester(), low, _started(2, 2048)], inflight)

    assert _victims(mgr) == [2]


def test_a_remote_tail_request_is_never_taken():
    """Checked on the predicate: a remote-tail request in the pass makes the scheduler run its
    mode exclusively, so the requester would never reach the allocation that preempts."""
    _, sched = _ctx_worker()
    low = _started(1, 128)
    assert sched._is_ctx_only_victim(low)
    low.py_csa2_remote_tail_mode = "source"
    assert not sched._is_ctx_only_victim(low)


def test_a_request_whose_kv_is_being_sent_is_never_taken():
    mgr, sched = _ctx_worker()
    sending = _started(1, 128)
    sending.state_value = DISAGG_CTX_TRANS_IN_PROGRESS
    sending.is_context_init_state = False

    sched.schedule_request([_requester(), sending], set())

    mgr.preempt_request.assert_not_called()


def test_generation_only_requests_are_never_taken():
    mgr, sched = _ctx_worker()
    gen = make_gen_request(1)
    gen.is_context_only_request = False
    gen.is_generation_only_request = True

    sched.schedule_request([_requester(), gen], set())

    mgr.preempt_request.assert_not_called()


@pytest.mark.parametrize("window", ["last", "prev"])
def test_a_request_scheduled_in_the_last_two_passes_is_not_taken(window):
    """Under the overlap executor the previous pass's chunk may still be on the GPU when this
    pass schedules."""
    mgr, sched = _ctx_worker()
    recent, idle = _started(1, 128), _started(2, 4096)
    setattr(sched, f"_ctx_scheduled_{window}", frozenset({recent.py_request_id}))

    sched.schedule_request([_requester(), recent, idle], set())

    assert _victims(mgr) == [2]


def test_a_launched_batch_stays_protected_through_passes_that_launch_nothing():
    """Under attention DP a launched batch stays unprocessed, and may still run, through passes
    that schedule other context work whose batch then does not run (no position moves); those
    passes must not age it out."""
    mgr = make_kv_cache_manager(has_cache_tier_below_gpu=False)
    sched = make_scheduler(mgr, max_num_tokens=1000, enable_recompute_pause=False)
    a, b = _started(1, 128), _started(2, 128)

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


def test_a_pass_with_no_context_work_releases_the_launched_batch():
    """If the last launched batch filled the pool, nothing launches again; the pass that then
    schedules no context work lets the executor process that batch, so it is released."""
    mgr = make_kv_cache_manager(has_cache_tier_below_gpu=False)
    sched = make_scheduler(mgr, max_num_tokens=1000, enable_recompute_pause=False)
    stuck = _started(1, 128)

    sched.schedule_request([stuck], set())
    stuck.context_current_position += 64  # launched
    mgr.prepare_context.side_effect = lambda req, reuse_limit=None: False
    sched.schedule_request([stuck], set())  # detects the launch; schedules nothing
    assert sched._ctx_launched == {1}
    sched.schedule_request([stuck], set())  # the previous pass had no context work

    assert sched._ctx_launched == frozenset()
    sched.schedule_request([stuck], set())  # ages out the two-pass window too
    assert sched._is_ctx_only_victim(stuck)


def test_a_batch_that_did_not_launch_is_not_recorded_as_launched():
    mgr = make_kv_cache_manager(has_cache_tier_below_gpu=False)
    sched = make_scheduler(mgr, max_num_tokens=1000, enable_recompute_pause=False)
    req = _started(1, 128)

    sched.schedule_request([req], set())
    sched.schedule_request([req], set())  # position unchanged: the first batch never ran

    assert sched._ctx_launched == frozenset()


def test_schedule_request_records_two_passes():
    mgr = make_kv_cache_manager(has_cache_tier_below_gpu=False)
    sched = make_scheduler(mgr, max_num_tokens=1000, enable_recompute_pause=False)
    a, b = make_ctx_request(1, 100), make_ctx_request(2, 100)

    sched.schedule_request([a], set())
    assert (sched._ctx_scheduled_last, sched._ctx_scheduled_prev) == ({1}, set())
    sched.schedule_request([b], set())
    assert (sched._ctx_scheduled_last, sched._ctx_scheduled_prev) == ({2}, {1})
    sched.schedule_request([], set())
    assert (sched._ctx_scheduled_last, sched._ctx_scheduled_prev) == (set(), {2})


def test_a_deferred_victim_is_not_scheduled_in_the_same_pass():
    """Regression: a connector save parks the victim in DISAGG_CONTEXT_TRANS_IN_PROGRESS while it
    still sits on the pass's pending list; scheduling it there sets a chunk size outside the
    context phase (C++ assertion)."""
    mgr, sched = _ctx_worker()
    victim = _started(1, 128)
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
