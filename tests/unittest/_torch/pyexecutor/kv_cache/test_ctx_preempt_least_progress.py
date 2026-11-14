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
"""KVCacheV2Scheduler context preemption victim order.

``preempt_least_prefill_progress`` (or ``TLLM_V2_CTX_PREEMPT_LEAST_PROGRESS=1``)
takes the started request with the least prefill progress; the default takes
the newest one.
"""

from unittest.mock import patch

import pytest
from _torch.pyexecutor.kv_cache.test_kv_cache_v2_scheduler import (
    defer_preemption,
    ids,
    make_ctx_request,
    make_gen_request,
    make_kv_cache_manager,
    make_scheduler,
)

from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2

pytestmark = pytest.mark.cpu_only

OPT_IN = "TLLM_V2_CTX_PREEMPT_LEAST_PROGRESS"


@pytest.fixture(autouse=True)
def _least_progress(monkeypatch):
    monkeypatch.setenv(OPT_IN, "1")
    monkeypatch.delenv("TLLM_V2_CTX_ONLY_PREEMPTION", raising=False)


def _out_of_pages_for(request_id):
    """resize_context that only fails for *request_id*."""
    return lambda req, n: req.py_request_id != request_id


def _started(request_id, position, reused=0, length=100):
    """A started context request that has computed ``position - reused`` prompt tokens."""
    req = make_ctx_request(request_id, length, is_first_context_chunk=False)
    req.context_current_position = position
    req.prepopulated_prompt_len = reused
    return req


def _preemptable(**manager):
    """A manager with GPU as the last cache level whose request 0 is out of pages."""
    return make_kv_cache_manager(
        **{"resize_context_fn": _out_of_pages_for(0), "has_cache_tier_below_gpu": False, **manager}
    )


def _victim(mgr):
    assert mgr.preempt_request.call_count == 1
    return mgr.preempt_request.call_args.args[0].py_request_id


# ---- victim order ----


@pytest.mark.parametrize(("raw", "expected"), [("1", 2), ("", 3), ("0", 3)])
def test_least_progress_victim_only_when_opted_in(monkeypatch, raw, expected):
    """Opted in: the least progressed started request; default: the newest one."""
    monkeypatch.setenv(OPT_IN, raw)
    mgr = _preemptable()
    sched = make_scheduler(mgr, max_num_tokens=1000)
    assert sched._ctx_preempt_least_progress is (expected == 2)
    reqs = [make_ctx_request(0, 100), _started(1, 4000), _started(2, 512), _started(3, 2048)]

    out = sched.schedule_request(reqs, set())

    assert _victim(mgr) == expected
    assert ids(out.recompute_paused_requests) == [expected]
    assert ids(out.context_requests) == []


@pytest.mark.parametrize("opted_in", [True, False])
def test_equal_progress_takes_the_newest(monkeypatch, opted_in):
    if not opted_in:
        monkeypatch.delenv(OPT_IN)
    mgr = _preemptable()
    sched = make_scheduler(mgr, max_num_tokens=1000)
    reqs = [make_ctx_request(0, 100), _started(1, 512), _started(2, 512), _started(3, 4096)]

    sched.schedule_request(reqs, set())

    assert _victim(mgr) == (2 if opted_in else 3)


def test_reused_prefix_is_not_progress():
    """Progress counts the prompt computed past the reused or loaded prefix."""
    mgr = _preemptable()
    sched = make_scheduler(mgr, max_num_tokens=1000)
    deep_reuse = _started(1, 5000, reused=4864)  # 136 tokens computed
    cold = _started(2, 1024)

    sched.schedule_request([make_ctx_request(0, 100), deep_reuse, cold], set())

    assert _victim(mgr) == 1


def test_generation_request_progress_is_its_prompt():
    """A generation victim's progress is the prompt it computed."""
    mgr = _preemptable()
    sched = make_scheduler(mgr, max_num_tokens=1000)
    gen = make_gen_request(1)  # 100-token prompt, all computed
    gen.prompt_len = 100
    reqs = [make_ctx_request(0, 100), _started(2, 64), gen, _started(3, 4096)]

    with patch.object(sched, "_try_schedule_generation") as schedule_gen:
        # Leave gen1 unscheduled in phase 1 so it stays an eligible victim.
        schedule_gen.return_value = (scheduler_v2.ScheduleAction.SKIP, 0, 0, len(reqs))
        sched.schedule_request(reqs, set())

    assert _victim(mgr) == 2


# ---- eligibility is unchanged ----


@pytest.mark.parametrize("disqualify", ["inflight", "suspended", "first_chunk"])
def test_an_ineligible_least_progressed_request_is_passed_over(disqualify):
    """Every eligibility rule still applies; the order only ranks eligible requests."""
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

    assert _victim(mgr) == 2


def test_requests_scheduled_this_pass_stay_protected():
    """A context request that took a chunk earlier in the pass is not a victim."""
    mgr = make_kv_cache_manager(
        resize_context_fn=_out_of_pages_for(5), has_cache_tier_below_gpu=False
    )
    sched = make_scheduler(mgr, max_num_tokens=1000)
    early = _started(1, 64)
    reqs = [early, make_ctx_request(5, 100), _started(2, 512), _started(3, 4096)]

    out = sched.schedule_request(reqs, set())

    assert ids(out.context_requests) == [1]
    assert _victim(mgr) == 2


def test_a_deferred_victim_ends_the_search():
    """A parked victim is not followed by a second one."""
    mgr = _preemptable()
    defer_preemption(mgr, 2)
    sched = make_scheduler(mgr, max_num_tokens=1000)
    reqs = [make_ctx_request(0, 100), _started(1, 4000), _started(2, 512), _started(3, 2048)]

    out = sched.schedule_request(reqs, set())

    assert [c.args[0].py_request_id for c in mgr.preempt_request.call_args_list] == [2]
    assert ids(out.recompute_paused_requests) == []


@pytest.mark.parametrize(
    ("manager", "scheduler"),
    [({"has_cache_tier_below_gpu": True}, {}), ({}, {"enable_recompute_pause": False})],
    ids=["cache_tier_below_gpu", "recompute_pause_off"],
)
def test_preemption_off_preempts_nothing(manager, scheduler):
    mgr = _preemptable(**manager)
    sched = make_scheduler(mgr, max_num_tokens=1000, **scheduler)

    sched.schedule_request([make_ctx_request(0, 100), _started(1, 512)], set())

    mgr.preempt_request.assert_not_called()


# ---- the generation path is unchanged ----


@pytest.mark.parametrize("opted_in", [True, False])
def test_generation_eviction_still_takes_the_tail(monkeypatch, opted_in):
    """Generation-side suspension keeps its tail order whatever the victims' progress."""
    if not opted_in:
        monkeypatch.delenv(OPT_IN)
    calls = [0]

    def alloc_fn(req):
        calls[0] += 1
        return calls[0] != 1

    mgr = make_kv_cache_manager(try_allocate_generation_fn=alloc_fn)
    sched = make_scheduler(mgr, max_num_tokens=100)
    reqs = [make_gen_request(0), _started(1, 64), _started(2, 8192)]

    out = sched.schedule_request(reqs, set())

    assert ids(out.paused_requests) == [2]
    mgr.preempt_request.assert_not_called()
