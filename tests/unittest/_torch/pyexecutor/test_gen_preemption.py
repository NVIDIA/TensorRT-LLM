# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GEN slot preemption (scheduler_config.gen_preemption), pause in place."""

import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor._util import compute_max_num_sequences
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import gen_preemption
from tensorrt_llm._torch.pyexecutor.scheduler.gen_admission_queue import (
    GenAdmissionConfig,
    GenAdmissionWaitingQueue,
)
from tensorrt_llm._torch.pyexecutor.scheduler.gen_preemption import (
    GenPreemption,
    GenPreemptionSettings,
    _Req,
    create_gen_preemption,
    resident_headroom,
)
from tensorrt_llm.llmapi.llm_args import GenPreemptionConfig

pytestmark = pytest.mark.cpu_only

GEN = LlmRequestState.GENERATION_IN_PROGRESS
ON = GenPreemptionSettings(tpot_budget=0.25)
SLO = GenPreemptionSettings(target_output_tps=40.0)
AUTO = GenAdmissionConfig(ttft_percentile=90.0)


def make_req(req_id, decoding_iter=100, generated=100, max_new_tokens=4096, state=GEN):
    req = SimpleNamespace(
        py_request_id=req_id,
        state=state,
        py_decoding_iter=decoding_iter,
        prompt_len=1000,
        py_max_new_tokens=max_new_tokens,
        is_dummy_request=False,
        generated=generated,
    )
    req.get_num_tokens = lambda beam: req.prompt_len + req.generated
    return req


def step(reqs, iters=1, tokens_per_iter=1):
    for req in reqs:
        req.py_decoding_iter += iters
        req.generated += iters * tokens_per_iter


def make_preemption(
    max_paused=4, decode_s=20.0, unpaused_tps=10.0, tpot_budget=0.5, target_output_tps=None, **knobs
):
    """D = 20 s (window 3 s at g = 1, max pause 7 s), cap ``max_paused``, no
    minimum run; a 50% TPOT budget unless ``target_output_tps`` is given.
    ``knobs`` override policy constants."""
    budget = None if target_output_tps is not None else tpot_budget
    p = GenPreemption(GenPreemptionSettings(budget, target_output_tps), 128)
    p.max_paused, p.mean_decode_s = max_paused, decode_s
    p.mean_tpot_s = None if unpaused_tps is None else 1.0 / unpaused_tps
    knobs = {"min_run_frac": 0.0, "warmup_min_run_s": 0.0, "min_remaining_tokens": 0, **knobs}
    for name, value in knobs.items():
        assert hasattr(p, name), name
        setattr(p, name, value)
    return p


def run_for(p, reqs, t0=0.0, seconds=10.0, rates=None):
    """First token at ``t0``, then ``rates`` (default 10) tokens/s."""
    p.observe(reqs, t0)
    for i, req in enumerate(reqs):
        n = int(round((10.0 if rates is None else rates[i]) * seconds))
        req.generated += n
        req.py_decoding_iter += n
    return t0 + seconds


def victims(p, reqs, now, n=1):
    p.observe(reqs, now)
    return [r.py_request_id for r in p._victims(reqs, n)]


def complete(p, now, decode_s, tokens, paused_s=0.0, target_ready=True, sacrificed=False):
    p._now = now
    start = now - decode_s - paused_s
    r = _Req(
        0, start, first_token_t=start, gen=tokens, paused_total=paused_s, sacrificed=sacrificed
    )
    p._on_left(r, target_ready)


def make_item(req_id, gen_arrival, orig_arrival):
    item = RequestQueueItem(req_id, Mock())
    item.admit_arrival, item.admit_orig_arrival = gen_arrival, orig_arrival
    return item


def slack_queue(x=10.0, g=100.0, now=100.0, items=()):
    q = GenAdmissionWaitingQueue(AUTO)
    q.adaptive.X, q.adaptive.G, q.adaptive.s2f_s = x, g, 0.5
    q.add_requests(list(items))
    q.set_now(now)
    return q


def at_risk_queue(now=100.0, n=1):
    """``n`` waiters with 0.3 s of slack against X = 10 s."""
    return slack_queue(now=now, items=[make_item(1000 + i, 99.0, 90.8) for i in range(n)])


def running(p, n):
    """``n`` requests decoding at 10 tokens/s since t=90."""
    reqs = [make_req(i) for i in range(n)]
    run_for(p, reqs, t0=90.0)
    return reqs


class TestSettingsAndPools:
    def test_from_config(self):
        assert not GenPreemptionSettings.from_config(GenPreemptionConfig()).enabled
        slo = GenPreemptionSettings.from_config(GenPreemptionConfig(target_output_tps=40))
        assert slo.enabled and slo.tpot_budget is None
        budget = GenPreemptionSettings.from_config(GenPreemptionConfig(tpot_budget=0.1))
        assert budget.enabled and budget.target_output_tps is None
        assert [resident_headroom(n, budget) for n in (128, 64, 1)] == [48, 24, 1]
        assert resident_headroom(128, GenPreemptionSettings()) == 0

    def test_pools_reserve_the_headroom(self, monkeypatch):
        mapping = Mock(pp_size=1)
        mapping.has_pp.return_value = False
        mamba = object.__new__(MambaHybridCacheManagerV2)
        mamba.max_batch_size, mamba.mapping = 128, mapping
        assert compute_max_num_sequences(mapping, 128, False) == 128
        assert mamba._max_resident_sequences() == 128
        monkeypatch.setattr(gen_preemption, "_active_settings", ON)
        assert compute_max_num_sequences(mapping, 128, False) == 176
        assert mamba._max_resident_sequences() == 176
        assert compute_max_num_sequences(mapping, 128, False, enable_overlap_headroom=True) == 304


class TestBudgetVictims:
    def test_longest_resident_first_ties_newest(self):
        p = make_preemption()
        old = [make_req(1), make_req(2)]
        p.observe(old, 0.0)
        reqs = old + [make_req(3)]
        p.observe(reqs, 5.0)
        assert victims(p, reqs, 6.0, n=3) == [2]  # one pause per pass
        p = make_preemption(max_per_pass=3)
        p.observe(old, 0.0)
        p.observe(reqs, 5.0)
        assert victims(p, reqs, 6.0, n=3) == [2, 1, 3]
        p._pause(p._victims(reqs, 1)[0])
        p.max_paused = 2  # the paused cap
        assert victims(p, reqs, 6.0, n=3) == [1]

    @pytest.mark.parametrize(
        "make, ok",
        [
            (lambda r: setattr(r, "py_decoding_iter", 0), False),
            (lambda r: setattr(r, "state", LlmRequestState.GENERATION_TO_COMPLETE), False),
            (lambda r: setattr(r, "is_dummy_request", True), False),
            (lambda r: setattr(r, "py_max_new_tokens", 132), False),  # 32 left
            (lambda r: setattr(r, "py_max_new_tokens", 133), True),
        ],
    )
    def test_eligibility(self, make, ok):
        p = make_preemption(min_remaining_tokens=32)
        req = make_req(1, generated=0)
        now = run_for(p, [req])
        make(req)
        assert victims(p, [req], now) == ([1] if ok else [])

    def test_min_run_hysteresis_and_pause_limit(self):
        p = make_preemption(decode_s=100.0, min_run_frac=0.07, hysteresis_iters=10)
        req = make_req(1)
        p.observe([req], 0.0)
        assert victims(p, [req], 6.9) == [] and victims(p, [req], 7.5) == [1]
        for n in (1, 2):
            p._pause(req)
            assert victims(p, [req], 7.5) == []  # paused
            p._resume(1)
            assert victims(p, [req], 7.5) == []  # hysteresis
            step([req], iters=10)
            assert victims(p, [req], 7.5) == ([1] if n == 1 else [])  # 2 pauses max

    def test_a_request_that_leaves_while_paused_is_forgotten(self):
        p = make_preemption()
        reqs = [make_req(1), make_req(2)]
        p.observe(reqs, 0.0)
        p._pause(reqs[1])
        p.observe(reqs[:1], 1.0)
        assert p.paused_ids == frozenset() and 2 not in p._reqs


class TestController:
    def test_warm_start_and_running_means(self):
        p = GenPreemption(ON, max_num_active_requests=128)
        assert p.max_paused == 48
        assert (p.gain, p.slack_s, p.max_pause_s, p.min_run_s) == (1.0, 0.0, 10.0, 2.0)
        complete(p, 100.0, decode_s=30.0, tokens=1500)  # seeds D and TPOT
        assert (p.mean_decode_s, p.mean_tpot_s) == (30.0, 0.02)
        assert (p.max_pause_s, p.min_run_s) == pytest.approx((10.5, 2.1))
        assert p.slack_s == pytest.approx(4.5 * p.gain)
        complete(p, 101.0, decode_s=80.0, tokens=1500)
        assert p.mean_decode_s == pytest.approx(30.0 + 50.0 / 500)

    def test_one_step_per_completion(self):
        p = GenPreemption(ON, max_num_active_requests=128)
        p.mean_tpot_s = 0.02  # setpoint 0.8 x 25% = 20%
        complete(p, 10.0, decode_s=20.0, tokens=1000)
        assert p.theta == pytest.approx(0.2 / (0.2 * 3000.0))
        theta = p.theta
        complete(p, 11.0, decode_s=20.0, tokens=1000, paused_s=5.0)  # inflation 25%
        assert p.theta == pytest.approx(theta - 0.05 / (0.2 * 3000.0))
        assert p.mean_tpot_s == pytest.approx(0.02)  # pauses excluded
        theta = p.theta
        complete(p, 12.0, decode_s=20.0, tokens=1000, paused_s=5.0, target_ready=False)
        assert p.theta == theta

    @pytest.mark.parametrize("paused_s, gain", [(0.0, 2.0), (10.0, 0.25)])
    def test_clamps(self, paused_s, gain):
        p = GenPreemption(ON, max_num_active_requests=128)
        for i in range(10000):
            complete(p, float(i), decode_s=20.0, tokens=1000, paused_s=paused_s)
        assert p.gain == pytest.approx(gain)

    @pytest.mark.parametrize("g0", [0.25, 1.0, 2.0])
    def test_converges_to_the_setpoint(self, g0):
        """Toy plant with inflation 0.25 g: the 20% setpoint is met at g = 0.8."""
        p = GenPreemption(ON, max_num_active_requests=128)
        p.theta = math.log(g0)
        for i in range(20000):
            complete(p, float(i), decode_s=20.0, tokens=1000, paused_s=0.25 * p.gain * 20.0)
        assert p.gain == pytest.approx(0.8, rel=0.02)


class TestAtRisk:
    def test_slack_bound(self):
        # 1: slack 0.3 s; 2: slack 4.5 s.
        q = slack_queue(items=[make_item(1, 99.0, 90.8), make_item(2, 99.5, 95.0)])
        assert [q.num_at_risk(s) for s in (5.0, 1.0, 0.2)] == [2, 1, 0]

    def test_late_and_guard_tier_heads_trigger(self):
        q = slack_queue(items=[make_item(1, 99.0, 80.0), make_item(2, 99.0, 85.0)])
        assert q.num_at_risk(5.0) == 2  # both late
        q = slack_queue(g=5.0, items=[make_item(2, 99.9, 90.8), make_item(1, 94.0, 94.0)])
        assert q.num_at_risk(5.0) == 2  # guard-tier head, slack 3.5 and 0.3
        # An on-time waiter (slack 9.5 s) is popped before a late one.
        q = slack_queue(items=[make_item(1, 99.0, 80.0), make_item(3, 99.9, 100.0)])
        assert q.num_at_risk(5.0) == 0

    @pytest.mark.parametrize("percentile", [90.0, 100.0])
    def test_no_target_no_risk(self, percentile):
        q = GenAdmissionWaitingQueue(GenAdmissionConfig(ttft_percentile=percentile))
        q.add_requests([make_item(1, 99.0, 91.0)])
        q.set_now(100.0)
        assert q.num_at_risk(100.0) == 0 and q.target_s is None


class TestStep:
    def test_pauses_only_when_full_and_at_risk(self):
        p = make_preemption()
        reqs = running(p, 4)
        assert p.step(reqs, 100.0, 5, 4, at_risk_queue()) == 4  # a free slot
        on_time = slack_queue(items=[make_item(1000, 99.0, 99.0)])  # slack 8.5 s
        assert p.step(reqs, 100.0, 4, 4, on_time) == 4
        assert p.step(reqs, 100.0, 4, 4, at_risk_queue()) == 3 and p.paused_ids == {3}

    def test_warm_up_window_is_zero(self):
        p = make_preemption(decode_s=None)
        reqs = running(p, 4)
        assert p.step(reqs, 100.0, 4, 4, at_risk_queue()) == 4
        late = slack_queue(items=[make_item(1000, 99.0, 80.0)])
        assert p.step(reqs, 100.0, 4, 4, late) == 3 and p.paused_ids == {3}

    def test_resume_lane_goes_before_new_admissions(self):
        p = make_preemption(max_paused=2, max_per_request=1)
        reqs = running(p, 4)
        assert p.step(reqs, 100.0, 4, 4, at_risk_queue()) == 3
        reqs.pop(0)  # the waiter took 3's slot, then 0 finished
        assert p.step(reqs, 100.1, 4, 4, at_risk_queue(100.1)) == 3
        assert p.paused_ids == {2}  # 3 resumed; 2 paused for the next waiter

    def test_forced_resume_after_max_pause(self):
        p = make_preemption(max_paused=2)
        reqs = running(p, 4)
        p.step(reqs, 100.0, 4, 4, at_risk_queue())
        assert p.step(reqs, 106.5, 4, 5, slack_queue(now=106.5)) == 4 and p.paused_ids == {3}
        assert p.step(reqs, 107.5, 4, 5, slack_queue(now=107.5)) == 5 and not p.paused_ids
        assert p.stats["forced"] == 1

    def test_fifo_resume(self):
        p = make_preemption(max_paused=3, max_per_pass=3)
        reqs = running(p, 6)
        assert p.step(reqs, 100.0, 6, 6, at_risk_queue(n=3)) == 3
        order = []
        for _ in range(3):
            before = p.paused_ids
            p.step(reqs, 100.0, 6, 5 + len(p.paused_ids), slack_queue())  # one slot free
            order += sorted(before - p.paused_ids)
        assert order == [5, 4, 3]

    def test_controller_starts_with_the_admission_target(self):
        p = make_preemption(decode_s=None, unpaused_tps=None)
        reqs = running(p, 4)
        p.observe(reqs, 100.0)
        p.step(reqs[1:], 100.0, 8, 3, slack_queue())  # 0 finished
        assert p.mean_decode_s == pytest.approx(10.0) and p.theta == pytest.approx(1 / 3000)
        q = GenAdmissionWaitingQueue(AUTO)  # no target yet
        q.set_now(100.0)
        p.step(reqs[2:], 100.0, 8, 2, q)
        assert p.theta == pytest.approx(1 / 3000)


def make_slo(**kwargs):
    """target_output_tps = 8 with D = 20 s: laxity victims need 0.4 s, the
    resume margin is 0.2 s, the max pause 7 s."""
    return make_preemption(target_output_tps=8.0, **kwargs)


def slo_reqs(p, specs, seconds=10.0):
    """Requests ``(max_new_tokens, tokens/s)`` from t=0; at 10 tokens/s for
    10 s the laxity is 0.025 x (max_new - 1) s, at 5 tokens/s it is negative."""
    reqs = [make_req(i, generated=1, max_new_tokens=m) for i, (m, _) in enumerate(specs)]
    now = run_for(p, reqs, seconds=seconds, rates=[r for _, r in specs])
    p.observe(reqs, now)
    return reqs, now


def ids(reqs):
    return [r.py_request_id for r in reqs]


class TestOutputTpsSlo:
    def test_laxity(self):
        p = make_slo()
        (a, b), now = slo_reqs(p, [(201, 10.0), (201, 5.0)])
        # Deadline 200 / 8 = 25 s; finish 10 + 100 / 10 = 20 s.
        assert p.laxity(a) == pytest.approx(5.0) and p.laxity(b) == pytest.approx(-15.0)
        p._pause(a)
        p.observe([a, b], now + 2.0)
        assert p.laxity(a) == pytest.approx(3.0)  # a pause costs its length
        p._resume(1)
        step([a], iters=10)
        p.observe([a, b], now + 3.0)
        assert p.laxity(a) == pytest.approx(3.0)  # the rate excludes pauses
        q = make_slo(unpaused_tps=None)
        q.observe([c := make_req(1, generated=1, max_new_tokens=201)], 0.0)
        assert q.laxity(c) is None  # no rate yet
        q.mean_tpot_s = 0.1
        assert q.laxity(c) == pytest.approx(5.0)

    def test_victim_tiers(self):
        p = make_slo(max_per_pass=4)
        reqs, _ = slo_reqs(p, [(201, 10.0), (401, 10.0), (201, 8.1), (201, 5.0)])
        # Laxity 5 s, 10 s, 0.31 s (under the 0.4 s floor), below the target.
        assert ids(p._victims(reqs, 4)) == [1, 0, 3]  # no credit: no sacrifice
        p.credit = 1.5
        assert ids(p._victims(reqs, 4)) == [1, 0, 3, 2]
        assert p._reqs[2].sacrificed and p.credit == pytest.approx(0.5)
        # Once sacrificed it is a below-target victim, at no further credit.
        assert ids(p._victims(reqs, 4)) == [1, 0, 3, 2] and p.credit == pytest.approx(0.5)

    def test_pause_limits_per_tier(self):
        p = make_slo()
        reqs, _ = slo_reqs(p, [(401, 10.0), (201, 5.0)])
        p.credit = 10.0
        p._reqs[1].pauses = 3  # below target: 3 pauses max
        assert ids(p._victims(reqs, 1)) == [0]
        p._reqs[0].pauses = 8  # laxity victim (or sacrifice): 8 max
        assert p._victims(reqs, 1) == [] and p.credit == 10.0

    def test_laxity_margin_resume_and_least_laxity_lane(self):
        p = make_slo(max_paused=4)
        reqs, _ = slo_reqs(p, [(401, 10.0), (201, 10.0), (161, 10.0), (201, 5.0)])
        for r in reqs:
            p._pause(r)
        # Due at: 0 -> 17 (max pause), 1 -> 14.8, 2 -> 13.8, 3 (below) -> 17.
        assert [p._resume_at(p._reqs[i]) for i in range(4)] == pytest.approx([17, 14.8, 13.8, 17])
        p._now = 13.79
        assert p._resume(0) == 0
        p._now = 13.81
        assert p._resume(1) == 1 and p.paused_ids == {0, 1, 3}  # 2 due (and the slot)
        assert p.laxity(reqs[2]) == pytest.approx(0.19)  # back with the margin left
        p._now = 14.0
        assert p._resume(1) == 1 and p.paused_ids == {0, 3}  # least laxity first
        p._now = 17.0
        assert p._resume(0) == 2 and p.stats["forced"] == 2

    def test_sacrificed_resume_last(self):
        p = make_slo(max_paused=4)
        reqs, now = slo_reqs(p, [(201, 5.0), (2001, 10.0)])
        p._reqs[0].sacrificed = True
        p._pause(reqs[0])  # due at 17 (max pause)
        p.observe(reqs, now + 2.0)
        p._pause(reqs[1])  # due at 19 (max pause; laxity 50 s)
        assert p._resume_at(p._reqs[0]) < p._resume_at(p._reqs[1])
        p.observe(reqs, now + 3.0)
        assert p._resume(1) == 1 and p.paused_ids == {0}  # sacrificed last
        p.observe(reqs, now + 7.0)
        assert p._resume(0) == 1 and p.stats["forced"] == 1  # still forced at the max pause

    def test_credit_bucket(self):
        p = GenPreemption(SLO, max_num_active_requests=128)
        complete(p, 10.0, decode_s=20.0, tokens=1000)  # 50 tokens/s
        assert p.credit == pytest.approx(0.2)
        complete(p, 11.0, decode_s=20.0, tokens=600)  # 30: below
        assert p.credit == pytest.approx(-0.6)
        complete(p, 12.0, decode_s=20.0, tokens=1000, paused_s=6.0)  # 38.5 with pauses: below
        assert p.credit == pytest.approx(-1.4)
        complete(p, 13.0, decode_s=20.0, tokens=1000, paused_s=6.0, sacrificed=True)  # refunded
        assert p.credit == pytest.approx(-1.2)
        complete(p, 14.0, decode_s=20.0, tokens=600, target_ready=False)
        assert p.credit == pytest.approx(-1.2) and p.theta == 0.0
        assert "below_target=3/4 credit=-1.2" in p.stats_line()
        for i in range(4000):
            complete(p, 20.0 + i, decode_s=20.0, tokens=1000)
        assert p.credit == pytest.approx(600.0)  # clamped to 3000 x 0.2
        for i in range(2000):
            complete(p, 5000.0 + i, decode_s=20.0, tokens=100)
        assert p.credit == pytest.approx(-600.0)

    def test_step(self):
        p = make_slo()
        reqs = running(p, 4)
        reqs[1].py_max_new_tokens = 8000  # the most laxity
        assert p.step(reqs, 100.0, 4, 4, at_risk_queue()) == 3 and p.paused_ids == {1}
        assert p.step(reqs[1:], 101.0, 4, 3, slack_queue(now=101.0)) == 3  # 1 back first
        assert not p.paused_ids


@pytest.mark.parametrize("settings", [ON, SLO])
def test_decisions_are_rank_identical(settings):
    """Only rank-identical inputs (token counts, max_tokens, completions, rank
    0's clock) are used: the local clock is never read."""
    results = []
    for _ in range(3):
        p = make_preemption(
            max_per_pass=3,
            hysteresis_iters=0,
            **{k: getattr(settings, k) for k in ("tpot_budget", "target_output_tps")},
        )
        reqs = [make_req(i, generated=1 + i, max_new_tokens=150 + 40 * i) for i in range(8)]
        clock = Mock(side_effect=AssertionError("local clock read"))
        with patch("time.monotonic", clock), patch("time.time", clock):
            p.observe(reqs, 90.0)
            for i, req in enumerate(reqs):
                step([req], iters=40 + 7 * i, tokens_per_iter=1 + i % 3)
            p.credit = 1.0
            p.step(reqs, 100.0, 8, 8, at_risk_queue(n=3))
            p.step(reqs[2:], 103.0, 8, 6, slack_queue(now=103.0))
            complete(p, 104.0, decode_s=10.0, tokens=50)
        assert any(r.pauses for r in p._reqs.values())
        results.append((p.paused_ids, p.credit, p.theta, sorted(map(repr, p._reqs.items()))))
    assert results[0] == results[1] == results[2]


class TestCreate:
    def _executor(self, **overrides):
        seq_slots = Mock()
        seq_slots.get_max_resource_count.return_value = 176
        ex = SimpleNamespace(
            scheduler=Mock(spec=["set_gen_paused_ids"]),
            _gen_admission=AUTO,
            enable_attention_dp=False,
            dist=SimpleNamespace(has_pp=False, rank=0),
            kv_cache_manager=Mock(max_admissible_sequences=352),
            draft_kv_cache_manager=None,
            max_num_active_requests=128,
            resource_manager=SimpleNamespace(
                resource_managers={ResourceManagerType.SEQ_SLOT_MANAGER: seq_slots}
            ),
        )
        ex.kv_cache_manager._max_resident_sequences.return_value = 176
        ex.__dict__.update(overrides)
        return ex

    def test_off_and_enabled(self):
        assert create_gen_preemption(self._executor(), GenPreemptionSettings()) is None
        p = create_gen_preemption(self._executor(), ON)
        assert isinstance(p, GenPreemption) and p.max_paused == 48

    @pytest.mark.parametrize(
        "overrides, needle",
        [
            ({"scheduler": object()}, "V2 scheduler"),
            ({"_gen_admission": GenAdmissionConfig(ttft_percentile=100.0)}, "auto GEN admission"),
            ({"enable_attention_dp": True}, "attention DP"),
            ({"max_num_active_requests": 129}, "sequence slot pool holds 176"),
            ({"draft_kv_cache_manager": Mock(spec=[], max_admissible_sequences=175)}, "175"),
        ],
    )
    def test_unsupported_warns_and_stays_off(self, overrides, needle):
        with patch.object(gen_preemption, "logger") as log:
            assert create_gen_preemption(self._executor(**overrides), SLO) is None
        assert needle in log.warning.call_args.args[0] and not log.info.called

    def test_pipeline_parallelism_raises(self):
        ex = self._executor(dist=SimpleNamespace(has_pp=True, rank=0))
        with pytest.raises(ValueError, match="pipeline parallelism"):
            create_gen_preemption(ex, SLO)


class TestUnsupportedHeadroomReason:
    """The creator-time mirror of create_gen_preemption's checks, deciding
    whether pools reserve the paused cap at all."""

    def _llm_args(self, **overrides):
        args = SimpleNamespace(
            pipeline_parallel_size=1,
            enable_attention_dp=False,
            kv_cache_config=SimpleNamespace(use_kv_cache_manager_v2=True),
            max_beam_width=1,
            sleep_config=None,
        )
        args.__dict__.update(overrides)
        return args

    def _reason(self, **overrides):
        return gen_preemption.unsupported_headroom_reason(
            GenPreemptionConfig(tpot_budget=0.25), self._llm_args(**overrides)
        )

    @pytest.mark.parametrize(
        "overrides, needle",
        [
            ({}, None),
            ({"pipeline_parallel_size": 2}, "pipeline parallelism"),
            ({"enable_attention_dp": True}, "attention DP"),
            (
                {"kv_cache_config": SimpleNamespace(use_kv_cache_manager_v2=True)},
                None,
            ),
            (
                {"kv_cache_config": SimpleNamespace(use_kv_cache_manager_v2=False)},
                "V2 scheduler",
            ),
            # "auto" is only resolved when the model is loaded: keep sizing
            # the pools; the executor decides with the real scheduler.
            (
                {"kv_cache_config": SimpleNamespace(use_kv_cache_manager_v2="auto")},
                None,
            ),
            # Sleep demotes a requested V2 manager to V1.
            ({"sleep_config": SimpleNamespace()}, "V2 scheduler"),
        ],
    )
    def test_reason(self, overrides, needle, monkeypatch):
        monkeypatch.setenv("TLLM_GEN_ADMISSION_TTFT_PERCENTILE", "90")
        reason = self._reason(**overrides)
        if needle is None:
            assert reason is None
        else:
            assert reason is not None and needle in reason

    def test_missing_auto_admission_policy_is_a_reason(self, monkeypatch):
        from tensorrt_llm._torch.pyexecutor.scheduler.gen_admission_queue import GEN_ADMISSION_ENV

        monkeypatch.delenv(GEN_ADMISSION_ENV, raising=False)
        assert self._reason() == (
            "needs the auto GEN admission policy (0 < TLLM_GEN_ADMISSION_TTFT_PERCENTILE < 99)"
        )

    def test_off_feature_never_gives_a_reason(self, monkeypatch):
        monkeypatch.delenv("TLLM_GEN_ADMISSION_TTFT_PERCENTILE", raising=False)
        assert (
            gen_preemption.unsupported_headroom_reason(
                None, self._llm_args(pipeline_parallel_size=2)
            )
            is None
        )
        assert (
            gen_preemption.unsupported_headroom_reason(
                GenPreemptionConfig(),
                self._llm_args(pipeline_parallel_size=2, enable_attention_dp=True),
            )
            is None
        )


def _drafter(kind, num_slots=4):
    """A DFlash or DSpark drafter worker and its metadata with ``num_slots``
    request slots plus a scratch slot (CPU stand-ins for device buffers)."""
    import collections
    from functools import partial

    import torch

    from tensorrt_llm._device_copy import DeviceCopyBuffer
    from tensorrt_llm._torch.speculative import dflash, dspark

    rows = num_slots + 1
    if kind == "dflash":
        worker = object.__new__(dflash.DFlashWorker)
        worker._ctx_buf_inited, worker._req_ctx_pos, worker._dummy_slot = True, {}, num_slots
        worker._ctx_len_host = [0] * rows
        worker._batch_to_slot_buffer = DeviceCopyBuffer(torch.full((num_slots,), num_slots))

        worker._ctx_base_host = [0] * rows
        worker._ctx_base_pages = torch.zeros(rows, dtype=torch.long)
        worker._ctx_block_tables = None
        worker._write_ctx_len = partial(dflash.DFlashWorker._write_ctx_len, worker)
        worker._assign_slot = partial(dflash.DFlashWorker._assign_slot, worker)
        worker._seed_generation_slot = partial(dflash.DFlashWorker._seed_generation_slot, worker)
        meta = object.__new__(dflash.DFlashSpecMetadata)
        meta._batch_indices_buffer = DeviceCopyBuffer(torch.zeros(num_slots, dtype=torch.int))
        meta.batch_indices_cuda = meta._batch_indices_buffer.tensor
        meta._dflash_worker = worker
        meta.host_position_ids = torch.zeros(num_slots, dtype=torch.long)
        meta.seq_lens = [1] * num_slots
    else:
        worker = object.__new__(dspark.DSv4DSparkWorker)
        worker._win_inited, worker._scratch_slot, worker._confidence_logits = True, num_slots, None
        worker._valid_len = torch.zeros(rows, dtype=torch.long)
        worker._position_initialized = torch.zeros(rows, dtype=torch.bool)
        worker._kv_windows = torch.zeros(rows, 1, 2, 2)
        worker._batch_to_slot = torch.full((num_slots,), num_slots, dtype=torch.long)
        meta = object.__new__(dspark.DSparkSpecMetadata)
        meta.batch_indices_cuda = torch.zeros(num_slots, dtype=torch.int)
        meta._dspark_worker = worker
    worker._req_to_slot, worker._free_slots = {}, collections.deque(range(num_slots))
    worker._graph_dummy_id_floor = 1 << 40
    worker._ctx_len = torch.zeros(rows, dtype=torch.long)
    return worker, meta


@pytest.mark.parametrize("kind", ["dflash", "dspark"])
def test_drafter_state_is_kept_for_paused_requests(kind):
    worker, meta = _drafter(kind)

    def prepare(request_ids, paused=()):
        meta.request_ids, meta.num_generations = list(request_ids), len(request_ids)
        meta.paused_in_place_ids = frozenset(paused)
        with patch(f"tensorrt_llm._torch.speculative.{kind}.prefer_pinned", return_value=False):
            meta.prepare()

    prepare([11, 12])
    slot = worker._req_to_slot[12]
    worker._ctx_len[slot] = 77
    prepare([11, 13, 14], paused={12})  # paused: slot and state kept, 4 slots used
    assert worker._req_to_slot[12] == slot and int(worker._ctx_len[slot]) == 77
    assert len(set(worker._req_to_slot.values())) == 4
    prepare([11, 12, 13])  # resumed into its slot; 14 finished and freed its slot
    assert worker._req_to_slot[12] == slot and int(worker._ctx_len[slot]) == 77
    assert 14 not in worker._req_to_slot and len(worker._free_slots) == 1


@pytest.mark.parametrize("kind", ["dflash", "dspark"])
@pytest.mark.parametrize("target_output_tps", [None, 8.0])
def test_executor_forced_resume_preserves_displaced_request(kind, target_output_tps):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import SchedulerOutput

    p = make_preemption(target_output_tps=target_output_tps)
    reqs = running(p, 2)
    for i, req in enumerate(reqs):
        req.py_batch_idx = i
    p._pause(reqs[0])
    reqs[0].py_batch_idx = None  # cleared by the scheduler while paused
    now = 108.0  # past the 7 s maximum pause, with the only batch slot full
    scheduler = Mock()
    scheduler.schedule_request.return_value = SchedulerOutput([], [], reqs[:1], [], [], 1)
    ex = SimpleNamespace(
        _gen_preemption=p,
        _get_request_admission_capacity=lambda: 1,
        request_broadcaster=SimpleNamespace(rank0_now=now),
        scheduler=scheduler,
        active_requests=reqs,
        inflight_req_ids=set(),
        enable_attention_dp=False,
        enable_iter_perf_stats=False,
        enable_batch_waiting=False,
        is_encoder_decoder=False,
        adp_router=SimpleNamespace(exclude_retiring_requests=False),
        dist=SimpleNamespace(rank=1, cp_config={}, cp_rank=0, cp_size=1),
        num_fetch_requests=0,
        kv_cache_transceiver=None,
        kv_cache_manager=Mock(spec=[]),
        _kv_load_tracker=Mock(begin_step=Mock(return_value=None)),
        _kv_cache_capacity={},
        _fetch_and_enqueue_requests=Mock(),
        _should_exclude_last_generation_logits=lambda: False,
        _pop_from_waiting_queue=Mock(return_value=([], {})),
        _maybe_record_hang_diagnostic_phase=Mock(),
        _cap_context_by_total_kv_len=lambda requests: requests,
    )
    queue = slack_queue(now=now)
    with patch("tensorrt_llm._torch.pyexecutor.py_executor.merge_requests", return_value=[]):
        assert PyExecutor._fetch_new_requests(ex, queue, reqs) == []
    assert not p.paused_ids and p.stats["forced"] == 1
    scheduler.set_gen_paused_ids.assert_called_once_with(frozenset())
    # step returns occupancy, so a forced resume blocks further admissions.
    ex._pop_from_waiting_queue.assert_called_once_with(queue, 2, None)

    worker, meta = _drafter(kind)
    meta.request_ids, meta.num_generations = [0, 1], 2
    with patch(f"tensorrt_llm._torch.speculative.{kind}.prefer_pinned", return_value=False):
        meta.prepare()
    old_slots = dict(worker._req_to_slot)
    worker._ctx_len[old_slots[1]] = 77

    scheduled, _, num_fitting = PyExecutor._schedule(ex)
    assert scheduled.generation_requests == reqs[:1] and num_fitting == 1
    assert scheduled.paused_in_place_ids == {1}
    assert reqs[1].py_batch_idx is None
    assert p.paused_ids == {1}
    assert p._reqs[1].paused_t == now
    assert p._reqs[1].pauses == 1
    meta.request_ids, meta.num_generations = [0], 1
    meta.paused_in_place_ids = scheduled.paused_in_place_ids
    with patch(f"tensorrt_llm._torch.speculative.{kind}.prefer_pinned", return_value=False):
        meta.prepare()
    assert worker._req_to_slot == old_slots
    assert int(worker._ctx_len[old_slots[1]]) == 77

    p.observe(reqs, 110.0)
    PyExecutor._schedule(ex)
    assert p._reqs[1].paused_t == now
    assert p._reqs[1].pauses == 1

    # The displaced request's maximum pause forces it ahead of the live
    # request, even though it appears last in the active list.
    ex.request_broadcaster.rank0_now = 116.0
    queue.set_now(116.0)
    with patch("tensorrt_llm._torch.pyexecutor.py_executor.merge_requests", return_value=[]):
        assert PyExecutor._fetch_new_requests(ex, queue, reqs) == []
    assert p.stats["forced"] == 2
    assert p._reqs[1].paused_t is None
    assert p._reqs[1].paused_total == 8.0
    assert p._reqs[1].resumed_at_iter == reqs[1].py_decoding_iter
    scheduler.schedule_request.return_value = SchedulerOutput([], [], reqs[1:], [], [], 1)
    resumed, _, _ = PyExecutor._schedule(ex)
    scheduler.schedule_request.assert_called_with([reqs[1], reqs[0]], set())
    assert resumed.paused_in_place_ids == {0}
    assert p.paused_ids == {0}
    meta.request_ids, meta.num_generations = [1], 1
    meta.paused_in_place_ids = resumed.paused_in_place_ids
    with patch(f"tensorrt_llm._torch.speculative.{kind}.prefer_pinned", return_value=False):
        meta.prepare()
    assert worker._req_to_slot == old_slots
    assert int(worker._ctx_len[old_slots[1]]) == 77

    # Completion charges the displacement to TPOT inflation rather than
    # increasing the controller's estimate of unpaused decode time.
    reqs[1].generated += 20
    p.observe(reqs, 117.0)
    p.observe(reqs[:1], 118.0, target_ready=True)
    assert p.mean_decode_s == pytest.approx(20.0)
    assert p.stats["scored"] == 1
    assert p.stats["infl"] > 0.0
