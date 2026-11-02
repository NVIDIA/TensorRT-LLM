# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for GEN slot preemption (scheduler_config.gen_preemption), pause in
place.

Covers:
- settings from the LLM API config, the startup log, and the resident
  headroom the sequence-slot, KV index and recurrent-state pools reserve
- victims: the longest resident request (ties: newest), and none while the
  TPOT budget max(0, R0 / target_output_tps - 1) is 0
- guardrails: first token out, min run time, per-request cap, hysteresis,
  near-finished requests, per-pass and paused-count budgets
- the window controller: warm start, one step per completion, convergence to
  0.8 x the budget, clamps, no update before the admission target or without
  a budget
- the at-risk test on the GEN admission queue
- the executor pass: resume lane ahead of new admissions, forced resume, no
  starvation, identical decisions on simulated ranks
- executor setup: pool checks, prerequisites, pipeline-parallel rejection
"""

import math
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor._util import compute_max_num_sequences
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import gen_preemption
from tensorrt_llm._torch.pyexecutor.scheduler.gen_admission_queue import (
    GenAdmissionConfig,
    GenAdmissionWaitingQueue,
)
from tensorrt_llm._torch.pyexecutor.scheduler.gen_preemption import (
    GenPreemption,
    GenPreemptionSettings,
    _ReqTrack,
    log_gen_preemption_settings,
    resident_headroom,
)
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.llmapi.llm_args import GenPreemptionConfig

pytestmark = pytest.mark.cpu_only

GEN = LlmRequestState.GENERATION_IN_PROGRESS
LOGGER = "tensorrt_llm._torch.pyexecutor.scheduler.gen_preemption.logger"
# Preemption on at 40 output tokens/s per request.
ON = GenPreemptionSettings(target_output_tps=40.0)


@pytest.fixture
def active():
    """Install settings for pool sizing; restore the previous ones after."""
    saved = gen_preemption.get_active_settings()
    yield gen_preemption.set_active_settings
    gen_preemption.set_active_settings(saved)


def make_req(
    req_id,
    decoding_iter=100,
    generated=100,
    prompt_len=1000,
    max_new_tokens=4096,
    state=GEN,
):
    req = SimpleNamespace(
        py_request_id=req_id,
        request_id=req_id,
        state=state,
        py_decoding_iter=decoding_iter,
        prompt_len=prompt_len,
        total_input_len_cp=prompt_len,
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


def make_preemption(max_paused=4, decode_s=20.0, target=1.0, unpaused_tps=10.0, **kwargs):
    """A preemption with a ``max_paused`` cap, a measured mean decode time of
    ``decode_s`` (window 0.15 x decode_s = 3 s at g = 1, max pause 7 s), a
    measured unpaused rate well above the target (TPOT budget 900%) and no
    minimum run unless asked."""
    kwargs.setdefault("min_run_frac", 0.0)
    kwargs.setdefault("warmup_min_run_s", 0.0)
    kwargs.setdefault("min_remaining_tokens", 0)
    settings = GenPreemptionSettings(target_output_tps=target, cap_frac=1.0, **kwargs)
    p = GenPreemption(settings, max_num_active_requests=max_paused)
    p.mean_decode_s = decode_s
    p.mean_tpot_s = None if unpaused_tps is None else 1.0 / unpaused_tps
    return p


def run_for(p, reqs, t0=0.0, seconds=10.0, rates=None):
    """First see ``reqs`` generating at ``t0``, then let each generate at its
    rate (default 10 tokens/s) for ``seconds``; returns the time then."""
    p.observe(reqs, t0)
    for i, req in enumerate(reqs):
        rate = 10.0 if rates is None else rates[i]
        req.generated += int(round(rate * seconds))
        req.py_decoding_iter += int(round(rate * seconds))
    return t0 + seconds


def victim_ids(p, reqs, now):
    p.observe(reqs, now)
    return [r.py_request_id for r in p.victim_order(reqs)]


def complete(p, now, decode_s, tokens, paused_s=0.0, target_ready=True):
    """Account one completed request directly."""
    p._now = now
    track = _ReqTrack(
        order=0,
        first_seen=now - decode_s - paused_s,
        first_token_t=now - decode_s - paused_s,
        first_token_gen=0,
        paused_total=paused_s,
        last_gen=tokens,
    )
    p._on_left(track, target_ready)


# ---------------------------------------------------------------------------
# Settings and pool sizing
# ---------------------------------------------------------------------------
class TestSettings:
    def test_default_is_off(self):
        for settings in (
            GenPreemptionSettings(),
            GenPreemptionSettings.from_config(None),
            GenPreemptionSettings.from_config(GenPreemptionConfig()),
        ):
            assert not settings.enabled
            assert resident_headroom(128, settings) == 0

    def test_from_llm_api_config(self):
        settings = GenPreemptionSettings.from_config(GenPreemptionConfig(target_output_tps=40))
        assert settings.enabled and settings.target_output_tps == 40.0
        # TPOT budget max(0, R0 / T - 1).
        assert settings.budget(50.0) == pytest.approx(0.25)
        assert settings.budget(40.0) == 0.0 and settings.budget(30.0) == 0.0
        assert settings.budget(None) == 0.0
        assert GenPreemptionSettings().budget(50.0) == 0.0
        # Paused cap 0.375 x max_num_active_requests, at least 1.
        assert settings.paused_cap(128) == 48 and settings.paused_cap(64) == 24
        assert settings.paused_cap(1) == 1
        assert resident_headroom(128, settings) == 48

    def test_internal_defaults(self):
        s = GenPreemptionSettings()
        assert (s.slack_decode_frac, s.gain_min, s.gain_max) == (0.15, 0.25, 2.0)
        assert (s.time_constant, s.ewma_n, s.target_frac) == (3000.0, 500.0, 0.8)
        assert (s.cap_frac, s.max_pause_frac, s.min_run_frac) == (0.375, 0.35, 0.07)
        assert s.late_trigger and (s.max_per_request, s.max_per_pass) == (2, 1)

    def test_tpot_budget_from_llm_api_config(self):
        settings = GenPreemptionSettings.from_config(GenPreemptionConfig(tpot_budget=0.1))
        assert settings.enabled and settings.tpot_budget == 0.1
        assert settings.target_output_tps is None
        # A fixed budget, whatever the measured rate.
        assert settings.budget(None) == settings.budget(5.0) == 0.1
        assert resident_headroom(128, settings) == 48

    @pytest.mark.parametrize("target", [0.0, -1.0, float("nan")])
    def test_invalid_target_raises(self, target):
        with pytest.raises(ValueError, match="target_output_tps"):
            GenPreemptionSettings(target_output_tps=target)

    @pytest.mark.parametrize("budget", [0.0, 1.0, -0.1, float("nan")])
    def test_invalid_budget_raises(self, budget):
        with pytest.raises(ValueError, match="tpot_budget"):
            GenPreemptionSettings(tpot_budget=budget)

    def test_budget_and_target_are_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            GenPreemptionSettings(tpot_budget=0.1, target_output_tps=40.0)

    def test_startup_log(self):
        with patch(LOGGER) as log:
            log_gen_preemption_settings(GenPreemptionSettings())
        log.info.assert_not_called() and log.warning.assert_not_called()
        with patch(LOGGER) as log:
            log_gen_preemption_settings(ON)
        msg = log.info.call_args.args[0]
        assert "enabled" in msg and "target_output_tps=40.0" in msg
        with patch(LOGGER) as log:
            log_gen_preemption_settings(ON, "needs the auto GEN admission policy")
        assert "disabled" in log.warning.call_args.args[0]

    def test_seq_slot_pool_headroom(self, active):
        mapping = Mock()
        mapping.has_pp.return_value = False
        active(GenPreemptionSettings())
        assert compute_max_num_sequences(mapping, 128, False) == 128
        active(ON)
        assert compute_max_num_sequences(mapping, 128, False) == 176
        assert compute_max_num_sequences(mapping, 128, False, enable_overlap_headroom=True) == 304
        mapping.has_pp.return_value = True
        mapping.pp_size = 2
        assert compute_max_num_sequences(mapping, 128, False) == 256

    def test_recurrent_state_pool_headroom(self, active):
        mgr = object.__new__(MambaHybridCacheManagerV2)
        mgr.max_batch_size = 128
        mgr.mapping = Mock(pp_size=1)
        mgr.mapping.has_pp.return_value = False
        active(GenPreemptionSettings())
        assert mgr._max_resident_sequences() == 128
        active(ON)
        assert mgr._max_resident_sequences() == 176
        mgr.mapping = Mock(pp_size=2)
        mgr.mapping.has_pp.return_value = True
        assert mgr._max_resident_sequences() == 256


# ---------------------------------------------------------------------------
# Victims: the longest resident request, only with a TPOT budget
# ---------------------------------------------------------------------------
class TestVictims:
    def test_longest_resident_first_ties_newest(self):
        p = make_preemption()
        old = [make_req(1), make_req(2)]
        p.observe(old, 0.0)
        new = make_req(3)
        p.observe(old + [new], 5.0)
        # 1 and 2 arrived together (tie: newest first), 3 is the most recent.
        assert victim_ids(p, old + [new], now=6.0) == [2, 1, 3]

    def test_victim_key_is_the_wait_order(self):
        p = make_preemption()
        reqs = [make_req(1), make_req(2)]
        p.observe(reqs[:1], 0.0)
        p.observe(reqs, 4.0)
        assert p.victim_key(reqs[0]) == (-4.0, -0)
        assert p.victim_key(reqs[1]) == (0.0, -1)

    @pytest.mark.parametrize("unpaused_tps", [None, 0.5, 1.0])
    def test_no_budget_no_victim(self, unpaused_tps):
        # R0 not measured yet, or at or below the target: nothing to spend.
        p = make_preemption(target=1.0, unpaused_tps=unpaused_tps)
        reqs = [make_req(i) for i in range(3)]
        p.observe(reqs, 0.0)
        assert p.tpot_budget == 0.0
        assert victim_ids(p, reqs, 1.0) == [2, 1, 0]  # eligible ...
        assert p.select_victims(reqs, num_at_risk=3) == []  # ... but not paused

    def test_budget_follows_the_measured_rate(self):
        p = make_preemption(target=40.0, unpaused_tps=50.0)
        assert p.unpaused_tps == pytest.approx(50.0)
        assert p.tpot_budget == pytest.approx(0.25)
        reqs = [make_req(i) for i in range(3)]
        p.observe(reqs, 0.0)
        assert [r.py_request_id for r in p.select_victims(reqs, 1)] == [2]

    def test_needs_a_first_token_time(self):
        p = make_preemption()
        req = make_req(1)
        p.observe([req], now=None)
        assert p.victim_order([req]) == []


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
class TestGuardrails:
    def test_min_run_since_first_token(self):
        # 0.07 x 100 s = 7 s.
        p = make_preemption(decode_s=100.0, min_run_frac=0.07)
        a = make_req(1)
        p.observe([a], 0.0)  # first seen generating at t=0
        b = make_req(2)
        p.observe([a, b], 1.0)
        step([a, b], iters=100)
        assert victim_ids(p, [a, b], now=6.9) == []
        assert victim_ids(p, [a, b], now=7.5) == [1]
        assert sorted(victim_ids(p, [a, b], now=8.5)) == [1, 2]

    def test_first_token_must_be_out(self):
        p = make_preemption()
        reqs = [make_req(1, decoding_iter=0), make_req(2, decoding_iter=1)]
        now = run_for(p, reqs)
        reqs[0].py_decoding_iter = 0
        assert victim_ids(p, reqs, now) == [2]

    def test_only_generating_requests(self):
        p = make_preemption()
        reqs = [make_req(i) for i in range(1, 5)]
        now = run_for(p, reqs)
        reqs[0].state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        reqs[1].state = LlmRequestState.GENERATION_TO_COMPLETE
        reqs[3].is_dummy_request = True
        assert victim_ids(p, reqs, now) == [3]

    def test_near_finished_requests_are_kept(self):
        p = make_preemption(min_remaining_tokens=32)
        reqs = [
            make_req(1, generated=0, max_new_tokens=132),
            make_req(2, generated=0, max_new_tokens=133),
        ]
        now = run_for(p, reqs)  # 100 tokens each
        assert victim_ids(p, reqs, now) == [2]

    def test_two_pauses_per_request(self):
        p = make_preemption(hysteresis_iters=0)
        req = make_req(1)
        now = run_for(p, [req])
        for n in (1, 2):
            assert victim_ids(p, [req], now) == [1]
            p.on_paused(req)
            assert p.victim_order([req]) == []  # paused
            p.on_resumed(req)
            assert p.num_pauses(1) == n
        assert victim_ids(p, [req], now) == []  # a third pause is refused
        assert p.totals["paused"] == p.totals["resumed"] == 2

    def test_hysteresis_after_resume(self):
        p = make_preemption(hysteresis_iters=10)
        req = make_req(1)
        now = run_for(p, [req])
        p.observe([req], now)
        p.on_paused(req)
        p.on_resumed(req)
        step([req], iters=9)
        assert victim_ids(p, [req], now) == []
        step([req], iters=1)
        assert victim_ids(p, [req], now) == [1]

    def test_per_pass_and_paused_budgets(self):
        reqs = [make_req(i) for i in range(6)]
        p = make_preemption(max_per_pass=2, max_paused=3)
        now = run_for(p, reqs)
        p.observe(reqs, now)
        assert len(p.select_victims(reqs, num_at_risk=5)) == 2
        assert len(p.select_victims(reqs, num_at_risk=1)) == 1
        assert p.select_victims(reqs, num_at_risk=0) == []
        for v in p.select_victims(reqs, num_at_risk=5):
            p.on_paused(v)
        assert len(p.select_victims(reqs, 5)) == 1
        p.on_paused(p.select_victims(reqs, 5)[0])
        assert p.select_victims(reqs, 5) == []  # 3 paused

    def test_requests_that_leave_are_forgotten(self):
        p = make_preemption()
        reqs = [make_req(1), make_req(2)]
        p.observe(reqs, 0.0)
        p.on_paused(reqs[1])
        p.observe(reqs[:1], 1.0)  # 2 was cancelled while paused
        assert p.num_paused == 0 and p.window["dropped"] == 1

    def test_stats_line(self):
        p = make_preemption()
        reqs = [make_req(1), make_req(2)]
        p.observe(reqs, 0.0)
        p.on_paused(reqs[1])
        p.observe(reqs, 3.0)
        p.on_resumed(reqs[1])
        line = p.stats_line()
        assert line.startswith("[gen_preempt] paused=1 resumed=1")
        assert "paused_now=0/4" in line and "mean_pause=3.00s" in line
        assert "g=1.000 slack=3.00s decode=20.00s unpaused_tps=10.00 budget=900.00%" in line
        assert "tpot_infl=na" in line
        assert "max_pause=7.00s min_run=0.00s" in line
        assert "paused=0 resumed=0" in p.stats_line()  # the window restarts


# ---------------------------------------------------------------------------
# Window controller
# ---------------------------------------------------------------------------
class TestController:
    def test_warm_start(self):
        p = GenPreemption(ON, max_num_active_requests=128)
        assert p.max_paused == 48
        assert (p.gain, p.slack_s, p.max_pause_s, p.min_run_s) == (1.0, 0.0, 10.0, 2.0)
        assert p.unpaused_tps is None and p.tpot_budget == 0.0
        # The first completion seeds D and R0: 30 s for 1500 tokens, 50 tokens/s
        # against the target 40 is a 25% TPOT budget.
        complete(p, 100.0, decode_s=30.0, tokens=1500)
        assert p.mean_decode_s == 30.0 and p.unpaused_tps == pytest.approx(50.0)
        assert p.tpot_budget == pytest.approx(0.25)
        assert p.slack_s == pytest.approx(0.15 * 30.0 * p.gain)
        assert (p.max_pause_s, p.min_run_s) == pytest.approx((10.5, 2.1))
        # Running means over 500 completions.
        complete(p, 101.0, decode_s=80.0, tokens=1500)
        assert p.mean_decode_s == pytest.approx(30.0 + 50.0 / 500)

    def test_one_step_per_completion(self):
        p = GenPreemption(ON, max_num_active_requests=128)
        p.mean_tpot_s = 0.02  # R0 50 tokens/s: budget 25%, setpoint 20%
        # Unpaused: inflation 0, theta += (0.2 - 0) / (0.2 x 3000).
        complete(p, 10.0, decode_s=20.0, tokens=1000)
        assert p.theta == pytest.approx(1.0 / 3000.0)
        # 5 s paused over 1000 tokens: 5 ms per token over a 20 ms TPOT = 25%.
        theta = p.theta
        complete(p, 11.0, decode_s=20.0, tokens=1000, paused_s=5.0)
        assert p.theta == pytest.approx(theta + (0.2 - 0.25) / (0.2 * 3000.0))
        # D and the TPOT exclude the pause.
        assert p.mean_decode_s == pytest.approx(20.0)
        assert p.mean_tpot_s == pytest.approx(0.02)

    def test_fixed_budget(self):
        p = GenPreemption(GenPreemptionSettings(tpot_budget=0.1), max_num_active_requests=4)
        # Known from the start: late waiters can take a slot before any
        # request completes (window 0 until then).
        assert p.tpot_budget == 0.1 and p.slack_s == 0.0
        reqs = [make_req(i) for i in range(2)]
        p.observe(reqs, 0.0)
        p.observe(reqs, 3.0)
        assert [r.py_request_id for r in p.select_victims(reqs, 1)] == [1]
        # Setpoint 0.8 x 10% = 8%: 1.6 s paused over a 20 s, 1000-token
        # decode is exactly on it.
        complete(p, 10.0, decode_s=20.0, tokens=1000)
        theta = p.theta
        assert theta == pytest.approx(1.0 / 3000.0)
        complete(p, 11.0, decode_s=20.0, tokens=1000, paused_s=1.6)
        assert p.theta == pytest.approx(theta)

    def test_no_update_before_the_admission_target(self):
        p = GenPreemption(ON, max_num_active_requests=128)
        for i in range(100):
            complete(p, float(i), decode_s=20.0, tokens=1000, target_ready=False)
        assert p.theta == 0.0 and p.mean_decode_s == pytest.approx(20.0)
        assert p.window["completed"] == 100

    def test_no_update_without_a_budget(self):
        p = GenPreemption(ON, max_num_active_requests=128)
        for i in range(100):  # 30 tokens/s < 40: budget 0
            complete(p, float(i), decode_s=20.0, tokens=600, paused_s=1.0)
        assert p.theta == 0.0 and p.tpot_budget == 0.0

    def test_clamps(self):
        p = GenPreemption(ON, max_num_active_requests=128)
        for i in range(10000):  # never paused: g opens up to its ceiling
            complete(p, float(i), decode_s=20.0, tokens=1000)
        assert p.gain == pytest.approx(2.0)
        for i in range(10000):  # paused far over the budget
            complete(p, float(i), decode_s=20.0, tokens=1000, paused_s=10.0)
        assert p.gain == pytest.approx(0.25)

    @pytest.mark.parametrize("g0", [0.25, 1.0, 2.0])
    def test_converges_to_the_setpoint(self, g0):
        """Toy plant: a completion's pause grows with the window, inflation
        0.25 g.  R0 50 tokens/s against 40 gives budget 25%, setpoint 20%,
        met at g* = 0.8."""
        p = GenPreemption(ON, max_num_active_requests=128)
        p.theta = math.log(g0)
        for i in range(20000):
            paused = 0.25 * p.gain * 20.0  # inflation 0.25 g of a 20 s decode
            complete(p, float(i), decode_s=20.0, tokens=1000, paused_s=paused)
        assert p.gain == pytest.approx(0.8, rel=0.02)

    def test_completions_through_observe(self):
        p = make_preemption(decode_s=None, unpaused_tps=None)
        reqs = [make_req(1, generated=0), make_req(2, generated=0)]
        p.observe(reqs, 0.0)  # first token at t=0
        step(reqs, iters=100)
        p.observe(reqs, 10.0)
        p.on_paused(reqs[1])
        p.observe(reqs, 12.0)
        p.on_resumed(reqs[1])
        step(reqs, iters=100)
        p.observe(reqs, 20.0)
        # Both finish at t=20: 1 decoded 20 s, 2 decoded 18 s + 2 s paused.
        p.observe([], 20.0, target_ready=True)
        assert p.window["completed"] == 2
        assert p.mean_decode_s == pytest.approx(20.0 - 2.0 / 500)
        assert p.mean_tpot_s == pytest.approx(0.1 + (0.09 - 0.1) / 500)
        # 2's inflation: 2 s / 200 tokens over the mean TPOT.
        assert "tpot_infl=+" in p.stats_line()

    def test_rank_identical(self):
        states = set()
        for _ in range(3):
            p = GenPreemption(ON, max_num_active_requests=128)
            for i in range(500):
                complete(
                    p,
                    float(i),
                    decode_s=20.0 + i % 7,
                    tokens=1200 + 13 * (i % 11),
                    paused_s=0.5 * (i % 3),
                )
            states.add((p.theta, p.mean_decode_s, p.mean_tpot_s))
        assert len(states) == 1


# ---------------------------------------------------------------------------
# At-risk test on the admission queue
# ---------------------------------------------------------------------------
AUTO = GenAdmissionConfig(ttft_percentile=90.0)


def make_item(req_id, gen_arrival, orig_arrival):
    item = RequestQueueItem(req_id, Mock())
    item.admit_arrival = gen_arrival
    item.admit_orig_arrival = orig_arrival
    return item


def slack_queue(x=10.0, g=100.0, s2f=0.5, now=100.0, items=()):
    q = GenAdmissionWaitingQueue(AUTO)
    q.adaptive.X, q.adaptive.G = x, g
    q.adaptive.s2f_s = s2f
    q.add_requests(list(items))
    q.set_now(now)
    return q


class TestAtRisk:
    def test_head_within_slack_is_at_risk(self):
        # 1: age 9.2 + s2f 0.5 -> slack 0.3; 2: age 5.0 -> slack 4.5.
        q = slack_queue(items=[make_item(1, 99.0, 90.8), make_item(2, 99.5, 95.0)])
        assert q.num_at_risk(slack_s=5.0) == 2
        assert q.num_at_risk(slack_s=1.0) == 1
        assert q.num_at_risk(slack_s=0.2) == 0

    def test_late_head_is_not_at_risk(self):
        q = slack_queue(items=[make_item(1, 99.0, 80.0)])  # 20.5 > X
        assert q.num_at_risk(slack_s=5.0) == 0

    def test_guard_tier_head_blocks(self):
        # 1 waited 6 s at GEN (guard tier, popped first); 2 is at risk but is
        # not the next pop.
        q = slack_queue(g=5.0, items=[make_item(2, 99.9, 90.8), make_item(1, 94.0, 94.0)])
        assert q.num_at_risk(slack_s=5.0) == 0

    def test_late_head_triggers_with_late_trigger(self):
        # No waiter can make X = 10 any more: 1 has slack -10.5, 2 has -5.5,
        # so the next pop (1) is from the late tier.
        q = slack_queue(items=[make_item(1, 99.0, 80.0), make_item(2, 99.0, 85.0)])
        assert q.num_at_risk(slack_s=5.0) == 0
        assert q.num_at_risk(slack_s=5.0, late_trigger=True) == 2
        # A waiter that can still make X is popped first (slack tier), so a
        # late one triggers only when nobody on time is waiting.
        q = slack_queue(items=[make_item(1, 99.0, 80.0), make_item(3, 99.9, 100.0)])
        assert q.num_at_risk(slack_s=5.0, late_trigger=True) == 0  # 3: slack 9.5

    def test_late_trigger_keeps_the_head_only_and_the_slack_bound(self):
        # The next pop has 9.5 s of slack: not at risk, whatever the flag.
        q = slack_queue(items=[make_item(3, 99.9, 100.0)])
        assert q.num_at_risk(slack_s=5.0, late_trigger=True) == 0
        # A guard-tier next pop triggers with the flag (sim late_ok).
        q = slack_queue(g=5.0, items=[make_item(2, 99.9, 90.8), make_item(1, 94.0, 94.0)])
        assert q.num_at_risk(slack_s=5.0) == 0
        assert q.num_at_risk(slack_s=5.0, late_trigger=True) == 2

    def test_no_target_no_risk(self):
        q = GenAdmissionWaitingQueue(AUTO)  # warming up
        q.add_requests([make_item(1, 99.0, 91.0)])
        q.set_now(100.0)
        assert q.num_at_risk(slack_s=100.0) == 0
        q = GenAdmissionWaitingQueue(GenAdmissionConfig(ttft_percentile=100.0))
        q.add_requests([make_item(1, 99.0, 91.0)])
        q.set_now(100.0)
        assert q.num_at_risk(slack_s=100.0) == 0


# ---------------------------------------------------------------------------
# Executor pass
# ---------------------------------------------------------------------------
def make_executor(preemption, capacity, now=100.0):
    stub = SimpleNamespace(
        _gen_preemption=preemption,
        request_broadcaster=SimpleNamespace(rank0_now=now),
        scheduler=Mock(),
        dist=SimpleNamespace(rank=1),
        _log_gen_preemption=lambda: None,
    )
    stub._get_request_admission_capacity = lambda: capacity
    stub.run = types.MethodType(PyExecutor._run_gen_preemption, stub)
    return stub


def paused_ids(ex):
    return set(ex.scheduler.set_gen_paused_ids.call_args.args[0])


def at_risk_queue(now=100.0, n=1):
    return slack_queue(now=now, items=[make_item(1000 + i, 99.0, 90.8) for i in range(n)])


def running(p, n):
    """``n`` requests decoding at 10 tokens/s since t=90 (the pass runs at
    t=100); all first seen together, so ties go to the newest."""
    reqs = [make_req(i) for i in range(n)]
    run_for(p, reqs, t0=90.0)
    return reqs


class TestExecutorPass:
    def test_pauses_for_an_at_risk_waiter_when_full(self):
        p = make_preemption()
        reqs = running(p, 4)
        ex = make_executor(p, capacity=4)
        assert ex.run(at_risk_queue(), reqs, 4) == 3
        assert paused_ids(ex) == {3}

    def test_no_pause_with_a_free_slot_or_no_risk(self):
        p = make_preemption()
        reqs = running(p, 4)
        ex = make_executor(p, capacity=5)
        assert ex.run(at_risk_queue(), reqs, 4) == 4
        ex = make_executor(p, capacity=4)
        q = slack_queue(items=[make_item(1000, 99.0, 99.0)])  # 8.5 s of slack
        assert ex.run(q, reqs, 4) == 4
        assert paused_ids(ex) == set()

    def test_no_pause_before_the_window_is_measured(self):
        # Warm-up: window 0 until a request completes; an on-time waiter does
        # not trigger, a late one does.
        p = make_preemption(decode_s=None)
        reqs = running(p, 4)
        ex = make_executor(p, capacity=4)
        assert ex.run(at_risk_queue(), reqs, 4) == 4 and paused_ids(ex) == set()
        late = slack_queue(items=[make_item(1000, 99.0, 80.0)])  # slack -10.5
        assert ex.run(late, reqs, 4) == 3 and paused_ids(ex) == {3}

    def test_late_waiter_triggers(self):
        for flag, expect in ((False, set()), (True, {3})):
            p = make_preemption(late_trigger=flag)
            reqs = running(p, 4)
            ex = make_executor(p, capacity=4)
            late = slack_queue(items=[make_item(1000, 99.0, 80.0)])  # slack -10.5
            assert ex.run(late, reqs, 4) == 4 - len(expect)
            assert paused_ids(ex) == expect, flag

    def test_a_request_is_paused_twice_then_refused(self):
        # One waiter at risk on every pass; the batch is full (4 resident) and
        # one request finishes between passes.
        p = make_preemption(max_paused=1, hysteresis_iters=10)
        reqs = running(p, 4)
        ex = make_executor(p, capacity=4)
        assert ex.run(at_risk_queue(), reqs, 4) == 3 and paused_ids(ex) == {3}
        # 0 finishes: 3 resumes first, then 2 is paused (3 is in hysteresis).
        reqs.pop(0)
        assert ex.run(at_risk_queue(), reqs, 4) == 3 and paused_ids(ex) == {2}
        # 10 decode steps later 1 finishes: 2 resumes, 3 is paused again.
        step(reqs, iters=10)
        reqs.pop(0)
        assert ex.run(at_risk_queue(), reqs, 4) == 3 and paused_ids(ex) == {3}
        assert p.num_pauses(3) == 2
        # 10 more steps, 2 finishes: 3 resumes and stays, it had its two pauses.
        step(reqs, iters=10)
        reqs.pop(0)
        assert ex.run(at_risk_queue(), reqs, 4) == 4 and paused_ids(ex) == set()
        assert p.num_pauses(3) == 2 and p.window["no_victim"] == 1
        assert p.totals["paused"] == p.totals["resumed"] == 3

    def test_resume_lane_goes_before_new_admissions(self):
        p = make_preemption(max_paused=2, max_per_request=1)
        reqs = running(p, 4)
        ex = make_executor(p, capacity=4)
        assert ex.run(at_risk_queue(), reqs, 4) == 3  # 3 paused, its slot freed
        # The waiter took the slot (4 counted again, 3 paused), then request 0
        # finished: the free slot goes back to 3, not to the queue.
        reqs.pop(0)
        ex.request_broadcaster.rank0_now = 100.1
        assert ex.run(at_risk_queue(100.1), reqs, 4) == 3
        # 3 resumed; still full and a waiter still at risk: pause another
        # (not 3, which had its one pause).
        assert not p.is_paused(3) and paused_ids(ex) == {2}

    def test_forced_resume_after_max_pause(self):
        p = make_preemption(max_paused=2)  # max pause 0.35 x 20 s = 7 s
        reqs = running(p, 4)
        ex = make_executor(p, capacity=4)
        ex.run(at_risk_queue(), reqs, 4)
        assert paused_ids(ex) == {3}
        # Still full (4 counted besides 3), no waiter: 3 waits ...
        ex.request_broadcaster.rank0_now = 106.5
        assert ex.run(slack_queue(now=106.5), reqs, 5) == 4
        assert paused_ids(ex) == {3}
        # ... until it has been paused for the max pause, then it is back
        # even though that takes the batch one over until a request finishes.
        ex.request_broadcaster.rank0_now = 107.5
        assert ex.run(slack_queue(now=107.5), reqs, 5) == 5
        assert paused_ids(ex) == set() and p.window["forced_resumes"] == 1

    def test_a_forced_resume_that_overfills_can_pause_for_a_waiter(self):
        p = make_preemption(max_paused=2)
        reqs = running(p, 5)
        ex = make_executor(p, capacity=4)
        p.on_paused(reqs[4])  # paused at t=90
        ex.run(at_risk_queue(), reqs, 5)
        # Forced back in (one over); a waiter at risk pauses the next victim.
        assert paused_ids(ex) == {3} and not p.is_paused(4)

    def test_no_starvation_fifo_resume(self):
        p = make_preemption(max_paused=3, max_per_pass=3)
        reqs = running(p, 6)
        ex = make_executor(p, capacity=6)
        assert ex.run(at_risk_queue(n=3), reqs, 6) == 3
        assert paused_ids(ex) == {3, 4, 5}
        # Slots free up one at a time; pauses come back oldest first.
        order = []
        for _ in range(3):
            before = p.paused_ids
            ex.run(slack_queue(now=100.0), reqs, 5 + p.num_paused)  # one slot free
            order += sorted(before - p.paused_ids)
        assert order == [5, 4, 3] and p.num_paused == 0

    def test_completions_feed_the_controller_once_the_target_is_set(self):
        p = make_preemption(decode_s=None)
        reqs = running(p, 4)
        ex = make_executor(p, capacity=8)
        p.observe(reqs, 100.0)
        # Request 0 finished: accounted at the pass, after X is set.
        ex.run(slack_queue(now=100.0), reqs[1:], 3)
        # Seeds D = 10 s and R0 = 10 tokens/s (budget 900%): the first
        # integral step of an unpaused completion is 1 / 3000.
        assert p.mean_decode_s == pytest.approx(10.0)
        assert p.unpaused_tps == pytest.approx(10.0)
        assert p.theta == pytest.approx(1.0 / 3000.0)
        # Without a target X the decode time is still measured, g is not.
        q = GenAdmissionWaitingQueue(AUTO)
        q.set_now(100.0)
        theta = p.theta
        ex.run(q, reqs[2:], 2)
        assert p.theta == theta and p.window["completed"] == 2

    def test_decisions_are_identical_across_ranks(self):
        """Same inputs (active requests, broadcast queue and clock) on every
        rank give the same pauses and controller state, whatever the local
        wall clock."""
        results = []
        for rank in range(3):
            p = make_preemption(max_per_pass=2)
            reqs = [make_req(i, generated=10 * i, prompt_len=100 + i) for i in range(8)]
            ex = make_executor(p, capacity=8)
            with patch("time.monotonic", return_value=1000.0 * rank):
                p.observe(reqs, 90.0)
                for i, req in enumerate(reqs):
                    step([req], iters=20, tokens_per_iter=1 + i % 3)
                ex.run(at_risk_queue(n=3), reqs[1:], 8)
            results.append((paused_ids(ex), p.theta, p.mean_decode_s))
        assert results[0] == results[1] == results[2]
        assert len(results[0][0]) == 2


class TestExecutorSetup:
    def _stub(self, **overrides):
        seq_slots = Mock()
        seq_slots.get_max_resource_count.return_value = 176
        stub = SimpleNamespace(
            _gen_preemption_settings=ON,
            scheduler=object.__new__(KVCacheV2Scheduler),
            _gen_admission=GenAdmissionConfig(ttft_percentile=90.0),
            enable_attention_dp=False,
            dist=SimpleNamespace(has_pp=False, pp_size=1, rank=0),
            kv_cache_manager=Mock(max_admissible_sequences=352),
            draft_kv_cache_manager=None,
            max_num_active_requests=128,
            resource_manager=SimpleNamespace(
                resource_managers={ResourceManagerType.SEQ_SLOT_MANAGER: seq_slots}
            ),
        )
        stub.kv_cache_manager._max_resident_sequences.return_value = 176
        stub.__dict__.update(overrides)
        stub._gen_preemption_disabled_reason = types.MethodType(
            PyExecutor._gen_preemption_disabled_reason, stub
        )
        return stub

    def _create(self, stub):
        with patch(LOGGER) as log:
            out = PyExecutor._create_gen_preemption(stub)
        calls = log.info.call_args_list + log.warning.call_args_list
        return out, [c.args[0] for c in calls]

    def test_off_creates_nothing(self):
        out, logs = self._create(self._stub(_gen_preemption_settings=GenPreemptionSettings()))
        assert out is None and logs == []

    def test_enabled(self):
        out, logs = self._create(self._stub())
        assert isinstance(out, GenPreemption) and out.max_paused == 48
        assert len(logs) == 1 and "enabled" in logs[0]

    def test_pipeline_parallelism_is_rejected(self):
        stub = self._stub(dist=SimpleNamespace(has_pp=True, pp_size=2, rank=0))
        with pytest.raises(ValueError, match="does not support pipeline parallelism"):
            PyExecutor._create_gen_preemption(stub)
        # Off: nothing to reject.
        stub._gen_preemption_settings = GenPreemptionSettings()
        assert PyExecutor._create_gen_preemption(stub) is None

    def test_draft_pool_is_checked(self):
        draft = Mock(max_admissible_sequences=175)
        del draft._max_resident_sequences
        out, logs = self._create(self._stub(draft_kv_cache_manager=draft))
        assert out is None and "index pool holds 175" in logs[0]

    @pytest.mark.parametrize(
        ("overrides", "needle"),
        [
            ({"_gen_admission": GenAdmissionConfig(ttft_percentile=100.0)}, "auto GEN admission"),
            ({"enable_attention_dp": True}, "attention DP"),
            ({"max_num_active_requests": 129}, "sequence slot pool holds 176"),
        ],
    )
    def test_disabled_with_one_warning(self, overrides, needle):
        stub = self._stub(**overrides)
        with patch(LOGGER) as log:
            out = PyExecutor._create_gen_preemption(stub)
        assert out is None and log.info.call_count == 0
        (msg,) = [c.args[0] for c in log.warning.call_args_list]
        assert "disabled" in msg and needle in msg

    def test_recurrent_state_pool_is_checked(self):
        stub = self._stub()
        stub.kv_cache_manager._max_resident_sequences.return_value = 136
        out, logs = self._create(stub)
        assert out is None and "recurrent state pool holds 136" in logs[0]


# ---------------------------------------------------------------------------
# Drafter state outside the KV cache: DSpark rolling windows
# ---------------------------------------------------------------------------
class TestDSparkWindows:
    def _setup(self, num_slots=4):
        import collections

        import torch

        from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata, DSv4DSparkWorker

        worker = object.__new__(DSv4DSparkWorker)
        worker._win_inited = True
        worker._req_to_slot = {}
        worker._free_slots = collections.deque(range(num_slots))
        rows = num_slots + 1
        worker._ctx_len = torch.zeros(rows, dtype=torch.long)
        worker._valid_len = torch.zeros(rows, dtype=torch.long)
        worker._position_initialized = torch.zeros(rows, dtype=torch.bool)
        worker._kv_windows = torch.zeros(rows, 1, 2, 2)
        worker._confidence_logits = None
        worker._scratch_slot = num_slots
        worker._graph_dummy_id_floor = 1 << 40
        worker._batch_to_slot = torch.full((num_slots,), num_slots, dtype=torch.long)
        meta = object.__new__(DSparkSpecMetadata)
        meta.batch_indices_cuda = torch.zeros(num_slots, dtype=torch.int)
        meta._dspark_worker = worker
        return worker, meta

    def _prepare(self, meta, request_ids, paused=frozenset()):
        meta.request_ids = list(request_ids)
        meta.num_generations = len(request_ids)
        meta.paused_in_place_ids = frozenset(paused)
        with patch("tensorrt_llm._torch.speculative.dspark.prefer_pinned", return_value=False):
            meta.prepare()

    def test_paused_requests_keep_their_window(self):
        worker, meta = self._setup()
        self._prepare(meta, [11, 12])
        slot = worker._req_to_slot[12]
        worker._ctx_len[slot] = 77
        worker._kv_windows[slot].fill_(1.0)
        # 12 paused in place: left out of the batch, window kept.
        self._prepare(meta, [11, 13], paused={12})
        assert worker._req_to_slot[12] == slot and int(worker._ctx_len[slot]) == 77
        assert bool(worker._kv_windows[slot].eq(1.0).all())
        # Resumed: same slot, same window.
        self._prepare(meta, [11, 12, 13])
        assert worker._req_to_slot[12] == slot and int(worker._ctx_len[slot]) == 77
        assert int(worker._batch_to_slot[1]) == slot

    def test_requests_not_in_the_batch_and_not_paused_release_their_window(self):
        worker, meta = self._setup()
        self._prepare(meta, [11, 12])
        slot = worker._req_to_slot[12]
        worker._ctx_len[slot] = 77
        self._prepare(meta, [11])
        assert 12 not in worker._req_to_slot and int(worker._ctx_len[slot]) == 0
        assert slot in worker._free_slots

    def test_slots_cover_paused_requests(self):
        # max_batch scheduled slots plus the paused cap kept ones (here 2 + 2).
        worker, meta = self._setup(num_slots=4)
        self._prepare(meta, [1, 2])
        self._prepare(meta, [3, 4], paused={1, 2})
        assert set(worker._req_to_slot) == {1, 2, 3, 4}
        assert len(set(worker._req_to_slot.values())) == 4


# ---------------------------------------------------------------------------
# Drafter state outside the KV cache: DFlash / standalone DSpark context slots
# ---------------------------------------------------------------------------
class TestDFlashSlots:
    def _setup(self, num_slots=4):
        import collections
        from functools import partial

        import torch

        from tensorrt_llm._device_copy import DeviceCopyBuffer
        from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, DFlashWorker

        worker = object.__new__(DFlashWorker)
        worker._ctx_buf_inited = True
        worker._req_to_slot = {}
        worker._req_ctx_pos = {}
        worker._free_slots = collections.deque(range(num_slots))
        worker._dummy_slot = num_slots
        worker._graph_dummy_id_floor = 1 << 40
        worker._ctx_len = torch.zeros(num_slots + 1, dtype=torch.long)
        worker._ctx_len_host = [0] * (num_slots + 1)
        worker._batch_to_slot_buffer = DeviceCopyBuffer(
            torch.full((num_slots,), num_slots, dtype=torch.long)
        )

        def write_ctx_len(updates):  # CPU stand-in for the device scatter
            for slot, value in updates.items():
                worker._ctx_len_host[slot] = value
                worker._ctx_len[slot] = value

        worker._write_ctx_len = write_ctx_len
        worker._assign_slot = partial(DFlashWorker._assign_slot, worker)
        meta = object.__new__(DFlashSpecMetadata)
        meta._batch_indices_buffer = DeviceCopyBuffer(torch.zeros(num_slots, dtype=torch.int))
        meta.batch_indices_cuda = meta._batch_indices_buffer.tensor
        meta._dflash_worker = worker
        return worker, meta

    def _prepare(self, meta, request_ids, paused=frozenset()):
        meta.request_ids = list(request_ids)
        meta.num_generations = len(request_ids)
        meta.paused_in_place_ids = frozenset(paused)
        with patch("tensorrt_llm._torch.speculative.dflash.prefer_pinned", return_value=False):
            meta.prepare()

    def test_paused_requests_keep_their_slot(self):
        worker, meta = self._setup()
        self._prepare(meta, [11, 12])
        slot = worker._req_to_slot[12]
        worker._ctx_len[slot] = 77
        worker._ctx_len_host[slot] = 77
        worker._req_ctx_pos[12] = 900
        # 12 paused in place: left out of the batch, drafter context kept.
        self._prepare(meta, [11, 13], paused={12})
        assert worker._req_to_slot[12] == slot and worker._req_ctx_pos[12] == 900
        assert int(worker._ctx_len[slot]) == 77 and worker._ctx_len_host[slot] == 77
        assert slot not in worker._free_slots
        # Resumed: same slot, same context length, mapped at its batch row.
        self._prepare(meta, [11, 12, 13])
        assert worker._req_to_slot[12] == slot and int(worker._ctx_len[slot]) == 77
        assert int(worker._batch_to_slot_buffer.tensor[1]) == slot

    def test_finished_requests_release_their_slot(self):
        worker, meta = self._setup()
        self._prepare(meta, [11, 12])
        slot = worker._req_to_slot[12]
        worker._ctx_len[slot] = 77
        worker._req_ctx_pos[12] = 900
        self._prepare(meta, [11], paused={12})
        # Finished (or cancelled) after the pause: neither batched nor paused.
        self._prepare(meta, [11])
        assert 12 not in worker._req_to_slot and 12 not in worker._req_ctx_pos
        assert int(worker._ctx_len[slot]) == 0 and slot in worker._free_slots

    def test_slots_cover_paused_requests(self):
        # max_batch scheduled slots plus the paused cap kept ones (here 2 + 2):
        # nobody falls back to the shared dummy slot.
        worker, meta = self._setup(num_slots=4)
        self._prepare(meta, [1, 2])
        with patch("tensorrt_llm._torch.speculative.dflash.logger") as log:
            self._prepare(meta, [3, 4], paused={1, 2})
        log.warning.assert_not_called()
        assert set(worker._req_to_slot) == {1, 2, 3, 4}
        assert len(set(worker._req_to_slot.values())) == 4

    def _lazy_init(self, max_batch):
        import torch

        from tensorrt_llm._torch.speculative.dflash import DFlashWorker

        class _Stop(Exception):
            pass

        worker = object.__new__(DFlashWorker)
        worker._validate_draft_attention_backend = Mock()
        worker._ctx_buf_inited = False
        worker._ctx_kv_manager = None
        worker.spec_config = Mock(max_draft_len=7, tokens_per_gen_step=8)
        # Stop once the slot bookkeeping is built, before any K/V allocation.
        worker._draft_block_width = Mock(side_effect=_Stop)
        attn_metadata = Mock(spec=["kv_cache_manager"])
        attn_metadata.kv_cache_manager = Mock(max_seq_len=4096)
        draft_model = Mock()
        draft_model.config = Mock(max_position_embeddings=4096)
        real_zeros = torch.zeros

        def cpu_zeros(*args, device=None, **kwargs):
            return real_zeros(*args, **kwargs)

        with patch("tensorrt_llm._torch.speculative.dflash.torch.zeros", side_effect=cpu_zeros):
            with pytest.raises(_Stop):
                worker._lazy_init_ctx_buffers(
                    draft_model, Mock(max_num_requests=max_batch), attn_metadata
                )
        return worker

    def test_slot_pool_reserves_the_resident_headroom(self, active):
        worker = self._lazy_init(128)
        assert worker._dummy_slot == 128 and len(worker._free_slots) == 128
        assert worker._ctx_len.shape[0] == 129
        active(ON)
        worker = self._lazy_init(128)
        # 128 scheduled + 48 paused request slots, then the dummy slot.
        assert worker._dummy_slot == 176 and sorted(worker._free_slots) == list(range(176))
        assert worker._ctx_len.shape[0] == 177 and len(worker._ctx_len_host) == 177
        assert worker._batch_to_slot_buffer.tensor.shape[0] == 128
