# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for GEN slot preemption (TLLM_GEN_PREEMPT_POLICY), pause in place.

Covers:
- GenPreemptionConfig env parsing, the startup log, and the resident headroom
  the sequence-slot, KV index and recurrent-state pools reserve
- victim order of each policy (newest, wait, tps, long)
- guardrails: first token out, min run time, per-request cap, hysteresis,
  near-finished requests, per-pass and paused-count budgets
- the at-risk test on the GEN admission queue
- the executor pass: resume lane ahead of new admissions, forced resume, no
  starvation, identical decisions on simulated ranks, and off == unchanged
"""

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
from tensorrt_llm._torch.pyexecutor.scheduler.gen_admission_queue import (
    GenAdmissionConfig,
    GenAdmissionWaitingQueue,
)
from tensorrt_llm._torch.pyexecutor.scheduler.gen_preemption import (
    POLICY_TPS,
    GenPreemption,
    GenPreemptionConfig,
    log_gen_preemption_config,
    resident_headroom,
)
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

pytestmark = pytest.mark.cpu_only

GEN = LlmRequestState.GENERATION_IN_PROGRESS
LOGGER = "tensorrt_llm._torch.pyexecutor.scheduler.gen_preemption.logger"
ON = {"TLLM_GEN_PREEMPT_POLICY": "wait", "TLLM_GEN_PREEMPT_MAX_PAUSED": "32"}


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


def make_preemption(policy="newest", **kwargs):
    kwargs.setdefault("max_paused", 4)
    kwargs.setdefault("min_run_s", 0.0)
    kwargs.setdefault("min_remaining_tokens", 0)
    return GenPreemption(GenPreemptionConfig(policy=policy, **kwargs))


def victim_ids(preemption, reqs, now=0.0):
    preemption.observe(reqs, now)
    return [r.py_request_id for r in preemption.victim_order(reqs)]


# ---------------------------------------------------------------------------
# Config and pool sizing
# ---------------------------------------------------------------------------
class TestConfig:
    def test_default_is_off(self):
        for env in ({}, {"TLLM_GEN_PREEMPT_POLICY": ""}, {"TLLM_GEN_PREEMPT_POLICY": "off"}):
            cfg = GenPreemptionConfig.from_env(env)
            assert cfg.policy == "off" and not cfg.enabled and cfg.disabled_reason is None
            assert resident_headroom(env) == 0
        # The headroom follows the policy: no pool grows while it is off.
        assert resident_headroom({"TLLM_GEN_PREEMPT_MAX_PAUSED": "32"}) == 0

    def test_needs_max_paused(self):
        cfg = GenPreemptionConfig.from_env({"TLLM_GEN_PREEMPT_POLICY": "wait"})
        assert not cfg.enabled and "TLLM_GEN_PREEMPT_MAX_PAUSED" in cfg.disabled_reason
        assert resident_headroom({"TLLM_GEN_PREEMPT_POLICY": "wait"}) == 0
        assert GenPreemptionConfig.from_env(ON).enabled and resident_headroom(ON) == 32

    def test_env_knobs(self):
        cfg = GenPreemptionConfig.from_env(
            {
                "TLLM_GEN_PREEMPT_POLICY": "Long",
                "TLLM_GEN_PREEMPT_MAX_PAUSED": "16",
                "TLLM_GEN_PREEMPT_SLACK_S": "3.5",
                "TLLM_GEN_PREEMPT_MIN_RUN_S": "1",
                "TLLM_GEN_PREEMPT_MAX_PAUSE_S": "20",
                "TLLM_GEN_PREEMPT_MAX_PER_REQUEST": "2",
                "TLLM_GEN_PREEMPT_MAX_PER_PASS": "3",
                "TLLM_GEN_PREEMPT_HYSTERESIS_ITERS": "10",
                "TLLM_GEN_PREEMPT_MIN_REMAINING_TOKENS": "4",
            }
        )
        assert cfg == GenPreemptionConfig(
            policy="long",
            max_paused=16,
            slack_s=3.5,
            min_run_s=1.0,
            max_pause_s=20.0,
            max_per_request=2,
            max_per_pass=3,
            hysteresis_iters=10,
            min_remaining_tokens=4,
        )
        assert cfg.enabled

    def test_defaults(self):
        cfg = GenPreemptionConfig()
        assert (cfg.policy, cfg.max_paused) == ("off", 0)
        assert (cfg.slack_s, cfg.min_run_s, cfg.max_pause_s) == (5.0, 2.0, 10.0)
        assert (cfg.max_per_request, cfg.max_per_pass) == (1, 1)
        assert cfg.target_tps == 0.0

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            GenPreemptionConfig.from_env({"TLLM_GEN_PREEMPT_POLICY": "lifo"})

    def test_negative_knob_raises(self):
        with pytest.raises(ValueError):
            GenPreemptionConfig(policy="newest", max_paused=-1)
        with pytest.raises(ValueError):
            GenPreemptionConfig(policy="newest", slack_s=float("nan"))

    def test_tps_without_target_is_disabled_with_a_startup_log(self):
        env = {**ON, "TLLM_GEN_PREEMPT_POLICY": "tps"}
        cfg = GenPreemptionConfig.from_env(env)
        assert not cfg.enabled and "TLLM_GEN_PREEMPT_TARGET_TPS" in cfg.disabled_reason
        assert resident_headroom(env) == 0
        with patch(LOGGER) as log:
            log_gen_preemption_config(cfg)
        (msg,) = [c.args[0] for c in log.info.call_args_list]
        assert "disabled" in msg and "TLLM_GEN_PREEMPT_TARGET_TPS" in msg

        env["TLLM_GEN_PREEMPT_TARGET_TPS"] = "20"
        cfg = GenPreemptionConfig.from_env(env)
        assert cfg.enabled and cfg.target_tps == 20.0
        with patch(LOGGER) as log:
            log_gen_preemption_config(cfg)
        assert "enabled" in log.info.call_args.args[0]

    def test_off_logs_nothing(self):
        with patch(LOGGER) as log:
            log_gen_preemption_config(GenPreemptionConfig())
        log.info.assert_not_called()

    def test_seq_slot_pool_headroom(self, monkeypatch):
        mapping = Mock()
        mapping.has_pp.return_value = False
        monkeypatch.delenv("TLLM_GEN_PREEMPT_POLICY", raising=False)
        assert compute_max_num_sequences(mapping, 128, False) == 128
        for k, v in ON.items():
            monkeypatch.setenv(k, v)
        assert compute_max_num_sequences(mapping, 128, False) == 160
        assert compute_max_num_sequences(mapping, 128, False, enable_overlap_headroom=True) == 288
        mapping.has_pp.return_value = True
        mapping.pp_size = 2
        assert compute_max_num_sequences(mapping, 128, False) == 256

    def test_recurrent_state_pool_headroom(self, monkeypatch):
        mgr = object.__new__(MambaHybridCacheManagerV2)
        mgr.max_batch_size = 128
        mgr.mapping = Mock(pp_size=1)
        mgr.mapping.has_pp.return_value = False
        monkeypatch.delenv("TLLM_GEN_PREEMPT_POLICY", raising=False)
        assert mgr._max_resident_sequences() == 128
        for k, v in ON.items():
            monkeypatch.setenv(k, v)
        assert mgr._max_resident_sequences() == 160
        mgr.mapping = Mock(pp_size=2)
        mgr.mapping.has_pp.return_value = True
        assert mgr._max_resident_sequences() == 256


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
class TestPolicies:
    def test_newest_takes_the_latest_admitted(self):
        reqs = [make_req(i) for i in (5, 3, 9)]
        assert victim_ids(make_preemption("newest"), reqs) == [9, 3, 5]

    def test_wait_takes_the_longest_resident(self):
        p = make_preemption("wait")
        old = [make_req(1), make_req(2)]
        p.observe(old, now=0.0)
        new = make_req(3)
        p.observe(old + [new], now=5.0)
        # 1 and 2 arrived together (tie: newest first), 3 is the most recent.
        assert victim_ids(p, old + [new], now=6.0) == [2, 1, 3]

    def test_long_takes_the_most_kv(self):
        reqs = [
            make_req(1, prompt_len=1000, generated=10),
            make_req(2, prompt_len=200, generated=2000),
            make_req(3, prompt_len=5000, generated=0),
        ]
        assert victim_ids(make_preemption("long"), reqs) == [3, 2, 1]

    def test_long_uses_the_global_prompt_under_helix(self):
        a = make_req(1, prompt_len=100)
        a.total_input_len_cp = 1600  # rank-local slice of a 1600-token prompt
        b = make_req(2, prompt_len=1000)
        assert victim_ids(make_preemption("long"), [a, b]) == [1, 2]

    def _tps_setup(self, target, rates, expected_pause_s=0.0):
        """Requests first seen generating at t=0 that then generate
        ``rate`` tokens/s for 10 s."""
        p = make_preemption(POLICY_TPS, target_tps=target, expected_pause_s=expected_pause_s)
        reqs = [make_req(i, generated=1) for i in range(len(rates))]
        p.observe(reqs, now=0.0)
        for req, rate in zip(reqs, rates):
            req.generated += int(rate * 10)
            req.py_decoding_iter += 10
        p.observe(reqs, now=10.0)
        return p, reqs

    def test_tps_below_target_is_never_preempted(self):
        p, reqs = self._tps_setup(target=50.0, rates=[10, 30, 50])
        assert p.victim_order(reqs) == []
        assert p.select_victims(reqs, num_at_risk=3) == []
        assert p.window["no_victim"] == 1

    def test_tps_above_target_largest_margin_first(self):
        p, reqs = self._tps_setup(target=50.0, rates=[40, 120, 60, 90])
        assert [r.py_request_id for r in p.victim_order(reqs)] == [1, 3, 2]
        assert p.achieved_tps(reqs[1]) == pytest.approx(120.0)
        assert p.tps_margin(reqs[1]) == pytest.approx(70.0)
        assert [r.py_request_id for r in p.select_victims(reqs, 1)] == [1]

    def test_tps_must_stay_on_target_over_the_pause(self):
        # 60 tok/s over 10 s; a 5 s pause brings it to 40 tok/s < 50.
        p, reqs = self._tps_setup(target=50.0, rates=[60, 100], expected_pause_s=5.0)
        assert [r.py_request_id for r in p.victim_order(reqs)] == [1]
        assert p.tps_margin(reqs[1]) == pytest.approx(100 * 10 / 15 - 50)

    def test_tps_uses_the_measured_pause(self):
        p, reqs = self._tps_setup(target=50.0, rates=[60, 100])
        assert [r.py_request_id for r in p.victim_order(reqs)] == [1, 0]
        p.mean_pause_s = 5.0
        assert [r.py_request_id for r in p.victim_order(reqs)] == [1]

    def test_tps_needs_a_first_token_time(self):
        p = make_preemption(POLICY_TPS, target_tps=1.0)
        req = make_req(1)
        p.observe([req], now=None)
        assert p.victim_order([req]) == []


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
class TestGuardrails:
    def test_min_run_time_since_first_token(self):
        p = make_preemption(min_run_s=2.0)
        a = make_req(1)
        p.observe([a], 0.0)  # first seen generating at t=0
        b = make_req(2)
        assert victim_ids(p, [a, b], now=1.5) == []
        assert victim_ids(p, [a, b], now=2.0) == [1]
        assert victim_ids(p, [a, b], now=3.5) == [2, 1]

    def test_first_token_must_be_out(self):
        p = make_preemption()
        reqs = [make_req(1, decoding_iter=0), make_req(2, decoding_iter=1)]
        assert victim_ids(p, reqs) == [2]

    def test_only_generating_requests(self):
        p = make_preemption()
        reqs = [
            make_req(1, state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS),
            make_req(2, state=LlmRequestState.GENERATION_TO_COMPLETE),
            make_req(3),
            make_req(4),
        ]
        reqs[-1].is_dummy_request = True
        assert victim_ids(p, reqs) == [3]

    def test_near_finished_requests_are_kept(self):
        p = make_preemption(min_remaining_tokens=32)
        reqs = [
            make_req(1, generated=100, max_new_tokens=132),
            make_req(2, generated=100, max_new_tokens=133),
        ]
        assert victim_ids(p, reqs) == [2]

    def test_one_pause_per_request(self):
        p = make_preemption(hysteresis_iters=0)
        req = make_req(1)
        p.observe([req], 0.0)
        p.on_paused(req)
        assert p.victim_order([req]) == []  # paused
        p.on_resumed(req)
        assert victim_ids(p, [req], now=1.0) == []
        assert p.num_pauses(1) == 1

    def test_hysteresis_after_resume(self):
        p = make_preemption(max_per_request=2, hysteresis_iters=10)
        req = make_req(1)
        p.observe([req], 0.0)
        p.on_paused(req)
        p.on_resumed(req)
        step([req], iters=9)
        assert victim_ids(p, [req]) == []
        step([req], iters=1)
        assert victim_ids(p, [req]) == [1]

    def test_per_pass_and_paused_budgets(self):
        reqs = [make_req(i) for i in range(6)]
        p = make_preemption(max_per_pass=2, max_paused=3)
        p.observe(reqs, 0.0)
        assert len(p.select_victims(reqs, num_at_risk=5)) == 2
        assert len(p.select_victims(reqs, num_at_risk=1)) == 1
        assert p.select_victims(reqs, num_at_risk=0) == []
        for v in p.select_victims(reqs, num_at_risk=5):
            p.on_paused(v)
        assert [r.py_request_id for r in p.select_victims(reqs, 5)] == [3]
        p.on_paused(reqs[3])
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
        assert line.startswith("[gen_preempt] policy=newest paused=1 resumed=1")
        assert "paused_now=0/4" in line and "mean_pause=3.00s" in line
        assert "paused=0 resumed=0" in p.stats_line()  # the window restarts


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


class TestExecutorPass:
    def test_pauses_for_an_at_risk_waiter_when_full(self):
        reqs = [make_req(i) for i in range(4)]
        ex = make_executor(make_preemption("newest"), capacity=4)
        assert ex.run(at_risk_queue(), reqs, 4) == 3
        assert paused_ids(ex) == {3}

    def test_no_pause_with_a_free_slot_or_no_risk(self):
        reqs = [make_req(i) for i in range(4)]
        ex = make_executor(make_preemption(), capacity=5)
        assert ex.run(at_risk_queue(), reqs, 4) == 4
        ex = make_executor(make_preemption(), capacity=4)
        q = slack_queue(items=[make_item(1000, 99.0, 99.0)])  # 9 s of slack
        assert ex.run(q, reqs, 4) == 4
        assert paused_ids(ex) == set()

    def test_resume_lane_goes_before_new_admissions(self):
        reqs = [make_req(i) for i in range(4)]
        p = make_preemption("newest", max_paused=2)
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
        reqs = [make_req(i) for i in range(4)]
        p = make_preemption("newest", max_paused=2, max_pause_s=10.0)
        ex = make_executor(p, capacity=4)
        ex.run(at_risk_queue(), reqs, 4)
        assert paused_ids(ex) == {3}
        # Still full (4 counted besides 3), no waiter: 3 waits ...
        ex.request_broadcaster.rank0_now = 109.9
        assert ex.run(slack_queue(now=109.9), reqs, 5) == 4
        assert paused_ids(ex) == {3}
        # ... until it has been paused for max_pause_s, then it is back even
        # though that takes the batch one over until a request finishes.
        ex.request_broadcaster.rank0_now = 110.0
        assert ex.run(slack_queue(now=110.0), reqs, 5) == 5
        assert paused_ids(ex) == set() and p.window["forced_resumes"] == 1

    def test_a_forced_resume_that_overfills_can_pause_for_a_waiter(self):
        reqs = [make_req(i) for i in range(5)]
        p = make_preemption("newest", max_paused=2, max_pause_s=10.0)
        ex = make_executor(p, capacity=4)
        p.observe(reqs, 90.0)
        p.on_paused(reqs[4])
        ex.run(at_risk_queue(), reqs, 5)
        # Forced back in (one over); a waiter at risk pauses the next victim.
        assert paused_ids(ex) == {3} and not p.is_paused(4)

    def test_no_starvation_fifo_resume(self):
        reqs = [make_req(i) for i in range(6)]
        p = make_preemption("newest", max_paused=3, max_per_pass=3)
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

    def test_decisions_are_identical_across_ranks(self):
        """Same inputs (active requests, broadcast queue and clock) on every
        rank give the same pauses, whatever the local wall clock."""
        for policy in ["newest", "wait", "long", POLICY_TPS]:
            results = []
            for rank in range(3):
                kw = {"target_tps": 1.0} if policy == POLICY_TPS else {}
                p = make_preemption(policy, max_per_pass=2, **kw)
                reqs = [make_req(i, generated=10 * i, prompt_len=100 + i) for i in range(8)]
                ex = make_executor(p, capacity=8)
                with patch("time.monotonic", return_value=1000.0 * rank):
                    p.observe(reqs, 90.0)
                    step(reqs, iters=20, tokens_per_iter=2)
                    ex.run(at_risk_queue(n=3), reqs, 8)
                results.append(paused_ids(ex))
            assert results[0] == results[1] == results[2], policy
            assert len(results[0]) == 2, policy


class TestExecutorSetup:
    def _stub(self, **overrides):
        seq_slots = Mock()
        seq_slots.get_max_resource_count.return_value = 160
        stub = SimpleNamespace(
            scheduler=object.__new__(KVCacheV2Scheduler),
            _gen_admission=GenAdmissionConfig(ttft_percentile=90.0),
            enable_attention_dp=False,
            dist=SimpleNamespace(has_pp=False, rank=0),
            kv_cache_manager=Mock(max_admissible_sequences=288),
            draft_kv_cache_manager=None,
            max_num_active_requests=128,
            resource_manager=SimpleNamespace(
                resource_managers={ResourceManagerType.SEQ_SLOT_MANAGER: seq_slots}
            ),
        )
        stub.kv_cache_manager._max_resident_sequences.return_value = 160
        stub.__dict__.update(overrides)
        stub._gen_preemption_disabled_reason = types.MethodType(
            PyExecutor._gen_preemption_disabled_reason, stub
        )
        return stub

    def _create(self, stub, env):
        with patch.dict("os.environ", env, clear=False), patch(LOGGER) as log:
            out = PyExecutor._create_gen_preemption(stub)
        return out, [c.args[0] for c in log.info.call_args_list]

    def test_off_creates_nothing(self, monkeypatch):
        monkeypatch.delenv("TLLM_GEN_PREEMPT_POLICY", raising=False)
        out, logs = self._create(self._stub(), {})
        assert out is None and logs == []

    def test_enabled(self):
        out, logs = self._create(self._stub(), ON)
        assert isinstance(out, GenPreemption) and "enabled" in logs[0]

    def test_draft_pool_is_checked(self):
        draft = Mock(max_admissible_sequences=159)
        del draft._max_resident_sequences
        out, logs = self._create(self._stub(draft_kv_cache_manager=draft), ON)
        assert out is None and "index pool holds 159" in logs[0]

    @pytest.mark.parametrize(
        ("overrides", "needle"),
        [
            ({"_gen_admission": GenAdmissionConfig(ttft_percentile=100.0)}, "auto GEN admission"),
            ({"enable_attention_dp": True}, "attention DP"),
            ({"max_num_active_requests": 129}, "sequence slot pool holds 160"),
        ],
    )
    def test_disabled_with_one_log(self, overrides, needle):
        out, logs = self._create(self._stub(**overrides), ON)
        assert out is None and len(logs) == 1
        assert "disabled" in logs[0] and needle in logs[0]

    def test_recurrent_state_pool_is_checked(self):
        stub = self._stub()
        stub.kv_cache_manager._max_resident_sequences.return_value = 136
        out, logs = self._create(stub, ON)
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
        # max_batch scheduled slots plus max_paused kept ones (here 2 + 2).
        worker, meta = self._setup(num_slots=4)
        self._prepare(meta, [1, 2])
        self._prepare(meta, [3, 4], paused={1, 2})
        assert set(worker._req_to_slot) == {1, 2, 3, 4}
        assert len(set(worker._req_to_slot.values())) == 4
