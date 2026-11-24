# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for GEN-side admission ordering (TLLM_GEN_ADMISSION_TTFT_PERCENTILE)."""

import collections
import math
import random
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.request_utils import get_from_waiting_queue
from tensorrt_llm._torch.pyexecutor.scheduler import (
    FCFSWaitingQueue,
    PriorityWaitingQueue,
    create_waiting_queue,
)
from tensorrt_llm._torch.pyexecutor.scheduler.gen_admission_queue import (
    TIER_GUARD,
    TIER_LATE,
    TIER_OLDEST,
    TIER_SLACK,
    TIER_UNKEYED,
    TIER_WARMUP,
    AdaptiveSlackConfig,
    FirstTokenTimer,
    GenAdmissionConfig,
    GenAdmissionWaitingQueue,
    get_orig_elapsed_s,
    parse_ttft_percentile,
    stamp_admission_times,
    work_quantile_for_ttft_percentile,
)
from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy

pytestmark = pytest.mark.cpu_only

ENV = "TLLM_GEN_ADMISSION_TTFT_PERCENTILE"
GAQ = "tensorrt_llm._torch.pyexecutor.scheduler.gen_admission_queue"
OLDEST = GenAdmissionConfig(ttft_percentile=100.0)
AUTO = GenAdmissionConfig(ttft_percentile=90.0)


def make_item(req_id, gen_arrival=None, orig_arrival=None, elapsed=None):
    """A queue item; ``elapsed`` sets py_disaggregated_params.orig_elapsed_s."""
    request = Mock()
    request.py_disaggregated_params = (
        SimpleNamespace(orig_elapsed_s=elapsed) if elapsed is not None else None
    )
    item = RequestQueueItem(req_id, request)
    item.admit_arrival = gen_arrival
    item.admit_orig_arrival = orig_arrival
    return item


def admit(q):
    """Pop the next request and admit it, as get_from_waiting_queue does."""
    item = q.pop_request()
    q.mark_admitted([item])
    return item


def drain_items(q):
    out = []
    while q:
        peeked = q.peek_request()
        popped = admit(q)
        assert peeked is popped
        out.append(popped)
    return out


def drain(q):
    return [item.id for item in drain_items(q)]


def tiers(items):
    return [item._gen_admission_tier for item in items]


def fixed_slack(x, g, cfg=AUTO):
    """An ``auto`` queue past warm-up with X and G pinned."""
    q = GenAdmissionWaitingQueue(cfg)
    q.adaptive.X, q.adaptive.G = x, g
    return q


def active(req_id, decoding_iter):
    return SimpleNamespace(py_request_id=req_id, py_decoding_iter=decoding_iter)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_default_is_fcfs_and_disabled(self):
        for env in ({}, {ENV: ""}, {ENV: "0"}, {ENV: " 0.0 "}):
            cfg = GenAdmissionConfig.from_env(env)
            assert cfg.policy == "fcfs" and cfg.ttft_percentile == 0.0
            assert not cfg.enabled and not cfg.needs_clock and cfg.auto is None

    @pytest.mark.parametrize("raw", ["99", "99.5", "100"])
    def test_p99_and_above_is_oldest_first(self, raw):
        cfg = GenAdmissionConfig.from_env({ENV: raw})
        assert cfg.policy == "oldest" and cfg.enabled and not cfg.needs_clock
        assert cfg.auto is None
        q = create_waiting_queue(gen_admission=cfg)
        assert isinstance(q, GenAdmissionWaitingQueue) and q.adaptive is None

    def test_just_below_p99_is_auto(self):
        cfg = GenAdmissionConfig.from_env({ENV: "98.9"})
        assert cfg.policy == "auto" and cfg.auto.q == pytest.approx(0.98, abs=0.01)

    def test_p90_is_auto_with_the_validated_constants(self):
        cfg = GenAdmissionConfig.from_env({ENV: "90"})
        assert cfg.policy == "auto" and cfg.enabled and cfg.needs_clock
        assert cfg.auto == AdaptiveSlackConfig()  # the defaults are P = 90's
        assert cfg.auto.q == pytest.approx(0.85) and cfg.auto.guard_k == 1.5
        assert "ttft_percentile=90 policy=auto q=0.85" in cfg.describe()

    @pytest.mark.parametrize(
        "raw, expected", [(" 99.5 ", 99.5), ("50", 50.0), ("1e1", 10.0), ("100.0", 100.0)]
    )
    def test_parse(self, raw, expected):
        assert parse_ttft_percentile(raw) == expected
        assert GenAdmissionConfig.from_env({ENV: raw}).ttft_percentile == expected

    @pytest.mark.parametrize("raw", ["p90", "auto", "oldest", "-1", "100.5", "nan", "inf", "9 0"])
    def test_invalid(self, raw):
        with pytest.raises(ValueError, match=ENV):
            GenAdmissionConfig.from_env({ENV: raw})

    @pytest.mark.parametrize(
        "p, q",
        [
            (1.0, 0.5),
            (70.0, 0.5),  # 0.7 - 5 * 0.09 = 0.25, clipped
            (80.0, 0.60),
            (85.0, 0.7375),
            (90.0, 0.85),
            (95.0, 0.9375),
            (98.0, 0.978),
            (98.99, 0.9899 - 5 * 0.0101**2),
        ],
    )
    def test_q_of_p(self, p, q):
        assert work_quantile_for_ttft_percentile(p) == pytest.approx(q)
        cfg = GenAdmissionConfig(ttft_percentile=p)
        assert cfg.auto == AdaptiveSlackConfig(q=cfg.auto.q)
        assert cfg.auto.q == pytest.approx(q)

    def test_explicit_auto_constants(self):
        a = AdaptiveSlackConfig(n=10)
        assert GenAdmissionConfig(ttft_percentile=90.0, auto=a).auto is a
        with pytest.raises(ValueError):
            GenAdmissionConfig(ttft_percentile=101.0)

    def test_auto_defaults(self):
        a = AdaptiveSlackConfig()
        assert (a.q, a.n, a.upd_s, a.alpha) == (0.85, 2000, 10.0, 0.02)
        assert (a.cap_up, a.cap_dn, a.freeze, a.margin) == (1.0, 0.2, 0.5, 0.5)
        assert (a.x_lo, a.x_hi, a.guard_k, a.queued_s, a.warm_frac) == (4.0, 60.0, 1.5, 0.1, 0.25)
        assert (a.s2f_seed_s, a.s2f_alpha) == (0.17, 0.05)

    def test_factory(self):
        assert isinstance(
            create_waiting_queue(WaitingQueuePolicy.FCFS, gen_admission=GenAdmissionConfig()),
            FCFSWaitingQueue,
        )
        assert isinstance(
            create_waiting_queue(WaitingQueuePolicy.PRIORITY, gen_admission=GenAdmissionConfig()),
            PriorityWaitingQueue,
        )
        auto = create_waiting_queue(WaitingQueuePolicy.PRIORITY, gen_admission=AUTO)
        assert isinstance(auto, GenAdmissionWaitingQueue) and auto.adaptive is not None
        with pytest.raises(ValueError):
            GenAdmissionWaitingQueue(GenAdmissionConfig())


# ---------------------------------------------------------------------------
# Stamping
# ---------------------------------------------------------------------------
class TestStamping:
    def test_elapsed_rebased_onto_local_clock(self):
        item = make_item(1, elapsed=3.5)
        item.admit_arrival = item.admit_orig_arrival = None
        stamp_admission_times(item, 100.0)
        assert item.admit_arrival == 100.0
        assert item.admit_orig_arrival == pytest.approx(96.5)

    @pytest.mark.parametrize("elapsed", [None, -1.0, float("nan"), float("inf"), 1e9, "x"])
    def test_bad_or_missing_elapsed_is_unkeyed(self, elapsed):
        request = Mock()
        request.py_disaggregated_params = SimpleNamespace(orig_elapsed_s=elapsed)
        assert get_orig_elapsed_s(request) is None
        item = RequestQueueItem(1, request)
        stamp_admission_times(item, 50.0)
        assert item.admit_arrival == 50.0
        assert item.admit_orig_arrival is None

    def test_no_disagg_params(self):
        request = Mock(spec=[])  # no py_disaggregated_params attribute
        assert get_orig_elapsed_s(request) is None

    def _queue(self, stamp):
        dist = Mock()
        dist.rank = 0
        return ExecutorRequestQueue(
            dist=dist,
            max_batch_size=8,
            enable_iter_perf_stats=False,
            batch_wait_timeout_ms=0.0,
            stamp_admission_times=stamp,
        )

    def _request(self, elapsed):
        request = Mock()
        request.disagg_request_id = None
        request.sampling_config.num_return_sequences = 1
        request.sampling_config.beam_width = 1
        request.py_disaggregated_params = SimpleNamespace(orig_elapsed_s=elapsed)
        return request

    def test_executor_request_queue_stamps_when_enabled(self):
        q = self._queue(stamp=True)
        q.enqueue_request(self._request(2.0))
        item = q.request_queue.get_nowait()
        assert item.admit_arrival is not None
        assert item.admit_arrival - item.admit_orig_arrival == pytest.approx(2.0)

    def test_executor_request_queue_default_does_not_stamp(self):
        q = self._queue(stamp=False)
        q.enqueue_request(self._request(2.0))
        item = q.request_queue.get_nowait()
        assert item.admit_arrival is None and item.admit_orig_arrival is None


# ---------------------------------------------------------------------------
# oldest
# ---------------------------------------------------------------------------
class TestOldest:
    def test_orders_by_original_arrival(self):
        q = GenAdmissionWaitingQueue(OLDEST)
        # GEN arrival order 1,2,3 but original arrivals 30,10,20.
        q.add_requests(
            [
                make_item(1, gen_arrival=100.0, orig_arrival=30.0),
                make_item(2, gen_arrival=101.0, orig_arrival=10.0),
                make_item(3, gen_arrival=102.0, orig_arrival=20.0),
            ]
        )
        q.set_now(103.0)
        assert [r.id for r in q] == [2, 3, 1]
        items = drain_items(q)
        assert [r.id for r in items] == [2, 3, 1]
        assert set(tiers(items)) == {TIER_OLDEST}

    def test_unkeyed_only_is_fcfs(self):
        q = GenAdmissionWaitingQueue(OLDEST)
        q.add_requests(make_item(i, gen_arrival=float(i)) for i in range(10))
        assert drain(q) == list(range(10))

    def test_unstamped_items_are_deterministic_fcfs(self):
        q = GenAdmissionWaitingQueue(OLDEST)
        q.add_request(make_item(1))
        q.add_request(make_item(2))
        q.add_request(make_item(3))
        assert drain(q) == [1, 2, 3]

    def test_prepend_keeps_place(self):
        q = GenAdmissionWaitingQueue(OLDEST)
        items = [make_item(i, gen_arrival=10.0, orig_arrival=1.0) for i in range(4)]
        q.add_requests(items)
        a = q.pop_request()
        b = q.pop_request()
        q.add_request(make_item(9, gen_arrival=11.0, orig_arrival=1.0))  # same key, later
        q.prepend_requests([a, b])
        assert drain(q) == [0, 1, 2, 3, 9]

    def test_remove_by_ids(self):
        q = GenAdmissionWaitingQueue(OLDEST)
        q.add_requests(make_item(i, gen_arrival=float(i), orig_arrival=-float(i)) for i in range(5))
        q.remove_by_ids({1, 3})
        assert len(q) == 3
        assert drain(q) == [4, 2, 0]

    def test_empty(self):
        q = GenAdmissionWaitingQueue(OLDEST)
        assert not q and len(q) == 0
        with pytest.raises(IndexError):
            q.pop_request()
        with pytest.raises(IndexError):
            q.peek_request()


# ---------------------------------------------------------------------------
# auto: the slack pick for a given X / G / s2f
# ---------------------------------------------------------------------------
class TestSlackPick:
    def test_tiers(self):
        q = fixed_slack(14.0, 20.0)
        now = 1000.0
        q.add_requests(
            [
                # late (age 18 > X), GEN wait 5
                make_item(1, gen_arrival=now - 5, orig_arrival=now - 18),
                # feasible (age 10), GEN wait 4
                make_item(2, gen_arrival=now - 4, orig_arrival=now - 10),
                # feasible (age 12), GEN wait 3 -> oldest feasible
                make_item(3, gen_arrival=now - 3, orig_arrival=now - 12),
                # aged at GEN (wait 21 >= 20) but young originally
                make_item(4, gen_arrival=now - 21, orig_arrival=now - 21.5),
                # late (age 30)
                make_item(5, gen_arrival=now - 1, orig_arrival=now - 30),
            ]
        )
        q.set_now(now, 0.0)
        items = drain_items(q)
        assert [r.id for r in items] == [4, 3, 2, 5, 1]
        assert tiers(items) == [TIER_GUARD, TIER_SLACK, TIER_SLACK, TIER_LATE, TIER_LATE]

    def test_guard_picks_oldest_original_among_aged(self):
        q = fixed_slack(14.0, 20.0)
        now = 100.0
        q.add_requests(
            [
                make_item(1, gen_arrival=now - 25, orig_arrival=now - 26),
                make_item(2, gen_arrival=now - 22, orig_arrival=now - 40),
                make_item(3, gen_arrival=now - 1, orig_arrival=now - 2),
            ]
        )
        q.set_now(now, 0.0)
        assert q.pop_request().id == 2
        assert q.pop_request().id == 1
        assert q.pop_request().id == 3

    def test_s2f_shrinks_feasible_window(self):
        q = fixed_slack(14.0, math.inf)
        now = 100.0
        q.add_requests(
            [
                make_item(1, gen_arrival=now, orig_arrival=now - 13.7),  # 13.7 + 0.5 > 14: late
                make_item(2, gen_arrival=now, orig_arrival=now - 13.0),  # feasible
            ]
        )
        q.set_now(now, 0.5)
        assert drain(q) == [2, 1]

    def test_s2f_none_keeps_the_last_value(self):
        q = fixed_slack(14.0, math.inf)
        assert q.adaptive.s2f_s == 0.17  # seed
        q.set_now(1.0, 0.5)
        q.set_now(2.0)
        assert q.adaptive.s2f_s == 0.5

    def test_no_clock_falls_back_to_oldest(self):
        q = fixed_slack(14.0, 20.0)
        q.add_requests(
            [
                make_item(1, gen_arrival=5.0, orig_arrival=3.0),
                make_item(2, gen_arrival=6.0, orig_arrival=1.0),
            ]
        )
        assert drain(q) == [2, 1]

    def test_unkeyed_only_is_exact_fcfs(self):
        rng = random.Random(0)
        q = fixed_slack(14.0, 20.0)
        arrivals = sorted(rng.uniform(0, 60) for _ in range(50))
        q.add_requests(make_item(i, gen_arrival=a) for i, a in enumerate(arrivals))
        q.set_now(61.0, 0.0)  # mix of GEN waits above and below the 20 s guard
        items = drain_items(q)
        assert [r.id for r in items] == list(range(50))
        assert set(tiers(items)) <= {TIER_GUARD, TIER_UNKEYED}

    def test_peek_pop_agree_and_prepend(self):
        q = fixed_slack(14.0, 20.0)
        now = 100.0
        q.add_requests(
            [
                make_item(1, gen_arrival=now - 1, orig_arrival=now - 30),
                make_item(2, gen_arrival=now - 1, orig_arrival=now - 10),
            ]
        )
        q.set_now(now, 0.0)
        first = q.pop_request()
        assert first.id == 2
        q.prepend_request(first)  # e.g. could not be scheduled this iteration
        assert q.peek_request() is first
        assert drain(q) == [2, 1]

    def test_peek_requests_follows_admission_order(self):
        q = fixed_slack(14.0, 20.0)
        now = 1000.0
        q.add_requests(
            [
                make_item(1, gen_arrival=now - 5, orig_arrival=now - 18),
                make_item(2, gen_arrival=now - 4, orig_arrival=now - 10),
                make_item(3, gen_arrival=now - 3, orig_arrival=now - 12),
                make_item(4, gen_arrival=now - 21, orig_arrival=now - 21.5),
                make_item(5, gen_arrival=now - 1, orig_arrival=now - 30),
            ]
        )
        q.set_now(now, 0.0)
        assert [r.id for r in q.peek_requests(3)] == [4, 3, 2]
        assert q.admission_groups(q.peek_requests(3)) == [0, 1, 2]
        assert q.peek_requests(0) == []
        assert admit(q).id == 4
        assert [r.id for r in q.peek_requests(10)] == [3, 2, 5, 1]
        assert len(q) == 4
        assert drain(q) == [3, 2, 5, 1]
        assert q.peek_requests(2) == []

    def test_stats_line(self):
        q = fixed_slack(14.0, 20.0)
        now = 100.0
        q.add_requests(
            [
                make_item(1, gen_arrival=now - 3, orig_arrival=now - 9),
                make_item(2, gen_arrival=now - 2),
            ]
        )
        q.set_now(now, 0.25)
        admit(q)
        line = q.stats_line(now)
        assert "qlen=1" in line and "oldest_gen_wait=2.00s" in line
        assert "slack=1" in line and "X=14.00" in line and "s2f=0.250" in line
        assert "admitted_since_last={}" in q.stats_line(now)

    def test_rank0_logging(self):
        """Rank 0 logs a stat line every 30 s at info level and each admission at
        debug level; other ranks log nothing."""
        for rank0 in (True, False):
            with patch(f"{GAQ}.logger") as log:
                log.is_debug_enabled.return_value = True
                q = GenAdmissionWaitingQueue(AUTO)
                if rank0:
                    q.enable_rank0()
                q.add_request(make_item(7, gen_arrival=1.0, orig_arrival=0.5))
                q.set_now(2.0)
                admit(q)
                q.set_now(31.0)
                q.set_now(32.0)
            infos = [c.args[0] for c in log.info.call_args_list]
            debugs = [c.args[0] for c in log.debug.call_args_list]
            if not rank0:
                assert not infos and not debugs
                continue
            assert infos[0].startswith("[gen_admission] GEN admission ordering enabled")
            assert [line.split(" qlen")[0] for line in infos[1:]] == [
                "[gen_admission] p90 policy=auto"
            ] * 2
            assert "admitted_since_last={warmup=1}" in infos[2]
            (debug,) = debugs
            assert "admit req_id=7 tier=warmup orig_age_s=1.500 gen_wait_s=1.000" in debug


# ---------------------------------------------------------------------------
# FirstTokenTimer: rank 0's admission -> first token estimate
# ---------------------------------------------------------------------------
class TestFirstTokenTimer:
    def test_ema_of_pop_to_first_token(self):
        t = FirstTokenTimer(seed_s=0.17, alpha=0.5)
        t.on_pop(1, 10.0)
        t.on_pop(2, 10.0)
        t.observe([active(1, 0), active(2, 0)], 10.2)  # in KV transfer
        assert t.value == 0.17 and t.n_samples == 0
        t.observe([active(1, 1), active(2, 0)], 10.5)
        assert t.value == pytest.approx(0.17 + 0.5 * (0.5 - 0.17))
        t.observe([active(1, 2), active(2, 3)], 11.0)  # 1 is sampled only once
        assert t.n_samples == 2
        assert t.value == pytest.approx(0.335 + 0.5 * (1.0 - 0.335))
        assert len(t) == 0

    def test_repop_restarts_the_timer(self):
        t = FirstTokenTimer(seed_s=0.0, alpha=1.0)
        t.on_pop(1, 10.0)
        t.on_pop(1, 12.0)  # put back and popped again
        t.observe([active(1, 1)], 12.25)
        assert t.value == pytest.approx(0.25)

    def test_outliers_dropped_and_lost_requests_expire(self):
        t = FirstTokenTimer(seed_s=0.2, alpha=1.0, max_sample_s=30.0, ttl_s=300.0)
        t.on_pop(1, 0.0)
        t.on_pop(2, 100.0)
        t.observe([active(1, 1)], 40.0)  # 40 s > max_sample_s: dropped
        assert t.value == 0.2 and t.n_samples == 0
        assert len(t) == 1
        t.observe([], 450.0)  # 2 never showed up here (other rank): expires
        assert len(t) == 0
        t.on_pop(3, 460.0)
        t.observe([active(3, 1)], 460.5)
        assert t.value == pytest.approx(0.5)

    def test_pending_is_bounded(self):
        t = FirstTokenTimer(seed_s=0.2, alpha=1.0, max_pending=3)
        for i in range(5):
            t.on_pop(i, float(i))
        assert len(t) == 3

    def test_queue_times_keyed_pops_on_rank0_only(self):
        q = GenAdmissionWaitingQueue(AUTO)
        assert q.first_token_timer is None  # not rank 0
        timer = q.enable_rank0()
        assert q.enable_rank0() is timer
        q.add_requests(
            [make_item(1, gen_arrival=0.0, orig_arrival=-1.0), make_item(2, gen_arrival=0.0)]
        )
        q.set_now(1.0)
        drain(q)
        assert len(timer) == 1  # unkeyed requests are not timed
        timer.observe([active(1, 1), active(2, 1)], 1.4)
        assert timer.n_samples == 1
        assert "s2f_samples=1" in q.stats_line(1.4)


# ---------------------------------------------------------------------------
# auto: adaptive slack target / guard
# ---------------------------------------------------------------------------
class RefAuto:
    """Numpy reference of the ``auto`` policy (X, G, s2f and the pick)."""

    def __init__(self, a: AdaptiveSlackConfig):
        self.a = a
        self.s2f = a.s2f_seed_s
        self.obs_ctx = collections.deque(maxlen=a.n)
        self.obs_gw = collections.deque(maxlen=a.n)
        self.X = self.A = self.last_upd = None
        self.upd_n = self.upd_q = self.n_q = self.n_upd = 0

    def first_token(self, dt):
        self.s2f += self.a.s2f_alpha * (dt - self.s2f)

    def target(self, ctx, gw):
        a = self.a
        v = float(np.quantile(np.fromiter(ctx, float), a.q)) + (float(np.mean(gw)) if gw else 0.0)
        return v + a.margin

    def update(self, t):
        a = self.a
        if self.last_upd is not None and t - self.last_upd < a.upd_s:
            return
        self.last_upd = t
        busy = self.upd_n > 0 and self.upd_q / self.upd_n >= a.freeze
        self.upd_n = self.upd_q = 0
        if self.n_q >= a.n:
            tgt = self.target(self.obs_ctx, self.obs_gw)
            if self.X is None:
                k = max(1, a.n // 4)
                self.X = self.target(list(self.obs_ctx)[-k:], list(self.obs_gw)[-k:])
            elif busy:
                self.X += max(-a.cap_dn, min(a.cap_up, a.alpha * (tgt - self.X)))
            self.X = max(a.x_lo, min(a.x_hi, self.X))
            self.A = self.X + a.guard_k * (float(np.mean(self.obs_gw)) if self.obs_gw else 0.0)
            self.n_upd += 1

    def pick(self, W, t):
        if self.X is None:
            return min(range(len(W)), key=lambda i: W[i]["t_arr"])
        aged = [i for i, e in enumerate(W) if t - e["t_arr"] >= self.A]
        if aged:
            return min(aged, key=lambda i: W[i]["t_orig"])
        feas = [i for i, e in enumerate(W) if (t - e["t_orig"]) + self.s2f <= self.X]
        if feas:
            return min(feas, key=lambda i: W[i]["t_orig"])
        return min(range(len(W)), key=lambda i: W[i]["t_orig"])

    def admit(self, e, t):
        gw = t - e["t_arr"]
        self.obs_ctx.append(t - e["t_orig"] + self.s2f - gw)
        self.obs_gw.append(gw)
        self.upd_n += 1
        self.upd_q += int(gw > self.a.queued_s)
        self.n_q += int(gw > self.a.queued_s)


# P = 90 with a short window and fast steps, so that X warms up and moves.
FAST = AdaptiveSlackConfig(n=40, upd_s=1.0, alpha=0.3, cap_up=1.0, cap_dn=0.5, s2f_alpha=0.2)


def auto_cfg(a=FAST):
    return GenAdmissionConfig(ttft_percentile=90.0, auto=a)


def run_auto(cfg, seed, steps=400, rate=(0, 4), pops=(0, 4), upstream=(0, 20)):
    """Drive a rank-0 queue, a replica fed only the broadcast values (another
    rank) and the reference with one random workload; every pop and every
    X / G / s2f value must agree."""
    rng = random.Random(seed)
    q = GenAdmissionWaitingQueue(cfg)
    timer = q.enable_rank0()
    replica = GenAdmissionWaitingQueue(cfg)
    ref = RefAuto(cfg.auto)
    waiting, now, next_id, xs = [], 1000.0, 0, []
    in_flight = {}  # req_id -> (pop time, first-token time)
    for _ in range(steps):
        for _ in range(rng.randint(*rate)):
            t_arr = now - rng.uniform(0.0, 0.05)
            e = dict(id=next_id, t_arr=t_arr, t_orig=t_arr - rng.uniform(*upstream))
            waiting.append(e)
            for queue in (q, replica):
                queue.add_request(
                    make_item(next_id, gen_arrival=e["t_arr"], orig_arrival=e["t_orig"])
                )
            next_id += 1
        now += rng.uniform(0.05, 0.6)
        # Rank 0: first tokens since the last iteration -> s2f -> broadcast.
        done = [rid for rid, (_, t_ft) in in_flight.items() if t_ft <= now]
        timer.observe([active(rid, int(rid in done)) for rid in in_flight], now)
        for rid in done:
            ref.first_token(now - in_flight.pop(rid)[0])
        s2f = timer.value
        q.set_now(now, s2f)
        replica.set_now(now, s2f)
        ref.update(now)
        for _ in range(rng.randint(*pops)):
            if not waiting:
                break
            e = waiting.pop(ref.pick(waiting, now))
            ref.admit(e, now)
            in_flight[e["id"]] = (now, now + rng.uniform(0.05, 1.5))
            assert admit(q).id == e["id"]
            assert admit(replica).id == e["id"]
        ada = q.adaptive
        assert ada.s2f_s == pytest.approx(ref.s2f, abs=1e-12)
        if ref.X is None:
            assert ada.X is None
        else:
            assert ada.X == pytest.approx(ref.X, abs=1e-9)
            assert ada.G == pytest.approx(ref.A, abs=1e-9)
        assert (replica.adaptive.X, replica.adaptive.G) == (ada.X, ada.G)
        assert list(replica.adaptive.obs_work) == list(ada.obs_work)
        xs.append(ada.X)
    return q, ref, xs


class TestAuto:
    def test_warmup_is_fcfs_by_gen_arrival(self):
        q = GenAdmissionWaitingQueue(auto_cfg())
        now = 100.0
        q.add_requests(
            [
                make_item(1, gen_arrival=now - 1, orig_arrival=now - 30),
                make_item(2, gen_arrival=now - 3, orig_arrival=now - 4),
                make_item(3, gen_arrival=now - 2, orig_arrival=now - 50),
            ]
        )
        q.set_now(now)
        items = drain_items(q)
        assert [r.id for r in items] == [2, 3, 1]
        assert set(tiers(items)) == {TIER_WARMUP}
        assert q.adaptive.X is None

    @pytest.mark.parametrize("seed", range(12))
    def test_matches_reference_at_p90(self, seed):
        assert FAST.q == pytest.approx(0.85)
        q, ref, xs = run_auto(auto_cfg(), seed)
        assert ref.X is not None, "X never warm-started"
        assert len(set(round(x, 6) for x in xs if x is not None)) > 3, "X never moved"
        assert q.first_token_timer.n_samples > 100
        assert q.adaptive.s2f_s != FAST.s2f_seed_s

    def test_freeze_holds_x_while_gen_idle(self):
        cfg = auto_cfg(AdaptiveSlackConfig(n=20, upd_s=1.0, alpha=1.0, cap_up=100, cap_dn=100))
        q = GenAdmissionWaitingQueue(cfg)
        now, rid = 0.0, 0

        def admit_batch(upstream, gen_wait):
            nonlocal now, rid
            for _ in range(10):
                q.add_request(
                    make_item(
                        rid, gen_arrival=now - gen_wait, orig_arrival=now - gen_wait - upstream
                    )
                )
                rid += 1
            q.set_now(now)
            while q:
                admit(q)
            now += 1.0

        for _ in range(3):  # 30 queued admissions: warm start
            admit_batch(upstream=10.0, gen_wait=1.0)
        q.set_now(now)
        x0 = q.adaptive.X
        assert x0 is not None
        # Upstream jumps to 30 s but nothing queues at GEN (wait 0): X is held.
        for _ in range(5):
            admit_batch(upstream=30.0, gen_wait=0.0)
        q.set_now(now)
        assert q.adaptive.X == pytest.approx(x0)
        # Queued again: X follows the new level.
        admit_batch(upstream=30.0, gen_wait=1.0)
        q.set_now(now + 1.0)
        assert q.adaptive.X > x0 + 5.0

    def test_warm_start_uses_freshest_quarter(self):
        a = AdaptiveSlackConfig(n=40, upd_s=1.0, margin=0.0, guard_k=0.0, q=0.5, s2f_seed_s=0.0)
        q = GenAdmissionWaitingQueue(auto_cfg(a))
        now = 0.0
        for i in range(40):  # first 30: work 5 s, last 10: work 20 s; GEN wait 1 s
            work = 5.0 if i < 30 else 20.0
            q.add_request(make_item(i, gen_arrival=now - 1.0, orig_arrival=now - 1.0 - work))
            q.set_now(now)
            admit(q)
            now += 0.01
        q.set_now(now + 1.0)  # first update after the 40 admissions
        # target over the freshest n/4 = 10 admissions: median 20 + mean wait 1
        assert q.adaptive.X == pytest.approx(21.0)
        assert q.adaptive.G == pytest.approx(21.0)  # guard_k 0

    def test_readmitted_request_observed_once(self):
        q = GenAdmissionWaitingQueue(auto_cfg())
        q.add_request(make_item(1, gen_arrival=0.0, orig_arrival=-5.0))
        q.set_now(1.0)
        item = admit(q)
        q.prepend_request(item)
        q.set_now(2.0)
        assert admit(q) is item
        assert len(q.adaptive.obs_work) == 1
        assert q.adaptive.obs_gw[0] == pytest.approx(1.0)
        assert q.adaptive.obs_work[0] == pytest.approx(5.0 + 0.17)

    def test_unkeyed_requests_not_observed(self):
        q = GenAdmissionWaitingQueue(auto_cfg())
        q.add_request(make_item(1, gen_arrival=0.0))
        q.set_now(1.0)
        admit(q)
        assert len(q.adaptive.obs_work) == 0

    def test_update_events_logged(self):
        with patch(f"{GAQ}.logger") as log:
            q, ref, _ = run_auto(auto_cfg(), 0)
        infos = [c.args[0] for c in log.info.call_args_list]
        events = [line for line in infos if line.startswith("[gen_admission] auto ")]
        assert any(e.startswith("[gen_admission] auto warm_start") for e in events)
        assert all(" X=" in e and "target=" in e and "s2f=" in e for e in events)
        # One line per update period once warmed up, held or not.
        assert len(events) == ref.n_upd
        assert any(line.startswith("[gen_admission] p90 policy=auto") for line in infos)


# ---------------------------------------------------------------------------
# Integration with get_from_waiting_queue
# ---------------------------------------------------------------------------
def test_adp_put_back_is_not_an_admission():
    """A request popped but put back (its attention-DP rank is full) is not
    observed, timed or counted; its GEN wait runs to its real admission."""
    q = GenAdmissionWaitingQueue(AUTO)
    timer = q.enable_rank0()
    pinned = make_item(1, gen_arrival=0.0, orig_arrival=-5.0)
    pinned.request.py_scheduling_params = SimpleNamespace(
        attention_dp_rank=0, attention_dp_relax=False
    )
    free = make_item(2, gen_arrival=0.5, orig_arrival=-1.0)
    free.request.py_scheduling_params = None
    q.add_requests([pinned, free])
    q.set_now(1.0)
    got = get_from_waiting_queue(q, 2, True, 1, all_ranks_num_active_requests=[1, 0])
    assert [r.id for r in got] == [2] and len(q) == 1
    assert len(q.adaptive.obs_work) == 1 and len(timer) == 1
    assert sum(q.admitted_by_tier.values()) == 1
    q.set_now(3.0)
    got = get_from_waiting_queue(q, 1, True, 1, all_ranks_num_active_requests=[0, 0])
    assert [r.id for r in got] == [1]
    assert q.adaptive.obs_gw[-1] == pytest.approx(3.0)
    assert sum(q.admitted_by_tier.values()) == 2


def test_get_from_waiting_queue_respects_capacity_and_order():
    q = fixed_slack(14.0, 20.0)
    now = 100.0
    for i in range(6):
        # Original ages 5..30 in reverse GEN arrival order; none aged at GEN.
        q.add_request(make_item(i, gen_arrival=now - 6 + i, orig_arrival=now - 30 + 5 * i))
    q.set_now(now, 0.0)
    # ages: 30, 25, 20, 15, 10, 5 -> feasible (<=14): ids 4 (10), 5 (5);
    # then late oldest-first: 0, 1, 2, 3.
    got = get_from_waiting_queue(q, 3, enable_attention_dp=False, max_num_active_requests=128)
    assert [r.id for r in got] == [4, 5, 0]
    assert len(q) == 3


def test_adp_put_backs_build_the_order_once():
    """Requests pinned to a full attention-DP rank are popped and put back
    without rescanning the queue per pop: one order build per pass."""
    q = fixed_slack(14.0, 20.0)
    now = 100.0
    for i in range(50):
        item = make_item(i, gen_arrival=now - 1, orig_arrival=now - 30 + 0.1 * i)
        item.request.py_scheduling_params = SimpleNamespace(
            attention_dp_rank=0 if i < 49 else 1, attention_dp_relax=False
        )
        q.add_request(item)
    q.set_now(now, 0.0)
    with patch.object(q, "_build_order", wraps=q._build_order) as build:
        got = get_from_waiting_queue(q, 2, True, 1, all_ranks_num_active_requests=[1, 0])
        assert build.call_count == 1
    assert [r.id for r in got] == [49]
    assert len(q) == 49 and [r.id for r in q] == list(range(49))
    assert drain(q) == list(range(49))


def test_pops_interleaved_with_changes():
    q = fixed_slack(14.0, 20.0)
    now = 100.0
    # ages 20, 18, 16, 12 (feasible), 10 (feasible)
    q.add_requests(
        make_item(i, gen_arrival=now - 1, orig_arrival=now - a)
        for i, a in enumerate([20, 18, 16, 12, 10])
    )
    q.set_now(now, 0.0)
    assert admit(q).id == 3
    q.add_request(make_item(5, gen_arrival=now, orig_arrival=now - 13))  # feasible, older
    assert q.peek_request().id == 5 and len(q) == 5
    q.remove_by_ids({5, 4})
    assert [r.id for r in q] == [0, 1, 2]
    q.add_request(make_item(6, gen_arrival=now - 25, orig_arrival=now - 25))  # guard
    assert drain(q) == [6, 0, 1, 2]


def test_request_finishing_on_its_first_token_is_sampled():
    """max_tokens=2 without overlap: token 1 is imported and token 2 finishes
    the request in one iteration, so it leaves before the next fetch."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    request = Mock()
    request.py_request_id = 7
    request.is_attention_dp_dummy = False
    request.py_kv_transfer_timed_out = False
    request.is_generation_only_request = True
    request.is_finished = True
    request.py_decoding_iter = 1
    request.return_perf_metrics = False
    request.py_total_draft_tokens = 0
    request.is_disagg_context_complete_state = False
    request.is_disagg_context_transmission_state = False
    executor = object.__new__(PyExecutor)
    executor.active_requests = [request]
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.perf_manager = Mock()
    executor.iter_counter = 1
    executor.stream_interval = 1
    executor.force_terminate_ctx_for_partial_reuse = False
    executor._disagg_coordinator = Mock()
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()
    executor._first_token_timer = FirstTokenTimer(seed_s=0.0, alpha=1.0)
    executor._first_token_timer.on_pop(7, 10.0)
    with patch("time.monotonic", return_value=10.3):
        PyExecutor._handle_responses(executor)
    executor._terminate_request.assert_called_once_with(request)
    assert executor.active_requests == []
    assert executor._first_token_timer.n_samples == 1
    assert executor._first_token_timer.value == pytest.approx(0.3)
