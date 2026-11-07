# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pause resident GEN requests to admit waiters at risk of missing TTFT.

The TPOT-budget policy prefers long-running requests; the output-rate policy
spends request laxity or bounded sacrifice credit. Decisions use rank-identical
request state and rank 0's clock. Paused requests retain all resident resources.
"""

import collections
import math
import time
from dataclasses import dataclass
from typing import Optional

from tensorrt_llm.logger import logger

from ..llm_request import LlmRequest, LlmRequestState
from .gen_admission_queue import STATS_LOG_PERIOD_S, GenAdmissionConfig, GenAdmissionWaitingQueue


@dataclass(frozen=True)
class GenPreemptionSettings:
    """``scheduler_config.gen_preemption``: at most one bound; neither is off."""

    tpot_budget: Optional[float] = None
    target_output_tps: Optional[float] = None

    @classmethod
    def from_config(cls, config) -> "GenPreemptionSettings":
        return cls(config.tpot_budget, config.target_output_tps) if config else cls()

    @property
    def enabled(self) -> bool:
        return self.tpot_budget is not None or self.target_output_tps is not None


def paused_cap(max_num_active_requests: int) -> int:
    """Requests paused at once, also the pools' resident headroom."""
    return max(1, int(round(0.375 * max_num_active_requests)))


# Settings of the executor being created; pools are sized from them.
_active_settings = GenPreemptionSettings()


def configure(config) -> None:
    """Install ``scheduler_config.gen_preemption`` for pool sizing."""
    global _active_settings
    _active_settings = GenPreemptionSettings.from_config(config)


def resident_headroom(max_batch_size: int, settings: Optional[GenPreemptionSettings] = None) -> int:
    """Entries a pool with one per resident request adds for paused requests."""
    settings = settings or _active_settings
    return paused_cap(max_batch_size) if settings.enabled else 0


def unsupported_headroom_reason(config, llm_args) -> Optional[str]:
    """Resolve prerequisites before reserving pool headroom.

    An unresolved V2 ``"auto"`` keeps headroom until the runtime check; explicit
    V1 or settings that demote V2 do not reserve it.
    """
    if config is None or not GenPreemptionSettings.from_config(config).enabled:
        return None
    if llm_args is None:
        return None
    if getattr(llm_args, "pipeline_parallel_size", 1) > 1:
        return "not supported with pipeline parallelism"
    if getattr(llm_args, "enable_attention_dp", False):
        return "not supported with attention DP"
    if not GenAdmissionConfig.from_env().needs_clock:
        return "needs the auto GEN admission policy (0 < TLLM_GEN_ADMISSION_TTFT_PERCENTILE < 99)"
    kv_cache_config = getattr(llm_args, "kv_cache_config", None)
    use_v2 = getattr(kv_cache_config, "use_kv_cache_manager_v2", None)
    if use_v2 is False:
        return "needs the KV cache manager V2 scheduler"
    if use_v2 is True:
        from .._util import resolved_kv_cache_manager_is_v2

        if not resolved_kv_cache_manager_is_v2(
            kv_cache_config,
            getattr(llm_args, "max_beam_width", None),
            enable_sleep=getattr(llm_args, "sleep_config", None) is not None,
        ):
            return "needs the KV cache manager V2 scheduler"
    return None


def generated_tokens(req) -> int:
    return max(0, req.get_num_tokens(0) - req.prompt_len)


@dataclass
class _Req:  # per-request state; times are rank 0's clock
    order: int  # admission order, breaks victim ties (newest first)
    first_seen: Optional[float]
    first_token_t: Optional[float] = None
    first_token_gen: int = 0
    gen: int = 0  # generated tokens at the last observation
    pauses: int = 0
    paused_t: Optional[float] = None  # set while paused
    paused_total: float = 0.0  # closed pauses
    resumed_at_iter: Optional[int] = None
    sacrificed: bool = False
    laxity_resume_t: Optional[float] = None


class GenPreemption:
    """Pause/resume bookkeeping and the two victim policies; ``step`` runs once
    per scheduling pass."""

    # Policy constants; D is the mean decode time, counts are in completions.
    slack_decode_frac = 0.15  # at-risk window at g = 1, x D
    gain_min, gain_max = 0.25, 2.0
    time_constant = 3000.0  # controller
    ewma_n = 500.0  # running means of D and TPOT
    target_frac = 0.8  # controller setpoint, x tpot_budget
    max_pause_frac, min_run_frac = 0.35, 0.07  # x D
    warmup_max_pause_s, warmup_min_run_s = 10.0, 2.0  # until the first completion
    max_per_request = 2
    hysteresis_iters = 64  # decode steps after a resume
    min_remaining_tokens = 32
    slo_eps, slo_credit_depth = 0.2, 3000.0  # share allowed below T; credit clamp, x eps
    laxity_min_frac, laxity_resume_frac = 0.02, 0.01  # x D
    laxity_max_pauses, below_target_max_pauses = 8, 3

    def __init__(self, settings: GenPreemptionSettings, max_num_active_requests: int) -> None:
        self.settings = settings
        self.slo = settings.target_output_tps is not None
        self.max_paused = paused_cap(max_num_active_requests)
        self._reqs: dict[int, _Req] = {}
        self._paused: dict = {}  # req_id -> request, in pause order
        self._resumed_ids: set[int] = set()
        self._order = 0
        self._now: Optional[float] = None
        self.theta = 0.0  # log g
        self.credit = 0.0  # sacrifices allowed, in completions
        self.mean_decode_s: Optional[float] = None  # D
        self.mean_tpot_s: Optional[float] = None
        self.stats = collections.Counter()  # since the last stat line
        self._last_log = 0.0

    @property
    def paused_ids(self) -> frozenset:
        return frozenset(self._paused)

    @property
    def resumed_ids(self) -> frozenset[int]:
        return frozenset(self._resumed_ids)

    @property
    def gain(self) -> float:
        return math.exp(self.theta)

    def _of_decode(self, frac: float, warmup: float) -> float:
        return warmup if self.mean_decode_s is None else frac * self.mean_decode_s

    @property
    def slack_s(self) -> float:
        return self.gain * self._of_decode(self.slack_decode_frac, 0.0)

    @property
    def max_pause_s(self) -> float:
        return self._of_decode(self.max_pause_frac, self.warmup_max_pause_s)

    @property
    def min_run_s(self) -> float:
        return self._of_decode(self.min_run_frac, self.warmup_min_run_s)

    def _paused_s(self, r: _Req) -> float:
        open_pause = 0.0 if r.paused_t is None else max(0.0, self._now - r.paused_t)
        return r.paused_total + open_pause

    def step(self, active_requests, now, capacity: int, num_active: int, waiting_queue) -> int:
        """Track, resume, then pause; returns the admission count (paused
        requests hold no slot)."""
        self._resumed_ids.clear()
        gen_queue = isinstance(waiting_queue, GenAdmissionWaitingQueue)
        self.observe(active_requests, now, gen_queue and waiting_queue.target_s is not None)
        num_active -= len(self._paused)
        num_active += self._resume(capacity - num_active)
        if num_active >= capacity and gen_queue:
            victim = self._victim(active_requests, waiting_queue.num_at_risk(self.slack_s))
            if victim is not None:
                self._pause(victim)
                num_active -= 1
        return num_active

    def scheduling_order(self, active_requests) -> list:
        """Give resumed requests the first available batch slots."""
        return sorted(active_requests, key=lambda req: req.py_request_id not in self._resumed_ids)

    def record_displaced(self, req) -> None:
        """Charge an unscheduled live decode interval as a bounded pause."""
        if req.py_request_id not in self._paused:
            self._pause(req)

    def observe(self, active_requests, now: Optional[float], target_ready: bool = False) -> None:
        """Track the active requests; account the ones that left.  Completions
        update the controller / credit only once the TTFT target is set."""
        self._now = now
        seen = set()
        for req in active_requests:
            rid = req.py_request_id
            seen.add(rid)
            r = self._reqs.get(rid)
            if r is None:
                r = self._reqs[rid] = _Req(order=self._order, first_seen=now)
                self._order += 1
            if r.first_seen is None:
                r.first_seen = now
            r.gen = generated_tokens(req)
            if (
                r.first_token_t is None
                and now is not None
                and req.state == LlmRequestState.GENERATION_IN_PROGRESS
            ):
                r.first_token_t, r.first_token_gen = now, r.gen
        for rid in [rid for rid in self._reqs if rid not in seen]:
            self._on_left(self._reqs.pop(rid), target_ready)
            self._paused.pop(rid, None)

    def _on_left(self, r: _Req, target_ready: bool) -> None:
        now = self._now
        if r.first_token_t is None or now is None:
            return
        paused = self._paused_s(r)
        tokens = r.gen - r.first_token_gen
        decode = now - r.first_token_t - paused
        if tokens < 1 or decode <= 0.0:
            return
        if self.mean_decode_s is None:
            self.mean_decode_s, self.mean_tpot_s = decode, decode / tokens
        else:
            self.mean_decode_s += (decode - self.mean_decode_s) / self.ewma_n
            self.mean_tpot_s += (decode / tokens - self.mean_tpot_s) / self.ewma_n
        if not target_ready:
            return
        inflation = paused / tokens / self.mean_tpot_s
        self.stats["scored"] += 1
        self.stats["infl"] += inflation
        if self.slo:
            below = tokens / (now - r.first_token_t) < self.settings.target_output_tps
            self.stats["below"] += below
            depth = self.slo_credit_depth * self.slo_eps
            credit = self.credit + r.sacrificed + self.slo_eps - below
            self.credit = min(depth, max(-depth, credit))
        else:
            target = self.target_frac * self.settings.tpot_budget
            theta = self.theta + (target - inflation) / (target * self.time_constant)
            self.theta = min(math.log(self.gain_max), max(math.log(self.gain_min), theta))

    def laxity(self, req) -> Optional[float]:
        """Seconds ``req`` can still be paused and finish at or above
        ``target_output_tps`` at its own unpaused rate (negative: below)."""
        r = self._reqs.get(req.py_request_id)
        if r is None or r.first_token_t is None or self._now is None:
            return None
        max_new, gen = req.py_max_new_tokens, generated_tokens(req)
        tokens = gen - r.first_token_gen
        unpaused = self._now - r.first_token_t - self._paused_s(r)
        if tokens > 0 and unpaused > 0.0:
            rate = tokens / unpaused
        else:
            rate = self.mean_tpot_s and 1.0 / self.mean_tpot_s
        if not rate:
            return None
        deadline = r.first_token_t + (max_new - r.first_token_gen) / self.settings.target_output_tps
        return deadline - (self._now + max(0, max_new - gen) / rate)

    def _resume_at(self, r: _Req) -> float:
        t = r.paused_t + self.max_pause_s
        return t if r.laxity_resume_t is None else min(t, r.laxity_resume_t)

    def _resume(self, free_slots: int) -> int:
        """Resume into the free slots, earliest due first (sacrificed requests
        last), and every request that is due regardless; returns the number
        resumed."""
        due_order = sorted(
            self._paused.items(),
            key=lambda e: (self._reqs[e[0]].sacrificed, self._resume_at(self._reqs[e[0]])),
        )
        n = 0
        for rid, req in due_order:
            r = self._reqs[rid]
            if n >= free_slots:
                if self._now < self._resume_at(r):
                    continue
                self.stats["forced"] += 1
            n += 1
            del self._paused[rid]
            self._resumed_ids.add(rid)
            r.resumed_at_iter = req.py_decoding_iter
            r.paused_total = self._paused_s(r)
            r.paused_t = r.laxity_resume_t = None
            self.stats["resumed"] += 1
        return n

    def _max_pauses(self) -> int:
        if self.slo:
            return max(self.laxity_max_pauses, self.below_target_max_pauses)
        return self.max_per_request

    def _key(self, req) -> tuple:
        """Longest resident first; ties: the newest."""
        r = self._reqs[req.py_request_id]
        return (r.first_seen, -r.order)

    def _eligible(self, req) -> bool:
        r = self._reqs[req.py_request_id]
        return (
            req.py_request_id not in self._paused
            and req.state == LlmRequestState.GENERATION_IN_PROGRESS
            and not req.is_dummy_request
            and req.py_decoding_iter >= 1
            and r.pauses < self._max_pauses()
            and r.first_token_t is not None
            and self._now is not None
            and self._now - r.first_token_t >= self.min_run_s
            and (
                r.resumed_at_iter is None
                or req.py_decoding_iter - r.resumed_at_iter >= self.hysteresis_iters
            )
            and req.py_max_new_tokens - generated_tokens(req) > self.min_remaining_tokens
        )

    def _victim(self, active_requests, num_at_risk: int) -> Optional[LlmRequest]:
        if num_at_risk <= 0 or len(self._paused) >= self.max_paused:
            return None
        candidates = [req for req in active_requests if self._eligible(req)]
        return (
            self._slo_victim(candidates)
            if self.slo
            else min(candidates, key=self._key, default=None)
        )

    def _slo_victim(self, candidates: list):
        """The largest laxity (>= 0.02 D), else the longest resident request
        already below the target, else a new sacrifice if the credit allows."""
        min_laxity = self.laxity_min_frac * (self.mean_decode_s or 0.0)
        lax, below, fresh = [], [], []
        for req in candidates:
            r = self._reqs[req.py_request_id]
            laxity = self.laxity(req)
            if r.sacrificed or (laxity is not None and laxity < 0.0):
                if r.pauses < self.below_target_max_pauses:
                    below.append(req)
            elif r.pauses < self.laxity_max_pauses:
                fresh.append(req)
                if laxity is not None and laxity >= min_laxity:
                    lax.append(((-laxity, self._key(req)), req))
        if lax:
            return min(lax, key=lambda e: e[0])[1]
        if below:
            return min(below, key=self._key)
        if fresh and self.credit >= 1.0:
            victim = min(fresh, key=self._key)
            self._reqs[victim.py_request_id].sacrificed = True
            self.credit -= 1.0
            return victim
        return None

    def _pause(self, req) -> None:
        r = self._reqs[req.py_request_id]
        if self.slo and not r.sacrificed:
            # Resume when the laxity is down to the margin (below target: no).
            laxity = self.laxity(req)
            if laxity is not None and laxity >= 0.0:
                margin = self.laxity_resume_frac * (self.mean_decode_s or 0.0)
                r.laxity_resume_t = self._now + laxity - margin
        self._paused[req.py_request_id] = req
        r.pauses += 1
        r.paused_t = self._now
        self.stats["paused"] += 1

    def stats_line(self) -> str:
        st, n = self.stats, self.stats["scored"]
        infl = f"{100.0 * st['infl'] / n:+.2f}%" if n else "na"
        decode = "na" if self.mean_decode_s is None else f"{self.mean_decode_s:.2f}s"
        line = (
            f"[gen_preempt] paused={st['paused']} resumed={st['resumed']} forced={st['forced']} "
            f"paused_now={len(self._paused)}/{self.max_paused} g={self.gain:.3f} "
            f"slack={self.slack_s:.2f}s decode={decode} tpot_infl={infl}"
        )
        if self.slo:
            line += f" below_target={st['below']}/{n} credit={self.credit:.1f}"
        st.clear()
        return line

    def maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < STATS_LOG_PERIOD_S:
            return
        self._last_log = now
        if +self.stats or self._paused:
            logger.info(self.stats_line())


def _unsupported(ex) -> Optional[str]:
    if not hasattr(ex.scheduler, "set_gen_paused_ids"):
        return "needs the KV cache manager V2 scheduler"
    if not ex._gen_admission.needs_clock:
        return "needs the auto GEN admission policy (0 < TLLM_GEN_ADMISSION_TTFT_PERCENTILE < 99)"
    if ex.enable_attention_dp:
        return "not supported with attention DP"
    from ..resource_manager import ResourceManagerType

    # Every pool with one entry per resident request must hold the paused ones.
    need = ex.max_num_active_requests + paused_cap(ex.max_num_active_requests)
    seq_slots = ex.resource_manager.resource_managers.get(ResourceManagerType.SEQ_SLOT_MANAGER)
    pools = [("sequence slot pool", seq_slots and seq_slots.get_max_resource_count())]
    for mgr in (ex.kv_cache_manager, ex.draft_kv_cache_manager):
        if mgr is not None:
            name = type(mgr).__name__
            pools.append((f"{name} index pool", getattr(mgr, "max_admissible_sequences", None)))
            if hasattr(mgr, "_max_resident_sequences"):
                pools.append((f"{name} recurrent state pool", mgr._max_resident_sequences()))
    for name, size in pools:
        if isinstance(size, int) and size < need:
            return f"the {name} holds {size} requests, fewer than active + paused cap = {need}"
    return None


def create_gen_preemption(executor, settings: Optional[GenPreemptionSettings]):
    """The executor's GenPreemption, or None (off, or unsupported: a warning)."""
    if settings is None or not settings.enabled:
        return None
    if executor.dist.has_pp:
        raise ValueError("GEN slot preemption is not supported with pipeline parallelism")
    reason = _unsupported(executor)
    if executor.dist.rank == 0:
        if reason is None:
            logger.info(f"[gen_preempt] GEN slot preemption (pause in place) enabled: {settings}")
        else:
            logger.warning(f"[gen_preempt] GEN slot preemption disabled: {reason}")
    return None if reason else GenPreemption(settings, executor.max_num_active_requests)
