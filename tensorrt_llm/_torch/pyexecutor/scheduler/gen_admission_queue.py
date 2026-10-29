# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GEN-side admission ordering for disaggregated serving.

Orders the GEN waiting queue by the original arrival at the disagg
orchestrator (``disaggregated_params.orig_elapsed_s``), configured by
``TLLM_GEN_ADMISSION_TTFT_PERCENTILE`` = P: 0 is FCFS, P >= 99 is oldest
original arrival first, and 0 < P < 99 is slack admission with a target X and
a guard G estimated online (``AdaptiveSlack``).

Every rank must pop the same requests, so all inputs are rank 0's: the
arrival stamps ride in the broadcast queue items, and its clock and ``s2f``
(admission -> first token time) ride with the request-count broadcast.
"""

import bisect
import collections
import itertools
import math
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Mapping, Optional

from tensorrt_llm.logger import logger

from .waiting_queue import WaitingQueue

GEN_ADMISSION_ENV = "TLLM_GEN_ADMISSION_TTFT_PERCENTILE"

STATS_LOG_PERIOD_S = 30.0

# Elapsed values outside [0, _MAX_ELAPSED_S] are treated as missing.
_MAX_ELAPSED_S = 86400.0

TIER_GUARD = "guard"
TIER_SLACK = "slack"
TIER_LATE = "late"
TIER_OLDEST = "oldest"
TIER_UNKEYED = "unkeyed"
TIER_WARMUP = "warmup"

OLDEST_FIRST_TTFT_PERCENTILE = 99.0


def work_quantile_for_ttft_percentile(percentile: float) -> float:
    """q = p - 5 * (1 - p)^2 with p = P / 100, clipped to [0.5, 0.99]."""
    # In percent units, so that P90 gives exactly 0.85.
    q = (100.0 * percentile - 5.0 * (100.0 - percentile) ** 2) / 10000.0
    return min(0.99, max(0.5, q))


@dataclass(frozen=True)
class AdaptiveSlackConfig:
    """Constants of the ``auto`` policy; only ``q`` depends on P (defaults: P90)."""

    q: float = 0.85  # quantile of the per-request work (TTFT - GEN wait)
    n: int = 2000  # rolling window and warm-up length (admissions)
    upd_s: float = 10.0  # X update period
    alpha: float = 0.02  # step toward the target per update
    cap_up: float = 1.0  # max X increase per update
    cap_dn: float = 0.2  # max X decrease per update
    freeze: float = 0.5  # hold X if < this share of the window's admissions queued
    queued_s: float = 0.1  # GEN wait above which an admission counts as queued
    margin: float = 0.5  # added to the estimate
    x_lo: float = 4.0
    x_hi: float = 60.0
    guard_k: float = 1.5  # G = X + guard_k * mean GEN wait
    warm_frac: float = 0.25  # warm start from the freshest n * warm_frac
    s2f_seed_s: float = 0.17  # admission -> first token before any sample
    s2f_alpha: float = 0.05  # EMA weight of a new admission -> first token sample


def parse_ttft_percentile(raw: Optional[str]) -> float:
    if raw is None or raw.strip() == "":
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not 0.0 <= value <= 100.0:  # also rejects nan
        raise ValueError(
            f"{GEN_ADMISSION_ENV}={raw!r} must be a number in [0, 100]: the TTFT "
            f"percentile to optimize (0: FCFS, >= 99: oldest first)"
        )
    return value


@dataclass(frozen=True)
class GenAdmissionConfig:
    """The TTFT percentile to optimize; ``auto`` is derived from it unless given."""

    ttft_percentile: float = 0.0
    auto: Optional[AdaptiveSlackConfig] = None

    def __post_init__(self):
        p = self.ttft_percentile
        if not 0.0 <= p <= 100.0:
            raise ValueError(f"GenAdmissionConfig.ttft_percentile={p!r} must be in [0, 100]")
        if self.policy == "auto" and self.auto is None:
            object.__setattr__(
                self, "auto", AdaptiveSlackConfig(q=work_quantile_for_ttft_percentile(p))
            )

    @property
    def policy(self) -> str:
        if self.ttft_percentile == 0.0:
            return "fcfs"
        if self.ttft_percentile >= OLDEST_FIRST_TTFT_PERCENTILE:
            return "oldest"
        return "auto"

    @property
    def enabled(self) -> bool:
        return self.policy != "fcfs"

    @property
    def needs_clock(self) -> bool:
        return self.policy == "auto"

    def describe(self) -> str:
        head = f"ttft_percentile={self.ttft_percentile:g} policy={self.policy}"
        if self.policy == "auto":
            a = self.auto
            return (
                f"{head} q={a.q:g} N={a.n} upd={a.upd_s:g}s "
                f"alpha={a.alpha:g} cap=+{a.cap_up:g}/-{a.cap_dn:g} freeze={a.freeze:g} "
                f"margin={a.margin:g} X=[{a.x_lo:g},{a.x_hi:g}] "
                f"guard_k={a.guard_k:g} s2f=EMA(seed {a.s2f_seed_s:g}s, alpha {a.s2f_alpha:g})"
            )
        return head

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "GenAdmissionConfig":
        env = os.environ if environ is None else environ
        return cls(ttft_percentile=parse_ttft_percentile(env.get(GEN_ADMISSION_ENV)))


def get_orig_elapsed_s(executor_request) -> Optional[float]:
    params = getattr(executor_request, "py_disaggregated_params", None)
    elapsed = getattr(params, "orig_elapsed_s", None)
    if elapsed is None:
        return None
    try:
        elapsed = float(elapsed)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(elapsed) or elapsed < 0.0 or elapsed > _MAX_ELAPSED_S:
        return None
    return elapsed


def stamp_admission_times(item, now: float) -> None:
    """Stamp GEN arrival and original arrival on a queue item (rank 0, at enqueue)."""
    item.admit_arrival = now
    elapsed = get_orig_elapsed_s(item.request) if item.request is not None else None
    item.admit_orig_arrival = None if elapsed is None else now - elapsed


def _quantile(values: Iterable[float], q: float) -> float:
    """Linear-interpolation quantile (``numpy.quantile`` default method)."""
    v = sorted(values)
    if not v:
        return float("nan")
    pos = q * (len(v) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def _mean(values) -> float:
    return math.fsum(values) / len(values) if values else 0.0


class FirstTokenTimer:
    """Rank 0's EMA of the admission -> first token time (``s2f``).

    Requests that never show a first token on rank 0 (another attention-DP
    rank, cancelled) expire after ``ttl_s``; samples above ``max_sample_s``
    (e.g. a retried KV transfer) are dropped.
    """

    def __init__(
        self,
        seed_s: float,
        alpha: float,
        max_sample_s: float = 30.0,
        ttl_s: float = 300.0,
        max_pending: int = 16384,
    ) -> None:
        self.value = seed_s
        self.n_samples = 0
        self._alpha = alpha
        self._max_sample_s = max_sample_s
        self._ttl_s = ttl_s
        self._max_pending = max_pending
        # req_id -> pop time, in pop order.
        self._pending: "collections.OrderedDict[int, float]" = collections.OrderedDict()

    def __len__(self) -> int:
        return len(self._pending)

    def on_pop(self, req_id: int, now: float) -> None:
        self._pending[req_id] = now
        self._pending.move_to_end(req_id)
        while len(self._pending) > self._max_pending:
            self._pending.popitem(last=False)

    def sample(self, dt: float) -> None:
        if not 0.0 <= dt <= self._max_sample_s:
            return
        self.value += self._alpha * (dt - self.value)
        self.n_samples += 1

    def observe(self, active_requests: Iterable, now: float) -> None:
        """Sample every pending request that has produced its first token."""
        if not self._pending:
            return
        pending = self._pending
        for req in active_requests:
            rid = getattr(req, "py_request_id", None)
            if rid in pending and getattr(req, "py_decoding_iter", 0) >= 1:
                self.sample(now - pending.pop(rid))
        while pending:
            rid, t_pop = next(iter(pending.items()))
            if now - t_pop <= self._ttl_s:
                break
            pending.popitem(last=False)


class AdaptiveSlack:
    """Online slack target X and guard G of the ``auto`` policy.

    Driven only by rank-consistent inputs, so the state is identical on every rank.
    """

    def __init__(self, cfg: AdaptiveSlackConfig) -> None:
        self.cfg = cfg
        self.s2f_s = cfg.s2f_seed_s
        self.obs_work: collections.deque = collections.deque(maxlen=cfg.n)
        self.obs_gw: collections.deque = collections.deque(maxlen=cfg.n)
        self.X: Optional[float] = None
        self.G: Optional[float] = None
        self.last_upd: Optional[float] = None
        self.upd_n = 0  # admissions since the last update
        self.upd_q = 0  # ... of which queued
        self.n_q = 0  # queued admissions overall (warm-up count)

    def _target(self, work, gw) -> float:
        return _quantile(work, self.cfg.q) + _mean(gw) + self.cfg.margin

    def on_clock(self, now: float) -> Optional[str]:
        """Update X and G if due; returns a log line when X was (re)computed."""
        c = self.cfg
        if self.last_upd is not None and now - self.last_upd < c.upd_s:
            return None
        self.last_upd = now
        upd_n, upd_q = self.upd_n, self.upd_q
        self.upd_n = self.upd_q = 0
        if self.n_q < c.n:
            return None
        tgt = self._target(self.obs_work, self.obs_gw)
        if self.X is None:
            # The early part of the window spans the start-up ramp.
            k = max(1, int(c.n * c.warm_frac))
            self.X = self._target(list(self.obs_work)[-k:], list(self.obs_gw)[-k:])
            kind = "warm_start"
        elif upd_n > 0 and upd_q / upd_n >= c.freeze:
            self.X += max(-c.cap_dn, min(c.cap_up, c.alpha * (tgt - self.X)))
            kind = "update"
        else:
            kind = "hold"
        self.X = max(c.x_lo, min(c.x_hi, self.X))
        self.G = self.X + c.guard_k * _mean(self.obs_gw)
        return (
            f"[gen_admission] auto {kind} t={now:.1f} X={self.X:.3f} G={self.G:.3f} "
            f"target={tgt:.3f} s2f={self.s2f_s:.3f} n={len(self.obs_work)} "
            f"n_queued={self.n_q} win_admits={upd_n} win_queued={upd_q} "
            f"held={int(kind == 'hold')}"
        )

    def on_admit(self, now: float, orig_arrival: float, gen_arrival: float) -> None:
        gw = now - gen_arrival
        self.obs_work.append(now - orig_arrival + self.s2f_s - gw)
        self.obs_gw.append(gw)
        queued = int(gw > self.cfg.queued_s)
        self.upd_n += 1
        self.upd_q += queued
        self.n_q += queued

    def describe_state(self) -> str:
        if self.X is None:
            return f"X=warmup({self.n_q}/{self.cfg.n}) G=none s2f={self.s2f_s:.3f}"
        return f"X={self.X:.2f} G={self.G:.2f} s2f={self.s2f_s:.3f}"


class GenAdmissionWaitingQueue(WaitingQueue):
    """Waiting queue ordered by original arrival, with optional slack/guard tiers.

    Entries are sorted by ``(orig_arrival, seq)``; prepended requests get a
    negative ``seq`` so they sort ahead of every request with the same key.
    The admission order only changes with the clock or the entries, so it is
    built once and popped from until then: a pass that pops and puts back
    many requests (attention-DP rank full) does not rescan per pop.
    """

    def __init__(self, config: GenAdmissionConfig) -> None:
        if not config.enabled:
            raise ValueError("GenAdmissionWaitingQueue needs 0 < ttft_percentile <= 100")
        self.config = config
        self.adaptive: Optional[AdaptiveSlack] = (
            AdaptiveSlack(config.auto) if config.policy == "auto" else None
        )
        self.first_token_timer: Optional[FirstTokenTimer] = None
        self._log = False
        self._last_stats_log = -math.inf
        # Sorted list of (orig_arrival, seq, gen_arrival, keyed, item).
        self._entries: list[tuple] = []
        # Admission order of the entries as (entry, tier), reversed; None when stale.
        self._order: Optional[list[tuple]] = None
        # seqs popped from _order but not yet dropped from _entries.
        self._popped: set[int] = set()
        self._counter = itertools.count()
        self._prepend_counter = itertools.count(-1, -1)
        self._now: Optional[float] = None
        # Key for never-stamped items: the latest key seen, the same on every rank.
        self._last_key = 0.0
        self.admitted_by_tier: collections.Counter = collections.Counter()

    def enable_rank0(self) -> Optional[FirstTokenTimer]:
        """Log on this rank and, for ``auto``, time admission -> first token."""
        self._log = True
        logger.info(
            f"[gen_admission] GEN admission ordering enabled: {self.config.describe()} "
            f"(requests without disaggregated_params.orig_elapsed_s are FCFS by GEN arrival)"
        )
        if self.adaptive is not None and self.first_token_timer is None:
            a = self.config.auto
            self.first_token_timer = FirstTokenTimer(a.s2f_seed_s, a.s2f_alpha)
        return self.first_token_timer

    def set_now(self, now: Optional[float], s2f_s: Optional[float] = None) -> None:
        """Set rank 0's clock and s2f (None keeps it); once per iteration, before the pops."""
        self._now = now
        self._order = None
        if self.adaptive is not None and s2f_s is not None:
            self.adaptive.s2f_s = s2f_s
        if now is None:
            return
        if self.adaptive is not None:
            line = self.adaptive.on_clock(now)
            if line and self._log:
                logger.info(line)
        if self._log and now - self._last_stats_log >= STATS_LOG_PERIOD_S:
            self._last_stats_log = now
            logger.info(self.stats_line(now))

    def _make_entry(self, item, seq: int) -> tuple:
        gen_arrival = item.admit_arrival
        if gen_arrival is None:
            gen_arrival = self._last_key
            item.admit_arrival = gen_arrival
        orig = item.admit_orig_arrival
        keyed = orig is not None
        if not keyed:
            orig = gen_arrival
        self._last_key = max(self._last_key, gen_arrival)
        return (orig, seq, gen_arrival, keyed, item)

    def _compact(self) -> None:
        if self._popped:
            self._entries = [e for e in self._entries if e[1] not in self._popped]
            self._popped.clear()

    def _insert(self, entry: tuple) -> None:
        self._compact()
        self._order = None
        bisect.insort(self._entries, entry, key=lambda e: (e[0], e[1]))

    def _build_order(self) -> list[tuple]:
        """(entry, tier) in admission order: each one is the pick once those before it are gone."""
        entries = self._entries
        ada = self.adaptive
        now = self._now
        if ada is None:
            return [(e, TIER_OLDEST) for e in entries]
        if now is None:
            return [(e, TIER_LATE) for e in entries]
        if ada.X is None:
            return [(e, TIER_WARMUP) for e in sorted(entries, key=lambda e: (e[2], e[1]))]
        # Guard (GEN wait >= G), then feasible, then late; each in original-arrival order.
        deadline = ada.X - ada.s2f_s
        guard, slack, late = [], [], []
        for e in entries:
            orig, _, gen_arrival, keyed, _ = e
            if now - gen_arrival >= ada.G:
                guard.append((e, TIER_GUARD))
            elif not keyed:
                slack.append((e, TIER_UNKEYED))
            elif now - orig <= deadline:
                slack.append((e, TIER_SLACK))
            else:
                late.append((e, TIER_LATE))
        return guard + slack + late

    def _next(self) -> tuple:
        if self._order is None:
            self._compact()
            self._order = self._build_order()[::-1]
        return self._order[-1]

    def mark_admitted(self, items: Iterable) -> None:
        now = self._now
        for item in items:
            tier = item._gen_admission_tier
            self.admitted_by_tier[tier] += 1
            gen_arrival, orig = item.admit_arrival, item.admit_orig_arrival
            keyed = orig is not None
            if keyed and now is not None:
                if self.adaptive is not None and not getattr(
                    item, "_gen_admission_observed", False
                ):
                    item._gen_admission_observed = True
                    self.adaptive.on_admit(now, orig, gen_arrival)
                if self.first_token_timer is not None:
                    self.first_token_timer.on_pop(item.id, now)
            if self._log and logger.is_debug_enabled():
                orig_age = f"{now - orig:.3f}" if keyed and now is not None else "na"
                gen_wait = f"{now - gen_arrival:.3f}" if now is not None else "na"
                logger.debug(
                    f"[gen_admission] admit req_id={item.id} tier={tier} "
                    f"orig_age_s={orig_age} gen_wait_s={gen_wait}"
                )

    def add_request(self, request) -> None:
        self._insert(self._make_entry(request, next(self._counter)))

    def add_requests(self, requests: Iterable) -> None:
        for request in requests:
            self.add_request(request)

    def pop_request(self):
        if not self:
            raise IndexError("pop from an empty queue")
        entry, tier = self._next()
        self._order.pop()
        self._popped.add(entry[1])
        item = entry[4]
        item._gen_admission_tier = tier
        return item

    def peek_request(self):
        if not self:
            raise IndexError("peek from an empty queue")
        return self._next()[0][4]

    def peek_requests(self, k: int) -> list:
        if k <= 0 or not self:
            return []
        self._next()  # builds the admission order
        return [entry[4] for entry, _ in self._order[: -k - 1 : -1]]

    def admission_groups(self, requests: list) -> list[int]:
        """One group per request: the admission order is kept."""
        return list(range(len(requests)))

    def prepend_request(self, request) -> None:
        self._insert(self._make_entry(request, next(self._prepend_counter)))

    def prepend_requests(self, requests: Iterable) -> None:
        # Reversed: the first request gets the most negative seq.
        for request in reversed(list(requests)):
            self.prepend_request(request)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        self._compact()
        self._order = None
        self._entries = [e for e in self._entries if e[4].id not in request_ids]

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return len(self._entries) - len(self._popped)

    def __iter__(self) -> Iterator:
        self._compact()
        return iter([e[4] for e in self._entries])

    def stats_line(self, now: float) -> str:
        self._compact()
        parts = [
            f"[gen_admission] p{self.config.ttft_percentile:g} policy={self.config.policy} "
            f"qlen={len(self._entries)}"
        ]
        if self.adaptive is not None:
            parts.append(self.adaptive.describe_state())
        if self.first_token_timer is not None:
            parts.append(f"s2f_samples={self.first_token_timer.n_samples}")
        if self._entries:
            oldest_gen_wait = now - min(e[2] for e in self._entries)
            keyed = [e[0] for e in self._entries if e[3]]
            parts.append(f"oldest_gen_wait={oldest_gen_wait:.2f}s")
            if keyed:
                parts.append(f"oldest_orig_age={now - min(keyed):.2f}s")
            parts.append(f"keyed={len(keyed)}")
        tiers = ",".join(f"{k}={v}" for k, v in sorted(self.admitted_by_tier.items()))
        parts.append(f"admitted_since_last={{{tiers}}}")
        self.admitted_by_tier.clear()
        return " ".join(parts)
