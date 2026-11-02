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
"""GEN-side slot preemption for disaggregated serving.

On a disagg GEN worker a request holds one of ``max_num_active_requests``
slots from admission on.  When every slot is taken and the request the GEN
admission queue would admit next is about to miss the TTFT target (or already
has), one running decode request is paused in place so the waiting request
can take its slot.

A paused request stays resident: its KV, KDA/Mamba state, draft KV and
sequence slot stay where they are, it is only left out of the scheduled batch,
so a pause costs no copy and no extra step.  Resident requests (running plus
paused) may therefore exceed ``max_batch_size`` by up to the paused cap; the
pools that hold one entry per resident request are sized for that (see
``resident_headroom``), while the scheduled batch, and so every CUDA graph,
stays within ``max_batch_size``.  A paused request resumes ahead of every new
admission once a slot frees up, or after the maximum pause regardless.

Configuration (``scheduler_config.gen_preemption``, at most one of the two;
neither leaves preemption off): ``tpot_budget`` B, the allowed mean-TPOT
increase from pauses (0.10 = +10%), or ``target_output_tps`` T, a
per-request output rate (output tokens/s, accepted draft tokens included, so
1 / TPOT) to keep, which is converted to B = max(0, R0 / T - 1)
(``GenPreemptionSettings.budget``): requests running faster than T donate
slot time to waiters at risk of missing the TTFT target.  The budget only
sets the pause controller's target; everything else is derived from
measurements of completed requests (rank-identical: they are the requests
leaving the active list, timed on rank 0's clock):

* R0 = 1 / (running mean TPOT of completed requests, pauses excluded) is the
  unpaused per-request output rate, and D their running mean decode time
  (first token to finish, pauses excluded), both over 500 completions;
* with a zero budget (R0 <= T, or T set and no completion yet) nothing is
  paused;
* at-risk window (slack against X) = g x 0.15 x D;
* g in [0.25, 2] is an integral controller, one update per completed request
  once X is set, time constant 3000 completions, that holds the mean TPOT
  inflation from pauses (pause per output token / mean unpaused TPOT) at
  0.8 x B;
* victims: the longest resident request first (ties: the newest), see
  ``GenPreemption.victim_key``;
* paused cap = 0.375 x ``max_num_active_requests`` (also the pool headroom);
* forced resume after 0.35 D, preemptible after 0.07 D of decoding;
* at most 2 pauses per request and 1 per scheduling pass.

Until the first request completes the window is 0 and the forced resume /
minimum run fall back to 10 s / 2 s; g starts at 1.

Preemption needs the ``auto`` GEN admission policy, whose target X and rank
0's admission -> first token estimate decide when a waiting request is at
risk.  Every input to a decision is identical on all ranks of the
model-parallel group (helix included): the active request list, token and
decode-iteration counts, the waiting queue (broadcast) and rank 0's clock
(broadcast with the request-count probe).  The controller adds only
completions (requests leaving the active list) and their token counts, timed
on the same clock, so it needs no collective either.

Pipeline parallelism and attention DP are rejected when the LLM arguments
are validated (and pipeline parallelism again at executor setup).
"""

import collections
import math
from dataclasses import dataclass
from typing import Iterable, Optional

from tensorrt_llm.logger import logger

from ..llm_request import LlmRequestState

# EMA weight of a new measured pause (stats line only).
_PAUSE_EMA_ALPHA = 0.1


@dataclass(frozen=True)
class GenPreemptionSettings:
    """Target and internal parameters.

    Only ``target_output_tps`` comes from the user
    (``scheduler_config.gen_preemption``); the rest are constants of the
    policy, fields only so that tests can construct other settings.
    """

    # Allowed mean-TPOT increase from pauses (0.10 = +10%).
    tpot_budget: Optional[float] = None
    # Per-request output tokens/s to keep, converted to a TPOT budget.
    target_output_tps: Optional[float] = None
    # At-risk window at g = 1, in mean decode times.
    slack_decode_frac: float = 0.15
    gain_min: float = 0.25
    gain_max: float = 2.0
    # Integrator time constant, in completed requests.
    time_constant: float = 3000.0
    # Running means of the decode time and TPOT, in completed requests.
    ewma_n: float = 500.0
    # Controller setpoint, as a fraction of the TPOT budget.
    target_frac: float = 0.8
    # Paused cap, as a fraction of max_num_active_requests.
    cap_frac: float = 0.375
    # Forced resume and minimum decode before a pause, in mean decode times.
    max_pause_frac: float = 0.35
    min_run_frac: float = 0.07
    # Forced resume and minimum run until the first request completes.
    warmup_max_pause_s: float = 10.0
    warmup_min_run_s: float = 2.0
    # The next pop also triggers when it is already late (slack < 0) or popped
    # by another tier (guard, breaker); its slack must still be within the
    # window.
    late_trigger: bool = True
    # Pauses over a request's lifetime and per scheduling pass.
    max_per_request: int = 2
    max_per_pass: int = 1
    # Decode steps after a resume before a request is preemptible again.
    hysteresis_iters: int = 64
    # Requests this close to max_new_tokens finish soon enough on their own.
    min_remaining_tokens: int = 32

    def __post_init__(self):
        if self.tpot_budget is not None and self.target_output_tps is not None:
            raise ValueError("gen_preemption: set tpot_budget or target_output_tps, not both")
        if self.tpot_budget is not None and not 0.0 < self.tpot_budget < 1.0:
            raise ValueError(
                f"gen_preemption tpot_budget must be in (0, 1), got {self.tpot_budget}"
            )
        if self.target_output_tps is not None and not self.target_output_tps > 0.0:
            raise ValueError(
                f"gen_preemption target_output_tps must be > 0, got {self.target_output_tps}"
            )
        if not 0.0 < self.gain_min <= self.gain_max:
            raise ValueError("gen_preemption: need 0 < gain_min <= gain_max")

    @classmethod
    def from_config(cls, config) -> "GenPreemptionSettings":
        """From ``scheduler_config.gen_preemption`` (None: off)."""
        if config is None:
            return cls()
        budget, target = config.tpot_budget, config.target_output_tps
        return cls(
            tpot_budget=None if budget is None else float(budget),
            target_output_tps=None if target is None else float(target),
        )

    @property
    def enabled(self) -> bool:
        return self.tpot_budget is not None or self.target_output_tps is not None

    def budget(self, unpaused_tps: Optional[float]) -> float:
        """Allowed mean-TPOT increase from pauses: ``tpot_budget``, or for
        ``target_output_tps`` T, max(0, R0 / T - 1) with R0 the measured
        unpaused per-request output rate (0 until it is measured)."""
        if self.tpot_budget is not None:
            return self.tpot_budget
        if self.target_output_tps is None or unpaused_tps is None:
            return 0.0
        return max(0.0, unpaused_tps / self.target_output_tps - 1.0)

    def paused_cap(self, max_num_active_requests: int) -> int:
        """Requests paused at once (at least 1)."""
        return max(1, int(round(self.cap_frac * max_num_active_requests)))

    def describe(self) -> str:
        if self.tpot_budget is not None:
            budget = f"tpot_budget={self.tpot_budget}"
        else:
            budget = f"target_output_tps={self.target_output_tps} (budget R0/target-1)"
        return (
            f"{budget} setpoint={self.target_frac}xbudget "
            f"window={self.slack_decode_frac}xD x g[{self.gain_min}, {self.gain_max}] "
            f"cap={self.cap_frac}xmax_num_active_requests "
            f"max_pause={self.max_pause_frac}xD min_run={self.min_run_frac}xD "
            f"max_per_request={self.max_per_request}"
        )


# The settings of the executor being created in this process; the pools that
# reserve resident headroom are sized from them (set_active_settings).
_active_settings = GenPreemptionSettings()


def set_active_settings(settings: GenPreemptionSettings) -> None:
    """Install the settings that ``resident_headroom`` sizes pools for.

    Called once per executor creation, before any pool is sized, from
    ``scheduler_config.gen_preemption``.
    """
    global _active_settings
    _active_settings = settings


def get_active_settings() -> GenPreemptionSettings:
    return _active_settings


def resident_headroom(max_batch_size: int, settings: Optional[GenPreemptionSettings] = None) -> int:
    """Resident requests above ``max_batch_size`` that preemption may hold.

    Paused requests keep their sequence slot, KV index slot and recurrent
    state slot while the requests that replaced them run, so every pool with
    one entry per resident request reserves this many more.  Every rank
    derives the same value from the same LLM arguments.
    """
    settings = _active_settings if settings is None else settings
    return settings.paused_cap(max_batch_size) if settings.enabled else 0


def generated_tokens(req) -> int:
    """Tokens generated so far (rank-identical, also under helix)."""
    return max(0, req.get_num_tokens(0) - req.prompt_len)


@dataclass
class _ReqTrack:
    """Per-request state, keyed by request id."""

    order: int  # admission order (first observed)
    first_seen: Optional[float]  # rank 0 clock at first observation
    # Rank 0 clock and generated tokens when first seen generating.
    first_token_t: Optional[float] = None
    first_token_gen: int = 0
    pauses: int = 0
    paused_t: Optional[float] = None
    resumed_at_iter: Optional[int] = None
    # Time spent paused (closed pauses) and generated tokens at the last
    # observation.
    paused_total: float = 0.0
    last_gen: int = 0


class GenPreemption:
    """Rank-identical pause bookkeeping, victim selection and the window
    controller.

    The executor calls, once per scheduling pass and in this order,
    ``observe`` with the active requests (the same on every rank),
    ``resume_candidates`` / ``on_resumed`` for the resume lane, then
    ``select_victims`` / ``on_paused``.  Paused requests stay in the
    executor's active requests; the scheduler skips them and the admission
    count excludes them.
    """

    def __init__(self, settings: GenPreemptionSettings, max_num_active_requests: int) -> None:
        self.settings = settings
        self.max_paused = settings.paused_cap(max_num_active_requests)
        self._tracks: dict[int, _ReqTrack] = {}
        self._order = 0
        # req_id -> request, in pause order.
        self._paused: "collections.OrderedDict[int, object]" = collections.OrderedDict()
        self._now: Optional[float] = None
        # EMA of measured pauses (pause -> resume, rank 0 clock).
        self.mean_pause_s: Optional[float] = None
        # Counters since the last stats line, and overall.
        self.window: collections.Counter = collections.Counter()
        self.totals: collections.Counter = collections.Counter()
        # Controller state: log g, the running means of the unpaused decode
        # time and TPOT of completed requests, and the TPOT inflation of the
        # completions since the last stats line.
        self.theta = 0.0
        self.mean_decode_s: Optional[float] = None
        self.mean_tpot_s: Optional[float] = None
        self._infl_sum = 0.0
        self._infl_n = 0

    # ---- state ----

    @property
    def num_paused(self) -> int:
        return len(self._paused)

    @property
    def paused_ids(self) -> frozenset:
        return frozenset(self._paused)

    def is_paused(self, req_id: int) -> bool:
        return req_id in self._paused

    def num_pauses(self, req_id: int) -> int:
        track = self._tracks.get(req_id)
        return track.pauses if track is not None else 0

    def _since_first_token(self, req) -> Optional[float]:
        track = self._tracks.get(req.py_request_id)
        if track is None or track.first_token_t is None or self._now is None:
            return None
        return self._now - track.first_token_t

    # ---- window, limits and their controller ----

    @property
    def gain(self) -> float:
        return math.exp(self.theta)

    @property
    def unpaused_tps(self) -> Optional[float]:
        """R0: per-request output rate without pauses (1 / mean TPOT)."""
        return None if not self.mean_tpot_s else 1.0 / self.mean_tpot_s

    @property
    def tpot_budget(self) -> float:
        return self.settings.budget(self.unpaused_tps)

    @property
    def slack_s(self) -> float:
        """At-risk window for the next pop's slack against X."""
        if self.mean_decode_s is None:
            return 0.0
        return self.gain * self.settings.slack_decode_frac * self.mean_decode_s

    @property
    def max_pause_s(self) -> float:
        if self.mean_decode_s is None:
            return self.settings.warmup_max_pause_s
        return self.settings.max_pause_frac * self.mean_decode_s

    @property
    def min_run_s(self) -> float:
        if self.mean_decode_s is None:
            return self.settings.warmup_min_run_s
        return self.settings.min_run_frac * self.mean_decode_s

    def _on_left(self, track: _ReqTrack, target_ready: bool) -> None:
        """Account a request that finished or was cancelled."""
        s = self.settings
        if track.first_token_t is None or self._now is None:
            return
        paused = track.paused_total
        if track.paused_t is not None:
            paused += max(0.0, self._now - track.paused_t)
        tokens = track.last_gen - track.first_token_gen
        decode = self._now - track.first_token_t - paused
        if tokens < 1 or decode <= 0.0:
            return
        tpot = decode / tokens
        if self.mean_decode_s is None:
            self.mean_decode_s, self.mean_tpot_s = decode, tpot
        else:
            self.mean_decode_s += (decode - self.mean_decode_s) / s.ewma_n
            self.mean_tpot_s += (tpot - self.mean_tpot_s) / s.ewma_n
        self.window["completed"] += 1
        target = s.target_frac * self.tpot_budget
        if not target_ready or target <= 0.0:
            return
        # This request's TPOT inflation from its pauses; integrate its error
        # against the setpoint, one step per completion.
        inflation = (paused / tokens) / self.mean_tpot_s
        theta = self.theta + (target - inflation) / (target * s.time_constant)
        self.theta = min(math.log(s.gain_max), max(math.log(s.gain_min), theta))
        self._infl_sum += inflation
        self._infl_n += 1

    def observe(
        self, active_requests: Iterable, now: Optional[float], target_ready: bool = False
    ) -> None:
        """Track the active requests; account and forget the ones that left.

        ``target_ready``: the admission queue has its TTFT target X; the
        controller integrates only from then on (rank-identical).
        """
        self._now = now
        seen = set()
        for req in active_requests:
            rid = req.py_request_id
            seen.add(rid)
            track = self._tracks.get(rid)
            if track is None:
                track = _ReqTrack(order=self._order, first_seen=now)
                self._order += 1
                self._tracks[rid] = track
            if track.first_seen is None:
                track.first_seen = now
            if (
                track.first_token_t is None
                and now is not None
                and req.state == LlmRequestState.GENERATION_IN_PROGRESS
            ):
                track.first_token_t = now
                track.first_token_gen = generated_tokens(req)
            track.last_gen = generated_tokens(req)
        for rid in [rid for rid in self._tracks if rid not in seen]:
            self._on_left(self._tracks.pop(rid), target_ready)
        for rid in [rid for rid in self._paused if rid not in seen]:
            # Cancelled or failed while paused.
            del self._paused[rid]
            self.window["dropped"] += 1

    # ---- resume lane ----

    def resume_candidates(self, free_slots: int) -> list:
        """Paused requests to resume, oldest pause first: as many as there are
        free slots, plus every one paused for ``max_pause_s`` or longer."""
        out = []
        max_pause_s = self.max_pause_s
        for req in self._paused.values():
            if len(out) < free_slots:
                out.append(req)
                continue
            paused_t = self._tracks[req.py_request_id].paused_t
            if self._now is not None and self._now - paused_t >= max_pause_s:
                out.append(req)
                self.window["forced_resumes"] += 1
        return out

    def on_resumed(self, req) -> None:
        rid = req.py_request_id
        self._paused.pop(rid, None)
        track = self._tracks.get(rid)
        if track is not None:
            track.resumed_at_iter = req.py_decoding_iter
            if track.paused_t is not None and self._now is not None:
                pause = max(0.0, self._now - track.paused_t)
                track.paused_total += pause
                self.mean_pause_s = (
                    pause
                    if self.mean_pause_s is None
                    else self.mean_pause_s + _PAUSE_EMA_ALPHA * (pause - self.mean_pause_s)
                )
            track.paused_t = None
        self.window["resumed"] += 1
        self.totals["resumed"] += 1

    # ---- pausing ----

    def is_eligible(self, req) -> bool:
        s = self.settings
        rid = req.py_request_id
        if rid in self._paused or req.state != LlmRequestState.GENERATION_IN_PROGRESS:
            return False
        if req.is_dummy_request or getattr(req, "is_attention_dp_dummy", False):
            return False
        if req.py_decoding_iter < 1:
            return False
        track = self._tracks.get(rid)
        if track is None or track.pauses >= s.max_per_request:
            return False
        since_first_token = self._since_first_token(req)
        if since_first_token is None or since_first_token < self.min_run_s:
            return False
        if (
            track.resumed_at_iter is not None
            and req.py_decoding_iter - track.resumed_at_iter < s.hysteresis_iters
        ):
            return False
        max_new = getattr(req, "py_max_new_tokens", None)
        if max_new is not None and max_new - generated_tokens(req) <= s.min_remaining_tokens:
            return False
        return True

    def victim_key(self, req) -> tuple:
        """Victim order, lowest first: ``wait``, the longest resident request
        (its first token is long out, it is the furthest from any TTFT
        deadline); ties go to the newest."""
        track = self._tracks[req.py_request_id]
        if self._now is None or track.first_seen is None:
            residence = float(req.py_decoding_iter)
        else:
            residence = self._now - track.first_seen
        return (-residence, -track.order)

    def victim_order(self, active_requests: Iterable) -> list:
        """Eligible requests, cheapest to pause first."""
        eligible = [req for req in active_requests if self.is_eligible(req)]
        return sorted(eligible, key=self.victim_key)

    def select_victims(self, active_requests: Iterable, num_at_risk: int) -> list:
        """Up to ``num_at_risk`` victims within the per-pass and paused-count
        budgets."""
        if self.tpot_budget <= 0.0:
            return []  # R0 <= target (or not measured yet): no TPOT to spend
        budget = min(num_at_risk, self.settings.max_per_pass, self.max_paused - self.num_paused)
        if budget <= 0:
            return []
        victims = self.victim_order(active_requests)[:budget]
        if not victims:
            self.window["no_victim"] += 1
        return victims

    def on_paused(self, req) -> None:
        rid = req.py_request_id
        self._paused[rid] = req
        track = self._tracks[rid]
        track.pauses += 1
        track.paused_t = self._now
        self.window["paused"] += 1
        self.totals["paused"] += 1

    # ---- logging ----

    def stats_line(self) -> str:
        w = self.window
        mean_pause = "na" if self.mean_pause_s is None else f"{self.mean_pause_s:.2f}s"
        decode = "na" if self.mean_decode_s is None else f"{self.mean_decode_s:.2f}s"
        r0 = "na" if self.unpaused_tps is None else f"{self.unpaused_tps:.2f}"
        infl = "na" if not self._infl_n else f"{100.0 * self._infl_sum / self._infl_n:+.2f}%"
        line = (
            f"[gen_preempt] paused={w['paused']} "
            f"resumed={w['resumed']} forced_resumes={w['forced_resumes']} "
            f"dropped={w['dropped']} no_victim={w['no_victim']} "
            f"paused_now={self.num_paused}/{self.max_paused} "
            f"mean_pause={mean_pause} total_paused={self.totals['paused']} "
            f"total_resumed={self.totals['resumed']} "
            f"g={self.gain:.3f} slack={self.slack_s:.2f}s decode={decode} "
            f"unpaused_tps={r0} budget={100.0 * self.tpot_budget:.2f}% "
            f"tpot_infl={infl} completed={w['completed']} "
            f"max_pause={self.max_pause_s:.2f}s min_run={self.min_run_s:.2f}s"
        )
        self._infl_sum = 0.0
        self._infl_n = 0
        self.window.clear()
        return line

    @property
    def has_activity(self) -> bool:
        return bool(+self.window) or self.num_paused > 0


def log_gen_preemption_settings(
    settings: GenPreemptionSettings, disabled_reason: Optional[str] = None
) -> None:
    if not settings.enabled:
        return
    if disabled_reason is not None:
        logger.warning(f"[gen_preempt] GEN slot preemption disabled: {disabled_reason}")
    else:
        logger.info(
            f"[gen_preempt] GEN slot preemption (pause in place) enabled: {settings.describe()}"
        )
