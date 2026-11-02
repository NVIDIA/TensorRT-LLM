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
1 / TPOT) used as a service-level objective.  Everything else is derived from
measurements of completed requests (rank-identical: they are the requests
leaving the active list, timed on rank 0's clock):

* D is the running mean decode time of completed requests (first token to
  finish, pauses excluded) and R0 = 1 / their running mean TPOT (pauses
  excluded), both over 500 completions;
* paused cap = 0.375 x ``max_num_active_requests`` (also the pool headroom);
* forced resume after 0.35 D, preemptible after 0.07 D of decoding;
* 1 pause per scheduling pass.

``tpot_budget``: the TPOT-budget controller (k3-disagg-sim phase 21).

* at-risk window (slack against X) = g x 0.15 x D;
* g in [0.25, 2] is an integral controller, one update per completed request
  once X is set, time constant 3000 completions, that holds the mean TPOT
  inflation from pauses (pause per output token / mean unpaused TPOT) at
  0.8 x B;
* victims: the longest resident request first (ties: the newest), see
  ``GenPreemption.victim_key``; at most 2 pauses per request;
* the resume lane takes the oldest pause first.

``target_output_tps``: T as a per-request floor that at least 1 - eps of the
requests meet (eps = 0.2, internal; k3-disagg-sim phase 24).  A request's
deadline is its first token time + (max_tokens - 1) / T, and its laxity L is
the deadline minus its projected finish at its own unpaused rate::

    L = t_first + (max_tokens - 1) / T - (now + remaining_tokens / r),
    r = tokens since the first token / (time since the first token - pauses)

(``max_tokens - 1`` counts the tokens after the first one; on a GEN worker the
first token arrives with the request).

* at-risk window = 0.15 x D (no controller);
* victims, in this order:

  1. the largest laxity among requests with L >= 0.02 D and fewer than 8
     pauses; such a victim is resumed when its laxity is down to 0.01 D (or
     after 0.35 D), so it still finishes at or above T;
  2. else a request that is already below T (L < 0, or sacrificed earlier)
     and has fewer than 3 pauses, the longest resident first: it costs the
     objective nothing more;
  3. else, while the credit is at least 1, a new sacrifice: the longest
     resident request (credit -= 1); a sacrificed or below-T victim resumes
     only after 0.35 D or when a slot frees.

* the credit is a token bucket in completions: each completion after X adds
  eps, minus 1 if it finished below T, plus 1 if it was sacrificed (so a
  sacrifice is counted once), clamped to +-3000 eps.  Credit saved in quiet
  periods is spent in load waves;
* the resume lane takes the earliest forced-resume time first, i.e. the
  least laxity; sacrificed and below-T victims, which only have the maximum
  pause, mostly come last.

Until the first request completes the window is 0 and the forced resume /
minimum run fall back to 10 s / 2 s; g starts at 1.

Preemption needs the ``auto`` GEN admission policy, whose target X and rank
0's admission -> first token estimate decide when a waiting request is at
risk.  Every input to a decision is identical on all ranks of the
model-parallel group (helix included): the active request list, token and
decode-iteration counts, the waiting queue (broadcast) and rank 0's clock
(broadcast with the request-count probe).  The controller and the
objective add only completions (requests leaving the active list), their
token counts and each request's ``max_tokens``, timed on the same clock, so
they need no collective either.

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

    Only ``tpot_budget`` / ``target_output_tps`` come from the user
    (``scheduler_config.gen_preemption``); the rest are constants of the
    policy, fields only so that tests can construct other settings.
    """

    # Allowed mean-TPOT increase from pauses (0.10 = +10%).
    tpot_budget: Optional[float] = None
    # Per-request output tokens/s floor, a service-level objective met by at
    # least 1 - slo_eps of the requests.
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
    # target_output_tps only.  Share of requests allowed below the target:
    # the sacrifice credit gains slo_eps per completion (after X), loses 1 per
    # completion below the target, and is clamped to +-slo_credit_depth x
    # slo_eps (a token bucket in completions).
    slo_eps: float = 0.2
    slo_credit_depth: float = 3000.0
    # Laxity victims need at least laxity_min_frac x D of laxity and resume
    # when it is down to laxity_resume_frac x D.
    laxity_min_frac: float = 0.02
    laxity_resume_frac: float = 0.01
    # Pauses per request: as a laxity victim (or new sacrifice), and as a
    # request already below the target.
    laxity_max_pauses: int = 8
    below_target_max_pauses: int = 3

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

    @property
    def output_tps_slo(self) -> bool:
        """``target_output_tps`` is set: laxity victims and the sacrifice
        credit instead of the TPOT-budget controller."""
        return self.target_output_tps is not None

    @property
    def max_pauses(self) -> int:
        """Pauses over a request's lifetime, any victim class."""
        if self.output_tps_slo:
            return max(self.laxity_max_pauses, self.below_target_max_pauses)
        return self.max_per_request

    def paused_cap(self, max_num_active_requests: int) -> int:
        """Requests paused at once (at least 1)."""
        return max(1, int(round(self.cap_frac * max_num_active_requests)))

    def describe(self) -> str:
        if self.output_tps_slo:
            policy = (
                f"target_output_tps={self.target_output_tps} "
                f"(floor for >= {1.0 - self.slo_eps:.0%} of requests: laxity victims "
                f">= {self.laxity_min_frac}xD resumed at {self.laxity_resume_frac}xD, "
                f"then below-target victims, then sacrifices on a credit of "
                f"{self.slo_eps}/completion clamped to +-{self.slo_credit_depth:.0f}x"
                f"{self.slo_eps}) window={self.slack_decode_frac}xD "
                f"max_per_request={self.laxity_max_pauses} "
                f"(below target {self.below_target_max_pauses})"
            )
        else:
            policy = (
                f"tpot_budget={self.tpot_budget} setpoint={self.target_frac}xbudget "
                f"window={self.slack_decode_frac}xD x g[{self.gain_min}, {self.gain_max}] "
                f"max_per_request={self.max_per_request}"
            )
        return (
            f"{policy} cap={self.cap_frac}xmax_num_active_requests "
            f"max_pause={self.max_pause_frac}xD min_run={self.min_run_frac}xD"
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
    # target_output_tps: counted against the objective once (a sacrifice),
    # and, while paused as a laxity victim, the time its laxity is down to
    # the resume margin (None: only the maximum pause applies).
    sacrificed: bool = False
    laxity_resume_t: Optional[float] = None


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
        # target_output_tps: the sacrifice credit, in completions, and the
        # laxity of this window's laxity victims when paused.
        self.credit = 0.0
        self._laxity_sum = 0.0

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
        """The TPOT budget of the controller (0 with ``target_output_tps``)."""
        return self.settings.tpot_budget or 0.0

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
        if not target_ready:
            return
        # This request's TPOT inflation from its pauses.
        inflation = (paused / tokens) / self.mean_tpot_s
        if s.output_tps_slo:
            self._infl_sum += inflation
            self._infl_n += 1
            self._on_left_slo(track, tokens)
            return
        target = s.target_frac * self.tpot_budget
        if target <= 0.0:
            return
        # Integrate the inflation's error against the setpoint, one step per
        # completion.
        theta = self.theta + (target - inflation) / (target * s.time_constant)
        self.theta = min(math.log(s.gain_max), max(math.log(s.gain_min), theta))
        self._infl_sum += inflation
        self._infl_n += 1

    def _on_left_slo(self, track: _ReqTrack, tokens: int) -> None:
        """Credit update for one completion after X: + eps, - 1 if it finished
        below the target, + 1 back if it was sacrificed (debited when chosen),
        clamped to +-slo_credit_depth x eps."""
        s = self.settings
        below = tokens / (self._now - track.first_token_t) < s.target_output_tps
        if track.sacrificed:
            self.credit += 1.0
        depth = s.slo_credit_depth * s.slo_eps
        self.credit = min(depth, max(-depth, self.credit + s.slo_eps - float(below)))
        for counter in (self.window, self.totals):
            counter["slo_completed"] += 1
            counter["below_target"] += int(below)

    def laxity(self, req) -> Optional[float]:
        """``target_output_tps``: seconds this request can still be paused and
        finish at or above the target, at its own unpaused rate::

            first token time + (max_tokens - 1) / T
              - (now + remaining tokens / unpaused rate)

        Negative: projected below the target.  None without a first token,
        ``max_tokens`` or a rate (no tokens yet and R0 unknown).  Inputs are
        rank-identical: token counts, ``max_tokens`` and rank 0's clock.
        """
        track = self._tracks.get(req.py_request_id)
        max_new = getattr(req, "py_max_new_tokens", None)
        if track is None or track.first_token_t is None or self._now is None or max_new is None:
            return None
        paused = track.paused_total
        if track.paused_t is not None:
            paused += max(0.0, self._now - track.paused_t)
        gen = generated_tokens(req)
        unpaused = self._now - track.first_token_t - paused
        tokens = gen - track.first_token_gen
        rate = tokens / unpaused if tokens > 0 and unpaused > 0.0 else self.unpaused_tps
        if not rate:
            return None
        deadline = track.first_token_t + (max_new - track.first_token_gen) / (
            self.settings.target_output_tps
        )
        return deadline - (self._now + max(0, max_new - gen) / rate)

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

    def resume_at(self, req) -> float:
        """When a paused request is resumed even without a free slot: after
        the maximum pause, or (laxity victims of ``target_output_tps``) when
        its laxity is down to the resume margin, whichever is first."""
        track = self._tracks[req.py_request_id]
        t = track.paused_t + self.max_pause_s
        if track.laxity_resume_t is not None:
            t = min(t, track.laxity_resume_t)
        return t

    def resume_candidates(self, free_slots: int) -> list:
        """Paused requests to resume: as many as there are free slots, plus
        every one due (``resume_at``).  The lane takes the oldest pause first,
        or with ``target_output_tps`` the earliest ``resume_at`` first (the
        least laxity; ties in pause order)."""
        out = []
        paused = list(self._paused.values())
        if self.settings.output_tps_slo:
            paused.sort(key=self.resume_at)
        for req in paused:
            if len(out) < free_slots:
                out.append(req)
                continue
            if self._now is not None and self._now >= self.resume_at(req):
                out.append(req)
                self.window["forced_resumes"] += 1
                track = self._tracks[req.py_request_id]
                if track.laxity_resume_t is not None and self._now >= track.laxity_resume_t:
                    self.window["laxity_resumes"] += 1
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
            track.laxity_resume_t = None
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
        if track is None or track.pauses >= s.max_pauses:
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
        s = self.settings
        if not s.output_tps_slo and self.tpot_budget <= 0.0:
            return []
        budget = min(num_at_risk, s.max_per_pass, self.max_paused - self.num_paused)
        if budget <= 0:
            return []
        if s.output_tps_slo:
            candidates = [req for req in active_requests if self.is_eligible(req)]
            victims = []
            while len(victims) < budget:
                victim = self._slo_victim(candidates)
                if victim is None:
                    break
                victims.append(victim)
                candidates.remove(victim)
        else:
            victims = self.victim_order(active_requests)[:budget]
        if not victims:
            self.window["no_victim"] += 1
        return victims

    def _slo_victim(self, candidates: list):
        """``target_output_tps`` victim among eligible ``candidates``: the
        largest laxity (at least 0.02 D), else the longest resident request
        already below the target, else a new sacrifice (longest resident)
        if the credit allows; None otherwise."""
        s = self.settings
        min_laxity = s.laxity_min_frac * (self.mean_decode_s or 0.0)
        by_laxity, below, fresh = [], [], []
        for req in candidates:
            track = self._tracks[req.py_request_id]
            laxity = self.laxity(req)
            if track.sacrificed or (laxity is not None and laxity < 0.0):
                if track.pauses < s.below_target_max_pauses:
                    below.append(req)
                continue
            if track.pauses >= s.laxity_max_pauses:
                continue
            if laxity is not None and laxity >= min_laxity:
                by_laxity.append((-laxity, self.victim_key(req), req))
            fresh.append(req)
        if by_laxity:
            neg_laxity, _, victim = min(by_laxity, key=lambda x: x[:2])
            self.window["laxity_victims"] += 1
            self._laxity_sum -= neg_laxity
            return victim
        if below:
            self.window["below_target_victims"] += 1
            return min(below, key=self.victim_key)
        if fresh and self.credit >= 1.0:
            victim = min(fresh, key=self.victim_key)
            self._tracks[victim.py_request_id].sacrificed = True
            self.credit -= 1.0
            self.window["sacrificed"] += 1
            self.totals["sacrificed"] += 1
            return victim
        return None

    def on_paused(self, req) -> None:
        rid = req.py_request_id
        track = self._tracks[rid]
        if self.settings.output_tps_slo and not track.sacrificed:
            # A laxity victim resumes when its laxity is down to the margin;
            # one already below the target only after the maximum pause.
            laxity = self.laxity(req)
            if laxity is not None and laxity >= 0.0 and self._now is not None:
                margin = self.settings.laxity_resume_frac * (self.mean_decode_s or 0.0)
                track.laxity_resume_t = self._now + laxity - margin
        self._paused[rid] = req
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
            f"unpaused_tps={r0} "
        )
        if self.settings.output_tps_slo:
            line += self._slo_stats() + " "
        else:
            line += f"budget={100.0 * self.tpot_budget:.2f}% "
        line += (
            f"tpot_infl={infl} completed={w['completed']} "
            f"max_pause={self.max_pause_s:.2f}s min_run={self.min_run_s:.2f}s"
        )
        self._infl_sum = 0.0
        self._infl_n = 0
        self._laxity_sum = 0.0
        self.window.clear()
        return line

    def _slo_stats(self) -> str:
        """``target_output_tps``: completions below the target (window /
        total), credit, victims by class, mean laxity of the window's laxity
        victims, laxity-margin resumes, and paused requests that are below
        the target or sacrificed."""
        w, t = self.window, self.totals
        n_lax = w["laxity_victims"]
        mean_laxity = f"{self._laxity_sum / n_lax:.2f}s" if n_lax else "na"
        paused_below = sum(1 for rid in self._paused if self._tracks[rid].laxity_resume_t is None)
        return (
            f"target_tps={self.settings.target_output_tps:.2f} "
            f"below_target={w['below_target']}/{w['slo_completed']} "
            f"below_target_total={t['below_target']}/{t['slo_completed']} "
            f"credit={self.credit:.1f} laxity_victims={n_lax} "
            f"below_target_victims={w['below_target_victims']} "
            f"sacrificed={w['sacrificed']} sacrificed_total={t['sacrificed']} "
            f"mean_laxity={mean_laxity} laxity_resumes={w['laxity_resumes']} "
            f"paused_below={paused_below}"
        )

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
