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
admission queue would admit next is about to miss the TTFT target, one running
decode request is paused in place so the waiting request can take its slot.

A paused request stays resident: its KV, KDA/Mamba state, draft KV and
sequence slot stay where they are, it is only left out of the scheduled batch,
so a pause costs no copy and no extra step.  Resident requests (running plus
paused) may therefore exceed ``max_batch_size`` by up to
``TLLM_GEN_PREEMPT_MAX_PAUSED``; the pools that hold one entry per resident
request are sized for that (see ``resident_headroom``), while the scheduled
batch, and so every CUDA graph, stays within ``max_batch_size``.  A paused
request resumes ahead of every new admission once a slot frees up, or after
``TLLM_GEN_PREEMPT_MAX_PAUSE_S`` regardless.

``TLLM_GEN_PREEMPT_POLICY`` selects the victim, the running request with the
lowest cost of pausing:

``off`` (the default)
    Nothing is ever paused.
``newest``
    The most recently admitted request (the scheduler's tail eviction order).
``wait``
    The longest resident request: its first token is long out and it is the
    furthest from any TTFT deadline.  The freed slot goes to the request the
    admission queue pops next, the oldest one that can still make the target.
``tps``
    Only requests ahead of their TPOT budget: the output rate since the first
    token, ``(tokens since first token) / (time since first token)``, must be
    above ``TLLM_GEN_PREEMPT_TARGET_TPS`` and stay at or above it over the
    expected pause.  The victim has the largest margin above the target; with
    none eligible nothing is paused.  Needs the target (per-request output
    tokens/s); without it preemption stays off.
``long``
    The request that holds the most KV (prompt + generated tokens).

``TLLM_GEN_PREEMPT_LATE_TRIGGER=1`` also pauses for a next pop that is
already late, not only for one that can still make the target.

Preemption needs the ``auto`` GEN admission policy, whose target X and rank
0's admission -> first token estimate decide when a waiting request is at
risk.  Every input to a decision is identical on all ranks of the
model-parallel group (helix included): the active request list, token and
decode-iteration counts, the waiting queue (broadcast) and rank 0's clock
(broadcast with the request-count probe).
"""

import collections
import os
from dataclasses import dataclass, fields
from typing import Callable, Iterable, Mapping, Optional

from tensorrt_llm.logger import logger

from ..llm_request import LlmRequestState

PREEMPT_POLICY_ENV = "TLLM_GEN_PREEMPT_POLICY"
MAX_PAUSED_ENV = "TLLM_GEN_PREEMPT_MAX_PAUSED"

POLICY_OFF = "off"
POLICY_NEWEST = "newest"
POLICY_WAIT = "wait"
POLICY_TPS = "tps"
POLICY_LONG = "long"
POLICIES = (POLICY_OFF, POLICY_NEWEST, POLICY_WAIT, POLICY_TPS, POLICY_LONG)

# EMA weight of a new measured pause (``tps`` with expected_pause_s = 0).
_PAUSE_EMA_ALPHA = 0.1


@dataclass(frozen=True)
class GenPreemptionConfig:
    """Victim policy and guardrails; every field but ``policy`` is read from
    ``TLLM_GEN_PREEMPT_<FIELD NAME IN UPPER CASE>``."""

    policy: str = POLICY_OFF
    # Requests paused at once; also the resident headroom the pools reserve.
    max_paused: int = 0
    # A waiting request is at risk once its slack, X - (age + s2f), is in
    # [0, slack_s]: a slot now still brings it in on time, but barely.
    slack_s: float = 5.0
    # 1: the next pop also triggers when it is already late (slack < 0) or
    # popped by another tier (guard, breaker); its slack must still be at
    # most slack_s.  0: only an on-time next pop of the slack tier.
    late_trigger: int = 0
    # Time since the first token before a request is preemptible.
    min_run_s: float = 2.0
    # A request paused this long resumes even without a free slot.
    max_pause_s: float = 10.0
    # Pauses over a request's lifetime.
    max_per_request: int = 1
    # Pauses per scheduling pass.
    max_per_pass: int = 1
    # Decode steps after a resume before a request is preemptible again.
    hysteresis_iters: int = 64
    # Requests this close to max_new_tokens finish soon enough on their own.
    min_remaining_tokens: int = 32
    # ``tps``: per-request output tokens/s a victim must stay above (0: unset).
    target_tps: float = 0.0
    # ``tps``: pause to plan for; 0 uses the measured mean pause (EMA).
    expected_pause_s: float = 0.0

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(
                f"{PREEMPT_POLICY_ENV}={self.policy!r} must be one of {', '.join(POLICIES)}"
            )
        if any(
            not getattr(self, f.name) >= 0 for f in fields(self) if f.name != "policy"
        ):  # also rejects nan
            raise ValueError("GenPreemptionConfig: values must be non-negative")

    @property
    def disabled_reason(self) -> Optional[str]:
        """Why a configured policy cannot run, or None."""
        if self.policy == POLICY_OFF:
            return None
        if self.max_paused <= 0:
            return f"{PREEMPT_POLICY_ENV}={self.policy} needs {MAX_PAUSED_ENV} > 0"
        if self.max_per_request <= 0 or self.max_per_pass <= 0:
            return "TLLM_GEN_PREEMPT_MAX_PER_REQUEST and _MAX_PER_PASS must be > 0"
        if self.policy == POLICY_TPS and self.target_tps <= 0.0:
            return f"{PREEMPT_POLICY_ENV}=tps needs TLLM_GEN_PREEMPT_TARGET_TPS > 0"
        return None

    @property
    def enabled(self) -> bool:
        return self.policy != POLICY_OFF and self.disabled_reason is None

    def describe(self) -> str:
        return " ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "GenPreemptionConfig":
        env = os.environ if environ is None else environ
        kwargs = {}
        policy = env.get(PREEMPT_POLICY_ENV, "").strip().lower()
        if policy:
            kwargs["policy"] = policy
        for f in fields(cls):
            if f.name == "policy":
                continue
            raw = env.get(f"TLLM_GEN_PREEMPT_{f.name.upper()}", "").strip()
            if raw:
                kwargs[f.name] = float(raw) if f.type is float else int(raw)
        return cls(**kwargs)


def resident_headroom(environ: Optional[Mapping[str, str]] = None) -> int:
    """Resident requests above ``max_batch_size`` that preemption may hold.

    Paused requests keep their sequence slot, KV index slot and recurrent
    state slot while the requests that replaced them run, so every pool with
    one entry per resident request reserves this many more.  Every rank
    derives the same value from its environment.
    """
    config = GenPreemptionConfig.from_env(environ)
    return config.max_paused if config.enabled else 0


def context_tokens(req) -> int:
    """Prompt length; the global one under helix (rank-identical)."""
    return getattr(req, "total_input_len_cp", None) or req.prompt_len


def generated_tokens(req) -> int:
    """Tokens generated so far (rank-identical, also under helix)."""
    return max(0, req.get_num_tokens(0) - req.prompt_len)


def kv_tokens(req) -> int:
    return context_tokens(req) + generated_tokens(req)


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


class GenPreemption:
    """Rank-identical pause bookkeeping and victim selection.

    The executor calls, once per scheduling pass and in this order,
    ``observe`` with the active requests (the same on every rank),
    ``resume_candidates`` / ``on_resumed`` for the resume lane, then
    ``select_victims`` / ``on_paused``.  Paused requests stay in the
    executor's active requests; the scheduler skips them and the admission
    count excludes them.
    """

    def __init__(self, config: GenPreemptionConfig) -> None:
        self.config = config
        self._score: Callable = {
            POLICY_NEWEST: self._score_newest,
            POLICY_WAIT: self._score_wait,
            POLICY_TPS: self._score_tps,
            POLICY_LONG: self._score_long,
        }.get(config.policy, self._score_newest)
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

    def achieved_tps(self, req) -> Optional[float]:
        """Output tokens/s since the first token, on rank 0's clock."""
        elapsed = self._since_first_token(req)
        if elapsed is None or elapsed <= 0.0:
            return None
        track = self._tracks[req.py_request_id]
        return (generated_tokens(req) - track.first_token_gen) / elapsed

    def tps_margin(self, req) -> Optional[float]:
        """Rate over the expected pause minus the target, if the request is
        above the target now and stays at or above it after the pause."""
        c = self.config
        rate = self.achieved_tps(req)
        if rate is None or rate <= c.target_tps:
            return None
        elapsed = self._since_first_token(req)
        pause = c.expected_pause_s or (self.mean_pause_s or 0.0)
        after = rate * elapsed / (elapsed + pause)
        if after < c.target_tps:
            return None
        return after - c.target_tps

    def observe(self, active_requests: Iterable, now: Optional[float]) -> None:
        """Track the active requests; forget the ones that left."""
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
        for rid in [rid for rid in self._tracks if rid not in seen]:
            del self._tracks[rid]
        for rid in [rid for rid in self._paused if rid not in seen]:
            # Cancelled or failed while paused.
            del self._paused[rid]
            self.window["dropped"] += 1

    # ---- resume lane ----

    def resume_candidates(self, free_slots: int) -> list:
        """Paused requests to resume, oldest pause first: as many as there are
        free slots, plus every one paused for ``max_pause_s`` or longer."""
        out = []
        for req in self._paused.values():
            if len(out) < free_slots:
                out.append(req)
                continue
            paused_t = self._tracks[req.py_request_id].paused_t
            if self._now is not None and self._now - paused_t >= self.config.max_pause_s:
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
        c = self.config
        rid = req.py_request_id
        if rid in self._paused or req.state != LlmRequestState.GENERATION_IN_PROGRESS:
            return False
        if req.is_dummy_request or getattr(req, "is_attention_dp_dummy", False):
            return False
        if req.py_decoding_iter < 1:
            return False
        track = self._tracks.get(rid)
        if track is None or track.pauses >= c.max_per_request:
            return False
        since_first_token = self._since_first_token(req)
        if since_first_token is None or since_first_token < c.min_run_s:
            return False
        if (
            track.resumed_at_iter is not None
            and req.py_decoding_iter - track.resumed_at_iter < c.hysteresis_iters
        ):
            return False
        max_new = getattr(req, "py_max_new_tokens", None)
        if max_new is not None and max_new - generated_tokens(req) <= c.min_remaining_tokens:
            return False
        if c.policy == POLICY_TPS and self.tps_margin(req) is None:
            return False
        return True

    def victim_order(self, active_requests: Iterable) -> list:
        """Eligible requests, cheapest to pause first (ties: newest first)."""
        eligible = [req for req in active_requests if self.is_eligible(req)]
        return sorted(
            eligible,
            key=lambda req: (self._score(req), -self._tracks[req.py_request_id].order),
        )

    def select_victims(self, active_requests: Iterable, num_at_risk: int) -> list:
        """Up to ``num_at_risk`` victims within the per-pass and paused-count
        budgets."""
        c = self.config
        budget = min(num_at_risk, c.max_per_pass, c.max_paused - self.num_paused)
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

    # ---- policies: cost of pausing, lowest first ----

    def _score_newest(self, req) -> float:
        return -self._tracks[req.py_request_id].order

    def _score_wait(self, req) -> float:
        first_seen = self._tracks[req.py_request_id].first_seen
        if self._now is None or first_seen is None:
            return -req.py_decoding_iter
        return -(self._now - first_seen)

    def _score_tps(self, req) -> float:
        # Only requests with a margin are eligible; the largest goes first.
        return -self.tps_margin(req)

    def _score_long(self, req) -> float:
        return -kv_tokens(req)

    # ---- logging ----

    def stats_line(self) -> str:
        w = self.window
        mean_pause = "na" if self.mean_pause_s is None else f"{self.mean_pause_s:.2f}s"
        line = (
            f"[gen_preempt] policy={self.config.policy} paused={w['paused']} "
            f"resumed={w['resumed']} forced_resumes={w['forced_resumes']} "
            f"dropped={w['dropped']} no_victim={w['no_victim']} "
            f"paused_now={self.num_paused}/{self.config.max_paused} "
            f"mean_pause={mean_pause} total_paused={self.totals['paused']} "
            f"total_resumed={self.totals['resumed']}"
        )
        self.window.clear()
        return line

    @property
    def has_activity(self) -> bool:
        return bool(+self.window) or self.num_paused > 0


def log_gen_preemption_config(
    config: GenPreemptionConfig, disabled_reason: Optional[str] = None
) -> None:
    if config.policy == POLICY_OFF:
        return
    disabled_reason = disabled_reason or config.disabled_reason
    if disabled_reason is not None:
        logger.info(f"[gen_preempt] GEN slot preemption disabled: {disabled_reason}")
    else:
        logger.info(
            f"[gen_preempt] GEN slot preemption (pause in place) enabled: {config.describe()}"
        )
