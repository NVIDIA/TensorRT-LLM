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

"""Load estimates for the TensorRT-LLM SMG gRPC adapter.

What the adapter can tell a router about its engine's load, from the requests it forwards and
from the engine's iteration statistics, in the shape the router consumes: the ``EngineLoad``
record on a ``SubscribeKvEvents`` stream and the ``SchedulerLoad`` of a ``GetLoads`` answer (see
``load_feed``). Plain Python without engine or protobuf imports, so every rule is unit-tested
without an engine.

The rules are the ones the vLLM servicer applies, so a router prices both engines alike:

- queued token-work: the uncached prompt tokens of the forwarded requests the engine has not
  started. The engine's queued count says how many of them wait (the youngest ones; the older
  ones are in prefill); each prompt is discounted by the recent cache hit rate because its own
  cached count is unknown until it starts. When the engine reports the prompt tokens of its
  queued requests, that figure is discounted instead;
- generation throughput: the tokens streamed to clients over the last :data:`THROUGHPUT_WINDOW`;
- cache hit rate: cached over prompt tokens of the last :data:`HIT_RATE_SAMPLES` first outputs;
- prefill capacity (``prefill_throughput``): computed input tokens (prompt minus cached) per
  second of engine time spent prefilling, the union of the requests' scheduled-to-first-token
  intervals on the engine clock: concurrent prefills count once, idle gaps and queueing not at
  all. It is an observed service rate, not kernel time. At least :data:`PREFILL_MIN_SAMPLES`
  samples are required and up to :data:`PREFILL_SAMPLES` are kept for :data:`PREFILL_SAMPLE_TTL`.
  A zero-work cache hit, a missing or non-finite timing, an interval under
  :data:`PREFILL_MIN_INTERVAL`, an aborted, cancelled or failed request and a disaggregated
  generation-only request do not calibrate it. The engine's per-request metrics carry no pause
  flag; a pause applies to requests past their first token, which leaves this interval alone.

Running and waiting counts, the admission window and the KV usage come from the engine's
iteration statistics while they flow (:class:`EngineStats`); the adapter's own counts stand in
when they do not (:func:`compose_record`).
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "HEARTBEAT_BACKOFF",
    "HEARTBEAT_INTERVAL",
    "HIT_RATE_SAMPLES",
    "LOAD_TICK",
    "PREFILL_MIN_INTERVAL",
    "PREFILL_MIN_SAMPLES",
    "PREFILL_SAMPLES",
    "PREFILL_SAMPLE_TTL",
    "STATS_FRESH",
    "THROUGHPUT_WINDOW",
    "EngineStats",
    "EngineStatsFold",
    "KvFigures",
    "LoadEstimate",
    "LoadRecord",
    "LoadTracker",
    "RecordPacer",
    "compose_record",
    "load_changed",
    "prefill_sample",
]

#: The window generation throughput is measured over (seconds).
THROUGHPUT_WINDOW = 2.0
#: How many first outputs the cache hit rate averages over.
HIT_RATE_SAMPLES = 64
#: How many prefill samples the capacity estimate keeps, and for how long (seconds).
PREFILL_SAMPLES = 64
PREFILL_SAMPLE_TTL = 60.0
#: Samples needed before a capacity is reported.
PREFILL_MIN_SAMPLES = 2
#: Shortest scheduled-to-first-token interval that calibrates capacity (seconds).
PREFILL_MIN_INTERVAL = 0.001
#: An engine iteration record older than this no longer supplies the counts (seconds).
STATS_FRESH = 2.0
#: How often a records-only stream checks the record for a change (seconds).
LOAD_TICK = 0.1
#: Silence after which a ``load_only`` heartbeat goes out (seconds).
HEARTBEAT_INTERVAL = 1.0
#: The heartbeat interval once two heartbeats in a row carried an unchanged record (seconds).
HEARTBEAT_BACKOFF = 5.0

Clock = Callable[[], float]


@dataclass(frozen=True)
class LoadEstimate:
    """The adapter's own figures for the record, at one instant."""

    #: Forwarded requests that produced output.
    running: int
    #: Forwarded requests without output yet.
    waiting: int
    #: Uncached prompt tokens of the waiting requests (see the module docstring).
    queued_token_work: int
    #: Prompt tokens of every waiting request: the engine's prefill work in hand.
    prefill_pending_tokens: int
    #: Tokens streamed per second over the window.
    gen_throughput: float
    #: Computed input tokens per active prefill second; ``None`` while unknown or expired.
    prefill_throughput: float | None
    #: Cached over prompt tokens of the recent first outputs, in ``[0, 1]``.
    cache_hit_rate: float


@dataclass(frozen=True)
class EngineStats:
    """One engine iteration record, as the LLM API's ``get_stats`` reports it."""

    #: When the record was read, on the adapter's clock.
    at: float
    #: The attention data-parallel rank the record describes.
    rank: int
    running: int
    waiting: int
    #: The admission window (``maxNumActiveRequests``); 0 when the record lacks it.
    max_running: int
    #: Prompt tokens of the queued context requests; ``None`` when the record lacks the figure.
    queued_prompt_tokens: int | None
    used_blocks: int
    max_blocks: int
    tokens_per_block: int

    @classmethod
    def from_iteration(cls, record: Mapping[str, Any], at: float) -> EngineStats | None:
        """The statistics of one iteration record, or ``None`` for a record without counts."""
        if "numActiveRequests" not in record:
            return None
        kv = record.get("kvCacheStats") or {}
        batching = record.get("inflightBatchingStats") or {}
        queued_tokens = batching.get("numQueuedCtxTokens")
        return cls(
            at=at,
            rank=int(record.get("attentionDpRank", record.get("rank", 0)) or 0),
            running=_count(record.get("numActiveRequests")),
            waiting=_count(record.get("numQueuedRequests")),
            max_running=_count(record.get("maxNumActiveRequests")),
            queued_prompt_tokens=None if queued_tokens is None else _count(queued_tokens),
            used_blocks=_count(kv.get("usedNumBlocks")),
            max_blocks=_count(kv.get("maxNumBlocks")),
            tokens_per_block=_count(kv.get("tokensPerBlock")),
        )


@dataclass(frozen=True)
class KvFigures:
    """The KV cache in use and its capacity, in tokens."""

    used_tokens: int
    max_tokens: int

    @property
    def usage(self) -> float:
        """The usage ratio in ``[0, 1]``; 0 without a known capacity."""
        if self.max_tokens <= 0:
            return 0.0
        return min(1.0, max(0.0, self.used_tokens / self.max_tokens))


class EngineStatsFold:
    """The newest iteration record of every attention data-parallel rank, folded into one.

    The LLM API serves the ranks behind one endpoint and does not name a request's rank, so
    the adapter reports the engine as one: counts and blocks summed over the ranks, the oldest
    rank's read time as the fold's.
    """

    def __init__(self) -> None:
        self._latest: dict[int, EngineStats] = {}

    def observe(self, stats: EngineStats) -> None:
        self._latest[stats.rank] = stats

    def current(self) -> EngineStats | None:
        """The folded record, or ``None`` before the first one."""
        ranks = list(self._latest.values())
        if not ranks:
            return None
        if len(ranks) == 1:
            return ranks[0]
        queued = [r.queued_prompt_tokens for r in ranks if r.queued_prompt_tokens is not None]
        return EngineStats(
            at=min(r.at for r in ranks),
            rank=0,
            running=sum(r.running for r in ranks),
            waiting=sum(r.waiting for r in ranks),
            max_running=sum(r.max_running for r in ranks),
            queued_prompt_tokens=sum(queued) if queued else None,
            used_blocks=sum(r.used_blocks for r in ranks),
            max_blocks=sum(r.max_blocks for r in ranks),
            tokens_per_block=max(r.tokens_per_block for r in ranks),
        )

    def kv(self) -> KvFigures | None:
        """The last known KV figures, whatever their age; ``None`` before the first record."""
        folded = self.current()
        if folded is None or folded.max_blocks <= 0:
            return None
        return KvFigures(
            used_tokens=folded.used_blocks * folded.tokens_per_block,
            max_tokens=folded.max_blocks * folded.tokens_per_block,
        )


@dataclass(frozen=True)
class LoadRecord:
    """The load record as the router reads it, before it goes on the wire.

    Every ``None`` is a field left unset on the wire (absent is distinct from zero to the
    router); the other fields are its routing core.
    """

    running_requests: int
    waiting_requests: int
    waiting_uncached_tokens: int | None
    token_usage: float
    gen_throughput: float
    max_running_requests: int
    #: Age of the engine figures behind the record in milliseconds (0: the adapter's own).
    age_ms: int
    cache_hit_rate: float | None
    num_used_tokens: int | None
    max_total_num_tokens: int | None
    prefill_pending_tokens: int | None
    prefill_throughput: float | None


def _count(value: Any) -> int:
    """A count as the record takes it: a missing or negative value reads as 0."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class LoadTracker:
    """The adapter's bookkeeping of the requests it forwards, for :class:`LoadEstimate`.

    ``clock`` is monotonic seconds (tests pass their own). Not thread-safe: it is driven from
    the servicer's event loop.
    """

    def __init__(self, *, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        # Forwarded requests without output yet: request id -> prompt tokens, oldest first.
        self._pending: dict[str, int] = {}
        # Forwarded requests that produced output and have not finished.
        self._running: set[str] = set()
        # Tokens streamed, by time, within the window.
        self._generated: deque[tuple[float, int]] = deque()
        self._generated_sum = 0
        # (cached, prompt) of the recent first outputs.
        self._hits: deque[tuple[int, int]] = deque(maxlen=HIT_RATE_SAMPLES)
        # (observed locally, computed tokens, engine start, engine end).
        self._prefills: deque[tuple[float, int, float, float]] = deque()

    def forwarded(self, request_id: str, prompt_tokens: int) -> None:
        """A request was handed to the engine."""
        if request_id in self._pending or request_id in self._running:
            return
        self._pending[request_id] = max(0, int(prompt_tokens))

    def started(self, request_id: str, cached_tokens: int) -> None:
        """A forwarded request produced its first output.

        ``cached_tokens`` of its prompt were served from the KV cache.
        """
        prompt = self._pending.pop(request_id, None)
        if prompt is None:
            return
        self._running.add(request_id)
        self._hits.append((min(prompt, max(0, int(cached_tokens or 0))), prompt))

    def generated(self, tokens: int) -> None:
        """``tokens`` were streamed to a client."""
        if tokens <= 0:
            return
        now = self._clock()
        self._generated.append((now, tokens))
        self._generated_sum += tokens
        self._trim_generated(now)

    def prefill_computed(self, tokens: int, start: float, end: float) -> None:
        """A request's prefill computed ``tokens`` between ``start`` and ``end``.

        Both are engine-clock seconds. A zero-work hit or an invalid interval does not calibrate
        capacity.
        """
        if (
            tokens <= 0
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0.0
            or end - start < PREFILL_MIN_INTERVAL
        ):
            return
        self._prefills.append((self._clock(), int(tokens), float(start), float(end)))
        while len(self._prefills) > PREFILL_SAMPLES:
            self._prefills.popleft()

    def finished(self, request_id: str) -> None:
        """A forwarded request is done: finished, aborted or failed."""
        self._pending.pop(request_id, None)
        self._running.discard(request_id)

    def estimate(self, num_waiting_reqs: int | None = None) -> LoadEstimate:
        """The figures now.

        ``num_waiting_reqs`` is the engine's queued count when known: only the youngest that many
        pending prompts are queued token-work; ``None`` counts them all.
        """
        now = self._clock()
        self._trim_generated(now)
        hit_rate = self._hit_rate()
        prompts = list(self._pending.values())
        if num_waiting_reqs is not None:
            prompts = prompts[len(prompts) - min(len(prompts), max(0, num_waiting_reqs)) :]
        queued = round(sum(prompts) * (1.0 - hit_rate))
        return LoadEstimate(
            running=len(self._running),
            waiting=len(self._pending),
            queued_token_work=max(0, queued),
            prefill_pending_tokens=sum(self._pending.values()),
            gen_throughput=self._generated_sum / THROUGHPUT_WINDOW,
            prefill_throughput=self._prefill_throughput(now),
            cache_hit_rate=hit_rate,
        )

    def _trim_generated(self, now: float) -> None:
        while self._generated and now - self._generated[0][0] > THROUGHPUT_WINDOW:
            self._generated_sum -= self._generated.popleft()[1]

    def _hit_rate(self) -> float:
        prompt_sum = sum(prompt for _, prompt in self._hits)
        if prompt_sum <= 0:
            return 0.0
        return min(1.0, sum(cached for cached, _ in self._hits) / prompt_sum)

    def _prefill_throughput(self, now: float) -> float | None:
        while self._prefills and now - self._prefills[0][0] > PREFILL_SAMPLE_TTL:
            self._prefills.popleft()
        if len(self._prefills) < PREFILL_MIN_SAMPLES:
            return None
        tokens = sum(sample[1] for sample in self._prefills)
        seconds = 0.0
        until = -math.inf
        for start, end in sorted((sample[2], sample[3]) for sample in self._prefills):
            # Concurrent intervals count once: each engine second is one second of capacity.
            seconds += max(0.0, end - max(start, until))
            until = max(until, end)
        if seconds <= 0.0:
            return None
        rate = tokens / seconds
        return rate if math.isfinite(rate) and rate > 0.0 else None


def prefill_sample(
    prompt_tokens: int,
    cached_tokens: int,
    first_scheduled: float | None,
    first_token: float | None,
    *,
    cancelled: bool = False,
    generation_only: bool = False,
) -> tuple[int, float, float] | None:
    """The ``(tokens, start, end)`` a request's first output contributes to the capacity estimate.

    ``None`` when it must not: the timings are the engine's steady-clock seconds (0 or ``None``
    when unset); a cancelled, aborted or failed request and a disaggregated generation-only
    request have no prefill of their own to measure.
    """
    if cancelled or generation_only or first_scheduled is None or first_token is None:
        return None
    if first_scheduled <= 0.0 or first_token <= 0.0:
        return None
    computed = max(0, int(prompt_tokens) - max(0, int(cached_tokens or 0)))
    if computed == 0:
        return None
    return computed, float(first_scheduled), float(first_token)


def compose_record(
    estimate: LoadEstimate,
    engine: EngineStats | None,
    kv: KvFigures | None,
    *,
    now: float,
    max_running_fallback: int,
) -> LoadRecord:
    """The record from the adapter's estimate and the engine's figures.

    With nothing forwarded and in flight the counts are zero whatever an old iteration record
    says (the engine emits none while idle). Otherwise an iteration record younger than
    :data:`STATS_FRESH` supplies the running and waiting counts, the admission window and the
    queued prompt tokens (discounted by the hit rate); the adapter's own counts stand in when
    the record is older or the engine emits none. The KV figures are the last known, whatever
    their age: the engine reports them only while it iterates.
    """
    fresh = engine is not None and now - engine.at <= STATS_FRESH
    in_flight = estimate.running + estimate.waiting
    queued: int | None
    if in_flight == 0:
        running, waiting, queued, age_ms = 0, 0, 0, 0
    elif fresh:
        running, waiting = engine.running, engine.waiting
        if engine.queued_prompt_tokens is not None:
            queued = max(0, round(engine.queued_prompt_tokens * (1.0 - estimate.cache_hit_rate)))
        else:
            queued = estimate.queued_token_work
        age_ms = max(0, int((now - engine.at) * 1000.0))
    else:
        running, waiting, queued, age_ms = (
            estimate.running,
            estimate.waiting,
            estimate.queued_token_work,
            0,
        )
    window = engine.max_running if fresh and engine.max_running > 0 else max_running_fallback
    return LoadRecord(
        running_requests=running,
        waiting_requests=waiting,
        waiting_uncached_tokens=queued,
        token_usage=kv.usage if kv is not None else 0.0,
        gen_throughput=estimate.gen_throughput,
        max_running_requests=max(0, int(window or 0)),
        age_ms=age_ms,
        cache_hit_rate=estimate.cache_hit_rate,
        num_used_tokens=kv.used_tokens if kv is not None else None,
        max_total_num_tokens=kv.max_tokens if kv is not None else None,
        prefill_pending_tokens=estimate.prefill_pending_tokens,
        prefill_throughput=estimate.prefill_throughput,
    )


def _throughput_changed(last: float, current: float) -> bool:
    """Meaningful absolute or relative movement of a rate, not noise."""
    delta = abs(last - current)
    return delta > 50.0 or delta > 0.05 * max(last, current)


def load_changed(last: LoadRecord, current: LoadRecord) -> bool:
    """Whether the record moved enough from ``last`` to be worth a ``load_only`` batch.

    Any queue, running, window or pending-prefill change, KV usage by half a percent, either
    throughput by 5 % or 50 tokens per second, or the measured capacity appearing or expiring.
    """
    if (last.prefill_throughput is None) != (current.prefill_throughput is None):
        return True
    if last.prefill_throughput is not None and current.prefill_throughput is not None:
        if _throughput_changed(last.prefill_throughput, current.prefill_throughput):
            return True
    return (
        last.running_requests != current.running_requests
        or last.waiting_requests != current.waiting_requests
        or last.waiting_uncached_tokens != current.waiting_uncached_tokens
        or last.max_running_requests != current.max_running_requests
        or abs(last.token_usage - current.token_usage) > 0.005
        or _throughput_changed(last.gen_throughput, current.gen_throughput)
        or last.prefill_pending_tokens != current.prefill_pending_tokens
    )


class RecordPacer:
    """When a stream sends its next record.

    On a change (:func:`load_changed`) or when its heartbeat is due: :data:`HEARTBEAT_INTERVAL`
    after the last record, :data:`HEARTBEAT_BACKOFF` once two heartbeats in a row carried an
    unchanged record. The first record always goes.
    """

    def __init__(
        self,
        *,
        heartbeat_interval: float = HEARTBEAT_INTERVAL,
        heartbeat_backoff: float = HEARTBEAT_BACKOFF,
    ) -> None:
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_backoff = heartbeat_backoff
        self._last: LoadRecord | None = None
        self._last_sent_at: float | None = None
        self._idle_heartbeats = 0

    def due(self, current: LoadRecord, now: float) -> bool:
        """Whether ``current`` goes out at ``now``; a ``True`` counts as sent."""
        changed = self._last is None or load_changed(self._last, current)
        due_after = (
            self.heartbeat_backoff if self._idle_heartbeats >= 2 else self.heartbeat_interval
        )
        silent_for = math.inf if self._last_sent_at is None else now - self._last_sent_at
        if not changed and silent_for < due_after:
            return False
        self._idle_heartbeats = 0 if changed else self._idle_heartbeats + 1
        self._last_sent_at = now
        self._last = current
        return True
