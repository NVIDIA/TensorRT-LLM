# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage context-server diagnostics of a pipeline-parallel executor (log-only).

On a disaggregated server at ``pipeline_parallel_size > 1`` every pipeline rank
counts, without a cap and without any device synchronization, what its context
path does, and logs the cumulative counts as one INFO line every
``DEFAULT_INTERVAL_S`` seconds and once more when the executor loop ends:

* **Stage agreement** (first rank, where the agreement runs,
  ``pp_stage_agree.py``). Per proposal round that every follower answered: the
  follower stage(s) whose answer was the smallest when it is below the proposed
  depth, and the tokens the round lost; the first rank itself (stage 0) when
  its own re-claim came back below the depth the followers allowed. Per
  admitted first chunk: the stage(s) whose own uncapped reusable prefix was the
  smallest, when the chunk was admitted below the largest one, and the tokens
  lost against that largest prefix and against the first rank's own prefix.
  Stages tied at the minimum are each counted with the full loss.
* **Reservations** (every rank): reservations made, lowered (and by how many
  tokens), dropped (and the tokens they held), claims that created no cache,
  and first chunks a follower adopted without a reservation at the admitted
  depth (claimed afresh at adoption).
* **Context suspend / evict / resume** (every rank): context caches suspended
  (refused pages) or evicted (for a generation step), preempted, and resumed,
  each split into first chunks and started prompts; the pages each resume
  brought onto the GPU from every lower cache tier (the cache's active pages
  that sat on that tier just before the resume) and the host time of the
  resume call (it enqueues the copies; their device time is not measured).
  A new cache starts suspended, so the first-chunk resumes count every
  first-chunk admission that activated its cache. A first chunk whose resize
  was refused pages is suspended by the KV cache manager itself and is not
  among the first-chunk suspends; a reserved one is counted with the
  reservations kept after a refused admission
  (``ctx_reservations_suspended``, of which ``ctx_reservations_page_refused``
  were refused pages rather than the token budget).
* **Admission holds**: first chunks an admission back-pressure held out of the
  scheduler pass (``admission_holds``; counted only where such a back-pressure
  reports its holds).
* **Interrupted prefills** (every rank): for each context chunk of a started
  prompt, the iterations since the prompt's previous chunk, bucketed as
  1 / 2-64 / 65-2000 / 2001+ and split at ``DEFAULT_DEEP_TOKENS`` cached
  tokens; a gap above ``DEFAULT_GAP_ITERS`` counts as interrupted.
* **Sample-state relay** (every rank): host time waiting in the pre-forward
  drain of the relay sends (``drain_wait_ms``, ``drain_wait_ms_max``), in the
  relay receive (``relay_recv_wait_ms``) and in the previous micro-batch's
  sampler-event synchronize (``sync_prev_ms``), each with its call count; the
  relay messages this rank sent compact or full (``pp_spec_relay.py``) and the largest sample-state payload among them
  (``relay_bytes_max``: the compact rows plus 8 B per request id, or the full
  store's tensor bytes; pickle framing and ``py_result`` diffs excluded).

* **Drafter context window guard host time** (every rank, with
  ``guard_timing``, which the executor sets): the calls of the KV cache manager into
  the reuse guard of a window-bounded drafter context
  (``draft_context_window.py``), per kind (``prefetch``: the store prefetch's
  uncounted claim limit; ``claim``: a reservation's or resume's claim limit;
  ``note``: a claim's note), with their summed and largest host time, and the
  guard's cumulative prefix hash count. Off, the guard calls are not timed and
  the line has no ``guard_*`` field.

The module imports nothing from the package. On every other executor
(``pipeline_parallel_size == 1`` included) nothing here is built and no line is
logged.
"""

import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_INTERVAL_S = 30.0
DEFAULT_GAP_ITERS = 1
DEFAULT_DEEP_TOKENS = 131072

# The token that starts every line this module formats.
LINE_TAG = "[pp_ctx_diag]"

# Buckets of the iterations between two chunks of one prompt: (lowest gap,
# highest gap or None, label).
GAP_BUCKETS: Tuple[Tuple[int, Optional[int], str], ...] = (
    (1, 1, "1"),
    (2, 64, "2-64"),
    (65, 2000, "65-2000"),
    (2001, None, "2001+"),
)

# Context chunks are counted per kind: the first chunk of a prompt, or a later
# chunk of a started prompt.
FIRST = "first"
STARTED = "started"
KINDS = (FIRST, STARTED)

# Kinds of drafter context window guard calls (``PPCtxDiag.guard_call``).
GUARD_CALL_KINDS = ("prefetch", "claim", "note")


def gap_bucket(gap: int) -> int:
    """Index in ``GAP_BUCKETS`` of a gap of ``gap`` iterations.

    Raises:
        ValueError: ``gap`` is below 1.
    """
    for index, (low, high, _) in enumerate(GAP_BUCKETS):
        if gap >= low and (high is None or gap <= high):
            return index
    raise ValueError(f"gap {gap} must be at least 1")


def _per_stage(values: Sequence[int]) -> str:
    return ",".join(f"s{stage}:{value}" for stage, value in enumerate(values))


def _per_level(values: Mapping[int, int]) -> str:
    if not values:
        return "-"
    return ",".join(f"L{level}:{values[level]}" for level in sorted(values))


def _per_bucket(values: Sequence[int]) -> str:
    return ",".join(f"{label}:{value}" for (_, _, label), value in zip(GAP_BUCKETS, values))


class PPCtxDiag:
    """One pipeline rank's cumulative context-path counters.

    The stage agreement, the scheduler, the KV cache managers and the executor
    call the recording methods below; the executor calls
    ``record_context_chunks`` once per loop iteration with the context chunks of
    the batch it runs, and logs ``line`` whenever ``due`` and once at the end.
    Every method is host-only bookkeeping.

    Args:
        pp_rank: this rank's pipeline stage.
        pp_size: pipeline-parallel size.
        interval_s: seconds between two periodic lines.
        gap_iters: a chunk of a started prompt that comes more than this many
            iterations after the prompt's previous chunk is interrupted.
        deep_tokens: a chunk that starts at this many cached tokens or more is deep.
        clock: monotonic clock in seconds.
        guard_timing: time the drafter context window guard's calls (``guard_call``).
    """

    def __init__(
        self,
        pp_rank: int,
        pp_size: int,
        interval_s: float = DEFAULT_INTERVAL_S,
        gap_iters: int = DEFAULT_GAP_ITERS,
        deep_tokens: int = DEFAULT_DEEP_TOKENS,
        clock: Callable[[], float] = time.monotonic,
        guard_timing: bool = False,
    ):
        if pp_size < 1 or not 0 <= pp_rank < pp_size:
            raise ValueError(f"pp_rank {pp_rank} is not a stage of a pipeline of {pp_size}")
        if not interval_s > 0:
            raise ValueError(f"interval_s {interval_s} must be positive")
        if gap_iters < 1:
            raise ValueError(f"gap_iters {gap_iters} must be at least 1")
        if deep_tokens < 1:
            raise ValueError(f"deep_tokens {deep_tokens} must be at least 1")
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.interval_s = interval_s
        self.gap_iters = gap_iters
        self.deep_tokens = deep_tokens
        self._clock = clock
        self._start = clock()
        self._next_line = self._start + interval_s
        self.num_lines = 0
        # Stage agreement, first rank: per proposal round, the stage(s) whose
        # answer bound it (stage 0: the first rank's own re-claim) and the
        # tokens lost; per admitted first chunk, the stage(s) with the smallest
        # own prefix and the tokens lost against the largest one.
        self.agree_rounds = 0
        self.agree_rounds_lowered = 0
        self.agree_rounds_tied = 0
        self.agree_round_bound = [0] * pp_size
        self.agree_round_lost_tokens = [0] * pp_size
        self.agree_admitted = 0
        self.agree_admitted_below_best = 0
        self.agree_admitted_without_prefixes = 0
        self.agree_binder = [0] * pp_size
        self.agree_binder_lost_tokens = [0] * pp_size
        self.agree_lost_vs_best_tokens = 0
        self.agree_lost_vs_leader_tokens = 0
        self.agree_held_chunk_iters = 0
        # Reservations, every rank.
        self.resv_made = 0
        self.resv_lowered = 0
        self.resv_lowered_tokens = 0
        self.resv_dropped = 0
        self.resv_dropped_tokens = 0
        self.resv_failed = 0
        self.adopt_first = 0
        self.adopt_fresh_claims = 0
        self.adopt_reclaims = 0
        # Context caches, every rank, per kind (first chunk / started prompt).
        self.ctx_suspended = {kind: 0 for kind in KINDS}
        self.ctx_evicted = {kind: 0 for kind in KINDS}
        self.ctx_preempted = {kind: 0 for kind in KINDS}
        self.ctx_resumed = {kind: 0 for kind in KINDS}
        self.ctx_resume_refused = {kind: 0 for kind in KINDS}
        self.ctx_reservations_suspended = 0
        self.ctx_reservations_page_refused = 0
        self.resume_onboarded_pages: Dict[str, Dict[int, int]] = {kind: {} for kind in KINDS}
        self.resume_host_s = {kind: 0.0 for kind in KINDS}
        self.resume_host_s_max = 0.0
        self.resume_pages_unknown = 0
        # Admission holds.
        self.admission_hold_iters = 0
        self.admission_held_chunk_iters = 0
        self.admission_held_requests = 0
        self.admission_held_now = 0
        self._admission_held_ids: set = set()
        # Context chunks, and the gaps between the chunks of one prompt.
        self.iteration = 0
        self.chunks = {kind: 0 for kind in KINDS}
        self.gaps_shallow = [0] * len(GAP_BUCKETS)
        self.gaps_deep = [0] * len(GAP_BUCKETS)
        self.interrupted = 0
        self.interrupted_deep = 0
        self.interrupted_max_gap = 0
        self._last_chunk_iteration: Dict[int, int] = {}
        # Sample-state relay, every rank.
        self.drain_waits = 0
        self.drain_wait_s = 0.0
        self.drain_wait_s_max = 0.0
        self.relay_recvs = 0
        self.relay_recv_wait_s = 0.0
        self.sync_prevs = 0
        self.sync_prev_s = 0.0
        self.relay_compact_sends = 0
        self.relay_full_sends = 0
        self.relay_bytes_max = 0
        # Drafter context window guard calls (with ``guard_timing``).
        self.guard_timing = bool(guard_timing)
        self.guard_calls = {kind: 0 for kind in GUARD_CALL_KINDS}
        self.guard_host_s = {kind: 0.0 for kind in GUARD_CALL_KINDS}
        self.guard_host_s_max = 0.0
        self.guard_prefix_hashes = 0

    # ------------------------------------------------------- stage agreement

    def agreement_round(self, depth: int, answers: Mapping[int, int]) -> None:
        """A proposal round at ``depth`` that every follower answered with a claim.

        Args:
            depth: the depth proposed.
            answers: follower stage -> depth it claimed.
        """
        self.agree_rounds += 1
        common = min(answers.values())
        if common >= depth:
            return
        self.agree_rounds_lowered += 1
        binders = [stage for stage, claimed in answers.items() if claimed == common]
        if len(binders) > 1:
            self.agree_rounds_tied += 1
        for stage in binders:
            self.agree_round_bound[stage] += 1
            self.agree_round_lost_tokens[stage] += depth - common

    def agreement_reclaim(self, allowed: int, claimed: Optional[int]) -> None:
        """The first rank re-claimed at most ``allowed`` tokens and got ``claimed`` (None: none)."""
        if claimed is not None and claimed < allowed:
            self.agree_round_bound[0] += 1
            self.agree_round_lost_tokens[0] += allowed - claimed

    def agreement_admitted(
        self, position: int, first_depth: int, follower_prefixes: Mapping[int, int]
    ) -> None:
        """A first chunk admitted at ``position`` under the stage agreement.

        Args:
            position: the depth the chunk was admitted at.
            first_depth: the first rank's own uncapped claim.
            follower_prefixes: follower stage -> its own uncapped reusable
                prefix (from its first answer; a stage that sent none is missing).
        """
        self.agree_admitted += 1
        self.agree_lost_vs_leader_tokens += max(first_depth - position, 0)
        own = {0: first_depth}
        own.update(follower_prefixes)
        if len(own) < self.pp_size:
            self.agree_admitted_without_prefixes += 1
        best = max(own.values())
        if position >= best:
            return
        self.agree_admitted_below_best += 1
        self.agree_lost_vs_best_tokens += best - position
        low = min(own.values())
        for stage, prefix in own.items():
            if prefix == low:
                self.agree_binder[stage] += 1
                self.agree_binder_lost_tokens[stage] += best - position

    def agreement_held(self, num_held: int) -> None:
        """First chunks the stage agreement held out of this iteration's pass."""
        self.agree_held_chunk_iters += num_held

    # ---------------------------------------------------------- reservations

    def reservation_claimed(self, held: Optional[int], claimed: Optional[int]) -> None:
        """A reservation claim: the depth ``held`` before it (None: none) and the
        depth ``claimed`` (None: no cache could be created)."""
        if claimed is None:
            self.resv_failed += 1
            if held is not None:
                self.resv_dropped += 1
                self.resv_dropped_tokens += held
            return
        if held is None:
            self.resv_made += 1
        elif claimed < held:
            self.resv_lowered += 1
            self.resv_lowered_tokens += held - claimed

    def reservation_released(self, held: Optional[int]) -> None:
        """A reservation dropped; ``held`` is the depth it held (None: there was none)."""
        if held is not None:
            self.resv_dropped += 1
            self.resv_dropped_tokens += held

    def first_chunk_adopted(self, held: Optional[int], position: int) -> None:
        """A follower adopts a first chunk at ``position`` holding a reservation at ``held``."""
        self.adopt_first += 1
        if held != position:
            self.adopt_fresh_claims += 1

    def first_chunk_reclaimed(self) -> None:
        """A follower's adoption dropped the cache it resumed and claimed again."""
        self.adopt_reclaims += 1

    # ------------------------------------------------- suspend, evict, resume

    @staticmethod
    def kind_of(started: bool) -> str:
        """The counter kind of a context cache: a started prompt's or a first chunk's."""
        return STARTED if started else FIRST

    def context_suspended(self, started: bool, evicted: bool = False) -> None:
        """An active context cache suspended (``evicted``: for a generation step)."""
        counts = self.ctx_evicted if evicted else self.ctx_suspended
        counts[self.kind_of(started)] += 1

    def reservation_suspended(self, page_refused: bool = False) -> None:
        """A reserved first chunk refused admission and kept, its caches suspended.

        Args:
            page_refused: the KV pool refused it pages (else the token budget
                or another limit of the pass refused it).
        """
        self.ctx_reservations_suspended += 1
        self.ctx_reservations_page_refused += int(page_refused)

    def context_preempted(self, started: bool) -> None:
        """A context request whose cache was given up for a re-prefill."""
        self.ctx_preempted[self.kind_of(started)] += 1

    def context_resumed(
        self,
        started: bool,
        resumed: bool,
        pages_by_level: Optional[Sequence[int]],
        seconds: float,
        count_event: bool = True,
    ) -> None:
        """A suspended context cache resumed (or refused) on this rank.

        Args:
            started: a started prompt's cache (else a first chunk's).
            resumed: whether the resume succeeded.
            pages_by_level: the cache's active pages per cache level (GPU first)
                just before the resume, or None when unknown; on success the
                pages of every lower level were brought onto the GPU.
            seconds: host time of the resume call.
            count_event: count the event (False for a second KV cache manager
                of the same request, whose pages and time still add up).
        """
        kind = self.kind_of(started)
        self.resume_host_s[kind] += seconds
        self.resume_host_s_max = max(self.resume_host_s_max, seconds)
        if not resumed:
            if count_event:
                self.ctx_resume_refused[kind] += 1
            return
        if count_event:
            self.ctx_resumed[kind] += 1
        if pages_by_level is None:
            self.resume_pages_unknown += 1
            return
        onboarded = self.resume_onboarded_pages[kind]
        for level, pages in enumerate(pages_by_level):
            if level > 0 and pages:
                onboarded[level] = onboarded.get(level, 0) + int(pages)

    # -------------------------------------------------------- admission holds

    def admission_holds(self, held: Iterable[int]) -> None:
        """First chunks an admission back-pressure held out of this iteration's pass.

        Args:
            held: the request ids held this iteration (empty: none).
        """
        held = set(held)
        self.admission_held_now = len(held)
        if not held:
            return
        self.admission_hold_iters += 1
        self.admission_held_chunk_iters += len(held)
        new = held - self._admission_held_ids
        self.admission_held_requests += len(new)
        self._admission_held_ids |= new

    # ----------------------------------------------------- sample-state relay

    def relay_drain_waited(self, seconds: float) -> None:
        """Host time of one pre-forward drain of the relay sends."""
        self.drain_waits += 1
        self.drain_wait_s += seconds
        self.drain_wait_s_max = max(self.drain_wait_s_max, seconds)

    def relay_recv_waited(self, seconds: float) -> None:
        """Host time of one relay receive."""
        self.relay_recvs += 1
        self.relay_recv_wait_s += seconds

    def sampler_sync_waited(self, seconds: float) -> None:
        """Host time of one synchronize on the previous micro-batch's sampler event."""
        self.sync_prevs += 1
        self.sync_prev_s += seconds

    def relay_sent(self, compact: bool, payload_bytes: int) -> None:
        """One relay message sent, compact or full, carrying ``payload_bytes``
        bytes of sample state (``pp_spec_relay.relay_payload_bytes``)."""
        if compact:
            self.relay_compact_sends += 1
        else:
            self.relay_full_sends += 1
        self.relay_bytes_max = max(self.relay_bytes_max, payload_bytes)

    # ------------------------------------------- drafter context window guard

    def guard_call(self, kind: str, seconds: float, prefix_hashes: int) -> None:
        """One call into the drafter context window guard (with ``guard_timing``).

        Args:
            kind: one of ``GUARD_CALL_KINDS``.
            seconds: host time of the call.
            prefix_hashes: the guard's cumulative prefix hash count after it.

        Raises:
            ValueError: ``kind`` is not a guard call kind.
        """
        if kind not in self.guard_calls:
            raise ValueError(f"unknown guard call kind {kind!r}")
        self.guard_calls[kind] += 1
        self.guard_host_s[kind] += seconds
        self.guard_host_s_max = max(self.guard_host_s_max, seconds)
        self.guard_prefix_hashes = prefix_hashes

    # --------------------------------------------------------- context chunks

    def record_context_chunks(self, requests: Iterable[Any]) -> None:
        """The context chunks of this iteration's batch; call once per loop iteration.

        Args:
            requests: the batch's context requests, each at the start of its
                chunk (``py_request_id``, ``context_current_position``,
                ``is_first_context_chunk``, ``is_last_context_chunk``).
        """
        iteration = self.iteration
        self.iteration += 1
        for req in requests:
            request_id = req.py_request_id
            first = bool(req.is_first_context_chunk)
            self.chunks[self.kind_of(not first)] += 1
            previous = self._last_chunk_iteration.get(request_id)
            if previous is not None and not first:
                gap = iteration - previous
                deep = req.context_current_position >= self.deep_tokens
                (self.gaps_deep if deep else self.gaps_shallow)[gap_bucket(gap)] += 1
                if gap > self.gap_iters:
                    self.interrupted += 1
                    self.interrupted_deep += int(deep)
                    self.interrupted_max_gap = max(self.interrupted_max_gap, gap)
            if req.is_last_context_chunk:
                self._last_chunk_iteration.pop(request_id, None)
            else:
                self._last_chunk_iteration[request_id] = iteration

    def forget_except(self, live_request_ids: Iterable[int]) -> None:
        """Drop the per-request state of requests no longer live (finished, cancelled)."""
        live = set(live_request_ids)
        for request_id in [r for r in self._last_chunk_iteration if r not in live]:
            del self._last_chunk_iteration[request_id]
        self._admission_held_ids &= live

    # ----------------------------------------------------------------- lines

    def due(self) -> bool:
        """Whether a periodic line is due; moves the next due time on if so."""
        now = self._clock()
        if now < self._next_line:
            return False
        while self._next_line <= now:
            self._next_line += self.interval_s
        return True

    def config_line(self) -> str:
        """The line logged once when the counters are built."""
        return (
            f"{LINE_TAG} kind=config pp_rank={self.pp_rank} pp_size={self.pp_size} "
            f"interval_s={self.interval_s:g} gap_iters={self.gap_iters} "
            f"deep_tokens={self.deep_tokens} gap_buckets="
            + ",".join(label for _, _, label in GAP_BUCKETS)
            + (" guard_timing=1" if self.guard_timing else "")
        )

    def line(self, kind: str, queue: Optional[Mapping[str, int]] = None) -> str:
        """One line of every cumulative count.

        Args:
            kind: ``periodic`` or ``final``.
            queue: point-in-time queue sizes to append, as ``q_<name>`` fields.
        """
        self.num_lines += 1
        fields: List[Tuple[str, Any]] = [
            ("kind", kind),
            ("pp_rank", self.pp_rank),
            ("pp_size", self.pp_size),
            ("seq", self.num_lines),
            ("elapsed_s", f"{self._clock() - self._start:.1f}"),
            ("iter", self.iteration),
            ("agree_rounds", self.agree_rounds),
            ("agree_rounds_lowered", self.agree_rounds_lowered),
            ("agree_rounds_tied", self.agree_rounds_tied),
            ("agree_round_bound", _per_stage(self.agree_round_bound)),
            ("agree_round_lost_tokens", _per_stage(self.agree_round_lost_tokens)),
            ("agree_admitted", self.agree_admitted),
            ("agree_admitted_below_best", self.agree_admitted_below_best),
            ("agree_admitted_without_prefixes", self.agree_admitted_without_prefixes),
            ("agree_binder", _per_stage(self.agree_binder)),
            ("agree_binder_lost_tokens", _per_stage(self.agree_binder_lost_tokens)),
            ("agree_lost_vs_best_tokens", self.agree_lost_vs_best_tokens),
            ("agree_lost_vs_leader_tokens", self.agree_lost_vs_leader_tokens),
            ("agree_held_chunk_iters", self.agree_held_chunk_iters),
            ("resv_made", self.resv_made),
            ("resv_lowered", self.resv_lowered),
            ("resv_lowered_tokens", self.resv_lowered_tokens),
            ("resv_dropped", self.resv_dropped),
            ("resv_dropped_tokens", self.resv_dropped_tokens),
            ("resv_failed", self.resv_failed),
            ("adopt_first", self.adopt_first),
            ("adopt_fresh_claims", self.adopt_fresh_claims),
            ("adopt_reclaims", self.adopt_reclaims),
        ]
        for name in KINDS:
            fields += [
                (f"ctx_suspended_{name}", self.ctx_suspended[name]),
                (f"ctx_evicted_{name}", self.ctx_evicted[name]),
                (f"ctx_preempted_{name}", self.ctx_preempted[name]),
                (f"ctx_resumed_{name}", self.ctx_resumed[name]),
                (f"ctx_resume_refused_{name}", self.ctx_resume_refused[name]),
                (f"resume_onboarded_pages_{name}", _per_level(self.resume_onboarded_pages[name])),
                (f"resume_host_ms_{name}", f"{self.resume_host_s[name] * 1e3:.3f}"),
            ]
        fields += [
            ("ctx_reservations_suspended", self.ctx_reservations_suspended),
            ("ctx_reservations_page_refused", self.ctx_reservations_page_refused),
            ("resume_host_ms_max", f"{self.resume_host_s_max * 1e3:.3f}"),
            ("resume_pages_unknown", self.resume_pages_unknown),
            ("admission_hold_iters", self.admission_hold_iters),
            ("admission_held_chunk_iters", self.admission_held_chunk_iters),
            ("admission_held_requests", self.admission_held_requests),
            ("admission_held_now", self.admission_held_now),
            ("chunks_first", self.chunks[FIRST]),
            ("chunks_started", self.chunks[STARTED]),
            ("gaps_shallow", _per_bucket(self.gaps_shallow)),
            ("gaps_deep", _per_bucket(self.gaps_deep)),
            ("interrupted", self.interrupted),
            ("interrupted_deep", self.interrupted_deep),
            ("interrupted_max_gap", self.interrupted_max_gap),
            ("drain_waits", self.drain_waits),
            ("drain_wait_ms", f"{self.drain_wait_s * 1e3:.3f}"),
            ("drain_wait_ms_max", f"{self.drain_wait_s_max * 1e3:.3f}"),
            ("relay_recvs", self.relay_recvs),
            ("relay_recv_wait_ms", f"{self.relay_recv_wait_s * 1e3:.3f}"),
            ("sync_prevs", self.sync_prevs),
            ("sync_prev_ms", f"{self.sync_prev_s * 1e3:.3f}"),
            ("relay_compact_sends", self.relay_compact_sends),
            ("relay_full_sends", self.relay_full_sends),
            ("relay_bytes_max", self.relay_bytes_max),
        ]
        if self.guard_timing:
            for name in GUARD_CALL_KINDS:
                fields += [
                    (f"guard_calls_{name}", self.guard_calls[name]),
                    (f"guard_host_ms_{name}", f"{self.guard_host_s[name] * 1e3:.3f}"),
                ]
            fields += [
                ("guard_host_ms_max", f"{self.guard_host_s_max * 1e3:.3f}"),
                ("guard_prefix_hashes", self.guard_prefix_hashes),
            ]
        fields += [(f"q_{name}", count) for name, count in (queue or {}).items()]
        return LINE_TAG + " " + " ".join(f"{name}={value}" for name, value in fields)
