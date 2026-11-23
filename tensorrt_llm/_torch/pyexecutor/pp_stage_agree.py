# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline stages agree on the reused prefix of every context chunk.

Under pipeline parallelism the first rank schedules each micro-batch and the
other ranks run it. With block reuse each stage owns a different KV pool -- its
own layers, its own sliding-window / full-attention mix, its own eviction
history -- so the prefix a stage can reuse for a new prompt need not be the
prefix the first rank reused. A stage that starts a request at another position
runs its layers on hidden states that the previous stage computed for other
tokens, at other RoPE positions and KV slots: the KV it writes, and that of
every later stage, is silently wrong.

On a disaggregated server at ``pipeline_parallel_size > 1`` whose layout the agreement
covers (``pp_stage_agree_layout``; the executor adds its own checks) every stage claims
the same prefix, exactly:

* **Exact depths.** A reuse claim capped at a depth ``m`` does not always return
  ``m``. The KV core prunes a match to the largest endpoint ``<= m`` at which
  every attention life cycle still holds the pages that endpoint needs: every
  block for full attention; the sink blocks and the blocks of the trailing
  window for a sliding window (``BlockRadixTree::pruneMatch``). The interior
  window pages of a committed prompt are usually gone, so on a stage with
  sliding-window layers a cap at an interior depth comes back shorter. The
  depth every stage can claim exactly is therefore found by asking each stage
  to claim a proposed depth: the first rank proposes its own claim; each
  follower claims at most that and answers what it got; if some stage got less,
  the first rank claims at most the smallest answer itself and proposes what it
  got. A round can only lower the depth, and no round goes below the largest
  depth valid on every stage, so the rounds stop there; depth 0 is valid
  everywhere.
* **Held claims.** The claims are *reservations*: KV caches created with the
  reuse match but not resumed. A created cache holds the pages of its matched
  prefix: a held page is never dropped (it can still move down to a lower cache
  tier, and stays where it is on the last one), so nothing evicts the agreed
  prefix between a stage's answer and the chunk; admitting the chunk resumes
  the reservation at exactly the agreed depth. The number of outstanding
  reservations is bounded (``max_reserved``).
* **Given back under page pressure.** Held prefixes that fill the last cache
  tier would leave no room for any chunk (every admission refused, the V2
  scheduler's deadlock check firing). When the first rank's pass schedules
  nothing and the pool refused pages, or a pass preempts or pauses a request,
  the first rank halves the outstanding reservations (unconfirmed ones and the
  newest first, never the oldest one) and lowers the bound with them; each pass
  without pressure raises the bound by one again. When the only reservation is
  itself refused pages in such a pass, it is given back too and tried again
  after every other waiting first chunk, after one pass with no reservation at
  all. Followers drop a reservation when the schedule message carries ``None``
  for it, so a follower never holds a reservation the first rank gave back
  before that message.
* **Followers' page pressure.** The same request pins different pages on
  different stages (an all-sliding-window stage keeps only a window of a
  prefix, a full-attention stage all of it), so the first rank's pool does not
  show a follower's pressure. A follower that already holds a reservation
  declines a further one that would leave it less room than one more first
  chunk needs: it releases it and answers that it claimed nothing, which the
  first rank treats as a failed proposal (asked again) and as a pass under
  page pressure (the halving above, the declined ones given back first). A
  follower never declines its only reservation -- with one reservation it pins
  the prefix a stock admission would claim -- so once the first rank is down
  to one reservation, the followers confirm it.
* **No new coupling.** Proposals ride on the schedule message (first rank to
  followers, every iteration). Answers ride on one small point-to-point report
  per follower and iteration, sent once the follower adopted that iteration's
  schedule; the first rank receives it ``lag`` iterations later
  (``probe_lag``), by which time the sample-state relay has already made it
  wait for the followers to get that far, so in steady state the receive does
  not block. A blocking collective at the top of each iteration would instead
  make the first rank wait for the last stage's previous micro-batch.
* Answers carry the iteration of the proposal they answer, so an answer to a
  proposal given back and made again at the same depth is not taken for an
  answer to the new one.
* The first rank holds a first chunk out of its scheduler pass until every stage
  confirmed the same depth, admits it capped at that depth, and takes it back
  out of the batch if its own claim landed elsewhere (a failed admission
  released its reservation and the fresh claim found less). Followers adopt the
  chunk from their reservation, and every context chunk's ``(position, size)``
  is compared with the first rank's.
* Context blocks are committed up to the end of the chunk being handled, not
  the live cursor, which under PP is ahead of it.
* **Followers release first.** The first rank applies a lowered or
  dropped reservation in ``step``, before its scheduler pass; a follower applies
  the same proposal only after it adopted that pass's batch. Without a cache
  tier below the GPU a held prefix counts as used, so for that one iteration a
  follower holds pages the first rank no longer held when it admitted the
  batch, and the KV core's resume gate (``max_util_for_resume``) can refuse a
  reserved first chunk the first rank resumed. With block reuse a follower
  therefore applies the releasing part of the message -- drops, and proposals
  below the depth it holds -- before it adopts the batch (``pre_release``); no
  such proposal is for a request in that batch. Its answers are unchanged
  (``answer_proposals`` does not claim them again).
* **Shortest-uncached-first reservations.** In arrival order, the first rank
  would hold every waiting first chunk without a confirmed reservation out of
  its scheduler pass, so the scheduler's shortest-prefill-first order would
  only ever see the ``max_reserved`` oldest first chunks. Under a long
  admission queue a warm follow-up turn would then wait behind cold prompts
  until its prefix is evicted. With block reuse the first rank therefore
  reserves the waiting first chunks with the fewest uncached tokens first (one
  prefix probe per request, memoized and refreshed by the reservation's own
  claim). A first chunk that has waited ``SPF_MAX_WAIT_ITERATIONS`` iterations
  is aged and goes ahead of the others in arrival order, but at most
  ``spf_aged_slots`` aged ones hold a reservation at a time, so a queue that
  waits longer than the aging bound does not fall back to arrival order. Only
  the order of new reservations changes; the protocol, the followers and the
  give-back under pressure (which keeps the first reservation of that order) do
  not.

Elsewhere (``pipeline_parallel_size == 1``, aggregated serving, or a layout the
agreement does not cover) none of this runs and the PP loop is unchanged.
"""

from collections import deque
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Set, Tuple

# Where the agreement is on, the first PP rank also sends each context request's
# position and chunk size with its schedule, and every other rank runs exactly
# the first rank's batch (``PyExecutor._pp_adopt_leader_schedule``): it adopts
# those chunks and, with KV cache manager V2, allocates generation steps its own
# pass skipped and undoes what its pass allocated outside the batch. It also
# sends the order in which its scheduler pass allocated the context chunks, and
# the other ranks adopt the chunks in that order rather than in the batch order:
# the batch order puts non-last chunks before last chunks
# (``KVCacheV2Scheduler._sort_requests``), so a follower could otherwise grow a
# started prompt before it resumes a reserved first chunk that the first rank
# resumed first, and a KV cache resume is refused above a pool utilization
# (``max_util_for_resume``) rather than when it does not fit.
#
# On the same layout, a pipeline stage whose own layers are all sliding-window
# keeps its quota in the KV cache manager V2 cross-rank capacity sync instead of
# being clamped to its fixed cost (``KVCacheManagerV2``), as an all-SWA stage
# clamped to its fixed cost cannot allocate the chunks it adopts.

# Iterations a first chunk waits before it is aged in the shortest-uncached-first
# reservation order (goes ahead in arrival order).
SPF_MAX_WAIT_ITERATIONS = 512

# Default bound on the first chunks the first rank has proposed or confirmed but
# not admitted yet. Each holds its matched prefix pages on every stage (never
# dropped; pinned where they are on the last cache tier) and one KV index slot,
# so the bound caps what reservations can pin; it needs to cover what the first
# rank admits in about ``lag`` iterations.
DEFAULT_MAX_RESERVED = 16

# The first rank logs, at INFO, this many admitted first chunks on which the
# stages disagreed; later ones are only counted in the summary logged when the
# PP loop ends.
MAX_LOGGED_DISAGREEMENTS = 16

# Every rank logs, at INFO, the pages this many of its reservations pin
# ("PP stage agreement: reservation pins ..."), per pool group and cache level.
MAX_LOGGED_PINS = 16

# Set on a request to cap its next first-chunk reuse claim
# (read by KVCacheManagerV2.prepare_context and KVCacheV2Scheduler).
REUSE_CAP_ATTR = "py_pp_reuse_cap"

# Set on a request while it holds a reservation: the depth reserved, which
# KVCacheManagerV2.probe_context_reuse reports for it (the shortest-prefill-first
# order would otherwise see a request holding a cache as one with no reuse).
RESERVED_DEPTH_ATTR = "py_pp_reserved_depth"


def pp_stage_agree_layout(
    *,
    pp_size: int,
    tp_size: int,
    cp_size: int,
    enable_attention_dp: bool,
    disaggregated: bool,
) -> bool:
    """Whether a deployment has the layout the stage agreement covers.

    True for a disaggregated server (one with a KV cache transceiver) at
    ``pipeline_parallel_size > 1`` with ``tensor_parallel_size ==
    context_parallel_size == 1`` and no attention DP. The executor adds the
    checks only it can make (KV cache manager V2, no KV cache connector, a KV
    cache manager that stops its context commit at the handled chunk) and runs
    without the agreement where one fails. False whenever ``pp_size == 1``.

    Args:
        pp_size: pipeline-parallel size.
        tp_size: tensor-parallel size.
        cp_size: context-parallel size.
        enable_attention_dp: whether attention DP is on.
        disaggregated: whether the server takes part in disaggregated serving.
    """
    return (
        pp_size > 1
        and tp_size == 1
        and cp_size == 1
        and not enable_attention_dp
        and bool(disaggregated)
    )


def spf_aged_slots(max_reserved: int) -> int:
    """Reservations aged first chunks may hold at a time: a quarter of ``max_reserved``, at least 1.

    Args:
        max_reserved: the bound on outstanding reservations.
    """
    return max(1, max_reserved // 4)


def reuse_cap(req: Any) -> Optional[int]:
    """The cross-stage reuse cap set on ``req``, or None."""
    return getattr(req, REUSE_CAP_ATTR, None)


def set_reuse_cap(req: Any, cap: Optional[int]) -> None:
    """Cap (or, with None, uncap) the next first-chunk reuse claim of ``req``."""
    setattr(req, REUSE_CAP_ATTR, cap)


def reserved_depth(req: Any) -> Optional[int]:
    """The depth ``req`` holds a reservation at, or None."""
    return getattr(req, RESERVED_DEPTH_ATTR, None)


def set_reserved_depth(req: Any, depth: Optional[int]) -> None:
    """Record (or, with None, clear) the depth of the reservation ``req`` holds."""
    setattr(req, RESERVED_DEPTH_ATTR, depth)


def is_waiting_context(req: Any) -> bool:
    """A context-init request other than a dummy: its prefix may be claimed later."""
    return (
        bool(getattr(req, "is_context_init_state", False))
        and not getattr(req, "is_dummy_request", False)
        and not getattr(req, "is_attention_dp_dummy", False)
    )


def is_waiting_first_chunk(req: Any) -> bool:
    """A waiting context request whose first chunk has not run yet."""
    return is_waiting_context(req) and req.is_first_context_chunk


def probe_lag(relay_lag: int) -> int:
    """Iterations between a follower's report and the first rank's receive of it.

    Before it schedules iteration ``t``, the first rank has received the sample
    state that the last stage relayed at iteration ``t + 1 - relay_lag``
    (``pp_ring.relay_lag``); every other follower is at least as far as the last
    stage. A report sent in that iteration has therefore arrived.

    Args:
        relay_lag: the sample-state relay lag of the ring.
    """
    return max(relay_lag - 1, 1)


# A reservation callback: claim the prefix of ``req`` capped at ``cap`` (None:
# uncapped) as a held, not resumed, cache -- dropping any reservation ``req``
# held before -- and return the depth claimed, or None when no cache could be
# created this iteration.
Reserve = Callable[[Any, Optional[int]], Optional[int]]

# One follower answer: (proposed depth, depth claimed or None, the stage's own
# uncapped reusable prefix or None).
Answer = Tuple[int, Optional[int], Optional[int]]


class PPStageAgreement:
    """One rank's state of the cross-stage reuse agreement.

    The executor drives it once per PP loop iteration. The first rank, before
    scheduling: ``accept`` the reports ``reports_due`` names, then ``step``
    (finish rounds, start new ones, get the requests to hold); after its
    scheduler pass: ``check_admitted`` per first chunk, then ``relieve`` (give
    reservations back under page pressure); into the schedule message:
    ``take_proposals``. A follower, after adopting the schedule:
    ``answer_proposals``, then send ``build_report``.

    Args:
        pp_rank: this rank's pipeline stage.
        pp_size: pipeline-parallel size.
        lag: iterations between a report and its receive (``probe_lag``).
        reuse_enabled: whether block reuse is on; without it every stage starts
            every prompt at position 0 and no report is exchanged.
        max_reserved: bound on proposed or confirmed, unadmitted first chunks.
        uncached_estimate: first rank, with block reuse: the
            uncached tokens of a waiting first chunk (prompt length minus this
            rank's reusable prefix, at least 1). With it, new reservations go
            shortest-uncached-first (``_reservation_order``); None keeps the
            arrival order.
        max_wait_iterations: iterations a first chunk waits before it is aged
            (only with ``uncached_estimate``).
        aged_slots: reservations aged first chunks may hold at a time (only with
            ``uncached_estimate``; default ``spf_aged_slots(max_reserved)``).
    """

    def __init__(
        self,
        pp_rank: int,
        pp_size: int,
        lag: int,
        reuse_enabled: bool,
        max_reserved: int = DEFAULT_MAX_RESERVED,
        uncached_estimate: Optional[Callable[[Any], int]] = None,
        max_wait_iterations: int = SPF_MAX_WAIT_ITERATIONS,
        aged_slots: Optional[int] = None,
    ):
        if pp_size < 2:
            raise ValueError("the PP stage agreement needs pipeline_parallel_size > 1")
        if lag < 1:
            raise ValueError(f"report lag {lag} must be at least 1")
        if max_reserved < 1:
            raise ValueError(f"max_reserved {max_reserved} must be at least 1")
        if max_wait_iterations < 0:
            raise ValueError(f"max_wait_iterations {max_wait_iterations} must be at least 0")
        if aged_slots is not None and aged_slots < 1:
            raise ValueError(f"aged_slots {aged_slots} must be at least 1")
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.lag = lag
        self.reuse_enabled = reuse_enabled
        self.max_reserved = max_reserved
        self.iteration = 0
        # First rank: the depth proposed per request, the followers' answers to
        # it (stage -> depth claimed, or None), the depths every stage
        # confirmed, and the proposals for the next schedule message (request
        # id -> depth, or None: drop the reservation).
        self._proposed: Dict[int, int] = {}
        self._answers: Dict[int, Dict[int, Optional[int]]] = {}
        self._confirmed: Dict[int, int] = {}
        self._outgoing: Dict[int, Optional[int]] = {}
        # First rank, per request being agreed: its own first claim, the
        # followers' own uncapped prefixes (first answers), the extra rounds.
        self._first_depth: Dict[int, int] = {}
        self._follower_prefixes: Dict[int, Dict[int, int]] = {}
        self._rounds: Dict[int, int] = {}
        # First rank, per request: the iteration its current proposal went
        # out (answers from other iterations are stale), and the order its
        # reservation started in (the newest are given back first).
        self._proposed_at: Dict[int, int] = {}
        self._started: Dict[int, int] = {}
        self._num_started = 0
        # First rank, page pressure (``relieve``): the bound on outstanding
        # reservations, below ``max_reserved`` after pressure; the requests
        # refused pages while holding the only reservation, tried again after
        # every other waiting first chunk; whether the next step makes no new
        # reservation; this step's order of the waiting first chunks; and
        # whether a full sweep found none admissible alone (the deadlock check
        # then counts the held first chunks, see ``exhausted``).
        self.budget = max_reserved
        self._tried_alone: Dict[int, None] = {}
        self._skip_new = False
        self._order: List[int] = []
        self.exhausted = False
        self._exhausted_waiting: frozenset = frozenset()
        self.num_pressure_passes = 0
        self.num_given_back = 0
        self.num_refused_alone = 0
        self.num_sweeps = 0
        # First rank: requests whose current proposal a follower declined for
        # lack of room (``relieve`` reads them), and the total received.
        self._declined: Set[int] = set()
        self.num_declines_received = 0
        # Followers: the answers for this iteration's report, the reservations
        # declined for lack of room in it, and isend handles the first rank may
        # not have matched yet.
        self._answers_out: Dict[int, Answer] = {}
        self._declined_out: List[int] = []
        self.num_declined = 0
        # Followers, with block reuse (``pre_release``): the
        # proposals of the current message already applied before the
        # adoption (request id -> depth claimed, or None for a drop), and
        # counters for the summary.
        self._pre_applied: Dict[int, Optional[int]] = {}
        self.num_early_releases = 0
        self.num_early_lowerings = 0
        self.max_early_lowered_by = 0
        self.num_early_skipped_in_batch = 0
        # Every rank: reservations whose pinned pages were logged.
        self.num_pin_logs = 0
        self.pending_sends: Deque[Any] = deque()
        self.num_held = 0
        # First-rank accounting of the admitted first chunks.
        self.num_admitted = 0
        self.num_followers_disagreed = 0
        self.num_leader_below = 0
        self.max_follower_spread = 0
        self.num_lowered = 0
        self.max_lowered_by = 0
        self.num_extra_rounds = 0
        self.num_undone = 0
        self.num_retries = 0
        self._num_logged = 0
        # Per-stage counters of the context path (pp_ctx_diag.py), set by the
        # executor; None leaves every step as it is.
        self.diag: Optional[Any] = None
        # First rank, with block reuse: the uncached-token
        # estimate, the aging bound and the aged share of the reservations; per
        # waiting first chunk, the iteration it was first seen waiting and its
        # memoized uncached tokens; counters for the summary.
        self.uncached_estimate = uncached_estimate
        self.max_wait_iterations = max_wait_iterations
        self.aged_slots = spf_aged_slots(max_reserved) if aged_slots is None else aged_slots
        self._first_seen: Dict[int, int] = {}
        self._uncached: Dict[int, int] = {}
        self.num_out_of_order = 0
        self.num_aged_reserved = 0
        self.num_probes = 0
        # First rank, opt-in, set by the caller: whether a waiting first chunk should wait; such
        # chunks are reserved after the others (aged ones and the refused-alone tail keep their
        # places).
        # None leaves the order as it is.
        self.deprioritize: Optional[Callable[[int], bool]] = None
        self.num_deprioritized = 0

    def _deprioritized_last(self, request_ids: List[int]) -> List[int]:
        """``request_ids`` with the ones ``deprioritize`` marks moved to the end, order kept."""
        if self.deprioritize is None:
            return request_ids
        front = [r for r in request_ids if not self.deprioritize(r)]
        if len(front) == len(request_ids):
            return request_ids
        back = [r for r in request_ids if self.deprioritize(r)]
        self.num_deprioritized += len(back)
        return front + back

    @property
    def spf_order(self) -> bool:
        """Whether new reservations go shortest-uncached-first."""
        return self.uncached_estimate is not None

    def _uncached_tokens(self, req: Any) -> int:
        """The memoized uncached tokens of a waiting first chunk (probed once)."""
        request_id = req.py_request_id
        estimate = self._uncached.get(request_id)
        if estimate is None:
            self.num_probes += 1
            estimate = self._uncached[request_id] = self.uncached_estimate(req)
        return estimate

    def _is_aged(self, request_id: int) -> bool:
        return self.iteration - self._first_seen[request_id] >= self.max_wait_iterations

    def _reservation_order(self, waiting: Mapping[int, Any]) -> List[int]:
        """This step's order of the waiting first chunks for new reservations.

        Without ``uncached_estimate``: arrival order. With it: the aged first
        chunks in arrival order -- those already holding a reservation, and
        new ones while fewer than ``aged_slots`` aged ones hold one --, then
        every other one by (uncached tokens, arrival). Either way the ones
        refused pages while holding the only reservation come last, in the
        order they were refused.
        """
        fresh = [r for r in waiting if r not in self._tried_alone]
        tail = list(self._tried_alone)
        if not self.spf_order:
            return self._deprioritized_last(fresh) + tail
        aged = [r for r in fresh if self._is_aged(r)]
        held_aged = {r for r in aged if r in self._proposed or r in self._confirmed}
        quota = max(self.aged_slots - len(held_aged), 0)
        new_aged = [r for r in aged if r not in held_aged][:quota]
        first = held_aged.union(new_aged)
        arrival = {r: i for i, r in enumerate(fresh)}
        rest = sorted(
            (r for r in fresh if r not in first),
            key=lambda r: (self._uncached_tokens(waiting[r]), arrival[r]),
        )
        return [r for r in aged if r in first] + self._deprioritized_last(rest) + tail

    @property
    def follower_stages(self) -> List[int]:
        return list(range(1, self.pp_size))

    def reports_due(self) -> Optional[int]:
        """Iteration whose follower reports the first rank receives now, or None."""
        due = self.iteration - self.lag
        return due if due >= 0 else None

    def reports_in_flight(self) -> int:
        """Reports per follower sent but not yet received by the first rank."""
        return min(self.lag, self.iteration)

    # ------------------------------------------------------------- first rank

    def accept(self, stage: int, report: Mapping[str, Any], expected_iteration: int) -> None:
        """Record a follower's report on the first rank.

        An answer counts only if it answers the proposal currently out for its
        request -- the same depth, sent in the iteration the report answers --;
        answers to an abandoned proposal are dropped, also when a later proposal
        has the same depth.

        Raises:
            RuntimeError: the report is not the one due now, i.e. the ranks ran
                different numbers of loop iterations.
        """
        if report["iteration"] != expected_iteration:
            raise RuntimeError(
                f"PP stage {stage} sent its reuse report of iteration "
                f"{report['iteration']}, but the first rank expected iteration "
                f"{expected_iteration}; the pipeline ranks are out of step."
            )
        declined = report.get("declined", ())
        for request_id, (depth, claimed, own_prefix) in report["answers"].items():
            if (
                self._proposed.get(request_id) != depth
                or self._proposed_at.get(request_id) != report["iteration"]
            ):
                continue
            self._answers.setdefault(request_id, {})[stage] = claimed
            if request_id in declined:
                self._declined.add(request_id)
                self.num_declines_received += 1
            if own_prefix is not None and self._rounds.get(request_id, 0) == 0:
                self._follower_prefixes.setdefault(request_id, {})[stage] = own_prefix

    def _propose(self, request_id: int, depth: int) -> None:
        self._proposed[request_id] = depth
        self._proposed_at[request_id] = self.iteration
        self._answers[request_id] = {}
        self._outgoing[request_id] = depth

    def _forget(self, request_id: int) -> None:
        self._proposed.pop(request_id, None)
        self._answers.pop(request_id, None)
        self._confirmed.pop(request_id, None)
        self._first_depth.pop(request_id, None)
        self._follower_prefixes.pop(request_id, None)
        self._rounds.pop(request_id, None)
        self._proposed_at.pop(request_id, None)
        self._started.pop(request_id, None)

    def _drop(self, request_id: int) -> None:
        """Forget a request this rank no longer holds, and tell the followers to
        drop theirs (a proposal for it later in the same message replaces this)."""
        self._forget(request_id)
        self._outgoing[request_id] = None

    def step(
        self,
        waiting_first_chunks: Iterable[Any],
        reserve: Reserve,
        holds_reservation: Callable[[Any], bool],
    ) -> Set[int]:
        """The first rank's agreement step for this iteration, before scheduling.

        Args:
            waiting_first_chunks: the waiting first chunks, oldest first.
            reserve: claims and holds a request's prefix on this rank.
            holds_reservation: whether this rank still holds a request's reservation.

        Returns:
            The request ids to hold out of this iteration's scheduler pass: every
            waiting first chunk whose depth is not confirmed on every stage.
        """
        waiting = {req.py_request_id: req for req in waiting_first_chunks}
        for request_id in [r for r in (*self._proposed, *self._confirmed) if r not in waiting]:
            # Cancelled or finished while being agreed: the request's normal
            # release frees the reservation on every stage.
            self._forget(request_id)
        if self.spf_order:
            # Not in ``_forget``: a reservation given back is made again later,
            # and keeps its waiting clock and estimate until then.
            for request_id in [r for r in self._first_seen if r not in waiting]:
                del self._first_seen[request_id]
            for request_id in [r for r in self._uncached if r not in waiting]:
                del self._uncached[request_id]
            for request_id in waiting:
                self._first_seen.setdefault(request_id, self.iteration)
        self._tried_alone = {r: None for r in self._tried_alone if r in waiting}
        if self.exhausted and frozenset(waiting) != self._exhausted_waiting:
            self.exhausted = False

        for request_id in list(self._confirmed):
            if not holds_reservation(waiting[request_id]):
                # A failed admission on this rank released the reservation; the
                # next claim may land elsewhere, so agree again (the followers
                # drop theirs unless a new proposal goes out below).
                self._drop(request_id)
                self.num_undone += 1

        for request_id, depth in list(self._proposed.items()):
            answers = self._answers.get(request_id, {})
            if len(answers) < len(self.follower_stages):
                continue
            req = waiting[request_id]
            if any(claimed is None for claimed in answers.values()):
                # A follower could not create the cache this iteration: ask again.
                self.num_retries += 1
                self._propose(request_id, depth)
                continue
            common = min(answers.values())
            if self.diag is not None:
                self.diag.agreement_round(depth, answers)
            if common == depth and holds_reservation(req):
                del self._proposed[request_id]
                self._answers.pop(request_id, None)
                self._confirmed[request_id] = depth
                continue
            # A stage claims less (or this rank lost its reservation): claim at
            # most that here and propose what this rank got, which its own
            # pruning may lower further.
            claimed = reserve(req, min(common, depth))
            if self.diag is not None:
                self.diag.agreement_reclaim(min(common, depth), claimed)
            self._rounds[request_id] = self._rounds.get(request_id, 0) + 1
            if claimed is None:
                self._drop(request_id)
                continue
            self._propose(request_id, claimed)

        # Waiting first chunks oldest first (or shortest-uncached-first, see
        # ``_reservation_order``), the ones refused pages while holding the
        # only reservation last (in the order they were refused).
        self._order = self._reservation_order(waiting)
        if (
            waiting
            and not self._proposed
            and not self._confirmed
            and len(self._tried_alone) == len(waiting)
        ):
            # Every waiting first chunk was refused pages holding the only
            # reservation, with no admission since: start the sweep again.
            self.exhausted = True
            self._exhausted_waiting = frozenset(waiting)
            self.num_sweeps += 1
            self._tried_alone.clear()
        skip_new, self._skip_new = self._skip_new, False
        # SPF order only: the waiting first chunks in arrival order, and how far
        # an arrival-order pass would have got (for the out-of-order count).
        arrival = [r for r in waiting if r not in self._tried_alone] if self.spf_order else []
        fifo = 0
        failed: Set[int] = set()
        for request_id in self._order:
            if skip_new or len(self._proposed) + len(self._confirmed) >= self.budget:
                break
            if request_id in self._proposed or request_id in self._confirmed:
                continue
            req = waiting[request_id]
            claimed = reserve(req, None)
            if claimed is None:
                failed.add(request_id)
                continue
            if self.spf_order:
                # The claim is this rank's reusable prefix now (it may have been
                # evicted since the probe).
                self._uncached[request_id] = max(req.prompt_len - claimed, 1)
                while fifo < len(arrival) and (
                    arrival[fifo] in self._proposed
                    or arrival[fifo] in self._confirmed
                    or arrival[fifo] in failed
                ):
                    fifo += 1
                if fifo < len(arrival) and arrival[fifo] != request_id:
                    self.num_out_of_order += 1
                if self._is_aged(request_id):
                    self.num_aged_reserved += 1
            self._started[request_id] = self._num_started
            self._num_started += 1
            self._first_depth[request_id] = claimed
            self._rounds[request_id] = 0
            self._propose(request_id, claimed)

        for request_id, depth in self._confirmed.items():
            set_reuse_cap(waiting[request_id], depth)
        held = {request_id for request_id in waiting if request_id not in self._confirmed}
        self.num_held = len(held)
        if self.diag is not None:
            self.diag.agreement_held(len(held))
        return held

    def take_proposals(self) -> Optional[Dict[int, Optional[int]]]:
        """The proposals for this iteration's schedule message, or None when there are none."""
        if not self._outgoing:
            return None
        proposals, self._outgoing = self._outgoing, {}
        return proposals

    def confirmed_depth(self, request_id: int) -> Optional[int]:
        """The depth every stage confirmed for a request, or None."""
        return self._confirmed.get(request_id)

    def check_admitted(self, request_id: int, position: int) -> Tuple[bool, Optional[str]]:
        """Account a first chunk the first rank's scheduler pass admitted at ``position``.

        Returns:
            ``(keep, log_line)``. ``keep`` is False when the chunk must leave the
            batch: its position is not the depth every stage confirmed, so the
            followers could not claim it exactly; the caller releases the
            admission and the request is agreed again. ``log_line`` is a line
            for one of the first ``MAX_LOGGED_DISAGREEMENTS`` admitted chunks on
            which the stages disagreed, else None.
        """
        depth = self._confirmed.get(request_id)
        if depth is None or depth != position:
            self._drop(request_id)
            self.num_undone += 1
            return False, None
        self.num_admitted += 1
        # The pool changed: every waiting first chunk may be admissible again.
        self._tried_alone.clear()
        self.exhausted = False
        first = self._first_depth.get(request_id, depth)
        prefixes = list(self._follower_prefixes.get(request_id, {}).values())
        if self.diag is not None:
            self.diag.agreement_admitted(
                position, first, self._follower_prefixes.get(request_id, {})
            )
        rounds = self._rounds.get(request_id, 0)
        self._forget(request_id)
        self.num_extra_rounds += rounds
        lowered = position < first
        followers_disagree = bool(prefixes) and max(prefixes) != min(prefixes)
        leader_below = bool(prefixes) and first < min(prefixes)
        if not (lowered or followers_disagree or leader_below):
            return True, None
        self.num_lowered += int(lowered)
        self.max_lowered_by = max(self.max_lowered_by, first - position)
        self.num_followers_disagreed += int(followers_disagree)
        self.num_leader_below += int(leader_below)
        if prefixes:
            self.max_follower_spread = max(self.max_follower_spread, max(prefixes) - min(prefixes))
        if self._num_logged >= MAX_LOGGED_DISAGREEMENTS:
            return True, None
        self._num_logged += 1
        reported = f"{min(prefixes)}..{max(prefixes)}" if prefixes else "unknown"
        return True, (
            f"PP stage agreement: request {request_id} admitted at position "
            f"{position}; the first rank's own prefix was {first}, follower "
            f"stages reported {reported} reusable tokens ({rounds} extra "
            "proposal round(s))."
        )

    def relieve(
        self,
        pressure: bool,
        stalled: bool,
        refused: Set[int],
        others: bool,
        release: Callable[[int], None],
    ) -> List[int]:
        """First rank, after its scheduler pass: give reservations back under page pressure.

        Reserved prefixes pin pages no other request can use (on the last cache
        tier); stock admission would have left them evictable. Under pressure
        the outstanding reservations are halved, the oldest one (in this step's
        order) always kept, the unconfirmed and then the newest given back
        first, and the bound lowered to what is kept; no new reservation is made
        until the bound allows it, and every pass without pressure raises the
        bound by one, up to ``max_reserved``. The one always kept is the first
        reserved request of this step's order: the oldest, or with an
        ``uncached_estimate`` the first of ``_reservation_order`` (an aged one, else the fewest uncached tokens);
        the others are given back newest-started first either way. When the
        only reservation was
        itself refused pages in a stalled pass, it is given back as well, the
        next step makes no reservation (so one pass runs with none), and the
        request is tried again after every other waiting first chunk -- unless
        it is the only request that could run at all, in which case giving it
        back would change nothing and it stays, as a stock pass would retry it.

        A follower's decline (``answer_proposals``) received since the last call
        counts as pressure too, and the declined reservations are given back
        before the others (the oldest still kept).

        Args:
            pressure: the pass stalled (scheduled nothing while the pool refused
                pages) or preempted or paused a request.
            stalled: the pass scheduled nothing while the pool refused pages.
            refused: waiting first chunks whose confirmed reservation the pass
                refused pages.
            others: requests other than waiting first chunks could have been
                scheduled in the pass (started context requests, generation).
            release: drops the reservation of a request id on this rank.

        Returns:
            The request ids given back; the next schedule message carries None
            for each, and every follower drops its reservation on receipt.
        """
        # A follower declined these for lack of room: its pool is under pressure
        # even when this rank's is not.
        declined = {r for r in self._declined if r in self._proposed or r in self._confirmed}
        self._declined.clear()
        pressure = pressure or bool(declined)
        if not pressure:
            self.budget = min(self.max_reserved, self.budget + 1)
            return []
        self.num_pressure_passes += 1
        reserved = [r for r in self._order if r in self._proposed or r in self._confirmed]
        if not reserved:
            return []
        head = reserved[0]
        if len(reserved) == 1:
            self.budget = 1
            if not (stalled and head in refused) or not (others or len(self._order) > 1):
                return []
            # Even alone this prefix and its chunk do not fit now: try the others.
            self._tried_alone[head] = None
            self.num_refused_alone += 1
            self._skip_new = True
            given_back = [head]
        else:
            self.budget = max(1, len(reserved) // 2)
            rest = sorted(
                reserved[1:],
                key=lambda r: (r not in declined, r in self._confirmed, -self._started.get(r, -1)),
            )
            given_back = rest[: len(reserved) - self.budget]
        for request_id in given_back:
            self._drop(request_id)
            release(request_id)
        self.num_given_back += len(given_back)
        return given_back

    def summary(self) -> str:
        """One line on the first chunks the first rank admitted under the agreement."""
        spf = ""
        if self.spf_order:
            spf = (
                f" Shortest-uncached-first reservations: {self.num_out_of_order} "
                f"made out of arrival order, {self.num_aged_reserved} aged "
                f"(max wait {self.max_wait_iterations} iterations, "
                f"{self.aged_slots} aged slot(s)), {self.num_probes} prefix probe(s)."
            )
        return (
            f"PP stage agreement: {self.num_admitted} first chunk(s) admitted; "
            f"follower stages disagreed on {self.num_followers_disagreed} "
            f"(largest spread {self.max_follower_spread} tokens); the first rank "
            f"reused less than every follower on {self.num_leader_below}; "
            f"admitted below the first rank's own prefix on {self.num_lowered} "
            f"(largest drop {self.max_lowered_by} tokens, {self.num_extra_rounds} "
            f"extra proposal round(s)); {self.num_undone} admission(s) undone; "
            f"{self.num_retries} reservation retry(ies); page pressure on "
            f"{self.num_pressure_passes} pass(es), {self.num_given_back} "
            f"reservation(s) given back, {self.num_refused_alone} refused alone, "
            f"{self.num_sweeps} full sweep(s); {self.num_declines_received} "
            "reservation(s) declined by a follower for lack of room." + spf
        )

    # -------------------------------------------------------------- followers

    def pre_release(
        self,
        proposals: Optional[Mapping[int, Optional[int]]],
        requests_by_id: Mapping[int, Any],
        in_batch: Set[int],
        reserve: Reserve,
        release: Callable[[Any], None],
    ) -> Dict[int, Tuple[int, Optional[int]]]:
        """A follower applies the releasing part of a schedule message before adopting its batch.

        With block reuse, on every follower. The first rank applied these in
        ``step`` (or gave the reservations back after its pass), so it held none
        of the released pages when its pass admitted the batch. For each
        proposal, drops first: a drop (``None``) of a reservation this rank
        holds releases it; a depth below the one this rank holds claims again
        at that depth (``reserve`` drops the deeper hold first), so the agreed
        prefix stays pinned. Anything else -- a new claim, the same or a deeper
        depth -- is left to ``answer_proposals`` after the adoption, as without
        ``pre_release``. Every such proposal is for a request outside the message's
        batch: the first rank holds a request out of its pass until it is
        confirmed, and takes one off the confirmed depth out of the batch; a
        request of ``in_batch`` is skipped all the same and counted.

        ``answer_proposals`` of the same message then answers what was applied
        here without claiming it again, so the report is the one it would have
        built alone.

        Args:
            proposals: request id -> proposed depth (None: drop the reservation).
            requests_by_id: this rank's active requests by id.
            in_batch: ids of the context requests of the message's batch.
            reserve: claims and holds a request's prefix on this rank.
            release: drops a request's reservation on this rank.

        Returns:
            request id -> (depth held before, depth claimed now or None) for
            every reservation released or lowered here.
        """
        self._pre_applied = {}
        applied: Dict[int, Tuple[int, Optional[int]]] = {}
        items = sorted((proposals or {}).items(), key=lambda item: item[1] is not None)
        for request_id, depth in items:
            req = requests_by_id.get(request_id)
            if req is None or not is_waiting_first_chunk(req):
                continue
            held = getattr(req, RESERVED_DEPTH_ATTR, None)
            if held is None or (depth is not None and depth >= held):
                continue
            if request_id in in_batch:
                self.num_early_skipped_in_batch += 1
                continue
            if depth is None:
                release(req)
                self._pre_applied[request_id] = None
                self.num_early_releases += 1
                applied[request_id] = (held, None)
                continue
            claimed = reserve(req, depth)
            self._pre_applied[request_id] = claimed
            self.num_early_lowerings += 1
            self.max_early_lowered_by = max(self.max_early_lowered_by, held - depth)
            applied[request_id] = (held, claimed)
        return applied

    def follower_summary(self) -> str:
        """One line on what this follower applied before adopting the batch."""
        return (
            f"PP stage agreement (follower): {self.num_early_releases} early "
            f"release(s), {self.num_early_lowerings} early lowering(s) (largest "
            f"lowered by {self.max_early_lowered_by} tokens) before adoption; "
            f"{self.num_early_skipped_in_batch} in-batch skip(s)."
        )

    def answer_proposals(
        self,
        proposals: Optional[Mapping[int, Optional[int]]],
        requests_by_id: Mapping[int, Any],
        reserve: Reserve,
        release: Callable[[Any], None],
        probe: Callable[[Any], Optional[int]],
        has_room: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        """A follower claims each proposed depth and records its answer.

        Drops (``None``) are applied first, so a reservation the first rank gave
        back no longer counts against the room of the ones it proposes. A
        reservation made while this rank already holds another one is kept only
        if ``has_room`` says the pools still have room for the request's first
        chunk; otherwise it is released at once and answered as claimed
        nothing (declined; the report lists it). The first rank cannot confirm
        a depth without this rank's claim, so a declined request is never
        admitted on the strength of a reservation this rank released.

        Args:
            proposals: request id -> proposed depth (None: drop the reservation).
            requests_by_id: this rank's active requests by id.
            reserve: claims and holds a request's prefix on this rank.
            release: drops a request's reservation on this rank.
            probe: this stage's own uncapped reusable prefix of a request.
            has_room: whether the pools still have room for a request's first
                chunk with every reservation held (None: never decline).

        Proposals ``pre_release`` already applied for this message are not
        claimed again: a drop is skipped, a lowered depth answers the depth
        claimed then (claimed once more if that claim failed, the one attempt
        an answer makes without ``pre_release``).
        """
        pre, self._pre_applied = self._pre_applied, {}
        items = sorted((proposals or {}).items(), key=lambda item: item[1] is not None)
        held = {
            request_id
            for request_id, req in requests_by_id.items()
            if getattr(req, RESERVED_DEPTH_ATTR, None) is not None
        }
        for request_id, depth in items:
            req = requests_by_id.get(request_id)
            if req is None or not is_waiting_first_chunk(req):
                if depth is not None:
                    self._answers_out[request_id] = (depth, None, None)
                continue
            held.discard(request_id)
            if depth is None:
                if request_id not in pre:
                    release(req)
                continue
            own_prefix = probe(req)
            claimed = pre.get(request_id) if request_id in pre else None
            if claimed is None:
                claimed = reserve(req, depth)
            if claimed is not None and held and has_room is not None and not has_room(req):
                # Another reservation already pins pages here and this one would
                # leave too little room for a chunk: decline it.
                release(req)
                claimed = None
                self._declined_out.append(request_id)
                self.num_declined += 1
            if claimed is not None:
                held.add(request_id)
            self._answers_out[request_id] = (depth, claimed, own_prefix)

    def build_report(self) -> Dict[str, Any]:
        """This follower's report for the current iteration: its answers, and the
        reservations among them it declined for lack of room."""
        answers, self._answers_out = self._answers_out, {}
        report = {"iteration": self.iteration, "answers": answers}
        if self._declined_out:
            report["declined"], self._declined_out = self._declined_out, []
        return report

    def advance(self) -> None:
        """End this rank's agreement step for the current iteration."""
        self.iteration += 1
