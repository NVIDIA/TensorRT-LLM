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
"""First-rank admission back-pressure for a pipeline-parallel DFlash context server.

Under pipeline parallelism the first rank schedules every micro-batch against its
own KV pools, and every other stage must allocate the same context chunks
(``PyExecutor._pp_adopt_leader_schedule``). On a DFlash context server
(``disaggregation/pp_dflash_context.py``) the last stage also holds the drafter's KV
layers, so its pools fill well before the first rank's. A follower that cannot
allocate a chunk the first rank already ran cannot skip it -- the earlier
stages have computed its hidden states -- so the first rank holds a new prompt
back until every stage has room for it.

With the PP stage agreement every follower sends the first rank a report
each iteration (``pp_stage_agree.py``); here the report also carries a snapshot
of the follower's KV pools (:class:`StageSnapshot`). The first rank keeps a
ledger (:class:`PPAdmissionLedger`):

* **Pages are counted in slots per constraint.** A constraint is one GPU pool
  group of one follower: its free plus evictable slots, and the slots that keep
  its utilization at or below ``max_util_for_resume`` (a suspended cache is not
  resumed above it). ``n`` new tokens take at most
  ``ceil(n / tokens_per_block) + 1`` slots per attention life cycle the pool
  group serves, the bound ``KVCacheManagerV2.has_room_for_context_tokens``
  uses.
* **What a report has not seen yet is charged.** A report sent after the
  follower adopted iteration ``r`` reaches the first rank ``lag`` iterations
  later. Every prompt the first rank scheduled a chunk of after ``r``, and every
  prompt that started and is not finished, is charged from where it stood at
  ``r`` to its end: every remaining block for a full-attention life cycle; for a
  sliding-window life cycle at most two chunks plus its window, since its stale
  pages go back as its chunks are committed; plus, for a first chunk after
  ``r``, the prefix it may bring back from a lower cache tier. Every reuse
  reservation proposed after ``r`` is charged its prefix too.
* **Started prompts are never held.** The charge above covers every later
  chunk of a started prompt, so the first rank never holds one back, and no
  stage fills up with prompts that have started but cannot finish.
* **Only first chunks wait.** Before its scheduler pass the first rank lets
  through the waiting first chunks, oldest first, while every follower's
  constraints still cover the charges above plus the whole demand of each one
  let through (its prefix and its commitment); the first one that does not fit,
  and every one after it, waits for the next iteration. The followers never see
  a held prompt in a batch, so every rank runs the same batch.

A follower's frees (finished contexts sent, reservations dropped) only add room
the ledger has not counted, so the charges bound what a follower allocates for
the first rank's batches from above.

The module imports nothing from the package; the executor passes it plain
numbers.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# Key of the pool snapshot in a follower's stage-agreement report.
ADMISSION_REPORT_KEY = "admission"

# The first rank logs, at INFO, this many iterations on which it starts holding
# first chunks back; later ones are only counted in the summary.
MAX_LOGGED_HOLDS = 16


def blocks_for(num_tokens: int, tokens_per_block: int) -> int:
    """Upper bound of the blocks ``num_tokens`` new tokens take in one life cycle (0 for none)."""
    if num_tokens <= 0:
        return 0
    return -(-num_tokens // tokens_per_block) + 1


@dataclass(frozen=True)
class Constraint:
    """One pool group of a follower's KV pools.

    Attributes:
        available: free plus evictable slots now.
        resume_room: the slots that can still be taken while the pool group stays
            at or below ``max_util_for_resume`` (negative when it is already
            above), or None when no such gate applies.
        capacity: the most slots the pool group can offer at all (the
            ``resume_room`` of the empty pool group).
        window_blocks: one entry per attention life cycle the pool group serves:
            None for full attention, else the most slots a claimed prefix holds
            in that sliding-window life cycle (sink blocks, the window with the
            generation headroom, and one block of alignment).
        extra_tokens: the tokens the KV cache manager allocates for a request
            beyond its history (spec decoding, a shared drafter's block).
    """

    available: int
    resume_room: Optional[int]
    capacity: int
    window_blocks: Tuple[Optional[int], ...]
    extra_tokens: int = 0

    @property
    def room(self) -> int:
        """The slots the first rank's batches may take."""
        if self.resume_room is None:
            return self.available
        return min(self.available, self.resume_room)

    def prefix_demand(self, depth: int, tokens_per_block: int) -> int:
        """Slots a claimed (or resumed) prefix of ``depth`` tokens can take here."""
        blocks = blocks_for(depth, tokens_per_block)
        return sum(
            blocks if window is None else min(blocks, window) for window in self.window_blocks
        )

    def remaining_demand(
        self, num_tokens: int, max_chunk_tokens: Optional[int], tokens_per_block: int
    ) -> int:
        """Slots the remaining ``num_tokens`` tokens of a started prompt can still take here.

        A sliding-window life cycle gives back the pages its window left behind as
        each chunk is committed, so on top of what it holds it takes at most two
        chunks plus its window; without a chunk bound it is charged like full
        attention.
        """
        blocks = blocks_for(num_tokens, tokens_per_block)
        if max_chunk_tokens is None:
            return len(self.window_blocks) * blocks
        chunks = 2 * blocks_for(max_chunk_tokens, tokens_per_block)
        return sum(
            blocks if window is None else min(blocks, chunks + window)
            for window in self.window_blocks
        )


@dataclass(frozen=True)
class StageSnapshot:
    """A follower's KV pools as its report carries them."""

    tokens_per_block: int
    constraints: Tuple[Constraint, ...]

    def to_wire(self) -> Dict[str, Any]:
        """Plain types for the report message."""
        return {
            "tpb": self.tokens_per_block,
            "c": [
                (c.available, c.resume_room, c.capacity, tuple(c.window_blocks), c.extra_tokens)
                for c in self.constraints
            ],
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> "StageSnapshot":
        return cls(
            int(wire["tpb"]),
            tuple(
                Constraint(
                    int(available),
                    None if resume_room is None else int(resume_room),
                    int(capacity),
                    tuple(None if w is None else int(w) for w in window_blocks),
                    int(extra_tokens),
                )
                for available, resume_room, capacity, window_blocks, extra_tokens in wire["c"]
            ),
        )

    @classmethod
    def merge(cls, snapshots: Sequence["StageSnapshot"]) -> "StageSnapshot":
        """One snapshot of the KV cache managers of one stage.

        Raises:
            ValueError: the managers use different block sizes.
        """
        sizes = {s.tokens_per_block for s in snapshots}
        if len(sizes) != 1:
            raise ValueError(f"the KV cache managers of one stage use block sizes {sorted(sizes)}")
        return cls(sizes.pop(), tuple(c for s in snapshots for c in s.constraints))


@dataclass(frozen=True)
class Candidate:
    """A waiting first chunk the first rank may let into its scheduler pass.

    Attributes:
        request_id: the request.
        depth: the prefix it is admitted at (the depth every stage confirmed).
        total_tokens: its prompt plus the draft tokens of its last chunk.
    """

    request_id: int
    depth: int
    total_tokens: int


@dataclass(frozen=True)
class ScheduledChunk:
    """A context chunk of the first rank's batch.

    Attributes:
        request_id: the request.
        position: the chunk's first token.
        num_tokens: the tokens it allocates (its size, plus the draft tokens on
            the last chunk).
        total_tokens: the request's prompt plus the draft tokens of its last chunk.
        is_first: whether it is the request's first chunk, whose prefix may be
            brought back from a lower cache tier.
    """

    request_id: int
    position: int
    num_tokens: int
    total_tokens: int
    is_first: bool


class PPAdmissionLedger:
    """The first rank's ledger of what each follower can still allocate.

    The executor drives it once per PP loop iteration on the first rank:
    ``accept`` the follower reports that arrived, then ``hold`` (before the
    scheduler pass; it returns the first chunks to leave out of the pass), then
    ``record`` (the batch and the reuse proposals of that iteration's schedule
    message). Followers only send snapshots.

    Args:
        pp_size: pipeline-parallel size.
        max_chunk_tokens: the most tokens a context chunk can have (the
            scheduler's ``max_num_tokens``), or None.
    """

    def __init__(self, pp_size: int, max_chunk_tokens: Optional[int]):
        if pp_size < 2:
            raise ValueError("admission back-pressure needs pipeline_parallel_size > 1")
        if max_chunk_tokens is not None and max_chunk_tokens < 1:
            raise ValueError(f"max_chunk_tokens {max_chunk_tokens} must be at least 1")
        self.pp_size = pp_size
        self.max_chunk_tokens = max_chunk_tokens
        # Per follower: the iteration of its latest report and its snapshot.
        self._reports: Dict[int, Tuple[int, StageSnapshot]] = {}
        # Per iteration some follower has not reported on yet: its chunks (with
        # the key of their prompt) and the depths of its reuse proposals.
        self._events: Deque[Tuple[int, List[Tuple[Tuple[int, int], ScheduledChunk]], List[int]]] = (
            deque()
        )
        # Started, unfinished prompts: key -> (total tokens, end of the last
        # chunk scheduled). A key is (request id, start count): a preempted
        # prompt that starts over is another prompt.
        self._started: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._epochs: Dict[int, int] = {}
        self._iteration: Optional[int] = None
        # Candidates whose demand exceeds a follower's whole pool group: each is
        # let through only once nothing else is charged there (warned once).
        self._oversized: Set[int] = set()
        self._warnings: List[str] = []
        # Whether the last ``hold`` held a candidate back for lack of room.
        self.blocked = False
        # Counters for the summary.
        self.num_held_iterations = 0
        self.num_startup_holds = 0
        self.max_held = 0
        self.num_logged_holds = 0
        self.num_oversized_admitted = 0
        # Host time of ``hold`` plus ``record`` and how many iterations ran them.
        self.host_seconds = 0.0
        self.num_iterations = 0

    @property
    def follower_stages(self) -> List[int]:
        return list(range(1, self.pp_size))

    def accept(self, stage: int, iteration: int, wire: Optional[Mapping[str, Any]]) -> None:
        """Record the pool snapshot of a follower's report of ``iteration``.

        Raises:
            RuntimeError: the report carries no snapshot, or it is not newer than
                the one already accepted from that follower.
        """
        if wire is None:
            raise RuntimeError(
                f"PP stage {stage} sent no KV pool snapshot with its report of "
                f"iteration {iteration}; every rank must run with the same "
                "admission back-pressure setting."
            )
        previous = self._reports.get(stage)
        if previous is not None and previous[0] >= iteration:
            raise RuntimeError(
                f"PP stage {stage} sent its KV pool snapshot of iteration "
                f"{iteration} after the one of iteration {previous[0]}."
            )
        self._reports[stage] = (iteration, StageSnapshot.from_wire(wire))

    def _key(self, request_id: int) -> Tuple[int, int]:
        return (request_id, self._epochs.get(request_id, 0))

    def _prune_started(self, live: Set[int], restarted: Set[int]) -> None:
        for key in list(self._started):
            request_id = key[0]
            if request_id in restarted:
                # It starts over: its next chunks belong to another prompt. The
                # pages of the old one stay charged until every follower
                # reported past its chunks.
                self._epochs[request_id] = key[1] + 1
                del self._started[key]
            elif request_id not in live:
                del self._started[key]
        for request_id in list(self._epochs):
            if request_id not in live:
                del self._epochs[request_id]
        self._oversized &= live

    def charge(
        self, c: Constraint, position: int, total_tokens: int, first: bool, tokens_per_block: int
    ) -> int:
        """Slots a prompt can still take on constraint ``c`` from ``position`` to its end.

        ``first``: its next chunk is its first, whose prefix of ``position``
        tokens may come back from a lower cache tier. The manager's extra tokens
        beyond the history are charged on top of the remaining tokens.
        """
        prefix = c.prefix_demand(position, tokens_per_block) if first else 0
        return prefix + c.remaining_demand(
            total_tokens - position + c.extra_tokens, self.max_chunk_tokens, tokens_per_block
        )

    def _needed(self, stage: int) -> Optional[List[int]]:
        """Slots per constraint the first rank's batches can still take on
        ``stage`` beyond its latest report; None before its first report."""
        report = self._reports.get(stage)
        if report is None:
            return None
        reported, snapshot = report
        tpb = snapshot.tokens_per_block
        needed = [0] * len(snapshot.constraints)
        # Where each prompt stood at the report: the start of its first chunk
        # scheduled after it, else the end of its last one.
        since: Dict[Tuple[int, int], Tuple[int, int, bool]] = {}
        for iteration, chunks, depths in self._events:
            if iteration <= reported:
                continue
            for key, chunk in chunks:
                since.setdefault(key, (chunk.position, chunk.total_tokens, chunk.is_first))
            for i, c in enumerate(snapshot.constraints):
                for depth in depths:
                    needed[i] += c.prefix_demand(depth, tpb)
        for key, (total, end) in self._started.items():
            since.setdefault(key, (end, total, False))
        for position, total, first in since.values():
            for i, c in enumerate(snapshot.constraints):
                needed[i] += self.charge(c, position, total, first, tpb)
        return needed

    def demand(self, c: Constraint, candidate: Candidate, tokens_per_block: int) -> int:
        """Slots a candidate can take on constraint ``c`` from admission to its end."""
        return self.charge(c, candidate.depth, candidate.total_tokens, True, tokens_per_block)

    def hold(
        self,
        iteration: int,
        candidates: Sequence[Candidate],
        live: Iterable[int],
        restarted: Iterable[int] = (),
    ) -> Tuple[Set[int], Optional[str]]:
        """The first chunks to leave out of this iteration's scheduler pass.

        Args:
            iteration: this PP loop iteration (the stage agreement's counter).
            candidates: the waiting first chunks the pass may admit, oldest first.
            live: every active request id; a started prompt not among them is
                forgotten.
            restarted: waiting first chunks; one that had started before (a
                preempted prompt starts over) has its commitment dropped.

        Returns:
            ``(held, log_line)``: the request ids to hold back, and a line for one
            of the first ``MAX_LOGGED_HOLDS`` iterations that start holding.

        A candidate whose demand bound exceeds what a follower's pool group can
        offer at all is held until nothing else is charged there and the pool
        group is wholly free, then let through (the bound is loose, so it may
        still fit); ``take_warnings`` reports it once.
        """
        start = time.perf_counter()
        try:
            return self._hold(iteration, candidates, live, restarted)
        finally:
            self.host_seconds += time.perf_counter() - start
            self.num_iterations += 1

    def _hold(
        self,
        iteration: int,
        candidates: Sequence[Candidate],
        live: Iterable[int],
        restarted: Iterable[int],
    ) -> Tuple[Set[int], Optional[str]]:
        self._iteration = iteration
        self._prune_started(set(live), set(restarted))
        if not candidates:
            self.blocked = False
            return set(), None
        needed = {stage: self._needed(stage) for stage in self.follower_stages}
        if any(n is None for n in needed.values()):
            # Before every follower's first report its room is unknown.
            self.num_startup_holds += 1
            self.blocked = False
            return {c.request_id for c in candidates}, None
        admitted = 0
        binding = None
        for candidate in candidates:
            demands = {}
            oversized = False
            for stage in self.follower_stages:
                snapshot = self._reports[stage][1]
                for i, c in enumerate(snapshot.constraints):
                    demand = self.demand(c, candidate, snapshot.tokens_per_block)
                    if demand > c.capacity:
                        oversized = True
                        if candidate.request_id not in self._oversized:
                            self._oversized.add(candidate.request_id)
                            self._warnings.append(
                                f"PP stage {stage}: context request {candidate.request_id} "
                                f"({candidate.total_tokens} tokens from position "
                                f"{candidate.depth}) may need up to {demand} KV cache "
                                f"slots of pool constraint {i}, which offers at most "
                                f"{c.capacity}; it waits until that pool is free and may "
                                "then fail to allocate there. Increase that stage's KV "
                                "cache size."
                            )
                        demand = c.capacity
                    if needed[stage][i] + demand > c.room and binding is None:
                        binding = (stage, i, c.room, needed[stage][i], demand)
                    demands[stage, i] = demand
            if binding is not None:
                break
            for (stage, i), demand in demands.items():
                needed[stage][i] += demand
            admitted += 1
            self.num_oversized_admitted += oversized
        held = {c.request_id for c in candidates[admitted:]}
        was_blocked, self.blocked = self.blocked, bool(held)
        line = None
        if held:
            self.num_held_iterations += 1
            self.max_held = max(self.max_held, len(held))
            if not was_blocked and self.num_logged_holds < MAX_LOGGED_HOLDS:
                self.num_logged_holds += 1
                stage, index, room, used, demand = binding
                line = (
                    f"PP admission back-pressure: holding {len(held)} first chunk(s) "
                    f"back at iteration {iteration}; PP stage {stage} pool constraint "
                    f"{index} has room for {room} slot(s), {used} are committed to "
                    "scheduled chunks, reservations and started prompts, and the "
                    f"oldest held prompt needs {demand}."
                )
        return held, line

    def record(self, chunks: Sequence[ScheduledChunk], reserved_depths: Iterable[int]) -> None:
        """Account the context chunks and reuse proposals of this iteration's schedule.

        Args:
            chunks: the context chunks of the batch the first rank sends.
            reserved_depths: the depths of the reuse reservations the schedule
                message proposes (dropped reservations left out).

        Raises:
            RuntimeError: ``hold`` did not run in this iteration.
        """
        if self._iteration is None:
            raise RuntimeError("PPAdmissionLedger.record without hold in this iteration")
        start = time.perf_counter()
        try:
            self._record(chunks, reserved_depths)
        finally:
            self.host_seconds += time.perf_counter() - start

    def _record(self, chunks: Sequence[ScheduledChunk], reserved_depths: Iterable[int]) -> None:
        iteration, self._iteration = self._iteration, None
        keyed = []
        for chunk in chunks:
            key = self._key(chunk.request_id)
            keyed.append((key, chunk))
            end = chunk.position + chunk.num_tokens
            if end >= chunk.total_tokens:
                self._started.pop(key, None)
            else:
                self._started[key] = (chunk.total_tokens, end)
        self._events.append((iteration, keyed, [d for d in reserved_depths if d]))
        if len(self._reports) == len(self.follower_stages):
            seen = min(reported for reported, _ in self._reports.values())
            while self._events and self._events[0][0] <= seen:
                self._events.popleft()

    @property
    def num_started(self) -> int:
        return len(self._started)

    def take_warnings(self) -> List[str]:
        """The warnings ``hold`` produced since the last call."""
        warnings, self._warnings = self._warnings, []
        return warnings

    def summary(self) -> str:
        return (
            "PP admission back-pressure: first chunks held back on "
            f"{self.num_held_iterations} iteration(s) (at most {self.max_held} at "
            f"once), and on {self.num_startup_holds} iteration(s) before the "
            f"followers' first reports; {self.num_oversized_admitted} prompt(s) let "
            "through above a pool's whole capacity bound; "
            f"{len(self._started)} started prompt(s) and "
            f"{len(self._events)} unreported iteration(s) at the end; "
            f"{self.host_seconds * 1e3:.1f} ms of host time over "
            f"{self.num_iterations} iteration(s)."
        )
