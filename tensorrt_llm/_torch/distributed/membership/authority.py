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
"""Membership authority for Wide-EP fault tolerance.

:class:`Authority` decides membership from the ranks' reports. It does no I/O and takes monotonic
time from its callers, so tests drive it with simulated ranks. It reads the wall clock only to
report :class:`TransactionSummary` times in Unix milliseconds.
"""

import time
from collections.abc import Iterable
from typing import Optional

from tensorrt_llm.logger import logger

from .records import (
    ADMISSION_TIMEOUT_S,
    HANG_TIMEOUT_S,
    MAX_RANKS,
    RENDEZVOUS_TIMEOUT_S,
    REPORT_INTERVAL_S,
    MembershipRecord,
    MembershipState,
    NodeReport,
    RankReport,
    Reply,
    ResumeState,
    TransactionSummary,
    all_ranks,
    has_rank,
    latest_epoch,
    next_epoch,
    rank_mask,
)


def _ranks(mask: int) -> list[int]:
    """Return the ranks set in ``mask``, lowest first."""
    return [r for r in range(mask.bit_length()) if has_rank(mask, r)]


def _chain_ends(ranks: Iterable[int], waits: dict[int, int]) -> int:
    """Return a mask of the ``ranks`` that are the end of a wait-for chain, so they may be hung.

    ``waits`` maps each rank to a mask of the ranks it waits on. A rank is an end when every rank
    it reaches through ``waits`` also reaches it, so it waits on nobody or only within its cycle.
    A cycle forms when a stopped process's last report names a peer that now waits on it. In
    ``0 -> 2 -> 3 -> 2``, ranks 2 and 3 are ends and rank 0 is not. A rank that waits on a rank
    that waits on nobody is never an end.

    The ends are the strongly connected components that no wait leaves. Tarjan's algorithm finds
    the components in one depth-first search over the waits.
    """
    order: dict[int, int] = {}
    low: dict[int, int] = {}
    stack: list[int] = []
    # The ranks whose component is found, and the ends among them.
    done = ends = 0
    for root in ranks:
        if root in order:
            continue
        order[root] = low[root] = len(order)
        stack.append(root)
        path = [(root, iter(_ranks(waits.get(root, 0))))]
        while path:
            r, peers = path[-1]
            q = next(peers, None)
            if q is not None:
                if q not in order:
                    order[q] = low[q] = len(order)
                    stack.append(q)
                    path.append((q, iter(_ranks(waits.get(q, 0)))))
                elif not has_rank(done, q):
                    low[r] = min(low[r], order[q])
                continue
            path.pop()
            if path:
                parent = path[-1][0]
                low[parent] = min(low[parent], low[r])
            if low[r] != order[r]:
                continue
            component = 0
            while not has_rank(component, r):
                component |= 1 << stack.pop()
            done |= component
            if not any(waits.get(q, 0) & ~component for q in _ranks(component)):
                ends |= component
    return ends


def _update_last_change(seen: dict[int, tuple], rank: int, value: tuple, now_s: float) -> float:
    """Store ``value`` for ``rank`` in ``seen`` if it changed and return when it last changed."""
    if seen.get(rank, (None,))[0] != value:
        seen[rank] = (value, now_s)
    return seen[rank][1]


class Authority:
    """Decide membership from the ranks' reports.

    A rank fails when its connection closes, or when it is the end of a wait-for chain that makes no
    progress for the timeout. Any failure while STARTING aborts. A replacement that fails to join is
    left out.

    Outside a change, a rank whose node supervisor reports its hang detector paused, in a report
    received within half the hang timeout, is never removed as hung, also once the rank's own
    reports stop. The ranks that wait on such a rank stall until their own hang detector or
    all-to-all timeout ends them.

    Call :meth:`tick` every report interval. ``record`` holds the current decision and
    ``last_transaction`` summarizes the last ended change. The authority process publishes
    ``stalled_since_s`` and ``stalled_replacements`` for health reporting.
    """

    def __init__(
        self,
        world_size: int,
        hang_timeout_s: float = HANG_TIMEOUT_S,
        rendezvous_timeout_s: float = RENDEZVOUS_TIMEOUT_S,
        admission_timeout_s: float = ADMISSION_TIMEOUT_S,
    ) -> None:
        """Start with every rank of the deployment in a STARTING record.

        Args:
            world_size: The deployment's ranks, at most ``MAX_RANKS``.
            hang_timeout_s: How long a rank that others wait on may make no progress before it is
                removed as hung.
            rendezvous_timeout_s: How long the startup waits for every rank to connect, and how
                long a starting rank or a replacement's startup may make no progress.
            admission_timeout_s: How long a joining rank that waits on no peer may make no progress
                in its admission before it is left out.

        Raises:
            ValueError: ``world_size`` is not in ``(0, MAX_RANKS]``.
        """
        if not 0 < world_size <= MAX_RANKS:
            raise ValueError(f"world_size must be in (0, {MAX_RANKS}], got {world_size}")
        self.world_size = world_size
        self.hang_timeout_s = hang_timeout_s
        self.rendezvous_timeout_s = rendezvous_timeout_s
        self.admission_timeout_s = admission_timeout_s
        self.record = MembershipRecord(
            generation=0, state=MembershipState.STARTING, active_mask=all_ranks(world_size)
        )
        self._committed_mask = self.record.active_mask
        self._rendezvous_began_s: Optional[float] = None
        # The ranks that connected at least once during the startup.
        self._connected_once = 0
        # rank -> (value, time it last changed), for the progress and the startup progress.
        self._progress: dict[int, tuple] = {}
        self._startup_progress: dict[int, tuple] = {}
        self._hung_since: dict[int, float] = {}
        # rank -> when the latest node report that counted it as paused arrived.
        self._last_paused_s: dict[int, float] = {}
        # rank -> the first decision time at which it had no report.
        self._unreported_since: dict[int, float] = {}
        # The ranks whose replacement asked to join and is not in a proposed change yet.
        self._requested: set[int] = set()
        # The replacements in the current change, until it commits or leaves them out.
        self._joining: set[int] = set()
        # rank -> generation of the change that removed it, so replacements for ranks that failed
        # together join in one change.
        self._failed_in: dict[int, int] = {}
        # Resume epoch of an INSTALL that a newer change replaced before it committed, or 0.
        self._replaced_install_epoch = 0
        # When the pending change's first failure began, or its first replacement was ready.
        self._detected_s: Optional[float] = None
        self._causes: dict[int, str] = {}
        # Since when the active ranks have waited on a rank that makes no progress, for longer than
        # a report interval.
        self.stalled_since_s: Optional[float] = None
        self._transaction_began_s: Optional[float] = None
        self._ready_s: Optional[float] = None
        self.last_transaction: Optional[TransactionSummary] = None
        self.stalled_replacements: tuple[int, ...] = ()
        self._changes: dict[str, int] = {}

    def request_join(self, rank: int) -> None:
        """Request that a replacement of ``rank``, which the membership no longer lists, join.

        The replacement joins once the change that removes the old process has committed.
        """
        self._requested.add(rank)

    def tick(
        self,
        now_s: float,
        reports: dict[int, RankReport],
        connected: int,
        node_reports: Iterable[NodeReport] = (),
    ) -> Optional[MembershipRecord]:
        """Decide membership at monotonic ``now_s`` and return a new record to publish, or None.

        ``connected`` masks the ranks whose current process has an open liveness connection.
        ``node_reports`` holds each connected node supervisor's latest report. Also update
        ``stalled_replacements`` and, when a change ends, ``last_transaction``.
        """
        before, committed = self.record.state, self._committed_mask
        paused_s: dict[int, float] = {}
        for node_report in node_reports:
            received_s = node_report.received_s
            if now_s - received_s <= self.hang_timeout_s / 2:
                for r in _ranks(node_report.paused_mask):
                    paused_s[r] = max(paused_s.get(r, received_s), received_s)
        record = self._decide(now_s, reports, connected, paused_s)
        if before != MembershipState.STARTING:
            self.stalled_replacements = tuple(
                r
                for r in sorted(self._requested)
                if not reports.get(r, RankReport()).joining_ready
                and not self._startup_advancing(r, reports, now_s)
            )
        if record is None:
            return None
        if before == MembershipState.COMMITTED:
            self._transaction_began_s = now_s
        if before != MembershipState.STARTING and record.state in (
            MembershipState.COMMITTED,
            MembershipState.ABORTED,
        ):
            self._record_transaction(committed, record, now_s)
        return record

    def _record_transaction(self, committed: int, record: MembershipRecord, now_s: float) -> None:
        """Count the change that ``record`` ends and summarize it in ``last_transaction``."""
        now_unix_ms = time.time() * 1e3

        def unix_ms(s: Optional[float]) -> Optional[float]:
            return None if s is None else now_unix_ms + (s - now_s) * 1e3

        began_s = self._transaction_began_s
        removed = tuple(_ranks(committed & ~record.active_mask))
        if record.state == MembershipState.ABORTED:
            kind = "abort"
        else:
            kind = "failover" if removed else "admission"
        self._changes[kind] = self._changes.get(kind, 0) + 1
        self.last_transaction = TransactionSummary(
            generation=record.generation,
            outcome=record.state.name,
            removed_ranks=removed,
            causes={r: self._causes[r] for r in removed if r in self._causes},
            added_ranks=tuple(_ranks(record.active_mask & ~committed)),
            detected_unix_ms=unix_ms(began_s if self._detected_s is None else self._detected_s),
            began_unix_ms=unix_ms(began_s),
            ready_unix_ms=unix_ms(self._ready_s),
            ended_unix_ms=now_unix_ms,
            changes=dict(self._changes),
            reason=record.reason if record.state == MembershipState.ABORTED else "",
        )
        self._detected_s = None
        self._causes = {}

    def _decide(
        self,
        now_s: float,
        reports: dict[int, RankReport],
        connected: int,
        paused_s: dict[int, float],
    ) -> Optional[MembershipRecord]:
        """Return the record that follows from the reports, or None if membership stays.

        ``paused_s`` maps each rank whose hang detector its node supervisor recently reported
        paused to when that report arrived.
        """
        state = self.record.state
        if state in (MembershipState.ABORTED, MembershipState.SHUTDOWN):
            return None
        if 0 in reports and reports[0].shutdown:
            return self._new_record(MembershipState.SHUTDOWN, self.record.active_mask)
        if state == MembershipState.STARTING:
            return self._await_startup(now_s, reports, connected)
        failed = self._detect_failures(now_s, reports, connected, self.hang_timeout_s, paused_s)
        # A replacement that disconnects or fails withdraws its join request.
        self._forget_joins((failed | ~connected) & all_ranks(self.world_size))
        if state == MembershipState.COMMITTED:
            return self._start_change(now_s, reports, failed)
        if failed:
            return self._propose(self.record.active_mask & ~failed)
        return self._advance_change(now_s, reports)

    def _forget_joins(self, gone_mask: int) -> None:
        gone = set(_ranks(gone_mask))
        self._requested -= gone
        self._joining -= gone

    def _start_change(
        self, now_s: float, reports: dict[int, RankReport], failed: int
    ) -> Optional[MembershipRecord]:
        """Propose to remove the ``failed`` ranks and to add the replacements ready to join."""
        ready = {r for r in self._requested if r in reports and reports[r].joining_ready}
        starting = {r for r in self._requested if self._startup_advancing(r, reports, now_s)}
        if not failed and not ready:
            self._detected_s = None
            return None
        if self._detected_s is None:
            self._detected_s = now_s
        if not failed and self._waits_for_replacements_of_same_failure(
            now_s, reports, ready, starting
        ):
            return None
        self._requested -= ready
        self._joining |= ready
        if ready:
            logger.info(f"Adding joining ranks {sorted(ready)} to the membership")
        return self._propose(
            (self.record.active_mask & ~failed) | rank_mask(ready, self.world_size)
        )

    def _advance_change(
        self, now_s: float, reports: dict[int, RankReport]
    ) -> Optional[MembershipRecord]:
        """Move a PROPOSE to INSTALL, or an INSTALL to COMMITTED, once every active rank replied.

        A survivor that replies FAILED aborts the deployment. A joining rank that replies FAILED is
        left out.
        """
        state = self.record.state
        active = _ranks(self.record.active_mask)
        failed_replies = {r for r in active if self._rank_replied(r, reports, Reply.FAILED)}
        failed_survivors = failed_replies - self._joining
        if failed_survivors:
            return self.abort(f"in {state.name}, ranks {sorted(failed_survivors)} replied FAILED")
        if failed_replies:
            self._joining -= failed_replies
            return self._propose(
                self.record.active_mask & ~rank_mask(failed_replies, self.world_size),
                reason="a replacement could not install the change that adds it",
            )
        expected = Reply.READY if state == MembershipState.PROPOSE else Reply.INSTALLED
        if not all(self._rank_replied(r, reports, expected) for r in active):
            return None
        if state == MembershipState.PROPOSE:
            return self._install(now_s, reports, active)
        return self._commit(active)

    def _rank_replied(self, r: int, reports: dict[int, RankReport], reply: Reply) -> bool:
        """Return whether rank ``r`` replied ``reply``, counting a joiner's readiness as READY."""
        report = reports.get(r)
        ready_to_join = r in self._joining and report is not None and report.joining_ready
        return (reply == Reply.READY and ready_to_join) or self._replied(report, reply)

    def _install(
        self, now_s: float, reports: dict[int, RankReport], active: list[int]
    ) -> MembershipRecord:
        """Publish INSTALL with the point where every active rank resumes."""
        survivors = [
            r for r in active if has_rank(self._committed_mask, r) and r not in self._joining
        ]
        # Resume after the furthest survivor, so no rank repeats an iteration that another finished.
        resume = ResumeState(
            iteration=max(
                [self.record.resume.iteration] + [reports[r].paused_at.iteration for r in survivors]
            ),
            epoch=self._resume_epoch([reports[r].paused_at.epoch for r in survivors]),
        )
        self._ready_s = now_s
        return self._new_record(MembershipState.INSTALL, self.record.active_mask, resume)

    def _resume_epoch(self, epochs: list[int]) -> int:
        """Return the host exchange epoch to resume from, given the survivors' paused epochs."""
        replaced = self._replaced_install_epoch
        # A newer PROPOSE can replace an INSTALL before its commit, after some ranks moved to that
        # INSTALL's resume epoch. Every round waits for every active rank, so if a survivor is still
        # before that epoch, no rank ran a round at it. Resume there again, which keeps the ranks
        # that installed it consistent.
        if replaced and any(latest_epoch([replaced, epoch]) != epoch for epoch in epochs):
            return replaced
        # Epochs wrap, so the resume epoch follows the latest one, not the largest.
        return next_epoch(latest_epoch(epochs))

    def _commit(self, active: list[int]) -> MembershipRecord:
        """Commit the change that every active rank installed."""
        record = self.record
        self._joining -= set(active)
        self._replaced_install_epoch = 0
        for r in _ranks(self._committed_mask & ~record.active_mask):
            self._failed_in[r] = record.generation
        for r in active:
            self._failed_in.pop(r, None)
        self._committed_mask = record.active_mask
        # A replacement's progress clock must not continue from the process it replaces.
        self._progress.clear()
        self._hung_since.clear()
        return self._new_record(MembershipState.COMMITTED, record.active_mask)

    def _await_startup(
        self, now_s: float, reports: dict[int, RankReport], connected: int
    ) -> Optional[MembershipRecord]:
        """Commit generation 0 once every rank connected and started.

        Abort if a connected rank fails, or if a rank does not connect within the rendezvous
        timeout.
        """
        if self._rendezvous_began_s is None:
            self._rendezvous_began_s = now_s
        everyone = self.record.active_mask
        self._connected_once |= connected
        failed = self._detect_failures(now_s, reports, connected, self.rendezvous_timeout_s, {})
        if failed & self._connected_once:
            return self.abort(
                f"ranks {_ranks(failed & self._connected_once)} failed during the startup"
            )
        absent = everyone & ~self._connected_once
        if absent and now_s - self._rendezvous_began_s > self.rendezvous_timeout_s:
            return self.abort(
                f"ranks {_ranks(absent)} did not reach the rendezvous within "
                f"{self.rendezvous_timeout_s:.0f} s"
            )
        if absent or not all(reports.get(r, RankReport()).started for r in _ranks(everyone)):
            return None
        logger.info("Every rank started")
        self._progress.clear()
        self._hung_since.clear()
        self._detected_s = None
        self._causes = {}
        return self._new_record(MembershipState.COMMITTED, everyone)

    def _waits_for_replacements_of_same_failure(
        self, now_s: float, reports: dict[int, RankReport], ready: set[int], starting: set[int]
    ) -> bool:
        """Return whether to delay adding ``ready`` for a replacement that is still starting.

        Delay while a ``starting`` replacement of a rank that one change removed together with a
        rank of ``ready`` still reports. The replacements of a lost node then join in one change,
        and the survivors pause once.
        """
        failed_with = {self._failed_in[r] for r in ready if r in self._failed_in}
        return any(
            self._failed_in.get(r) in failed_with and self._reports_advance(r, reports, now_s)
            for r in starting - ready
        )

    def _startup_advancing(self, r: int, reports: dict[int, RankReport], now_s: float) -> bool:
        """Return whether the startup of rank ``r``'s replacement progressed within the timeout."""
        report = reports.get(r, RankReport())
        progress = (report.incarnation, report.progress)
        # A replacement waits for its predecessor's memory until the active ranks commit a
        # membership without the predecessor. This wait is not a stall.
        if has_rank(self._committed_mask, r):
            self._startup_progress.pop(r, None)
        changed_s = _update_last_change(self._startup_progress, r, progress, now_s)
        return now_s - changed_s <= self.rendezvous_timeout_s

    def _reports_advance(self, r: int, reports: dict[int, RankReport], now_s: float) -> bool:
        """Return whether rank ``r`` sent a report within half the hang timeout.

        A rank whose reports stopped is a stopped process, so its last ``waiting_for`` is stale.
        """
        return now_s - self._received_s(r, reports, now_s) <= self.hang_timeout_s / 2

    def _received_s(self, r: int, reports: dict[int, RankReport], now_s: float) -> float:
        """Return when rank ``r``'s latest report arrived, or since when it has sent none."""
        if r in reports:
            self._unreported_since.pop(r, None)
            return reports[r].received_s
        return self._unreported_since.setdefault(r, now_s)

    def _propose(self, active_mask: int, reason: str = "") -> MembershipRecord:
        """Publish PROPOSE of ``active_mask`` in a new generation, or ABORTED without rank 0."""
        self._ready_s = None
        # Rank 0 hosts the authority and serves the requests, so the deployment ends without it.
        if not has_rank(active_mask, 0):
            logger.error("Deployment aborted: rank 0 failed")
            return self._new_record(MembershipState.ABORTED, active_mask, reason="rank 0 failed")
        if self.record.state == MembershipState.INSTALL:
            self._replaced_install_epoch = self.record.resume.epoch
        return self._new_record(
            MembershipState.PROPOSE,
            active_mask,
            generation=self.record.generation + 1,
            reason=reason,
        )

    def abort(self, reason: str) -> MembershipRecord:
        """Publish ABORTED with ``reason``, also for a fatal error outside the reports."""
        logger.error(f"Deployment aborted: {reason}")
        return self._new_record(MembershipState.ABORTED, self.record.active_mask, reason=reason)

    def _new_record(
        self,
        state: MembershipState,
        active_mask: int,
        resume: Optional[ResumeState] = None,
        generation: Optional[int] = None,
        reason: str = "",
    ) -> MembershipRecord:
        """Replace ``record`` with one in ``state``, keeping the generation and resume unless given."""
        self.record = MembershipRecord(
            generation=self.record.generation if generation is None else generation,
            state=state,
            active_mask=active_mask,
            resume=self.record.resume if resume is None else resume,
            joining_mask=rank_mask(self._joining, self.world_size) & active_mask,
            reason=reason,
        )
        return self.record

    def _replied(self, report: Optional[RankReport], reply: Reply) -> bool:
        return (
            report is not None
            and report.reply_generation == self.record.generation
            and report.reply == reply
        )

    def _detect_failures(
        self,
        now_s: float,
        reports: dict[int, RankReport],
        connected: int,
        timeout_s: float,
        paused_s: dict[int, float],
    ) -> int:
        """Return a mask of active ranks that are dead, or hung for ``timeout_s``.

        A rank whose connection closed is dead. A rank is hung when other ranks wait on it, it is
        the end of a wait-for chain (:func:`_chain_ends`), and it made no progress for the timeout.
        Progress counts its iteration, install steps, and the all-to-all rounds and host
        collective epochs its peers wait on. During a change the ranks that have not replied are
        waited on, and during startup the ranks that have not started. A rank's waits count only
        while its reports advance. Outside a change, a rank in ``paused_s``, whose node supervisor
        recently reported its hang detector paused, is never hung, and once it leaves the pause its
        stall starts no earlier than the latest node report that counted it as paused. A joining
        rank that waits on no peer is hung after at most the admission timeout, so the survivors do
        not pause a full hang timeout for it.

        Also record the failure causes and since when the active ranks stall.
        """
        state, active_mask = self.record.state, self.record.active_mask
        active = _ranks(active_mask)
        expected = {MembershipState.PROPOSE: Reply.READY, MembershipState.INSTALL: Reply.INSTALLED}
        waited_on = 0
        if state in expected:
            waited_on = rank_mask(
                (r for r in active if not self._rank_replied(r, reports, expected[state])),
                self.world_size,
            )
        # During the startup a rank that started waits on every rank that has not.
        not_started = 0
        if state == MembershipState.STARTING:
            not_started = rank_mask(
                (r for r in active if not reports.get(r, RankReport()).started), self.world_size
            )
        # A rank's waits count only while its reports advance.
        waits: dict[int, int] = {}
        rounds: dict[int, set[tuple]] = {}
        for r in active:
            report = reports.get(r, RankReport())
            if self._reports_advance(r, reports, now_s):
                waiting_for = report.waiting_for | (not_started if report.started else 0)
                waits[r] = waiting_for & active_mask & ~(1 << r)
                waited_on |= waits[r]
            for peer, round_flag in report.waited_flags:
                rounds.setdefault(peer, set()).add(("all-to-all", round_flag))
            for peer, epoch in report.waited_epochs:
                rounds.setdefault(peer, set()).add(("host exchange", epoch))
        chain_ends = _chain_ends(active, waits)
        failed = active_mask & ~connected
        for r in _ranks(failed):
            self._causes.setdefault(r, "disconnected")
        self.stalled_since_s = None
        for r in active:
            report = reports.get(r, RankReport())
            progress = (report.progress, report.install_steps, frozenset(rounds.get(r, ())))
            progressed_s = _update_last_change(self._progress, r, progress, now_s)
            # A rank in a call that its hang detector's pause covers, such as rank 0 waiting for
            # requests, keeps the others waiting but is not hung, also once its reports stop: the
            # call may hold the GIL that its membership client thread needs. Its node supervisor
            # reads the pause from shared memory, so a stopped rank or one that left the pause is
            # not taken as paused. During a change every rank pauses its hang detector, so the
            # pause counts only outside a change.
            paused = state == MembershipState.COMMITTED and r in paused_s
            if paused:
                self._last_paused_s[r] = paused_s[r]
            if has_rank(waited_on, r) and not paused and has_rank(chain_ends, r):
                # A rank whose reports stopped advancing has stalled since its last report, or
                # since it left a pause, whichever is later.
                stalled_s = now_s if r in waits else self._received_s(r, reports, now_s)
                since_s = max(
                    self._hung_since.setdefault(r, stalled_s),
                    progressed_s,
                    self._last_paused_s.get(r, progressed_s),
                )
                # Publish a stall only once it outlasts a report interval.
                if now_s - since_s > REPORT_INTERVAL_S and (
                    self.stalled_since_s is None or since_s < self.stalled_since_s
                ):
                    self.stalled_since_s = since_s
                # A joiner that waits on a peer may be held up by it, so it gets the full timeout.
                stalled_joiner = r in self._joining and not waits.get(r)
                limit_s = min(timeout_s, self.admission_timeout_s) if stalled_joiner else timeout_s
                if now_s - since_s > limit_s:
                    failed |= 1 << r
                    self._causes.setdefault(r, "hung")
                    if self._detected_s is None or since_s < self._detected_s:
                        self._detected_s = since_s
                    logger.warning(
                        f"Rank {r} is hung. It made no progress for {now_s - since_s:.1f} s while "
                        "active ranks wait for it. Last reports: "
                        f"{self._describe(active, reports, waits, now_s, paused_s)}"
                    )
            else:
                self._hung_since.pop(r, None)
        return failed

    def _describe(
        self,
        active: list[int],
        reports: dict[int, RankReport],
        waits: dict[int, int],
        now_s: float,
        paused_s: dict[int, float],
    ) -> str:
        described = []
        for r in active:
            report = reports.get(r, RankReport())
            described.append(
                f"rank {r}: iteration {report.progress}, paused {r in paused_s}, "
                f"waits for {_ranks(waits.get(r, 0))}, "
                f"all-to-all rounds {list(report.waited_flags)}, "
                f"host-exchange epochs {list(report.waited_epochs)}, reported "
                f"{now_s - self._received_s(r, reports, now_s):.1f} s ago"
            )
        return "; ".join(described)
