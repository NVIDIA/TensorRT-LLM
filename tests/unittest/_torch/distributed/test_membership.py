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
"""The Wide-EP fault-tolerance authority's decisions, driven by simulated ranks.

The authority process and the membership clients run over loopback sockets and a real TCPStore.
"""

import contextlib
import logging
import os
import random
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import weakref
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _torch.distributed.store_group_test_utils import InMemoryStore
from pydantic import ValidationError
from torch.distributed import TCPStore

from tensorrt_llm._torch.distributed import membership
from tensorrt_llm._torch.distributed.membership import (
    LIVENESS_KEY,
    RECORD_KEY,
    RENDEZVOUS_ENV,
    SHARED_SECRET_ENV,
    STALLED_SINCE_KEY,
    TRANSACTION_KEY,
    Authority,
    AuthorityProcess,
    MembershipClient,
    MembershipRecord,
    MembershipState,
    NodeReport,
    NodeView,
    RankReport,
    Reply,
    ResumeState,
    SignedStore,
    TransactionSummary,
    UnsignedValueError,
    authority,
    authority_process,
    liveness,
    open_liveness_connection,
    open_node_supervisor_connection,
    open_store,
    rendezvous_store,
)
from tensorrt_llm._torch.distributed.membership.liveness import Lines, hello_token
from tensorrt_llm.logger import logger

pytestmark = pytest.mark.cpu_only

WORLD = 4
ALL = (1 << WORLD) - 1
# How long a test waits for a condition or for a thread to end.
TIMEOUT_S = 60.0
JOIN_TIMEOUT_S = 10.0


def _without(*ranks):
    """Return the mask of every rank except ``ranks``."""
    return ALL & ~sum(1 << rank for rank in ranks)


@pytest.fixture(autouse=True)
def _report_interval(monkeypatch):
    """Shorten the report and decision interval for the tests, and set the shared secret."""
    for module in (authority, authority_process, liveness):
        monkeypatch.setattr(module, "REPORT_INTERVAL_S", 0.05)
    monkeypatch.setenv(SHARED_SECRET_ENV, "ab" * 32)


class Cluster:
    """Simulate ranks that report their progress and reply to the authority's records."""

    def __init__(self, world=WORLD, **timeouts):
        self.authority = Authority(world, **timeouts)
        started = {r: RankReport(started=True) for r in range(world)}
        record = self.authority.tick(0.0, started, (1 << world) - 1)
        assert record.state == MembershipState.COMMITTED
        self.now = 0.0
        self.alive = set(range(world))  # processes that hold their connection
        self.hung = set()  # connected but making no progress
        self.stopped = set()  # stopped by SIGSTOP, still connected, but no longer reporting
        # Hang detector paused, as while waiting for requests, which the node supervisor reports.
        self.paused = set()
        self.shutdown = set()  # ranks whose report says that the deployment ends normally
        self.waiting_for = {}  # rank -> mask it waits for
        self.waited_flags = {}  # rank -> ((peer, round flag its kernels wait for), ...)
        # rank -> ((peer, epoch of the host-exchange round it waits in), ...)
        self.waited_epochs = {}
        self.installing = set()  # completing a step of an install at every report
        self.unprepared = set()  # replacements that have not reported they are ready to join
        self.install_reply = {r: Reply.INSTALLED for r in range(world)}
        self.reports = {r: RankReport() for r in range(world)}
        self.paused_at = {r: ResumeState(iteration=10 + r, epoch=100 + r) for r in range(world)}
        self.published = []
        self.published_at = []

    def step(self, seconds=0.05):
        self.now += seconds
        record = self.authority.record
        for r in self.alive:
            if r in self.stopped:
                continue
            old = self.reports[r]
            reply, generation = old.reply, old.reply_generation
            if r not in self.hung and record.is_active(r):
                if record.state == MembershipState.PROPOSE:
                    reply, generation = Reply.READY, record.generation
                elif record.state == MembershipState.INSTALL:
                    reply, generation = self.install_reply[r], record.generation
            busy = r not in self.hung and r not in self.paused
            self.reports[r] = RankReport(
                received_s=self.now,
                progress=old.progress + (1 if busy else 0),
                waiting_for=self.waiting_for.get(r, 0),
                waited_flags=self.waited_flags.get(r, ()),
                waited_epochs=self.waited_epochs.get(r, ()),
                install_steps=old.install_steps + (1 if r in self.installing else 0),
                reply_generation=generation,
                reply=reply,
                paused_at=self.paused_at[r],
                joining_ready=r not in self.unprepared,
                shutdown=r in self.shutdown,
            )
        connected = sum(1 << r for r in self.alive)
        node = NodeReport(paused_mask=sum(1 << r for r in self.paused), received_s=self.now)
        new = self.authority.tick(self.now, self.reports, connected, [node])
        if new is not None:
            self.published.append(new)
            self.published_at.append(self.now)
        return new

    def run(self, seconds):
        for _ in range(round(seconds / 0.05)):
            self.step()
        return self.authority.record

    def run_until(self, state):
        """Step until the authority's record is in ``state``."""
        for _ in range(100):
            if self.authority.record.state == state:
                break
            self.step()
        assert self.authority.record.state == state


def states(cluster):
    return [r.state for r in cluster.published]


TRANSACTION = [MembershipState.PROPOSE, MembershipState.INSTALL, MembershipState.COMMITTED]


@pytest.mark.parametrize("rank", [-1, WORLD, 1 << 40])
def test_rank_mask_rejects_ranks_outside_world(rank):
    with pytest.raises(ValueError, match=f"got {rank}$"):
        membership.rank_mask([0, rank], WORLD)


def test_rank_mask_counts_repeated_rank_once():
    assert membership.rank_mask([1, 1, 3], WORLD) == 0b1010


def test_healthy_ranks_change_nothing():
    cluster = Cluster()
    cluster.run(5.0)
    assert cluster.published == []
    assert cluster.authority.record == MembershipRecord(
        generation=0, state=MembershipState.COMMITTED, active_mask=ALL
    )


@pytest.mark.parametrize(
    "epochs, resume_epoch", [({}, 104), ({0: 0xFFFFFFFE, 1: 1, 3: 1}, 2)], ids=["plain", "wrapped"]
)
def test_closed_connection_removes_the_rank(epochs, resume_epoch):
    cluster = Cluster()
    for r, epoch in epochs.items():  # ranks 1 and 3 went on past the wrap
        cluster.paused_at[r] = ResumeState(iteration=10 + r, epoch=epoch)
    cluster.alive.discard(2)
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert record.generation == 1
    assert record.active_mask == _without(2)
    # Resume from the furthest survivor, with the next epoch.
    assert record.resume.iteration == 13
    assert record.resume.epoch == resume_epoch


@pytest.mark.parametrize("waits_in", ["all-to-all", "host exchange"])
def test_delivered_rounds_are_not_a_hang(waits_in):
    """Keep a rank whose iteration stalls while every round its peers wait for still arrives."""
    cluster = Cluster(hang_timeout_s=1.0)
    rounds = cluster.waited_flags if waits_in == "all-to-all" else cluster.waited_epochs
    cluster.hung.add(3)  # its iteration does not advance
    cluster.waiting_for.update({r: 1 << 3 for r in range(3)})
    for round_ in range(1, 13):  # a new round every 0.25 s for 3 s
        rounds.update({r: ((3, round_),) for r in range(3)})
        cluster.run(0.25)
    assert cluster.published == []
    record = cluster.run(1.5)  # the same round for longer than the hang timeout
    assert record.state == MembershipState.COMMITTED
    assert not record.is_active(3)


# The hang rule. Only the end of a wait-for chain that does not move for the hang timeout is hung.


def _first_change(cluster, seconds):
    """Return the first record the authority publishes within ``seconds``, or None."""
    for _ in range(round(seconds / 0.05)):
        if cluster.step() is not None:
            return cluster.published[-1]
    return None


def _release(cluster):
    """Resume the active ranks after the authority removed the hung ones."""
    cluster.hung.clear()
    cluster.waiting_for.clear()
    cluster.waited_flags.clear()
    cluster.waited_epochs.clear()


@pytest.mark.parametrize("stopped", [False, True])
def test_only_wait_chain_end_is_hung(stopped):
    """Remove only rank 2, stopped or hung, when rank 3 waits on it and ranks 0 and 1 on both."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.run(0.5)
    (cluster.stopped if stopped else cluster.hung).add(2)
    cluster.hung.update({0, 1, 3})
    cluster.waiting_for.update({0: 0b1100, 1: 0b1100, 3: 1 << 2})
    cluster.waited_flags[3] = ((2, 7),)
    hung_s = cluster.now
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == _without(2)
    assert 1.0 < cluster.now - hung_s <= 1.2
    _release(cluster)
    assert cluster.run(1.0).active_mask == _without(2)
    assert states(cluster) == TRANSACTION


@pytest.mark.parametrize("stopped_in", ["all-to-all", "host exchange", "stale wait"])
def test_stopped_process_is_hung_anywhere(stopped_in):
    """Remove rank 2 one hang timeout after it stops while waiting on a peer, keeping the peer.

    The peer then waits on rank 2, except for a stale wait, where the peer works on.
    """
    cluster = Cluster(hang_timeout_s=1.0)
    peer = 3
    cluster.waiting_for[2] = 1 << peer
    if stopped_in == "all-to-all":
        cluster.waited_flags[2] = ((peer, 7),)
    elif stopped_in == "host exchange":
        cluster.waited_epochs[2] = ((peer, 40),)
    cluster.run(0.5)
    cluster.stopped.add(2)
    waiting = (0, 1) if stopped_in == "stale wait" else (0, 1, 3)
    cluster.hung.update(waiting)
    cluster.waiting_for.update({r: 1 << 2 for r in waiting})
    stopped_s = cluster.now
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == _without(2)
    assert 1.0 < cluster.now - stopped_s <= 1.2
    _release(cluster)
    assert cluster.run(1.0).state == MembershipState.COMMITTED
    assert MembershipState.ABORTED not in states(cluster)


@pytest.mark.parametrize("reports", ["stop", "go on"])
@pytest.mark.parametrize("paused", [True, False])
def test_paused_rank_is_not_hung(paused, reports):
    """Keep rank 2 while its latest report says its hang detector is paused, and remove it if not.

    Ranks 0, 1 and 3 wait on it in a host exchange for three hang timeouts. It makes no progress,
    and either stops reporting, as in a paused call that holds the GIL, or reports on.
    """
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.hung.add(2)
    if paused:
        cluster.paused.add(2)
    cluster.run(0.1)
    if reports == "stop":
        cluster.stopped.add(2)
    waiting = (0, 1, 3)
    cluster.hung.update(waiting)
    cluster.waiting_for.update({r: 1 << 2 for r in waiting})
    cluster.waited_epochs.update({r: ((2, 40),) for r in waiting})
    record = _first_change(cluster, 3.0)
    if paused:
        assert record is None
    else:
        assert record.state == MembershipState.PROPOSE
        assert record.active_mask == _without(2)


def test_late_rank_after_idle_not_hung():
    """Keep a rank late to the first collective after an idle period, since idle time is no stall."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.paused.update(range(WORLD))  # rank 0 waits for requests and the others wait on it
    cluster.waiting_for.update({r: 1 for r in range(1, WORLD)})
    cluster.run(5.0)
    cluster.paused.clear()  # a request arrives, and rank 2 is late to the next collective
    cluster.hung.update(range(WORLD))
    cluster.waiting_for.update({r: 1 << 2 for r in (0, 1, 3)})
    cluster.run(0.3)
    _release(cluster)
    cluster.run(2.0)
    assert cluster.published == []


@pytest.mark.parametrize("how", ["hung", "stopped"])
@pytest.mark.parametrize("phase", [MembershipState.PROPOSE, MembershipState.INSTALL])
def test_hang_during_transaction_restarts_it(phase, how):
    """Fail a rank that hangs in a phase like a crashed one, after the hang timeout."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(1)  # a crash starts the transaction
    cluster.run_until(phase)
    getattr(cluster, how).add(3)
    phase_s = cluster.now
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == 0b0101
    assert 1.0 < cluster.now - phase_s <= 1.2
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == 0b0101
    assert MembershipState.ABORTED not in states(cluster)


@pytest.mark.parametrize("cause", ["install", "rank 0"])
def test_failed_install_or_rank0_loss_aborts(cause):
    """Abort on a failed survivor install or the loss of rank 0."""
    cluster = Cluster()
    if cause == "install":
        cluster.install_reply[1] = Reply.FAILED
    cluster.alive.discard(0 if cause == "rank 0" else 3)
    assert cluster.run(1.0).state == MembershipState.ABORTED
    assert MembershipState.COMMITTED not in states(cluster)
    assert cluster.authority.last_transaction.outcome == "ABORTED"
    assert cluster.authority.last_transaction.changes == {"abort": 1}
    assert cluster.authority.last_transaction.reason == cluster.authority.record.reason != ""


def test_hang_detection_dated_at_stall_start():
    """Date a hang's transaction from the start of the stall, which the authority publishes."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.hung.add(1)
    cluster.waiting_for[0] = 1 << 1
    cluster.run(0.5)
    assert cluster.authority.stalled_since_s == pytest.approx(0.05)
    record = cluster.run(2.0)
    assert record.state == MembershipState.COMMITTED
    assert cluster.authority.stalled_since_s is None
    summary = cluster.authority.last_transaction
    assert summary.causes == {1: "hung"}
    assert summary.began_unix_ms - summary.detected_unix_ms == pytest.approx(1050.0, abs=60.0)


def test_survivor_failure_during_admission_commits():
    """Commit when rank 1 fails while rank 2's replacement is admitted after ranks 2 and 3 failed."""
    cluster = Cluster()
    cluster.alive -= {2, 3}
    assert cluster.run(1.0).state == MembershipState.COMMITTED
    _join(cluster, 2)  # rank 3's replacement did not ask to join, so rank 2's does not wait for it
    cluster.run_until(MembershipState.PROPOSE)
    cluster.alive.discard(1)
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == 0b0101
    assert MembershipState.ABORTED not in states(cluster)


def test_commit_waits_for_every_install():
    """Never commit a generation that a rank replied READY to but did not install."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(3)
    cluster.run(1.0)  # in generation 1, every survivor's last reply is INSTALLED
    cluster.alive.discard(2)
    cluster.run_until(MembershipState.INSTALL)
    cluster.hung.add(1)  # replied READY to generation 2, never INSTALLED
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == 0b0001
    committed = [r.generation for r in cluster.published if r.state == MembershipState.COMMITTED]
    assert committed == [1, 3]  # never generation 2, which rank 1 did not install


def test_hang_clock_restarts_at_commit():
    """Give a rank waited on during the install a full hang timeout after the commit."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(3)
    cluster.run_until(MembershipState.INSTALL)
    cluster.hung.add(1)
    cluster.waiting_for[2] = 1 << 1
    assert cluster.run(0.8).state == MembershipState.INSTALL
    # It replies without progress (the hang detector stays paused through the change), and the
    # change commits.
    cluster.hung.discard(1)
    cluster.paused.add(1)
    assert _first_change(cluster, 0.2).state == MembershipState.COMMITTED
    cluster.paused.discard(1)
    cluster.hung.add(1)  # its first step after the commit
    cluster.run(0.5)
    _release(cluster)
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == 0b0111
    assert states(cluster) == TRANSACTION


def test_progressing_install_is_not_hung():
    """Count each completed install step as progress, so only a stalled install is a hang."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(1)
    cluster.run_until(MembershipState.INSTALL)
    cluster.hung.add(3)  # replies nothing and its iteration does not advance ...
    cluster.installing.add(3)  # ... but it completes install steps
    cluster.run(3.0)
    assert cluster.authority.record.state == MembershipState.INSTALL
    cluster.installing.discard(3)
    stalled_s = cluster.now
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == 0b0101
    assert 1.0 < cluster.now - stalled_s <= 1.2


def test_stalled_joiner_left_out_early():
    """Leave out a joining rank that stalls in its admission after the admission timeout."""
    admission_s = 2.0
    cluster = Cluster(admission_timeout_s=admission_s)
    cluster.alive.discard(2)
    cluster.run(1.0)
    _join(cluster, 2)
    cluster.hung.add(2)  # ready to join, but it never installs the change that adds it
    cluster.run_until(MembershipState.INSTALL)
    stalled_s = cluster.now
    record = _first_change(cluster, admission_s + 1.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == _without(2)
    assert cluster.now - stalled_s <= admission_s + 0.2


def test_ready_joiner_not_hung_while_survivor_finishes_step():
    """Keep a joiner whose readiness counts as READY while a survivor finishes its step.

    The joiner sends no READY of its own and makes no progress, so the authority must not wait on
    it, or its admission timeout would leave it out.
    """
    admission_s = 2.0
    cluster = Cluster(admission_timeout_s=admission_s)
    cluster.alive.discard(2)
    cluster.run(1.0)
    _join(cluster, 2)
    cluster.run_until(MembershipState.PROPOSE)
    proposed = cluster.authority.record
    cluster.hung.update({1, 2})  # neither replies READY
    cluster.installing.add(1)  # rank 1 still makes progress in its step
    assert cluster.run(admission_s + 1.0) == proposed
    _release(cluster)
    cluster.installing.clear()
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == ALL


def test_frozen_survivor_removed_before_joiner():
    """Remove a survivor that froze during an admission, not the joiner it holds up."""
    cluster = Cluster(hang_timeout_s=4.0, admission_timeout_s=1.0)
    cluster.alive.discard(3)
    cluster.run(1.0)
    _join(cluster, 3)
    cluster.run_until(MembershipState.INSTALL)
    # Rank 2 freezes while it waits for the joiner's slot, and the joiner waits on rank 2's.
    cluster.waiting_for.update({2: 1 << 3, 3: 1 << 2})
    cluster.hung.update({2, 3})
    cluster.step()
    cluster.stopped.add(2)
    record = _first_change(cluster, 8.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == _without(2)


def test_unconnected_replacement_join_request_dropped():
    """Drop the join request of a replacement that never connects."""
    cluster = Cluster()
    cluster.alive.discard(3)
    cluster.run(1.0)
    cluster.reports[3] = RankReport()
    cluster.authority.request_join(3)  # the replacement never connects
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert not record.is_active(3)


def _join(cluster, rank):
    cluster.alive.add(rank)  # a new process for the rank, with fresh reports
    cluster.reports[rank] = RankReport()
    cluster.authority.request_join(rank)


def test_stalled_replacement_startup_reported():
    """Report a replacement whose startup stalls for the rendezvous timeout, but not a slow one.

    The wait for the memory of the process it replaces is not a stall.
    """
    cluster = Cluster(rendezvous_timeout_s=1.0)
    cluster.hung.add(1)  # Rank 1 holds the change that removes rank 3 before its commit.
    cluster.alive.discard(3)
    cluster.step()
    _join(cluster, 3)
    cluster.hung.add(3)
    cluster.unprepared.add(3)
    cluster.run(2.0)
    assert cluster.authority.stalled_replacements == ()
    cluster.hung.discard(1)
    cluster.alive -= {2, 3}
    cluster.run(1.0)
    assert not cluster.authority.record.active_mask & 0b1100
    _join(cluster, 2)
    _join(cluster, 3)
    cluster.unprepared.add(2)
    cluster.run(3.0)
    assert cluster.authority.stalled_replacements == (3,)


@pytest.mark.parametrize("late", ["not ready", "no report"])
def test_replacements_added_in_one_transaction(late):
    """Add a node's replacements together, waiting for one that is not ready or has not reported.

    The authority's clock is far from zero, as a monotonic clock is.
    """
    cluster = Cluster()
    cluster.now = 1e6
    cluster.alive -= {2, 3}
    cluster.run(1.0)
    cluster.published.clear()
    _join(cluster, 2)
    _join(cluster, 3)
    if late == "not ready":
        cluster.unprepared.add(3)
    else:
        del cluster.reports[3]  # connected, but no report arrived yet
        cluster.stopped.add(3)
    cluster.run(2.0)
    assert cluster.published == []  # waits for rank 3's replacement
    cluster.unprepared.discard(3)
    cluster.stopped.discard(3)
    cluster.reports.setdefault(3, RankReport())
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert record.active_mask == ALL
    assert cluster.authority.last_transaction.added_ranks == (2, 3)
    assert cluster.authority.last_transaction.changes == {"failover": 1, "admission": 1}
    # The transaction is detected at rank 2's readiness and includes the wait for rank 3's.
    summary = cluster.authority.last_transaction
    assert summary.began_unix_ms - summary.detected_unix_ms >= 2000


@pytest.mark.parametrize("failure", ["failed_reply", "hang"])
@pytest.mark.parametrize("phase", [MembershipState.PROPOSE, MembershipState.INSTALL])
def test_failed_replacement_left_out_of_change(phase, failure):
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(2)
    cluster.run(1.0)
    _join(cluster, 2)
    cluster.run_until(phase)
    if failure == "failed_reply" and phase == MembershipState.INSTALL:
        cluster.install_reply[2] = Reply.FAILED
    else:
        cluster.hung.add(2)  # never replies again
        if failure == "failed_reply":
            cluster.reports[2] = cluster.reports[2].model_copy(
                update=dict(
                    reply=Reply.FAILED, reply_generation=cluster.authority.record.generation
                )
            )
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == _without(2)
    assert record.joining_mask == 0
    assert MembershipState.ABORTED not in states(cluster)
    if failure == "failed_reply":
        # The node supervisor reads why the change left the replacement out.
        assert any("could not install" in r.reason for r in cluster.published)


@pytest.mark.parametrize("phase", [MembershipState.PROPOSE, MembershipState.INSTALL])
@pytest.mark.parametrize("victim", [1, 2, 3])
def test_second_failure_in_each_phase_commits(phase, victim):
    cluster = Cluster()
    first = 1 if victim != 1 else 2
    cluster.alive.discard(first)
    cluster.run_until(phase)
    cluster.alive.discard(victim)
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED
    assert not record.is_active(first)
    assert not record.is_active(victim)


def test_rank_0_shutdown_report_ends_deployment():
    """End the deployment on rank 0's shutdown report only, so leaving ranks are not failures."""
    cluster = Cluster()
    cluster.shutdown.add(1)
    cluster.run(1.0)
    assert cluster.published == []
    cluster.shutdown.add(0)
    cluster.step()
    assert cluster.authority.record.state == MembershipState.SHUTDOWN
    cluster.alive.clear()
    cluster.run(2.0)
    assert [record.state for record in cluster.published] == [MembershipState.SHUTDOWN]


# The startup stays STARTING until every rank has started. Any loss before then aborts at once.


def _starting_reports(started=(), received_s=0.0):
    return {r: RankReport(started=r in started, received_s=received_s) for r in range(WORLD)}


@pytest.mark.parametrize("failed_rank", [0, 2])
def test_startup_rank_failure_aborts_at_once(failed_rank):
    """Abort without waiting for the rendezvous timeout, and give the reason in the record."""
    authority = Authority(WORLD)
    authority.tick(0.0, _starting_reports(), ALL)
    record = authority.tick(0.05, _starting_reports(started={1}), _without(failed_rank))
    assert record.state == MembershipState.ABORTED
    assert record.reason == f"ranks [{failed_rank}] failed during the startup"


@pytest.mark.parametrize("failure", ["absent", "stopped", "not started"])
def test_startup_failure_aborts_at_rendezvous_timeout(failure):
    """Abort when a rank fails the startup for the rendezvous timeout, and name it in the record."""
    authority = Authority(WORLD, rendezvous_timeout_s=60.0)
    for second in range(0, 91):
        connected = ALL
        if failure == "absent":
            reports = _starting_reports(started={0, 1, 2}, received_s=second)
            connected = _without(3)
        elif failure == "stopped":
            reports = {
                r: RankReport(received_s=second, progress=3, waiting_for=1 << 3)
                for r in range(WORLD)
            }
            reports[3] = RankReport(received_s=min(second, 10), progress=3)  # stopped at 10 s
        else:
            reports = {
                r: RankReport(received_s=second, progress=3, started=r != 3) for r in range(WORLD)
            }
        record = authority.tick(float(second), reports, connected)
        if second <= 60:
            assert record is None
    assert authority.record.state == MembershipState.ABORTED
    if failure == "absent":
        assert authority.record.reason == "ranks [3] did not reach the rendezvous within 60 s"
    else:
        assert authority.record.reason == "ranks [3] failed during the startup"


@pytest.mark.parametrize("peer", ["hangs", "progresses"])
def test_startup_collective_hang_aborts(peer):
    """Abort when a waited-on rank completes no startup collective for the rendezvous timeout."""
    authority = Authority(WORLD, rendezvous_timeout_s=600.0)
    for second in range(0, 1300, 10):
        reports = {
            r: RankReport(received_s=second, progress=3, waiting_for=1 << 2) for r in range(WORLD)
        }
        reports[2] = RankReport(
            received_s=second, progress=3 + (second if peer == "progresses" else 0)
        )
        record = authority.tick(float(second), reports, ALL)
        if second <= 600 or peer == "progresses":
            assert record is None
    if peer == "hangs":
        assert authority.record.state == MembershipState.ABORTED
        assert authority.record.reason == "ranks [2] failed during the startup"


def test_first_failover_cause_after_staggered_startup():
    """Give a hung rank's cause, not that it connected after the others during the startup."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.authority = Authority(WORLD, hang_timeout_s=1.0)
    cluster.authority.tick(0.0, _starting_reports(), _without(1))
    record = cluster.authority.tick(0.05, _starting_reports(started=range(WORLD)), ALL)
    assert record.state == MembershipState.COMMITTED
    cluster.now = 0.05
    cluster.hung.add(1)
    cluster.waiting_for[0] = 1 << 1
    assert cluster.run(2.0).state == MembershipState.COMMITTED
    assert cluster.authority.last_transaction.removed_ranks == (1,)
    assert cluster.authority.last_transaction.causes == {1: "hung"}


class SignedTcpStore:
    """A real TCPStore that each thread reaches through its own signing client, as a process does."""

    def __init__(self):
        self._server = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
        self._clients = threading.local()
        self.host, self.port = "127.0.0.1", self._server.port

    def __getattr__(self, name):
        if not hasattr(self._clients, "store"):
            client = TCPStore(self.host, self.port, is_master=False, timeout=timedelta(seconds=30))
            self._clients.store = SignedStore(client)
        return getattr(self._clients.store, name)


@contextlib.contextmanager
def _deployment(world, store=None, reply=None, **authority_options):
    """Run an authority process and a client per rank, yield the store and clients, then shut down.

    The store is in memory unless ``store`` is given. The clients answer records with ``reply``,
    by default as an executor does.
    """
    store = InMemoryStore() if store is None else store
    thread, returned = _serve(
        store, AuthorityProcess(world, world, store, Authority(world, **authority_options))
    )
    clients = []
    try:
        clients += [_member(store, r, world, reply=reply) for r in range(world)]
        _until(
            lambda: all(
                client.record is not None and client.record.state == MembershipState.COMMITTED
                for client in clients
            )
        )
        yield store, clients
        clients[0].report(shutdown=True)
        _until(
            lambda: (
                MembershipRecord.model_validate_json(store.get(RECORD_KEY)).state
                == MembershipState.SHUTDOWN
            )
        )
        # The store lives in the authority's process, which serves until every active rank has read
        # the record.
        thread.join(timeout=0.2)  # A brief wait, after which it still serves.
        assert thread.is_alive()
    finally:
        _leave(clients)
        thread.join(JOIN_TIMEOUT_S)
    assert [record.state for record in returned] == [MembershipState.SHUTDOWN]


def _serve(store, process):
    """Run ``process`` on a thread until it serves, and return the thread and what its run returned.

    A run that raises a ``RuntimeError`` returns the error.
    """
    returned = []

    def run():
        try:
            returned.append(process.run(host="127.0.0.1"))
        except RuntimeError as error:
            returned.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    _until(lambda: store.check([membership.LIVENESS_KEY]))
    return thread, returned


def _until(condition, timeout_s=TIMEOUT_S):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _member(store, rank, world, replacement=False, on_error=None, reply=None):
    """Start a rank's client that replies to each record with ``reply``, or as an executor does."""
    reply = _reply if reply is None else reply
    client = MembershipClient(
        rank,
        store,
        lambda record: reply(client, record),
        replacement=replacement,
        on_error=on_error,
    )
    client.start()
    if replacement:
        client.report(joining_ready=True)
    else:
        client.report(started=True)
    return client


def _reply(client, record):
    """Reply to ``record`` as an executor does."""
    if record.state == MembershipState.PROPOSE:
        client.report(reply=Reply.READY, reply_generation=record.generation)
    elif record.state == MembershipState.INSTALL:
        client.report(reply=Reply.INSTALLED, reply_generation=record.generation)


def _refused(store, rank, replacement):
    """Return whether the authority closes a liveness connection for ``rank``."""
    with open_liveness_connection(store, rank, replacement) as connection:
        connection.sock.settimeout(10)
        return connection.sock.recv(1 << 16) == b""


def _leave(clients):
    for client in clients:
        if client._thread.is_alive():
            client.stop()


@pytest.mark.parametrize("end", [MembershipState.SHUTDOWN, MembershipState.ABORTED])
def test_authority_ends_with_rank_connected(end):
    """End the authority after rank 0 read SHUTDOWN, or a hang timeout after an abort."""
    store = InMemoryStore()
    parent_read, parent_write = os.pipe()
    process = AuthorityProcess(2, 2, store, Authority(2, hang_timeout_s=5.0), parent=parent_read)
    thread, returned = _serve(store, process)
    states_at_close = []
    clients = []
    try:
        clients.append(_member(store, 0, 2))
        clients.append(
            _member(
                store, 1, 2, on_error=lambda error: states_at_close.append(clients[1].record.state)
            )
        )
        _until(lambda: all(c.record is not None and _committed(c, 0b11) for c in clients))
        if end == MembershipState.SHUTDOWN:
            clients[0].report(shutdown=True)
            _until(lambda: clients[0].record.state == MembershipState.SHUTDOWN)
        clients[0].stop()  # The deployment aborts unless it shuts down.
        if end == MembershipState.SHUTDOWN:
            os.close(parent_write)  # as when rank 0's process ends
        _until(lambda: clients[1].record.state == end)
        thread.join(TIMEOUT_S)
        assert not thread.is_alive()
        if end == MembershipState.SHUTDOWN:
            _until(lambda: states_at_close)
            assert states_at_close == [MembershipState.SHUTDOWN]
    finally:
        _leave(clients)
        os.close(parent_read)
        if end == MembershipState.ABORTED:
            os.close(parent_write)
    assert [record.state for record in returned] == [end]


def test_authority_aborts_only_at_end_of_rank_0_pipe():
    """Keep serving when bytes arrive on rank 0's pipe, and abort once the pipe ends."""
    store = InMemoryStore()
    parent_read, parent_write = os.pipe()
    process = AuthorityProcess(1, 1, store, Authority(1), parent=parent_read)
    thread, returned = _serve(store, process)
    clients = []
    try:
        clients.append(_member(store, 0, 1))
        _until(lambda: _committed(clients[0], 0b1))
        os.write(parent_write, b"\n")
        time.sleep(10 * authority.REPORT_INTERVAL_S)
        assert _committed(clients[0], 0b1)
        os.close(parent_write)
        parent_write = None
        _until(lambda: clients[0].record.state == MembershipState.ABORTED)
    finally:
        _leave(clients)
        thread.join(TIMEOUT_S)
        os.close(parent_read)
        if parent_write is not None:
            os.close(parent_write)
    assert [record.state for record in returned] == [MembershipState.ABORTED]
    assert returned[0].reason == "rank 0's process ended"


@pytest.mark.parametrize("end", [MembershipState.SHUTDOWN, MembershipState.ABORTED])
def test_authority_process_runs_until_rank_0_ends(end):
    """Run the authority process as rank 0 starts it, and end it as rank 0 does.

    The process writes the store's address to ``--address-fd`` once the store listens there, runs
    on every CPU though started bound to one, ignores SIGINT and SIGTERM, and exits with 0 after a
    shutdown, else with 1.
    """
    # Reserved with SO_REUSEADDR until the process ends, so that the authority can still bind it.
    reserved = socket.socket()
    reserved.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    reserved.bind(("127.0.0.1", 0))
    address = f"127.0.0.1:{reserved.getsockname()[1]}"
    env = {
        **os.environ,
        RENDEZVOUS_ENV: address,
        "TRT_LLM_NO_LIB_INIT": "1",
        "MPI4PY_RC_INITIALIZE": "false",
        "CUDA_VISIBLE_DEVICES": "",
    }
    env.pop("TLLM_NUMA_AWARE_WORKER_AFFINITY", None)
    allowed = os.sched_getaffinity(0)
    read_fd, write_fd = os.pipe()
    os.sched_setaffinity(0, {min(allowed)})
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorrt_llm._torch.distributed.membership",
                "--world-size=1",
                "--gpus-per-node=1",
                f"--address-fd={write_fd}",
            ],
            env=env,
            pass_fds=(write_fd,),
            stdin=subprocess.PIPE,
        )
    finally:
        os.sched_setaffinity(0, allowed)
        os.close(write_fd)
    clients = []
    try:
        with os.fdopen(read_fd) as address_pipe:
            assert select.select([address_pipe], [], [], TIMEOUT_S)[0]
            assert address_pipe.readline() == f"{address}\n"
        store = open_store(address, timeout_s=TIMEOUT_S)
        store.wait([LIVENESS_KEY], timedelta(seconds=TIMEOUT_S))
        assert os.sched_getaffinity(process.pid) >= allowed
        for signum in (signal.SIGINT, signal.SIGTERM):
            process.send_signal(signum)
        clients.append(_member(store, 0, 1))
        _until(lambda: _committed(clients[0], 0b1))
        if end == MembershipState.SHUTDOWN:
            clients[0].report(shutdown=True)
        else:
            process.stdin.close()  # as when rank 0's process ends
        _until(lambda: clients[0].record.state == end)
        clients[0].stop()
        assert process.wait(TIMEOUT_S) == (0 if end == MembershipState.SHUTDOWN else 1)
    finally:
        _leave(clients)
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdin.close()
        reserved.close()


def test_stall_published_in_unix_milliseconds():
    """Publish since when the ranks wait on a stalled rank in Unix milliseconds, then clear it."""
    with _deployment(2) as (store, clients):
        began_ms = time.time() * 1e3
        clients[0].report(waiting_for=0b10)
        _until(lambda: store.check([STALLED_SINCE_KEY]) and store.get(STALLED_SINCE_KEY))
        assert began_ms - 50 <= int(store.get(STALLED_SINCE_KEY)) <= time.time() * 1e3
        clients[0].report(waiting_for=0)
        _until(lambda: store.get(STALLED_SINCE_KEY) == b"")


def test_node_report_keeps_paused_rank():
    """Keep a rank that others wait on while its node supervisor reports it paused, and no longer.

    Rank 1 makes no progress while rank 0 waits on it. Its node supervisor reports it paused every
    report interval for three hang timeouts, and then reports it armed.
    """
    with _deployment(2, hang_timeout_s=1.0) as (store, clients):
        node = _NodeSupervisor(store, 0, [0, 1])
        node.until(lambda view: True)
        clients[0].report(waiting_for=0b10)
        kept_until_s = time.monotonic() + 3.0
        while time.monotonic() < kept_until_s:
            node.report(paused_mask=0b10)
            time.sleep(0.05)
        assert _committed(clients[0], 0b11)
        node.report()
        _until(lambda: _committed(clients[0], 0b01))


def test_rendezvous_store_signs_with_shared_secret(monkeypatch):
    """Open the store named in the environment with a client that signs with the shared secret."""
    server = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    address = f"127.0.0.1:{server.port}"
    monkeypatch.setenv(RENDEZVOUS_ENV, address)
    rendezvous_store(timeout_s=TIMEOUT_S).set("key", b"value")
    assert open_store(address, TIMEOUT_S).get("key") == b"value"
    with pytest.raises(UnsignedValueError):
        open_store(address, TIMEOUT_S, secret="cd" * 32).get("key")


def test_rendezvous_store_requires_address(monkeypatch):
    """Refuse to open the rendezvous store without an address, naming the variable."""
    monkeypatch.delenv(RENDEZVOUS_ENV, raising=False)
    with pytest.raises(ValueError, match=f"requires host:port in {RENDEZVOUS_ENV}"):
        rendezvous_store(timeout_s=TIMEOUT_S)


def test_failed_client_start_closes_connection(monkeypatch):
    """Close the liveness connection when the first record does not arrive, and stop() returns.

    The authority admits a replacement for the rank only once the connection closes.
    """
    monkeypatch.setattr(liveness, "HELLO_TIMEOUT_S", 0.2)
    store = InMemoryStore()
    closed = threading.Event()
    with socket.create_server(("127.0.0.1", 0)) as listener:
        store.set(membership.LIVENESS_KEY, f"127.0.0.1:{listener.getsockname()[1]}".encode())

        def silent_authority():
            connection, _ = listener.accept()
            with connection:
                while connection.recv(1 << 16):
                    pass
            closed.set()

        thread = threading.Thread(target=silent_authority, daemon=True)
        thread.start()
        client = MembershipClient(0, store, lambda record: None)
        with pytest.raises(TimeoutError):
            client.start()
        assert closed.wait(JOIN_TIMEOUT_S)
        client.stop()
        thread.join(JOIN_TIMEOUT_S)


@pytest.mark.parametrize("shutdown", [False, True], ids=["idle", "shutdown"])
def test_authority_heartbeats_keep_ranks_connected(monkeypatch, shutdown):
    """Keep the ranks of an idle or ending deployment connected with the authority's heartbeats."""
    monkeypatch.setattr(authority_process, "HEARTBEAT_INTERVAL_S", 0.1)
    monkeypatch.setattr(liveness, "AUTHORITY_SILENCE_TIMEOUT_S", 2.0)
    with _deployment(2) as (_, clients):
        if shutdown:
            clients[0].report(shutdown=True)
            _until(lambda: all(c.record.state == MembershipState.SHUTDOWN for c in clients))
        time.sleep(2 * liveness.AUTHORITY_SILENCE_TIMEOUT_S)
        # A client that took the authority as hung has ended.
        assert all(client._thread.is_alive() for client in clients)


def test_authority_logs_past_blocked_output(monkeypatch):
    """Log from the authority's loop without waiting for an output stream that blocks."""
    released, written = threading.Event(), []

    class BlockedOutput(logging.Handler):
        def emit(self, record):
            released.wait(JOIN_TIMEOUT_S)
            written.append(record.getMessage())

    monkeypatch.setattr(logging.getLogger(logger.PREFIX), "handlers", [BlockedOutput()])
    listener = authority_process.log_from_own_thread()
    try:
        logger.error("Membership: logged while the output blocks")
        assert written == []
    finally:
        released.set()
        listener.stop()
    assert len(written) == 1


def test_liveness_connections_configured(monkeypatch):
    """Send records at once and enable TCP keepalive at both ends of the connections of a rank and
    of a node supervisor."""
    configure, accepted = liveness.configure_connection, []
    monkeypatch.setattr(
        authority_process,
        "configure_connection",
        lambda sock: configure(sock) or accepted.append(sock),
    )
    with _deployment(1) as (store, _):
        supervisor = _NodeSupervisor(store, 0, [0])
        supervisor.until(lambda view: True)
        # The authority's ends of the rank's and the node supervisor's connections.
        assert len(accepted) == 2
        for sock in (supervisor.connection.sock, *accepted):
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
            assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
            idle_s = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE)
            interval_s = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL)
            probes = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT)
            assert idle_s + interval_s * probes == membership.DEAD_TIMEOUT_S
            user_timeout_ms = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT)
            assert user_timeout_ms == membership.DEAD_TIMEOUT_S * 1000


@pytest.mark.parametrize("published", [False, True])
def test_waiting_for_the_authority_is_logged(monkeypatch, published):
    """Wait in the store and log once, naming the store, if no liveness address arrives in time."""
    monkeypatch.setattr(liveness, "_AUTHORITY_WAIT_WARNING_S", 0.05)
    warnings = []
    monkeypatch.setattr(liveness.logger, "warning", warnings.append)
    tcp_store = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    if published:
        tcp_store.set(LIVENESS_KEY, b"")
    store = Mock(host="127.0.0.1", port=tcp_store.port, wait=Mock(wraps=tcp_store.wait))
    liveness._warn_while_no_authority(store)
    store.wait.assert_called_once()
    assert len(warnings) == (0 if published else 1)
    assert published or f"127.0.0.1:{tcp_store.port}" in warnings[0]


def test_unchanged_membership_sends_nothing():
    """Send no record and write nothing to the store while the membership does not change."""
    with _deployment(2) as (store, clients):
        clients[1].stop()
        _until(lambda: clients[0].record.active_mask == 0b01)
        _until(lambda: clients[0].record.state == MembershipState.COMMITTED)
        writes = Mock(wraps=store.set)
        store.set = writes
        # No transaction starts for a replacement that has not reported that it is ready to join.
        with open_liveness_connection(store, 1, replacement=True) as connection:
            received = b""
            deadline_s = time.monotonic() + 3 * membership.records.HEARTBEAT_INTERVAL_S
            while (left_s := deadline_s - time.monotonic()) > 0:
                connection.sock.settimeout(left_s)
                with contextlib.suppress(socket.timeout):
                    data = connection.sock.recv(1 << 16)
                    assert data, "the authority closed the connection"
                    received += data
            # Only the record that gives it its incarnation arrives, besides empty heartbeat lines.
            assert len([line for line in connection.lines.decode(received) if line]) == 1
            writes.assert_not_called()  # until the connection closes


def test_foreign_or_second_replacement_refused(monkeypatch):
    """Refuse a hello with another deployment's token, or a rank's second connected replacement."""
    with _deployment(2) as (store, clients):
        clients[1].stop()
        _until(lambda: _committed(clients[0], 0b01))
        monkeypatch.setenv(SHARED_SECRET_ENV, "cd" * 32)
        assert _refused(store, 1, True)
        with pytest.raises(ConnectionError, match="refused this process of rank 1"):
            _member(store, 1, 2, replacement=True)
        monkeypatch.setenv(SHARED_SECRET_ENV, "ab" * 32)
        with open_liveness_connection(store, 1, replacement=True) as earlier:
            earlier.sock.settimeout(10)
            assert earlier.sock.recv(1 << 16), "the replacement was refused"
            assert _refused(store, 1, True)
        assert _committed(clients[0], 0b01)


class _NodeSupervisor:
    """Node supervisor connection that reads the views the authority sends."""

    def __init__(self, store, node, ranks):
        self.connection = open_node_supervisor_connection(store, node, ranks, f"host-{node}")
        self.connection.sock.settimeout(10)

    def until(self, predicate) -> NodeView:
        """Return the first view that satisfies ``predicate``."""
        while True:
            data = self.connection.sock.recv(1 << 16)
            assert data, "the authority closed the connection"
            views = [
                NodeView.model_validate_json(line) for line in self.connection.lines.decode(data)
            ]
            if views and predicate(views[-1]):
                return views[-1]

    def report(self, given_up=(), paused_mask=0):
        """Report the ranks that the supervisor gave up and those it found paused."""
        line = NodeReport(given_up=given_up, paused_mask=paused_mask).model_dump_json().encode()
        self.connection.sock.sendall(self.connection.lines.encode(line))


@pytest.mark.parametrize("lost_by", ["close", "foreign given-up rank", "foreign paused rank"])
def test_node_supervisor_loss_fails_its_ranks(monkeypatch, lost_by):
    """Remove a lost supervisor's ranks at once, though their processes run, and mark them given up.

    The removal is one transaction. A record resent with only the given-up ranks changed starts none.
    """
    error = Mock()
    monkeypatch.setattr(authority.logger, "error", error)
    with _deployment(4) as (store, clients):
        states = []

        def on_record(record):
            states.append(record.state)
            _reply(clients[0], record)

        clients[0].bind(on_record)
        # Node 0's supervisor stays connected, because its loss would abort the deployment.
        node_0 = _NodeSupervisor(store, 0, [0, 1])
        node_1 = _NodeSupervisor(store, 1, [2, 3])
        node_1.until(lambda view: True)
        if lost_by == "close":
            node_1.connection.sock.close()
        elif lost_by == "foreign given-up rank":
            node_1.report(given_up=(0,))
        else:
            node_1.report(paused_mask=0b0001)
        _until(lambda: _committed(clients[0], 0b0011))
        node_0.until(lambda view: view.record.active_mask == 0b0011)
        assert clients[0].record.given_up_mask == 0b1100
        generation = clients[0].record.generation
        transaction = store.get(TRANSACTION_KEY)
        summary = TransactionSummary.model_validate_json(transaction)
        assert (summary.generation, summary.removed_ranks) == (generation, (2, 3))
        assert summary.changes == {"failover": 1}
        assert states.count(MembershipState.PROPOSE) == 1

        # Node 1's supervisor connects again and gives up none of its ranks.
        node_1 = _NodeSupervisor(store, 1, [2, 3])
        node_1.report()
        _until(lambda: clients[0].record.given_up_mask == 0)
        assert _committed(clients[0], 0b0011)
        assert clients[0].record.generation == generation
        assert states.count(MembershipState.PROPOSE) == 1
        assert store.get(TRANSACTION_KEY) == transaction
    invalid = {
        "foreign given-up rank": "node 1 gave up ranks [0], which are not all its own",
        "foreign paused rank": (
            "node 1 reported the ranks of mask 0x1 paused, which are not all its own"
        ),
    }
    if lost_by in invalid:
        (message,) = [call.args[0] for call in error.call_args_list if "node 1" in call.args[0]]
        assert message == (
            "Closing the connection of the supervisor of node 1, because it sent an invalid line: "
            f"{invalid[lost_by]}. Its ranks [2, 3] count as failed"
        )


def test_lost_node_rejoins_in_one_change():
    """Admit a lost node's new supervisor and its replacements, which join in one change."""
    with _deployment(4) as (store, clients):
        states = []

        def on_record(record):
            states.append(record.state)
            _reply(clients[0], record)

        clients[0].bind(on_record)
        node_0 = _NodeSupervisor(store, 0, [0, 1])
        node_1 = _NodeSupervisor(store, 1, [2, 3])
        node_1.until(lambda view: True)
        node_1.connection.sock.close()
        _until(lambda: _committed(clients[0], 0b0011))
        node_1 = _NodeSupervisor(store, 1, [2, 3])
        view = node_1.until(lambda view: True)
        assert (view.record.state, view.record.active_mask) == (MembershipState.COMMITTED, 0b0011)
        node_1.report()
        _until(lambda: clients[0].record.given_up_mask == 0)
        # The supervisor starts both replacements at once, so both connect before either is ready.
        joiners = {}
        for rank in (2, 3):
            joiners[rank] = MembershipClient(
                rank,
                store,
                lambda record, rank=rank: _reply(joiners[rank], record),
                replacement=True,
            )
            joiners[rank].start()
            clients.append(joiners[rank])
        for joiner in joiners.values():
            joiner.report(joining_ready=True)
        _until(lambda: _committed(clients[0], 0b1111))
        assert states.count(MembershipState.PROPOSE) == 2  # the failover and the admission
        summary = TransactionSummary.model_validate_json(store.get(TRANSACTION_KEY))
        assert summary.added_ranks == (2, 3)
        assert clients[0].record.incarnations == (0, 0, 1, 1)
        node_0.until(lambda view: view.record.active_mask == 0b1111)


@pytest.mark.parametrize("node, ranks", [(1, [2, 3]), (2, [3])], ids=["same node", "same rank"])
def test_second_node_supervisor_refused(node, ranks):
    """Refuse a supervisor for a node, or a rank, that a connected supervisor holds."""
    with _deployment(4) as (store, _):
        node_1 = _NodeSupervisor(store, 1, [2, 3])
        node_1.until(lambda view: True)
        with open_node_supervisor_connection(store, node, ranks, f"host-{node}") as second:
            second.sock.settimeout(10)
            assert second.sock.recv(1 << 16) == b"", "a second supervisor was admitted"
        # The authority still admits another node's supervisor.
        node_0 = _NodeSupervisor(store, 0, [0, 1])
        assert node_0.until(lambda view: True).record.active_mask == 0b1111


def test_given_up_ranks_published_during_change(monkeypatch):
    """Publish the ranks a node supervisor gave up at once, or during a change with its next record."""
    proposed, resumed = threading.Event(), threading.Event()
    decide = Authority.tick

    def tick(self, now_s, reports, connected_mask, node_reports=()):
        if proposed.is_set() and not resumed.is_set():
            return None
        record = decide(self, now_s, reports, connected_mask, node_reports)
        if record is not None and record.state == MembershipState.PROPOSE:
            proposed.set()
        return record

    monkeypatch.setattr(Authority, "tick", tick)
    with _deployment(4) as (store, clients):
        node_0 = _NodeSupervisor(store, 0, [0, 1])
        node_1 = _NodeSupervisor(store, 1, [2, 3])
        node_1.until(lambda view: True)
        received = []

        def on_record(record):
            received.append(record)
            _reply(clients[0], record)

        clients[0].bind(on_record)
        clients[3].stop()
        _until(lambda: clients[0].record.state == MembershipState.PROPOSE)
        node_1.report(given_up=(3,))
        node_0.until(lambda view: view.record.given_up_mask == 0b1000)
        resumed.set()
        _until(lambda: _committed(clients[0], 0b0111))
        assert [record.state for record in received].count(MembershipState.PROPOSE) == 1
        assert clients[0].record.given_up_mask == 0b1000
        node_1.report()
        _until(lambda: clients[0].record.given_up_mask == 0)
        stored = MembershipRecord.model_validate_json(store.get(RECORD_KEY))
        assert stored.given_up_mask == 0


def test_node_view_names_connected_ranks(monkeypatch):
    """Send node supervisors the ranks with a connected process, again once one leaves."""
    deciding = threading.Event()
    deciding.set()
    decide = Authority.tick

    def tick(self, now_s, reports, connected_mask, node_reports=()):
        return (
            decide(self, now_s, reports, connected_mask, node_reports)
            if deciding.is_set()
            else None
        )

    monkeypatch.setattr(Authority, "tick", tick)
    with _deployment(4) as (store, clients):
        node_0 = _NodeSupervisor(store, 0, [0, 1])
        node_0.until(lambda view: view.connected_mask == 0b1111)
        # No membership change follows the leave, so only the connected ranks change.
        deciding.clear()
        clients[3].stop()
        view = node_0.until(lambda view: view.connected_mask == 0b0111)
        assert view.record.active_mask == 0b1111
        deciding.set()


@pytest.mark.parametrize("connected_nodes", [[0], []], ids=["one", "none"])
def test_startup_without_node_supervisor_aborts(monkeypatch, connected_nodes):
    """Abort a startup in which a node's supervisor did not connect, naming the nodes and the fix."""
    monkeypatch.setattr(authority_process, "NODE_SUPERVISOR_TIMEOUT_S", 0.5)
    store = InMemoryStore()
    thread, returned = _serve(store, AuthorityProcess(4, 2, store, Authority(4)))
    supervisors = [
        _NodeSupervisor(store, node, [2 * node, 2 * node + 1]) for node in connected_nodes
    ]
    try:
        for supervisor in supervisors:
            supervisor.until(lambda view: view.record.state == MembershipState.ABORTED)
        thread.join(JOIN_TIMEOUT_S)
    finally:
        for supervisor in supervisors:
            supervisor.connection.sock.close()
    assert [record.state for record in returned] == [MembershipState.ABORTED]
    missing = [node for node in (0, 1) if node not in connected_nodes]
    assert returned[0].reason.startswith(f"the supervisors of nodes {missing} did not connect")
    assert SHARED_SECRET_ENV in returned[0].reason


@pytest.mark.parametrize("store_type", [InMemoryStore, SignedTcpStore])
def test_replacement_refused_until_rank_removed(monkeypatch, store_type):
    """Refuse a replacement for rank 0, or for a rank until the membership removes it.

    The replacement gets the rank's next incarnation, and the first process is no longer admitted.
    Also over a real store with signed values, as the deployment's processes use.
    """
    deciding = threading.Event()
    deciding.set()
    connected = []
    decide = Authority.tick

    def tick(self, now_s, reports, connected_mask, node_reports=()):
        connected.append(connected_mask)
        return (
            decide(self, now_s, reports, connected_mask, node_reports)
            if deciding.is_set()
            else None
        )

    monkeypatch.setattr(Authority, "tick", tick)
    with _deployment(3, store_type()) as (store, clients):
        assert _refused(store, 0, True)
        assert _refused(store, 2, True)
        deciding.clear()
        clients[2].stop()
        _until(lambda: connected[-1] == 0b011)
        assert clients[0].record.is_active(2)
        assert _refused(store, 2, True)
        deciding.set()
        _until(lambda: _committed(clients[0], 0b011))
        clients.append(_member(store, 2, 3, replacement=True))
        assert clients[-1].incarnation == 1
        _until(lambda: _committed(clients[0], 0b111))
        assert clients[0].record.incarnations == (0, 0, 1)
        assert _refused(store, 2, False)


def test_second_failure_during_failover_install():
    """Fail a second rank while a survivor installs the first one's failover, with real clients.

    Rank 3 holds its reply to the failover's INSTALL, so the INSTALL is open when rank 2 fails.
    The failover's generation never commits, and the next change commits without both ranks.
    """
    held, seen = [], []

    def holding_reply(client, record):
        if client.rank == 0:
            seen.append((record.generation, record.state))
        if client.rank == 3 and record.state == MembershipState.INSTALL and not held:
            held.append(record.generation)
            return
        _reply(client, record)

    with _deployment(4, reply=holding_reply) as (store, clients):
        clients[1].stop()
        _until(lambda: held)
        clients[2].stop()
        _until(lambda: _committed(clients[0], 0b1001))
        summary = TransactionSummary.model_validate_json(store.get(TRANSACTION_KEY))
    (failover,) = held
    assert (failover, MembershipState.INSTALL) in seen
    assert (failover, MembershipState.COMMITTED) not in seen
    assert summary.generation > failover
    assert summary.removed_ranks == (1, 2)


def test_authority_stamps_reports_on_receipt(monkeypatch):
    """Check that the authority stamps each report with the time it received it."""
    stamps = []
    decide = Authority.tick

    def tick(self, now_s, reports, connected_mask, node_reports=()):
        stamps.extend((now_s, report.received_s) for report in reports.values())
        return decide(self, now_s, reports, connected_mask, node_reports)

    monkeypatch.setattr(Authority, "tick", tick)
    with _deployment(2):
        _until(lambda: len(stamps) >= 4)
    assert all(0.0 <= now_s - received_s < 10.0 for now_s, received_s in stamps)


def test_first_process_refused_after_startup(monkeypatch):
    """Refuse a first process once the deployment started, also for a failed rank never replaced."""
    warning = Mock()
    monkeypatch.setattr(authority.logger, "warning", warning)
    with _deployment(3) as (store, clients):
        clients[2]._sock.shutdown(socket.SHUT_RDWR)
        _until(lambda: _committed(clients[0], 0b011))
        assert clients[0].record.incarnations == (0, 0, 0)
        assert _refused(store, 2, False)
    (message,) = [call.args[0] for call in warning.call_args_list if "liveness" in call.args[0]]
    assert message == (
        "Closing a liveness connection: rank 2 joins only as a replacement once the deployment has "
        "started"
    )


def test_replacement_outside_deployment_refused(monkeypatch):
    """Refuse a replacement of a rank outside the deployment, which keeps serving."""
    warning = Mock()
    monkeypatch.setattr(authority.logger, "warning", warning)
    with _deployment(2) as (store, clients):
        assert _refused(store, 2, True)
        assert _committed(clients[0], 0b11)
    (message,) = [call.args[0] for call in warning.call_args_list if "liveness" in call.args[0]]
    assert message == "Closing a liveness connection: rank 2 is not in the deployment"


def _committed(client, active_mask):
    """Return whether ``client``'s newest record is COMMITTED with ``active_mask``."""
    record = client.record
    return record.state == MembershipState.COMMITTED and record.active_mask == active_mask


def test_authority_survives_misbehaving_peer(monkeypatch):
    """Keep serving after a peer whose first record fails to send and who sends an invalid line."""
    with _deployment(2) as (store, clients):
        clients[1].stop()
        _until(lambda: _committed(clients[0], 0b01))
        host, port = store.get(membership.LIVENESS_KEY).decode().rsplit(":", 1)
        hello = Lines.encode(f"{hello_token()} 1 1".encode())
        with socket.create_connection((host, int(port)), timeout=10) as peer:
            send, address = socket.socket.send, peer.getsockname()

            def send_failing_to_peer(sock, data, *args):
                if sock.getpeername() == address:
                    raise ConnectionResetError("the peer reset the connection")
                return send(sock, data, *args)

            monkeypatch.setattr(socket.socket, "send", send_failing_to_peer)
            peer.sendall(hello + b"not a report\n")
            while peer.recv(1 << 16):
                pass
        clients.append(_member(store, 1, 2, replacement=True))
        _until(lambda: _committed(clients[0], 0b11))


def test_authority_failure_aborts_with_reason(monkeypatch):
    """Send the ranks an ABORTED record naming an unexpected authority error, then raise it."""
    store = InMemoryStore()
    thread, raised = _serve(store, AuthorityProcess(2, 2, store, Authority(2)))
    clients = []
    try:
        clients += [_member(store, r, 2) for r in range(2)]
        _until(lambda: clients[0].record is not None and _committed(clients[0], 0b11))
        node = _NodeSupervisor(store, 0, [0, 1])
        node.until(lambda view: True)
        node.report(given_up=(1,))
        _until(lambda: clients[0].record.given_up_mask == 0b10)
        monkeypatch.setattr(store, "set", Mock(side_effect=RuntimeError("the store failed")))
        clients[1].stop()
        _until(lambda: clients[0].record.state == MembershipState.ABORTED)
        thread.join(JOIN_TIMEOUT_S)
    finally:
        _leave(clients)
    assert clients[0].record.reason == "the authority failed: RuntimeError('the store failed')"
    assert clients[0].record.given_up_mask == 0b10
    assert [str(error) for error in raised] == ["the store failed"]


def test_fork_close_keeps_no_closed_socket():
    """Keep no reference to a closed liveness socket for the close in a forked child."""
    with _deployment(1) as (store, _):
        with open_liveness_connection(store, 0) as connection:
            sock = weakref.ref(connection.sock)
        del connection
        assert sock() is None


def _scripted_authority(listener, resend_s, stop):
    """Answer a rank's hello with a record, then resend it every ``resend_s`` if set."""
    sock, _ = listener.accept()
    with sock:
        hello = b""
        while not hello.endswith(b"\n"):
            hello += sock.recv(256)
        lines = Lines()
        record = MembershipRecord(
            generation=0, state=MembershipState.COMMITTED, active_mask=0b1, incarnations=(0,)
        )
        sock.sendall(lines.encode(record.model_dump_json().encode()))
        while not stop.wait(resend_s or None):
            sock.sendall(lines.encode(record.model_dump_json().encode()))


@contextlib.contextmanager
def _scripted_client(errors, resend_s=None):
    """Yield a started client of rank 0, connected to a scripted authority.

    The authority closes the connection first on exit, so the client may then add an error.
    """
    store = InMemoryStore()
    stop = threading.Event()
    with socket.create_server(("127.0.0.1", 0)) as listener:
        store.set(membership.LIVENESS_KEY, f"127.0.0.1:{listener.getsockname()[1]}".encode())
        thread = threading.Thread(
            target=_scripted_authority, args=(listener, resend_s, stop), daemon=True
        )
        thread.start()
        client = MembershipClient(0, store, lambda record: None, on_error=errors.append)
        client.start()
        try:
            yield client
        finally:
            stop.set()
            thread.join(JOIN_TIMEOUT_S)
            _leave([client])


@pytest.mark.parametrize("resend_s", [None, 0.1], ids=["silent", "busy"])
def test_rank_ends_on_silent_authority(monkeypatch, resend_s):
    """End a rank's client once the authority is silent for the silence timeout, not before."""
    silence_s = 2.0
    monkeypatch.setattr(liveness, "AUTHORITY_SILENCE_TIMEOUT_S", silence_s)
    errors = []
    # Before the client's thread starts, which is when its silence timeout starts.
    started_s = time.monotonic()
    with _scripted_client(errors, resend_s):
        if resend_s is None:
            _until(lambda: errors)
            assert time.monotonic() - started_s >= silence_s
            assert isinstance(errors[0], TimeoutError)
        else:
            time.sleep(2 * silence_s)
            assert not errors


@pytest.mark.parametrize(
    "changes", [{"waiting_on": 0b10}, {"progress": "first"}], ids=["unknown", "mistyped"]
)
def test_report_rejects_invalid_field(changes):
    """Refuse a report change that names no report field or has a value of the wrong type."""
    errors = []
    with _scripted_client(errors) as client:
        with pytest.raises(ValidationError):
            client.report(**changes)
        assert not errors


def test_client_holds_first_record_after_start(monkeypatch):
    """Hold the authority's first record once start() returns, before the thread delivers it."""
    monkeypatch.setattr(liveness.MembershipClient, "_serve", lambda self: self._stop.wait())
    with _scripted_client([]) as client:
        assert client.record is not None
        assert client.record.incarnations[0] == client.incarnation


def test_late_wakeup_keeps_live_authority(monkeypatch):
    """Keep a client whose thread wakes after the silence timeout while the authority's lines wait."""
    silence_s = 1.0
    monkeypatch.setattr(liveness, "AUTHORITY_SILENCE_TIMEOUT_S", silence_s)
    skipped_s = [0.0]
    monkeypatch.setattr(
        liveness, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + skipped_s[0])
    )
    calls = []

    def select_with_late_wakeup(watched, *rest):
        calls.append(watched)
        if len(calls) > 1:
            return select.select(watched, *rest)
        # The authority's line has arrived, but the thread wakes only after the silence timeout.
        select.select(watched[:1], [], [])
        skipped_s[0] += 2 * silence_s
        return [], [], []

    monkeypatch.setattr(liveness, "select", SimpleNamespace(select=select_with_late_wakeup))
    errors = []
    with _scripted_client(errors, resend_s=0.1):
        _until(lambda: errors or len(calls) > 2)
        assert not errors


def test_callback_error_logged_with_traceback(monkeypatch):
    """Report an error of a client's callback with its traceback, not as a lost authority."""
    logged, errors = [], []
    monkeypatch.setattr(liveness.logger, "error", logged.append)

    def on_tick():
        raise AttributeError("a bug in the callback")

    with _deployment(2) as (_, clients):
        clients[1].bind(lambda record: None, on_tick=on_tick, on_error=errors.append)
        _until(lambda: errors)
    assert isinstance(errors[0], AttributeError)
    (message,) = [text for text in logged if "callback" in text]
    assert message.startswith("The tick callback of the membership client of rank 1 failed")
    assert "Traceback" in message


def test_client_thread_error_closes_connection(monkeypatch):
    """Report any error that ends a client's thread, and close its liveness connection."""
    logged, errors = [], []
    monkeypatch.setattr(liveness.logger, "error", logged.append)
    with _scripted_client(errors) as client:
        monkeypatch.setattr(client._lines, "encode", Mock(side_effect=KeyError("a bug")))
        client.report(progress=1)
        _until(lambda: errors)
        assert isinstance(errors[0], KeyError)
        assert client._sock.fileno() == -1
        (message,) = logged
        assert message.startswith("Membership client of rank 0 failed")
        assert "Traceback" in message


def test_transaction_summary_spans_propose_to_commit():
    cluster = Cluster()
    cluster.run(1.0)
    assert cluster.authority.last_transaction is None
    cluster.alive.discard(2)
    cluster.run_until(MembershipState.INSTALL)
    cluster.alive.discard(3)  # restarts the transaction, which keeps its start
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    began = cluster.published_at[0]
    ended = cluster.published_at[-1]
    summary = cluster.authority.last_transaction
    assert (summary.generation, summary.outcome) == (record.generation, "COMMITTED")
    assert (summary.removed_ranks, summary.added_ranks) == ((2, 3), ())
    assert summary.causes == {2: "disconnected", 3: "disconnected"}
    assert summary.ended_unix_ms - summary.began_unix_ms == pytest.approx((ended - began) * 1e3)
    assert ended - began > 0
    # A closed connection is proposed in the tick that sees it, and the simulated ranks install in
    # one step.
    assert summary.detected_unix_ms == summary.began_unix_ms
    assert summary.ended_unix_ms - summary.ready_unix_ms == pytest.approx(50.0)


def _interleave(cluster, rng):
    """Apply a random crash, hang, join, readiness or install outcome, and delay random reports."""
    record, r = cluster.authority.record, rng.randrange(1, cluster.authority.world_size)
    event = rng.randrange(5)
    if event == 0:
        cluster.alive.discard(r)
    elif event == 1 and r not in cluster.alive:
        _join(cluster, r)
        cluster.unprepared.add(r)
        cluster.hung.discard(r)
        cluster.install_reply[r] = Reply.INSTALLED
    elif event == 2:
        cluster.unprepared.discard(r)
    elif event == 3 and rng.random() < 0.1:
        cluster.hung.add(r)
    elif event == 4 and (record.is_joining(r) or rng.random() < 0.01):
        cluster.install_reply[r] = Reply.FAILED
    cluster.stopped = {q for q in cluster.alive if rng.random() < 0.2}


@pytest.mark.parametrize("seed", range(10))
def test_random_interleavings_keep_invariants(seed):
    """Check the protocol's invariants after every tick of random failures, joins and late replies."""
    rng, world = random.Random(seed), 4 + seed % 5
    cluster = Cluster(world, hang_timeout_s=1.0)
    expected = {MembershipState.PROPOSE: Reply.READY, MembershipState.INSTALL: Reply.INSTALLED}
    record = cluster.authority.record
    committed, joining = record.active_mask, 0
    for step in range(600):
        if step < 400:
            _interleave(cluster, rng)
        else:
            cluster.stopped, cluster.unprepared = set(), set()
        previous, new = record, cluster.step()
        record = cluster.authority.record
        assert record.generation >= previous.generation
        if record.state == MembershipState.PROPOSE and new is not None:
            assert record.generation > previous.generation
        if record.state == MembershipState.ABORTED:
            # Only the loss of rank 0, or a surviving rank that fails its install, requires an abort.
            survivors = [
                r for r in range(world) if previous.is_active(r) and not previous.is_joining(r)
            ]
            assert not record.is_active(0) or any(
                cluster.install_reply[r] == Reply.FAILED for r in survivors
            ), record.reason
            return
        assert record.active_mask & ~sum(1 << r for r in cluster.alive) == 0
        active = [r for r in range(world) if record.is_active(r)]
        replied = [
            cluster.reports[r].reply_generation == record.generation
            and cluster.reports[r].reply == expected.get(previous.state)
            for r in active
        ]
        if record.state in expected:
            joining = record.joining_mask
            assert record.active_mask & ~committed & ~joining == 0
            assert new is not None or not all(replied)
        elif new is not None:
            assert previous.state == MembershipState.INSTALL
            assert all(replied)
            assert record.active_mask & ~committed & ~joining == 0
            committed = record.active_mask
    assert record.state == MembershipState.COMMITTED
