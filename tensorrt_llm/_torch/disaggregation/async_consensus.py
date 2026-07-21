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

"""Nonblocking agreement for Python disaggregated KV-transfer state.

The protocol deliberately separates two state machines:

* terminal votes are immutable and produce one authoritative outcome after
  every participant is locally quiescent; and
* generation-first readiness is withdrawable until a rank grants its prepare
  lease; the authoritative PP schedule then activates the same request on
  every rank before acknowledged completion retires the in-flight epoch.

The MPI transport uses a duplicated communicator and fixed-width packets.  It
never introduces a collective after construction.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import numpy as np

from tensorrt_llm._utils import mpi_comm

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


PROTOCOL_VERSION = 3
_PACKET_FIELDS = 7
_DEFAULT_MAX_MESSAGES_PER_POLL = 256
_DEFAULT_MAX_COMPLETED_EPOCHS = 65_536
_DEFAULT_MAX_PENDING_SENDS = 65_536
_DEFAULT_MAX_OPEN_ROUNDS = 65_536
_DEFAULT_ROUND_TIMEOUT_S = 600.0
_DEFAULT_MAX_SEND_TESTS_PER_PROGRESS = 256


class _ConsensusBackpressure(RuntimeError):
    """Transient capacity rejection before any MPI send is issued."""


class ConsensusPhase(IntEnum):
    READY = 1
    TERMINAL = 2


class ConsensusOutcome(IntEnum):
    READY = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4
    WITHDRAWN = 5


class ConsensusEventKind(IntEnum):
    READY_PREPARE = 1
    READY_RELEASE = 2
    READY_COMPLETE = 3
    READY_ABORT = 4
    TERMINAL_COMMIT = 5
    READY_ABORT_FINALIZE = 6


class _MessageKind(IntEnum):
    VOTE = 1
    WITHDRAW = 2
    READY_PREPARE = 3
    READY_ACK = 4
    READY_RELEASE = 5
    READY_ACTIVATE_ACK = 6
    READY_ABORT = 7
    TERMINAL_COMMIT = 8
    CLOSE = 9
    CLOSE_ACK = 10
    READY_ABORT_ACK = 11
    READY_ABORT_FINALIZE = 12
    FAIL_STOP = 13
    READY_COMPLETE = 14


class _CoordinatorAction(IntEnum):
    VOTES_COMPLETE = 1
    READY_ACKS_COMPLETE = 2
    READY_ABORT_ACKS_COMPLETE = 3
    READY_ABORT_START = 4
    FAIL_STOP = 5
    READY_ACTIVATE_ACKS_COMPLETE = 6


@dataclass(frozen=True)
class ConsensusEvent:
    kind: ConsensusEventKind
    request_id: int
    epoch: int
    outcome: ConsensusOutcome


@dataclass(frozen=True)
class _Packet:
    kind: _MessageKind
    phase: ConsensusPhase
    request_id: int
    epoch: int
    outcome: ConsensusOutcome
    source: int

    def encode(self) -> np.ndarray:
        return np.asarray(
            [
                PROTOCOL_VERSION,
                int(self.kind),
                int(self.phase),
                self.request_id,
                self.epoch,
                int(self.outcome),
                self.source,
            ],
            dtype=np.uint64,
        )

    @classmethod
    def decode(cls, fields: np.ndarray) -> "_Packet":
        if fields.shape != (_PACKET_FIELDS,):
            raise RuntimeError(f"invalid consensus packet shape: {fields.shape}")
        if int(fields[0]) != PROTOCOL_VERSION:
            raise RuntimeError(
                f"unsupported consensus protocol version {int(fields[0])}; "
                f"expected {PROTOCOL_VERSION}"
            )
        try:
            return cls(
                kind=_MessageKind(int(fields[1])),
                phase=ConsensusPhase(int(fields[2])),
                request_id=int(fields[3]),
                epoch=int(fields[4]),
                outcome=ConsensusOutcome(int(fields[5])),
                source=int(fields[6]),
            )
        except ValueError as error:
            raise RuntimeError(f"invalid consensus packet fields: {fields.tolist()}") from error


class ConsensusTransport(Protocol):
    rank: int
    participants: tuple[int, ...]

    def send(self, packet: _Packet, destination: int) -> None: ...

    def send_many(self, messages: Sequence[tuple[_Packet, int]]) -> None:
        """Atomically accept a fan-out or reject it before sending.

        `_ConsensusBackpressure` is the only recoverable rejection. Once any
        packet is accepted, an implementation must either accept the complete
        batch or terminate the participant group; returning an error after a
        strict prefix would make an authoritative decision unsafe to retry.
        """
        ...

    def progress(self) -> None: ...

    def receive(self, limit: int) -> list[_Packet]: ...

    @property
    def pending_send_count(self) -> int: ...

    def close(self, timeout_s: float) -> None: ...


@dataclass
class _PendingSend:
    buffer: np.ndarray
    request: object


@dataclass
class _RoundProgress:
    started_at: float
    last_progress_at: float


class MpiConsensusTransport:
    """Fixed-buffer point-to-point transport on a dedicated communicator."""

    def __init__(
        self,
        participants: Sequence[int],
        *,
        max_pending_sends: int = _DEFAULT_MAX_PENDING_SENDS,
        max_send_tests_per_progress: int = _DEFAULT_MAX_SEND_TESTS_PER_PROGRESS,
    ):
        if MPI is None:
            raise RuntimeError("mpi4py is required for asynchronous Python consensus")
        if max_pending_sends <= 0:
            raise ValueError("max_pending_sends must be positive")
        if max_send_tests_per_progress <= 0:
            raise ValueError("max_send_tests_per_progress must be positive")
        participants_tuple = tuple(int(rank) for rank in participants)
        minimum_fanout = max(1, len(participants_tuple) - 1)
        if max_pending_sends < minimum_fanout:
            raise ValueError(
                "max_pending_sends must accommodate one participant fan-out: "
                f"minimum={minimum_fanout}, configured={max_pending_sends}"
            )
        world = mpi_comm()
        world_size = world.Get_size()
        # The initial qualified topology is TP1/CP1/PP>1, so the PP domain is
        # the full worker world.  Rejecting subsets avoids communicator-rank
        # translation mistakes until attention-DP lanes are qualified.
        if participants_tuple != tuple(range(world_size)):
            raise RuntimeError(
                "asynchronous Python consensus currently requires the participant "
                "domain to equal the MPI worker world"
            )
        self.participants = participants_tuple
        comm = world.Dup()
        # A recoverable exception after Isend accepts a strict fan-out prefix
        # cannot be repaired by another protocol message: an early recipient
        # may already expose the decision. Capacity and packet construction are
        # therefore checked before the first Isend, and all subsequent MPI
        # failures terminate the communicator group instead of returning to
        # Python with an ambiguous partial-send state.
        try:
            comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)
            rank = int(comm.Get_rank())
        except Exception:
            # Dup() creates an owned communicator before any transport member
            # can be published. Roll it back transactionally if the remaining
            # MPI setup fails.
            try:
                comm.Free()
            except Exception:
                pass
            raise
        self._comm = comm
        self.rank = rank
        self._pending: deque[_PendingSend] = deque()
        self._max_pending_sends = max_pending_sends
        self._max_send_tests_per_progress = max_send_tests_per_progress
        self._receive_sources = tuple(rank for rank in self.participants if rank != self.rank)
        self._receive_cursor = 0
        self._closed = False

    def send(self, packet: _Packet, destination: int) -> None:
        self.send_many(((packet, destination),))

    def send_many(self, messages: Sequence[tuple[_Packet, int]]) -> None:
        if self._closed:
            raise RuntimeError("cannot send after consensus transport shutdown")
        messages_tuple = tuple(messages)
        for _, destination in messages_tuple:
            if destination not in self.participants:
                raise RuntimeError(f"consensus destination {destination} is not a participant")
        self.progress()
        projected_pending = len(self._pending) + len(messages_tuple)
        if projected_pending > self._max_pending_sends:
            kinds = sorted({packet.kind.name for packet, _ in messages_tuple})
            raise _ConsensusBackpressure(
                "asynchronous consensus send backpressure limit exceeded: "
                f"rank={self.rank}, batch={len(messages_tuple)}, kinds={kinds}, "
                f"pending={len(self._pending)}, projected={projected_pending}, "
                f"limit={self._max_pending_sends}"
            )
        buffers = tuple(packet.encode() for packet, _ in messages_tuple)
        for buffer, (_, destination) in zip(buffers, messages_tuple):
            request = self._comm.Isend(buffer, dest=destination, tag=0)
            self._pending.append(_PendingSend(buffer=buffer, request=request))

    def progress(self) -> None:
        # Bound every hot-path progress call. Incomplete requests rotate to the
        # tail so a permanently slow send cannot starve later completions.
        to_test = min(len(self._pending), self._max_send_tests_per_progress)
        for _ in range(to_test):
            pending = self._pending.popleft()
            if not pending.request.Test():
                self._pending.append(pending)

    def receive(self, limit: int) -> list[_Packet]:
        packets: list[_Packet] = []
        if limit <= 0:
            return packets
        if not self._receive_sources:
            return packets
        # Probe each source at most once per sweep, then rotate the first source
        # for the next sweep. A continuously busy low rank therefore cannot
        # starve a higher rank when ``limit`` is smaller than the world size.
        while len(packets) < limit:
            made_progress = False
            source_count = len(self._receive_sources)
            start = self._receive_cursor
            for offset in range(source_count):
                if len(packets) >= limit:
                    break
                source_index = (start + offset) % source_count
                source = self._receive_sources[source_index]
                if not self._comm.Iprobe(source=source, tag=0):
                    continue
                fields = np.empty(_PACKET_FIELDS, dtype=np.uint64)
                self._comm.Recv(fields, source=source, tag=0)
                packet = _Packet.decode(fields)
                if packet.source != source:
                    raise RuntimeError(
                        f"consensus packet source mismatch: payload={packet.source}, mpi={source}"
                    )
                packets.append(packet)
                made_progress = True
                self._receive_cursor = (source_index + 1) % source_count
            if not made_progress:
                self._receive_cursor = (start + 1) % source_count
                break
        return packets

    @property
    def pending_send_count(self) -> int:
        return len(self._pending)

    def close(self, timeout_s: float) -> None:
        if self._closed:
            return
        if timeout_s <= 0:
            raise ValueError("transport close timeout must be positive")
        deadline = time.monotonic() + timeout_s
        while self._pending:
            self.progress()
            if self._pending and time.monotonic() >= deadline:
                raise RuntimeError(
                    f"timed out draining {len(self._pending)} asynchronous consensus sends"
                )
            if self._pending:
                time.sleep(0.001)
        self._comm.Free()
        self._closed = True


_Key = tuple[ConsensusPhase, int, int]
_IntentKey = tuple[_MessageKind, _Key]
_UNKNOWN_FAILURE_KEY: _Key = (ConsensusPhase.TERMINAL, 0, 0)


@dataclass(frozen=True)
class _LocalIntent:
    packet: _Packet
    destination: int


_NORMAL_LOCAL_INTENT_KINDS = (
    _MessageKind.VOTE,
    _MessageKind.WITHDRAW,
    _MessageKind.READY_ACK,
    _MessageKind.READY_ABORT_ACK,
    _MessageKind.READY_ACTIVATE_ACK,
)
_COORDINATOR_MESSAGE_KINDS = {
    _MessageKind.READY_PREPARE,
    _MessageKind.READY_RELEASE,
    _MessageKind.READY_COMPLETE,
    _MessageKind.READY_ABORT,
    _MessageKind.READY_ABORT_FINALIZE,
    _MessageKind.TERMINAL_COMMIT,
    _MessageKind.CLOSE_ACK,
}
_READY_OUTCOME_MESSAGE_KINDS = {
    _MessageKind.READY_PREPARE,
    _MessageKind.READY_ACK,
    _MessageKind.READY_RELEASE,
    _MessageKind.READY_ACTIVATE_ACK,
    _MessageKind.READY_COMPLETE,
}
_WITHDRAWN_OUTCOME_MESSAGE_KINDS = {
    _MessageKind.WITHDRAW,
    _MessageKind.READY_ABORT,
    _MessageKind.READY_ABORT_ACK,
    _MessageKind.READY_ABORT_FINALIZE,
    _MessageKind.CLOSE,
    _MessageKind.CLOSE_ACK,
}
_PRIORITY_LOCAL_INTENT_LIMIT = 2


class AsyncConsensusCoordinator:
    """Asynchronous authoritative agreement over one PP participant domain."""

    def __init__(
        self,
        transport: ConsensusTransport,
        *,
        scheduling_rank: int | None = None,
        max_messages_per_poll: int = _DEFAULT_MAX_MESSAGES_PER_POLL,
        max_completed_epochs: int = _DEFAULT_MAX_COMPLETED_EPOCHS,
        max_open_rounds: int = _DEFAULT_MAX_OPEN_ROUNDS,
        round_timeout_s: float = _DEFAULT_ROUND_TIMEOUT_S,
        ready_lease_timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if len(transport.participants) < 2:
            raise ValueError("asynchronous consensus requires at least two participants")
        if transport.rank not in transport.participants:
            raise ValueError("local rank is not in the consensus participant domain")
        if max_messages_per_poll <= 0:
            raise ValueError("max_messages_per_poll must be positive")
        if max_completed_epochs <= 0:
            raise ValueError("max_completed_epochs must be positive")
        if max_open_rounds <= 0:
            raise ValueError("max_open_rounds must be positive")
        if round_timeout_s <= 0:
            raise ValueError("round_timeout_s must be positive")
        if ready_lease_timeout_s is not None and ready_lease_timeout_s <= 0:
            raise ValueError("ready_lease_timeout_s must be positive when set")
        self._transport = transport
        self._participants = transport.participants
        self._participant_set = set(self._participants)
        self._coordinator_rank = self._participants[-1]
        self._scheduling_rank = (
            self._participants[0] if scheduling_rank is None else int(scheduling_rank)
        )
        if self._scheduling_rank not in self._participant_set:
            raise ValueError("scheduling rank is not in the consensus participant domain")
        self._max_messages_per_poll = max_messages_per_poll
        self._max_completed_epochs = max_completed_epochs
        self._max_open_rounds = max_open_rounds
        self._round_timeout_s = round_timeout_s
        # A fully prepared READY round can legitimately remain silent while
        # rank zero waits for scheduler/KV capacity.  That interval is an
        # application lease, not missing protocol progress, so it must not
        # inherit the short vote/ack watchdog.  By default the executor's own
        # request and hang policies bound it; deployments may opt into a
        # distinct lease deadline without weakening the protocol watchdog.
        self._ready_lease_timeout_s = ready_lease_timeout_s
        self._clock = clock

        self._local_votes: dict[_Key, ConsensusOutcome] = {}
        self._votes: dict[_Key, dict[int, ConsensusOutcome]] = {}
        self._ready_required_acks: dict[_Key, set[int]] = {}
        self._ready_acks: dict[_Key, set[int]] = {}
        self._ready_activation_required_acks: dict[_Key, set[int]] = {}
        self._ready_activation_acks: dict[_Key, set[int]] = {}
        self._ready_abort_required_acks: dict[_Key, set[int]] = {}
        self._ready_abort_acks: dict[_Key, set[int]] = {}
        self._ready_abort_requested: set[_Key] = set()
        self._local_ready_prepared: set[_Key] = set()
        self._local_ready_acknowledged: set[_Key] = set()
        self._local_ready_released: set[_Key] = set()
        self._local_ready_activated: set[_Key] = set()
        self._local_ready_aborting: set[_Key] = set()
        self._local_ready_abort_acknowledged: set[_Key] = set()
        self._completed_epoch: OrderedDict[tuple[ConsensusPhase, int], int] = OrderedDict()
        self._events: deque[ConsensusEvent] = deque()
        self._round_progress: dict[_Key, _RoundProgress] = {}
        self._round_deadlines: OrderedDict[_Key, float] = OrderedDict()
        self._ready_lease_deadlines: OrderedDict[_Key, float] = OrderedDict()
        self._rounds_by_request: dict[tuple[ConsensusPhase, int], set[_Key]] = {}
        self._open_ready_epoch: dict[int, _Key] = {}
        # Normal local intents are bounded by open rounds: protocol ordering
        # permits at most one unsent intent per round. FAIL_STOP and CLOSE use a
        # separate reserved queue so overload cannot prevent fail-closed
        # propagation or shutdown.
        self._local_outbox: OrderedDict[_IntentKey, _LocalIntent] = OrderedDict()
        self._priority_local_outbox: OrderedDict[_IntentKey, _LocalIntent] = OrderedDict()
        self._coordinator_actions: deque[tuple[_CoordinatorAction, _Key]] = deque()
        self._queued_coordinator_actions: set[tuple[_CoordinatorAction, _Key]] = set()
        self._fatal_key: _Key | None = None
        self._fatal_error: str | None = None
        self._fail_stop_propagated = False

        self._shutdown_started = False
        self._closed_peers: set[int] = set()
        self._close_sent = False
        self._close_acknowledged = False
        self._close_ack_sent = False

    @property
    def rank(self) -> int:
        return self._transport.rank

    @property
    def coordinator_rank(self) -> int:
        return self._coordinator_rank

    @property
    def scheduling_rank(self) -> int:
        """Rank that may transition the request after READY_RELEASE."""
        return self._scheduling_rank

    def diagnostic_snapshot(self) -> dict[str, object]:
        """Return a read-only, bounded view of the oldest open round."""
        now = self._clock()
        ready_rounds = sum(key[0] == ConsensusPhase.READY for key in self._round_progress)
        terminal_rounds = len(self._round_progress) - ready_rounds
        snapshot: dict[str, object] = {
            "rank": self.rank,
            "coordinator_rank": self._coordinator_rank,
            "is_coordinator": self.rank == self._coordinator_rank,
            "scheduling_rank": self._scheduling_rank,
            "open_ready_rounds": ready_rounds,
            "open_terminal_rounds": terminal_rounds,
            "local_outbox": len(self._local_outbox),
            "priority_outbox": len(self._priority_local_outbox),
            "coordinator_actions": len(self._coordinator_actions),
            "pending_sends": self._transport.pending_send_count,
            "pending_events": len(self._events),
        }
        if not self._round_progress:
            return snapshot

        oldest_key, progress = min(
            self._round_progress.items(),
            key=lambda item: item[1].started_at,
        )
        phase, request_id, epoch = oldest_key
        snapshot.update(
            {
                "oldest_phase": phase.name,
                "oldest_request_id": request_id,
                "oldest_epoch": epoch,
                "oldest_age_s": round(now - progress.started_at, 6),
                "oldest_idle_s": round(now - progress.last_progress_at, 6),
            }
        )
        if self.rank != self._coordinator_rank:
            snapshot.update(
                {
                    "missing_votes": None,
                    "missing_ready_acks": None,
                    "missing_activation_acks": None,
                }
            )
            return snapshot

        participants = self._participant_set
        votes = set(self._votes.get(oldest_key, {}))
        required_ready_acks = self._ready_required_acks.get(oldest_key, set())
        ready_acks = self._ready_acks.get(oldest_key, set())
        required_activation_acks = self._ready_activation_required_acks.get(oldest_key, set())
        activation_acks = self._ready_activation_acks.get(oldest_key, set())
        snapshot.update(
            {
                "missing_votes": sorted(participants - votes),
                "missing_ready_acks": sorted(required_ready_acks - ready_acks),
                "missing_activation_acks": sorted(required_activation_acks - activation_acks),
            }
        )
        return snapshot

    def publish_ready(self, request_id: int, epoch: int = 0) -> None:
        self._publish(
            ConsensusPhase.READY,
            request_id,
            epoch,
            ConsensusOutcome.READY,
        )

    def withdraw_ready(self, request_id: int, epoch: int = 0) -> bool:
        """Withdraw a readiness vote before this rank grants its lease.

        ``True`` means that this epoch is, or is becoming, aborted. ``False``
        means this rank already acknowledged READY_PREPARE (or observed the
        epoch's final decision), so the caller must let readiness finish and
        treat cancellation as part of the request's next lifecycle phase.
        """
        self._check_running()
        key = (ConsensusPhase.READY, int(request_id), int(epoch))
        current = self._local_votes.get(key)
        if current == ConsensusOutcome.WITHDRAWN:
            return True
        if key in self._local_ready_aborting:
            return True
        if key in self._local_ready_acknowledged:
            return False
        if self._is_stale(key):
            return False
        self._ensure_no_open_ready_epoch(key)
        self._reserve_local_round(key)
        packet = self._packet(
            _MessageKind.WITHDRAW,
            key,
            ConsensusOutcome.WITHDRAWN,
        )
        if self.rank == self._coordinator_rank:
            self._local_votes[key] = ConsensusOutcome.WITHDRAWN
            self._touch_round(key)
            self._request_ready_abort(key)
        else:
            vote_intent_key = (_MessageKind.VOTE, key)
            # A cancellation may arrive before an unsent readiness vote reaches
            # MPI. The coordinator accepts withdrawal without a preceding local
            # vote, so replace that intent rather than consuming two outbox
            # credits or later resurrecting the round.
            self._local_outbox.pop(vote_intent_key, None)
            self._enqueue_local_intent(packet, self._coordinator_rank)
            self._local_votes[key] = ConsensusOutcome.WITHDRAWN
            self._touch_round(key)
            self._drain_local_outbox(1)
        return True

    def publish_terminal(
        self,
        request_id: int,
        outcome: ConsensusOutcome,
        epoch: int = 0,
    ) -> None:
        if outcome not in (
            ConsensusOutcome.COMPLETED,
            ConsensusOutcome.FAILED,
            ConsensusOutcome.CANCELLED,
        ):
            raise ValueError(f"invalid terminal consensus outcome: {outcome}")
        self._publish(ConsensusPhase.TERMINAL, request_id, epoch, outcome)

    def acknowledge_ready(self, request_id: int, epoch: int = 0) -> None:
        """Grant an irrevocable local lease after applying READY_PREPARE.

        Once this method succeeds, :meth:`withdraw_ready` cannot abort on
        behalf of this rank. The request remains prepared until it is selected
        by the authoritative PP schedule or an abort is finalized.
        """
        self._check_running()
        key = (ConsensusPhase.READY, int(request_id), int(epoch))
        if self._local_votes.get(key) != ConsensusOutcome.READY:
            raise RuntimeError(f"cannot acknowledge unpublished readiness for {key}")
        if key not in self._local_ready_prepared:
            raise RuntimeError(f"cannot acknowledge readiness before READY_PREPARE for {key}")
        if key in self._local_ready_aborting:
            raise RuntimeError(f"cannot acknowledge readiness while aborting {key}")
        if key in self._local_ready_acknowledged:
            return
        packet = self._packet(_MessageKind.READY_ACK, key, ConsensusOutcome.READY)
        if self.rank == self._coordinator_rank:
            self._local_ready_acknowledged.add(key)
            self._touch_round(key)
            self._record_ready_ack(key, self.rank)
        else:
            self._enqueue_local_intent(packet, self._coordinator_rank)
            self._local_ready_acknowledged.add(key)
            self._touch_round(key)
            self._drain_local_outbox(1)

    def acknowledge_ready_abort(self, request_id: int, epoch: int = 0) -> None:
        """Acknowledge that the local READY_ABORT rollback has been applied."""
        self._check_running()
        key = (ConsensusPhase.READY, int(request_id), int(epoch))
        if key not in self._local_ready_aborting:
            raise RuntimeError(f"cannot acknowledge readiness abort before READY_ABORT for {key}")
        if key in self._local_ready_abort_acknowledged:
            return
        packet = self._packet(
            _MessageKind.READY_ABORT_ACK,
            key,
            ConsensusOutcome.WITHDRAWN,
        )
        if self.rank == self._coordinator_rank:
            self._local_ready_abort_acknowledged.add(key)
            self._touch_round(key)
            self._record_ready_abort_ack(key, self.rank)
        else:
            self._enqueue_local_intent(packet, self._coordinator_rank)
            self._local_ready_abort_acknowledged.add(key)
            self._touch_round(key)
            self._drain_local_outbox(1)

    def acknowledge_ready_activation(self, request_id: int, epoch: int = 0) -> None:
        """Acknowledge activation by the authoritative PP schedule.

        PREPARE keeps follower requests hidden.  Rank zero receives
        READY_RELEASE after every prepare ACK and can select the request.  The
        resulting PP schedule is the activation token: each rank calls this
        method only after the same scheduled request is locally visible.  The
        coordinator retains the epoch until every participant acknowledges.
        """
        self._check_running()
        key = (ConsensusPhase.READY, int(request_id), int(epoch))
        if key not in self._local_ready_acknowledged:
            raise RuntimeError(
                f"cannot activate readiness before PREPARE acknowledgement for {key}"
            )
        if self.rank == self._scheduling_rank and key not in self._local_ready_released:
            raise RuntimeError(
                f"scheduling rank activated readiness before READY_RELEASE for {key}"
            )
        if key in self._local_ready_activated:
            return
        packet = self._packet(_MessageKind.READY_ACTIVATE_ACK, key, ConsensusOutcome.READY)
        if self.rank == self._coordinator_rank:
            self._local_ready_activated.add(key)
            self._touch_round(key)
            self._record_ready_activation_ack(key, self.rank)
        else:
            self._enqueue_local_intent(packet, self._coordinator_rank)
            self._local_ready_activated.add(key)
            self._touch_round(key)
            self._drain_local_outbox(1)

    def poll(self) -> list[ConsensusEvent]:
        try:
            self._transport.progress()
        except RuntimeError as error:
            first_failure = self._fail_stop_from_runtime_error(
                _UNKNOWN_FAILURE_KEY,
                error,
            )
            if not first_failure:
                self._raise_if_fatal()
            raise
        self._drain_local_outbox(self._max_messages_per_poll)
        try:
            packets = self._transport.receive(self._max_messages_per_poll)
        except RuntimeError as error:
            first_failure = self._fail_stop_from_runtime_error(
                _UNKNOWN_FAILURE_KEY,
                error,
            )
            if not first_failure:
                self._raise_if_fatal()
            raise
        for packet in packets:
            try:
                self._handle_packet(packet)
            except RuntimeError as error:
                first_failure = self._fail_stop_from_runtime_error(
                    (packet.phase, packet.request_id, packet.epoch),
                    error,
                )
                if not first_failure:
                    self._raise_if_fatal()
                raise
        if self.rank == self._coordinator_rank:
            self._advance_coordinator(self._max_messages_per_poll)
        if not self._shutdown_started:
            self._check_round_watchdogs()
        self._raise_if_fatal()
        events = list(self._events)
        self._events.clear()
        return events

    def shutdown(self, timeout_s: float = 30.0) -> None:
        if timeout_s <= 0:
            raise ValueError("shutdown timeout must be positive")
        if self._close_acknowledged:
            self._transport.close(timeout_s)
            return
        self._shutdown_started = True
        self._local_outbox.clear()
        close_key = (ConsensusPhase.READY, 0, 0)
        if self.rank == self._coordinator_rank:
            self._closed_peers.add(self.rank)
        elif not self._close_sent:
            self._enqueue_local_intent(
                self._packet(_MessageKind.CLOSE, close_key, ConsensusOutcome.WITHDRAWN),
                self._coordinator_rank,
                priority=True,
            )
            self._close_sent = True
            self._drain_local_outbox(1)
        deadline = time.monotonic() + timeout_s
        while not self._close_acknowledged:
            self.poll()
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "timed out waiting for asynchronous consensus shutdown acknowledgement"
                )
            time.sleep(0.001)
        self._transport.close(max(0.001, deadline - time.monotonic()))

    def _publish(
        self,
        phase: ConsensusPhase,
        request_id: int,
        epoch: int,
        outcome: ConsensusOutcome,
    ) -> None:
        self._check_running()
        key = (phase, int(request_id), int(epoch))
        if self._is_stale(key):
            raise RuntimeError(f"cannot publish a stale consensus epoch: {key}")
        if phase == ConsensusPhase.READY:
            self._ensure_no_open_ready_epoch(key)
            if key in self._local_ready_aborting:
                raise RuntimeError(f"cannot publish readiness while aborting {key}")
        previous = self._local_votes.get(key)
        if previous is not None:
            if previous != outcome:
                raise RuntimeError(
                    f"local consensus vote changed for {key}: {previous.name} -> {outcome.name}"
                )
            return
        self._reserve_local_round(key)
        packet = self._packet(_MessageKind.VOTE, key, outcome)
        if self.rank == self._coordinator_rank:
            self._local_votes[key] = outcome
            self._touch_round(key)
            self._record_vote(key, self.rank, outcome)
        else:
            self._enqueue_local_intent(packet, self._coordinator_rank)
            self._local_votes[key] = outcome
            self._touch_round(key)
            self._drain_local_outbox(1)

    def _enqueue_local_intent(
        self,
        packet: _Packet,
        destination: int,
        *,
        priority: bool = False,
    ) -> None:
        intent_key = (packet.kind, (packet.phase, packet.request_id, packet.epoch))
        intent = _LocalIntent(packet=packet, destination=destination)
        outbox = self._priority_local_outbox if priority else self._local_outbox
        previous = outbox.get(intent_key)
        if previous is not None:
            if previous != intent:
                raise RuntimeError(f"local consensus intent changed for {intent_key}")
            return
        limit = _PRIORITY_LOCAL_INTENT_LIMIT if priority else self._max_open_rounds
        if len(outbox) >= limit:
            raise RuntimeError(
                "asynchronous consensus local-intent limit exceeded: "
                f"rank={self.rank}, priority={int(priority)}, kind={packet.kind.name}, "
                f"request_id={packet.request_id}, epoch={packet.epoch}, "
                f"queued={len(outbox)}, limit={limit}"
            )
        outbox[intent_key] = intent

    def _drain_local_outbox(self, limit: int) -> None:
        if limit <= 0:
            return
        sent = 0
        for outbox in (self._priority_local_outbox, self._local_outbox):
            while outbox and sent < limit:
                intent_key, intent = next(iter(outbox.items()))
                try:
                    self._transport.send(intent.packet, intent.destination)
                except _ConsensusBackpressure:
                    return
                except RuntimeError as error:
                    key = intent_key[1]
                    already_fatal = self._fatal_key is not None
                    self._enter_local_fail_stop(
                        key,
                        "asynchronous consensus transport failed while sending a durable "
                        f"local intent: rank={self.rank}, kind={intent.packet.kind.name}, "
                        f"phase={key[0].name}, request_id={key[1]}, epoch={key[2]}, "
                        f"error={error}",
                        notify_coordinator=True,
                    )
                    if already_fatal:
                        if self._shutdown_started:
                            return
                        if self._fatal_error is not None:
                            raise RuntimeError(self._fatal_error) from error
                    raise
                outbox.pop(intent_key)
                sent += 1

    def _enter_local_fail_stop(
        self,
        key: _Key,
        diagnostic: str,
        *,
        notify_coordinator: bool,
    ) -> None:
        if self._fatal_key is not None:
            return
        self._fatal_key = key
        self._fatal_error = diagnostic
        self._local_outbox.clear()
        self._events.clear()
        if notify_coordinator and self.rank != self._coordinator_rank:
            self._enqueue_local_intent(
                self._packet(_MessageKind.FAIL_STOP, key, ConsensusOutcome.FAILED),
                self._coordinator_rank,
                priority=True,
            )

    def _fail_stop_from_runtime_error(self, key: _Key, error: RuntimeError) -> bool:
        """Record the first local protocol/transport error and notify peers.

        Returns ``True`` only when ``error`` established the local fatal state.
        A best-effort notification is attempted immediately, while its durable
        intent/action remains queued if transport progress is unavailable.
        """
        first_failure = self._fatal_key is None
        if first_failure:
            if self.rank == self._coordinator_rank:
                self._request_fail_stop(
                    key,
                    reported_by=self.rank,
                    diagnostic=str(error),
                )
            else:
                self._enter_local_fail_stop(
                    key,
                    str(error),
                    notify_coordinator=True,
                )
        try:
            if self.rank == self._coordinator_rank:
                self._advance_coordinator(1)
            else:
                self._drain_local_outbox(1)
        except RuntimeError:
            # The first diagnostic remains authoritative. FAIL_STOP is a
            # reserved durable obligation and a later poll/shutdown retries it.
            pass
        return first_failure

    def _handle_packet(self, packet: _Packet) -> None:
        if packet.source not in self._participant_set:
            raise RuntimeError(f"packet source {packet.source} is not a participant")
        self._validate_packet_contract(packet)
        key = (packet.phase, packet.request_id, packet.epoch)
        if packet.kind == _MessageKind.CLOSE:
            self._require_coordinator()
            self._closed_peers.add(packet.source)
            return
        if packet.kind == _MessageKind.CLOSE_ACK:
            if packet.source != self._coordinator_rank:
                raise RuntimeError("close acknowledgement did not come from coordinator")
            self._close_acknowledged = True
            return
        if packet.kind == _MessageKind.FAIL_STOP:
            self._require_outcome(packet, ConsensusOutcome.FAILED)
            if self.rank == self._coordinator_rank:
                self._request_fail_stop(key, reported_by=packet.source)
            elif packet.source == self._coordinator_rank:
                self._enter_local_fail_stop(
                    key,
                    (
                        "asynchronous consensus entered coordinated fail-stop: "
                        f"rank={self.rank}, coordinator={self._coordinator_rank}, "
                        f"phase={key[0].name}, request_id={key[1]}, epoch={key[2]}"
                    ),
                    notify_coordinator=False,
                )
            else:
                raise RuntimeError(
                    "consensus fail-stop message did not come from the coordinator: "
                    f"source={packet.source}, coordinator={self._coordinator_rank}"
                )
            return
        if self._fatal_key is not None:
            return
        if self._is_stale(key):
            return
        if packet.kind == _MessageKind.VOTE:
            self._require_coordinator()
            if key in self._ready_abort_requested:
                return
            self._record_vote(key, packet.source, packet.outcome)
        elif packet.kind == _MessageKind.WITHDRAW:
            self._require_coordinator()
            if packet.phase != ConsensusPhase.READY:
                raise RuntimeError("only READY consensus can be withdrawn")
            # MPI preserves per-source ordering. If this source's ACK was
            # already observed, its lease is irrevocable and a late or buggy
            # withdrawal cannot overturn the round.
            if packet.source in self._ready_acks.get(key, set()):
                return
            self._request_ready_abort(key)
        elif packet.kind == _MessageKind.READY_PREPARE:
            self._require_phase(packet, ConsensusPhase.READY)
            self._require_outcome(packet, ConsensusOutcome.READY)
            if key in self._local_ready_aborting:
                return
            local_vote = self._local_votes.get(key)
            if local_vote == ConsensusOutcome.WITHDRAWN:
                # PREPARE and a local cancellation can cross in flight.  The
                # already-published withdrawal makes the coordinator abort the
                # round; never acknowledge or resurrect it here.
                return
            if local_vote != ConsensusOutcome.READY:
                raise RuntimeError(f"received READY_PREPARE without a local READY vote: {key}")
            if key in self._local_ready_prepared:
                return
            self._touch_round(key)
            self._local_ready_prepared.add(key)
            self._events.append(
                ConsensusEvent(
                    ConsensusEventKind.READY_PREPARE,
                    packet.request_id,
                    packet.epoch,
                    ConsensusOutcome.READY,
                )
            )
        elif packet.kind == _MessageKind.READY_ACK:
            self._require_phase(packet, ConsensusPhase.READY)
            self._require_coordinator()
            self._record_ready_ack(key, packet.source)
        elif packet.kind == _MessageKind.READY_ACTIVATE_ACK:
            self._require_phase(packet, ConsensusPhase.READY)
            self._require_coordinator()
            self._record_ready_activation_ack(key, packet.source)
        elif packet.kind == _MessageKind.READY_RELEASE:
            self._require_phase(packet, ConsensusPhase.READY)
            self._require_outcome(packet, ConsensusOutcome.READY)
            if self.rank != self._scheduling_rank:
                raise RuntimeError("non-scheduling rank received READY_RELEASE")
            if key in self._local_ready_released:
                return
            self._touch_round(key)
            self._local_ready_released.add(key)
            self._events.append(
                ConsensusEvent(
                    ConsensusEventKind.READY_RELEASE,
                    packet.request_id,
                    packet.epoch,
                    ConsensusOutcome.READY,
                )
            )
        elif packet.kind == _MessageKind.READY_COMPLETE:
            self._require_phase(packet, ConsensusPhase.READY)
            self._require_outcome(packet, ConsensusOutcome.READY)
            self._complete_local(key)
            self._events.append(
                ConsensusEvent(
                    ConsensusEventKind.READY_COMPLETE,
                    packet.request_id,
                    packet.epoch,
                    ConsensusOutcome.READY,
                )
            )
        elif packet.kind == _MessageKind.READY_ABORT:
            self._require_phase(packet, ConsensusPhase.READY)
            self._require_outcome(packet, ConsensusOutcome.WITHDRAWN)
            self._apply_ready_abort(key)
        elif packet.kind == _MessageKind.READY_ABORT_ACK:
            self._require_phase(packet, ConsensusPhase.READY)
            self._require_outcome(packet, ConsensusOutcome.WITHDRAWN)
            self._require_coordinator()
            self._record_ready_abort_ack(key, packet.source)
        elif packet.kind == _MessageKind.READY_ABORT_FINALIZE:
            self._require_phase(packet, ConsensusPhase.READY)
            self._require_outcome(packet, ConsensusOutcome.WITHDRAWN)
            if key not in self._local_ready_aborting:
                raise RuntimeError(f"received READY_ABORT_FINALIZE before READY_ABORT for {key}")
            self._complete_local(key)
            self._events.append(
                ConsensusEvent(
                    ConsensusEventKind.READY_ABORT_FINALIZE,
                    packet.request_id,
                    packet.epoch,
                    ConsensusOutcome.WITHDRAWN,
                )
            )
        elif packet.kind == _MessageKind.TERMINAL_COMMIT:
            self._require_phase(packet, ConsensusPhase.TERMINAL)
            if packet.outcome not in (
                ConsensusOutcome.COMPLETED,
                ConsensusOutcome.FAILED,
                ConsensusOutcome.CANCELLED,
            ):
                raise RuntimeError(
                    f"message TERMINAL_COMMIT has invalid outcome {packet.outcome.name}"
                )
            if self._local_votes.get(key) not in (
                ConsensusOutcome.COMPLETED,
                ConsensusOutcome.FAILED,
                ConsensusOutcome.CANCELLED,
            ):
                raise RuntimeError(f"received terminal commit before local quiescence: {key}")
            self._complete_local(key)
            self._events.append(
                ConsensusEvent(
                    ConsensusEventKind.TERMINAL_COMMIT,
                    packet.request_id,
                    packet.epoch,
                    packet.outcome,
                )
            )
        else:
            raise RuntimeError(f"unhandled consensus message kind {packet.kind}")

    def _advance_coordinator(self, limit: int) -> None:
        for _ in range(limit):
            if not self._coordinator_actions:
                break
            action, key = self._coordinator_actions.popleft()
            action_key = (action, key)
            if action_key not in self._queued_coordinator_actions:
                continue
            if action != _CoordinatorAction.FAIL_STOP and self._is_stale(key):
                self._queued_coordinator_actions.discard(action_key)
                continue
            try:
                if action == _CoordinatorAction.VOTES_COMPLETE:
                    votes = self._votes.get(key)
                    if votes is not None and len(votes) == len(self._participants):
                        if key[0] == ConsensusPhase.TERMINAL:
                            outcome = self._reduce_terminal(votes.values())
                            self._broadcast_terminal_commit(key, outcome)
                            self._votes.pop(key, None)
                        elif key not in self._ready_required_acks:
                            self._prepare_ready(key)
                elif action == _CoordinatorAction.READY_ACKS_COMPLETE:
                    required = self._ready_required_acks.get(key)
                    if (
                        required is not None
                        and key not in self._ready_abort_requested
                        and self._ready_acks.get(key, set()) == required
                    ):
                        self._release_ready(key)
                elif action == _CoordinatorAction.READY_ACTIVATE_ACKS_COMPLETE:
                    required = self._ready_activation_required_acks.get(key)
                    if (
                        required is not None
                        and self._ready_activation_acks.get(key, set()) == required
                    ):
                        self._complete_ready(key)
                elif action == _CoordinatorAction.READY_ABORT_ACKS_COMPLETE:
                    required = self._ready_abort_required_acks.get(key)
                    if required is not None and self._ready_abort_acks.get(key, set()) == required:
                        self._finalize_ready_abort(key)
                elif action == _CoordinatorAction.READY_ABORT_START:
                    if key in self._ready_abort_requested:
                        self._start_ready_abort(key)
                elif action == _CoordinatorAction.FAIL_STOP:
                    if key == self._fatal_key and not self._fail_stop_propagated:
                        self._propagate_fail_stop(key)
                else:
                    raise RuntimeError(f"unhandled coordinator action {action}")
            except _ConsensusBackpressure:
                self._coordinator_actions.appendleft(action_key)
                return
            except RuntimeError as error:
                if action == _CoordinatorAction.FAIL_STOP:
                    self._coordinator_actions.appendleft(action_key)
                    if self._shutdown_started:
                        return
                    if self._fatal_error is not None:
                        raise RuntimeError(self._fatal_error) from error
                    raise
                # A decision fan-out can fail after the transport has accepted
                # a strict prefix. Never retry that original action: a duplicate
                # or second partial decision could let ranks diverge. Atomically
                # replace it with the reserved global FAIL_STOP obligation.
                self._queued_coordinator_actions.discard(action_key)
                self._fail_stop_from_runtime_error(key, error)
                raise
            self._queued_coordinator_actions.discard(action_key)

        if (
            self._shutdown_started
            and not self._close_ack_sent
            and self._closed_peers == self._participant_set
        ):
            close_key = (ConsensusPhase.READY, 0, 0)
            close_messages = tuple(
                (
                    self._packet(
                        _MessageKind.CLOSE_ACK,
                        close_key,
                        ConsensusOutcome.WITHDRAWN,
                    ),
                    rank,
                )
                for rank in self._participants
                if rank != self.rank
            )
            try:
                self._transport.send_many(close_messages)
            except _ConsensusBackpressure:
                return
            self._votes.clear()
            self._ready_required_acks.clear()
            self._ready_acks.clear()
            self._ready_activation_required_acks.clear()
            self._ready_activation_acks.clear()
            self._ready_abort_required_acks.clear()
            self._ready_abort_acks.clear()
            self._ready_abort_requested.clear()
            self._local_votes.clear()
            self._round_progress.clear()
            self._round_deadlines.clear()
            self._ready_lease_deadlines.clear()
            self._rounds_by_request.clear()
            self._open_ready_epoch.clear()
            self._local_outbox.clear()
            self._priority_local_outbox.clear()
            self._coordinator_actions.clear()
            self._queued_coordinator_actions.clear()
            self._close_ack_sent = True
            self._close_acknowledged = True

    def _prepare_ready(self, key: _Key) -> None:
        required = set(self._participants)
        messages = tuple(
            (
                self._packet(
                    _MessageKind.READY_PREPARE,
                    key,
                    ConsensusOutcome.READY,
                ),
                rank,
            )
            for rank in self._participants
            if rank != self.rank
        )
        self._transport.send_many(messages)
        self._ready_required_acks[key] = required
        self._ready_acks[key] = set()
        self._touch_round(key)
        for rank in self._participants:
            if rank == self.rank:
                self._local_ready_prepared.add(key)
                self._events.append(
                    ConsensusEvent(
                        ConsensusEventKind.READY_PREPARE,
                        key[1],
                        key[2],
                        ConsensusOutcome.READY,
                    )
                )

    def _record_vote(
        self,
        key: _Key,
        source: int,
        outcome: ConsensusOutcome,
    ) -> None:
        if source not in self._participant_set:
            raise RuntimeError(f"vote source {source} is not a participant")
        if key[0] == ConsensusPhase.READY and outcome != ConsensusOutcome.READY:
            raise RuntimeError(f"invalid readiness vote {outcome.name}")
        if key[0] == ConsensusPhase.TERMINAL and outcome not in (
            ConsensusOutcome.COMPLETED,
            ConsensusOutcome.FAILED,
            ConsensusOutcome.CANCELLED,
        ):
            raise RuntimeError(f"invalid terminal vote {outcome.name}")
        self._reserve_coordinator_round(key)
        votes = self._votes.setdefault(key, {})
        previous = votes.get(source)
        if previous is not None and previous != outcome:
            raise RuntimeError(
                f"participant {source} changed vote for {key}: {previous.name} -> {outcome.name}"
            )
        votes[source] = outcome
        self._touch_round(key)
        if len(votes) == len(self._participants):
            self._enqueue_coordinator_action(_CoordinatorAction.VOTES_COMPLETE, key)

    def _record_ready_ack(self, key: _Key, source: int) -> None:
        if key in self._ready_abort_requested:
            return
        required = self._ready_required_acks.get(key)
        if required is None or source not in required:
            raise RuntimeError(f"unexpected readiness acknowledgement from {source} for {key}")
        self._ready_acks[key].add(source)
        self._touch_round(key)
        if self._ready_acks[key] == required:
            self._enqueue_coordinator_action(_CoordinatorAction.READY_ACKS_COMPLETE, key)

    def _record_ready_activation_ack(self, key: _Key, source: int) -> None:
        required = self._ready_activation_required_acks.get(key)
        if required is None or source not in required:
            raise RuntimeError(
                f"unexpected readiness activation acknowledgement from {source} for {key}"
            )
        self._ready_activation_acks[key].add(source)
        self._touch_round(key)
        if self._ready_activation_acks[key] == required:
            self._enqueue_coordinator_action(
                _CoordinatorAction.READY_ACTIVATE_ACKS_COMPLETE,
                key,
            )

    def _record_ready_abort_ack(self, key: _Key, source: int) -> None:
        required = self._ready_abort_required_acks.get(key)
        if required is None or source not in required:
            raise RuntimeError(
                f"unexpected readiness abort acknowledgement from {source} for {key}"
            )
        self._ready_abort_acks[key].add(source)
        self._touch_round(key)
        if self._ready_abort_acks[key] == required:
            self._enqueue_coordinator_action(
                _CoordinatorAction.READY_ABORT_ACKS_COMPLETE,
                key,
            )

    def _enqueue_coordinator_action(self, action: _CoordinatorAction, key: _Key) -> None:
        action_key = (action, key)
        if action_key in self._queued_coordinator_actions:
            return
        self._queued_coordinator_actions.add(action_key)
        self._coordinator_actions.append(action_key)

    def _request_fail_stop(
        self,
        key: _Key,
        reported_by: int,
        diagnostic: str | None = None,
    ) -> None:
        if self.rank != self._coordinator_rank:
            raise RuntimeError("only the coordinator can propagate consensus fail-stop")
        if self._fatal_key is not None:
            return
        self._fatal_key = key
        self._fatal_error = diagnostic or (
            "asynchronous consensus entered coordinated fail-stop: "
            f"rank={self.rank}, coordinator={self._coordinator_rank}, "
            f"reported_by={reported_by}, phase={key[0].name}, "
            f"request_id={key[1]}, epoch={key[2]}"
        )
        self._local_outbox.clear()
        self._events.clear()
        self._coordinator_actions.clear()
        self._queued_coordinator_actions.clear()
        action_key = (_CoordinatorAction.FAIL_STOP, key)
        self._queued_coordinator_actions.add(action_key)
        self._coordinator_actions.appendleft(action_key)

    def _propagate_fail_stop(self, key: _Key) -> None:
        messages = tuple(
            (
                self._packet(_MessageKind.FAIL_STOP, key, ConsensusOutcome.FAILED),
                rank,
            )
            for rank in self._participants
            if rank != self.rank
        )
        self._transport.send_many(messages)
        self._fail_stop_propagated = True

    def _release_ready(self, key: _Key) -> None:
        if self.rank == self._scheduling_rank:
            raise RuntimeError("ready coordinator must differ from the scheduling rank")
        self._transport.send(
            self._packet(_MessageKind.READY_RELEASE, key, ConsensusOutcome.READY),
            self._scheduling_rank,
        )
        self._ready_activation_required_acks[key] = set(self._participants)
        self._ready_activation_acks[key] = set()
        self._touch_round(key)

    def _complete_ready(self, key: _Key) -> None:
        messages = tuple(
            (
                self._packet(_MessageKind.READY_COMPLETE, key, ConsensusOutcome.READY),
                rank,
            )
            for rank in self._participants
            if rank != self.rank
        )
        self._transport.send_many(messages)
        self._complete_local(key)
        self._events.append(
            ConsensusEvent(
                ConsensusEventKind.READY_COMPLETE,
                key[1],
                key[2],
                ConsensusOutcome.READY,
            )
        )
        self._votes.pop(key, None)
        self._ready_required_acks.pop(key, None)
        self._ready_acks.pop(key, None)
        self._ready_activation_required_acks.pop(key, None)
        self._ready_activation_acks.pop(key, None)

    def _request_ready_abort(self, key: _Key) -> None:
        if self.rank != self._coordinator_rank:
            raise RuntimeError("only the coordinator can abort readiness")
        if self._is_stale(key) or key in self._ready_abort_requested:
            return
        self._reserve_coordinator_round(key)
        self._ready_abort_requested.add(key)
        self._touch_round(key)
        self._queued_coordinator_actions.discard((_CoordinatorAction.VOTES_COMPLETE, key))
        self._queued_coordinator_actions.discard((_CoordinatorAction.READY_ACKS_COMPLETE, key))
        self._enqueue_coordinator_action(_CoordinatorAction.READY_ABORT_START, key)

    def _start_ready_abort(self, key: _Key) -> None:
        messages = tuple(
            (
                self._packet(
                    _MessageKind.READY_ABORT,
                    key,
                    ConsensusOutcome.WITHDRAWN,
                ),
                rank,
            )
            for rank in self._participants
            if rank != self.rank
        )
        self._transport.send_many(messages)
        self._ready_abort_required_acks[key] = set(self._participants)
        self._ready_abort_acks[key] = set()
        self._touch_round(key)
        self._votes.pop(key, None)
        self._ready_required_acks.pop(key, None)
        self._ready_acks.pop(key, None)
        self._ready_activation_required_acks.pop(key, None)
        self._ready_activation_acks.pop(key, None)
        self._apply_ready_abort(key)

    def _apply_ready_abort(self, key: _Key) -> None:
        if key in self._local_ready_aborting:
            return
        # READY_ABORT supersedes every normal outbound intent for the open
        # readiness round. In particular, a follower can receive an abort from
        # another rank's withdrawal while its own VOTE or READY_ACK is still
        # waiting for transport capacity. Discarding those obsolete intents
        # preserves the one-unsent-intent-per-round bound; the abort ACK below
        # becomes the round's sole remaining outbound obligation.
        for kind in (
            _MessageKind.VOTE,
            _MessageKind.WITHDRAW,
            _MessageKind.READY_ACK,
        ):
            self._local_outbox.pop((kind, key), None)
        self._local_ready_aborting.add(key)
        self._touch_round(key)
        self._events.append(
            ConsensusEvent(
                ConsensusEventKind.READY_ABORT,
                key[1],
                key[2],
                ConsensusOutcome.WITHDRAWN,
            )
        )

    def _finalize_ready_abort(self, key: _Key) -> None:
        if self.rank != self._coordinator_rank:
            raise RuntimeError("only the coordinator can finalize readiness abort")
        messages = tuple(
            (
                self._packet(
                    _MessageKind.READY_ABORT_FINALIZE,
                    key,
                    ConsensusOutcome.WITHDRAWN,
                ),
                rank,
            )
            for rank in self._participants
            if rank != self.rank
        )
        self._transport.send_many(messages)
        self._complete_local(key)
        self._events.append(
            ConsensusEvent(
                ConsensusEventKind.READY_ABORT_FINALIZE,
                key[1],
                key[2],
                ConsensusOutcome.WITHDRAWN,
            )
        )

    def _broadcast_terminal_commit(
        self,
        key: _Key,
        outcome: ConsensusOutcome,
    ) -> None:
        messages = tuple(
            (
                self._packet(_MessageKind.TERMINAL_COMMIT, key, outcome),
                rank,
            )
            for rank in self._participants
            if rank != self.rank
        )
        self._transport.send_many(messages)
        self._complete_local(key)
        self._events.append(
            ConsensusEvent(
                ConsensusEventKind.TERMINAL_COMMIT,
                key[1],
                key[2],
                outcome,
            )
        )

    @staticmethod
    def _reduce_terminal(outcomes: Iterable[ConsensusOutcome]) -> ConsensusOutcome:
        values = set(outcomes)
        if ConsensusOutcome.CANCELLED in values:
            return ConsensusOutcome.CANCELLED
        if ConsensusOutcome.FAILED in values:
            return ConsensusOutcome.FAILED
        if values == {ConsensusOutcome.COMPLETED}:
            return ConsensusOutcome.COMPLETED
        raise RuntimeError(f"invalid terminal vote set: {values}")

    def _complete_local(self, key: _Key) -> None:
        epoch_key = (key[0], key[1])
        self._completed_epoch[epoch_key] = max(key[2], self._completed_epoch.get(epoch_key, -1))
        self._completed_epoch.move_to_end(epoch_key)
        while len(self._completed_epoch) > self._max_completed_epochs:
            self._completed_epoch.popitem(last=False)
        # Completing epoch N tombstones every older epoch for this request and
        # phase.  Purge partial coordinator state too, otherwise an incomplete
        # old round can remain resident forever after a newer round commits.
        request_rounds = self._rounds_by_request.get(epoch_key, set())
        candidates = tuple(candidate for candidate in request_rounds if candidate[2] <= key[2])
        for candidate in candidates:
            for state in (
                self._local_votes,
                self._votes,
                self._ready_required_acks,
                self._ready_acks,
                self._ready_activation_required_acks,
                self._ready_activation_acks,
                self._ready_abort_required_acks,
                self._ready_abort_acks,
            ):
                state.pop(candidate, None)
            self._local_ready_prepared.discard(candidate)
            self._local_ready_acknowledged.discard(candidate)
            self._local_ready_released.discard(candidate)
            self._local_ready_activated.discard(candidate)
            self._local_ready_aborting.discard(candidate)
            self._local_ready_abort_acknowledged.discard(candidate)
            self._ready_abort_requested.discard(candidate)
            self._round_progress.pop(candidate, None)
            self._round_deadlines.pop(candidate, None)
            self._ready_lease_deadlines.pop(candidate, None)
            for action in _CoordinatorAction:
                self._queued_coordinator_actions.discard((action, candidate))
            for kind in _NORMAL_LOCAL_INTENT_KINDS:
                self._local_outbox.pop((kind, candidate), None)
            request_rounds.discard(candidate)
        if not request_rounds:
            self._rounds_by_request.pop(epoch_key, None)
        if key[0] == ConsensusPhase.READY:
            open_ready = self._open_ready_epoch.get(key[1])
            if open_ready is not None and open_ready[2] <= key[2]:
                self._open_ready_epoch.pop(key[1], None)

    def _reserve_local_round(self, key: _Key) -> None:
        self._reserve_round(key, "local publication")

    def _reserve_coordinator_round(self, key: _Key) -> None:
        self._reserve_round(key, "coordinator receive")

    def _reserve_round(self, key: _Key, operation: str) -> None:
        if key in self._round_progress:
            return
        if key[0] == ConsensusPhase.READY:
            open_key = self._open_ready_epoch.get(key[1])
            if open_key is not None and open_key != key:
                raise RuntimeError(
                    "cannot reuse a readiness request ID before its prior epoch finalizes: "
                    f"open={open_key}, new={key}"
                )
        if len(self._round_progress) >= self._max_open_rounds:
            oldest_key = next(iter(self._round_progress), None)
            raise RuntimeError(
                "asynchronous consensus open-round limit exceeded: "
                f"rank={self.rank}, operation={operation}, key={key}, "
                f"open={len(self._round_progress)}, limit={self._max_open_rounds}, "
                f"oldest={oldest_key}, pending_sends={self._transport.pending_send_count}"
            )

    def _touch_round(self, key: _Key) -> None:
        now = self._clock()
        progress = self._round_progress.get(key)
        if progress is None:
            self._reserve_round(key, "protocol progress")
            self._round_progress[key] = _RoundProgress(
                started_at=now,
                last_progress_at=now,
            )
            self._arm_round_watchdog(key, now)
            self._rounds_by_request.setdefault((key[0], key[1]), set()).add(key)
            if key[0] == ConsensusPhase.READY:
                self._open_ready_epoch[key[1]] = key
            return
        progress.last_progress_at = now
        # The watchdog measures lack of protocol progress, not total round
        # lifetime. Readiness may legitimately remain open while the scheduler
        # waits for capacity, and later votes/acknowledgements must grant a new
        # idle window. Moving the key preserves deadline order for the O(1)
        # oldest-round check below.
        self._arm_round_watchdog(key, now)

    def _is_silent_ready_lease(self, key: _Key) -> bool:
        if key[0] != ConsensusPhase.READY:
            return False
        if key in self._ready_abort_requested or key in self._local_ready_aborting:
            return False
        if self.rank == self._coordinator_rank:
            required = self._ready_activation_required_acks.get(key)
            return required is not None and not self._ready_activation_acks.get(key)
        return key in self._local_ready_acknowledged and key not in self._local_ready_activated

    def _arm_round_watchdog(self, key: _Key, now: float) -> None:
        self._round_deadlines.pop(key, None)
        self._ready_lease_deadlines.pop(key, None)
        if self._is_silent_ready_lease(key):
            if self._ready_lease_timeout_s is not None:
                self._ready_lease_deadlines[key] = now + self._ready_lease_timeout_s
            return
        self._round_deadlines[key] = now + self._round_timeout_s

    def _check_round_watchdogs(self) -> None:
        candidates: list[tuple[_Key, float]] = []
        if self._round_deadlines:
            candidates.append(next(iter(self._round_deadlines.items())))
        if self._ready_lease_deadlines:
            candidates.append(next(iter(self._ready_lease_deadlines.items())))
        if not candidates:
            return
        key, deadline = min(candidates, key=lambda candidate: candidate[1])
        now = self._clock()
        if now < deadline:
            return
        progress = self._round_progress[key]
        votes = self._votes.get(key, {})
        required_acks = self._ready_required_acks.get(key, set())
        ready_acks = self._ready_acks.get(key, set())
        required_activation_acks = self._ready_activation_required_acks.get(key, set())
        activation_acks = self._ready_activation_acks.get(key, set())
        required_abort_acks = self._ready_abort_required_acks.get(key, set())
        abort_acks = self._ready_abort_acks.get(key, set())
        missing_votes = sorted(self._participant_set - set(votes))
        missing_ready_acks = sorted(required_acks - ready_acks)
        missing_activation_acks = sorted(required_activation_acks - activation_acks)
        missing_abort_acks = sorted(required_abort_acks - abort_acks)
        local_vote = self._local_votes.get(key)
        diagnostic = (
            "asynchronous consensus round watchdog expired without a global decision: "
            f"rank={self.rank}, coordinator={self._coordinator_rank}, "
            f"phase={key[0].name}, request_id={key[1]}, epoch={key[2]}, "
            f"age_s={now - progress.started_at:.3f}, "
            f"idle_s={now - progress.last_progress_at:.3f}, "
            f"local_vote={local_vote.name if local_vote is not None else 'NONE'}, "
            f"missing_votes={missing_votes}, missing_ready_acks={missing_ready_acks}, "
            f"missing_activation_acks={missing_activation_acks}, "
            f"missing_abort_acks={missing_abort_acks}, "
            f"pending_sends={self._transport.pending_send_count}"
        )
        if self.rank == self._coordinator_rank:
            self._request_fail_stop(
                key,
                reported_by=self.rank,
                diagnostic=diagnostic,
            )
            self._advance_coordinator(1)
            return
        self._enter_local_fail_stop(
            key,
            diagnostic,
            notify_coordinator=True,
        )
        # Best effort only: local fail-stop must not depend on notification
        # capacity. The reserved intent remains queued for shutdown progress if
        # the transport is currently full.
        self._drain_local_outbox(1)

    def _raise_if_fatal(self) -> None:
        if self._fatal_error is not None and not self._shutdown_started:
            raise RuntimeError(self._fatal_error)

    def _is_stale(self, key: _Key) -> bool:
        return key[2] <= self._completed_epoch.get((key[0], key[1]), -1)

    def _ensure_no_open_ready_epoch(self, key: _Key) -> None:
        open_key = self._open_ready_epoch.get(key[1])
        if open_key is not None and open_key != key:
            raise RuntimeError(
                "cannot reuse a readiness request ID before its prior epoch finalizes: "
                f"open={open_key}, new={key}"
            )

    def _packet(
        self,
        kind: _MessageKind,
        key: _Key,
        outcome: ConsensusOutcome,
    ) -> _Packet:
        return _Packet(kind, key[0], key[1], key[2], outcome, self.rank)

    def _validate_packet_contract(self, packet: _Packet) -> None:
        if packet.kind in _COORDINATOR_MESSAGE_KINDS and packet.source != self._coordinator_rank:
            raise RuntimeError(
                f"message {packet.kind.name} did not come from coordinator "
                f"{self._coordinator_rank}: source={packet.source}"
            )

        if packet.kind in _READY_OUTCOME_MESSAGE_KINDS:
            self._require_phase(packet, ConsensusPhase.READY)
            self._require_outcome(packet, ConsensusOutcome.READY)
        elif packet.kind in _WITHDRAWN_OUTCOME_MESSAGE_KINDS:
            self._require_phase(packet, ConsensusPhase.READY)
            self._require_outcome(packet, ConsensusOutcome.WITHDRAWN)
        elif packet.kind == _MessageKind.TERMINAL_COMMIT:
            self._require_phase(packet, ConsensusPhase.TERMINAL)
            if packet.outcome not in (
                ConsensusOutcome.COMPLETED,
                ConsensusOutcome.FAILED,
                ConsensusOutcome.CANCELLED,
            ):
                raise RuntimeError(
                    f"message TERMINAL_COMMIT has invalid outcome {packet.outcome.name}"
                )
        elif packet.kind == _MessageKind.FAIL_STOP:
            self._require_outcome(packet, ConsensusOutcome.FAILED)
        elif packet.kind == _MessageKind.VOTE:
            if packet.phase == ConsensusPhase.READY:
                self._require_outcome(packet, ConsensusOutcome.READY)
            elif packet.outcome not in (
                ConsensusOutcome.COMPLETED,
                ConsensusOutcome.FAILED,
                ConsensusOutcome.CANCELLED,
            ):
                raise RuntimeError(
                    f"message VOTE has invalid terminal outcome {packet.outcome.name}"
                )

    def _require_coordinator(self) -> None:
        if self.rank != self._coordinator_rank:
            raise RuntimeError("only the coordinator may receive this consensus message")

    @staticmethod
    def _require_phase(packet: _Packet, phase: ConsensusPhase) -> None:
        if packet.phase != phase:
            raise RuntimeError(
                f"message {packet.kind.name} has phase {packet.phase.name}, expected {phase.name}"
            )

    @staticmethod
    def _require_outcome(packet: _Packet, outcome: ConsensusOutcome) -> None:
        if packet.outcome != outcome:
            raise RuntimeError(
                f"message {packet.kind.name} has outcome {packet.outcome.name}, "
                f"expected {outcome.name}"
            )

    def _check_running(self) -> None:
        if self._shutdown_started:
            raise RuntimeError("cannot publish consensus state after shutdown starts")
        if self._fatal_key is not None:
            raise RuntimeError("cannot publish consensus state after coordinated fail-stop starts")
