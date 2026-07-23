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

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.async_consensus as async_consensus_module
from tensorrt_llm._torch.disaggregation.async_consensus import (
    AsyncConsensusCoordinator,
    ConsensusEvent,
    ConsensusEventKind,
    ConsensusOutcome,
    ConsensusPhase,
    MpiConsensusTransport,
    _ConsensusBackpressure,
    _CoordinatorAction,
    _MessageKind,
    _Packet,
    _PendingSend,
)


class _FakeNetwork:
    def __init__(self, participants: Sequence[int]) -> None:
        self.participants = tuple(participants)
        self._queues: dict[int, deque[_Packet]] = {rank: deque() for rank in self.participants}
        self._lock = threading.Lock()

    def send(self, packet: _Packet, destination: int) -> None:
        with self._lock:
            self._queues[destination].append(packet)

    def receive(self, rank: int, limit: int) -> list[_Packet]:
        packets: list[_Packet] = []
        with self._lock:
            queue = self._queues[rank]
            while queue and len(packets) < limit:
                packets.append(queue.popleft())
        return packets

    def queued(self, rank: int) -> int:
        with self._lock:
            return len(self._queues[rank])


class _FakeTransport:
    def __init__(self, network: _FakeNetwork, rank: int) -> None:
        self.rank = rank
        self.participants = network.participants
        self._network = network
        self.receive_limits: list[int] = []
        self.progress_count = 0
        self.close_timeouts: list[float] = []

    def send(self, packet: _Packet, destination: int) -> None:
        self._network.send(packet, destination)

    def send_many(self, messages: Sequence[tuple[_Packet, int]]) -> None:
        for packet, destination in messages:
            self.send(packet, destination)

    def progress(self) -> None:
        self.progress_count += 1

    def receive(self, limit: int) -> list[_Packet]:
        self.receive_limits.append(limit)
        return self._network.receive(self.rank, limit)

    @property
    def pending_send_count(self) -> int:
        return 0

    def close(self, timeout_s: float) -> None:
        self.close_timeouts.append(timeout_s)


class _AtomicCapacityTransport(_FakeTransport):
    def __init__(self, network: _FakeNetwork, rank: int, capacity: int) -> None:
        super().__init__(network, rank)
        self.capacity = capacity

    def send_many(self, messages: Sequence[tuple[_Packet, int]]) -> None:
        if len(messages) > self.capacity:
            raise _ConsensusBackpressure(
                f"send backpressure limit exceeded: batch={len(messages)}, limit={self.capacity}"
            )
        super().send_many(messages)


class _FanoutFailureTransport(_FakeTransport):
    """Reject one matching fan-out before delivering any packet."""

    def __init__(
        self,
        network: _FakeNetwork,
        rank: int,
        failing_kind: _MessageKind,
        *,
        defer_fail_stop_once: bool = False,
    ) -> None:
        super().__init__(network, rank)
        self._failing_kind = failing_kind
        self._defer_fail_stop_once = defer_fail_stop_once
        self._fail_stop_deferred = False
        self.failed = False
        self.fanout_kinds: list[_MessageKind] = []

    def send_many(self, messages: Sequence[tuple[_Packet, int]]) -> None:
        messages_tuple = tuple(messages)
        if messages_tuple:
            self.fanout_kinds.append(messages_tuple[0][0].kind)
        if (
            messages_tuple
            and messages_tuple[0][0].kind == _MessageKind.FAIL_STOP
            and self._defer_fail_stop_once
            and not self._fail_stop_deferred
        ):
            self._fail_stop_deferred = True
            raise _ConsensusBackpressure("synthetic fail-stop backpressure")
        if not self.failed and messages_tuple and messages_tuple[0][0].kind == self._failing_kind:
            self.failed = True
            raise RuntimeError(f"synthetic {self._failing_kind.name} fan-out failure")
        super().send_many(messages_tuple)


class _ReceiveFailureTransport(_FakeTransport):
    def __init__(self, network: _FakeNetwork, rank: int, diagnostic: str) -> None:
        super().__init__(network, rank)
        self._diagnostic = diagnostic
        self._failed = False

    def receive(self, limit: int) -> list[_Packet]:
        if not self._failed:
            self._failed = True
            raise RuntimeError(self._diagnostic)
        return super().receive(limit)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _NoIterationDict(dict):
    def __iter__(self):
        raise AssertionError("hot-path state must not be globally scanned")


def _make_group(
    size: int = 4,
    *,
    max_messages_per_poll: int = 256,
    max_completed_epochs: int = 65_536,
    max_open_rounds: int = 65_536,
    round_timeout_s: float = 600.0,
    ready_lease_timeout_s: float | None = None,
    clock: _FakeClock | None = None,
) -> tuple[_FakeNetwork, list[_FakeTransport], list[AsyncConsensusCoordinator]]:
    network = _FakeNetwork(range(size))
    transports = [_FakeTransport(network, rank) for rank in range(size)]
    coordinators = [
        AsyncConsensusCoordinator(
            transport,
            max_messages_per_poll=max_messages_per_poll,
            max_completed_epochs=max_completed_epochs,
            max_open_rounds=max_open_rounds,
            round_timeout_s=round_timeout_s,
            ready_lease_timeout_s=ready_lease_timeout_s,
            clock=clock if clock is not None else time.monotonic,
        )
        for transport in transports
    ]
    return network, transports, coordinators


def _poll_rounds(
    coordinators: Sequence[AsyncConsensusCoordinator],
    rounds: int = 4,
) -> dict[int, list[ConsensusEvent]]:
    events: dict[int, list[ConsensusEvent]] = defaultdict(list)
    for _ in range(rounds):
        for coordinator in coordinators:
            events[coordinator.rank].extend(coordinator.poll())
    return events


def _terminal_events(events: Sequence[ConsensusEvent]) -> list[ConsensusEvent]:
    return [event for event in events if event.kind == ConsensusEventKind.TERMINAL_COMMIT]


def _release_scheduler_activate_all_and_complete(
    coordinators: Sequence[AsyncConsensusCoordinator],
    request_id: int,
    epoch: int = 0,
) -> dict[int, list[ConsensusEvent]]:
    """Use the authoritative schedule token, then complete every lease."""
    scheduling_rank = coordinators[0].scheduling_rank
    coordinator = coordinators[-1]
    events: dict[int, list[ConsensusEvent]] = defaultdict(list)

    assert coordinator.poll() == []
    scheduler = next(instance for instance in coordinators if instance.rank == scheduling_rank)
    release_events = scheduler.poll()
    events[scheduler.rank].extend(release_events)
    assert [event.kind for event in release_events] == [ConsensusEventKind.READY_RELEASE]

    # The test now simulates delivery of the exact rank-zero PP schedule.
    for instance in coordinators:
        instance.acknowledge_ready_activation(request_id, epoch)
    completed = _poll_rounds(coordinators)
    for rank, rank_events in completed.items():
        events[rank].extend(rank_events)
        assert [event.kind for event in rank_events] == [ConsensusEventKind.READY_COMPLETE]
    return events


def test_terminal_success_waits_for_every_staggered_vote() -> None:
    _, _, coordinators = _make_group()
    request_id = 101

    for rank in (2, 0, 3):
        coordinators[rank].publish_terminal(request_id, ConsensusOutcome.COMPLETED)
        events = _poll_rounds(coordinators)
        assert all(not _terminal_events(rank_events) for rank_events in events.values())

    coordinators[1].publish_terminal(request_id, ConsensusOutcome.COMPLETED)
    events = _poll_rounds(coordinators)

    for rank in range(4):
        assert _terminal_events(events[rank]) == [
            ConsensusEvent(
                ConsensusEventKind.TERMINAL_COMMIT,
                request_id,
                0,
                ConsensusOutcome.COMPLETED,
            )
        ]


@pytest.mark.parametrize(
    ("votes", "expected"),
    [
        (
            [
                ConsensusOutcome.COMPLETED,
                ConsensusOutcome.FAILED,
                ConsensusOutcome.COMPLETED,
                ConsensusOutcome.COMPLETED,
            ],
            ConsensusOutcome.FAILED,
        ),
        (
            [
                ConsensusOutcome.FAILED,
                ConsensusOutcome.CANCELLED,
                ConsensusOutcome.COMPLETED,
                ConsensusOutcome.FAILED,
            ],
            ConsensusOutcome.CANCELLED,
        ),
    ],
)
def test_terminal_outcome_precedence(
    votes: list[ConsensusOutcome], expected: ConsensusOutcome
) -> None:
    _, _, coordinators = _make_group()

    for coordinator, outcome in zip(coordinators, votes):
        coordinator.publish_terminal(102, outcome)
    events = _poll_rounds(coordinators)

    for rank_events in events.values():
        assert [event.outcome for event in _terminal_events(rank_events)] == [expected]


def test_duplicate_vote_is_idempotent_and_changed_vote_is_rejected() -> None:
    network, _, coordinators = _make_group()
    request_id = 103
    follower = coordinators[0]
    coordinator = coordinators[-1]

    follower.publish_terminal(request_id, ConsensusOutcome.COMPLETED)
    follower.publish_terminal(request_id, ConsensusOutcome.COMPLETED)
    assert network.queued(coordinator.rank) == 1
    with pytest.raises(RuntimeError, match="local consensus vote changed"):
        follower.publish_terminal(request_id, ConsensusOutcome.FAILED)

    duplicate = _Packet(
        _MessageKind.VOTE,
        ConsensusPhase.TERMINAL,
        request_id,
        0,
        ConsensusOutcome.COMPLETED,
        follower.rank,
    )
    network.send(duplicate, coordinator.rank)
    coordinator.poll()

    changed = _Packet(
        _MessageKind.VOTE,
        ConsensusPhase.TERMINAL,
        request_id,
        0,
        ConsensusOutcome.FAILED,
        follower.rank,
    )
    network.send(changed, coordinator.rank)
    with pytest.raises(RuntimeError, match="participant 0 changed vote"):
        coordinator.poll()


@pytest.mark.parametrize(
    ("kind", "phase", "outcome"),
    [
        (_MessageKind.READY_PREPARE, ConsensusPhase.READY, ConsensusOutcome.READY),
        (_MessageKind.READY_RELEASE, ConsensusPhase.READY, ConsensusOutcome.READY),
        (_MessageKind.READY_COMPLETE, ConsensusPhase.READY, ConsensusOutcome.READY),
        (_MessageKind.READY_ABORT, ConsensusPhase.READY, ConsensusOutcome.WITHDRAWN),
        (
            _MessageKind.READY_ABORT_FINALIZE,
            ConsensusPhase.READY,
            ConsensusOutcome.WITHDRAWN,
        ),
        (
            _MessageKind.TERMINAL_COMMIT,
            ConsensusPhase.TERMINAL,
            ConsensusOutcome.COMPLETED,
        ),
        (_MessageKind.CLOSE_ACK, ConsensusPhase.READY, ConsensusOutcome.WITHDRAWN),
    ],
)
def test_authoritative_messages_must_come_from_coordinator(
    kind: _MessageKind,
    phase: ConsensusPhase,
    outcome: ConsensusOutcome,
) -> None:
    network, _, coordinators = _make_group()
    follower = coordinators[0]
    network.send(_Packet(kind, phase, 1031, 0, outcome, 1), follower.rank)

    with pytest.raises(RuntimeError, match=rf"message {kind.name} did not come from coordinator"):
        follower.poll()
    assert not follower._events
    assert follower._fatal_key == (phase, 1031, 0)
    assert follower._fatal_error is not None
    assert network.queued(coordinators[-1].rank) == 1


def test_packet_contract_rejects_wrong_ack_outcome() -> None:
    network, _, coordinators = _make_group()
    coordinator = coordinators[-1]
    network.send(
        _Packet(
            _MessageKind.READY_ACK,
            ConsensusPhase.READY,
            1032,
            0,
            ConsensusOutcome.FAILED,
            0,
        ),
        coordinator.rank,
    )

    with pytest.raises(RuntimeError, match="message READY_ACK has outcome FAILED"):
        coordinator.poll()
    assert coordinator._fatal_key == (ConsensusPhase.READY, 1032, 0)
    assert coordinator._fatal_error == "message READY_ACK has outcome FAILED, expected READY"
    assert coordinator._fail_stop_propagated
    for follower in coordinators[:-1]:
        with pytest.raises(RuntimeError, match="coordinated fail-stop"):
            follower.poll()


def test_protocol_version_mismatch_is_fatal_not_retryable_backpressure() -> None:
    fields = _Packet(
        _MessageKind.VOTE,
        ConsensusPhase.TERMINAL,
        1033,
        0,
        ConsensusOutcome.COMPLETED,
        0,
    ).encode()
    fields[0] = np.uint64(0)

    with pytest.raises(RuntimeError, match="unsupported consensus protocol version") as error:
        _Packet.decode(fields)
    assert type(error.value) is RuntimeError


def test_receive_decode_failure_preserves_first_error_and_notifies_all_ranks() -> None:
    network = _FakeNetwork(range(4))
    diagnostic = "synthetic packet decode failure"
    transports: list[_FakeTransport] = [
        _ReceiveFailureTransport(network, 0, diagnostic),
        *[_FakeTransport(network, rank) for rank in range(1, 4)],
    ]
    coordinators = [AsyncConsensusCoordinator(transport) for transport in transports]
    follower = coordinators[0]

    with pytest.raises(RuntimeError, match=diagnostic):
        follower.poll()
    assert follower._fatal_error == diagnostic
    assert follower._fatal_key == (ConsensusPhase.TERMINAL, 0, 0)
    assert not follower._priority_local_outbox

    # The decode failure's reserved notification reaches the coordinator,
    # which fans out authoritative fail-stop to every participant.
    with pytest.raises(RuntimeError, match="coordinated fail-stop"):
        coordinators[-1].poll()
    for instance in coordinators[1:-1]:
        with pytest.raises(RuntimeError, match="coordinated fail-stop"):
            instance.poll()

    # Further polling must report the first local diagnostic, not overwrite it
    # with the coordinator's later generic fail-stop message.
    with pytest.raises(RuntimeError, match=diagnostic):
        follower.poll()
    assert follower._fatal_error == diagnostic


@pytest.mark.parametrize(
    "failing_kind",
    [_MessageKind.READY_PREPARE, _MessageKind.TERMINAL_COMMIT],
)
def test_decision_fanout_hard_failure_is_replaced_by_fail_stop(
    failing_kind: _MessageKind,
) -> None:
    network = _FakeNetwork(range(4))
    transports: list[_FakeTransport] = [_FakeTransport(network, rank) for rank in range(3)]
    coordinator_transport = _FanoutFailureTransport(
        network,
        3,
        failing_kind,
        defer_fail_stop_once=True,
    )
    transports.append(coordinator_transport)
    coordinators = [AsyncConsensusCoordinator(transport) for transport in transports]
    coordinator = coordinators[-1]

    for instance in coordinators:
        if failing_kind == _MessageKind.READY_PREPARE:
            instance.publish_ready(1034)
        else:
            instance.publish_terminal(1034, ConsensusOutcome.COMPLETED)

    diagnostic = f"synthetic {failing_kind.name} fan-out failure"
    with pytest.raises(RuntimeError, match=diagnostic):
        coordinator.poll()
    assert coordinator._fatal_error == diagnostic
    assert not coordinator._fail_stop_propagated
    assert coordinator_transport.fanout_kinds == [
        failing_kind,
        _MessageKind.FAIL_STOP,
    ]
    assert [action for action, _ in coordinator._coordinator_actions] == [
        _CoordinatorAction.FAIL_STOP
    ]
    assert not coordinator._events

    # A later poll retries only the reserved fail-stop action. The failed
    # decision can never be retried.
    with pytest.raises(RuntimeError, match=diagnostic):
        coordinator.poll()
    assert coordinator._fail_stop_propagated
    assert not coordinator._coordinator_actions
    assert coordinator_transport.fanout_kinds == [
        failing_kind,
        _MessageKind.FAIL_STOP,
        _MessageKind.FAIL_STOP,
    ]

    # No participant can observe a decision from the rejected atomic fan-out.
    expected_phase = (
        ConsensusPhase.READY
        if failing_kind == _MessageKind.READY_PREPARE
        else ConsensusPhase.TERMINAL
    )
    for follower in coordinators[:-1]:
        with pytest.raises(RuntimeError, match="coordinated fail-stop"):
            follower.poll()
        assert not follower._events
        assert follower._fatal_key == (expected_phase, 1034, 0)

    # Subsequent progress reports the original diagnostic and can never retry
    # the failed decision action.
    with pytest.raises(RuntimeError, match=diagnostic):
        coordinator.poll()
    assert coordinator_transport.fanout_kinds == [
        failing_kind,
        _MessageKind.FAIL_STOP,
        _MessageKind.FAIL_STOP,
    ]


def test_publication_and_withdrawal_absorb_send_backpressure() -> None:
    network, transports, coordinators = _make_group()
    follower = coordinators[0]
    transport = transports[0]
    original_send = transport.send
    fail_next = True

    def fail_once(packet: _Packet, destination: int) -> None:
        nonlocal fail_next
        if fail_next:
            fail_next = False
            raise _ConsensusBackpressure("synthetic send backpressure")
        original_send(packet, destination)

    transport.send = fail_once
    terminal_key = (ConsensusPhase.TERMINAL, 104, 0)
    follower.publish_terminal(104, ConsensusOutcome.COMPLETED)
    assert network.queued(coordinators[-1].rank) == 0
    assert follower._local_votes[terminal_key] == ConsensusOutcome.COMPLETED
    assert (_MessageKind.VOTE, terminal_key) in follower._local_outbox
    follower.poll()
    assert network.queued(coordinators[-1].rank) == 1
    assert follower._local_votes[terminal_key] == ConsensusOutcome.COMPLETED
    assert not follower._local_outbox

    follower.publish_ready(105)
    ready_key = (ConsensusPhase.READY, 105, 0)
    fail_next = True
    assert follower.withdraw_ready(105) is True
    assert follower._local_votes[ready_key] == ConsensusOutcome.WITHDRAWN
    assert (_MessageKind.WITHDRAW, ready_key) in follower._local_outbox
    follower.poll()
    assert not follower._local_outbox


def test_hard_send_failure_enters_local_fail_stop_and_queues_notification() -> None:
    _, transports, coordinators = _make_group()
    follower = coordinators[0]
    transport = transports[follower.rank]

    def fail_send(_packet: _Packet, _destination: int) -> None:
        raise RuntimeError("synthetic hard transport failure")

    transport.send = fail_send
    with pytest.raises(RuntimeError, match="synthetic hard transport failure"):
        follower.publish_terminal(1041, ConsensusOutcome.COMPLETED)

    key = (ConsensusPhase.TERMINAL, 1041, 0)
    assert follower._fatal_key == key
    assert not follower._local_outbox
    assert list(follower._priority_local_outbox) == [(_MessageKind.FAIL_STOP, key)]
    with pytest.raises(RuntimeError, match="after coordinated fail-stop starts"):
        follower.publish_terminal(1042, ConsensusOutcome.COMPLETED)


def test_unsent_readiness_vote_is_coalesced_into_withdrawal() -> None:
    network, transports, coordinators = _make_group()
    follower = coordinators[0]
    transport = transports[0]
    original_send = transport.send

    def reject_send(_packet: _Packet, _destination: int) -> None:
        raise _ConsensusBackpressure("synthetic send backpressure")

    transport.send = reject_send
    follower.publish_ready(1051)
    key = (ConsensusPhase.READY, 1051, 0)
    assert list(follower._local_outbox) == [(_MessageKind.VOTE, key)]

    assert follower.withdraw_ready(1051) is True
    assert list(follower._local_outbox) == [(_MessageKind.WITHDRAW, key)]

    transport.send = original_send
    follower.poll()
    queued = network.receive(coordinators[-1].rank, 2)
    assert [packet.kind for packet in queued] == [_MessageKind.WITHDRAW]


def test_ready_abort_supersedes_unsent_normal_intent_for_same_round() -> None:
    network, transports, coordinators = _make_group(max_open_rounds=1)
    follower = coordinators[0]
    transport = transports[follower.rank]

    def reject_send(_packet: _Packet, _destination: int) -> None:
        raise _ConsensusBackpressure("synthetic send backpressure")

    transport.send = reject_send
    follower.publish_ready(1052)
    key = (ConsensusPhase.READY, 1052, 0)
    assert list(follower._local_outbox) == [(_MessageKind.VOTE, key)]

    network.send(
        _Packet(
            _MessageKind.READY_ABORT,
            ConsensusPhase.READY,
            1052,
            0,
            ConsensusOutcome.WITHDRAWN,
            follower.coordinator_rank,
        ),
        follower.rank,
    )
    assert [event.kind for event in follower.poll()] == [ConsensusEventKind.READY_ABORT]
    assert not follower._local_outbox

    # The reserved round credit is sufficient for the only remaining outbound
    # obligation even though transport pressure persists.
    follower.acknowledge_ready_abort(1052)
    assert list(follower._local_outbox) == [(_MessageKind.READY_ABORT_ACK, key)]


def test_ready_ack_intents_survive_backpressure_after_prepare_event_is_consumed() -> None:
    _, transports, coordinators = _make_group()
    coordinator = coordinators[-1]
    follower = coordinators[1]

    for instance in coordinators:
        instance.publish_ready(1052)
    coordinator.poll()
    events_by_rank = {instance.rank: instance.poll() for instance in coordinators[:-1]}
    coordinator_events = [
        ConsensusEvent(
            ConsensusEventKind.READY_PREPARE,
            1052,
            0,
            ConsensusOutcome.READY,
        )
    ]
    assert all(events == coordinator_events for events in events_by_rank.values())

    original_send = transports[follower.rank].send
    reject_next_ack = True

    def reject_ack_once(packet: _Packet, destination: int) -> None:
        nonlocal reject_next_ack
        if reject_next_ack and packet.kind == _MessageKind.READY_ACK:
            reject_next_ack = False
            raise _ConsensusBackpressure("synthetic ACK backpressure")
        original_send(packet, destination)

    transports[follower.rank].send = reject_ack_once
    follower.acknowledge_ready(1052)
    key = (ConsensusPhase.READY, 1052, 0)
    assert key in follower._local_ready_acknowledged
    assert (_MessageKind.READY_ACK, key) in follower._local_outbox

    for instance in coordinators:
        if instance is not follower:
            instance.acknowledge_ready(1052)
    assert coordinator.poll() == []

    follower.poll()
    _release_scheduler_activate_all_and_complete(coordinators, 1052)


def test_ready_activation_ack_survives_backpressure_before_completion() -> None:
    _, transports, coordinators = _make_group()
    request_id = 1053
    coordinator = coordinators[-1]
    follower = coordinators[1]
    scheduling_rank = coordinators[0]

    for instance in coordinators:
        instance.publish_ready(request_id)
    coordinator.poll()
    for instance in coordinators[:-1]:
        instance.poll()
    for instance in coordinators:
        instance.acknowledge_ready(request_id)

    assert coordinator.poll() == []
    assert [event.kind for event in scheduling_rank.poll()] == [ConsensusEventKind.READY_RELEASE]

    original_send = transports[follower.rank].send
    reject_next_ack = True

    def reject_activation_ack_once(packet: _Packet, destination: int) -> None:
        nonlocal reject_next_ack
        if reject_next_ack and packet.kind == _MessageKind.READY_ACTIVATE_ACK:
            reject_next_ack = False
            raise _ConsensusBackpressure("synthetic activation-ACK backpressure")
        original_send(packet, destination)

    transports[follower.rank].send = reject_activation_ack_once
    follower.acknowledge_ready_activation(request_id)
    key = (ConsensusPhase.READY, request_id, 0)
    assert (_MessageKind.READY_ACTIVATE_ACK, key) in follower._local_outbox

    scheduling_rank.acknowledge_ready_activation(request_id)
    coordinators[2].acknowledge_ready_activation(request_id)
    coordinator.acknowledge_ready_activation(request_id)
    assert coordinator.poll() == []

    follower.poll()
    assert [event.kind for event in coordinator.poll()] == [ConsensusEventKind.READY_COMPLETE]
    assert [event.kind for event in scheduling_rank.poll()] == [ConsensusEventKind.READY_COMPLETE]


def test_terminal_fanout_reserves_capacity_before_any_rank_commits() -> None:
    network = _FakeNetwork(range(4))
    transports: list[_FakeTransport] = [_FakeTransport(network, rank) for rank in range(3)]
    coordinator_transport = _AtomicCapacityTransport(network, 3, capacity=2)
    transports.append(coordinator_transport)
    coordinators = [AsyncConsensusCoordinator(transport) for transport in transports]

    for instance in coordinators:
        instance.publish_terminal(106, ConsensusOutcome.COMPLETED)
    assert coordinators[-1].poll() == []

    assert all(network.queued(rank) == 0 for rank in range(3))
    assert not coordinators[-1]._events
    action = coordinators[-1]._coordinator_actions[0]
    assert action[1] == (ConsensusPhase.TERMINAL, 106, 0)

    coordinator_transport.capacity = 3
    events = {3: coordinators[-1].poll()}
    for instance in coordinators[:-1]:
        events[instance.rank] = instance.poll()
    assert all(len(_terminal_events(rank_events)) == 1 for rank_events in events.values())


def test_readiness_prepare_ack_release_orders_scheduler_last() -> None:
    _, _, coordinators = _make_group()
    request_id = 201
    scheduling_rank = coordinators[0]
    coordinator = coordinators[-1]
    assert all(instance.scheduling_rank == scheduling_rank.rank for instance in coordinators)

    for instance in coordinators:
        instance.publish_ready(request_id)

    coordinator_events = coordinator.poll()
    assert coordinator_events == [
        ConsensusEvent(
            ConsensusEventKind.READY_PREPARE,
            request_id,
            0,
            ConsensusOutcome.READY,
        )
    ]
    scheduling_events = scheduling_rank.poll()
    rank_one_events = coordinators[1].poll()
    rank_two_events = coordinators[2].poll()
    assert [event.kind for event in scheduling_events] == [ConsensusEventKind.READY_PREPARE]
    assert [event.kind for event in rank_one_events] == [ConsensusEventKind.READY_PREPARE]
    assert [event.kind for event in rank_two_events] == [ConsensusEventKind.READY_PREPARE]

    scheduling_rank.acknowledge_ready(request_id)
    coordinators[1].acknowledge_ready(request_id)
    coordinator.acknowledge_ready(request_id)
    assert coordinator.poll() == []

    coordinators[2].acknowledge_ready(request_id)
    events = _release_scheduler_activate_all_and_complete(coordinators, request_id)
    for rank in range(4):
        assert ConsensusEventKind.READY_COMPLETE in [event.kind for event in events[rank]]
    assert ConsensusEventKind.READY_RELEASE in [
        event.kind for event in events[scheduling_rank.rank]
    ]


def test_released_readiness_lease_does_not_use_protocol_idle_watchdog() -> None:
    clock = _FakeClock()
    _, _, coordinators = _make_group(round_timeout_s=5.0, clock=clock)
    request_id = 2021
    coordinator = coordinators[-1]
    scheduling_rank = coordinators[0]

    for instance in coordinators:
        instance.publish_ready(request_id)
    assert [event.kind for event in coordinator.poll()] == [ConsensusEventKind.READY_PREPARE]
    for instance in coordinators[:-1]:
        assert [event.kind for event in instance.poll()] == [ConsensusEventKind.READY_PREPARE]
    for instance in coordinators:
        instance.acknowledge_ready(request_id)
    assert coordinator.poll() == []
    assert [event.kind for event in scheduling_rank.poll()] == [ConsensusEventKind.READY_RELEASE]

    # Rank zero may wait arbitrarily longer than the vote/ack watchdog for KV
    # capacity. No consensus packet is expected during this healthy lease.
    clock.advance(50.0)
    assert all(instance.poll() == [] for instance in coordinators)

    for instance in coordinators:
        instance.acknowledge_ready_activation(request_id)
    events = _poll_rounds(coordinators)
    assert all(
        [event.kind for event in events[rank]] == [ConsensusEventKind.READY_COMPLETE]
        for rank in range(len(coordinators))
    )


def test_readiness_ack_before_prepare_is_rejected() -> None:
    _, _, coordinators = _make_group()
    follower = coordinators[1]

    follower.publish_ready(203)
    with pytest.raises(RuntimeError, match="before READY_PREPARE"):
        follower.acknowledge_ready(203)


def test_readiness_withdraw_aborts_epoch_and_tombstones_late_messages() -> None:
    network, _, coordinators = _make_group()
    request_id = 202
    coordinator = coordinators[-1]

    for instance in coordinators:
        instance.publish_ready(request_id)
    prepare_events = coordinator.poll()
    assert [event.kind for event in prepare_events] == [ConsensusEventKind.READY_PREPARE]

    coordinators[1].withdraw_ready(request_id)
    coordinator_events = coordinator.poll()
    assert [event.kind for event in coordinator_events] == [ConsensusEventKind.READY_ABORT]
    abort_events = {coordinator.rank: coordinator_events}
    for instance in coordinators[:-1]:
        abort_events[instance.rank] = instance.poll()
    assert [event.kind for event in abort_events[0]] == [
        ConsensusEventKind.READY_PREPARE,
        ConsensusEventKind.READY_ABORT,
    ]
    # READY_PREPARE was already in flight when rank 1 withdrew. The withdrawing
    # rank must suppress that stale preparation, while another follower may
    # legally observe PREPARE followed by ABORT. Neither path may release the
    # scheduling rank.
    assert [event.kind for event in abort_events[1]] == [ConsensusEventKind.READY_ABORT]
    assert [event.kind for event in abort_events[2]] == [
        ConsensusEventKind.READY_PREPARE,
        ConsensusEventKind.READY_ABORT,
    ]
    assert [event.kind for event in abort_events[coordinator.rank]] == [
        ConsensusEventKind.READY_ABORT
    ]

    # READY_ABORT is only the rollback command. No rank may reuse the request
    # ID until every rollback is applied and the coordinator finalizes it.
    for instance in coordinators:
        with pytest.raises(RuntimeError, match="before its prior epoch finalizes"):
            instance.publish_ready(request_id, epoch=1)
    for instance in (coordinators[0], coordinators[1], coordinator):
        instance.acknowledge_ready_abort(request_id)

    # Finalization is itself a consensus point: one unapplied rollback keeps
    # the request ID/epoch leased everywhere.
    assert coordinator.poll() == []
    coordinators[2].acknowledge_ready_abort(request_id)

    coordinator_events = coordinator.poll()
    assert [event.kind for event in coordinator_events] == [ConsensusEventKind.READY_ABORT_FINALIZE]
    finalize_events = {coordinator.rank: coordinator_events}
    for instance in coordinators[:-1]:
        finalize_events[instance.rank] = instance.poll()
    for rank_events in finalize_events.values():
        assert [event.kind for event in rank_events] == [ConsensusEventKind.READY_ABORT_FINALIZE]

    stale_vote = _Packet(
        _MessageKind.VOTE,
        ConsensusPhase.READY,
        request_id,
        0,
        ConsensusOutcome.READY,
        0,
    )
    network.send(stale_vote, coordinator.rank)
    assert coordinator.poll() == []
    with pytest.raises(RuntimeError, match="stale consensus epoch"):
        coordinators[0].publish_ready(request_id, epoch=0)

    for instance in coordinators:
        instance.publish_ready(request_id, epoch=1)
    assert [event.kind for event in coordinator.poll()] == [ConsensusEventKind.READY_PREPARE]


def test_readiness_ack_is_an_irrevocable_lease() -> None:
    network, _, coordinators = _make_group()
    request_id = 205
    coordinator = coordinators[-1]

    for instance in coordinators:
        instance.publish_ready(request_id)
    coordinator.poll()
    for instance in coordinators[:-1]:
        instance.poll()

    coordinators[1].acknowledge_ready(request_id)
    assert coordinators[1].withdraw_ready(request_id) is False
    # Even a malformed late withdrawal cannot overturn a lease whose ACK was
    # observed first from the same source.
    network.send(
        _Packet(
            _MessageKind.WITHDRAW,
            ConsensusPhase.READY,
            request_id,
            0,
            ConsensusOutcome.WITHDRAWN,
            1,
        ),
        coordinator.rank,
    )
    coordinators[0].acknowledge_ready(request_id)
    coordinators[2].acknowledge_ready(request_id)
    coordinator.acknowledge_ready(request_id)

    _release_scheduler_activate_all_and_complete(coordinators, request_id)


def test_withdraw_before_ack_aborts_even_after_another_rank_leases() -> None:
    _, _, coordinators = _make_group()
    request_id = 206
    coordinator = coordinators[-1]

    for instance in coordinators:
        instance.publish_ready(request_id)
    coordinator.poll()
    for instance in coordinators[:-1]:
        instance.poll()

    coordinators[1].acknowledge_ready(request_id)
    assert coordinators[2].withdraw_ready(request_id) is True
    coordinator_events = coordinator.poll()
    assert [event.kind for event in coordinator_events] == [ConsensusEventKind.READY_ABORT]

    events = {coordinator.rank: coordinator_events}
    for instance in coordinators[:-1]:
        events[instance.rank] = instance.poll()
    assert all(
        ConsensusEventKind.READY_RELEASE not in [event.kind for event in rank_events]
        for rank_events in events.values()
    )
    assert [event.kind for event in events[1]] == [ConsensusEventKind.READY_ABORT]
    assert coordinators[1].withdraw_ready(request_id) is True

    for instance in coordinators:
        instance.acknowledge_ready_abort(request_id)
    assert [event.kind for event in coordinator.poll()] == [ConsensusEventKind.READY_ABORT_FINALIZE]
    for instance in coordinators[:-1]:
        assert [event.kind for event in instance.poll()] == [
            ConsensusEventKind.READY_ABORT_FINALIZE
        ]


def test_ready_abort_ack_intent_survives_backpressure() -> None:
    _, transports, coordinators = _make_group()
    coordinator = coordinators[-1]
    follower = coordinators[1]
    request_id = 2061

    for instance in coordinators:
        instance.publish_ready(request_id)
    coordinator.poll()
    for instance in coordinators[:-1]:
        instance.poll()
    assert coordinators[2].withdraw_ready(request_id) is True
    assert [event.kind for event in coordinator.poll()] == [ConsensusEventKind.READY_ABORT]
    for instance in coordinators[:-1]:
        assert ConsensusEventKind.READY_ABORT in [event.kind for event in instance.poll()]

    original_send = transports[follower.rank].send
    reject_next_ack = True

    def reject_abort_ack_once(packet: _Packet, destination: int) -> None:
        nonlocal reject_next_ack
        if reject_next_ack and packet.kind == _MessageKind.READY_ABORT_ACK:
            reject_next_ack = False
            raise _ConsensusBackpressure("synthetic abort-ACK backpressure")
        original_send(packet, destination)

    transports[follower.rank].send = reject_abort_ack_once
    for instance in coordinators:
        instance.acknowledge_ready_abort(request_id)
    key = (ConsensusPhase.READY, request_id, 0)
    assert key in follower._local_ready_abort_acknowledged
    assert (_MessageKind.READY_ABORT_ACK, key) in follower._local_outbox
    assert coordinator.poll() == []

    follower.poll()
    assert [event.kind for event in coordinator.poll()] == [ConsensusEventKind.READY_ABORT_FINALIZE]
    for instance in coordinators[:-1]:
        assert [event.kind for event in instance.poll()] == [
            ConsensusEventKind.READY_ABORT_FINALIZE
        ]


def test_withdraw_without_local_vote_aborts_partial_readiness_round() -> None:
    _, _, coordinators = _make_group()
    request_id = 207
    coordinator = coordinators[-1]

    # One rank has observed peer metadata and voted. A second rank receives a
    # cancellation before it becomes locally ready, while the remaining ranks
    # have not entered the round at all.
    coordinators[0].publish_ready(request_id)
    assert coordinators[1].withdraw_ready(request_id) is True

    coordinator_events = coordinator.poll()
    assert [event.kind for event in coordinator_events] == [ConsensusEventKind.READY_ABORT]
    events = {coordinator.rank: coordinator_events}
    for instance in coordinators[:-1]:
        events[instance.rank] = instance.poll()
    assert all(
        [event.kind for event in rank_events] == [ConsensusEventKind.READY_ABORT]
        for rank_events in events.values()
    )

    # Every participant rolls back and acknowledges before any rank can reuse
    # the request ID at the next epoch.
    for instance in coordinators:
        with pytest.raises(RuntimeError, match="before its prior epoch finalizes"):
            instance.publish_ready(request_id, epoch=1)
        instance.acknowledge_ready_abort(request_id)
    assert [event.kind for event in coordinator.poll()] == [ConsensusEventKind.READY_ABORT_FINALIZE]
    for instance in coordinators[:-1]:
        assert [event.kind for event in instance.poll()] == [
            ConsensusEventKind.READY_ABORT_FINALIZE
        ]
    for instance in coordinators:
        instance.publish_ready(request_id, epoch=1)


def test_poll_processes_at_most_configured_messages() -> None:
    network, transports, coordinators = _make_group(max_messages_per_poll=2)
    coordinator = coordinators[-1]

    for request_id in range(3):
        coordinators[0].publish_terminal(request_id, ConsensusOutcome.COMPLETED)
    assert network.queued(coordinator.rank) == 3

    assert coordinator.poll() == []
    assert network.queued(coordinator.rank) == 1
    assert transports[coordinator.rank].receive_limits == [2]
    assert coordinator.poll() == []
    assert network.queued(coordinator.rank) == 0
    assert transports[coordinator.rank].receive_limits == [2, 2]


def test_newer_commit_purges_incomplete_older_epoch() -> None:
    _, _, coordinators = _make_group()
    coordinator = coordinators[-1]
    request_id = 204

    coordinators[0].publish_terminal(request_id, ConsensusOutcome.COMPLETED, epoch=0)
    coordinator.poll()

    for instance in coordinators:
        instance.publish_terminal(request_id, ConsensusOutcome.COMPLETED, epoch=1)
    _poll_rounds(coordinators)

    assert (ConsensusPhase.TERMINAL, request_id, 0) not in coordinator._votes


def test_completion_and_ready_epoch_checks_use_direct_indexes() -> None:
    _, _, coordinators = _make_group(max_open_rounds=512)
    follower = coordinators[0]
    for request_id in range(9000, 9256):
        follower.publish_terminal(request_id, ConsensusOutcome.COMPLETED)

    follower._local_votes = _NoIterationDict(follower._local_votes)
    completed_key = (ConsensusPhase.TERMINAL, 9000, 0)
    follower._complete_local(completed_key)
    assert completed_key not in follower._local_votes
    assert len(follower._round_progress) == 255

    follower.publish_ready(9300)
    follower.publish_ready(9301)
    with pytest.raises(RuntimeError, match="before its prior epoch finalizes"):
        follower.publish_ready(9300, epoch=1)


def test_completed_epoch_tombstones_have_a_bounded_lru_window() -> None:
    _, _, coordinators = _make_group(max_completed_epochs=2)

    for request_id in (401, 402, 403):
        for instance in coordinators:
            instance.publish_terminal(request_id, ConsensusOutcome.COMPLETED)
        _poll_rounds(coordinators)

    for instance in coordinators:
        assert len(instance._completed_epoch) == 2
        assert (ConsensusPhase.TERMINAL, 401) not in instance._completed_epoch
        assert list(instance._completed_epoch) == [
            (ConsensusPhase.TERMINAL, 402),
            (ConsensusPhase.TERMINAL, 403),
        ]


def test_open_round_limit_rejects_new_work_without_mutating_state() -> None:
    network, _, coordinators = _make_group(max_open_rounds=2)
    follower = coordinators[0]
    coordinator = coordinators[-1]

    follower.publish_terminal(501, ConsensusOutcome.COMPLETED)
    follower.publish_terminal(502, ConsensusOutcome.COMPLETED)
    with pytest.raises(RuntimeError, match="open-round limit exceeded"):
        follower.publish_terminal(503, ConsensusOutcome.COMPLETED)

    assert network.queued(coordinator.rank) == 2
    assert (ConsensusPhase.TERMINAL, 503, 0) not in follower._local_votes


def test_coordinator_open_round_limit_bounds_untrusted_remote_votes() -> None:
    network, _, coordinators = _make_group(max_open_rounds=2)
    coordinator = coordinators[-1]

    for request_id in (511, 512, 513):
        network.send(
            _Packet(
                _MessageKind.VOTE,
                ConsensusPhase.TERMINAL,
                request_id,
                0,
                ConsensusOutcome.COMPLETED,
                0,
            ),
            coordinator.rank,
        )

    with pytest.raises(RuntimeError, match="operation=coordinator receive"):
        coordinator.poll()
    assert len(coordinator._round_progress) == 2
    assert (ConsensusPhase.TERMINAL, 513, 0) not in coordinator._votes


def test_round_watchdog_reports_missing_consensus_and_never_commits() -> None:
    clock = _FakeClock()
    _, _, coordinators = _make_group(round_timeout_s=5.0, clock=clock)
    follower = coordinators[0]

    follower.publish_terminal(521, ConsensusOutcome.COMPLETED)
    assert follower.poll() == []
    clock.advance(5.1)

    with pytest.raises(
        RuntimeError,
        match=r"watchdog expired.*phase=TERMINAL.*request_id=521.*missing_votes=\[0, 1, 2, 3\]",
    ):
        follower.poll()
    assert follower._events == deque()


def test_round_watchdog_refreshes_on_progress_then_expires_when_idle() -> None:
    clock = _FakeClock()
    _, _, coordinators = _make_group(round_timeout_s=5.0, clock=clock)
    coordinator = coordinators[-1]

    coordinators[0].publish_terminal(525, ConsensusOutcome.COMPLETED)
    assert coordinator.poll() == []

    # A second distinct vote is healthy protocol progress. It arrives near the
    # original deadline and must grant the round a fresh idle window.
    clock.advance(4.0)
    coordinators[1].publish_terminal(525, ConsensusOutcome.COMPLETED)
    assert coordinator.poll() == []

    clock.advance(1.1)
    assert coordinator.poll() == []

    # No further vote or acknowledgement arrives, so the refreshed idle
    # window still fails closed once it expires.
    clock.advance(4.0)
    with pytest.raises(RuntimeError, match=r"watchdog expired.*request_id=525"):
        coordinator.poll()


def test_watchdog_propagates_coordinated_fail_stop_to_uninvolved_ranks() -> None:
    clock = _FakeClock()
    _, _, coordinators = _make_group(round_timeout_s=5.0, clock=clock)
    coordinators[0].publish_terminal(522, ConsensusOutcome.COMPLETED)
    clock.advance(5.1)

    with pytest.raises(RuntimeError, match="watchdog expired"):
        coordinators[0].poll()
    with pytest.raises(RuntimeError, match="coordinated fail-stop"):
        coordinators[-1].poll()
    for instance in coordinators[1:-1]:
        with pytest.raises(RuntimeError, match="coordinated fail-stop"):
            instance.poll()

    assert all(not instance._events for instance in coordinators)
    with pytest.raises(RuntimeError, match="shutdown acknowledgement"):
        coordinators[1].shutdown(0.001)
    with ThreadPoolExecutor(max_workers=len(coordinators)) as executor:
        futures = [executor.submit(instance.shutdown, 2.0) for instance in coordinators]
        for future in futures:
            future.result(timeout=3.0)


def test_coordinator_watchdog_fails_locally_when_fail_stop_fanout_is_backpressured() -> None:
    clock = _FakeClock()
    network = _FakeNetwork(range(4))
    transports: list[_FakeTransport] = [_FakeTransport(network, rank) for rank in range(3)]
    coordinator_transport = _AtomicCapacityTransport(network, 3, capacity=0)
    transports.append(coordinator_transport)
    coordinators = [
        AsyncConsensusCoordinator(transport, round_timeout_s=1.0, clock=clock)
        for transport in transports
    ]
    coordinator = coordinators[-1]

    for instance in coordinators:
        instance.publish_terminal(523, ConsensusOutcome.COMPLETED)
    assert coordinator.poll() == []
    clock.advance(1.1)

    with pytest.raises(RuntimeError, match=r"watchdog expired.*request_id=523"):
        coordinator.poll()
    assert coordinator._fatal_key == (ConsensusPhase.TERMINAL, 523, 0)
    assert coordinator._fatal_error is not None
    assert not coordinator._fail_stop_propagated
    assert coordinator._coordinator_actions[0][0] == _CoordinatorAction.FAIL_STOP
    assert all(network.queued(rank) == 0 for rank in range(3))


def test_follower_watchdog_fails_locally_and_reserves_notification_under_backpressure() -> None:
    clock = _FakeClock()
    _, transports, coordinators = _make_group(round_timeout_s=1.0, clock=clock)
    follower = coordinators[0]

    def reject_send(_packet: _Packet, _destination: int) -> None:
        raise _ConsensusBackpressure("permanent synthetic backpressure")

    transports[follower.rank].send = reject_send
    follower.publish_terminal(524, ConsensusOutcome.COMPLETED)
    clock.advance(1.1)

    with pytest.raises(RuntimeError, match=r"watchdog expired.*request_id=524"):
        follower.poll()
    key = (ConsensusPhase.TERMINAL, 524, 0)
    assert follower._fatal_key == key
    assert not follower._local_outbox
    assert list(follower._priority_local_outbox) == [(_MessageKind.FAIL_STOP, key)]


def test_ready_action_queue_does_not_scan_incomplete_rounds() -> None:
    _, _, coordinators = _make_group(max_messages_per_poll=256)
    coordinator = coordinators[-1]

    for request_id in range(100):
        coordinators[0].publish_terminal(request_id, ConsensusOutcome.COMPLETED)
    coordinator.poll()
    assert not coordinator._coordinator_actions

    request_id = 601
    for instance in coordinators:
        instance.publish_terminal(request_id, ConsensusOutcome.COMPLETED)
    events = coordinator.poll()
    assert _terminal_events(events) == [
        ConsensusEvent(
            ConsensusEventKind.TERMINAL_COMMIT,
            request_id,
            0,
            ConsensusOutcome.COMPLETED,
        )
    ]


class _ControllableRequest:
    def __init__(self) -> None:
        self.complete = False
        self.test_count = 0

    def Test(self) -> bool:
        self.test_count += 1
        return self.complete


class _FakeMpiComm:
    def __init__(self) -> None:
        self.requests: list[_ControllableRequest] = []
        self.sent_destinations: list[int] = []
        self.receive_queues: dict[int, deque[_Packet]] = defaultdict(deque)
        self.error_handler = None
        self.freed = False

    def Get_size(self) -> int:
        return 4

    def Dup(self):
        return self

    def Get_rank(self) -> int:
        return 3

    def Set_errhandler(self, error_handler) -> None:
        self.error_handler = error_handler

    def Isend(self, _buffer, dest: int, tag: int) -> _ControllableRequest:
        assert tag == 0
        request = _ControllableRequest()
        self.requests.append(request)
        self.sent_destinations.append(dest)
        return request

    def Iprobe(self, source: int, tag: int) -> bool:
        assert tag == 0
        return bool(self.receive_queues[source])

    def Recv(self, fields, source: int, tag: int) -> None:
        assert tag == 0
        fields[:] = self.receive_queues[source].popleft().encode()

    def Free(self) -> None:
        self.freed = True


def _make_unit_mpi_transport(
    comm: _FakeMpiComm,
    *,
    max_pending_sends: int = 4,
    max_send_tests_per_progress: int = 256,
) -> MpiConsensusTransport:
    transport = object.__new__(MpiConsensusTransport)
    transport.participants = (0, 1, 2, 3)
    transport._comm = comm
    transport.rank = 3
    transport._pending = deque()
    transport._max_pending_sends = max_pending_sends
    transport._max_send_tests_per_progress = max_send_tests_per_progress
    transport._receive_sources = (0, 1, 2)
    transport._receive_cursor = 0
    transport._closed = False
    return transport


def test_mpi_transport_makes_post_preflight_errors_process_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comm = _FakeMpiComm()
    fatal_handler = object()
    monkeypatch.setattr(
        async_consensus_module,
        "MPI",
        SimpleNamespace(ERRORS_ARE_FATAL=fatal_handler),
    )
    monkeypatch.setattr(async_consensus_module, "mpi_comm", lambda: comm)

    transport = MpiConsensusTransport(range(4))

    assert transport._comm is comm
    assert comm.error_handler is fatal_handler


def test_mpi_transport_frees_duplicated_communicator_when_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comm = _FakeMpiComm()
    comm.Set_errhandler = Mock(side_effect=RuntimeError("MPI setup failed"))
    monkeypatch.setattr(
        async_consensus_module,
        "MPI",
        SimpleNamespace(ERRORS_ARE_FATAL=object()),
    )
    monkeypatch.setattr(async_consensus_module, "mpi_comm", lambda: comm)

    with pytest.raises(RuntimeError, match="MPI setup failed"):
        MpiConsensusTransport(range(4))

    assert comm.freed


def test_mpi_transport_applies_send_backpressure_before_allocating() -> None:
    comm = _FakeMpiComm()
    transport = _make_unit_mpi_transport(comm, max_pending_sends=1)
    key = (ConsensusPhase.TERMINAL, 701, 0)
    packet = _Packet(
        _MessageKind.VOTE,
        key[0],
        key[1],
        key[2],
        ConsensusOutcome.COMPLETED,
        transport.rank,
    )

    transport.send(packet, 0)
    with pytest.raises(
        _ConsensusBackpressure,
        match=r"backpressure limit exceeded.*pending=1, projected=2, limit=1",
    ):
        transport.send(packet, 1)
    assert comm.sent_destinations == [0]
    assert transport.pending_send_count == 1

    comm.requests[0].complete = True
    transport.send(packet, 1)
    assert comm.sent_destinations == [0, 1]
    assert transport.pending_send_count == 1


def test_mpi_transport_progress_checks_a_bounded_rotating_window() -> None:
    comm = _FakeMpiComm()
    transport = _make_unit_mpi_transport(
        comm,
        max_pending_sends=16,
        max_send_tests_per_progress=3,
    )
    packet = _Packet(
        _MessageKind.VOTE,
        ConsensusPhase.TERMINAL,
        702,
        0,
        ConsensusOutcome.COMPLETED,
        transport.rank,
    )
    for destination in (0, 1, 2, 0, 1, 2):
        # Bypass send() so its own bounded progress does not affect the test
        # window being measured.
        buffer = packet.encode()
        transport._pending.append(
            _PendingSend(buffer=buffer, request=comm.Isend(buffer, dest=destination, tag=0))
        )

    transport.progress()
    assert [request.test_count for request in comm.requests] == [1, 1, 1, 0, 0, 0]
    transport.progress()
    assert [request.test_count for request in comm.requests] == [1, 1, 1, 1, 1, 1]
    transport.progress()
    assert [request.test_count for request in comm.requests] == [2, 2, 2, 1, 1, 1]


def test_mpi_transport_rotates_receive_sources_under_a_busy_sender() -> None:
    comm = _FakeMpiComm()
    transport = _make_unit_mpi_transport(comm)
    for source in (0, 0, 0, 1, 2):
        comm.receive_queues[source].append(
            _Packet(
                _MessageKind.VOTE,
                ConsensusPhase.TERMINAL,
                800 + source,
                0,
                ConsensusOutcome.COMPLETED,
                source,
            )
        )

    observed_sources = [transport.receive(1)[0].source for _ in range(3)]
    assert observed_sources == [0, 1, 2]


def test_shutdown_drains_all_ranks_and_rejects_future_publication() -> None:
    _, transports, coordinators = _make_group()

    coordinators[0].publish_terminal(301, ConsensusOutcome.COMPLETED)
    with ThreadPoolExecutor(max_workers=len(coordinators)) as executor:
        futures = [executor.submit(instance.shutdown, 2.0) for instance in coordinators]
        for future in futures:
            future.result(timeout=3.0)

    assert all(transport.close_timeouts for transport in transports)
    with pytest.raises(RuntimeError, match="after shutdown starts"):
        coordinators[0].publish_ready(302)


def test_shutdown_retry_does_not_duplicate_close_request() -> None:
    network, _, coordinators = _make_group()
    follower = coordinators[0]
    coordinator = coordinators[-1]

    with pytest.raises(RuntimeError, match="shutdown acknowledgement"):
        follower.shutdown(0.001)
    assert network.queued(coordinator.rank) == 1

    with ThreadPoolExecutor(max_workers=len(coordinators)) as executor:
        futures = [executor.submit(instance.shutdown, 2.0) for instance in coordinators]
        for future in futures:
            future.result(timeout=3.0)

    assert network.queued(coordinator.rank) == 0
