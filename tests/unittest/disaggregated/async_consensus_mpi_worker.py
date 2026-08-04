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

"""Four-rank CPU exercise for the asynchronous Python consensus protocol."""

from __future__ import annotations

import json
import sys
import time
import traceback

from mpi4py import MPI

from tensorrt_llm._torch.disaggregation.async_consensus import (
    AsyncConsensusCoordinator,
    ConsensusEvent,
    ConsensusEventKind,
    ConsensusOutcome,
    ConsensusPhase,
    MpiConsensusTransport,
    _MessageKind,
    _Packet,
)

_WORLD_SIZE = 4
_STEP_TIMEOUT_S = 10.0


def _wait_for_event(
    coordinator: AsyncConsensusCoordinator,
    expected: set[ConsensusEventKind],
    *,
    reject: set[ConsensusEventKind] | None = None,
) -> ConsensusEvent:
    deadline = time.monotonic() + _STEP_TIMEOUT_S
    observed: list[str] = []
    while time.monotonic() < deadline:
        for event in coordinator.poll():
            observed.append(event.kind.name)
            if reject is not None and event.kind in reject:
                raise AssertionError(
                    f"rank {coordinator.rank} observed rejected event {event}; history={observed}"
                )
            if event.kind in expected:
                return event
        time.sleep(0.001)
    raise AssertionError(
        f"rank {coordinator.rank} timed out waiting for "
        f"{sorted(kind.name for kind in expected)}; history={observed}"
    )


def _drain_pending(transport: MpiConsensusTransport) -> None:
    deadline = time.monotonic() + _STEP_TIMEOUT_S
    while transport.pending_send_count:
        transport.progress()
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"rank {transport.rank} could not drain {transport.pending_send_count} sends"
            )
        time.sleep(0.001)


def _run() -> None:
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    if world.Get_size() != _WORLD_SIZE:
        raise AssertionError(f"expected {_WORLD_SIZE} MPI ranks, got {world.Get_size()}")

    # A message on COMM_WORLD must remain untouched by the duplicated
    # consensus communicator even though both use tag zero.
    world_request = None
    if rank == 0:
        world_request = world.isend("world-communicator-sentinel", dest=3, tag=0)

    transport = MpiConsensusTransport(range(_WORLD_SIZE), max_pending_sends=256)
    coordinator = AsyncConsensusCoordinator(
        transport,
        max_messages_per_poll=8,
        max_open_rounds=64,
        round_timeout_s=5.0,
    )

    # Terminal votes arrive in a different order on each rank. Failure must be
    # reduced once, then delivered as the same authoritative outcome.
    time.sleep(0.005 * ((_WORLD_SIZE - rank) % _WORLD_SIZE))
    local_outcome = ConsensusOutcome.FAILED if rank == 1 else ConsensusOutcome.COMPLETED
    coordinator.publish_terminal(1001, local_outcome)
    terminal = _wait_for_event(
        coordinator,
        {ConsensusEventKind.TERMINAL_COMMIT},
    )
    if terminal.outcome != ConsensusOutcome.FAILED:
        raise AssertionError(f"rank {rank} observed terminal outcome {terminal.outcome}")
    world.Barrier()
    _drain_pending(transport)
    world.Barrier()

    # Force a coordinator fan-out below its required three credits. The first
    # attempt must emit nothing and retain a retryable action; restoring the
    # capacity must then commit exactly once on every rank.
    if rank == coordinator.coordinator_rank:
        transport._max_pending_sends = 2
    world.Barrier()
    coordinator.publish_terminal(1004, ConsensusOutcome.COMPLETED)
    if rank == coordinator.coordinator_rank:
        deadline = time.monotonic() + _STEP_TIMEOUT_S
        while not coordinator._coordinator_actions:
            if coordinator.poll():
                raise AssertionError("capacity-rejected fan-out emitted a local event")
            if time.monotonic() >= deadline:
                raise AssertionError("timed out reaching the capacity-rejected fan-out")
            time.sleep(0.001)
        if transport.pending_send_count:
            raise AssertionError("capacity-rejected fan-out issued a partial MPI send")
        transport._max_pending_sends = 256
    capacity_terminal = _wait_for_event(
        coordinator,
        {ConsensusEventKind.TERMINAL_COMMIT},
    )
    if capacity_terminal.outcome != ConsensusOutcome.COMPLETED:
        raise AssertionError(
            f"rank {rank} observed capacity retry outcome {capacity_terminal.outcome}"
        )
    world.Barrier()

    # Exercise PREPARE racing a pre-ACK withdrawal. No scheduling rank may be
    # released, and epoch reuse remains prohibited through rollback finalize.
    coordinator.publish_ready(1002, epoch=0)
    if rank == 2 and not coordinator.withdraw_ready(1002, epoch=0):
        raise AssertionError("pre-ACK readiness withdrawal was rejected")
    aborted = _wait_for_event(
        coordinator,
        {ConsensusEventKind.READY_ABORT},
        reject={ConsensusEventKind.READY_RELEASE, ConsensusEventKind.READY_COMPLETE},
    )
    if aborted.outcome != ConsensusOutcome.WITHDRAWN:
        raise AssertionError(f"rank {rank} observed abort outcome {aborted.outcome}")
    try:
        coordinator.publish_ready(1002, epoch=1)
    except RuntimeError as error:
        if "before its prior epoch finalizes" not in str(error):
            raise
    else:
        raise AssertionError("request ID was reused before readiness abort finalized")
    coordinator.acknowledge_ready_abort(1002, epoch=0)
    _wait_for_event(coordinator, {ConsensusEventKind.READY_ABORT_FINALIZE})
    world.Barrier()

    coordinator.publish_ready(1002, epoch=1)
    _wait_for_event(coordinator, {ConsensusEventKind.READY_PREPARE})
    coordinator.acknowledge_ready(1002, epoch=1)
    if rank == coordinator.scheduling_rank:
        released = _wait_for_event(coordinator, {ConsensusEventKind.READY_RELEASE})
        if released.epoch != 1:
            raise AssertionError(f"rank {rank} observed wrong ready epoch {released.epoch}")
    elif rank == coordinator.coordinator_rank:
        key = (ConsensusPhase.READY, 1002, 1)
        deadline = time.monotonic() + _STEP_TIMEOUT_S
        while key not in coordinator._ready_activation_required_acks:
            coordinator.poll()
            if time.monotonic() >= deadline:
                raise AssertionError("timed out releasing the scheduling rank")
            time.sleep(0.001)
    world.Barrier()
    # The barrier models delivery of rank zero's authoritative PP schedule.
    coordinator.acknowledge_ready_activation(1002, epoch=1)
    completed = _wait_for_event(coordinator, {ConsensusEventKind.READY_COMPLETE})
    if completed.epoch != 1:
        raise AssertionError(f"rank {rank} observed wrong ready epoch {completed.epoch}")
    world.Barrier()

    # A low-rank flood must not starve the two other senders when receiving one
    # packet per poll. These packets exercise only the transport and are
    # intentionally not passed to the coordinator state machine.
    if rank < 3:
        send_count = 8 if rank == 0 else 1
        for sequence in range(send_count):
            transport.send(
                _Packet(
                    _MessageKind.CLOSE,
                    ConsensusPhase.READY,
                    2000 + sequence,
                    0,
                    ConsensusOutcome.WITHDRAWN,
                    rank,
                ),
                3,
            )
    world.Barrier()
    if rank == 3:
        deadline = time.monotonic() + _STEP_TIMEOUT_S
        while not all(transport._comm.Iprobe(source=source, tag=0) for source in (0, 1, 2)):
            if time.monotonic() >= deadline:
                raise AssertionError("timed out waiting for all fairness senders")
            time.sleep(0.001)
        first_sources = [transport.receive(1)[0].source for _ in range(3)]
        if set(first_sources) != {0, 1, 2}:
            raise AssertionError(f"rotating receive fairness failed: {first_sources}")
        remaining = 7
        deadline = time.monotonic() + _STEP_TIMEOUT_S
        while remaining:
            packets = transport.receive(remaining)
            remaining -= len(packets)
            if time.monotonic() >= deadline:
                raise AssertionError(f"timed out draining {remaining} fairness packets")
            if not packets:
                time.sleep(0.001)
    world.Barrier()
    _drain_pending(transport)

    if rank == 3:
        sentinel = world.recv(source=0, tag=0)
        if sentinel != "world-communicator-sentinel":
            raise AssertionError(f"unexpected communicator sentinel: {sentinel}")
    if rank == 0:
        world_request.wait()
    world.Barrier()

    # A deliberately incomplete round must diagnose and stop; a watchdog must
    # never fabricate a terminal commit to make progress.
    coordinator._round_timeout_s = 0.1
    if rank == 0:
        coordinator.publish_terminal(1003, ConsensusOutcome.COMPLETED)
    watchdog_seen = False
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            events = coordinator.poll()
        except RuntimeError as error:
            if not any(
                marker in str(error)
                for marker in (
                    "watchdog expired without a global decision",
                    "coordinated fail-stop",
                )
            ):
                raise
            watchdog_seen = True
            break
        if any(event.kind == ConsensusEventKind.TERMINAL_COMMIT for event in events):
            raise AssertionError("watchdog path invented a terminal consensus outcome")
        time.sleep(0.001)
    if not watchdog_seen:
        raise AssertionError(
            f"rank {rank} watchdog state mismatch: observed={watchdog_seen}, "
            f"coordinator={coordinator.coordinator_rank}"
        )
    world.Barrier()

    # Rank zero times out once before the other ranks enter shutdown, then
    # retries the same close handshake. CLOSE must remain idempotent and the
    # staggered communicator free must finish on every rank.
    retried_shutdown = False
    if rank == 0:
        try:
            coordinator.shutdown(0.02)
        except RuntimeError as error:
            if "shutdown acknowledgement" not in str(error):
                raise
            retried_shutdown = True
        else:
            raise AssertionError("rank zero shutdown unexpectedly completed before peers")
        time.sleep(0.15)
    else:
        time.sleep(0.05 + 0.01 * rank)
    coordinator.shutdown(10.0)
    world.Barrier()

    retries = world.gather(retried_shutdown, root=0)
    if rank == 0:
        if retries != [True, False, False, False]:
            raise AssertionError(f"unexpected shutdown retry flags: {retries}")
        print(
            "ASYNC_CONSENSUS_MPI_OK "
            + json.dumps(
                {
                    "communicator_isolation": True,
                    "fanout_backpressure_retry": True,
                    "epoch_reuse": True,
                    "receive_fairness": True,
                    "shutdown_retry": True,
                    "terminal_outcome": terminal.outcome.name,
                    "watchdog_fail_closed": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    try:
        _run()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
        raise
