# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""1c.3 (stub): MPI FT subcomm broadcast thread.

Production PR 1c.3 sets up a dedicated MPI sub-communicator
(``MPI_Comm_split`` from ``COMM_WORLD``) with ``MPI_ERRORS_RETURN``,
spawns a CPU thread that uses non-blocking ``Isend``/``Irecv`` + ``Test`` for
FT signaling, opportunistically uses ULFM, and implements multi-failure
consensus.

This stub does the absolute minimum: **one** ``Isend``/``Irecv`` pair on a
dedicated thread per process, broadcasting "rank K just died" from any
discoverer to every other rank, and applying received notifications to the
local ``EPGroupHealth``. No subcomm split — uses ``COMM_WORLD`` directly with
a fixed tag. Single-failure only; multi-failure consensus deferred to
production PR 1c.6.

Stub contract — exercises these seams:

  * **Cross-rank propagation latency** — ``t_watchdog_fires`` on rank A vs.
    ``t_mark_failed_propagated`` on the slowest survivor.
  * **EPGroupHealth as the consensus point** — every rank's local view ends
    up the same after one full broadcast cycle, even if multiple discoverers
    raced to detect.
  * **Failure of MPI machinery on the dead rank** — the dead rank's ``Isend``
    never lands; survivors must not block forever waiting for it. Validates
    the audit-1a Day 2 finding that ``MPI_ERRORS_RETURN`` + main-thread
    polling is the right pattern.

Stub deliberately omits, vs. production:

  * ``MPI_Comm_split`` for the FT subcomm — uses ``COMM_WORLD``.
  * ``MPI_Errhandler_set(MPI_ERRORS_RETURN)`` on a dedicated subcomm.
  * ULFM ``MPI_Comm_revoke`` opportunistic use.
  * Multi-failure consensus protocol (PR 1c.6).
  * Iteration-barrier piggyback (PR 1c.5).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

try:
    from mpi4py import MPI
except ImportError as exc:  # pragma: no cover — pure environment check
    raise ImportError(
        "mpi4py is required for the MPI FT subcomm stub; install it before running the prototype"
    ) from exc

from tensorrt_llm._torch.modules.fused_moe.ep_group_health import EPGroupHealth

# A single sentinel tag for FT death notifications. Production code allocates
# a tag range owned by the FT subcomm; the stub's choice is arbitrary but
# deliberately picked outside the typical TRT-LLM tag range.
_FT_DEATH_NOTIFY_TAG = 0x7F71D0


@dataclass(frozen=True)
class MpiFtSubcommConfig:
    """Tunables. Production code reads these from LLMArgs (PR 1d.1)."""

    poll_interval_sec: float = 0.05
    """How often the broadcast thread checks ``Test`` on outstanding requests."""


class MpiFtSubcommStub:
    """Single-thread broadcaster of ``mark_failed`` notifications across ``COMM_WORLD``.

    On ``broadcast_failure(rank)`` (called from a discoverer's watchdog),
    posts a non-blocking ``Isend`` to every other rank with the failed rank's
    index. A background thread on every rank ``Irecv``s these messages and
    applies them locally via ``health.mark_failed``.

    Args:
        ep_size: Number of ranks (must equal ``MPI.COMM_WORLD.Get_size()``).
        local_rank: This process's rank.
        health: Shared :class:`EPGroupHealth`.
        config: Tunables.
        on_failure_received: Optional logging callback (timeline hook).
    """

    def __init__(
        self,
        ep_size: int,
        local_rank: int,
        health: EPGroupHealth,
        config: MpiFtSubcommConfig = MpiFtSubcommConfig(),
        on_failure_received: Optional[Callable[[int, int, float], None]] = None,
    ) -> None:
        comm_size = MPI.COMM_WORLD.Get_size()
        comm_rank = MPI.COMM_WORLD.Get_rank()
        if ep_size != comm_size:
            raise ValueError(f"ep_size ({ep_size}) must match MPI_COMM_WORLD size ({comm_size})")
        if local_rank != comm_rank:
            raise ValueError(
                f"local_rank ({local_rank}) must match MPI_COMM_WORLD rank ({comm_rank})"
            )
        self._ep_size = ep_size
        self._local_rank = local_rank
        self._health = health
        self._config = config
        self._on_failure_received = on_failure_received
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._comm = MPI.COMM_WORLD

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("mpi ft subcomm stub already started")
        self._thread = threading.Thread(
            target=self._recv_loop, name="mpi-ft-subcomm-stub", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def broadcast_failure(self, failed_rank: int) -> None:
        """Notify every other rank that ``failed_rank`` is dead.

        Posts non-blocking sends and returns immediately; cleanup of the
        request handles is best-effort (the prototype tolerates leaks because
        runs are short and the dead-peer's ``Isend`` will never complete
        anyway).
        """
        if not 0 <= failed_rank < self._ep_size:
            raise ValueError(f"failed_rank {failed_rank} not in [0, {self._ep_size})")
        payload = [failed_rank]
        for peer in range(self._ep_size):
            if peer == self._local_rank or peer == failed_rank:
                continue
            try:
                self._comm.isend(payload, dest=peer, tag=_FT_DEATH_NOTIFY_TAG)
            except MPI.Exception:  # noqa: BLE001 — production code uses MPI_ERRORS_RETURN
                # Send to a poisoned peer can fail; that's fine, we'll
                # discover via the recv loop on this rank that the peer is
                # gone via someone else's broadcast.
                pass

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                status = MPI.Status()
                if self._comm.iprobe(
                    source=MPI.ANY_SOURCE, tag=_FT_DEATH_NOTIFY_TAG, status=status
                ):
                    src = status.Get_source()
                    payload = self._comm.recv(source=src, tag=_FT_DEATH_NOTIFY_TAG)
                    failed_rank = int(payload[0])
                    if 0 <= failed_rank < self._ep_size:
                        if self._health.mark_failed(failed_rank):
                            if self._on_failure_received is not None:
                                self._on_failure_received(failed_rank, src, time.monotonic())
            except MPI.Exception:  # noqa: BLE001 — keep polling on transient errors
                pass
            self._stop.wait(self._config.poll_interval_sec)
