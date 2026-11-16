# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py.util import pkl5


class PrefixLoadCompletionTracker:
    """Collect worker reports without waiting for transfers or peer progress.

    The leader publishes completed IDs through the executor's ordered request
    queue. Tracking ends when that decision has been applied on every rank.
    """

    def __init__(self, comm: pkl5.Intracomm | None = None) -> None:
        self._comm = comm
        self._rank = comm.Get_rank() if comm is not None else 0
        self._size = comm.Get_size() if comm is not None else 1
        self._pending: set[int] = set()
        self._send_request: pkl5.Request | None = None
        self._receive_requests: dict[int, pkl5.Request] = {}
        self._workers_finished: dict[int, set[int]] = {}
        self._completed: set[int] = set()

    def track(self, reservation_id: int) -> None:
        if self._rank == 0:
            self._workers_finished[reservation_id] = set()

    def forget(self, reservation_id: int) -> None:
        self._workers_finished.pop(reservation_id, None)
        self._completed.discard(reservation_id)
        self._pending.discard(reservation_id)

    def report(self, reservation_ids: set[int]) -> None:
        """Queue newly completed local transfers; sending never waits for the leader."""
        if self._rank == 0:
            self._record(0, reservation_ids)
        else:
            self._pending.update(reservation_ids)

    def _record(self, rank: int, reservation_ids: set[int]) -> None:
        for reservation_id in reservation_ids:
            workers = self._workers_finished.get(reservation_id)
            if workers is None:
                continue
            workers.add(rank)
            if len(workers) == self._size:
                self._completed.add(reservation_id)

    def poll(self) -> None:
        """Progress at most one report per peer without a blocking receive."""
        if self._comm is None:
            return
        if self._rank != 0:
            if self._send_request is not None:
                finished, _ = self._send_request.test()
                if not finished:
                    return
                self._send_request = None
            if self._pending:
                self._send_request = self._comm.isend(self._pending, dest=0, tag=0)
                self._pending = set()
            return

        if not self._workers_finished and not self._receive_requests:
            return
        for rank in range(1, self._size):
            request = self._receive_requests.get(rank)
            if request is None:
                message = self._comm.improbe(source=rank, tag=0)
                if message is None:
                    continue
                request = message.irecv()
                self._receive_requests[rank] = request
            finished, reservation_ids = request.test()
            if finished:
                del self._receive_requests[rank]
                self._record(rank, reservation_ids)

    def take_completed(self) -> list[int]:
        """Return leader decisions to distribute before the next scheduling pass."""
        self.poll()
        completed = sorted(self._completed)
        self._completed.clear()
        return completed

    def close(self) -> None:
        """Release the private communicator after the executor has stopped."""
        if self._send_request is not None:
            self._send_request.wait()
            self._send_request = None
        for request in self._receive_requests.values():
            request.wait()
        self._receive_requests.clear()
        if self._comm is not None:
            self._comm.Free()
            self._comm = None
