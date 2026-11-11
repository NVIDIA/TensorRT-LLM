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
"""Reports from every rank of a multi-rank MPI executor.

A report request -- ``LLM.get_memory_report()`` -- reaches rank 0 only under
the MPI executor. Rank 0 asks the other ranks over a communicator duplicated
for this purpose when the executor starts, and each of them answers from a
listener thread with a report of its own process.

Set up only while the monitor is on: the duplicated communicator and one
listener thread per rank are the whole cost, and the listener polls for a
request every fraction of a second rather than blocking in ``recv``, which
some MPI implementations implement by spinning a CPU core.

A rank that does not answer in time is reported as such rather than holding
up the others; its late answer is recognised by its sequence number and
dropped. A report is larger than MPI's eager limit, so a reply only
completes once rank 0 receives it; replies are therefore sent without
blocking, and a listener that is stopped while its reply is uncollected
abandons it instead of waiting in MPI. Rank 0 drains uncollected replies
when it stops.
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Any

from tensorrt_llm._torch.pyexecutor.mpi_polling import message_ready, try_recv
from tensorrt_llm.logger import logger

__all__ = ["RankReporter"]

_REQUEST_TAG = 2626
_REPLY_TAG = 2627
_POLL_S = 0.2
_SEND_POLL_S = 0.05
_GATHER_POLL_S = 0.05
_DEFAULT_TIMEOUT_S = 30.0

#: Replies whose send was abandoned when a listener stopped. MPI may still
#: read their buffers, so they stay referenced for the life of the process.
_abandoned_sends: list[Any] = []


class RankReporter:
    """Rank 0 gathers; every other rank answers from a listener thread."""

    def __init__(self, comm: Any, rank: int, world_size: int, device: int | None) -> None:
        self._comm = comm
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._stop = threading.Event()
        self._gather_lock = threading.Lock()
        self._sequence = itertools.count(1)
        self._thread: threading.Thread | None = None
        if rank != 0:
            self._thread = threading.Thread(
                target=self._listen, name="memory-monitor-rank-listener", daemon=True
            )
            self._thread.start()

    def _listen(self) -> None:
        import torch

        from .triggers import report_now

        if self._device is not None:
            torch.cuda.set_device(self._device)
        while not self._stop.is_set():
            try:
                if not message_ready(self._comm, 0, _REQUEST_TAG):
                    self._stop.wait(_POLL_S)
                    continue
                sequence = self._comm.recv(source=0, tag=_REQUEST_TAG)
                report = report_now(tag="on_demand", rank=self._rank, force=True, save=False)
                if not self._deliver(self._comm.isend((sequence, report), dest=0, tag=_REPLY_TAG)):
                    return
            except Exception as exc:  # a diagnostic must not kill the rank
                logger.warning(
                    f"memory monitor: rank listener failed ({type(exc).__name__}: {exc})"
                )
                self._stop.wait(_POLL_S)

    def _deliver(self, request: Any) -> bool:
        """Wait for a reply to be received; ``False`` if stopped first.

        ``request`` is what the communicator's ``isend`` returns: an
        ``MPI.Request`` or, for the pickle-5 communicator the executor
        uses, a ``pkl5.Request``; both offer ``test()``, returning
        ``(done, payload)``. A request abandoned on stop is kept referenced
        until the process exits, because its send buffer may still be read.
        """
        while not request.test()[0]:
            if self._stop.wait(_SEND_POLL_S):
                _abandoned_sends.append(request)
                logger.warning(
                    f"memory monitor: rank {self._rank} stopped before rank 0 collected its report"
                )
                return False
        return True

    def gather(self, timeout: float = _DEFAULT_TIMEOUT_S) -> list[str]:
        """Reports from ranks 1..N-1, in rank order. Rank 0 only."""
        if self._rank != 0:
            return []
        with self._gather_lock:
            sequence = next(self._sequence)
            for dest in range(1, self._world_size):
                self._comm.send(sequence, dest=dest, tag=_REQUEST_TAG)
            replies: dict[int, str] = {}
            deadline = time.monotonic() + timeout
            pending = set(range(1, self._world_size))

            def current(reply: tuple[int, str | None]) -> bool:
                # A reply to an earlier request that timed out is dropped.
                return reply[0] == sequence

            while pending and time.monotonic() < deadline:
                for source in sorted(pending):
                    received, reply = try_recv(self._comm, source, _REPLY_TAG, current)
                    if not received:
                        continue
                    replies[source] = reply[1] or f"rank {source}: the report could not be built"
                    pending.discard(source)
                if pending:
                    time.sleep(_GATHER_POLL_S)
            for source in pending:
                replies[source] = f"rank {source}: no report within {timeout:g} s"
            return [replies[source] for source in sorted(replies)]

    def _drain_replies(self) -> None:
        """Receive replies nobody waits for any more, so no send stays pending."""
        for source in range(1, self._world_size):
            try_recv(self._comm, source, _REPLY_TAG, accept=lambda _reply: False)

    def stop(self) -> None:
        """End this rank's listener. Each rank stops its own."""
        self._stop.set()
        # A gather still running receives its own replies.
        if self._rank == 0 and self._gather_lock.acquire(blocking=False):
            try:
                self._drain_replies()
            finally:
                self._gather_lock.release()
        if self._thread is not None:
            timeout = 2 * _POLL_S + 5
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    f"memory monitor: rank {self._rank}'s report listener did not stop "
                    f"within {timeout:g} s"
                )
