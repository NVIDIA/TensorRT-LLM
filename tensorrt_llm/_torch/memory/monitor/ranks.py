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
dropped.
"""

from __future__ import annotations

import itertools
import threading
import time

from tensorrt_llm.logger import logger

__all__ = ["RankReporter"]

_REQUEST_TAG = 2626
_REPLY_TAG = 2627
_POLL_S = 0.2
_DEFAULT_TIMEOUT_S = 30.0


class RankReporter:
    """Rank 0 gathers; every other rank answers from a listener thread."""

    def __init__(self, comm, rank: int, world_size: int, device: int | None) -> None:
        self._comm = comm
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._stop = threading.Event()
        self._gather_lock = threading.Lock()
        self._sequence = itertools.count(1)
        self._thread = None
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
                if not self._comm.Iprobe(source=0, tag=_REQUEST_TAG):
                    self._stop.wait(_POLL_S)
                    continue
                sequence = self._comm.recv(source=0, tag=_REQUEST_TAG)
                report = report_now(tag="on_demand", rank=self._rank, force=True)
                self._comm.send((sequence, report), dest=0, tag=_REPLY_TAG)
            except Exception as exc:  # noqa: BLE001 - a diagnostic must not kill the rank
                logger.warning(
                    f"memory monitor: rank listener failed ({type(exc).__name__}: {exc})"
                )
                self._stop.wait(_POLL_S)

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
            while pending and time.monotonic() < deadline:
                for source in sorted(pending):
                    if not self._comm.Iprobe(source=source, tag=_REPLY_TAG):
                        continue
                    answered, report = self._comm.recv(source=source, tag=_REPLY_TAG)
                    if answered != sequence:
                        continue  # a late answer to an earlier request
                    replies[source] = report or f"rank {source}: the report could not be built"
                    pending.discard(source)
                if pending:
                    time.sleep(0.05)
            for source in pending:
                replies[source] = f"rank {source}: no report within {timeout:g} s"
            return [replies[source] for source in sorted(replies)]

    def stop(self) -> None:
        """End this rank's listener. Each rank stops its own."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * _POLL_S + 5)
