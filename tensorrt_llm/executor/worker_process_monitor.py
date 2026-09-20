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

"""Linux process-liveness monitoring for locally spawned executor workers."""

from __future__ import annotations

import logging
import os
import select
import socket
import threading
from typing import Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)


class WorkerProcessIdentity(NamedTuple):
    """Stable identity for one executor worker process."""

    rank: int
    pid: int
    start_time: Optional[int]
    hostname: str
    pid_namespace: Optional[int]


def _read_process_state(pid: int) -> Optional[tuple[str, int]]:
    """Read the Linux process state and start time from ``/proc``.

    Returns:
        A ``(state, start_time)`` tuple, or ``None`` when the process does not
        exist or procfs is unavailable.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
            stat = stat_file.read()
    except (FileNotFoundError, ProcessLookupError):
        return None

    # The second field is parenthesized and may contain spaces or parentheses.
    command_end = stat.rfind(")")
    if command_end < 0:
        return None
    fields = stat[command_end + 2 :].split()
    if len(fields) <= 19:
        return None
    return fields[0], int(fields[19])


def _read_pid_namespace(pid: int) -> Optional[int]:
    """Return the inode identifying a Linux PID namespace."""
    try:
        return os.stat(f"/proc/{pid}/ns/pid").st_ino
    except OSError:
        return None


def capture_worker_process_identity(rank: int, pid: Optional[int] = None) -> WorkerProcessIdentity:
    """Capture the calling worker's rank, PID, and Linux start time."""
    pid = pid if pid is not None else os.getpid()
    try:
        process_state = _read_process_state(pid)
    except (OSError, ValueError):
        process_state = None
    start_time = process_state[1] if process_state is not None else None
    return WorkerProcessIdentity(
        rank=rank,
        pid=pid,
        start_time=start_time,
        hostname=socket.gethostname(),
        pid_namespace=_read_pid_namespace(pid),
    )


class WorkerProcessMonitor:
    """Monitor locally spawned workers without relying on MPI futures.

    Linux pidfds are used when Python exposes ``os.pidfd_open``.  A procfs
    identity check is retained as a fallback for older Python versions.  If
    neither mechanism is available, the worker is left to the existing MPI
    future/error-queue checks.
    """

    def __init__(self) -> None:
        self._pidfd_to_identity: Dict[int, WorkerProcessIdentity] = {}
        self._procfs_identities: List[WorkerProcessIdentity] = []
        self._dead_identity: Optional[WorkerProcessIdentity] = None
        self._poller = select.poll() if hasattr(select, "poll") else None
        self._lock = threading.Lock()
        self._local_hostname = socket.gethostname()
        self._local_pid_namespace = _read_pid_namespace(os.getpid())

    def register(self, identities: List[WorkerProcessIdentity]) -> None:
        """Register locally spawned worker identities for liveness checks."""
        with self._lock:
            self._close_unlocked()
            pidfd_open = getattr(os, "pidfd_open", None)
            for identity in identities:
                if identity.hostname != self._local_hostname:
                    continue
                if identity.pid_namespace != self._local_pid_namespace:
                    continue
                if pidfd_open is not None and self._poller is not None:
                    try:
                        pidfd = pidfd_open(identity.pid)
                    except ProcessLookupError:
                        self._dead_identity = identity
                        continue
                    except OSError as error:
                        logger.debug("pidfd_open(%d) failed: %s", identity.pid, error)
                    else:
                        if identity.start_time is not None:
                            try:
                                process_state = _read_process_state(identity.pid)
                            except (OSError, ValueError) as error:
                                logger.debug(
                                    "Failed to validate process identity for pid %d: %s",
                                    identity.pid,
                                    error,
                                )
                            else:
                                if (
                                    process_state is None
                                    or process_state[0] == "Z"
                                    or process_state[1] != identity.start_time
                                ):
                                    os.close(pidfd)
                                    self._dead_identity = identity
                                    continue
                        self._pidfd_to_identity[pidfd] = identity
                        self._poller.register(
                            pidfd, select.POLLIN | select.POLLERR | select.POLLHUP
                        )
                        continue

                if identity.start_time is not None:
                    self._procfs_identities.append(identity)

    def find_dead_worker(self) -> Optional[WorkerProcessIdentity]:
        """Return the first worker known to have exited, if any."""
        with self._lock:
            if self._dead_identity is not None:
                return self._dead_identity

            if self._poller is not None:
                for pidfd, _ in self._poller.poll(0):
                    identity = self._pidfd_to_identity.get(pidfd)
                    if identity is not None:
                        self._dead_identity = identity
                        return identity

            for identity in self._procfs_identities:
                try:
                    process_state = _read_process_state(identity.pid)
                except (OSError, ValueError) as error:
                    logger.debug(
                        "Failed to check process state for pid %d: %s", identity.pid, error
                    )
                    continue
                if (
                    process_state is None
                    or process_state[0] == "Z"
                    or process_state[1] != identity.start_time
                ):
                    self._dead_identity = identity
                    return identity
            return None

    def close(self) -> None:
        """Close all process handles and reset the monitor."""
        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        for pidfd in self._pidfd_to_identity:
            if self._poller is not None:
                try:
                    self._poller.unregister(pidfd)
                except (KeyError, OSError):
                    pass
            try:
                os.close(pidfd)
            except OSError:
                pass
        self._pidfd_to_identity.clear()
        self._procfs_identities.clear()
        self._dead_identity = None
