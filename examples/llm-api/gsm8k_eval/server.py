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
"""The managed ``trtllm-serve`` process: launch, reported address, GPU memory and cleanup.

A managed server is started in its own session and binds a kernel-chosen port that it writes
to its ``--report_addr`` file. Stopping it terminates the session leader, then kills every
remaining process of that session and every descendant of the leader. Nothing else is ever
signalled: no process is looked up by port or by name, and an existing endpoint is never
touched.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

#: Seconds the stopped server may take to exit before its processes are killed.
STOP_GRACE_S = 120.0


def serve_argv(base_argv: Sequence[str], addr_path: str | os.PathLike) -> list[str]:
    """``base_argv`` bound to 127.0.0.1 on a kernel-chosen port reported to ``addr_path``."""
    return [*base_argv, "--host", "127.0.0.1", "--port", "0", "--report_addr", str(addr_path)]


def _process_stat(pid: int) -> Optional[tuple[int, int, bytes]]:
    """Linux parent PID, session ID and kernel start time; None if the process is gone."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_bytes().rsplit(b")", 1)[1].split()
        return int(fields[1]), int(fields[3]), fields[19]
    except (OSError, IndexError, ValueError):
        return None


def owned_processes(leader: int, start_time: bytes) -> dict[int, bytes]:
    """Snapshot identities of the leader's session and descendants, excluding this caller."""
    table = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit() and (stat := _process_stat(int(entry))) is not None:
            table[int(entry)] = stat
    if leader not in table or table[leader][2] != start_time:
        return {}
    children: dict[int, list[int]] = {}
    owned = {pid for pid, (_, sid, _) in table.items() if sid == leader}
    for pid, (parent, _, _) in table.items():
        children.setdefault(parent, []).append(pid)
    stack = [leader]
    descendants = {leader}
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in descendants:
                descendants.add(child)
                stack.append(child)
    return {pid: table[pid][2] for pid in owned | descendants if pid != os.getpid()}


def _signal_process(pid: int, start_time: bytes, sig: int) -> None:
    """Signal this process incarnation through a pidfd, never a recycled PID."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        fd = os.pidfd_open(pid)
        try:
            stat = _process_stat(pid)
            if stat is not None and stat[2] == start_time:
                signal.pidfd_send_signal(fd, sig)
        finally:
            os.close(fd)


@contextlib.contextmanager
def _ignore_cleanup_signals() -> Iterator[None]:
    """A repeated interrupt must not abandon cleanup; restore the caller's handlers."""
    saved = {}
    try:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                saved[sig] = signal.signal(sig, signal.SIG_IGN)
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def _wait_unreaped(pid: int, timeout: float) -> None:
    """Wait for exit without releasing the leader PID before the final ownership scan."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                return
        except ChildProcessError:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.05, remaining))


def stop(process: subprocess.Popen, grace_s: float = STOP_GRACE_S) -> Optional[int]:
    """Terminate an owned server and identity-checked workers; return its exit code."""
    with _ignore_cleanup_signals():
        # A reaped leader's PID may already belong to another process.
        stat = _process_stat(process.pid) if process.returncode is None else None
        if stat is None:
            return process.poll()
        start_time = stat[2]
        owned = owned_processes(process.pid, start_time)
        try:
            _signal_process(process.pid, start_time, signal.SIGTERM)
            _wait_unreaped(process.pid, grace_s)
        finally:
            owned.update(owned_processes(process.pid, start_time))
            for pid, identity in owned.items():
                _signal_process(pid, identity, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=60)
    return process.returncode


class _ManagedServer(subprocess.Popen):
    """Keep the leader unreaped when polled; stop() reaps it after the worker sweep."""

    def poll(self) -> Optional[int]:
        if self.returncode is not None:
            return self.returncode
        status = os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if status is None:
            return None
        return status.si_status if status.si_code == os.CLD_EXITED else -status.si_status


@contextlib.contextmanager
def launch(
    argv: Sequence[str], env: Mapping[str, str], log_path: str | os.PathLike
) -> Iterator[subprocess.Popen]:
    """Run ``argv`` in a new session with output to ``log_path``; always stop it on exit."""
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        process = _ManagedServer(
            list(argv), env=dict(env), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            yield process
        finally:
            stop(process)


def wait_for_addr(
    addr_path: str | os.PathLike, timeout_s: float, process: Optional[subprocess.Popen] = None
) -> tuple[str, int]:
    """The ``host, port`` the server wrote to its ``--report_addr`` file."""
    deadline = time.monotonic() + timeout_s
    path = Path(addr_path)
    while time.monotonic() < deadline:
        if process is not None and (code := process.poll()) is not None:
            raise RuntimeError(f"the server exited with code {code} before reporting its address")
        try:
            reported = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            reported = ""
        if reported:
            host, _, port = reported.rpartition(":")
            return host, int(port)
        time.sleep(0.5)
    raise TimeoutError(f"the server did not report its address within {timeout_s:.0f} s")


@contextlib.contextmanager
def gpu_memory_poller(
    csv_path: str | os.PathLike, gpu_ids: Sequence[str]
) -> Iterator[Optional[subprocess.Popen]]:
    """Poll per-GPU used memory once a second into ``csv_path`` until the context ends.

    Does nothing when no GPU is named (the server's devices are unknown).
    """
    if not gpu_ids:
        yield None
        return
    with open(csv_path, "w", encoding="utf-8") as out:
        poller = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,uuid,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-lms",
                "1000",
                "-i",
                ",".join(gpu_ids),
            ],
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        try:
            yield poller
        finally:
            poller.terminate()
            try:
                poller.wait(timeout=10)
            except subprocess.TimeoutExpired:
                poller.kill()
                poller.wait()
