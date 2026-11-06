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
import time
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

#: Seconds the stopped server may take to exit before its processes are killed.
STOP_GRACE_S = 120.0


def serve_argv(base_argv: Sequence[str], addr_path: str | os.PathLike) -> list[str]:
    """``base_argv`` bound to 127.0.0.1 on a kernel-chosen port reported to ``addr_path``."""
    return [*base_argv, "--host", "127.0.0.1", "--port", "0", "--report_addr", str(addr_path)]


def _session_members(sid: int) -> set[int]:
    """PIDs of the live processes whose session id is ``sid`` (Linux ``/proc``)."""
    members = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.getsid(int(entry)) == sid:
                members.add(int(entry))
        except (ProcessLookupError, PermissionError):
            continue
    return members


def _descendants(pid: int) -> set[int]:
    """PIDs of every live descendant of ``pid`` (Linux ``/proc`` parent links)."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8", errors="replace") as f:
                stat = f.read()
        except OSError:
            continue
        # The command name is parenthesized and may contain spaces; the ppid follows it.
        fields = stat.rsplit(")", 1)[-1].split()
        if len(fields) > 1 and fields[1].isdigit():
            children.setdefault(int(fields[1]), []).append(int(entry))
    found, stack = set(), [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def owned_processes(leader: int) -> set[int]:
    """The leader's session members and descendants, excluding the caller itself."""
    return (_session_members(leader) | _descendants(leader)) - {os.getpid()}


def stop(process: subprocess.Popen, grace_s: float = STOP_GRACE_S) -> Optional[int]:
    """Terminate an owned server and every process it started; return its exit code."""
    owned = owned_processes(process.pid)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
    owned |= owned_processes(process.pid)
    for pid in sorted(owned):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
    if process.poll() is None:
        process.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=60)
    return process.poll()


@contextlib.contextmanager
def launch(
    argv: Sequence[str], env: Mapping[str, str], log_path: str | os.PathLike
) -> Iterator[subprocess.Popen]:
    """Run ``argv`` in a new session with output to ``log_path``; always stop it on exit."""
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
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
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"the server exited with code {process.returncode} before reporting its address"
            )
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
