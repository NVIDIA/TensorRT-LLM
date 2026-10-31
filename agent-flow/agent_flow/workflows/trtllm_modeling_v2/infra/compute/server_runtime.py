# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight server lifecycle helpers, also loadable by path without agent_flow."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Mapping, Sequence

DEVICE_RELEASE_TIMEOUT_S = 120

MULTI_TASK_SLURM_VARS = (
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_STEP_ID",
    "SLURM_TASK_PID",
    "SLURM_STEP_NUM_TASKS",
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def tail(path: Path, lines: int = 40) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-lines:])
    except OSError:
        return ""


def wait_healthy(port: int, server: subprocess.Popen, log_path: Path, timeout_s: int) -> float:
    start = time.monotonic()
    deadline = start + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SystemExit(
                f"server exited with {server.returncode} during boot; tail of {log_path}:\n"
                f"{tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return time.monotonic() - start
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(5)
    raise SystemExit(f"server not healthy after {timeout_s}s; see {log_path}")


def server_devices(count: int) -> str:
    """The first ``count`` devices this job was given, as ``nvidia-smi -i`` takes them.

    nvidia-smi ignores ``CUDA_VISIBLE_DEVICES`` and reports the whole node, other jobs' devices too.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") or ",".join(str(i) for i in range(count))
    return ",".join(visible.split(",")[: max(count, 1)])


def device_memory_used_mib(count: int = 1) -> int | None:
    """The most memory in use on any of the first ``count`` visible devices (a server's)."""
    out = subprocess.run(
        ["nvidia-smi", "-i", server_devices(count), "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout  # fmt: skip
    used = [int(line) for line in out.split() if line.isdigit()]
    return max(used) if used else None


def device_holders(count: int = 1) -> set[int]:
    """Pids that hold memory on the first ``count`` devices this job was given."""
    out = subprocess.run(
        ["nvidia-smi", "-i", server_devices(count), "--query-compute-apps=pid",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout  # fmt: skip
    return {int(line) for line in out.split() if line.isdigit()}


def _alive(pid: int) -> bool:
    """Whether ``pid`` still runs: a zombie has exited and holds no device."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat.rpartition(")")[2].split()[0] != "Z"


def reap(pids: set[int], grace_s: float = 30.0) -> list[int]:
    """SIGTERM ``pids``, SIGKILL any still alive after ``grace_s``; the pids signalled.

    A server under ``mpirun`` keeps its ranks outside the launcher's process group, so
    signalling that group can leave a rank holding a device for the next server to measure.
    """
    signalled: list[int] = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = sorted(p for p in pids if _alive(p))
        if not alive:
            break
        for pid in alive:
            try:
                os.kill(pid, sig)
                signalled.append(pid)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline and any(_alive(p) for p in alive):
            time.sleep(0.5)
    return signalled


def rank_launch(world_size: int, launcher: str = "trtllm-llmapi-launch") -> list[str]:
    """The argv prefix that creates ``world_size`` ranks for one server, or none for one.

    Slurm task variables are cleared so the launcher reads the new MPI world's
    rank and size instead of the enclosing one-task allocation.
    """
    if world_size <= 1:
        return []
    scrub = [arg for name in MULTI_TASK_SLURM_VARS for arg in ("-u", name)]
    return [
        "env", *scrub,
        "mpirun", "-n", str(world_size), "--oversubscribe", "--allow-run-as-root", launcher,
    ]  # fmt: skip


def _terminate(signum, frame):  # noqa: ARG001
    # A cancelled job gets SIGTERM. Python's default would exit without running
    # the caller's `finally`, and the server -- in its own session -- would outlive
    # the job holding the GPU. Raising turns the signal into that teardown.
    raise SystemExit(128 + signum)


def start_server(argv: Sequence[str], log_path: Path, env: Mapping[str, str]) -> subprocess.Popen:
    """Start a server in its own process group with its log kept on disk."""
    with log_path.open("wb") as log:
        return subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )


def stop_server(
    server: subprocess.Popen,
    world: int,
    before: set[int],
    *,
    grace_s: float = 90,
    holders: Callable[[int], set[int]] = device_holders,
    log: Callable[[str], None] | None = None,
    report_remaining: bool = True,
) -> dict:
    """Stop the group and only new holders of the job's assigned devices."""
    try:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    except ProcessLookupError:
        server.wait()
    stray = holders(world) - before - {os.getpid()}
    if stray and log is not None:
        log(f"[perf] server left device holders {sorted(stray)}; reaping")
    reaped = sorted(set(reap(stray))) if stray else []
    deadline = time.monotonic() + DEVICE_RELEASE_TIMEOUT_S
    while stray and holders(world) & stray and time.monotonic() < deadline:
        time.sleep(1)
    return {
        "returncode": server.returncode,
        "reaped": reaped,
        "left": sorted(holders(world) & stray) if stray and report_remaining else [],
    }
