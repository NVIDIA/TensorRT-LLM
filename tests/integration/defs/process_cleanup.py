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
"""Last-resort cleanup for processes owned by one CI test command."""

import os
import signal
import subprocess
import time
import uuid
import warnings
from collections.abc import Sequence

import psutil

PROCESS_OWNER_ENV_VAR = "TRTLLM_TEST_PROCESS_OWNER_TOKEN"

_GONE_PROCESS_ERRORS = (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError)


def new_process_owner_token() -> str:
    """Return a unique identifier for one command and its descendants."""
    return uuid.uuid4().hex


def cleanup_owned_processes(
    owner_token: str,
    *,
    description: str,
    terminate_grace_seconds: float = 5.0,
    kill_wait_seconds: float = 5.0,
) -> list[int]:
    """Terminate all live processes carrying `owner_token`.

    The environment marker is intentionally prefixed with `TRTLLM` because `MpiPoolSession` forwards
    such variables to spawned workers. Unlike parent-PID, process-group, or session tracking, the
    marker survives MPI reparenting and workers that create a new session.

    Returns the PIDs that remain after SIGKILL verification. The caller should fail the owning test
    command when this list is non-empty.
    """
    if not owner_token:
        raise ValueError("owner_token must not be empty")

    processes = _owned_processes(owner_token)
    if not processes:
        return []

    # Record start times alongside PIDs so a PID recycled by an unrelated process cannot later be
    # mistaken for one of our leftovers.
    identities = _process_identities(processes)
    found_pids = set(identities)
    warnings.warn(f"Found leftover processes from {description}: {sorted(found_pids)}")

    # Sample NVML while the leftovers are still alive. Seeing one of them here is what proves that
    # NVML reports PIDs in our namespace; the driver normally translates them, but that is not
    # verified for every CI container, so we refuse to draw conclusions without the evidence.
    gpu_pids_before_cleanup = _gpu_process_pids()

    _signal_processes(processes, signal.SIGTERM)
    _wait_until_no_owned_processes(owner_token, terminate_grace_seconds)

    # Rescan in a loop: it picks up descendants created while the original processes were
    # terminating - including ones forked during the kill itself - as well as workers that changed
    # process group or session.
    kill_deadline = time.monotonic() + kill_wait_seconds
    while True:
        survivors = _owned_processes(owner_token)
        if not survivors:
            break
        identities.update(_process_identities(survivors))
        found_pids.update(identities)
        _signal_processes(survivors, signal.SIGKILL)
        time_left = kill_deadline - time.monotonic()
        if time_left <= 0:
            break
        _wait_until_no_owned_processes(owner_token, min(0.5, time_left))

    remaining = {process.pid for process in _owned_processes(owner_token)}
    remaining.update(
        _verify_gpu_contexts_released(
            identities,
            gpu_pids_before_cleanup,
            kill_wait_seconds,
            description=description,
        )
    )
    if remaining:
        warnings.warn(
            f"Unable to clean all leftover processes or GPU contexts from {description}: "
            f"{sorted(remaining)}"
        )
    return sorted(remaining)


def _process_identities(processes: Sequence[psutil.Process]) -> dict[int, float | None]:
    """Map each PID to its process start time, or to `None` when that cannot be read."""
    identities: dict[int, float | None] = {}
    for process in processes:
        try:
            identities[process.pid] = process.create_time()
        except _GONE_PROCESS_ERRORS:
            # The process may already be gone, but its GPU context can outlive it, so keep the PID
            # and accept that we will not be able to rule out PID reuse for it.
            identities[process.pid] = None
        except (PermissionError, psutil.Error):
            identities[process.pid] = None
    return identities


def _is_recycled_pid(pid: int, create_time: float | None) -> bool:
    """Return whether `pid` now belongs to a process other than the one we recorded."""
    if create_time is None:
        return False
    try:
        return psutil.Process(pid).create_time() != create_time
    except _GONE_PROCESS_ERRORS:
        # Nothing alive holds the PID, so a GPU context under it belongs to our dead process.
        return False
    except (PermissionError, psutil.Error):
        return False


def _is_live(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except _GONE_PROCESS_ERRORS:
        return False
    except (PermissionError, psutil.Error) as error:
        warnings.warn(f"Unable to inspect process {process.pid} ({type(error).__name__}): {error}")
        return False


def _owned_processes(owner_token: str) -> list[psutil.Process]:
    current_uid = os.getuid()
    owned = []
    for process in psutil.process_iter(["pid", "uids"]):
        try:
            uids = process.info["uids"]
            if uids is None or current_uid not in uids:
                continue
            if not _is_live(process):
                continue
            if process.environ().get(PROCESS_OWNER_ENV_VAR) != owner_token:
                continue
            owned.append(process)
        except _GONE_PROCESS_ERRORS:
            continue
        except (PermissionError, psutil.Error) as error:
            # An exiting process still reports as live for a moment after `/proc/<pid>/environ`
            # becomes unreadable, so give it a beat before treating this as a real problem.
            time.sleep(0.01)
            if not _is_live(process):
                continue
            warnings.warn(
                f"Unable to inspect process {process.pid} while looking for owned workers "
                f"({type(error).__name__}): {error}"
            )
    return sorted(owned, key=lambda process: process.pid)


def _signal_processes(processes: Sequence[psutil.Process], sig: signal.Signals) -> None:
    for process in processes:
        try:
            if _is_live(process):
                process.send_signal(sig)
        except _GONE_PROCESS_ERRORS:
            continue
        except (PermissionError, psutil.Error) as error:
            warnings.warn(
                f"Unable to send {sig.name} to process {process.pid} ({type(error).__name__}): "
                f"{error}"
            )


def _wait_until_no_owned_processes(owner_token: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    # Each scan reads /proc/<pid>/environ for every same-uid process, so back off rather than
    # hammering a loaded CI node at a fixed high rate.
    interval = 0.05
    while _owned_processes(owner_token):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(interval, remaining))
        interval = min(interval * 2, 0.5)


def _gpu_process_pids() -> set[int] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        warnings.warn(f"Unable to verify that GPU contexts were released: {error}")
        return None
    if result.returncode != 0:
        detail = result.stderr.strip()
        warnings.warn(
            f"Unable to verify that GPU contexts were released: nvidia-smi "
            f"exited with {result.returncode}{f': {detail}' if detail else ''}"
        )
        return None

    pids = set()
    for line in result.stdout.splitlines():
        try:
            pids.add(int(line.strip()))
        except ValueError:
            warnings.warn(f"Ignoring unexpected nvidia-smi PID output: {line!r}")
    return pids


def _verify_gpu_contexts_released(
    identities: dict[int, float | None],
    gpu_pids_before_cleanup: set[int] | None,
    timeout: float,
    *,
    description: str,
) -> list[int]:
    """Return the PIDs whose GPU contexts outlived the processes we killed.

    `gpu_pids_before_cleanup` is the NVML sample taken while those processes were still alive. It
    decides whether the comparison is meaningful at all: an empty result is only reported as "no
    leaked contexts" when we can tell that apart from "we could not see our own processes".
    """
    if gpu_pids_before_cleanup is None:
        # Already warned about by `_gpu_process_pids`.
        return []
    if not gpu_pids_before_cleanup:
        # No compute process held a context in the first place, so there is nothing to wait for.
        return []
    if not gpu_pids_before_cleanup & set(identities):
        warnings.warn(
            f"Skipping GPU context verification for {description}: nvidia-smi reported compute "
            f"processes {sorted(gpu_pids_before_cleanup)} but none of the leftover processes "
            f"{sorted(identities)}. Either those contexts belong to another job, or NVML reports "
            f"PIDs from a different namespace, in which case leaked contexts cannot be detected "
            f"by PID."
        )
        return []

    return _wait_until_gpu_contexts_released(identities, timeout)


def _wait_until_gpu_contexts_released(
    identities: dict[int, float | None], timeout: float
) -> list[int]:
    deadline = time.monotonic() + timeout
    while True:
        gpu_pids = _gpu_process_pids()
        if gpu_pids is None:
            return []
        remaining = sorted(
            pid for pid in gpu_pids & set(identities) if not _is_recycled_pid(pid, identities[pid])
        )
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(min(0.25, deadline - time.monotonic()))
