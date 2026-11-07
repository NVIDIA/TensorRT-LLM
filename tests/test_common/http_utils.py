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

import errno
import os
import subprocess
import time

import requests

from test_common.error_utils import check_error, report_error


def _fail_with_server_logs(message: str, check_files: list[str] | None):
    """Raise RuntimeError(message), appending the tail of each server log.

    A server that never becomes healthy often prints its story only to its own
    log file; without dumping it here the CI log shows nothing but the client
    poll loop, making the failure unclassifiable.
    """
    if check_files:
        # report_error raises RuntimeError(message + error context / log tails).
        report_error(message, check_files)
    raise RuntimeError(message)


def fail_if_proc_died(
    proc: subprocess.Popen | None,
    what: str,
    check_files: list[str] | None = None,
):
    """Event-driven fail-fast: raise (with server-log tails) if ``proc`` exited.

    Process death is an event, not a timeout: a babysitter that checks its
    child converts a dead server into an immediate, diagnosable failure
    instead of burning GPUs until a timeout. The resulting nonzero rank exit
    ends that rank's SLURM step (``srun --kill-on-bad-exit=1`` in the perf
    scripts); the remaining steps then fail fast on the dead endpoint via the
    bounded ready-wait.
    """
    if proc is None:
        return
    exit_code = proc.poll()
    if exit_code is not None:
        _fail_with_server_logs(
            f"{what} exited unexpectedly with code {exit_code} while it was still needed.",
            check_files,
        )


def wait_for_endpoint_ready(
    url: str,
    timeout: int = 300,
    check_files: list[str] | None = None,
    server_proc: subprocess.Popen | None = None,
    check_interval: float = 30.0,
):
    """Poll ``url`` until it returns 200, failing fast and loudly otherwise.

    Fail-fast paths (all of which dump the tails of ``check_files`` so the
    server-side story lands in the CI log):
      - ``server_proc`` exited -> no point polling a dead server for the
        remaining timeout;
      - an error keyword appears in a ``check_files`` log (scanned every
        ``check_interval`` seconds; time-based rather than iteration-based);
      - ``timeout`` elapses without the endpoint becoming ready.
    """
    start = time.monotonic()
    next_file_check = start + check_interval
    missing_warned: set[str] = set()
    while time.monotonic() - start < timeout:
        # Check server_proc if provided (singular)
        fail_if_proc_died(server_proc, "Server process (before becoming ready)", check_files)

        if check_files and time.monotonic() >= next_file_check:
            next_file_check = time.monotonic() + check_interval
            for check_file in check_files:
                if not os.path.exists(check_file):
                    if check_file not in missing_warned:
                        missing_warned.add(check_file)
                        print(
                            f"[WARNING] server log {check_file} does not exist "
                            "(yet?); cannot scan it for errors"
                        )
                    continue
                error_lines = check_error(check_file)
                if error_lines:
                    error_lines_str = ", ".join(
                        [f"line {line_idx}: {line_str}" for line_idx, line_str in error_lines]
                    )
                    _fail_with_server_logs(
                        f"Found error in server file {check_file}: {error_lines_str}",
                        check_files,
                    )
        try:
            time.sleep(1)
            if requests.get(url, timeout=5).status_code == 200:
                print(f"endpoint {url} is ready")
                return
        except Exception as err:
            print(f"endpoint {url} is not ready, with exception: {err}")
    _fail_with_server_logs(
        f"Endpoint {url} did not become ready within {timeout} seconds", check_files
    )


def wait_for_endpoint_down(url: str, timeout: int = 300) -> None:
    """Wait until a request fails with connection refused or the deadline passes."""
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            response = requests.get(url, timeout=min(5, remaining))
        except requests.ConnectionError as err:
            if _is_connection_refused(err):
                print(f"endpoint {url} refused the connection and is down")
                return
            print(f"Shutdown probe for {url} will retry after a connection error: {err}")
        except requests.Timeout as err:
            print(
                f"Shutdown probe for {url} timed out while waiting for the endpoint to stop: {err}"
            )
        else:
            print(f"endpoint {url} is still up and returned status code {response.status_code}")
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise RuntimeError(f"Endpoint {url} did not become down within {timeout} seconds")


def _is_connection_refused(error: BaseException) -> bool:
    """Return whether an HTTP connection error wraps ECONNREFUSED."""
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if getattr(current, "errno", None) == errno.ECONNREFUSED:
            return True
        pending.extend(
            nested
            for nested in (
                current.__cause__,
                current.__context__,
                getattr(current, "reason", None),
                *current.args,
            )
            if isinstance(nested, BaseException)
        )
    return False
