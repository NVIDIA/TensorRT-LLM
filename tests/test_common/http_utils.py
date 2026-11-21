# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    additional_server_procs: list[tuple[subprocess.Popen | None, str]] | None = None,
):
    """Poll ``url`` until it returns 200, failing fast and loudly otherwise.

    Fail-fast paths (all of which dump the tails of ``check_files`` so the
    server-side story lands in the CI log):
      - ``server_proc`` or an ``additional_server_procs`` entry exited ->
        no point polling when a required server is dead;
      - an error keyword appears in a ``check_files`` log (scanned every
        ``check_interval`` seconds; time-based rather than iteration-based);
      - ``timeout`` elapses without the endpoint becoming ready.
    """
    start = time.monotonic()
    next_file_check = start + check_interval
    missing_warned: set[str] = set()
    while time.monotonic() - start < timeout:
        fail_if_proc_died(server_proc, "Server process (before becoming ready)", check_files)
        for additional_proc, what in additional_server_procs or []:
            fail_if_proc_died(additional_proc, what, check_files)

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


def wait_for_endpoint_down(url: str, timeout: float = 300, required_failures: int = 10) -> None:
    """Wait for ``required_failures`` consecutive connection refusals.

    An HTTP response or any other request error resets the count. Probes run
    one second apart, so the timeout must allow enough time for the full streak.
    """
    deadline = time.monotonic() + timeout
    consecutive_refusals = 0
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            response = requests.get(url, timeout=min(5, remaining))
        except requests.RequestException as err:
            if isinstance(err, requests.ConnectionError) and _is_connection_refused(err):
                consecutive_refusals += 1
                if consecutive_refusals >= required_failures:
                    print(f"endpoint {url} refused the connection and is down")
                    return
            else:
                consecutive_refusals = 0
                print(f"Shutdown probe for {url} will retry after a request error: {err}")
        else:
            consecutive_refusals = 0
            print(f"endpoint {url} returned status code {response.status_code}")
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise RuntimeError(
        f"Endpoint {url} did not refuse {required_failures} consecutive probes "
        f"within {timeout} seconds"
    )


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
