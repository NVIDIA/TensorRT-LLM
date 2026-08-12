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


def wait_for_endpoint_down(url: str, timeout: int = 300):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            if requests.get(url, timeout=5).status_code >= 100:
                print(f"endpoint {url} returned status code {requests.get(url).status_code}")
                time.sleep(1)
        except Exception as err:
            print(f"endpoint {url} is down, with exception: {err}")
            return
    raise RuntimeError(f"Endpoint {url} did not become down within {timeout} seconds")
