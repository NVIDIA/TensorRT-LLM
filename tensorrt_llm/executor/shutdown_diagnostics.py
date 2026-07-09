# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import faulthandler
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

_FORCE_EXIT_ENV = "TLLM_SHUTDOWN_FORCE_EXIT"
_TIMEOUT_ENV = "TLLM_SHUTDOWN_TRACE_TIMEOUT_SEC"
_TRACE_DIR_ENV = "TLLM_SHUTDOWN_TRACE_DIR"

logger = logging.getLogger(__name__)


def get_shutdown_timeout_seconds() -> Optional[float]:
    value = os.environ.get(_TIMEOUT_ENV)
    if value is None:
        return None
    try:
        timeout = float(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", _TIMEOUT_ENV, value)
        return None
    if timeout <= 0:
        logger.warning("Ignoring non-positive %s=%r", _TIMEOUT_ENV, value)
        return None
    return timeout


def trace_shutdown_phase(component: str, phase: str, **fields: Any) -> None:
    trace_directory = os.environ.get(_TRACE_DIR_ENV)
    if trace_directory is None:
        return

    details = " ".join(f"{key}={str(value).replace(chr(10), ' ')}" for key, value in fields.items())
    record = (
        f"time_ns={time.time_ns()} pid={os.getpid()} "
        f"tid={threading.get_native_id()} component={component} phase={phase}"
        f"{(' ' + details) if details else ''}\n"
    )
    trace_path = os.path.join(trace_directory, f"shutdown_trace.{os.getpid()}.log")
    try:
        fd = os.open(trace_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o600)
        try:
            os.write(fd, record.encode("utf-8", errors="backslashreplace"))
        finally:
            os.close(fd)
    except OSError:
        logger.debug("Failed to write shutdown phase trace", exc_info=True)


def dump_shutdown_stacks(component: str, phase: str) -> None:
    trace_directory = os.environ.get(_TRACE_DIR_ENV)
    if trace_directory is None:
        return

    safe_phase = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{component}.{phase}")
    trace_path = os.path.join(
        trace_directory, f"shutdown_stacks.{os.getpid()}.{safe_phase}.{time.time_ns()}.log"
    )
    try:
        with open(trace_path, "w", encoding="utf-8") as trace_file:
            faulthandler.dump_traceback(file=trace_file, all_threads=True)
    except OSError:
        logger.debug("Failed to write shutdown stack trace", exc_info=True)


def _watchdog_expired(component: str, phase: str, force_exit: bool) -> None:
    trace_shutdown_phase(component, phase, state="timeout")
    dump_shutdown_stacks(component, phase)
    if force_exit and os.environ.get(_FORCE_EXIT_ENV) == "1":
        trace_shutdown_phase(component, phase, state="force_exit")
        os._exit(1)


@contextmanager
def shutdown_watchdog(component: str, phase: str, *, force_exit: bool = False) -> Iterator[None]:
    """Trace a shutdown phase and dump stacks if its diagnostic deadline expires."""
    timeout = get_shutdown_timeout_seconds()
    started = time.monotonic()
    trace_shutdown_phase(component, phase, state="begin", timeout=timeout)
    timer = None
    if timeout is not None:
        timer = threading.Timer(timeout, _watchdog_expired, (component, phase, force_exit))
        timer.daemon = True
        timer.start()
    try:
        yield
    except BaseException as error:
        trace_shutdown_phase(
            component,
            phase,
            state="error",
            error_type=type(error).__name__,
            elapsed_seconds=time.monotonic() - started,
        )
        raise
    else:
        trace_shutdown_phase(
            component, phase, state="end", elapsed_seconds=time.monotonic() - started
        )
    finally:
        if timer is not None:
            timer.cancel()
            timer.join()


def wait_for_shutdown_event(event: threading.Event, component: str, phase: str) -> None:
    """Wait normally unless the diagnostic timeout requests a bounded wait."""
    timeout = get_shutdown_timeout_seconds()
    trace_shutdown_phase(component, phase, state="begin", timeout=timeout)
    if not event.wait(timeout=timeout):
        trace_shutdown_phase(component, phase, state="timeout")
        dump_shutdown_stacks(component, phase)
        raise TimeoutError(f"{component} timed out after {timeout} seconds in {phase}")
    trace_shutdown_phase(component, phase, state="end")
