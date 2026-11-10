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
"""When the monitor reports.

The memory lives in the executor's process, not the one that created the
``LLM``. A report taken from the parent describes the parent -- its CUDA
context and nothing else -- so every trigger here has to fire inside the
worker.

Two triggers fire there. In full mode startup ends with one breakdown --
tagged ``startup_failed`` if startup raised. After that, an out-of-memory
error reports, which is the moment the breakdown answers a question somebody
is actually asking.

A report is built only for an out-of-memory error nobody handled. Some
callers allocate optimistically and fall back when that fails -- KV cache
sizing, warmup and graph capture, optional workspaces in attention and MoE
kernels -- and torch's OOM observer fires before any of them can catch the
error. The observer therefore only records the failed allocation. The report
is built when the error is seen to escape: the executor's error handler and
the sleep/wakeup listener pass their message to :func:`notify_error`,
TensorRT-LLM's logger passes every ERROR message through it -- background
threads such as the KV transfer workers catch their failures and log them --
and a thread that dies of an out-of-memory error reaches it through
``threading.excepthook``. The recorded allocation, if recent, goes into the
report's headline: how much was asked for and what PyTorch held at that
moment, before the failing step's memory was released.

The same path covers allocations made outside the PyTorch allocator --
cuBLAS workspaces, NCCL buffers, TensorRT-LLM's own pools -- which torch's
observer never sees. :func:`notify_error` recognises their messages; NCCL's
generic CUDA error is taken for an out-of-memory error only when the device
is nearly full. Host-memory failures are not GPU memory and are ignored.

Reporting is suspended while an executor is being created
(:func:`suspend_reporting`), since creation runs allocations into the wall on
purpose, and every executor that armed reporting releases it when it shuts
down (:func:`release_oom_reporter`).

With ``TLLM_MEM_MONITOR=1`` and ``TLLM_MEM_MONITOR_INTERVAL`` set to a
number of seconds, a background thread also reports at that interval, so
memory that grows slowly while serving shows up before it runs out. It only
wakes on its timer; nothing is added to the executor loop.

With ``TLLM_MEM_MONITOR_DIR`` set, every report except on-demand ones is
also written to its own file there, named by rank, process, time and tag.

Reports are built on the executor's device, recorded when reporting is
armed: the thread that logs an error may never have selected a device, and
reading device 0 from it would describe the wrong GPU and could create a
CUDA context there.

The handlers allocate no device memory and never raise. Out-of-memory
reports are logged at ERROR, so that the default log level shows them, and
limited by time rather than by count: a retry loop that fails repeatedly
collapses to one report per interval, while an unrelated failure hours later
is still reported. A report that could not be built does not use up the
interval.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "OOM_INTERVAL_VAR",
    "PERIODIC_INTERVAL_VAR",
    "install_oom_reporter",
    "notify_error",
    "release_oom_reporter",
    "report_now",
    "suspend_reporting",
    "uninstall_oom_reporter",
]

#: Directory reports are also written to, one file each.
DIR_VAR = "TLLM_MEM_MONITOR_DIR"

#: Seconds between periodic reports in full mode; unset or ``0`` for none.
PERIODIC_INTERVAL_VAR = "TLLM_MEM_MONITOR_INTERVAL"
#: Shortest periodic interval; each report takes an allocator snapshot.
MIN_PERIODIC_INTERVAL_S = 1.0

#: Minimum seconds between two OOM reports. ``0`` reports every one.
OOM_INTERVAL_VAR = "TLLM_MEM_MONITOR_OOM_INTERVAL"
_DEFAULT_OOM_INTERVAL_S = 60.0

#: A failed PyTorch allocation older than this is not linked to an error.
_FAILED_ALLOCATION_WINDOW_S = 30.0

#: Whether the callback is attached to torch. Never cleared: torch keeps
#: every observer it is handed and offers no way to remove one, so a second
#: attach would double every report.
_attached = False

#: Whether OOM reporting is armed. Independent of ``_attached``: errors
#: passed to :func:`notify_error` are reported even where torch offers no
#: observer.
_enabled_flag = False
#: Executors that armed reporting and have not released it.
_armed_count = 0
#: Device reports are built on; the most recently armed executor's.
_device: int | None = None
#: Nesting depth of :func:`suspend_reporting`.
_suspended = 0
_last_report_at: float | None = None
#: Whether a thread is building an OOM report right now.
_report_in_flight = False
_rate_lock = threading.Lock()
#: Set on a thread while it builds a report, so the report's own logging
#: cannot trigger another.
_reporting = threading.local()

#: Messages that mean a device allocation failed, from CUDA, the driver,
#: PyTorch, cuBLAS and TensorRT-LLM's C++ code.
_OOM_PATTERN = re.compile(
    r"out of memory|OUT_OF_MEMORY|cudaErrorMemoryAllocation|OutOfMemoryError"
    r"|CUBLAS_STATUS_ALLOC_FAILED|Failed to create cublas handle\. Status: 3\b",
    re.IGNORECASE,
)
#: Failures that are often, but not always, an allocation failing: NCCL
#: reports a failed ``cudaMalloc`` as a generic CUDA error.
_MAYBE_OOM_PATTERN = re.compile(r"unhandled cuda error", re.IGNORECASE)
#: A device this full makes a :data:`_MAYBE_OOM_PATTERN` error an OOM.
_NEARLY_FULL_FRACTION = 0.02
_NEARLY_FULL_BYTES = 1 << 30
#: Host-memory failures, which mention "out of memory" too.
_HOST_MEMORY_PATTERN = re.compile(
    r"HostOOMError|Host out of memory|cuMemHostRegister|cudaHostAlloc|cudaMallocHost"
    r"|pinned memory",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _FailedAllocation:
    """What torch's observer saw when a PyTorch allocation failed."""

    device: int
    requested_bytes: int
    device_total: int
    device_free: int
    live_bytes: int | None
    reserved_bytes: int | None
    at: float

    def describe(self) -> str:
        gib = 1 << 30
        held = ""
        if self.live_bytes is not None and self.reserved_bytes is not None:
            held = (
                f"; PyTorch held {self.live_bytes / gib:.2f} GiB live, "
                f"{self.reserved_bytes / gib:.2f} GiB reserved"
            )
        return (
            f"PyTorch could not allocate {self.requested_bytes / gib:.2f} GiB on device "
            f"{self.device} (device free {self.device_free / gib:.2f} of "
            f"{self.device_total / gib:.2f} GiB{held})"
        )


_last_failed_allocation: _FailedAllocation | None = None


def _oom_interval_s() -> float:
    from . import seconds_setting

    return seconds_setting(OOM_INTERVAL_VAR, _DEFAULT_OOM_INTERVAL_S)


def _periodic_interval_s() -> float:
    """Seconds between periodic reports; ``0`` for none."""
    from . import seconds_setting, warn_once

    interval = seconds_setting(PERIODIC_INTERVAL_VAR, 0.0)
    if 0 < interval < MIN_PERIODIC_INTERVAL_S:
        warn_once(
            PERIODIC_INTERVAL_VAR,
            os.environ.get(PERIODIC_INTERVAL_VAR, ""),
            f"is below the minimum of {MIN_PERIODIC_INTERVAL_S:g} s; using the minimum",
        )
        return MIN_PERIODIC_INTERVAL_S
    return interval


def _check_settings() -> None:
    """Read every setting once, so an invalid one is reported when arming."""
    from . import mode, selected_ranks, stacks_requested

    mode()
    selected_ranks()
    stacks_requested()
    _oom_interval_s()
    _periodic_interval_s()


def report_now(
    tag: str,
    rank: int | None = None,
    *,
    headline: str = "",
    record_baseline: bool = False,
    suggestion: str = "",
    startup_steps: bool = False,
    force: bool = False,
    device_index: int | None = None,
    save: bool = True,
) -> str | None:
    """Render the monitor for this process, or ``None`` if it is disabled.

    ``force`` renders it even when disabled; see :func:`capture`.

    Returns the text without logging it; the caller decides where it goes
    (the log, an exception message, a crash dump). With ``save`` it is also
    written to ``TLLM_MEM_MONITOR_DIR`` when that is set. ``record_baseline``
    makes this reading the one later reports compare against.
    ``device_index`` is the device to describe, by default the calling
    thread's current one; the report is built with it selected.
    """
    from . import capture, render_capture

    selected = (
        torch.cuda.device(device_index) if device_index is not None else contextlib.nullcontext()
    )
    try:
        with selected:
            snapshot = capture(
                tag=tag,
                rank=rank,
                device_index=device_index,
                headline=headline,
                record_baseline=record_baseline,
                suggestion=suggestion,
                startup_steps=startup_steps,
                force=force,
            )
    except Exception as exc:  # a report must not add a failure
        logger.warning(f"memory monitor: capture for {tag!r} failed ({type(exc).__name__}: {exc})")
        return None

    if snapshot is None:
        return None

    try:
        text = render_capture(snapshot)
    except Exception as exc:  # a report must not add a failure
        logger.warning(f"memory monitor: rendering {tag!r} failed ({type(exc).__name__}: {exc})")
        return None
    if save:
        _save(text, tag, snapshot.rank)
    return text


_UNSAFE_FILE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _save(text: str, tag: str, rank: int | None) -> str | None:
    """Write a report to ``TLLM_MEM_MONITOR_DIR``; the path, or ``None``."""
    directory = os.environ.get(DIR_VAR, "").strip()
    if not directory:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
    rank_part = "x" if rank is None else str(rank)
    safe_tag = _UNSAFE_FILE_CHARS.sub("_", tag) or "report"
    path = os.path.join(
        directory, f"memory_rank{rank_part}_pid{os.getpid()}_{stamp}_{safe_tag}.txt"
    )
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            handle.write(text + "\n")
    except OSError as exc:
        logger.warning(f"memory monitor: could not write {path} ({exc})")
        return None
    return path


def _claim_report_slot() -> bool:
    """Whether this thread may build the next OOM report."""
    global _report_in_flight
    with _rate_lock:
        if _report_in_flight:
            return False
        now = time.monotonic()
        if _last_report_at is not None and now - _last_report_at < _oom_interval_s():
            return False
        _report_in_flight = True
        return True


def _finish_report_slot(produced: bool) -> None:
    """Free the slot; only a report that was produced starts the interval."""
    global _report_in_flight, _last_report_at
    with _rate_lock:
        _report_in_flight = False
        if produced:
            _last_report_at = time.monotonic()


def _report_oom(headline: str) -> None:
    """Emit one OOM report, subject to the shared rate limit. Never raises."""
    if not _enabled_flag or _suspended or getattr(_reporting, "active", False):
        return
    if not _claim_report_slot():
        return
    produced = False
    _reporting.active = True
    try:
        # Forced: in the default mode the full monitor is off, and an armed
        # reporter is all the permission this report needs.
        report = report_now(
            tag="out_of_memory", headline=headline, force=True, device_index=_device
        )
        produced = report is not None
        logger.error(f"memory monitor:\n{report}" if produced else f"memory monitor: {headline}")
    except Exception:  # never add a second failure to an OOM
        pass
    finally:
        _reporting.active = False
        _finish_report_slot(produced)


def _on_out_of_memory(
    device: int, requested_bytes: int, device_total: int, device_free: int
) -> None:
    """torch's OOM observer: record the failed allocation, report nothing.

    torch calls it before the caller can catch the error; whether the error
    is a failure is decided later, when it reaches :func:`notify_error`.
    """
    global _last_failed_allocation
    if not _enabled_flag or _suspended:
        return
    live = reserved = None
    try:
        live = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
    except RuntimeError:
        pass
    _last_failed_allocation = _FailedAllocation(
        device=device,
        requested_bytes=requested_bytes,
        device_total=device_total,
        device_free=device_free,
        live_bytes=live,
        reserved_bytes=reserved,
        at=time.monotonic(),
    )


def _recent_failed_allocation() -> _FailedAllocation | None:
    """The last failed PyTorch allocation, if recent; consumed by the call."""
    global _last_failed_allocation
    failed, _last_failed_allocation = _last_failed_allocation, None
    if failed is None or time.monotonic() - failed.at > _FAILED_ALLOCATION_WINDOW_S:
        return None
    return failed


def _device_nearly_full() -> bool:
    """Whether the reporting device has almost no free memory left."""
    try:
        if _device is None:
            free, total = torch.cuda.mem_get_info()
        else:
            free, total = torch.cuda.mem_get_info(_device)
    except RuntimeError:
        return False
    return free < max(_NEARLY_FULL_BYTES, _NEARLY_FULL_FRACTION * total)


def is_device_oom_message(message: str) -> bool:
    """Whether an error message describes a failed device allocation."""
    if _HOST_MEMORY_PATTERN.search(message):
        return False
    if _OOM_PATTERN.search(message):
        return True
    return bool(_MAYBE_OOM_PATTERN.search(message)) and _device_nearly_full()


def notify_error(message: str | None) -> None:
    """Report if ``message`` describes a failed device allocation. Never raises.

    Called with errors that escaped their caller: the executor's error
    handler, logged ERROR messages and uncaught thread exceptions. Cheap
    when reporting is not armed or the message is unrelated.
    """
    try:
        if not _enabled_flag or _suspended or not message:
            return
        if getattr(_reporting, "active", False) or not is_device_oom_message(message):
            return
        first_line = message.strip().splitlines()[0][:200]
        failed = _recent_failed_allocation()
        if failed is not None:
            headline = f"OUT OF MEMORY: {failed.describe()}. Error: {first_line}"
        else:
            headline = f"OUT OF MEMORY: {first_line}"
        _report_oom(headline)
    except Exception:  # an error path must not gain a second error
        pass


class _ErrorLogFilter(logging.Filter):
    """Passes ERROR messages to :func:`notify_error`; filters nothing out."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            notify_error(record.getMessage())
        return True


_log_filter = _ErrorLogFilter()
_previous_thread_hook = None


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    try:
        notify_error(f"{getattr(args.exc_type, '__name__', 'Error')}: {args.exc_value}")
    finally:
        (_previous_thread_hook or threading.__excepthook__)(args)


def _install_error_hooks() -> None:
    global _previous_thread_hook
    py_logger = logger._logger
    if _log_filter not in py_logger.filters:
        py_logger.addFilter(_log_filter)
    if threading.excepthook is not _thread_excepthook:
        _previous_thread_hook = threading.excepthook
        threading.excepthook = _thread_excepthook


def _remove_error_hooks() -> None:
    global _previous_thread_hook
    logger._logger.removeFilter(_log_filter)
    if threading.excepthook is _thread_excepthook:
        threading.excepthook = _previous_thread_hook or threading.__excepthook__
        _previous_thread_hook = None


class _PeriodicReporter:
    """Reports every ``interval`` seconds from a daemon thread until stopped."""

    def __init__(self, interval: float, rank: int | None, device: int | None) -> None:
        self._stop = threading.Event()
        self._interval = interval
        self._rank = rank
        self._device = device
        self._thread = threading.Thread(
            target=self._run, name="memory-monitor-periodic", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            report = report_now(tag="periodic", rank=self._rank, device_index=self._device)
            if report is not None:
                logger.info(f"memory monitor:\n{report}")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            logger.warning("memory monitor: the periodic reporter did not stop within 5 s")


_periodic: _PeriodicReporter | None = None


def _start_periodic(rank: int | None, device: int | None) -> None:
    """(Re)start the periodic reporter on ``device``, if an interval is set."""
    global _periodic
    _stop_periodic()
    interval = _periodic_interval_s()
    if interval > 0:
        _periodic = _PeriodicReporter(interval, rank, device)


def _stop_periodic() -> None:
    global _periodic
    if _periodic is not None:
        _periodic.stop()
        _periodic = None


@contextlib.contextmanager
def suspend_reporting() -> Iterator[None]:
    """No OOM report while this is open, in any thread.

    Opened around executor creation, which deliberately runs allocations
    out of memory and catches the error. Process-wide: an executor already
    serving in the same process is not reported on meanwhile either.
    """
    global _suspended
    with _rate_lock:
        _suspended += 1
    try:
        yield
    finally:
        with _rate_lock:
            _suspended -= 1


def install_oom_reporter(rank: int | None = None, device: int | None = None) -> bool:
    """Arm OOM reporting for an executor that has finished starting.

    See the module docstring for why arming earlier is wrong. ``device`` is
    the executor's device, by default the calling thread's current one;
    reports describe it. Returns whether reporting is armed. Each successful
    call must be paired with :func:`release_oom_reporter`. The callback is
    attached to torch at most once per process, because torch keeps every
    observer it is handed and cannot remove one.
    """
    global _attached, _enabled_flag, _armed_count, _device

    from . import full_mode, oom_reports_enabled

    _check_settings()
    if not oom_reports_enabled(rank):
        return False

    if device is None and torch.cuda.is_available():
        device = torch.cuda.current_device()
    _device = device
    _armed_count += 1
    # Armed even if torch's observer cannot be attached: errors from outside
    # the PyTorch allocator still arrive through notify_error.
    _enabled_flag = True
    _install_error_hooks()
    if full_mode(rank):
        _start_periodic(rank, device)
    if _attached:
        return True

    attach = getattr(torch._C, "_cuda_attach_out_of_memory_observer", None)
    if attach is None:
        logger.debug("memory monitor: this torch has no OOM observer")
        return True

    try:
        attach(_on_out_of_memory)
    except RuntimeError as exc:
        logger.debug(f"memory monitor: could not attach the OOM observer ({exc})")
        return True

    _attached = True
    return True


def release_oom_reporter() -> None:
    """Release one :func:`install_oom_reporter`; the last release disarms."""
    global _armed_count
    if _armed_count <= 0:
        return
    _armed_count -= 1
    if _armed_count == 0:
        uninstall_oom_reporter()


def uninstall_oom_reporter() -> None:
    """Stop OOM reporting regardless of how many executors armed it.

    torch offers no way to detach an observer, so the callback stays
    attached and is made inert instead. ``_attached`` deliberately survives:
    re-installing must not attach a second copy.
    """
    global _enabled_flag, _last_report_at, _armed_count, _device, _last_failed_allocation
    _enabled_flag = False
    _armed_count = 0
    _device = None
    _last_report_at = None
    _last_failed_allocation = None
    _remove_error_hooks()
    _stop_periodic()
