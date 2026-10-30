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

Two triggers fire there. Startup ends with one breakdown -- tagged
``startup_failed`` if startup raised, which includes an unhandled OOM while
loading. After that, out of memory reports, which is the moment the
breakdown answers a question somebody is actually asking.

The OOM observer is only armed once startup is over. KV cache sizing,
warmup and graph capture deliberately run allocations into the wall and
catch the error -- the flashinfer planner, for one, notes that estimation
forwards "run with device memory deliberately exhausted". torch's observer
fires before anyone catches, so arming it earlier reports every one of
those as if it were a failure.

torch's observer only sees the PyTorch allocator. Allocations made in C++ --
cuBLAS workspaces, NCCL buffers, TensorRT-LLM's own pools -- fail with a
plain error that the executor catches per iteration, so the executor's error
handler also passes its message through :func:`notify_error`, which reports
when the message describes an out-of-memory condition. Both paths share one
rate limit, so a torch OOM that the executor then handles is reported once.

Errors that never reach the executor's handler are caught two more ways
once reporting is armed: TensorRT-LLM's logger passes every ERROR message
through :func:`notify_error` -- background threads such as the KV transfer
workers catch their failures and log them -- and a thread that dies of an
uncaught out-of-memory error is reported through ``threading.excepthook``.

With ``TLLM_MEM_MONITOR_INTERVAL`` set to a number of seconds, a background
thread also reports at that interval, so memory that grows slowly while
serving shows up before it runs out. It only wakes on its timer; nothing is
added to the executor loop.

With ``TLLM_MEM_MONITOR_DIR`` set, every report is also written to its own
file there, named by rank, process and time. Logs from several ranks
interleave and are often cut short when a job dies; a file per report is
what survives to be read.

The handler runs while an allocation is failing, so it allocates no device
memory and swallows every exception. Reports are limited by time rather
than by count: a retry loop that OOMs repeatedly collapses to one report
per interval, while an unrelated OOM hours later is still reported.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "INTERVAL_VAR",
    "install_oom_reporter",
    "notify_error",
    "report_now",
    "uninstall_oom_reporter",
]

#: Directory every report is also written to, one file each.
DIR_VAR = "TLLM_MEM_MONITOR_DIR"

#: Seconds between periodic reports while serving; unset or ``0`` for none.
INTERVAL_VAR = "TLLM_MEM_MONITOR_INTERVAL"

#: Minimum seconds between two OOM reports. ``0`` reports every one.
_INTERVAL_VAR = "TLLM_MEM_MONITOR_OOM_INTERVAL"
_DEFAULT_INTERVAL_S = 60.0

#: Whether the callback is attached to torch. Never cleared: torch keeps
#: every observer it is handed and offers no way to remove one, so a second
#: attach would double every report.
_attached = False

#: Whether OOM reporting is armed. Independent of ``_attached``: errors
#: passed to :func:`notify_error` are reported even where torch offers no
#: observer.
_enabled_flag = False
_last_report_at: float | None = None
_rate_lock = threading.Lock()
#: Set on a thread while it builds a report, so the report's own logging
#: cannot trigger another.
_reporting = threading.local()

#: Messages from CUDA, the driver, NCCL and TensorRT-LLM's C++ code that mean
#: an allocation failed.
_OOM_PATTERN = re.compile(
    r"out of memory|OUT_OF_MEMORY|cudaErrorMemoryAllocation|OutOfMemoryError",
    re.IGNORECASE,
)


def _interval_s() -> float:
    raw = os.environ.get(_INTERVAL_VAR, "")
    if not raw:
        return _DEFAULT_INTERVAL_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_INTERVAL_S


def report_now(
    tag: str,
    rank: int | None = None,
    *,
    headline: str = "",
    record_baseline: bool = False,
    suggestion: str = "",
    startup_steps: bool = False,
    force: bool = False,
) -> str | None:
    """Render the monitor for this process, or ``None`` if it is disabled.

    ``force`` renders it even when disabled; see :func:`capture`.

    Returns the text without logging it; the caller decides where it goes
    (the log, an exception message, a crash dump). It is also written to
    ``TLLM_MEM_MONITOR_DIR`` when that is set.
    ``record_baseline`` makes this reading the one later reports compare
    against.
    """
    from . import capture, render_capture

    try:
        snapshot = capture(
            tag=tag,
            rank=rank,
            headline=headline,
            record_baseline=record_baseline,
            suggestion=suggestion,
            startup_steps=startup_steps,
            force=force,
        )
    except Exception as exc:  # noqa: BLE001 - a report must not add a failure
        logger.warning(f"memory monitor: capture for {tag!r} failed ({type(exc).__name__}: {exc})")
        return None

    if snapshot is None:
        return None

    try:
        text = render_capture(snapshot)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"memory monitor: rendering {tag!r} failed ({type(exc).__name__}: {exc})")
        return None
    _save(text, tag, snapshot.rank)
    return text


def _save(text: str, tag: str, rank: int | None) -> str | None:
    """Write a report to ``TLLM_MEM_MONITOR_DIR``; the path, or ``None``."""
    directory = os.environ.get(DIR_VAR, "")
    if not directory:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
    rank_part = "x" if rank is None else str(rank)
    path = os.path.join(directory, f"memory_rank{rank_part}_pid{os.getpid()}_{stamp}_{tag}.txt")
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            handle.write(text + "\n")
    except OSError as exc:
        logger.warning(f"memory monitor: could not write {path} ({exc})")
        return None
    return path


def _report_oom(headline: str) -> None:
    """Emit one OOM report, subject to the shared rate limit. Never raises."""
    global _last_report_at

    if not _enabled_flag or getattr(_reporting, "active", False):
        return
    # Background threads report too, so two may race for the same OOM.
    with _rate_lock:
        now = time.monotonic()
        if _last_report_at is not None and now - _last_report_at < _interval_s():
            return
        _last_report_at = now

    _reporting.active = True
    try:
        report = report_now(tag="out_of_memory", headline=headline)
        # WARNING rather than ERROR: the caller may yet handle the OOM, and
        # the error itself is what reports the failure.
        logger.warning(
            f"memory monitor: {headline}" if report is None else f"memory monitor:\n{report}"
        )
    except Exception:  # noqa: BLE001 - never add a second failure to an OOM
        pass
    finally:
        _reporting.active = False


def _on_out_of_memory(device: int, allocation_bytes: int, allocated: int, reserved: int) -> None:
    # Fires before anyone has had a chance to catch the error, and some
    # callers do catch it and fall back; the wording leaves room for that.
    _report_oom(
        f"OUT OF MEMORY: PyTorch could not allocate {allocation_bytes / (1 << 30):.2f} GiB "
        f"on device {device} (the caller may still recover)."
    )


def notify_error(message: str | None) -> None:
    """Report if ``message`` describes an out-of-memory error. Never raises.

    For errors raised outside the PyTorch allocator, which torch's observer
    does not see. Cheap when the monitor is off or the message is unrelated.
    """
    try:
        if not _enabled_flag or not message or not _OOM_PATTERN.search(message):
            return
        first_line = message.strip().splitlines()[0][:200]
        _report_oom(f"OUT OF MEMORY reported by the executor: {first_line}")
    except Exception:  # noqa: BLE001
        pass


class _ErrorLogFilter(logging.Filter):
    """Passes ERROR messages to :func:`notify_error`; filters nothing out."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            try:
                notify_error(record.getMessage())
            except Exception:  # noqa: BLE001 - logging must not fail
                pass
        return True


_log_filter = _ErrorLogFilter()
_previous_thread_hook = None


def _thread_excepthook(args) -> None:
    try:
        notify_error(f"{getattr(args.exc_type, '__name__', 'Error')}: {args.exc_value}")
    finally:
        (_previous_thread_hook or threading.__excepthook__)(args)


def _install_error_hooks() -> None:
    global _previous_thread_hook
    py_logger = getattr(logger, "_logger", None)
    if py_logger is not None and _log_filter not in py_logger.filters:
        py_logger.addFilter(_log_filter)
    if threading.excepthook is not _thread_excepthook:
        _previous_thread_hook = threading.excepthook
        threading.excepthook = _thread_excepthook


def _remove_error_hooks() -> None:
    global _previous_thread_hook
    py_logger = getattr(logger, "_logger", None)
    if py_logger is not None:
        py_logger.removeFilter(_log_filter)
    if threading.excepthook is _thread_excepthook:
        threading.excepthook = _previous_thread_hook or threading.__excepthook__
        _previous_thread_hook = None


class _PeriodicReporter:
    """Reports every ``interval`` seconds from a daemon thread until stopped."""

    def __init__(self, interval: float, rank: int | None) -> None:
        self._stop = threading.Event()
        self._interval = interval
        self._rank = rank
        # A new thread starts on device 0; report on the one being served.
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._thread = threading.Thread(
            target=self._run, name="memory-monitor-periodic", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        if self._device is not None:
            torch.cuda.set_device(self._device)
        while not self._stop.wait(self._interval):
            report = report_now(tag="periodic", rank=self._rank)
            if report is not None:
                logger.info(f"memory monitor:\n{report}")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


_periodic: _PeriodicReporter | None = None


def _periodic_interval_s() -> float:
    try:
        return max(0.0, float(os.environ.get(INTERVAL_VAR, "") or 0))
    except ValueError:
        logger.warning(f"memory monitor: ignoring {INTERVAL_VAR}, not a number of seconds")
        return 0.0


def _start_periodic(rank: int | None) -> None:
    global _periodic
    interval = _periodic_interval_s()
    if interval > 0 and _periodic is None:
        _periodic = _PeriodicReporter(interval, rank)


def _stop_periodic() -> None:
    global _periodic
    if _periodic is not None:
        _periodic.stop()
        _periodic = None


def install_oom_reporter(rank: int | None = None) -> bool:
    """Report the monitor the next time an allocation runs out of memory.

    Call once startup is over; see the module docstring for why earlier is
    wrong. Returns whether reporting is active. Safe to call repeatedly: the
    callback is attached to torch at most once per process, because torch
    keeps every observer it is handed and cannot remove one, so attaching
    again would duplicate every report.
    """
    global _attached, _enabled_flag

    from . import enabled

    if not enabled(rank):
        return False

    # Armed even if torch's observer cannot be attached: errors from outside
    # the PyTorch allocator still arrive through notify_error.
    _enabled_flag = True
    _install_error_hooks()
    _start_periodic(rank)
    if _attached:
        return True

    attach = getattr(torch._C, "_cuda_attach_out_of_memory_observer", None)
    if attach is None:
        logger.debug("memory monitor: this torch has no OOM observer")
        return True

    try:
        attach(_on_out_of_memory)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            f"memory monitor: could not attach the OOM observer ({type(exc).__name__}: {exc})"
        )
        return True

    _attached = True
    return True


def uninstall_oom_reporter() -> None:
    """Stop reporting on OOM.

    torch offers no way to detach an observer, so the callback stays
    attached and is made inert instead. ``_attached`` deliberately survives:
    re-installing must not attach a second copy.
    """
    global _enabled_flag, _last_report_at
    _enabled_flag = False
    _last_report_at = None
    _remove_error_hooks()
    _stop_periodic()
