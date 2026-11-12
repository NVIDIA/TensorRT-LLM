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
``threading.excepthook``. The recorded allocation goes into the report's
headline only when the error is that failure -- PyTorch's message names the
same size on the same device: how much was asked for and what PyTorch held
at that moment, before the failing step's memory was released. Any other
error, such as a native allocation failing after PyTorch recovered from its
own, is described by its own first line.

The same path covers allocations made outside the PyTorch allocator --
cuBLAS workspaces, NCCL buffers, TensorRT-LLM's own pools -- which torch's
observer never sees. :func:`notify_error` recognises their messages; NCCL's
generic CUDA error is taken for an out-of-memory error only when the device
is nearly full. Host-memory failures are not GPU memory and are ignored.

Reporting is suspended while an executor is being created
(:func:`suspend_reporting`), since creation runs allocations into the wall on
purpose, and every executor that armed reporting releases it when it shuts
down (:func:`release_oom_reporter`). Reports describe the device of the most
recently armed executor still alive, and the periodic reporter follows it.

In full mode with ``periodic_report_interval`` (``TLLM_MEM_MONITOR_INTERVAL``)
set to a number of seconds, a background thread also reports at that interval, so
memory that grows slowly while serving shows up before it runs out. It only
wakes on its timer; nothing is added to the executor loop.

With ``report_dir`` (``TLLM_MEM_MONITOR_DIR``) set, every report except on-demand ones is
also written to its own file there, named by rank, process, time and tag.
The directory is created readable by its owner only (``0o700``) and each
file likewise (``0o600``); only the newest :data:`MAX_PERIODIC_FILES`
periodic files of a rank and process are kept.

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

from .capability import tagging_supported
from .capture import capture, render_capture
from .schema import GIB
from .settings import (
    DIR_VAR,
    FULL,
    OOM_INTERVAL_VAR,
    PERIODIC_INTERVAL_VAR,
    STACKS_ENV_VAR,
    check_rank_filter,
    full_mode,
    mode,
    oom_reports_enabled,
    seconds_setting,
    selected_ranks,
    setting,
    stacks_requested,
    warn_once,
)

__all__ = [
    "DIR_VAR",
    "MAX_PERIODIC_FILES",
    "OOM_INTERVAL_VAR",
    "PERIODIC_INTERVAL_VAR",
    "install_oom_reporter",
    "is_device_oom_message",
    "notify_error",
    "release_oom_reporter",
    "report_now",
    "suspend_reporting",
    "uninstall_oom_reporter",
]

#: Shortest periodic interval; each report takes an allocator snapshot.
MIN_PERIODIC_INTERVAL_S = 1.0

#: Periodic report files kept per rank and process in ``TLLM_MEM_MONITOR_DIR``;
#: older ones are deleted as new ones are written.
MAX_PERIODIC_FILES = 100

#: Minimum seconds between two OOM reports when none is configured.
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


@dataclass(frozen=True)
class _Armed:
    """An executor that armed reporting and has not released it."""

    owner: object | None
    rank: int | None
    device: int | None


#: Executors that armed reporting, oldest first.
_armed: list[_Armed] = []
#: Device reports are built on: the most recently armed live executor's.
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
#: The thread building a report for a logged error, while it runs.
_background_report: threading.Thread | None = None
#: How long disarming waits for a report still being built for a logged error.
_BACKGROUND_REPORT_JOIN_S = 30.0

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
_NEARLY_FULL_BYTES = GIB
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
    #: What PyTorch may hold on the device: the device total, or the
    #: per-process limit once ``set_per_process_memory_fraction`` set one.
    pytorch_limit: int
    device_free: int
    live_bytes: int | None
    reserved_bytes: int | None
    at: float

    def describe(self) -> str:
        held = ""
        if self.live_bytes is not None and self.reserved_bytes is not None:
            held = (
                f"; PyTorch held {self.live_bytes / GIB:.2f} GiB live, "
                f"{self.reserved_bytes / GIB:.2f} GiB reserved"
            )
        return (
            f"PyTorch could not allocate {self.requested_bytes / GIB:.2f} GiB on device "
            f"{self.device} (device free {self.device_free / GIB:.2f} GiB; PyTorch may use up "
            f"to {self.pytorch_limit / GIB:.2f} GiB{held})"
        )

    def is_reported_by(self, message: str) -> bool:
        """Whether ``message`` is PyTorch's error for this failed allocation.

        PyTorch's message names the size with two decimals in the largest
        fitting unit and the device by index, e.g. "Tried to allocate
        20.00 MiB. GPU 0"; the size matches within that rounding.
        """
        match = _TRIED_TO_ALLOCATE.search(message)
        if match is None or int(match.group("device")) != self.device:
            return False
        unit = _SIZE_UNITS[match.group("unit")]
        stated = float(match.group("size")) * unit
        return abs(stated - self.requested_bytes) <= 0.005 * unit + 1


#: PyTorch's "CUDA out of memory" message: the size it tried and the device.
_TRIED_TO_ALLOCATE = re.compile(
    r"Tried to allocate (?P<size>\d+(?:\.\d+)?) (?P<unit>bytes|KiB|MiB|GiB)\. "
    r"GPU (?P<device>\d+)"
)
_SIZE_UNITS: dict[str, int] = {"bytes": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": GIB}

_last_failed_allocation: _FailedAllocation | None = None


def _oom_interval_s() -> float:
    return seconds_setting(OOM_INTERVAL_VAR, _DEFAULT_OOM_INTERVAL_S)


def _periodic_interval_s() -> float:
    """Seconds between periodic reports; ``0`` for none."""
    interval = seconds_setting(PERIODIC_INTERVAL_VAR, 0.0)
    if 0 < interval < MIN_PERIODIC_INTERVAL_S:
        warn_once(
            PERIODIC_INTERVAL_VAR,
            setting(PERIODIC_INTERVAL_VAR) or "",
            f"is below the minimum of {MIN_PERIODIC_INTERVAL_S:g} s; using the minimum",
        )
        return MIN_PERIODIC_INTERVAL_S
    return interval


def _check_settings(world_size: int | None) -> None:
    """Read every setting once, so an invalid or inert one is reported when arming.

    A periodic interval and allocation stacks act only in full mode; asking
    for them in another mode, or for stacks where the allocator drops the
    tags they are stored in, is reported rather than silently ignored.
    """
    current = mode()
    selected_ranks()
    if world_size is not None:
        check_rank_filter(world_size)
    stacks = stacks_requested()
    _oom_interval_s()
    interval = _periodic_interval_s()
    if current != FULL:
        if interval > 0:
            warn_once(
                PERIODIC_INTERVAL_VAR,
                setting(PERIODIC_INTERVAL_VAR) or "",
                f"takes effect only in full mode, and the mode is {current}; "
                "no periodic report is taken",
            )
        if stacks:
            warn_once(
                STACKS_ENV_VAR,
                setting(STACKS_ENV_VAR) or "",
                f"takes effect only in full mode, and the mode is {current}; "
                "stacks are not recorded",
            )
    elif stacks and not tagging_supported():
        warn_once(
            STACKS_ENV_VAR,
            setting(STACKS_ENV_VAR) or "",
            "needs an allocator that keeps allocation metadata, and this one does "
            "not; stacks are not recorded",
        )


def report_now(
    tag: str,
    rank: int | None = None,
    *,
    headline: str = "",
    record_baseline: bool = False,
    suggestion: str = "",
    advise: bool = True,
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
    makes this reading the one later reports compare against. Without
    ``advise`` the report gives no tuning advice, for a failure that tuning
    memory would not fix.
    ``device_index`` is the device to describe, by default the calling
    thread's current one; the report is built with it selected.
    """
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
                advise=advise,
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
#: Tag of the reports :data:`MAX_PERIODIC_FILES` applies to.
_PERIODIC_TAG = "periodic"


def _save(text: str, tag: str, rank: int | None) -> str | None:
    """Write a report to ``TLLM_MEM_MONITOR_DIR``; the path, or ``None``.

    The directory is created ``0o700`` and the file ``0o600``: a report
    names the job's layout and sizes, and the directory may be shared.
    Periodic reports keep only the newest :data:`MAX_PERIODIC_FILES` per
    rank and process. A file that cannot be written is reported once per
    directory and never raises.
    """
    directory = (setting(DIR_VAR) or "").strip()
    if not directory:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
    rank_part = "x" if rank is None else str(rank)
    safe_tag = _UNSAFE_FILE_CHARS.sub("_", tag) or "report"
    prefix = f"memory_rank{rank_part}_pid{os.getpid()}_"
    path = os.path.join(directory, f"{prefix}{stamp}_{safe_tag}.txt")
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(text + "\n")
        if safe_tag == _PERIODIC_TAG:
            _prune_periodic(directory, prefix)
    except (OSError, ValueError) as exc:
        # ValueError: a path with a NUL byte.
        logger.warning_once(
            f"memory monitor: {DIR_VAR}={directory!r} cannot be written to "
            f"({type(exc).__name__}: {exc}); reports are not saved there",
            key=("memory monitor: save", directory),
        )
        return None
    return path


def _prune_periodic(directory: str, prefix: str) -> None:
    """Delete this rank and process's oldest periodic files beyond the cap."""
    suffix = f"_{_PERIODIC_TAG}.txt"
    names = sorted(
        name for name in os.listdir(directory) if name.startswith(prefix) and name.endswith(suffix)
    )
    for name in names[:-MAX_PERIODIC_FILES]:
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(directory, name))


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


def _report_oom(headline: str, *, background: bool = False) -> None:
    """Emit one OOM report, subject to the shared rate limit. Never raises.

    With ``background`` the report is built on a short-lived thread, so the
    caller -- a thread logging an error -- returns at once. The rate-limit
    slot is still taken here, so a burst of errors starts one report.
    """
    global _background_report
    if not _enabled_flag or _suspended or getattr(_reporting, "active", False):
        return
    if not _claim_report_slot():
        return
    device = _device
    if not background:
        _build_oom_report(headline, device)
        return
    if device is None:
        try:
            device = torch.cuda.current_device()
        except (RuntimeError, AssertionError):
            device = None
    try:
        # Not a daemon: a report already started still reaches the log when
        # the process exits normally.
        thread = threading.Thread(
            target=_build_oom_report,
            args=(headline, device),
            name="memory-monitor-oom-report",
        )
        thread.start()
        _background_report = thread
    except RuntimeError:  # no thread can be started, e.g. at interpreter exit
        _build_oom_report(headline, device)


def _build_oom_report(headline: str, device: int | None) -> None:
    """Build and log the OOM report for a slot already claimed. Never raises."""
    produced = False
    _reporting.active = True
    try:
        # Forced: in the default mode the full monitor is off, and an armed
        # reporter is all the permission this report needs.
        report = report_now(tag="out_of_memory", headline=headline, force=True, device_index=device)
        produced = report is not None
        logger.error(f"memory monitor:\n{report}" if produced else f"memory monitor: {headline}")
    except Exception:  # never add a second failure to an OOM
        pass
    finally:
        _reporting.active = False
        _finish_report_slot(produced)


def _on_out_of_memory(
    device: int, requested_bytes: int, pytorch_limit: int, device_free: int
) -> None:
    """torch's OOM observer: record the failed allocation, report nothing.

    torch passes the device total as ``pytorch_limit``, or the per-process
    limit when one is set.

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
        pytorch_limit=pytorch_limit,
        device_free=device_free,
        live_bytes=live,
        reserved_bytes=reserved,
        at=time.monotonic(),
    )


def _failed_allocation_of(message: str) -> _FailedAllocation | None:
    """The recorded failed PyTorch allocation, if ``message`` is its error.

    Consumed when it matches. Matched by size and device rather than by
    time alone: callers catch some PyTorch OOMs and recover, and a later,
    unrelated failure must not be blamed on the allocation they handled.
    """
    global _last_failed_allocation
    failed = _last_failed_allocation
    if failed is None or time.monotonic() - failed.at > _FAILED_ALLOCATION_WINDOW_S:
        return None
    if not failed.is_reported_by(message):
        return None
    _last_failed_allocation = None
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
    """Whether an error message describes a failed device allocation.

    The monitor's one test for a GPU out-of-memory error. Host-memory
    failures are not; NCCL's generic "unhandled cuda error" is only when
    the reporting device is nearly full.
    """
    if _HOST_MEMORY_PATTERN.search(message):
        return False
    if _OOM_PATTERN.search(message):
        return True
    return bool(_MAYBE_OOM_PATTERN.search(message)) and _device_nearly_full()


def notify_error(message: str | None, *, background: bool = False) -> None:
    """Report if ``message`` describes a failed device allocation. Never raises.

    Called with errors that escaped their caller: the executor's error
    handler, logged ERROR messages and uncaught thread exceptions. Cheap
    when reporting is not armed or the message is unrelated. With
    ``background`` the report is built on its own thread; see
    :func:`_report_oom`.
    """
    try:
        if not _enabled_flag or _suspended or not message:
            return
        if getattr(_reporting, "active", False) or not is_device_oom_message(message):
            return
        first_line = message.strip().splitlines()[0][:200]
        failed = _failed_allocation_of(message)
        if failed is not None:
            headline = f"OUT OF MEMORY: {failed.describe()}. Error: {first_line}"
        else:
            headline = f"OUT OF MEMORY: {first_line}"
        _report_oom(headline, background=background)
    except Exception:  # an error path must not gain a second error
        pass


class _ErrorLogFilter(logging.Filter):
    """Passes ERROR messages to :func:`notify_error`; filters nothing out.

    The report is built on its own thread: the thread logging the error is
    often the executor's, which still has error responses to send and a
    crash watchdog to arm.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            notify_error(record.getMessage(), background=True)
        return True


_log_filter = _ErrorLogFilter()
_previous_thread_hook = None
#: Set on a thread while :func:`_thread_excepthook` runs on it.
_in_thread_hook = threading.local()


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    """Report an uncaught thread exception, then hand it to the previous hook.

    A hook installed after this one may chain back to it; re-entered on the
    same thread, it hands the exception to ``threading.__excepthook__``
    instead, which prints it, rather than recursing until ``RecursionError``.
    """
    if getattr(_in_thread_hook, "active", False):
        threading.__excepthook__(args)
        return
    _in_thread_hook.active = True
    try:
        try:
            notify_error(f"{getattr(args.exc_type, '__name__', 'Error')}: {args.exc_value}")
        finally:
            (_previous_thread_hook or threading.__excepthook__)(args)
    finally:
        _in_thread_hook.active = False


def _install_error_hooks() -> None:
    global _previous_thread_hook
    py_logger = logger.python_logger
    if _log_filter not in py_logger.filters:
        py_logger.addFilter(_log_filter)
    if threading.excepthook is not _thread_excepthook:
        _previous_thread_hook = threading.excepthook
        threading.excepthook = _thread_excepthook


def _remove_error_hooks() -> None:
    global _previous_thread_hook
    logger.python_logger.removeFilter(_log_filter)
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


def install_oom_reporter(
    rank: int | None = None,
    device: int | None = None,
    *,
    owner: object | None = None,
    world_size: int | None = None,
) -> bool:
    """Arm OOM reporting for an executor that has finished starting.

    See the module docstring for why arming earlier is wrong. ``device`` is
    the executor's device, by default the calling thread's current one;
    reports describe it. ``owner`` identifies the executor, so that
    :func:`release_oom_reporter` releases its arming and not another's.
    ``world_size`` is the job's rank count: a rank filter that selects none
    of its ranks is reported and ignored.

    Returns whether reporting is armed. Each successful call must be paired
    with :func:`release_oom_reporter`. The callback is attached to torch at
    most once per process, because torch keeps every observer it is handed
    and cannot remove one.
    """
    global _attached, _enabled_flag, _device

    _check_settings(world_size)
    if not oom_reports_enabled(rank):
        return False

    if device is None and torch.cuda.is_available():
        device = torch.cuda.current_device()
    armed = _Armed(owner=owner, rank=rank, device=device)
    _armed.append(armed)
    _device = device
    # Armed even if torch's observer cannot be attached: errors from outside
    # the PyTorch allocator still arrive through notify_error.
    _enabled_flag = True
    try:
        _install_error_hooks()
        _follow_newest_executor()
    except RuntimeError as exc:
        # A diagnostic must not fail the executor it serves: undo this arming
        # (another executor's stays) and report that it is off.
        _armed.remove(armed)
        # The thread that failed to start for this executor may fail again
        # for the previous one; that one then goes without periodic reports.
        with contextlib.suppress(RuntimeError):
            _restore_after_release()
        logger.warning(f"memory monitor: could not arm out-of-memory reporting ({exc})")
        return False
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


def _follow_newest_executor() -> None:
    """Point reports and the periodic reporter at the newest armed executor."""
    global _device
    newest = _armed[-1]
    _device = newest.device
    if full_mode(newest.rank):
        _start_periodic(newest.rank, newest.device)
    else:
        _stop_periodic()


def _restore_after_release() -> None:
    """Follow the newest executor still armed, or disarm if none is."""
    if _armed:
        _follow_newest_executor()
    else:
        uninstall_oom_reporter()


def release_oom_reporter(owner: object | None = None) -> None:
    """Release one :func:`install_oom_reporter`; the last release disarms.

    Releases the arming made with ``owner``, or the newest one when
    ``owner`` is ``None``. Reports then describe the newest executor still
    armed, and its periodic reporter takes over.
    """
    if not _armed:
        return
    index = len(_armed) - 1
    if owner is not None:
        matches = [i for i, armed in enumerate(_armed) if armed.owner is owner]
        if not matches:
            return
        index = matches[-1]
    was_newest = index == len(_armed) - 1
    del _armed[index]
    if not _armed:
        uninstall_oom_reporter()
    elif was_newest:
        _follow_newest_executor()


def uninstall_oom_reporter() -> None:
    """Stop OOM reporting regardless of how many executors armed it.

    torch offers no way to detach an observer, so the callback stays
    attached and is made inert instead. ``_attached`` deliberately survives:
    re-installing must not attach a second copy.
    """
    global _enabled_flag, _last_report_at, _device, _last_failed_allocation, _background_report
    _enabled_flag = False
    building, _background_report = _background_report, None
    if building is not None and building is not threading.current_thread():
        building.join(_BACKGROUND_REPORT_JOIN_S)
    _armed.clear()
    _device = None
    _last_report_at = None
    _last_failed_allocation = None
    _remove_error_hooks()
    _stop_periodic()
