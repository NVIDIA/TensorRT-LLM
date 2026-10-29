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

The handler runs while an allocation is failing, so it allocates no device
memory and swallows every exception. Reports are limited by time rather
than by count: a retry loop that OOMs repeatedly collapses to one report
per interval, while an unrelated OOM hours later is still reported.
"""

from __future__ import annotations

import os
import re
import time

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "install_oom_reporter",
    "notify_error",
    "report_now",
    "uninstall_oom_reporter",
]

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
) -> str | None:
    """Render the monitor for this process, or ``None`` if it is disabled.

    Returns the text as well as logging it so a caller can attach it to an
    exception or a crash dump rather than hoping the log survives.
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
        )
    except Exception as exc:  # noqa: BLE001 - a report must not add a failure
        logger.warning(f"memory monitor: capture for {tag!r} failed ({type(exc).__name__}: {exc})")
        return None

    if snapshot is None:
        return None

    try:
        return render_capture(snapshot)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"memory monitor: rendering {tag!r} failed ({type(exc).__name__}: {exc})")
        return None


def _report_oom(headline: str) -> None:
    """Emit one OOM report, subject to the shared rate limit. Never raises."""
    global _last_report_at

    if not _enabled_flag:
        return
    now = time.monotonic()
    if _last_report_at is not None and now - _last_report_at < _interval_s():
        return
    _last_report_at = now

    try:
        report = report_now(tag="out_of_memory", headline=headline)
        # WARNING rather than ERROR: the caller may yet handle the OOM, and
        # the error itself is what reports the failure.
        logger.warning(
            f"memory monitor: {headline}" if report is None else f"memory monitor:\n{report}"
        )
    except Exception:  # noqa: BLE001 - never add a second failure to an OOM
        pass


def _on_out_of_memory(device: int, allocation_bytes: int, allocated: int, reserved: int) -> None:
    _report_oom(
        f"OUT OF MEMORY: PyTorch could not allocate {allocation_bytes / (1 << 30):.2f} GiB "
        f"on device {device}."
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
