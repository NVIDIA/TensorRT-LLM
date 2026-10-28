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
"""When the ledger reports.

The memory lives in the executor's process, not the one that created the
``LLM``. A report taken from the parent describes the parent -- its CUDA
context and nothing else -- so every trigger here has to fire inside the
worker.

Out of memory is the trigger worth having: it is the moment the breakdown
answers a question somebody is actually asking. The handler runs while an
allocation is failing, so it allocates no device memory, swallows every
exception, and reports at most once per process by default -- a retry loop
that OOMs repeatedly would otherwise bury the first and most informative
report under identical copies.
"""

from __future__ import annotations

import os

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "install_oom_reporter",
    "report_now",
    "uninstall_oom_reporter",
]

#: Set to an integer to allow more than one OOM report per process.
_MAX_REPORTS_VAR = "TLLM_MEM_LEDGER_OOM_REPORTS"

#: Whether the callback is attached to torch. Never cleared: torch keeps
#: every observer it is handed and offers no way to remove one, so a second
#: attach would double every report.
_attached = False

_enabled_flag = False
_reports_emitted = 0


def _max_reports() -> int:
    raw = os.environ.get(_MAX_REPORTS_VAR, "")
    if not raw:
        return 1
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def report_now(tag: str, rank: int | None = None) -> str | None:
    """Render the ledger for this process, or ``None`` if it is disabled.

    Returns the text as well as logging it so a caller can attach it to an
    exception or a crash dump rather than hoping the log survives.
    """
    from . import capture, render_capture

    try:
        snapshot = capture(tag=tag, rank=rank)
    except Exception as exc:  # noqa: BLE001 - a report must not add a failure
        logger.warning(f"memory ledger: capture for {tag!r} failed ({type(exc).__name__}: {exc})")
        return None

    if snapshot is None:
        return None

    try:
        return render_capture(snapshot)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"memory ledger: rendering {tag!r} failed ({type(exc).__name__}: {exc})")
        return None


def _on_out_of_memory(device: int, allocation_bytes: int, allocated: int, reserved: int) -> None:
    global _reports_emitted

    if not _enabled_flag or _reports_emitted >= _max_reports():
        return
    _reports_emitted += 1

    gib = 1 << 30
    try:
        header = (
            f"memory ledger: out of memory on device {device} -- "
            f"request {allocation_bytes / gib:.2f} GiB, "
            f"torch allocated {allocated / gib:.2f} GiB, "
            f"torch reserved {reserved / gib:.2f} GiB"
        )
        report = report_now(tag="out_of_memory")
        logger.error(header if report is None else f"{header}\n{report}")
    except Exception:  # noqa: BLE001 - never add a second failure to an OOM
        pass


def install_oom_reporter(rank: int | None = None) -> bool:
    """Report the ledger the next time an allocation runs out of memory.

    Returns whether reporting is active. Safe to call repeatedly: the
    callback is attached to torch at most once per process, because torch
    keeps every observer it is handed and cannot remove one, so attaching
    again would duplicate every report.
    """
    global _attached, _enabled_flag

    from . import enabled

    if not enabled(rank):
        return False

    if _attached:
        _enabled_flag = True
        return True

    attach = getattr(torch._C, "_cuda_attach_out_of_memory_observer", None)
    if attach is None:
        logger.debug("memory ledger: this torch has no OOM observer")
        return False

    try:
        attach(_on_out_of_memory)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            f"memory ledger: could not attach the OOM observer ({type(exc).__name__}: {exc})"
        )
        return False

    _attached = True
    _enabled_flag = True
    return True


def uninstall_oom_reporter() -> None:
    """Stop reporting on OOM.

    torch offers no way to detach an observer, so the callback stays
    attached and is made inert instead. ``_attached`` deliberately survives:
    re-installing must not attach a second copy.
    """
    global _enabled_flag, _reports_emitted
    _enabled_flag = False
    _reports_emitted = 0
