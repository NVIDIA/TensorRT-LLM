# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common decode reporting policy with explicit target-owned state and callbacks."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Callable, Optional, Protocol

import torch


class _Logger(Protocol):
    info: Callable[[str], None]
    warning: Callable[[str], None]


class ReportingTarget(Protocol):
    """Each target keeps its own flags, lock, logged keys, exception and callbacks."""

    _require: list[bool]
    _lock: AbstractContextManager[bool]
    _logged: set
    _log_once: Callable[[Callable[[str], None], str, str, str], None]
    missed: Callable[[str, Optional[int], str], None]
    logger: _Logger
    mpi_rank: Callable[[], int]
    FastPathUnavailable: type[RuntimeError]


def set_require(state: ReportingTarget, /, require: bool) -> None:
    """Whether a miss raises (``modeling_v2='require'``); set by the target at construction."""
    state._require[0] = bool(require)


def require_mode(state: ReportingTarget, /) -> bool:
    return state._require[0]


def _log_once(
    state: ReportingTarget, /, level: Callable[[str], None], path: str, reason: str, message: str
) -> None:
    """Log eager diagnostics once; compiled regions do not lock or log."""
    if torch.compiler.is_compiling():
        return
    key = (path, reason)
    with state._lock:
        if key in state._logged:
            return
        state._logged.add(key)
    level(message)


def missed(state: ReportingTarget, /, path: str, rows: Optional[int], reason: str) -> None:
    """Raise under require, including during tracing; otherwise report only on eager steps."""
    if not state._require[0] and torch.compiler.is_compiling():
        return
    at = "" if rows is None else f" at {rows} rows"
    if state._require[0]:
        raise state.FastPathUnavailable(
            f"modeling_v2='require': the MiniMax-M3 decode path {path!r} covers this step{at} but cannot run: {reason}"
        )
    state._log_once(
        state.logger.warning,
        path,
        reason,
        f"[M3 dispatch] {path}{at}: {reason}; falling back (logged once per reason)",
    )


def capture_missed(state: ReportingTarget, /, path: str, rows: Optional[int], what: str) -> None:
    """``what`` is built on eager steps but a CUDA-graph capture found it missing (see ``missed``)."""
    state.missed(path, rows, f"{what} is not built and a CUDA graph is being captured")


def taken(state: ReportingTarget, /, path: str, rows: Optional[int], what: str = "") -> None:
    """Report eager engagement once per (path, rows, what) on each rank.

    Compiled regions do not report. ``what`` is an optional detail such as the projection.
    """
    if torch.compiler.is_compiling():
        return
    at = "" if rows is None else f" at {rows} rows"
    detail = f" ({what})" if what else ""
    state._log_once(
        state.logger.info,
        path,
        f"taken: {rows}" + (f": {what}" if what else ""),
        f"[M3 dispatch] rank {state.mpi_rank()}: {path} taken{at}{detail}",
    )


def uncovered(state: ReportingTarget, /, path: str, reason: str) -> None:
    """The step is outside ``path``'s covered shapes: log once per (path, reason); the generic path runs."""
    state._log_once(
        state.logger.info, path, reason, f"[M3 dispatch] {path} not taken: {reason} (logged once)"
    )


def reset_logged(state: ReportingTarget, /) -> None:
    """Forget which (path, reason) pairs were logged."""
    with state._lock:
        state._logged.clear()
