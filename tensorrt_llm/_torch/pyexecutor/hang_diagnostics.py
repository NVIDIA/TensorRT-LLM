# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, silent-until-failure diagnostics for PyExecutor hangs."""

import os
import time
from typing import Any, Optional, Protocol

HANG_DIAGNOSTICS_ENV = "TLLM_HANG_DIAGNOSTICS"

# Sentinel default for `ExecutorHangDiagnostics.record`'s `forward_completion_event`
# so "leave the previous event untouched" is distinguishable from an explicit `None`
# ("clear it").
_KEEP_CURRENT_EVENT: Any = object()


class _QueryEvent(Protocol):
    def query(self) -> bool: ...


class ExecutorHangDiagnostics:
    """Keep the executor's latest phase available to the hang detector.

    The executor only constructs this object when explicitly enabled. Updates
    replace one immutable tuple, so the hang-detector thread never waits for a
    lock held by the thread it is diagnosing.
    """

    def __init__(self, rank: int) -> None:
        self._rank = rank
        self._snapshot: tuple[str, float, int, str, Optional[_QueryEvent]] = (
            "initialized",
            time.monotonic(),
            -1,
            "",
            None,
        )

    @classmethod
    def from_environment(cls, rank: int) -> Optional["ExecutorHangDiagnostics"]:
        if os.environ.get(HANG_DIAGNOSTICS_ENV) != "1":
            return None
        return cls(rank)

    def record(
        self,
        phase: str,
        iteration: int,
        details: str = "",
        forward_completion_event: Optional[_QueryEvent] = _KEEP_CURRENT_EVENT,
    ) -> None:
        """Record one phase transition.

        `forward_completion_event` defaults to leaving the previously recorded
        event untouched; pass `None` explicitly to clear it, or an event to
        replace it.
        """
        if forward_completion_event is _KEEP_CURRENT_EVENT:
            forward_completion_event = self._snapshot[4]
        self._snapshot = (
            phase,
            time.monotonic(),
            iteration,
            details,
            forward_completion_event,
        )

    def get_status_dump(self) -> str:
        phase, started_at, iteration, details, forward_event = self._snapshot
        phase_age = max(0.0, time.monotonic() - started_at)
        if forward_event is None:
            cuda_completion = "not-recorded"
        else:
            try:
                cuda_completion = "complete" if forward_event.query() else "pending"
            except Exception as error:  # noqa: BLE001 - diagnostics must not mask a hang
                cuda_completion = f"query-failed({type(error).__name__}: {error})"

        status = (
            f"PyExecutor hang diagnostics: rank={self._rank}, "
            f"iteration={iteration}, phase={phase}, phase_age={phase_age:.1f}s, "
            f"forward_cuda_completion={cuda_completion}"
        )
        if details:
            status += f", {details}"
        return status
