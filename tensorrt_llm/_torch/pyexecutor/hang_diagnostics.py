# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, silent-until-failure diagnostics for PyExecutor hangs."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Protocol

from .hang_detector import HangDetector

if TYPE_CHECKING:
    import torch

HANG_DIAGNOSTICS_ENV = "TLLM_HANG_DIAGNOSTICS"
_INITIALIZATION_HANG_TIMEOUT_SECONDS = 300
_INITIALIZATION_REPORT_CONTEXT = (
    "PyExecutor initialization diagnostic (dump only; process will continue)"
)

# Sentinel default for `ExecutorHangDiagnostics.record`'s `forward_completion_event`
# so "leave the previous event untouched" is distinguishable from an explicit `None`
# ("clear it").
_KEEP_CURRENT_EVENT: Any = object()


@contextmanager
def monitor_executor_initialization() -> Iterator[None]:
    """Dump worker stacks if opt-in executor initialization stalls."""
    if os.environ.get(HANG_DIAGNOSTICS_ENV) != "1":
        yield
        return

    detector = HangDetector(
        timeout=_INITIALIZATION_HANG_TIMEOUT_SECONDS,
        report_context=_INITIALIZATION_REPORT_CONTEXT,
    )
    detector.register_status_provider(lambda: "PyExecutor initialization has not completed.")
    with detector:
        detector.checkpoint()
        yield


class _CudaEvent(Protocol):
    def query(self) -> bool: ...

    def record(self, stream: torch.cuda.Stream) -> None: ...


class _CudaEventFactory(Protocol):
    def __call__(self, *, enable_timing: bool) -> _CudaEvent: ...


class _StatusProviderRegistry(Protocol):
    def register_status_provider(self, provider: Callable[[], str]) -> None: ...


class ExecutorHangDiagnostics:
    """Keep the executor's latest phase available to the hang detector.

    The executor only constructs this object when explicitly enabled. Updates
    replace one immutable tuple, so the hang-detector thread never waits for a
    lock held by the thread it is diagnosing.
    """

    def __init__(self, rank: int) -> None:
        self._rank = rank
        self._snapshot: tuple[str, float, int, str, Optional[_CudaEvent]] = (
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
        forward_completion_event: Optional[_CudaEvent] = _KEEP_CURRENT_EVENT,
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

    @staticmethod
    def record_forward_completion_event(
        event: _CudaEvent,
        stream: torch.cuda.Stream,
    ) -> Optional[_CudaEvent]:
        """Record a CUDA event without allowing diagnostics to affect execution."""
        try:
            event.record(stream)
        except Exception:  # noqa: BLE001 - diagnostics must not affect execution
            return None
        return event

    def get_status_dump(self) -> str:
        phase, started_at, iteration, details, forward_event = self._snapshot
        phase_age = max(0.0, time.monotonic() - started_at)
        if forward_event is None:
            cuda_completion = "not-recorded"
        else:
            # The hang detector queries only after the executor has stopped
            # checkpointing, so this event is not expected to be re-recorded.
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


def create_executor_hang_diagnostics(
    rank: int,
    status_provider_registry: _StatusProviderRegistry,
    event_factory: _CudaEventFactory,
) -> tuple[Optional[ExecutorHangDiagnostics], Optional[_CudaEvent]]:
    """Create and register enabled diagnostics and their CUDA completion event."""
    diagnostics = ExecutorHangDiagnostics.from_environment(rank)
    if diagnostics is None:
        return None, None

    try:
        event = event_factory(enable_timing=False)
    except Exception:  # noqa: BLE001 - diagnostics must not affect execution
        event = None
    status_provider_registry.register_status_provider(diagnostics.get_status_dump)
    return diagnostics, event
