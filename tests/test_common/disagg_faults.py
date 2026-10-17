# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scripted faults at asynchronous test boundaries, without background tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Optional, TypeVar

_T = TypeVar("_T")


class InjectedFault(RuntimeError):
    """Failure deliberately injected by a test scenario."""


@dataclass(frozen=True)
class Fault:
    """A one-shot fault on the nth call to a boundary for a particular key.

    A delay waits for an explicit gate release; it does not simulate elapsed
    time. A drop skips the operation and returns None. A raise fails before
    calling the operation. Keys can identify requests, workers or storage keys.
    """

    boundary: str
    key: str
    action: Literal["delay", "drop", "raise"]
    occurrence: int = 1
    gate: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.boundary or not self.key:
            raise ValueError("Fault boundary and key must be nonempty.")
        if self.action not in ("delay", "drop", "raise"):
            raise ValueError(f"Unknown fault action: {self.action!r}.")
        if isinstance(self.occurrence, bool) or not isinstance(self.occurrence, int):
            raise ValueError("Fault occurrence must be a positive integer.")
        if self.occurrence < 1:
            raise ValueError("Fault occurrence must be a positive integer.")
        if self.action == "delay" and not self.gate:
            raise ValueError("A delay fault requires a named gate.")
        if self.action != "delay" and self.gate is not None:
            raise ValueError("Only a delay fault can have a gate.")


@dataclass(frozen=True)
class TraceEvent:
    """One boundary transition; no payloads or wall-clock timestamps are retained."""

    boundary: str
    key: str
    occurrence: int
    event: str


class FaultScenario:
    """Apply faults to caller-owned coroutines at explicit test boundaries.

    Pass a coroutine factory, not a created coroutine: dropped operations must
    never start. The caller owns and joins tasks created by its test. This
    helper creates no tasks, modifies no global state and patches no clocks.
    """

    def __init__(self, faults: Sequence[Fault] = ()) -> None:
        self._faults: dict[tuple[str, str, int], Fault] = {}
        self._gates: dict[str, tuple[asyncio.Event, asyncio.Event]] = {}
        for fault in faults:
            coordinate = (fault.boundary, fault.key, fault.occurrence)
            if coordinate in self._faults:
                raise ValueError(f"Multiple faults target {coordinate!r}.")
            self._faults[coordinate] = fault
            if fault.gate is not None:
                if fault.gate in self._gates:
                    raise ValueError(f"Duplicate gate name: {fault.gate!r}.")
                self._gates[fault.gate] = (asyncio.Event(), asyncio.Event())
        self._counts: dict[tuple[str, str], int] = {}
        self._consumed: set[tuple[str, str, int]] = set()
        self._active: set[tuple[str, str, int]] = set()
        self.trace: list[TraceEvent] = []

    async def wait_until_blocked(self, gate: str) -> None:
        """Wait until the selected operation reaches its delay gate."""
        reached, _ = self._gates[gate]
        await reached.wait()

    def release(self, gate: str) -> None:
        """Release an operation held at a named delay gate."""
        _, released = self._gates[gate]
        released.set()

    async def run(
        self, boundary: str, key: str, operation: Callable[[], Awaitable[_T]]
    ) -> Optional[_T]:
        """Run one operation, or skip it when its scripted action is a drop."""
        counter_key = (boundary, key)
        occurrence = self._counts.get(counter_key, 0) + 1
        self._counts[counter_key] = occurrence
        coordinate = (boundary, key, occurrence)
        self._active.add(coordinate)

        def record(event: str) -> None:
            self.trace.append(TraceEvent(boundary, key, occurrence, event))

        record("entered")
        try:
            fault = self._faults.get(coordinate)
            if fault is not None:
                self._consumed.add(coordinate)
                if fault.action == "drop":
                    record("dropped")
                    return None
                if fault.action == "raise":
                    record("injected_failure")
                    raise InjectedFault(f"Injected failure at {boundary!r} for {key!r}.")
                assert fault.gate is not None
                reached, released = self._gates[fault.gate]
                record("blocked")
                reached.set()
                await released.wait()
                record("released")
            result = await operation()
            record("completed")
            return result
        except asyncio.CancelledError:
            record("cancelled")
            raise
        except Exception:
            record("failed")
            raise
        finally:
            self._active.remove(coordinate)

    def assert_consumed(self) -> None:
        """Fail if a configured fault was never reached by the test."""
        unused = self._faults.keys() - self._consumed
        assert not unused, f"Unconsumed faults: {sorted(unused)!r}"

    def assert_idle(self) -> None:
        """Fail if an operation is still executing or blocked at a gate."""
        assert not self._active, f"Active boundary calls: {sorted(self._active)!r}"
