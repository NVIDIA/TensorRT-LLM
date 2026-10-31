# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep private coordination responsive independently of inference callbacks."""

import asyncio
import threading
from collections.abc import Callable

from .coordinator import CoordinationError, Coordinator


class CoordinatorReactor:
    """Own the synchronous coordinator without blocking the launcher loop.

    The health callback reads cached status only. Frontend 0 still sends its
    heartbeat from the launcher loop, so a sufficiently long stall can expire
    its lease and stop the group.
    """

    def __init__(
        self,
        directory: str,
        count: int,
        instance_id: str,
        healthy: Callable[[], bool],
        fail: Callable[[str], None],
    ) -> None:
        self._owner_loop = asyncio.get_running_loop()
        self._fail = fail
        self._owner_failure_lock = threading.RLock()
        self._failure: str | None = None
        self._closing = False
        self._start_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._coordinator = Coordinator(
            directory, count, instance_id, healthy, self._fence, self._owner_failure_lock
        )

    async def start(self) -> None:
        if self._start_task is not None or self._closing:
            raise RuntimeError("Coordinator already started or closed")
        self._start_task = asyncio.create_task(asyncio.to_thread(self._coordinator.start))
        try:
            await asyncio.shield(self._start_task)
        except BaseException:
            await asyncio.shield(self.close())
            raise

    def _fence(self, reason: str) -> None:
        with self._owner_failure_lock:
            self._coordinator.ready = False
            self._coordinator.stopping = True
            if self._failure is not None:
                return
            self._failure = reason
        try:
            self._owner_loop.call_soon_threadsafe(self._fail, reason)
        except RuntimeError:
            # A closed owner loop cannot resume serving; admission is fenced.
            pass

    def report_owner_failure(self, reason: str) -> None:
        """Publish terminal owner failure before any later reservation commits."""
        with self._owner_failure_lock:
            if self._failure is None:
                self._failure = reason
            self._coordinator.ready = False
            self._coordinator.stopping = True

    async def set_ready(self) -> None:
        with self._owner_failure_lock:
            if self._failure is not None:
                raise CoordinationError(f"OpenEngine coordinator is fenced: {self._failure}")
            if self._closing:
                raise CoordinationError("OpenEngine coordinator is fenced")
            self._coordinator.ready = True

    async def stop_admission(self) -> None:
        """Keep private transport alive while frontends release reservations."""
        with self._owner_failure_lock:
            self._closing = True
            self._coordinator.ready = False
            self._coordinator.stopping = True

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        await self.stop_admission()
        try:
            if self._start_task is not None:
                await asyncio.shield(self._start_task)
        finally:
            await asyncio.to_thread(self._coordinator.close)
