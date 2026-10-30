# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep private coordination responsive independently of inference callbacks."""

import asyncio
import concurrent.futures
import threading
from collections.abc import Callable, Coroutine

import uvloop

from .coordinator import CoordinationError, Coordinator


class CoordinatorReactor:
    """Own the coordinator and its gRPC objects on a dedicated event-loop thread.

    The health callback only reads cached status. LLM operations, request handles,
    frontend clients, and data IPC remain on their existing owner loop.
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
        self._directory = directory
        self._count = count
        self._instance_id = instance_id
        self._healthy = healthy
        self._fail = fail
        self._loop_ready: concurrent.futures.Future[asyncio.AbstractEventLoop] = (
            concurrent.futures.Future()
        )
        self._started: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._finished: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._thread: threading.Thread | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._coordinator: Coordinator | None = None
        self._stop: asyncio.Event | None = None
        self._start_task: asyncio.Task[None] | None = None
        self._closing = False
        self._failure: str | None = None
        self._reactor_loop: asyncio.AbstractEventLoop | None = None
        self._owner_failure_lock = threading.RLock()
        self._pending_owner_failure: str | None = None

    async def start(self) -> None:
        if self._thread is not None or self._closing:
            raise RuntimeError("Coordinator reactor already started or closed")
        self._thread = threading.Thread(target=self._thread_main, name="openengine-coordinator")
        try:
            self._thread.start()
        except BaseException:
            # No loop exists to receive shutdown commands if thread creation fails.
            self._thread = None
            self._closing = True
            raise
        try:
            await asyncio.shield(asyncio.wrap_future(self._started))
        except BaseException:
            await asyncio.shield(self.close())
            raise

    def _thread_main(self) -> None:
        try:
            uvloop.run(self._run(), loop_factory=uvloop.new_event_loop)
        except BaseException as error:
            for future in (self._loop_ready, self._started, self._finished):
                if not future.done():
                    future.set_exception(error)
            self._notify_owner(f"OpenEngine coordinator reactor failed: {error}")
        else:
            if not self._started.done():
                self._started.cancel()
            self._finished.set_result(None)

    async def _run(self) -> None:
        self._coordinator = Coordinator(
            self._directory,
            self._count,
            self._instance_id,
            self._owner_healthy,
            self._fence,
            admission_lock=self._owner_failure_lock,
        )
        self._stop = asyncio.Event()
        with self._owner_failure_lock:
            self._reactor_loop = asyncio.get_running_loop()
            pending_failure = self._pending_owner_failure
        self._loop_ready.set_result(self._reactor_loop)
        try:
            if pending_failure is not None:
                self._fence(pending_failure)
            self._start_task = asyncio.create_task(self._coordinator.start())
            try:
                await self._start_task
            except asyncio.CancelledError:
                if self._closing:
                    return
                raise
            if self._closing:
                return
            self._started.set_result(None)
            await self._stop.wait()
        finally:
            self._coordinator.ready = False
            self._coordinator.stopping = True
            await self._coordinator.close()

    def _notify_owner(self, reason: str) -> None:
        try:
            self._owner_loop.call_soon_threadsafe(self._fail, reason)
        except RuntimeError:
            # A closed owner loop cannot resume serving; admission is already fenced.
            pass

    def _owner_healthy(self) -> bool:
        with self._owner_failure_lock:
            return self._pending_owner_failure is None and self._healthy()

    def _fence(self, reason: str) -> None:
        assert self._coordinator is not None
        self._coordinator.ready = False
        self._coordinator.stopping = True
        if self._failure is None:
            self._failure = reason
            self._notify_owner(reason)

    async def _submit(self, coroutine: Coroutine[None, None, None]) -> None:
        try:
            if self._thread is None or self._finished.done():
                raise CoordinationError("OpenEngine coordinator reactor is not running")
            loop = await asyncio.shield(asyncio.wrap_future(self._loop_ready))
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        except BaseException:
            coroutine.close()
            raise
        await asyncio.shield(asyncio.wrap_future(future))

    def report_owner_failure(self, reason: str) -> None:
        """Retain terminal owner failures even before the receiver thread starts."""
        with self._owner_failure_lock:
            if self._pending_owner_failure is None:
                self._pending_owner_failure = reason
            loop = self._reactor_loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._fence, reason)
            except RuntimeError:
                # The closed receiver can no longer admit requests.
                pass

    async def set_ready(self) -> None:
        async def apply() -> None:
            if self._failure is not None or self._closing:
                raise CoordinationError("OpenEngine coordinator reactor is fenced")
            assert self._coordinator is not None
            self._coordinator.ready = True

        await self._submit(apply())

    async def stop_admission(self) -> None:
        """Keep private transport alive while frontends release their reservations."""
        if self._thread is None or self._finished.done():
            self._closing = True
            return

        async def stop() -> None:
            self._closing = True
            assert self._coordinator is not None
            self._coordinator.ready = False
            self._coordinator.stopping = True

        await self._submit(stop())

    def _request_stop(self) -> None:
        self._closing = True
        if self._coordinator is not None:
            self._coordinator.ready = False
            self._coordinator.stopping = True
        assert self._stop is not None
        self._stop.set()
        if self._start_task is not None and not self._start_task.done():
            self._start_task.cancel()

    async def close(self) -> None:
        if self._thread is None:
            self._closing = True
            return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        try:
            loop = await asyncio.shield(asyncio.wrap_future(self._loop_ready))
            if not self._finished.done():
                loop.call_soon_threadsafe(self._request_stop)
            await asyncio.shield(asyncio.wrap_future(self._finished))
        finally:
            assert self._thread is not None
            await asyncio.to_thread(self._thread.join)
