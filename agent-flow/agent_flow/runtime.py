from __future__ import annotations

import asyncio
import inspect
from queue import Queue
from threading import Event, Lock, Thread

import anyio

from .types import AgentRequest


class PortalRunner:
    """Run async work on a dedicated background asyncio loop."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._thread: Thread | None = None
        self._queue: Queue | None = None
        self._ready = Event()

    @property
    def started(self) -> bool:
        return self._thread is not None

    def _run_worker(self) -> None:
        queue = Queue()
        self._queue = queue
        self._ready.set()
        with asyncio.Runner() as runner:
            while True:
                func, args, kwargs, result_queue = queue.get()
                try:
                    if func is None:
                        break
                    result = runner.run(self._invoke(func, *args, **kwargs))
                except Exception as exc:
                    result_queue.put((False, exc))
                else:
                    result_queue.put((True, result))
                finally:
                    # Drop the last submitted callable so bound methods do not
                    # keep their owning layer alive after the task completes.
                    func = None
                    args = ()
                    kwargs = {}
                    result_queue = None

    def _ensure_worker(self) -> Thread:
        with self._lock:
            if self._thread is None:
                self._ready = Event()
                self._thread = Thread(target=self._run_worker, daemon=True)
                self._thread.start()
                self._ready.wait()
            if self._thread is None or self._queue is None:
                raise RuntimeError("Background worker failed to start.")
            return self._thread

    async def _invoke(self, func, *args, **kwargs):
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def call(self, func, *args, **kwargs):
        self._ensure_worker()
        if self._queue is None:
            raise RuntimeError("Background worker queue was not initialized.")

        result_queue = Queue(maxsize=1)
        self._queue.put((func, args, kwargs, result_queue))
        ok, payload = result_queue.get()
        if ok:
            return payload
        raise payload

    async def acall(self, func, *args, **kwargs):
        return await anyio.to_thread.run_sync(self.call, func, *args, **kwargs)

    def close(self) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
            queue = self._queue
            self._queue = None
        if queue is not None:
            queue.put((None, (), {}, Queue()))
        if thread is not None:
            thread.join()


def build_request(
    content: str,
    system_prompt: str | None = None,
) -> AgentRequest:
    return AgentRequest(content=content, system_prompt=system_prompt)
