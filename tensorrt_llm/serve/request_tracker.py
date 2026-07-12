# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol-neutral tracking for requests submitted to an LLM instance."""

import asyncio
from collections.abc import Iterable
from typing import Protocol

from tensorrt_llm.logger import logger


class AbortableResult(Protocol):
    def abort(self) -> None:
        """Abort the associated engine request."""


class RequestTracker:
    """Track active engine results for admission, cancellation, and drain."""

    def __init__(self, llm: object) -> None:
        self.llm = llm
        self._requests: dict[str, AbortableResult] = {}
        self._external_requests = 0
        self._draining = False
        self._changed = asyncio.Condition()

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def active_count(self) -> int:
        return len(self._requests) + self._external_requests

    @property
    def active_requests(self) -> dict[str, AbortableResult]:
        """Compatibility view for protocol request managers."""
        return self._requests

    def admit(self, request_id: str, result: AbortableResult) -> None:
        if self._draining:
            raise RuntimeError("TensorRT-LLM is draining")
        if request_id in self._requests:
            raise ValueError(f"Request {request_id!r} is already active")
        self._requests[request_id] = result

    async def finish(self, request_id: str) -> None:
        async with self._changed:
            self._requests.pop(request_id, None)
            self._changed.notify_all()

    def begin_external(self) -> None:
        if self._draining:
            raise RuntimeError("TensorRT-LLM is draining")
        self._external_requests += 1

    async def finish_external(self) -> None:
        async with self._changed:
            self._external_requests = max(0, self._external_requests - 1)
            self._changed.notify_all()

    async def abort(self, request_id: str) -> bool:
        result = self._requests.get(request_id)
        if result is None:
            return False
        try:
            result.abort()
        except (RuntimeError, AssertionError) as error:
            logger.warning("Failed to abort request %s: %s", request_id, error)
            return False
        finally:
            await self.finish(request_id)
        return True

    async def abort_all(self) -> int:
        tracked_by_identity = {id(result): result for result in self._requests.values()}
        executor = getattr(self.llm, "_executor", None)
        executor_results = getattr(executor, "_results", None)
        if isinstance(executor_results, dict):
            tracked_by_identity.update(
                (id(result), result) for result in tuple(executor_results.values())
            )

        aborted = 0
        for result in tracked_by_identity.values():
            try:
                result.abort()
            except (RuntimeError, AssertionError) as error:
                logger.warning("Failed to abort engine request: %s", error)
            else:
                aborted += 1

        async with self._changed:
            self._requests.clear()
            self._changed.notify_all()
        return aborted

    async def start_drain(self) -> int:
        async with self._changed:
            self._draining = True
            self._changed.notify_all()
        return self.active_count

    async def wait_empty(self, timeout: float | None = None) -> bool:
        async def _wait() -> None:
            async with self._changed:
                await self._changed.wait_for(lambda: self.active_count == 0)

        if self.active_count == 0:
            return True
        try:
            await asyncio.wait_for(_wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return True

    async def health(self) -> tuple[bool, str]:
        if not hasattr(self.llm, "_executor"):
            return True, "OK"
        executor = getattr(self.llm, "_executor", None)
        if executor is None:
            return False, "Executor is not available"
        try:
            healthy = executor.check_health()
        except (RuntimeError, AttributeError) as error:
            return False, f"Executor health check failed: {error}"
        if healthy:
            return True, "OK"
        fatal_error = getattr(executor, "_fatal_error", None)
        if fatal_error is None:
            return False, "Executor is unhealthy"
        lines = str(fatal_error).splitlines()
        short = (lines[0] if lines else type(fatal_error).__name__)[:200]
        return False, f"{type(fatal_error).__name__}: {short}"

    def iter_results(self) -> Iterable[AbortableResult]:
        return tuple(self._requests.values())


async def track_http_response(response: object, tracker: RequestTracker) -> object:
    """Keep HTTP admission active until a streaming response body closes."""
    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is None:
        await tracker.finish_external()
        return response

    async def tracked_body():
        try:
            async for chunk in body_iterator:
                yield chunk
        finally:
            await tracker.finish_external()

    response.body_iterator = tracked_body()
    return response
