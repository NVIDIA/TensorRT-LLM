# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One serving-clock deadline for a no-replay disaggregated request."""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TypeVar

_T = TypeVar("_T")


class DisaggRequestTimeoutError(TimeoutError):
    """The pair-serving budget expired, independently of worker deadlines."""

    def __init__(self, total_seconds: float) -> None:
        """Describe the expired serving budget without attributing a network cause.

        Args:
            total_seconds: Configured request-wide budget in seconds.
        """
        super().__init__(f"Disaggregated request exceeded its {total_seconds:g}-second E2E budget")


class DisaggRequestDeadline:
    """Keep one absolute deadline in the pair-serving event loop's clock domain."""

    def __init__(self, total_seconds: float) -> None:
        """Start the request budget before placement or worker dispatch.

        Args:
            total_seconds: Validated positive, finite request budget in seconds.
        """
        self.total_seconds = total_seconds
        self._loop = asyncio.get_running_loop()
        self._deadline = self._loop.time() + total_seconds
        self._failure_observed = False
        self._failure: Exception | None = None

    def record_failure(self, error: Exception | None = None) -> None:
        """Preserve an observed worker failure before cancellation cleanup starts.

        Args:
            error: Selected exception, or None for an existing in-band error stream.
        """
        if not self._failure_observed and self._loop.time() < self._deadline:
            self._failure_observed = True
            self._failure = error

    def _expiration_error(self) -> Exception:
        """Return the earlier observed failure, or a new request-budget error."""
        return self._failure or DisaggRequestTimeoutError(self.total_seconds)

    async def run(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        """Await one phase using the original budget and drain cancelled work.

        Args:
            operation: Factory invoked only while request time remains.

        Returns:
            The operation's result, including ownership of a returned stream.

        Raises:
            DisaggRequestTimeoutError: The request-wide deadline expires.

        Worker errors and caller cancellation retain their original identity.
        Cleanup can outlive the deadline; it does not establish physical retirement.
        """
        remaining = self._deadline - self._loop.time()
        if remaining <= 0:
            raise self._expiration_error()

        completed_at: float | None = None

        async def invoke() -> _T:
            """Run the operation in a task owned by this wait."""
            nonlocal completed_at
            if self._loop.time() >= self._deadline:
                raise self._expiration_error()
            result = await operation()
            completed_at = self._loop.time()
            return result

        task = asyncio.create_task(invoke())
        claimed = False
        try:
            completed, _ = await asyncio.wait((task,), timeout=remaining)
            if not completed:
                if not self._failure_observed:
                    raise self._expiration_error()
                # The workflow already selected a terminal failure. Its cleanup
                # is not more request work and must not receive another cancel.
                await asyncio.shield(task)
            result = task.result()
            assert completed_at is not None
            if completed_at >= self._deadline and not self._failure_observed:
                raise self._expiration_error()
            claimed = True
            return result
        finally:
            if not claimed:
                if not task.done():
                    task.cancel()

                async def reclaim() -> None:
                    """Drain work and close an unclaimed stream without a second cancellation."""
                    await asyncio.gather(task, return_exceptions=True)
                    if not task.cancelled() and task.exception() is None:
                        result = task.result()
                        if isinstance(result, AsyncGenerator):
                            # Setup can return a stream while cancellation races it.
                            await result.aclose()

                cleanup = asyncio.gather(reclaim(), return_exceptions=True)
                cancelled = False
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        # Paired cleanup can cancel this waiter after its own
                        # deadline already cancelled the worker operation.
                        cancelled = True
                cleanup_error = cleanup.result()[0]
                if isinstance(cleanup_error, BaseException):
                    self._loop.call_exception_handler(
                        {
                            "message": "Failed to close an unclaimed request stream",
                            "exception": cleanup_error,
                        }
                    )
                if cancelled:
                    raise asyncio.CancelledError
