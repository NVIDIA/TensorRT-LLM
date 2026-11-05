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
        self._failure_cleanup_started = False
        self._cancelled_work: set[asyncio.Task] = set()

    def record_failure(self, error: Exception | None = None) -> bool:
        """Preserve an observed worker failure before cancellation cleanup starts.

        Args:
            error: Selected exception, or None for an existing in-band error stream.

        Returns:
            Whether a worker failure was selected before the request deadline.
        """
        if not self._failure_observed and self._loop.time() < self._deadline:
            self._failure_observed = True
            self._failure = error
        return self._failure_observed

    def _expiration_error(self) -> Exception:
        """Return the earlier observed failure, or a new request-budget error."""
        return self._failure or DisaggRequestTimeoutError(self.total_seconds)

    def begin_failure_cleanup(self) -> None:
        """Allow cleanup-owning waits to drain after a selected failure.

        The caller must cancel its unfinished work before this transition.
        Independent live reads must use run with allow_failure_cleanup=False.
        Recording a failure alone does not stop the deadline.
        """
        self._failure_cleanup_started = self._failure_observed

    def cancel_work(self, task: asyncio.Task) -> None:
        """Cancel unfinished work once, without interrupting owned cleanup.

        Args:
            task: Request task whose remaining work should stop.
        """
        if not task.done() and task not in self._cancelled_work:
            self._cancelled_work.add(task)
            task.cancel()

    async def run_cleanup(self, operation: Callable[[], Awaitable[None]]) -> None:
        """Drain cleanup even when its waiter is cancelled repeatedly.

        Args:
            operation: Cleanup only; must not dispatch or continue request work.

        Raises:
            asyncio.CancelledError: Caller cancellation, after cleanup completes.
        """
        cleanup = asyncio.gather(operation(), return_exceptions=True)
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        error = cleanup.result()[0]
        if isinstance(error, BaseException):
            if not self._failure_observed and not cancelled:
                raise error
            self._loop.call_exception_handler(
                {"message": "Request cleanup failed after a terminal outcome", "exception": error}
            )
        if cancelled:
            raise asyncio.CancelledError

    async def run(
        self, operation: Callable[[], Awaitable[_T]], *, allow_failure_cleanup: bool = True
    ) -> _T:
        """Await one phase using the original budget and drain cancelled work.

        Args:
            operation: Factory invoked only while request time remains.
            allow_failure_cleanup: Permit a cleanup-owning wait to outlive expiry
                after begin_failure_cleanup. Disable for independent HTTP reads.

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
            try:
                if self._loop.time() >= self._deadline:
                    raise self._expiration_error()
                return await operation()
            finally:
                completed_at = self._loop.time()

        task = asyncio.create_task(invoke())
        claimed = False
        try:
            completed, _ = await asyncio.wait((task,), timeout=remaining)
            if not completed:
                if not allow_failure_cleanup or not self._failure_cleanup_started:
                    raise self._expiration_error()
                # Only explicit cleanup may outlive the budget. Recording an
                # error does not exempt remaining HTTP work from cancellation.
                await asyncio.shield(task)
            if not task.cancelled():
                assert completed_at is not None
                if completed_at >= self._deadline and not self._failure_observed:
                    raise self._expiration_error()
            result = task.result()
            claimed = True
            return result
        finally:
            if not claimed:
                self.cancel_work(task)

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
