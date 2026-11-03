# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated, cancellable iteration over one OpenEngine Generate response stream."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import TracebackType
from typing import Any

import grpc

from tensorrt_llm.grpc.openengine.bindings import generation_pb2, openengine_pb2_grpc
from tensorrt_llm.logger import logger

from ._aio import discard
from .errors import OpenEngineEngineError, OpenEngineProtocolError, OpenEngineRpcError

# How long to keep reading after a finished or error terminal event while
# waiting for the server to close the stream with OK.
DRAIN_TIMEOUT_SECONDS = 1.0

# Returned by GenerateStream._read when the call was cancelled locally.
_CALL_CANCELLED = object()

_KNOWN_FINISH_REASONS = frozenset(generation_pb2.FinishReason.values()) - {
    generation_pb2.FINISH_REASON_UNSPECIFIED
}

# Strong references to the tasks that read the rest of a call after its
# prefill_ready event, so they are not garbage-collected while running.
_BACKGROUND_DRAINS: set[asyncio.Task[None]] = set()


class GenerateStream:
    """One Generate call, iterated as validated ``GenerateResponse`` messages.

    The request is sent at the first iteration. Iteration ends after the
    terminal event: every output's ``finished`` event or a ``prefill_ready``
    event. A terminal ``error`` event is raised as
    :class:`OpenEngineEngineError` instead of being yielded.

    Before the terminal event, a non-OK status is raised as
    :class:`OpenEngineRpcError`, except the ``CANCELLED`` status of a call this
    stream cancelled, which ends iteration quietly; a response that breaks the
    stream contract is raised as :class:`OpenEngineProtocolError`. A response
    with an event type this client does not know is logged and skipped.

    After the terminal event, the stream reads the call to its end and logs any
    further responses and a non-OK status. After a ``finished`` or ``error``
    terminal event it waits up to :data:`DRAIN_TIMEOUT_SECONDS` for the server
    to close the call and then cancels it. After a ``prefill_ready`` event it
    never cancels the call, because the server would abort the request and
    could release the KV the event hands off; iteration ends at once and the
    rest of the call is read in the background until the server closes it, the
    call's deadline passes, or the client is closed.

    Cancelling the stream cancels the RPC; an OpenEngine server must treat
    that as an abort of the request. Use the stream as an async context manager
    so that leaving the block early always cancels it::

        async with client.generate(request) as stream:
            async for response in stream:
                ...

    Created by :meth:`OpenEngineClient.generate`; not constructed directly.
    """

    def __init__(
        self,
        stub: openengine_pb2_grpc.InferenceStub,
        request: generation_pb2.GenerateRequest,
        *,
        timeout: float | None,
        target: str,
        on_done: Callable[[GenerateStream], None] | None = None,
    ) -> None:
        self._stub = stub
        self._request = request
        self._timeout = timeout
        self._target = target
        self._on_done = on_done
        sampling = request.sampling
        self._num_outputs = (
            sampling.num_sequences
            if sampling.HasField("num_sequences") and sampling.num_sequences > 1
            else 1
        )
        self._call: grpc.aio.UnaryStreamCall | None = None
        self._finished_outputs: set[int] = set()
        self._output_seen = False
        self._prompt_seen = False
        self._terminal = False
        self._prefill_ready = False
        self._done = False
        self._cancelled = False
        self._pending_error: OpenEngineEngineError | None = None

    @property
    def request_id(self) -> str:
        """The request's ``request_id``."""
        return self._request.request_id

    @property
    def done(self) -> bool:
        """Whether iteration has ended, normally, by error, or by cancellation."""
        return self._done

    @property
    def cancelled(self) -> bool:
        """Whether the stream was cancelled before its terminal event."""
        return self._cancelled

    def cancel(self) -> None:
        """Cancel the request.

        Idempotent. Has no effect once the terminal event has been received.
        A stream cancelled before its first iteration never sends the request
        and is done at once. Iterating a cancelled stream ends without raising;
        check :attr:`cancelled` to tell it from a stream that completed.
        """
        if self._terminal or self._done:
            return
        self._cancelled = True
        if self._call is None:
            self._finish()
        else:
            self._cancel_call()

    async def aclose(self) -> None:
        """End the stream, cancelling the request if it has not finished yet."""
        if self._done:
            return
        try:
            if self._terminal:
                await self._end_after_terminal()
            else:
                self.cancel()
        finally:
            self._finish()

    async def __aenter__(self) -> GenerateStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    def __aiter__(self) -> GenerateStream:
        return self

    async def __anext__(self) -> generation_pb2.GenerateResponse:
        if self._done:
            raise StopAsyncIteration
        if self._cancelled:
            self._finish()
            raise StopAsyncIteration
        if self._terminal:
            try:
                await self._end_after_terminal()
            finally:
                self._finish()
            raise StopAsyncIteration
        if self._call is None:
            try:
                self._call = self._stub.Generate(self._request, timeout=self._timeout)
            except BaseException:
                self._finish()
                raise
        while True:
            try:
                response = await self._read()
            except asyncio.CancelledError:
                self._finish()
                raise
            except grpc.aio.AioRpcError as error:
                self._finish()
                if self._cancelled and error.code() == grpc.StatusCode.CANCELLED:
                    raise StopAsyncIteration from None
                raise OpenEngineRpcError(
                    error.code(),
                    error.details() or "",
                    rpc="Generate",
                    target=self._target,
                    request_id=self.request_id,
                ) from error
            except BaseException:
                self._cancel_call()
                self._finish()
                raise
            if response is _CALL_CANCELLED:
                self._finish()
                if self._cancelled:
                    raise StopAsyncIteration
                raise OpenEngineRpcError(
                    grpc.StatusCode.CANCELLED,
                    "the call was cancelled outside the stream",
                    rpc="Generate",
                    target=self._target,
                    request_id=self.request_id,
                )
            if response is grpc.aio.EOF:
                self._finish()
                raise self._protocol_error("the stream ended before a terminal event")
            try:
                known = self._validate(response)
            except OpenEngineProtocolError:
                self._cancel_call()
                self._finish()
                raise
            if not known:
                logger.warning(
                    f"OpenEngine engine {self._target} sent a response with an unknown event "
                    f"for request {self.request_id}; skipping it"
                )
                continue
            if self._pending_error is not None:
                try:
                    await self._end_after_terminal()
                finally:
                    self._finish()
                raise self._pending_error
            return response

    async def _read(self, timeout: float | None = None) -> Any:
        """Read the next message from the call.

        Returns the message, ``grpc.aio.EOF``, or ``_CALL_CANCELLED`` when the
        call was cancelled locally. Raises ``asyncio.TimeoutError`` after
        ``timeout`` seconds. If the calling task is cancelled, cancels the
        call, marks the stream cancelled unless its terminal event has already
        arrived, and re-raises. Waiting with ``asyncio.wait`` keeps those two kinds of
        cancellation apart: a cancelled read completes the wait, while
        cancelling this task raises out of it.
        """
        assert self._call is not None
        read = asyncio.ensure_future(self._call.read())
        try:
            done, _ = await asyncio.wait((read,), timeout=timeout)
        except asyncio.CancelledError:
            discard(read)
            if not self._terminal:
                self._cancelled = True
            self._cancel_call()
            raise
        if not done:
            discard(read)
            raise asyncio.TimeoutError
        if read.cancelled():
            return _CALL_CANCELLED
        return read.result()

    def _validate(self, response: generation_pb2.GenerateResponse) -> bool:
        """Check one response against the stream contract and update the stream state.

        Returns False for a response whose event type this client does not know.
        """
        if response.request_id and response.request_id != self.request_id:
            raise self._protocol_error(f"a response carries request_id '{response.request_id}'")
        event = response.WhichOneof("event")
        if event is None:
            return False
        if event == "prompt":
            if self._prompt_seen or self._output_seen:
                raise self._protocol_error(
                    "the prompt event is repeated or follows a token or finished event"
                )
            self._prompt_seen = True
        elif event in ("token", "finished"):
            payload = response.token if event == "token" else response.finished
            if not payload.HasField("output_index"):
                raise self._protocol_error(f"a {event} event has no output_index")
            index = payload.output_index
            if index >= self._num_outputs:
                raise self._protocol_error(
                    f"a {event} event has output_index {index}, but the request asked "
                    f"for {self._num_outputs} output(s)"
                )
            if index in self._finished_outputs:
                raise self._protocol_error(f"a {event} event follows output {index}'s finish")
            self._output_seen = True
            if event == "finished":
                reason = response.finished.reason
                if reason not in _KNOWN_FINISH_REASONS:
                    # An unknown value counts as FINISH_REASON_UNSPECIFIED.
                    raise self._protocol_error(
                        f"output {index} finished without a known reason ({reason})"
                    )
                self._finished_outputs.add(index)
                self._terminal = len(self._finished_outputs) == self._num_outputs
        else:
            # prefill_ready and error end the request.
            self._terminal = True
            self._prefill_ready = event == "prefill_ready"
        if response.HasField("usage") and not self._terminal:
            raise self._protocol_error(f"a non-terminal {event} event carries usage")
        if event == "error":
            error = response.error
            self._pending_error = OpenEngineEngineError(
                error.code,
                error.message,
                retryable=error.retryable,
                usage=response.usage if response.HasField("usage") else None,
                target=self._target,
                request_id=self.request_id,
            )
        return True

    async def _end_after_terminal(self) -> None:
        """Read the rest of the call after the terminal event; see the class docstring."""
        if self._call is None:
            return
        if self._prefill_ready:
            task = asyncio.get_running_loop().create_task(self._drain(timeout=None))
            _BACKGROUND_DRAINS.add(task)
            task.add_done_callback(_BACKGROUND_DRAINS.discard)
            return
        await self._drain(timeout=DRAIN_TIMEOUT_SECONDS)

    async def _drain(self, timeout: float | None) -> None:
        """Read to the end of the call after its terminal event.

        The request's outcome is settled at this point, so extra events or a
        non-OK status are logged rather than raised. With a ``timeout``, cancels
        the call if the server has not closed it within that many seconds.
        """
        assert self._call is not None
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        try:
            while True:
                try:
                    response = await self._read(
                        timeout=None if deadline is None else max(0.0, deadline - loop.time())
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"OpenEngine engine {self._target} did not close request "
                        f"{self.request_id} within {timeout:g}s of its terminal "
                        "event; cancelling the call"
                    )
                    self._cancel_call()
                    return
                if response is _CALL_CANCELLED or response is grpc.aio.EOF:
                    return
                event = response.WhichOneof("event") or "unknown"
                logger.warning(
                    f"OpenEngine engine {self._target} sent a {event} event after request "
                    f"{self.request_id} ended; ignoring it"
                )
        except grpc.aio.AioRpcError as error:
            logger.warning(
                f"OpenEngine engine {self._target} ended request {self.request_id} with "
                f"{error.code().name} after its terminal event: {error.details()}"
            )

    def _cancel_call(self) -> None:
        if self._call is not None:
            self._call.cancel()

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        if self._on_done is not None:
            self._on_done(self)

    def _protocol_error(self, problem: str) -> OpenEngineProtocolError:
        return OpenEngineProtocolError(
            f"OpenEngine engine {self._target} broke the Generate stream contract for "
            f"request {self.request_id}: {problem}",
            target=self._target,
            request_id=self.request_id,
        )


__all__ = ["DRAIN_TIMEOUT_SECONDS", "GenerateStream"]
