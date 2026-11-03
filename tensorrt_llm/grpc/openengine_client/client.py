# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async client for an OpenEngine v1 server."""

from __future__ import annotations

import asyncio
import enum
import uuid
import weakref
from collections.abc import Awaitable, Callable, Sequence
from types import TracebackType
from typing import Any, TypeVar

import grpc
from google.protobuf.message import Message

from tensorrt_llm.grpc.openengine.bindings import (
    generation_pb2,
    kv_pb2,
    lifecycle_pb2,
    model_pb2,
    openengine_pb2_grpc,
    server_pb2,
)

from ._aio import discard
from .errors import (
    OpenEngineClientError,
    OpenEngineIncompatibleServerError,
    OpenEngineProtocolError,
    OpenEngineRpcError,
)
from .stream import GenerateStream

SUPPORTED_SCHEMA_REVISION = 1
"""int: The newest OpenEngine schema revision this client implements."""

MAX_MESSAGE_BYTES = 64 * 1024 * 1024
"""int: Default send and receive limit for one gRPC message."""

_READY_POLL_INTERVAL_SECONDS = 0.5
_NOT_READY_CODES = frozenset({grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED})

_ResponseT = TypeVar("_ResponseT")


def new_request_id() -> str:
    """Return a fresh Generate ``request_id``."""
    return f"trtllm-{uuid.uuid4().hex}"


class AbortOutcome(enum.Enum):
    """Result of aborting one request by ``request_id``.

    Callers can treat every outcome as success.
    """

    ABORTED = "aborted"
    ALREADY_FINISHED = "already_finished"
    NOT_FOUND = "not_found"


class OpenEngineClient:
    """Async client for one OpenEngine v1 server.

    Create it with :meth:`connect`, which waits for the engine to report ready
    and checks its schema revision. All methods must run on the event loop the
    client was created on.

    Each Control method takes an optional ``timeout`` that replaces
    ``control_timeout`` for that call. A Control call still in flight when the
    client is closed raises :class:`OpenEngineClientError`.
    """

    def __init__(
        self,
        target: str,
        channel: grpc.aio.Channel,
        *,
        control_timeout: float,
        allow_newer_schema: bool,
    ) -> None:
        """Wrap an open channel without contacting the engine; :meth:`connect` is the public entry.

        Args:
            target: The engine address, ``host:port``.
            channel: A ``grpc.aio`` channel to ``target``. The client closes it.
            control_timeout: Default deadline in seconds for Control RPCs.
            allow_newer_schema: See :meth:`connect`.
        """
        self._target = target
        self._channel = channel
        self._control_timeout = control_timeout
        self._allow_newer_schema = allow_newer_schema
        self._control = openengine_pb2_grpc.ControlStub(channel)
        self._inference = openengine_pb2_grpc.InferenceStub(channel)
        self._streams: weakref.WeakSet[GenerateStream] = weakref.WeakSet()
        self._server_info: server_pb2.ServerInfo | None = None
        self._model_info: model_pb2.ModelInfo | None = None
        self._closed = False

    @classmethod
    async def connect(
        cls,
        target: str,
        *,
        ready_timeout: float = 60.0,
        control_timeout: float = 5.0,
        allow_newer_schema: bool = False,
        channel_options: Sequence[tuple[str, Any]] = (),
    ) -> OpenEngineClient:
        """Connect to an engine and wait until it can serve requests.

        Polls ``Health`` until it reports ``HEALTH_STATE_READY``. Until
        ``GetServerInfo`` answers, each poll also tries it and checks the
        schema revision it reports, so an incompatible engine that answers
        ``GetServerInfo`` while still loading is rejected without waiting for it
        to finish. Once the engine is ready, reads and checks ``ServerInfo``
        again and caches ``ModelInfo``.

        Args:
            target: The engine address, ``host:port``.
            ready_timeout: Seconds to wait for the engine to become ready. The
                reads after it is ready each use ``control_timeout``.
            control_timeout: Default deadline in seconds for Control RPCs.
            allow_newer_schema: Accept an engine whose ``schema_revision`` is
                newer than :data:`SUPPORTED_SCHEMA_REVISION`, as long as its
                ``minimum_client_revision`` still allows this client.
            channel_options: Extra gRPC channel options. They take precedence
                over the client's defaults.

        Raises:
            OpenEngineRpcError: The engine did not become ready in time, or a
                Control RPC failed.
            OpenEngineIncompatibleServerError: The engine's schema revision is
                not supported.
        """
        options = {
            "grpc.max_send_message_length": MAX_MESSAGE_BYTES,
            "grpc.max_receive_message_length": MAX_MESSAGE_BYTES,
        }
        options.update(channel_options)
        channel = grpc.aio.insecure_channel(target, options=list(options.items()))
        client = cls(
            target,
            channel,
            control_timeout=control_timeout,
            allow_newer_schema=allow_newer_schema,
        )
        deadline = asyncio.get_running_loop().time() + ready_timeout
        try:
            await client._wait_until_ready(deadline, ready_timeout)
            await client.refresh()
        except BaseException:
            await client.close()
            raise
        return client

    @property
    def target(self) -> str:
        """The engine address."""
        return self._target

    @property
    def server_info(self) -> server_pb2.ServerInfo:
        """The ``ServerInfo`` read by the last :meth:`connect` or :meth:`refresh`."""
        if self._server_info is None:
            raise OpenEngineClientError("ServerInfo has not been read", target=self._target)
        return self._server_info

    @property
    def model_info(self) -> model_pb2.ModelInfo:
        """The ``ModelInfo`` read by the last :meth:`connect` or :meth:`refresh`."""
        if self._model_info is None:
            raise OpenEngineClientError("ModelInfo has not been read", target=self._target)
        return self._model_info

    async def refresh(self, *, timeout: float | None = None) -> None:
        """Re-read ``ServerInfo`` and ``ModelInfo`` and re-check the schema revision.

        Args:
            timeout: Deadline in seconds for each of the two RPCs; ``None``
                means ``control_timeout``.

        Raises:
            OpenEngineRpcError: A Control RPC failed.
            OpenEngineIncompatibleServerError: The engine's schema revision is
                not supported.
        """
        server_info = await self._call_control(
            self._control.GetServerInfo,
            server_pb2.GetServerInfoRequest(),
            rpc="GetServerInfo",
            timeout=timeout,
        )
        self._check_schema(server_info)
        model_info = await self._call_control(
            self._control.GetModelInfo,
            model_pb2.GetModelInfoRequest(),
            rpc="GetModelInfo",
            timeout=timeout,
        )
        self._server_info = server_info
        self._model_info = model_info

    async def health(
        self, *, include_inference_probe: bool = False, timeout: float | None = None
    ) -> lifecycle_pb2.HealthResponse:
        """Return the engine's health.

        Args:
            include_inference_probe: Ask the engine to run a small generation
                as part of the check. An engine may take longer than
                ``control_timeout`` to finish the probe; pass a longer
                ``timeout`` for it.
            timeout: Deadline in seconds; ``None`` means ``control_timeout``.
        """
        return await self._call_control(
            self._control.Health,
            lifecycle_pb2.HealthRequest(include_inference_probe=include_inference_probe),
            rpc="Health",
            timeout=timeout,
        )

    async def get_load(self, *, timeout: float | None = None) -> server_pb2.LoadInfo | None:
        """Return the engine's load, or ``None`` if the engine does not implement ``GetLoad``.

        Args:
            timeout: Deadline in seconds; ``None`` means ``control_timeout``.
        """
        try:
            return await self._call_control(
                self._control.GetLoad, server_pb2.GetLoadRequest(), rpc="GetLoad", timeout=timeout
            )
        except OpenEngineRpcError as error:
            if error.code == grpc.StatusCode.UNIMPLEMENTED:
                return None
            raise

    async def abort(self, request_id: str, *, timeout: float | None = None) -> AbortOutcome:
        """Abort one request by ``request_id``.

        Args:
            request_id: The ``request_id`` of the Generate request to abort.
            timeout: Deadline in seconds; ``None`` means ``control_timeout``.

        Returns:
            How the engine answered. ``ALREADY_FINISHED`` and ``NOT_FOUND`` mean
            the engine had no running request with that id.

        Raises:
            ValueError: ``request_id`` is empty.
            OpenEngineRpcError: The Abort RPC failed with a status other than
                ``NOT_FOUND``.
            OpenEngineProtocolError: The engine returned an abort status this
                client does not recognize.
        """
        if not request_id:
            raise ValueError("request_id must be non-empty")
        try:
            response = await self._call_control(
                self._control.Abort,
                lifecycle_pb2.AbortRequest(request_id=request_id),
                rpc="Abort",
                timeout=timeout,
            )
        except OpenEngineRpcError as error:
            if error.code == grpc.StatusCode.NOT_FOUND:
                return AbortOutcome.NOT_FOUND
            raise
        return self._abort_outcome(response, f"request {request_id}", request_id=request_id)

    async def abort_all(self, *, timeout: float | None = None) -> lifecycle_pb2.AbortResponse:
        """Abort every request on the engine.

        Args:
            timeout: Deadline in seconds; ``None`` means ``control_timeout``.

        Raises:
            OpenEngineRpcError: The Abort RPC failed.
            OpenEngineProtocolError: The engine returned an abort status this
                client does not recognize.
        """
        response = await self._call_control(
            self._control.Abort,
            lifecycle_pb2.AbortRequest(all_requests=lifecycle_pb2.AllRequests()),
            rpc="Abort",
            timeout=timeout,
        )
        self._abort_outcome(response, "all requests")
        return response

    async def abort_kv_session(
        self, kv_session: kv_pb2.KvSessionRef, *, timeout: float | None = None
    ) -> lifecycle_pb2.AbortResponse:
        """Ask a prefill engine to release the KV it holds for a disaggregated handoff.

        Args:
            kv_session: The ``KvSessionRef`` from the prefill request's
                ``prefill_ready`` event.
            timeout: Deadline in seconds; ``None`` means ``control_timeout``.

        Raises:
            OpenEngineRpcError: The Abort RPC failed, for example with
                ``UNIMPLEMENTED`` when the engine does not support releasing KV
                sessions.
            OpenEngineProtocolError: The engine returned an abort status this
                client does not recognize.
        """
        response = await self._call_control(
            self._control.Abort,
            lifecycle_pb2.AbortRequest(kv_session=kv_session),
            rpc="Abort",
            timeout=timeout,
        )
        self._abort_outcome(response, f"KV session {kv_session.session_id}")
        return response

    def generate(
        self,
        request: generation_pb2.GenerateRequest,
        *,
        timeout: float | None = None,
    ) -> GenerateStream:
        """Prepare a Generate call. The RPC is sent on the first iteration of the returned stream.

        Args:
            request: The request. The stream sends a copy taken now, so later
                changes to ``request`` are not sent. When its ``request_id`` is
                empty, the copy gets a new id; ``request`` itself is not
                modified.
            timeout: Deadline in seconds for the RPC, counted from the first
                iteration; ``None`` means no deadline.

        Returns:
            The response stream. Use it as an async context manager.
        """
        self._ensure_open()
        sent = generation_pb2.GenerateRequest()
        sent.CopyFrom(request)
        if not sent.request_id:
            sent.request_id = new_request_id()
        stream = GenerateStream(
            self._inference,
            sent,
            timeout=timeout,
            target=self._target,
            on_done=self._streams.discard,
        )
        self._streams.add(stream)
        return stream

    async def close(self) -> None:
        """Cancel every stream that has not reached its terminal event and close the channel.

        Closing the channel also cancels the calls still being read in the
        background after a ``prefill_ready`` event. Safe to call again.
        """
        self._closed = True
        for stream in list(self._streams):
            stream.cancel()
        await self._channel.close()

    async def __aenter__(self) -> OpenEngineClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def _call_control(
        self,
        method: Callable[..., Awaitable[_ResponseT]],
        request: Message,
        *,
        rpc: str,
        timeout: float | None = None,
        wait_for_ready: bool | None = None,
    ) -> _ResponseT:
        self._ensure_open()
        call = method(
            request,
            timeout=self._control_timeout if timeout is None else timeout,
            wait_for_ready=wait_for_ready,
        )
        # As in GenerateStream._read: a call cancelled locally completes the
        # wait, while cancelling the calling task raises out of it. The client
        # never hands out its calls, so only closing the channel cancels one.
        response = asyncio.ensure_future(call)
        try:
            await asyncio.wait((response,))
        except asyncio.CancelledError:
            discard(response)
            raise
        if response.cancelled():
            raise OpenEngineClientError(
                f"the client was closed during the {rpc} call", target=self._target
            )
        try:
            return response.result()
        except grpc.aio.AioRpcError as error:
            raise OpenEngineRpcError(
                error.code(), error.details() or "", rpc=rpc, target=self._target
            ) from error

    async def _try_check_schema(self, timeout: float) -> bool:
        """Check ``ServerInfo`` if the engine answers; return whether it answered."""
        try:
            server_info = await self._call_control(
                self._control.GetServerInfo,
                server_pb2.GetServerInfoRequest(),
                rpc="GetServerInfo",
                timeout=timeout,
                wait_for_ready=True,
            )
        except OpenEngineRpcError as error:
            if error.code not in _NOT_READY_CODES:
                raise
            return False
        self._check_schema(server_info)
        return True

    async def _wait_until_ready(self, deadline: float, ready_timeout: float) -> None:
        loop = asyncio.get_running_loop()
        last_problem = "no response"
        schema_checked = False
        while True:
            remaining = deadline - loop.time()
            if remaining > 0 and not schema_checked:
                schema_checked = await self._try_check_schema(min(remaining, self._control_timeout))
                remaining = deadline - loop.time()
            if remaining <= 0:
                raise OpenEngineRpcError(
                    grpc.StatusCode.UNAVAILABLE,
                    f"not ready within {ready_timeout:g}s ({last_problem})",
                    rpc="Health",
                    target=self._target,
                )
            try:
                response = await self._call_control(
                    self._control.Health,
                    lifecycle_pb2.HealthRequest(),
                    rpc="Health",
                    timeout=min(remaining, self._control_timeout),
                    wait_for_ready=True,
                )
            except OpenEngineRpcError as error:
                if error.code not in _NOT_READY_CODES:
                    raise
                last_problem = f"{error.code.name}: {error.details}"
            else:
                if response.state == lifecycle_pb2.HEALTH_STATE_READY:
                    return
                last_problem = f"state {lifecycle_pb2.HealthState.Name(response.state)}"
            await asyncio.sleep(min(_READY_POLL_INTERVAL_SECONDS, max(0.0, deadline - loop.time())))

    def _abort_outcome(
        self,
        response: lifecycle_pb2.AbortResponse,
        aborted: str,
        *,
        request_id: str | None = None,
    ) -> AbortOutcome:
        if response.status == lifecycle_pb2.ABORT_STATUS_ABORTED:
            return AbortOutcome.ABORTED
        if response.status == lifecycle_pb2.ABORT_STATUS_ALREADY_FINISHED:
            return AbortOutcome.ALREADY_FINISHED
        raise OpenEngineProtocolError(
            f"OpenEngine engine {self._target} answered Abort for {aborted} "
            f"with status {response.status}",
            target=self._target,
            request_id=request_id,
        )

    def _check_schema(self, info: server_pb2.ServerInfo) -> None:
        problem = None
        if info.schema_revision == 0:
            problem = "reports schema_revision 0, which is invalid"
        elif info.minimum_client_revision > SUPPORTED_SCHEMA_REVISION:
            problem = (
                f"requires client revision {info.minimum_client_revision}, but this client "
                f"implements revision {SUPPORTED_SCHEMA_REVISION}"
            )
        elif info.schema_revision > SUPPORTED_SCHEMA_REVISION and not self._allow_newer_schema:
            problem = (
                f"implements schema revision {info.schema_revision}, newer than the "
                f"supported revision {SUPPORTED_SCHEMA_REVISION}; pass allow_newer_schema=True "
                "to accept it"
            )
        if problem is not None:
            raise OpenEngineIncompatibleServerError(
                f"OpenEngine engine {self._target} (GetServerInfo) {problem}",
                target=self._target,
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise OpenEngineClientError("the client is closed", target=self._target)


__all__ = [
    "MAX_MESSAGE_BYTES",
    "SUPPORTED_SCHEMA_REVISION",
    "AbortOutcome",
    "OpenEngineClient",
    "new_request_id",
]
