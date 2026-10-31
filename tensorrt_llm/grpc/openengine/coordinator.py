# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private, local coordination for one multiprocess OpenEngine instance.

Only request ownership and control messages cross this service. Generation
handles and token streams stay in the frontend that submitted the request.
The Unix sockets live in the launcher's mode-0700 temporary directory.
"""

import asyncio
import concurrent.futures
import json
import queue
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from time import monotonic
from typing import Any

import grpc

from tensorrt_llm.logger import logger

_RPC_TIMEOUT = 5.0
_ABORT_TIMEOUT = 30.0
_HEARTBEAT_TIMEOUT = 10.0
_CONTROL_THREADS = 4
_ABORT_THREADS = 4
_MAX_REQUEST_ID_BYTES = 1024
# At most six JSON bytes per UTF-8 byte, plus the fixed UUID and framing:
# 128 accepted entries occupy less than 1 MiB, well below the 64 MiB limit.
_ABORT_BATCH_SIZE = 128
_SERVICE = "trtllm.private.OpenEngineFrontend"
_OPTIONS = [
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
]


class CoordinationError(RuntimeError):
    """A private coordination operation could not be completed."""


class DuplicateRequestError(ValueError):
    """An external request ID is already reserved by another stream."""


class InvalidRequestIdError(ValueError):
    """A request ID cannot safely be carried by the private control protocol."""


def _validate_request_id(request_id: str) -> None:
    # Check length first to avoid allocating an encoded copy of an oversized ID.
    # Even worst-case JSON escaping of an accepted ID stays well below the
    # private transport limit. Reject before creating any ownership tasks.
    if len(request_id) > _MAX_REQUEST_ID_BYTES or len(request_id.encode()) > _MAX_REQUEST_ID_BYTES:
        raise InvalidRequestIdError(
            f"request_id must not exceed {_MAX_REQUEST_ID_BYTES} UTF-8 bytes with multiple frontends"
        )


def _encode(message: dict) -> bytes:
    return json.dumps(message).encode()


def _address(directory: str, name: str) -> str:
    return f"unix:{Path(directory) / name}"


def _server(
    address: str, handler: Callable, server: grpc.aio.Server | None = None
) -> grpc.aio.Server:
    if server is None:
        server = grpc.aio.server(options=_OPTIONS)
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                _SERVICE,
                {
                    "Call": grpc.unary_unary_rpc_method_handler(
                        handler, request_deserializer=json.loads, response_serializer=_encode
                    )
                },
            ),
        )
    )
    server.add_insecure_port(address)
    return server


class _Connection:
    def __init__(self, address: str) -> None:
        self.channel = grpc.aio.insecure_channel(address, options=_OPTIONS)
        self.call = self.channel.unary_unary(
            f"/{_SERVICE}/Call", request_serializer=_encode, response_deserializer=json.loads
        )

    async def request(self, message: dict) -> dict:
        timeout = _ABORT_TIMEOUT if message.get("operation") == "abort" else _RPC_TIMEOUT
        return await self.call(message, timeout=timeout)

    async def close(self) -> None:
        await self.channel.close()


class _SyncConnection:
    """A synchronous private channel used only by coordinator handlers."""

    def __init__(self, address: str) -> None:
        self.channel = grpc.insecure_channel(address, options=_OPTIONS)
        self.call = self.channel.unary_unary(
            f"/{_SERVICE}/Call", request_serializer=_encode, response_deserializer=json.loads
        )

    def request(self, message: dict, timeout: float) -> grpc.Future:
        return self.call.future(message, timeout=timeout)

    def close(self) -> None:
        self.channel.close()


class Coordinator:
    """Own request state and serve private control without another aio loop.

    Slow aborts use their own endpoint and pool so they cannot occupy the
    heartbeat and reservation handlers. All state changes hold one lock.
    """

    def __init__(
        self,
        directory: str,
        count: int,
        instance_id: str,
        healthy: Callable[[], bool],
        fail: Callable[[str], None],
        admission_lock: AbstractContextManager[Any] | None = None,
    ) -> None:
        self.instance_id = instance_id
        self.ready = False
        self.stopping = False
        self._heartbeats: list[float | None] = [None] * count
        self._healthy = healthy
        self._fail = fail
        self._admission_lock = admission_lock if admission_lock is not None else threading.RLock()
        self._requests: dict[str, tuple[str, int]] = {}
        self._directory = directory
        self._peers: list[_SyncConnection] = []
        self._servers: list[grpc.Server] = []
        self._pools: list[concurrent.futures.ThreadPoolExecutor] = []

    def start(self) -> None:
        try:
            for frontend in range(len(self._heartbeats)):
                self._peers.append(
                    _SyncConnection(_address(self._directory, f"frontend-{frontend}"))
                )
            for name, handler, workers, limit in (
                ("coordinator", self._dispatch, _CONTROL_THREADS, None),
                ("coordinator-abort", self._abort, _ABORT_THREADS, _ABORT_THREADS),
            ):
                pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix=f"openengine-{name}"
                )
                self._pools.append(pool)
                server = grpc.server(pool, options=_OPTIONS, maximum_concurrent_rpcs=limit)
                self._servers.append(server)
                server.add_generic_rpc_handlers(
                    (
                        grpc.method_handlers_generic_handler(
                            _SERVICE,
                            {
                                "Call": grpc.unary_unary_rpc_method_handler(
                                    handler,
                                    request_deserializer=json.loads,
                                    response_serializer=_encode,
                                )
                            },
                        ),
                    )
                )
                address = _address(self._directory, name)
                if not server.add_insecure_port(address):
                    raise RuntimeError(f"Failed to bind OpenEngine private endpoint {address}")
                server.start()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        with self._admission_lock:
            self.ready = False
            self.stopping = True
        try:
            stopped = [server.stop(0) for server in self._servers]
            for peer in self._peers:
                peer.close()
            for event in stopped:
                event.wait()
        finally:
            for pool in self._pools:
                pool.shutdown(wait=True)
            self._servers.clear()
            self._peers.clear()
            self._pools.clear()

    def _group_ready(self) -> bool:
        """Check group readiness while holding the admission lock."""
        if not self.ready or self.stopping:
            return False
        now = monotonic()
        for frontend, last_seen in enumerate(self._heartbeats):
            if last_seen is None or now - last_seen >= _HEARTBEAT_TIMEOUT:
                self.ready = False
                self._fail(f"OpenEngine frontend {frontend} heartbeat expired")
                return False
        return self._healthy()

    def _dispatch(self, message: dict, context: grpc.ServicerContext) -> dict:
        operation = message["operation"]
        with self._admission_lock:
            if operation == "heartbeat":
                # A late heartbeat cannot resurrect an expired group.
                self._group_ready()
                self._heartbeats[message["frontend"]] = monotonic()
                operation = "status"
            if operation == "status":
                return {
                    "ready": self._group_ready(),
                    "instance_id": self.instance_id,
                    "count": len(self._requests),
                    "stopping": self.stopping,
                }
            if operation == "withdraw":
                self.ready = False
                if not self.stopping:
                    self.stopping = True
                    self._fail(f"OpenEngine frontend {message['frontend']} is stopping")
                return {}
            if operation == "reserve":
                if not self._group_ready():
                    return {"ready": False, "reserved": False}
                request_id = message["request_id"]
                owner = (message["token"], message["frontend"])
                current = self._requests.get(request_id)
                if current is not None and current != owner:
                    return {"reserved": False}
                self._requests[request_id] = owner
                return {"reserved": True}
            if operation == "release":
                request_id = message["request_id"]
                if self._requests.get(request_id) == (message["token"], message["frontend"]):
                    del self._requests[request_id]
                return {}
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unknown private operation")

    def _abort(self, message: dict, context: grpc.ServicerContext) -> dict:
        if message.get("operation") != "abort":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unknown private operation")
        # Keep generation tokens in the snapshot, then release the state lock
        # before any RPC; a slow frontend must not block normal control traffic.
        with self._admission_lock:
            request_id = message.get("request_id")
            if request_id is None:
                snapshot = list(self._requests.items())
            else:
                owner = self._requests.get(request_id)
                snapshot = [(request_id, owner)] if owner is not None else []
        by_frontend: dict[int, list] = defaultdict(list)
        for key, (token, frontend) in snapshot:
            by_frontend[frontend].append([key, token])

        deadline = monotonic() + _ABORT_TIMEOUT - _RPC_TIMEOUT
        completed: queue.Queue[tuple[int | None, grpc.Future | None]] = queue.Queue()
        if not context.add_callback(lambda: completed.put((None, None))):
            context.abort(grpc.StatusCode.CANCELLED, "Abort request cancelled")
        active: dict[int, grpc.Future] = {}
        offsets = dict.fromkeys(by_frontend, 0)
        aborted = failed = 0

        def submit(frontend: int) -> bool:
            remaining = deadline - monotonic()
            if remaining <= 0:
                return False
            offset = offsets[frontend]
            batch = by_frontend[frontend][offset : offset + _ABORT_BATCH_SIZE]
            future = self._peers[frontend].request(
                {"requests": batch}, min(_RPC_TIMEOUT, remaining)
            )
            active[frontend] = future
            future.add_done_callback(lambda result: completed.put((frontend, result)))
            return True

        try:
            for frontend in by_frontend:
                if not submit(frontend):
                    failed += len(by_frontend[frontend])
            while active:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    failed += sum(len(by_frontend[i]) - offsets[i] for i in active)
                    break
                try:
                    frontend, future = completed.get(timeout=remaining)
                except queue.Empty:
                    failed += sum(len(by_frontend[i]) - offsets[i] for i in active)
                    break
                if frontend is None:
                    context.abort(grpc.StatusCode.CANCELLED, "Abort request cancelled")
                if active.get(frontend) is not future:
                    continue
                del active[frontend]
                try:
                    reply = future.result()
                except grpc.RpcError as error:
                    if error.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
                        with self._admission_lock:
                            if self.stopping or not context.is_active():
                                context.abort(grpc.StatusCode.CANCELLED, "Abort request cancelled")
                            self.ready = False
                            self._fail(f"Frontend control connection failed: {error}")
                        context.abort(grpc.StatusCode.UNAVAILABLE, "Frontend control failed")
                    failed += len(by_frontend[frontend]) - offsets[frontend]
                    continue
                aborted += reply["aborted"]
                failed += reply["failed"]
                offsets[frontend] += min(
                    _ABORT_BATCH_SIZE, len(by_frontend[frontend]) - offsets[frontend]
                )
                if offsets[frontend] < len(by_frontend[frontend]) and not submit(frontend):
                    failed += len(by_frontend[frontend]) - offsets[frontend]
            return {"aborted": aborted, "failed": failed}
        finally:
            for future in active.values():
                future.cancel()


class Reservation:
    """A frontend-owned generation, including the pre-submission interval."""

    def __init__(self, client: "FrontendClient", request_id: str) -> None:
        self.client = client
        self.request_id = request_id
        self.token = uuid.uuid4().hex
        self.handle: Any = None
        self.aborted = False
        self.releasing = False
        self.reserve_task = asyncio.create_task(
            client.request(
                "reserve", request_id=request_id, token=self.token, frontend=client.frontend_id
            )
        )
        self.release_task: asyncio.Task | None = None

    def bind(self, handle: Any) -> None:
        """Bind without yielding, so an abort cannot miss a submitted handle."""
        self.handle = handle
        if self.aborted:
            handle.abort()

    def abort(self) -> bool:
        if self.releasing:
            return False
        if self.handle is not None:
            self.handle.abort()
        self.aborted = True
        return True

    def release(self) -> asyncio.Task:
        """Release even if the caller is cancelled or suspended in a yield."""
        if self.release_task is None:
            self.releasing = True
            self.release_task = self.client.cleanup(self._release())
        return self.release_task

    async def _release(self) -> None:
        try:
            # The reservation may have committed before Generate was cancelled.
            # Never race Release ahead of Reserve, or discard its unknown reply.
            reply = await self.reserve_task
            if reply["reserved"]:
                await self.client.request(
                    "release",
                    request_id=self.request_id,
                    token=self.token,
                    frontend=self.client.frontend_id,
                )
        finally:
            self.client.reservations.pop(self.token, None)


class FrontendClient:
    """Private control client and fenced abort endpoint for one frontend."""

    def __init__(self, directory: str, frontend_id: int, fail: Callable[[str], None]) -> None:
        self.frontend_id = frontend_id
        self.reservations: dict[str, Reservation] = {}
        self._fail = fail
        self._cleanup: set[asyncio.Task] = set()
        self._connection = _Connection(_address(directory, "coordinator"))
        self._abort_connection = _Connection(_address(directory, "coordinator-abort"))
        self._server = _server(_address(directory, f"frontend-{frontend_id}"), self._abort)

    async def start(self) -> None:
        await self._server.start()
        await self.request("heartbeat", frontend=self.frontend_id)

    async def request(self, operation: str, **fields: Any) -> dict:
        if operation == "abort" and "request_id" in fields:
            _validate_request_id(fields["request_id"])
        try:
            connection = self._abort_connection if operation == "abort" else self._connection
            return await connection.request({"operation": operation, **fields})
        except grpc.aio.AioRpcError as error:
            if operation == "abort" and error.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                # The bounded abort pool is busy; ownership remains intact and
                # a caller may retry without invalidating the serving group.
                raise CoordinationError("OpenEngine abort capacity exhausted") from error
            # A reserve reply may have been lost after committing. Fail the
            # entire group rather than retry with ambiguous ownership.
            self._fail(f"OpenEngine coordinator unavailable: {error}")
            raise CoordinationError("OpenEngine coordinator unavailable") from error

    async def reserve(self, request_id: str) -> Reservation:
        _validate_request_id(request_id)
        reservation = Reservation(self, request_id)
        self.reservations[reservation.token] = reservation
        try:
            reply = await asyncio.shield(reservation.reserve_task)
            if reply.get("ready") is False:
                raise CoordinationError("Frontend group is not ready")
            if not reply["reserved"]:
                raise DuplicateRequestError(f"request_id '{request_id}' is already active")
            return reservation
        except BaseException:
            reservation.release()
            raise

    def cleanup(self, coroutine: Any) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._cleanup.add(task)
        task.add_done_callback(self._cleanup_done)
        return task

    def _cleanup_done(self, task: asyncio.Task) -> None:
        self._cleanup.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            self._fail(f"OpenEngine ownership cleanup failed: {error}")

    async def _abort(self, message: dict, context: grpc.aio.ServicerContext) -> dict:
        aborted = failed = 0
        for request_id, token in message["requests"]:
            reservation = self.reservations.get(token)
            if reservation is None or reservation.request_id != request_id:
                continue
            try:
                aborted += reservation.abort()
            except Exception as error:
                # Engine-specific abort exceptions must retain their failure
                # outcome instead of being reported as ALREADY_FINISHED.
                logger.warning(f"Failed to abort OpenEngine request {request_id}: {error}")
                failed += 1
        return {"aborted": aborted, "failed": failed}

    async def close(self) -> None:
        for reservation in list(self.reservations.values()):
            reservation.release()
        if self._cleanup:
            await asyncio.gather(*self._cleanup, return_exceptions=True)
        await self._server.stop(0)
        await self._connection.close()
        await self._abort_connection.close()
