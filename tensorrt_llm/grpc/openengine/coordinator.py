# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private, local coordination for one multiprocess OpenEngine instance.

Only request ownership and control messages cross this service. Generation
handles and token streams stay in the frontend that submitted the request.
The Unix sockets live in the launcher's mode-0700 temporary directory.
"""

import asyncio
import json
import uuid
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

import grpc

from tensorrt_llm.logger import logger

_RPC_TIMEOUT = 5.0
_ABORT_TIMEOUT = 30.0
_HEARTBEAT_TIMEOUT = 10.0
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
    """The serving group can no longer guarantee request ownership."""


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


def _server(address: str, handler: Callable) -> grpc.aio.Server:
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


class Coordinator:
    """Serialize ownership changes and route fenced control operations."""

    def __init__(
        self,
        directory: str,
        count: int,
        instance_id: str,
        healthy: Callable[[], bool],
        fail: Callable[[str], None],
    ) -> None:
        self.instance_id = instance_id
        self.ready = False
        self.stopping = False
        self._heartbeats: list[float | None] = [None] * count
        self._healthy = healthy
        self._fail = fail
        self._requests: dict[str, tuple[str, int]] = {}
        self._peers = [_Connection(_address(directory, f"frontend-{i}")) for i in range(count)]
        self._server = _server(_address(directory, "coordinator"), self._dispatch)

    async def start(self) -> None:
        await self._server.start()

    async def close(self) -> None:
        self.ready = False
        await self._server.stop(0)
        await asyncio.gather(*(peer.close() for peer in self._peers))

    def _group_ready(self) -> bool:
        if not self.ready or self.stopping:
            return False
        now = monotonic()
        for frontend, last_seen in enumerate(self._heartbeats):
            if last_seen is None or now - last_seen >= _HEARTBEAT_TIMEOUT:
                self.ready = False
                self._fail(f"OpenEngine frontend {frontend} heartbeat expired")
                return False
        return self._healthy()

    async def _abort_frontend(self, frontend: int, requests: list, deadline: float) -> dict:
        aborted = failed = 0
        # One outstanding batch per frontend bounds fan-out independently of
        # the snapshot size. Each batch retains the original generation tokens.
        for start in range(0, len(requests), _ABORT_BATCH_SIZE):
            remaining = deadline - monotonic()
            if remaining <= 0:
                return {"aborted": aborted, "failed": failed + len(requests) - start}
            try:
                reply = await asyncio.wait_for(
                    self._peers[frontend].request(
                        {"requests": requests[start : start + _ABORT_BATCH_SIZE]}
                    ),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                # The current batch has no confirmed outcome. Include it and
                # unsent batches as failures without invalidating the group.
                return {"aborted": aborted, "failed": failed + len(requests) - start}
            except grpc.aio.AioRpcError as error:
                if error.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise
                return {"aborted": aborted, "failed": failed + len(requests) - start}
            aborted += reply["aborted"]
            failed += reply["failed"]
        return {"aborted": aborted, "failed": failed}

    async def _dispatch(self, message: dict, context: grpc.aio.ServicerContext) -> dict:
        operation = message["operation"]
        if operation == "heartbeat":
            # Check expiry before refreshing: a late heartbeat cannot resurrect
            # a group that has already lost a frontend's responsiveness guarantee.
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
        if operation == "abort":
            # Snapshot before awaiting peer RPCs. Every dispatched item carries
            # its generation token, including entries in abort-all snapshots.
            # Reserve one control-RPC interval to deliver partial outcomes
            # before the caller's overall abort deadline expires.
            deadline = monotonic() + _ABORT_TIMEOUT - _RPC_TIMEOUT
            request_id = message.get("request_id")
            if request_id is None:
                snapshot = list(self._requests.items())
            else:
                owner = self._requests.get(request_id)
                snapshot = [(request_id, owner)] if owner is not None else []
            by_frontend: dict[int, list] = defaultdict(list)
            for key, (token, frontend) in snapshot:
                by_frontend[frontend].append([key, token])
            tasks = [
                asyncio.create_task(self._abort_frontend(frontend, requests, deadline))
                for frontend, requests in by_frontend.items()
            ]
            try:
                if tasks:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                    for task in done:
                        if (error := task.exception()) is not None:
                            raise error
                replies = [task.result() for task in tasks]
                return {
                    "aborted": sum(reply["aborted"] for reply in replies),
                    "failed": sum(reply["failed"] for reply in replies),
                }
            except grpc.aio.AioRpcError as error:
                self.ready = False
                self._fail(f"Frontend control connection failed: {error}")
                await context.abort(grpc.StatusCode.UNAVAILABLE, "Frontend control failed")
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unknown private operation")


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
        self._server = _server(_address(directory, f"frontend-{frontend_id}"), self._abort)

    async def start(self) -> None:
        await self._server.start()
        await self.request("heartbeat", frontend=self.frontend_id)

    async def request(self, operation: str, **fields: Any) -> dict:
        if operation == "abort" and "request_id" in fields:
            _validate_request_id(fields["request_id"])
        try:
            return await self._connection.request({"operation": operation, **fields})
        except grpc.aio.AioRpcError as error:
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
