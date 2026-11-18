# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Coordinator HTTP server for disaggregated serving.

One coordinator process owns all cluster state (routers, readiness, worker
events, and -- for the centralized router -- the single ZMQ event-ingest bind)
and answers the internal coordination API that the forked worker processes call:

    POST /select   {"role", "routing_key", "req_id", "exclude_server"}
      -> {"server": "host:port", "info": {...}, "req_id": <int|null>}
         ("server" is null, and nothing is reserved, when the router needs
         more routing input, e.g. info["routing_content_required"])
    POST /finish   {"role", "req_id", "success"}  -> {}
    POST /renew    {"role", "req_id"}  -> {}
    GET  /cluster_info -> {...}
    GET  /health   -> 200 when ready
    GET  /version

The routing key is produced worker-side by ``Router.routing_key`` and consumed
here by ``Router.get_next_server_by_key`` (see ``serve/router.py``), so this
endpoint is generic across the stateful router types that use it (centralized ->
block hashes, conversation -> conversation_id). Single-process by design; it owns
the ZMQ ingest bind for centralized mode.
"""

import asyncio
import socket
from contextlib import asynccontextmanager
from typing import Awaitable, Optional, TypeVar

import msgpack
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

from tensorrt_llm.logger import logger
from tensorrt_llm.serve._telemetry import create_uvicorn_server
from tensorrt_llm.serve.cluster_storage import HttpClusterStorageServer
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.output_length_predictor import MAX_TOKENS, RECORD_BATCH_MAX, is_token_count
from tensorrt_llm.serve.seat_pool import NoGenerationSeatError
from tensorrt_llm.version import __version__ as VERSION

TIMEOUT_KEEP_ALIVE = 10  # seconds
# Accept queue of the fleet UNIX socket: a burst opens one connect per concurrent
# /select, and a connect to a full queue fails with ENOTCONN on its first send.
# The kernel caps the value at net.core.somaxconn.
COORDINATOR_UDS_BACKLOG = 16384
# uvicorn's default accept queue: a cap at or above it is only logged, not warned about.
UVICORN_DEFAULT_BACKLOG = 2048
MSGPACK_MEDIA_TYPE = "application/msgpack"
# How often serving checks whether a uvicorn server began its shutdown (uvicorn's own tick).
SHUTDOWN_POLL_S = 0.1
_T = TypeVar("_T")


class _ClientDisconnected(Exception):
    """The client of a waiting request closed its connection."""


async def _wait_for_disconnect(raw_req: Request) -> None:
    """Return once the client of ``raw_req``, whose body was read, closes its connection."""
    while (await raw_req.receive())["type"] != "http.disconnect":
        pass


async def _until_disconnect(raw_req: Request, selection: Awaitable[_T]) -> _T:
    """Await ``selection``; cancel it and raise ``_ClientDisconnected`` if the client leaves first.

    uvicorn does not cancel a handler whose client disconnected, so a pooled select of a fleet
    worker that died would otherwise keep its place in the seat queue and later take a seat.
    """
    selecting = asyncio.ensure_future(selection)
    watching = asyncio.ensure_future(_wait_for_disconnect(raw_req))
    try:
        await asyncio.wait((selecting, watching), return_when=asyncio.FIRST_COMPLETED)
        if not selecting.done() and not watching.cancelled() and watching.exception() is None:
            selecting.cancel()
            try:
                return await selecting
            except asyncio.CancelledError:
                raise _ClientDisconnected() from None
        return await selecting
    finally:
        watching.cancel()
        # Also when this handler itself is cancelled.
        selecting.cancel()


def _effective_uds_backlog() -> int:
    """COORDINATOR_UDS_BACKLOG as capped by net.core.somaxconn, where Linux exposes it."""
    try:
        with open("/proc/sys/net/core/somaxconn") as f:
            return min(COORDINATOR_UDS_BACKLOG, int(f.read()))
    except (OSError, ValueError):
        return COORDINATOR_UDS_BACKLOG


class CoordinatorServer:
    """Serve a :class:`DisaggCoordinatorService`'s coordination API over HTTP."""

    def __init__(self, coordinator: DisaggCoordinatorService) -> None:
        self._coordinator = coordinator
        # The uvicorn servers of the running __call__.
        self.servers: list[uvicorn.Server] = []

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self._coordinator.start()
            yield
            await self._coordinator.stop()

        self.app = FastAPI(lifespan=lifespan)
        self.app.add_api_route("/select", self.select, methods=["POST"])
        self.app.add_api_route("/finish", self.finish, methods=["POST"])
        self.app.add_api_route("/renew", self.renew, methods=["POST"])
        self.app.add_api_route("/output_length", self.output_length, methods=["POST"])
        self.app.add_api_route("/cluster_info", self.cluster_info, methods=["GET"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        cluster_storage = self._coordinator.cluster_storage
        if isinstance(cluster_storage, HttpClusterStorageServer):
            cluster_storage.add_routes(self.app)

    @staticmethod
    def _response(content: object, status_code: int = 200) -> Response:
        return Response(
            content=msgpack.packb(content, use_bin_type=True),
            status_code=status_code,
            media_type=MSGPACK_MEDIA_TYPE,
        )

    async def select(self, raw_req: Request) -> Response:
        try:
            body = msgpack.unpackb(await raw_req.body(), raw=False)
        except Exception as e:
            return self._response({"error": f"invalid MessagePack body: {e}"}, status_code=400)
        if (
            not isinstance(body, dict)
            or "role" not in body
            or "routing_key" not in body
            or "req_id" not in body
        ):
            return self._response(
                {"error": "body must include 'role', 'routing_key', and 'req_id'"},
                status_code=400,
            )
        role = body["role"]
        if role not in ("context", "ctx", "generation", "gen"):
            return self._response({"error": f"invalid role: {role}"}, status_code=400)
        req_id = body["req_id"]
        if not isinstance(req_id, int) or isinstance(req_id, bool):
            return self._response({"error": "req_id must be an integer"}, status_code=400)
        exclude_server = body.get("exclude_server")
        if exclude_server is not None and not isinstance(exclude_server, str):
            return self._response({"error": "exclude_server must be a string"}, status_code=400)
        selection = self._coordinator.select(role, body["routing_key"], req_id, exclude_server)
        try:
            if self._pooled(role):
                server, info, req_id = await _until_disconnect(raw_req, selection)
            else:
                server, info, req_id = await selection
        except _ClientDisconnected:
            # Nobody reads this reply; the cancelled wait gave its place or seat back.
            return self._response({"error": "client disconnected"}, status_code=503)
        except NoGenerationSeatError as e:
            # Overload, not a fault: the frontend answers its client 503 too.
            return self._response({"error": str(e), "no_generation_seat": True}, status_code=503)
        except ValueError as e:
            return self._response({"error": str(e)}, status_code=503)
        except Exception as e:  # noqa: BLE001
            logger.error(f"CoordinatorServer.select failed: {e}")
            return self._response({"error": str(e)}, status_code=500)
        return self._response({"server": server, "info": info, "req_id": req_id})

    def _pooled(self, role: str) -> bool:
        """Whether a select for ``role`` may wait for a seat (a pooled router)."""
        if role in ("context", "ctx"):
            return self._coordinator.ctx_router.pooled
        return self._coordinator.gen_router.pooled

    async def finish(self, raw_req: Request) -> Response:
        try:
            body = msgpack.unpackb(await raw_req.body(), raw=False)
        except Exception as e:
            return self._response({"error": f"invalid MessagePack body: {e}"}, status_code=400)
        if not isinstance(body, dict) or "role" not in body or "req_id" not in body:
            return self._response(
                {"error": "body must include 'role' and 'req_id'"}, status_code=400
            )
        role = body["role"]
        if role not in ("context", "ctx", "generation", "gen"):
            return self._response({"error": f"invalid role: {role}"}, status_code=400)
        req_id = body["req_id"]
        if not isinstance(req_id, int) or isinstance(req_id, bool):
            return self._response({"error": "req_id must be an integer"}, status_code=400)
        success = body.get("success", True)
        if not isinstance(success, bool):
            return self._response({"error": "success must be a boolean"}, status_code=400)
        await self._coordinator.finish(role, req_id, success)
        return self._response({})

    @staticmethod
    def _output_length_error(
        conv: object, isl: object, out: object = 0.0, key: object = None
    ) -> Optional[str]:
        """Why a prediction's or reply's fields are invalid, or None."""
        if conv is not None and not isinstance(conv, str):
            return "conv must be a string or null"
        if not is_token_count(isl):
            return f"isl must be a finite number in [0, {MAX_TOKENS:.0f}]"
        if key is not None and (not isinstance(key, int) or isinstance(key, bool)):
            return "key must be an integer or null"
        if out is None and key is not None:
            return None  # a failed request: its prediction row is dropped
        if not is_token_count(out):
            return f"out must be a finite number in [0, {MAX_TOKENS:.0f}], or null with a key"
        return None

    async def output_length(self, raw_req: Request) -> Response:
        """Output-length prediction (``output_length_prediction``) for frontends.

        POST {"op": "predict", "conv", "isl"} -> {"tokens", "key"}, and
        POST {"op": "record", "records": [[conv, isl, out, key], ...]} -> {"learned"} (``out``
        null with a ``key``: the request failed, its prediction row is dropped).
        """
        try:
            body = msgpack.unpackb(await raw_req.body(), raw=False)
        except Exception as e:
            return self._response({"error": f"invalid MessagePack body: {e}"}, status_code=400)
        if not isinstance(body, dict) or body.get("op") not in ("predict", "record"):
            return self._response({"error": "op must be 'predict' or 'record'"}, status_code=400)
        try:
            if body["op"] == "predict":
                conv, isl = body.get("conv"), body.get("isl")
                error = self._output_length_error(conv, isl)
                if error:
                    return self._response({"error": error}, status_code=400)
                tokens, key = await self._coordinator.predict_output_length(conv, isl)
                return self._response({"tokens": tokens, "key": key})
            records = body.get("records")
            if not isinstance(records, list) or len(records) > RECORD_BATCH_MAX:
                return self._response(
                    {"error": f"records must be a list of at most {RECORD_BATCH_MAX} entries"},
                    status_code=400,
                )
            for record in records:
                if not isinstance(record, list) or len(record) != 4:
                    return self._response(
                        {"error": "a record is [conv, isl, out, key]"}, status_code=400
                    )
                error = self._output_length_error(*record)
                if error:
                    return self._response({"error": error}, status_code=400)
            learned = await self._coordinator.record_output_lengths(
                [
                    (conv, float(isl), None if out is None else float(out), key)
                    for conv, isl, out, key in records
                ]
            )
            return self._response({"learned": learned})
        except ValueError as e:
            return self._response({"error": str(e)}, status_code=503)

    async def renew(self, raw_req: Request) -> Response:
        try:
            body = msgpack.unpackb(await raw_req.body(), raw=False)
        except Exception as e:
            return self._response({"error": f"invalid MessagePack body: {e}"}, status_code=400)
        if not isinstance(body, dict) or "role" not in body or "req_id" not in body:
            return self._response(
                {"error": "body must include 'role' and 'req_id'"}, status_code=400
            )
        role = body["role"]
        if role not in ("context", "ctx", "generation", "gen"):
            return self._response({"error": f"invalid role: {role}"}, status_code=400)
        req_id = body["req_id"]
        if not isinstance(req_id, int) or isinstance(req_id, bool):
            return self._response({"error": "req_id must be an integer"}, status_code=400)
        try:
            await self._coordinator.renew(role, req_id)
        except ValueError as e:
            return self._response({"error": str(e)}, status_code=503)
        except Exception as e:  # noqa: BLE001
            logger.error(f"CoordinatorServer.renew failed: {e}")
            return self._response({"error": str(e)}, status_code=500)
        return self._response({})

    async def cluster_info(self) -> Response:
        return self._response(await self._coordinator.cluster_info())

    async def health(self) -> Response:
        return Response(status_code=200 if await self._coordinator.is_ready() else 503)

    async def version(self) -> Response:
        return self._response({"version": VERSION})

    async def __call__(
        self,
        host: str,
        port: int,
        uds: Optional[str] = None,
        keep_alive_timeout: int = TIMEOUT_KEEP_ALIVE,
        sockets: Optional[list[socket.socket]] = None,
    ) -> None:
        # Single-process (owns routing state + the centralized ZMQ ingest bind);
        # workers=1 forced so a leaked WEB_CONCURRENCY can't fork it. When ``uds``
        # is set the co-located fleet uses it for the hot /select,/finish path
        # (avoids the TCP loopback overhead that dominated per-request latency).
        # keep_alive_timeout comes from the disaggregated config's
        # ``server_keep_alive_timeout`` so both listeners are tuned by one key.
        # ``sockets`` carries a socket the caller already bound, which is how the
        # other disaggregated listeners start. Letting uvicorn bind ``host``
        # itself means it resolves the name, and a name whose AAAA record is
        # link-local resolves to fe80:: with scope id 0 -- an address the kernel
        # always refuses ("invalid argument"), since the scope id can only come
        # from an interface name.
        kwargs = dict(workers=1, log_level="info", timeout_keep_alive=keep_alive_timeout)
        if uds:
            # uvicorn.Config binds uds XOR host:port, so run two Servers: UDS for
            # the fleet (hot path) and TCP for health/external clients.
            await self._coordinator.start()
            backlog = _effective_uds_backlog()
            if backlog < COORDINATOR_UDS_BACKLOG:
                log = logger.warning if backlog < UVICORN_DEFAULT_BACKLOG else logger.info
                log(
                    f"Coordinator UNIX socket accept queue is {backlog}, capped by "
                    f"net.core.somaxconn (requested {COORDINATOR_UDS_BACKLOG}); raise the "
                    "sysctl if routers report coordinator connect failures under bursts."
                )
            try:
                uds_cfg = uvicorn.Config(
                    self.app, uds=uds, lifespan="off", backlog=COORDINATOR_UDS_BACKLOG, **kwargs
                )
                tcp_cfg = uvicorn.Config(self.app, host=host, port=port, lifespan="off", **kwargs)
                await self._serve(
                    [
                        (create_uvicorn_server(uds_cfg), None),
                        (create_uvicorn_server(tcp_cfg), sockets),
                    ]
                )
            finally:
                await self._coordinator.stop()
        else:
            config = uvicorn.Config(self.app, host=host, port=port, **kwargs)
            await self._serve([(create_uvicorn_server(config), sockets)])

    async def _serve(
        self, servers: list[tuple[uvicorn.Server, Optional[list[socket.socket]]]]
    ) -> None:
        """Run ``servers`` (with their sockets); fail the seat waiters once one of them stops.

        uvicorn's graceful shutdown waits for every request in flight, and a pooled /select
        waiting for a generation seat ends only at its seat or its KV transfer deadline (none
        without one), so the lifespan exit or the caller's ``stop()`` would come too late.
        """
        self.servers = [server for server, _ in servers]
        watcher = asyncio.create_task(self._fail_seat_waiters_at_shutdown())
        try:
            # A listener without sockets (UDS) binds its own.
            await asyncio.gather(
                *(
                    server.serve() if socks is None else server.serve(sockets=socks)
                    for server, socks in servers
                )
            )
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    async def _fail_seat_waiters_at_shutdown(self) -> None:
        while not any(server.should_exit for server in self.servers):
            await asyncio.sleep(SHUTDOWN_POLL_S)
        await self._coordinator.fail_seat_waiters()


def serve_coordinator(host: str, port: int, coordinator: DisaggCoordinatorService) -> None:
    asyncio.run(CoordinatorServer(coordinator)(host, port))
