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
    POST /finish   {"role", "req_id", "success"}  -> {}
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
from contextlib import asynccontextmanager
from typing import Optional

import msgpack
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.cluster_storage import HttpClusterStorageServer
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.version import __version__ as VERSION

TIMEOUT_KEEP_ALIVE = 10  # seconds
MSGPACK_MEDIA_TYPE = "application/msgpack"


class CoordinatorServer:
    """Serve a :class:`DisaggCoordinatorService`'s coordination API over HTTP."""

    def __init__(self, coordinator: DisaggCoordinatorService) -> None:
        self._coordinator = coordinator

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self._coordinator.start()
            yield
            await self._coordinator.stop()

        self.app = FastAPI(lifespan=lifespan)
        self.app.add_api_route("/select", self.select, methods=["POST"])
        self.app.add_api_route("/finish", self.finish, methods=["POST"])
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
        if not isinstance(body, dict) or "role" not in body or "routing_key" not in body:
            return self._response(
                {"error": "body must include 'role' and 'routing_key'"}, status_code=400
            )
        role = body["role"]
        if role not in ("context", "ctx", "generation", "gen"):
            return self._response({"error": f"invalid role: {role}"}, status_code=400)
        req_id = body.get("req_id")
        if req_id is not None and (not isinstance(req_id, int) or isinstance(req_id, bool)):
            return self._response({"error": "req_id must be an integer"}, status_code=400)
        exclude_server = body.get("exclude_server")
        if exclude_server is not None and not isinstance(exclude_server, str):
            return self._response({"error": "exclude_server must be a string"}, status_code=400)
        try:
            server, info, req_id = await self._coordinator.select(
                role, body["routing_key"], req_id, exclude_server
            )
        except ValueError as e:
            return self._response({"error": str(e)}, status_code=503)
        except Exception as e:  # noqa: BLE001
            logger.error(f"CoordinatorServer.select failed: {e}")
            return self._response({"error": str(e)}, status_code=500)
        return self._response({"server": server, "info": info, "req_id": req_id})

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

    async def cluster_info(self) -> Response:
        return self._response(await self._coordinator.cluster_info())

    async def health(self) -> Response:
        return Response(status_code=200 if await self._coordinator.is_ready() else 503)

    async def version(self) -> Response:
        return self._response({"version": VERSION})

    async def __call__(self, host: str, port: int, uds: Optional[str] = None) -> None:
        # Single-process (owns routing state + the centralized ZMQ ingest bind);
        # workers=1 forced so a leaked WEB_CONCURRENCY can't fork it. When ``uds``
        # is set the co-located fleet uses it for the hot /select,/finish path
        # (avoids the TCP loopback overhead that dominated per-request latency).
        kwargs = dict(workers=1, log_level="info", timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        if uds:
            # uvicorn.Config binds uds XOR host:port, so run two Servers: UDS for
            # the fleet (hot path) and TCP for health/external clients.
            import asyncio as _asyncio

            await self._coordinator.start()
            try:
                uds_cfg = uvicorn.Config(self.app, uds=uds, lifespan="off", **kwargs)
                tcp_cfg = uvicorn.Config(self.app, host=host, port=port, lifespan="off", **kwargs)
                await _asyncio.gather(
                    uvicorn.Server(uds_cfg).serve(), uvicorn.Server(tcp_cfg).serve()
                )
            finally:
                await self._coordinator.stop()
        else:
            config = uvicorn.Config(self.app, host=host, port=port, **kwargs)
            await uvicorn.Server(config).serve()


def serve_coordinator(host: str, port: int, coordinator: DisaggCoordinatorService) -> None:
    asyncio.run(CoordinatorServer(coordinator)(host, port))
