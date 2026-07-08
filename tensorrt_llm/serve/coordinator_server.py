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
import time
from contextlib import asynccontextmanager
from typing import Optional

import orjson
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse, Response

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.version import __version__ as VERSION

# Timing headers to decompose the coordinator IPC round-trip (co-located, so
# wall-clock time.time() is comparable): client stamps send time, coordinator
# stamps recv/send, from which the client derives req_wire/handler/resp_wire.
HDR_CLIENT_SEND = "x-client-send-time"
HDR_COORD_RECV = "x-coord-recv-time"
HDR_COORD_SEND = "x-coord-send-time"

# The coordinator is a single event loop serving /select,/finish for the whole
# fleet; each /select body carries a routing_key (a flat list of block hashes,
# hundreds of int64 for long prompts). stdlib json parse+serialize on that hot
# path keeps the loop busy and lengthens the queue (client-observed /select was
# ~hundreds of ms while the handler itself is ~0.1ms). orjson is 2-5x faster at
# both ends, freeing the loop sooner. JSONResponse -> ORJSONResponse and
# request bodies parsed with orjson.loads (bytes, no text decode).

TIMEOUT_KEEP_ALIVE = 10  # seconds


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

    def _timing_headers(self, raw_req: Request, recv_t: float) -> dict:
        """Echo the client send time + stamp coord recv/send times so the client
        can decompose the IPC round-trip. recv_t is captured at handler entry;
        send time is 'now' (just before responding).
        """
        h = {HDR_COORD_RECV: repr(recv_t), HDR_COORD_SEND: repr(time.time())}
        cs = raw_req.headers.get(HDR_CLIENT_SEND)
        if cs is not None:
            h[HDR_CLIENT_SEND] = cs
        return h

    async def select(self, raw_req: Request) -> Response:
        _recv = time.time()
        try:
            body = orjson.loads(await raw_req.body())
        except Exception as e:
            return ORJSONResponse(status_code=400, content={"error": f"invalid JSON body: {e}"})
        if not isinstance(body, dict) or "role" not in body:
            return ORJSONResponse(
                status_code=400, content={"error": "body must include 'role' and 'routing_key'"}
            )
        try:
            server, info, req_id = await self._coordinator.select(
                body["role"],
                body.get("routing_key"),
                body.get("req_id"),
                body.get("exclude_server"),
            )
        except ValueError as e:
            return ORJSONResponse(status_code=503, content={"error": str(e)})
        except Exception as e:  # noqa: BLE001
            logger.error(f"CoordinatorServer.select failed: {e}")
            return ORJSONResponse(status_code=500, content={"error": str(e)})
        return ORJSONResponse(
            content={"server": server, "info": info, "req_id": req_id},
            headers=self._timing_headers(raw_req, _recv),
        )

    async def finish(self, raw_req: Request) -> Response:
        _recv = time.time()
        try:
            body = orjson.loads(await raw_req.body())
        except Exception as e:
            return ORJSONResponse(status_code=400, content={"error": f"invalid JSON body: {e}"})
        await self._coordinator.finish(
            body.get("role", "gen"), body.get("req_id"), body.get("success", True)
        )
        return ORJSONResponse(content={}, headers=self._timing_headers(raw_req, _recv))

    async def cluster_info(self) -> Response:
        return ORJSONResponse(content=await self._coordinator.cluster_info())

    async def health(self) -> Response:
        return Response(status_code=200 if await self._coordinator.is_ready() else 503)

    async def version(self) -> Response:
        return ORJSONResponse(content={"version": VERSION})

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

            uds_cfg = uvicorn.Config(self.app, uds=uds, **kwargs)
            tcp_cfg = uvicorn.Config(self.app, host=host, port=port, **kwargs)
            await _asyncio.gather(uvicorn.Server(uds_cfg).serve(), uvicorn.Server(tcp_cfg).serve())
        else:
            config = uvicorn.Config(self.app, host=host, port=port, **kwargs)
            await uvicorn.Server(config).serve()


def serve_coordinator(host: str, port: int, coordinator: DisaggCoordinatorService) -> None:
    asyncio.run(CoordinatorServer(coordinator)(host, port))
