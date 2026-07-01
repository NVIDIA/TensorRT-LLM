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

    POST /select   {"role", "routing_key", "exclude_server"}
      -> {"server": "host:port", "info": {...}, "handle": <str|null>}
    POST /finish   {"role", "handle", "success"}  -> {}
    GET  /cluster_info -> {...}
    GET  /health   -> 200 when ready
    GET  /version

The routing key is produced client-side by ``Router.extract_routing_key`` and
consumed here by ``Router.select_by_key`` (see ``serve/router.py``), so this
endpoint is generic across router types (centralized -> block hashes,
conversation -> conversation_id, round-robin -> empty). Single-process by design;
it owns the ZMQ ingest bind for centralized mode.
"""

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.version import __version__ as VERSION

TIMEOUT_KEEP_ALIVE = 10  # seconds


class CoordinatorServer:
    """Serve a :class:`DisaggCoordinatorService`'s coordination API over HTTP."""

    def __init__(self, cluster_manager: DisaggCoordinatorService) -> None:
        self._cluster = cluster_manager

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self._cluster.start()
            yield
            await self._cluster.stop()

        self.app = FastAPI(lifespan=lifespan)
        self.app.add_api_route("/select", self.select, methods=["POST"])
        self.app.add_api_route("/finish", self.finish, methods=["POST"])
        self.app.add_api_route("/cluster_info", self.cluster_info,
                               methods=["GET"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])

    async def select(self, raw_req: Request) -> Response:
        try:
            body = await raw_req.json()
        except Exception as e:
            return JSONResponse(status_code=400,
                                content={"error": f"invalid JSON body: {e}"})
        if not isinstance(body, dict) or "role" not in body:
            return JSONResponse(
                status_code=400,
                content={"error": "body must include 'role' and 'routing_key'"})
        try:
            server, info, handle = await self._cluster.select(
                body["role"], body.get("routing_key"),
                body.get("exclude_server"))
        except ValueError as e:
            return JSONResponse(status_code=503, content={"error": str(e)})
        except Exception as e:  # noqa: BLE001
            logger.error(f"CoordinatorServer.select failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
        return JSONResponse(content={"server": server, "info": info,
                                     "handle": handle})

    async def finish(self, raw_req: Request) -> Response:
        try:
            body = await raw_req.json()
        except Exception as e:
            return JSONResponse(status_code=400,
                                content={"error": f"invalid JSON body: {e}"})
        await self._cluster.finish(body.get("role", "gen"), body.get("handle"),
                                   body.get("success", True))
        return JSONResponse(content={})

    async def cluster_info(self) -> Response:
        return JSONResponse(content=await self._cluster.cluster_info())

    async def health(self) -> Response:
        return Response(status_code=200 if await self._cluster.is_ready()
                        else 503)

    async def version(self) -> Response:
        return JSONResponse(content={"version": VERSION})

    async def __call__(self, host: str, port: int) -> None:
        # Single-process by design: owns routing state + the centralized ZMQ
        # ingest bind. workers=1 forced so a leaked WEB_CONCURRENCY can't fork it.
        config = uvicorn.Config(self.app, host=host, port=port, workers=1,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()


def serve_coordinator(host: str, port: int,
                      cluster_manager: DisaggCoordinatorService) -> None:
    asyncio.run(CoordinatorServer(cluster_manager)(host, port))
