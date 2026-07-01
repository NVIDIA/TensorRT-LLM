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
"""Standalone HTTP server exposing a router's block-hash placement decision.

Wraps a router that can place a request from **pre-computed block hashes** and
serves the decision over HTTP. Tokenization and block-hashing happen in the
caller (the orchestrator / :class:`~tensorrt_llm.serve.router.RemoteHttpRouter`),
so only the hash list crosses the wire -- the request body (potentially tens of
thousands of tokens) never does, and all tokenization compute stays local to the
orchestrator:

    POST /select   {"block_hashes": [<int>, ...],
                    "exclude_server": "http://host:port" | null}
      -> {"server": "http://host:port",
          "matched_blocks": <int>,
          "dp_rank": <int> | null}

    GET  /health   -> 200 when at least one server is prepared
    GET  /servers  -> {"servers": [...], "prepared": <int>}
    GET  /version

The wrapped router must expose ``select_by_block_hashes(block_hashes,
exclude_server) -> (server, matched_blocks, dp_rank)`` (the centralized KV-cache
router does).
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm.llmapi.disagg_utils import MetadataServerConfig, RouterConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.metadata_server import (JsonDictionary,
                                                create_metadata_server)
from tensorrt_llm.serve.router import Router, create_router
from tensorrt_llm.version import __version__ as VERSION

TIMEOUT_KEEP_ALIVE = 10  # seconds


class RouterHttpServer:
    """Serve a :class:`Router`'s ``get_next_server`` decision over HTTP.

    Args:
        router: The router that owns the server pool and placement policy.
        monitor_interval_s: If > 0 and a metadata server is configured, poll it
            for live servers at this cadence; otherwise prepare the static list.
    """

    def __init__(self,
                 router: Router,
                 monitor_interval_s: float = 0.0) -> None:
        if not hasattr(router, "select_by_block_hashes"):
            raise TypeError(
                f"{type(router).__name__} does not support block-hash "
                "placement; RouterHttpServer requires a router exposing "
                "select_by_block_hashes() (e.g. CentralizedKVCacheRouter).")
        self._router = router
        self._monitor_interval_s = monitor_interval_s

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self._router.prepare_servers()
            if self._monitor_interval_s > 0:
                await self._router.start_server_monitoring(
                    self._monitor_interval_s)
            yield
            if self._monitor_interval_s > 0:
                await self._router.stop_server_monitoring()
            await self._router.close()

        self.app = FastAPI(lifespan=lifespan)
        self._register_routes()

    def _register_routes(self) -> None:
        self.app.add_api_route("/select", self.select, methods=["POST"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/servers", self.servers, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])

    async def select(self, raw_req: Request) -> Response:
        try:
            body = await raw_req.json()
        except Exception as e:
            return JSONResponse(status_code=400,
                                content={"error": f"invalid JSON body: {e}"})
        if not isinstance(body, dict) or "block_hashes" not in body:
            return JSONResponse(
                status_code=400,
                content={"error": "body must be {'block_hashes': [...]}"})
        block_hashes = body["block_hashes"]
        exclude_server = body.get("exclude_server")

        try:
            server, matched, dp_rank = self._router.select_by_block_hashes(
                block_hashes, exclude_server=exclude_server)
        except ValueError as e:
            # No available servers.
            return JSONResponse(status_code=503, content={"error": str(e)})
        except Exception as e:  # noqa: BLE001
            logger.error(f"RouterHttpServer.select failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

        return JSONResponse(content={
            "server": server,
            "matched_blocks": matched,
            "dp_rank": dp_rank,
        })

    async def health(self) -> Response:
        if self._router.num_prepared_servers > 0:
            return Response(status_code=200)
        return Response(status_code=503)

    async def servers(self) -> Response:
        return JSONResponse(content={
            "servers": list(self._router.servers),
            "prepared": self._router.num_prepared_servers,
        })

    async def version(self) -> Response:
        return JSONResponse(content={"version": VERSION})

    async def __call__(self, host: str, port: int) -> None:
        # The router server is intentionally single-process: it owns the routing
        # state (and, for a wrapped centralized router, the single ZMQ ingest
        # bind). workers=1 is forced so a leaked WEB_CONCURRENCY can never fork
        # it into multiple processes that would fragment state / collide on the
        # port.
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                workers=1,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()


def create_router_http_server(
    router_config: RouterConfig,
    servers: list[str],
    metadata_server_cfg: Optional[MetadataServerConfig] = None,
    metadata_server: Optional[JsonDictionary] = None,
    monitor_interval_s: float = 0.0,
) -> RouterHttpServer:
    """Build a :class:`RouterHttpServer` from a router config + server list."""
    if metadata_server is None:
        metadata_server = create_metadata_server(metadata_server_cfg)
    router = create_router(router_config, servers, metadata_server_cfg,
                           metadata_server)
    return RouterHttpServer(router, monitor_interval_s=monitor_interval_s)


def main(host: str = "0.0.0.0",
         port: int = 8080,
         router_config: Optional[RouterConfig] = None,
         servers: Optional[list[str]] = None,
         monitor_interval_s: float = 0.0) -> None:
    server = create_router_http_server(
        router_config or RouterConfig(),
        servers or [],
        monitor_interval_s=monitor_interval_s)
    asyncio.run(server(host, port))
