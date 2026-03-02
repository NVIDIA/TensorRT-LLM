# Copyright (c) 2025, NVIDIA CORPORATION.
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
#!/usr/bin/env python

# yapf: disable
import asyncio
import signal
import socket
import traceback
from contextlib import asynccontextmanager
from typing import Callable, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.executor import CppExecutorError
from tensorrt_llm.llmapi import tracing
from tensorrt_llm.llmapi.disagg_utils import (DisaggServerConfig,
                                              MetadataServerConfig, ServerRole,
                                              get_ctx_gen_server_addrs)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.cluster_storage import (HttpClusterStorageServer,
                                                create_cluster_storage)
from tensorrt_llm.serve.metadata_server import create_metadata_server
from tensorrt_llm.serve.openai_client import OpenAIClient, OpenAIHttpClient
from tensorrt_llm.serve.openai_disagg_service import (
    OpenAIDisaggregatedService, ResponseHooks)
from tensorrt_llm.serve.openai_protocol import (UCompletionRequest,
                                                UCompletionResponse)
from tensorrt_llm.serve.perf_metrics import DisaggPerfMetricsCollector
from tensorrt_llm.serve.responses_utils import (ServerArrivalTimeMiddleware,
                                                get_steady_clock_now_in_seconds)
from tensorrt_llm.serve.router import Router, create_router
from tensorrt_llm.version import __version__ as VERSION

# yapf: enale
TIMEOUT_KEEP_ALIVE = 10  # seconds.

class RawRequestResponseHooks(ResponseHooks):
    def __init__(self, raw_req: Request, perf_metrics_collector: DisaggPerfMetricsCollector):
        self.raw_req = raw_req
        self.ctx_server = ""
        self.gen_server = ""
        self.request_arrival_time = raw_req.state.server_arrival_time
        self.server_first_token_time = 0
        self.perf_metrics_collector = perf_metrics_collector

    def on_req_begin(self, request: UCompletionRequest):
        self.perf_metrics_collector.queue_latency_seconds.observe(get_steady_clock_now_in_seconds() - self.request_arrival_time)

    def on_ctx_resp(self, ctx_server: str, response: UCompletionResponse):
        self.ctx_server = ctx_server

    def on_first_token(self, gen_server: str, request: UCompletionRequest, response: UCompletionResponse = None):
        self.gen_server = gen_server
        self.server_first_token_time = get_steady_clock_now_in_seconds()

    def on_resp_done(self, gen_server: str, request: UCompletionRequest, response: UCompletionResponse = None):
        if request.disaggregated_params:
            ctx_req_id = request.disaggregated_params.ctx_request_id
            asyncio.create_task(self.perf_metrics_collector.add_per_request_metrics(self.ctx_server, gen_server, ctx_req_id, self.raw_req.state.server_arrival_time, self.server_first_token_time))


class OpenAIDisaggServer:

    def __init__(self,
                 config: DisaggServerConfig,
                 req_timeout_secs: int = 180,
                 server_start_timeout_secs: int = 180,
                 metadata_server_cfg: Optional[MetadataServerConfig] = None,
                 metrics_interval_secs: int = 0):
        self._config = config
        self._req_timeout_secs = req_timeout_secs
        self._server_start_timeout_secs = server_start_timeout_secs
        self._metadata_server_cfg = metadata_server_cfg
        self._metrics_interval_secs = metrics_interval_secs

        self._ctx_servers, self._gen_servers = get_ctx_gen_server_addrs(config.server_configs)
        self._ctx_router = create_router(config.ctx_router_config, self._ctx_servers, metadata_server_cfg, create_metadata_server(metadata_server_cfg), self._sync_server_clock)
        self._gen_router = create_router(config.gen_router_config, self._gen_servers, metadata_server_cfg, create_metadata_server(metadata_server_cfg), self._sync_server_clock)
        self._metadata_server = create_metadata_server(metadata_server_cfg)
        self._perf_metrics_collector = DisaggPerfMetricsCollector(config.perf_metrics_max_requests)

        self._disagg_cluster_storage = create_cluster_storage(config.disagg_cluster_config.cluster_uri, config.disagg_cluster_config.cluster_name) if config.disagg_cluster_config else None

        self._service = OpenAIDisaggregatedService(
            self._config, self._ctx_router, self._gen_router, self._create_client,
            metadata_server=self._metadata_server,
            metadata_config=self._metadata_server_cfg,
            req_timeout_secs=self._req_timeout_secs,
            server_start_timeout_secs=self._server_start_timeout_secs,
            perf_metrics_collector=self._perf_metrics_collector,
            disagg_cluster_storage=self._disagg_cluster_storage)

        try:
            otlp_cfg = config.otlp_config
            if otlp_cfg and otlp_cfg.otlp_traces_endpoint:
                tracing.init_tracer("trt.llm", otlp_cfg.otlp_traces_endpoint)
                logger.info(
                    f"Initialized OTLP tracer successfully, endpoint: {otlp_cfg.otlp_traces_endpoint}"
                )
        except Exception as e:
            logger.error(f"Failed to initialize OTLP tracer: {e}")


        @asynccontextmanager
        async def lifespan(app) -> None:
            # Prepare servers (sync server clock) when static ctx/gen server list is used
            await self._ctx_router.prepare_servers()
            await self._gen_router.prepare_servers()
            await self._service.setup()
            yield
            await self._service.teardown()

        self.app = FastAPI(lifespan=lifespan)

        self.app.add_middleware(ServerArrivalTimeMiddleware)

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(_, exc):
            self._perf_metrics_collector.validation_exceptions.inc()
            return JSONResponse(status_code=400, content={"error": str(exc)})

        self.register_routes()

    def _create_client(self, router: Router, role: ServerRole, max_retries: int = 1) -> OpenAIClient:
        client = OpenAIHttpClient(router, role, self._req_timeout_secs, max_retries)
        self._perf_metrics_collector.add_client(client)
        return client

    def register_routes(self):
        self.app.add_api_route("/v1/completions", self._wrap_entry_point(self._service.openai_completion), methods=["POST"])
        self.app.add_api_route("/v1/chat/completions", self._wrap_entry_point(self._service.openai_chat_completion), methods=["POST"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/cluster_info", self.cluster_info, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/perf_metrics", self._perf_metrics_collector.get_perf_metrics, methods=["GET"])
        # import prometheus_client lazily to break the `set_prometheus_multiproc_dir`
        from prometheus_client import make_asgi_app
        self.app.mount("/prometheus/metrics", make_asgi_app())
        if self._disagg_cluster_storage and isinstance(self._disagg_cluster_storage, HttpClusterStorageServer):
            self._disagg_cluster_storage.add_routes(self.app)

    def _wrap_entry_point(self, entry_point: Callable) -> Callable:
        async def wrapper(req: UCompletionRequest, raw_req: Request) -> Response:
            try:
                self._perf_metrics_collector.total_requests.inc()
                if req.stream:
                    self._perf_metrics_collector.stream_requests.inc()
                else:
                    self._perf_metrics_collector.nonstream_requests.inc()
                hooks = RawRequestResponseHooks(raw_req, self._perf_metrics_collector)
                response_or_generator = await entry_point(req, hooks)
                self._perf_metrics_collector.total_responses.inc()
                if req.stream:
                    return StreamingResponse(content=response_or_generator, media_type="text/event-stream")
                else:
                    return JSONResponse(content=response_or_generator.model_dump())
            except Exception as e:
                self._handle_exception(e)
        return wrapper

    def _handle_exception(self, exception):
        if isinstance(exception, CppExecutorError):
            logger.error("CppExecutorError: ", traceback.format_exc())
            signal.raise_signal(signal.SIGINT)
        elif isinstance(exception, HTTPException):
            self._perf_metrics_collector.http_exceptions.inc()
            logger.error(f"HTTPException {exception.status_code} {exception.detail}: ", traceback.format_exc())
            raise exception
        else:
            self._perf_metrics_collector.internal_errors.inc()
            logger.error("Internal server error: ", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Internal server error {str(exception)}")


    async def health(self) -> Response:
        if not await self._service.is_ready():
            return Response(status_code=500)
        return Response(status_code=200)

    async def cluster_info(self) -> JSONResponse:
        return JSONResponse(content=await self._service.cluster_info())

    async def version(self) -> JSONResponse:
        return JSONResponse(content={"version": VERSION})

    async def __call__(self, host: str, port: int, sockets: list[socket.socket] | None = None):
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level=logger.level,
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve(sockets=sockets)

    async def _sync_server_clock(self, server: str):
        """ Sync the ctx/gen server's steady clock with the disagg-server's steady clock (in case NTP service is not running). """
        async def query_steady_clock_offset(session: aiohttp.ClientSession, server_url: str) -> tuple[Optional[float], Optional[float]]:
            try:
                originate_ts = get_steady_clock_now_in_seconds()
                async with session.get(server_url) as response:
                    destination_ts = get_steady_clock_now_in_seconds()
                    if response.status == 200:
                        response_content = await response.json()
                        # Compute the steady clock timestamp difference using the NTP clock synchronization algorithm. https://en.wikipedia.org/wiki/Network_Time_Protocol#Clock_synchronization_algorithm
                        receive_ts = response_content['receive_ts']
                        transmit_ts = response_content['transmit_ts']
                        delay = (destination_ts - originate_ts) - (transmit_ts - receive_ts)
                        offset = ((receive_ts - originate_ts) + (transmit_ts - destination_ts)) / 2
                        return delay, offset
                    else:
                        return None, None
            except Exception:
                return None, None

        async def set_steady_clock_offset(session: aiohttp.ClientSession, server_url: str, offset: float) -> None:
            payload = {"offset": offset}
            async with session.post(server_url, json=payload) as response:
                if response.status != 200:
                    logger.warning(f"Cannot set disagg server steady clock offset for server {server_url}, the perf metrics timestamps could be mis-aligned")

        async def align_steady_clock_offset(session: aiohttp.ClientSession, server_url: str) -> None:
            delay, offset = await query_steady_clock_offset(session, server_url)
            if delay is None or offset is None:
                logger.warning(f"Unable to measure steady clock offset for {server_url}; skipping adjustment")
                return
            logger.info(f'Server: {server_url}, delay: {delay} second, offset: {offset} second')
            # Negate the offset so that worker servers can adjust their steady clock by adding the new offset
            await set_steady_clock_offset(session, server_url, -offset)

        server_scheme = "http://" if not server.startswith("http://") else ""
        server_url = f"{server_scheme}{server}/steady_clock_offset"

        try:
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=True),
                timeout=aiohttp.ClientTimeout(total=self._req_timeout_secs)) as session:
                await align_steady_clock_offset(session, server_url)
        except (aiohttp.ClientError, OSError) as e:
            logger.warning(f"Unable to align steady clock offset for {server_url}: {e}; skipping adjustment")
