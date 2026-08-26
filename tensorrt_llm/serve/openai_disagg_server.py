# Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.executor import CppExecutorError
from tensorrt_llm.llmapi import tracing
from tensorrt_llm.llmapi.disagg_utils import (DisaggServerConfig,
                                              MetadataServerConfig, ServerRole)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve._telemetry import create_uvicorn_server
from tensorrt_llm.serve.cluster_storage import (
    HttpClusterStorageServer, create_cluster_storage,
    validate_http_cluster_storage_scope)
from tensorrt_llm.serve.conversation_id import resolve_request_conversation_id
from tensorrt_llm.serve.disagg_coordinator import (CoordinatorClient,
                                                   DisaggCoordinatorService)
from tensorrt_llm.serve.openai_client import OpenAIClient, OpenAIHttpClient
from tensorrt_llm.serve.openai_disagg_service import (
    OpenAIDisaggregatedService, ResponseHooks)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest, CompletionRequest, UCompletionRequest,
    UCompletionResponse, ensure_request_chat_template_allowed)
from tensorrt_llm.serve.perf_metrics import (DisaggPerfMetricsCollector,
                                             PerfMetricsJsonlWriter,
                                             PerfMetricsMiddleware,
                                             combine_disagg_metrics)
from tensorrt_llm.serve.responses_utils import (ServerArrivalTimeMiddleware,
                                                get_steady_clock_now_in_seconds)
from tensorrt_llm.serve.router import Router
from tensorrt_llm.usage import TerminalOutcome, record_termination_observation
from tensorrt_llm.version import __version__ as VERSION

# yapf: enale
_LOG_CONTROL_CHARACTERS = {
    code: f"\\x{code:02x}"
    for code in (*range(32), 127)
}

# Most round trips to spend estimating one ctx/gen server's steady-clock
# offset. Only the least-delayed sample is kept; see `_sync_server_clock`.
# Each probe costs ~0.2 s because the `/steady_clock_offset` handler sleeps
# between its two timestamps, and servers are prepared sequentially, so the
# loop stops early (below) instead of always spending the full budget.
_CLOCK_SYNC_PROBES = 5

# A round trip this fast is already conclusive -- its offset estimate is
# accurate to +/- 2.5 ms -- so stop probing. This is the common case on a
# healthy server, keeping the added startup cost to one extra round trip.
_CLOCK_SYNC_GOOD_DELAY_SECONDS = 0.005

# Per-request timeout for the handshake, and a wall-clock budget for the whole
# probe loop. Servers are prepared one at a time, so an unresponsive worker must
# not be able to stall startup for `_req_timeout_secs` once per probe. A healthy
# handshake takes ~0.2 s per round trip.
_CLOCK_SYNC_REQUEST_TIMEOUT_SECONDS = 10.0
_CLOCK_SYNC_TOTAL_BUDGET_SECONDS = 15.0

# Largest round-trip delay for which the estimated offset is still worth
# applying. The NTP estimate is only accurate to +/- delay/2, so this caps the
# error the handshake can inject into perf-metric timestamps at 5 ms. Co-located
# servers share CLOCK_MONOTONIC (true offset exactly 0) and NTP-synced hosts are
# aligned to well under a millisecond, so discarding a noisier estimate is
# strictly better than applying it.
_CLOCK_SYNC_MAX_DELAY_SECONDS = 0.010

class RawRequestResponseHooks(ResponseHooks):
    def __init__(self, raw_req: Request, queue_latency_metric,
                 collect_perf_metrics: bool):
        self.raw_req = raw_req
        self.queue_latency_metric = queue_latency_metric
        self.collect_perf_metrics = collect_perf_metrics
        self.ctx_server = ""
        self.gen_server = ""
        self.request_id = ""
        self.disagg_request_id = None
        self.request_arrival_time = raw_req.state.server_arrival_time
        self.server_first_token_time = 0
        self.ctx_dispatch_time = 0
        self.ctx_metrics = None
        self.gen_metrics = None

    def on_req_begin(self, request: UCompletionRequest):
        params = request.disaggregated_params
        if params is not None:
            self.disagg_request_id = params.disagg_request_id
            request_id = params.disagg_request_id or params.ctx_request_id
            self.request_id = str(request_id or "")
        self.queue_latency_metric.observe(
            get_steady_clock_now_in_seconds() - self.request_arrival_time)

    def on_disagg_request_id(self, disagg_request_id: int):
        self.disagg_request_id = disagg_request_id
        self.request_id = str(disagg_request_id)

    def on_ctx_dispatch(self, request: UCompletionRequest):
        self.ctx_dispatch_time = get_steady_clock_now_in_seconds()

    def on_perf_metrics(self, server: str, role: str, metrics: dict):
        if role == "ctx":
            self.ctx_server = server
            self.ctx_metrics = metrics
        elif role == "gen":
            self.gen_server = server
            self.gen_metrics = metrics

    def on_ctx_resp(self, ctx_server: str, response: UCompletionResponse):
        self.ctx_server = ctx_server

    def on_first_token(
            self, gen_server: str, request: UCompletionRequest,
            response: UCompletionResponse = None):
        self.gen_server = gen_server
        self.server_first_token_time = get_steady_clock_now_in_seconds()

    def on_resp_done(
            self, gen_server: str, request: UCompletionRequest,
            response: UCompletionResponse = None):
        self.gen_server = gen_server
        if not self.collect_perf_metrics:
            return
        disagg_phase = {
            "ctx_server": self.ctx_server,
            "gen_server": self.gen_server,
            "timing_metrics": {
                "arrival_time": self.request_arrival_time,
                "last_token_time": get_steady_clock_now_in_seconds(),
                "server_arrival_time": self.request_arrival_time,
                "ctx_dispatch_time": self.ctx_dispatch_time or None,
                "server_first_token_time": self.server_first_token_time or None,
            },
        }
        self.raw_req.state.perf_metrics_records.append(
            combine_disagg_metrics(
                self.request_id,
                disagg_phase,
                self.ctx_metrics,
                self.gen_metrics,
                disagg_request_id=self.disagg_request_id,
            ))


class OpenAIDisaggServer:
    def __init__(self,
                 config: DisaggServerConfig,
                 req_timeout_secs: int = 180,
                 server_start_timeout_secs: int = 180,
                 metadata_server_cfg: Optional[MetadataServerConfig] = None,
                 metrics_interval_secs: int = 0,
                 coordinator_url: Optional[str] = None):
        self._config = config
        self._req_timeout_secs = req_timeout_secs
        self._server_start_timeout_secs = server_start_timeout_secs
        self._metadata_server_cfg = metadata_server_cfg
        self._metrics_interval_secs = metrics_interval_secs
        self._allow_request_chat_template = getattr(
            config, "allow_request_chat_template", False)
        # When set, this is a forked worker: routing/readiness are delegated to
        # the coordinator at coordinator_url (CoordinatorClient). Otherwise this
        # process owns the routers + cluster state (DisaggCoordinatorService).
        self._coordinator_url = coordinator_url

        self._perf_metrics_collector = DisaggPerfMetricsCollector(
            config.perf_metrics_max_requests)
        self._expose_perf_metrics = config.return_perf_metrics
        self._collect_perf_metrics = (
            config.return_perf_metrics
            or config.perf_metrics_output_dir is not None)
        self._perf_metrics_writer = PerfMetricsJsonlWriter(
            config.perf_metrics_output_dir, "disagg")

        self._disagg_cluster_storage = None
        if config.disagg_cluster_config:
            validate_http_cluster_storage_scope(
                config.disagg_cluster_config.cluster_uri, config.hostname)
            self._disagg_cluster_storage = create_cluster_storage(
                config.disagg_cluster_config.cluster_uri,
                config.disagg_cluster_config.cluster_name)
        # The server doesn't build routers -- the coordinator object does:
        # DisaggCoordinatorService (owner) or CoordinatorClient (delegating). The
        # server just reads .ctx_router / .gen_router off whichever it holds.
        if self._coordinator_url:
            self._coordinator = CoordinatorClient(
                self._coordinator_url, self._config, metadata_server_cfg,
                request_timeout_s=self._req_timeout_secs,
                startup_timeout_s=self._server_start_timeout_secs)
        else:
            self._coordinator = DisaggCoordinatorService(
                self._config, self._create_client,
                metadata_config=self._metadata_server_cfg,
                server_preparation_func=self._sync_server_clock,
                server_start_timeout_secs=self._server_start_timeout_secs)
        self._ctx_router = self._coordinator.ctx_router
        self._gen_router = self._coordinator.gen_router

        self._service = OpenAIDisaggregatedService(
            self._config, self._coordinator, self._create_client,
            req_timeout_secs=self._req_timeout_secs)

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
            # The cluster manager (via setup) owns server preparation + monitoring.
            await self._perf_metrics_writer.start()
            await self._service.setup()
            yield
            await self._service.teardown()
            await self._perf_metrics_writer.close()

        self.app = FastAPI(lifespan=lifespan)

        if self._collect_perf_metrics:
            self.app.add_middleware(
                PerfMetricsMiddleware,
                expose_headers=self._expose_perf_metrics,
                writer=self._perf_metrics_writer)
        self.app.add_middleware(ServerArrivalTimeMiddleware)

        # Log request-body validation failures so a client/server schema mismatch
        # shows up server-side. Throttled (first, then every 1000th) to avoid
        # flooding the event loop when every request fails identically.
        self._val_err_n = 0
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc):
            self._perf_metrics_collector.validation_exceptions.inc()
            self._val_err_n += 1
            if self._val_err_n == 1 or self._val_err_n % 1000 == 0:
                try:
                    errs = exc.errors()
                    # Compact: [{loc, type, msg}] -- drops the (large) echoed input.
                    brief = [{"loc": e.get("loc"), "type": e.get("type"),
                              "msg": e.get("msg")} for e in errs][:8]
                except Exception:  # noqa: BLE001
                    brief = str(exc)[:500]
                method = request.method.translate(_LOG_CONTROL_CHARACTERS)
                path = request.url.path.translate(_LOG_CONTROL_CHARACTERS)
                logger.warning(
                    f"[validation] {method} {path} 400 "
                    f"(n={self._val_err_n}): {brief}")
            return JSONResponse(status_code=400, content={"error": str(exc)})

        self.register_routes()

    def _create_client(self, router: Router, role: ServerRole, max_retries: int = 1) -> OpenAIClient:
        async def disagg_id_generator():
            return await self._coordinator.get_disagg_request_id()
        client = OpenAIHttpClient(
            router, role, self._req_timeout_secs, max_retries,
            disagg_id_generator=disagg_id_generator,
            request_perf_metrics=self._collect_perf_metrics,
            internal_disagg_auth_key=self._config.internal_request_auth_key)
        return client

    def register_routes(self):
        # The disagg service owns only the request-serving endpoints (/v1/*) and
        # perf metrics. Readiness / cluster topology are the coordinator's state,
        # so /health and /cluster_info hook straight to self._coordinator.
        self.app.add_api_route("/v1/completions", self._wrap_entry_point(self._service.openai_completion, CompletionRequest), methods=["POST"])
        self.app.add_api_route("/v1/chat/completions", self._wrap_entry_point(self._service.openai_chat_completion, ChatCompletionRequest), methods=["POST"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/cluster_info", self.cluster_info, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        # import prometheus_client lazily to break the `set_prometheus_multiproc_dir`
        from prometheus_client import (CollectorRegistry, make_asgi_app,
                                       multiprocess)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        self.app.mount("/prometheus/metrics", make_asgi_app(registry=registry))
        # Single-process (local coordinator): mount the in-process HTTP cluster
        # storage routes on this app. In worker mode the coordinator is remote and
        # owns those routes (CoordinatorClient has no cluster_storage).
        cluster_storage = getattr(self._coordinator, "cluster_storage", None)
        if isinstance(cluster_storage, HttpClusterStorageServer):
            cluster_storage.add_routes(self.app)
        elif (isinstance(self._coordinator, CoordinatorClient)
              and isinstance(self._disagg_cluster_storage,
                             HttpClusterStorageServer)):
            # Keep the configured public cluster_uri valid in fleet mode while
            # the coordinator remains the sole owner of the HTTP storage state.
            for path, method in (("/set", "POST"), ("/get", "GET"),
                                 ("/delete", "DELETE"), ("/expire", "GET"),
                                 ("/get_prefix", "GET")):
                self.app.add_api_route(path,
                                       self._proxy_cluster_storage_request,
                                       methods=[method])

    async def _proxy_cluster_storage_request(self,
                                             raw_req: Request) -> Response:
        try:
            body, status, content_type = (
                await self._coordinator.proxy_cluster_storage_request(
                    raw_req.method, raw_req.url.path,
                    list(raw_req.query_params.multi_items()),
                    await raw_req.body(), raw_req.headers.get("Content-Type")))
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.warning(f"Failed to proxy cluster storage request: {e}")
            return JSONResponse(status_code=502,
                                content={"error": "coordinator unavailable"})
        headers = {"Content-Type": content_type} if content_type else None
        return Response(content=body, status_code=status, headers=headers)

    @staticmethod
    def _extract_conversation_id(req: UCompletionRequest, raw_req: Request):
        """Populate conversation_params.conversation_id from supported headers.

        Body ``conversation_params.conversation_id`` is canonical. Headers are
        used only when the body does not provide an id.
        """
        resolve_request_conversation_id(req, raw_req.headers)

    def _wrap_entry_point(self, entry_point: Callable, request_type: type = UCompletionRequest) -> Callable:
        # Bind the concrete request model per route so FastAPI validates against it.
        # The bare Union UCompletionRequest (no discriminator) makes Pydantic try
        # CompletionRequest first and 400 every chat body, so override the wrapper's
        # annotation with request_type (as openai_server.py does).
        @tracing.trace_span("disaggregated_request")
        async def wrapper(req: request_type, raw_req: Request) -> Response:
            try:
                self._perf_metrics_collector.total_requests.inc()
                if req.stream:
                    self._perf_metrics_collector.stream_requests.inc()
                else:
                    self._perf_metrics_collector.nonstream_requests.inc()
                try:
                    ensure_request_chat_template_allowed(
                        req, self._allow_request_chat_template)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e)) from e
                self._extract_conversation_id(req, raw_req)
                hooks = RawRequestResponseHooks(
                    raw_req, self._perf_metrics_collector.queue_latency_seconds,
                    self._collect_perf_metrics)
                response_or_generator = await entry_point(req, hooks)
                self._perf_metrics_collector.total_responses.inc()
                if req.stream:
                    return StreamingResponse(
                        content=response_or_generator,
                        media_type="text/event-stream")
                return JSONResponse(content=response_or_generator.model_dump())
            except Exception as e:
                self._handle_exception(e)
        return wrapper

    def _handle_exception(self, exception):
        if isinstance(exception, CppExecutorError):
            logger.error("CppExecutorError: ", traceback.format_exc())
            record_termination_observation(
                TerminalOutcome(
                    termination_kind="worker_failure",
                    component="disagg_worker",
                    reporting_source="supervisor",
                    exit_code_known=False,
                ))
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
        if not await self._coordinator.is_ready():
            return Response(status_code=503)
        return Response(status_code=200)

    async def cluster_info(self) -> JSONResponse:
        return JSONResponse(content=await self._coordinator.cluster_info())

    async def version(self) -> JSONResponse:
        return JSONResponse(content={"version": VERSION})

    async def __call__(self, host: str, port: int, sockets: list[socket.socket] | None = None):
        keep_alive_timeout = self._config.server_keep_alive_timeout
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level=logger.level,
                                timeout_keep_alive=keep_alive_timeout)
        await create_uvicorn_server(config).serve(sockets=sockets)

    async def _sync_server_clock(self, server: str):
        """ Sync the ctx/gen server's steady clock with the disagg-server's steady clock (in case NTP service is not running).

        The offset is estimated with the NTP algorithm from an HTTP round trip,
        so its error is bounded by half the round-trip delay: a round trip whose
        two legs are asymmetric is indistinguishable from a clock offset. The
        handshake runs while the ctx/gen servers are still finishing startup, so
        a single sample regularly lands on a stalled event loop and yields tens
        of milliseconds of pure error, which is then baked into every perf-metric
        timestamp those servers report.

        Two mitigations, both standard NTP practice:

        * Probe ``_CLOCK_SYNC_PROBES`` times and keep the sample with the
          smallest delay (NTP's clock filter). The least-delayed round trip is
          the most symmetric one, so it carries the least error. A throwaway
          warm-up request first keeps DNS resolution and connection setup --
          which are paid entirely on the outbound leg -- out of the samples.
        * Skip the adjustment entirely when even the best sample is delayed by
          more than ``_CLOCK_SYNC_MAX_DELAY_SECONDS``. Past that point the
          estimate is worth less than the zero it would replace: co-located
          servers share CLOCK_MONOTONIC and are already exactly aligned, and a
          cross-host deployment running NTP is aligned to well under a
          millisecond.
        """
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
            # Warm-up probe: DNS resolution and connection setup are paid on the
            # outbound leg only, so folding them into a measured sample biases
            # the offset by half their cost. Its result is deliberately dropped.
            await query_steady_clock_offset(session, server_url)

            # NTP clock filter: keep the least-delayed round trip, since it is
            # the most symmetric one and hence carries the least error. Stop as
            # soon as a sample is conclusive so a healthy server costs one probe.
            best = None
            probes = 0
            deadline = get_steady_clock_now_in_seconds() + _CLOCK_SYNC_TOTAL_BUDGET_SECONDS
            for _ in range(_CLOCK_SYNC_PROBES):
                probes += 1
                sample = await query_steady_clock_offset(session, server_url)
                if sample[0] is None or sample[1] is None:
                    # The server is unreachable or erroring; retrying it four
                    # more times only delays startup for every later server.
                    break
                if best is None or sample[0] < best[0]:
                    best = sample
                if best[0] <= _CLOCK_SYNC_GOOD_DELAY_SECONDS:
                    break
                if get_steady_clock_now_in_seconds() >= deadline:
                    break
            if best is None:
                logger.warning(f"Unable to measure steady clock offset for {server_url}; skipping adjustment")
                return

            delay, offset = best
            logger.info(f'Server: {server_url}, delay: {delay} second, offset: {offset} second '
                        f'(best of {probes} probes)')
            if delay > _CLOCK_SYNC_MAX_DELAY_SECONDS:
                logger.warning(
                    f"Steady clock handshake with {server_url} was too slow to be conclusive "
                    f"(best round-trip delay {delay * 1e3:.1f} ms > {_CLOCK_SYNC_MAX_DELAY_SECONDS * 1e3:.1f} ms); "
                    f"the offset estimate is only accurate to +/-{delay / 2 * 1e3:.1f} ms, so it is discarded "
                    "rather than applied. Perf-metric timestamps stay on each server's own steady clock.")
                return
            # Negate the offset so that worker servers can adjust their steady clock by adding the new offset
            await set_steady_clock_offset(session, server_url, -offset)

        server_scheme = "http://" if not server.startswith("http://") else ""
        server_url = f"{server_scheme}{server}/steady_clock_offset"

        try:
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=True),
                timeout=aiohttp.ClientTimeout(total=min(
                    self._req_timeout_secs,
                    _CLOCK_SYNC_REQUEST_TIMEOUT_SECONDS))) as session:
                await align_steady_clock_offset(session, server_url)
        except (aiohttp.ClientError, OSError) as e:
            logger.warning(f"Unable to align steady clock offset for {server_url}: {e}; skipping adjustment")
