# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prometheus collection and the optional, leader-owned HTTP listener."""

import asyncio
import os
import socket
import tempfile
import threading
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import grpc

from tensorrt_llm.logger import logger
from tensorrt_llm.metrics.collector import MetricsCollector

from .coordinator import CoordinationError

if TYPE_CHECKING:
    from tensorrt_llm.disaggregated_params import DisaggregatedParams
    from tensorrt_llm.llmapi.llm import RequestOutput

    from .bindings.generation_pb2 import GenerateResponse

_DIRECTORY_ENV = "TLLM_OPENENGINE_METRICS_DIR"
_COLLECTION_INTERVAL_SECONDS = 1.0


class MetricsDirectory:
    """Own a fresh serving-group directory; attached frontends only inherit it.

    The operator's PROMETHEUS_MULTIPROC_DIR is a parent directory, never a
    directory whose existing contents this server may delete or aggregate.
    """

    def __init__(self, attached: bool) -> None:
        self._directory = None
        self._previous = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
        if attached:
            path = os.environ.get(_DIRECTORY_ENV)
            if not path or path != self._previous or not os.path.isdir(path):
                raise RuntimeError("OpenEngine frontend has no shared Prometheus directory")
        else:
            self._directory = tempfile.TemporaryDirectory(
                prefix="tllm-openengine-metrics-", dir=self._previous
            )
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = self._directory.name
            os.environ[_DIRECTORY_ENV] = self._directory.name

    def close(self) -> None:
        """Remove owned storage after all serving-group processes have stopped."""
        if self._directory is not None:
            self._directory.cleanup()
            os.environ.pop(_DIRECTORY_ENV, None)
            if self._previous is None:
                os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            else:
                os.environ["PROMETHEUS_MULTIPROC_DIR"] = self._previous


class RpcOutcome(str, Enum):
    REJECTED = "rejected"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    ENGINE_ERROR = "engine_error"
    TIMEOUT = "timeout"
    STALLED = "stalled"


class RequestMetrics:
    """Own one RPC's classification and exactly-once outcome recording."""

    def __init__(self, metrics: "OpenEngineMetrics") -> None:
        self._metrics = metrics
        self._phase: Literal["aggregated", "prefill", "decode"] = "aggregated"
        self._finished = False
        self._outcome = RpcOutcome.REJECTED

    def begin(self, params: "DisaggregatedParams | None") -> None:
        if params is None:
            return
        if params.request_type == "context_only":
            self._phase = "prefill"
        elif params.request_type == "generation_only":
            self._phase = "decode"
        else:
            self._phase = "aggregated"

    def accepted(self) -> None:
        # An accepted request that closes before a terminal result is cancelled.
        self._outcome = RpcOutcome.CANCELLED

    def observe_exception(self, error: BaseException) -> None:
        if self._finished:
            return
        # RPC cancellation and coordination failure describe the eventual
        # transport result, even if the engine already completed or failed.
        if isinstance(error, asyncio.CancelledError):
            self._outcome = RpcOutcome.CANCELLED
        elif isinstance(error, CoordinationError):
            self._outcome = RpcOutcome.UNAVAILABLE
        else:
            self._outcome = RpcOutcome.ENGINE_ERROR

    def engine_failed(self) -> None:
        """An engine operation failed; its exception type does not change its origin."""
        if not self._finished:
            self._outcome = RpcOutcome.ENGINE_ERROR

    def observe_response(self, response: "GenerateResponse") -> None:
        if response.HasField("error"):
            self.engine_failed()

    def stalled(self) -> None:
        if not self._finished:
            self._outcome = RpcOutcome.STALLED
            self.finish()

    def complete(self, result: "RequestOutput", *, handoff_ready: bool) -> None:
        if self._finished:
            return
        # An error already sent to the client survives normal engine completion.
        if self._outcome != RpcOutcome.ENGINE_ERROR:
            reasons = {output.finish_reason for output in result.outputs}
            if "cancelled" in reasons:
                self._outcome = RpcOutcome.CANCELLED
            elif "timeout" in reasons:
                self._outcome = RpcOutcome.TIMEOUT
            elif self._phase == "prefill" and not handoff_ready:
                self._outcome = RpcOutcome.ENGINE_ERROR
            else:
                self._outcome = RpcOutcome.COMPLETED
        # Engine candidate accounting is independent of the RPC's outcome.
        self._metrics.record_result(result)

    def finish(self, status: grpc.StatusCode | None = None) -> None:
        """Record at most once, including when a suspended stream resumes later."""
        if self._finished:
            return
        if status == grpc.StatusCode.CANCELLED:
            self._outcome = RpcOutcome.CANCELLED
        elif status == grpc.StatusCode.DEADLINE_EXCEEDED:
            self._outcome = RpcOutcome.TIMEOUT
        self._finished = True
        self._metrics.record_outcome(self._phase, self._outcome.value)


class OpenEngineMetrics:
    """Collect frontend metrics; only the engine owner serves and drains stats.

    Args:
        llm: Initialized LLM with engine performance statistics enabled.
        model: Stable served model label; request-supplied model aliases are ignored.
        host: HTTP bind address.
        port: HTTP port, or zero for an ephemeral port in embedded use.
        owner: Whether this process owns the engine and HTTP endpoint.
    """

    def __init__(self, llm: Any, model: str, host: str, port: int, owner: bool) -> None:
        # All imports follow MetricsDirectory creation and precede construction
        # of any metric, including in re-executed attached frontends.
        from prometheus_client import CollectorRegistry, Counter, make_asgi_app, multiprocess
        from prometheus_client.values import ValueClass

        if not ValueClass._multiprocess:
            raise RuntimeError(
                "OpenEngine metrics require PROMETHEUS_MULTIPROC_DIR before importing "
                "prometheus_client; start a fresh trtllm-serve process"
            )
        self._llm = llm
        self.host = host
        self.port = port
        self.owner = owner
        self._collect_requests = llm.args.return_perf_metrics
        self._collect_iterations = llm.args.enable_iter_perf_stats
        registry = CollectorRegistry()
        self.collector = MetricsCollector.from_args(
            llm.args, model, registry=registry, gauge_multiprocess_mode="livemostrecent"
        )
        self._outcomes = Counter(
            "trtllm_openengine_rpc_outcomes_total",
            "Generate RPC outcomes, independent of output-candidate completion metrics.",
            ["model_name", "phase", "outcome"],
            registry=registry,
        )
        self._model = model
        self._errors = Counter(
            "trtllm_openengine_metrics_errors_total",
            "Metrics collection failures; inference results are preserved.",
            ["model_name", "source"],
            registry=registry,
        )
        self._last_error_log = 0.0
        self._http = None
        self._thread = None
        self._socket = None
        self._task = None
        self._read_task = None
        self._stopping = threading.Event()
        if owner:
            self.collector.log_config_from_args(llm.args, model)
            scrape_registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(scrape_registry)
            prometheus_app = make_asgi_app(registry=scrape_registry)

            async def app(scope, receive, send) -> None:
                if scope["path"] != "/prometheus/metrics":
                    await send({"type": "http.response.start", "status": 404, "headers": []})
                    await send({"type": "http.response.body", "body": b"Not found"})
                    return
                await prometheus_app(scope, receive, send)

            import uvicorn

            self._http = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=host,
                    port=port,
                    lifespan="off",
                    access_log=False,
                    limit_concurrency=4,
                    backlog=8,
                    timeout_keep_alive=5,
                    timeout_graceful_shutdown=2,
                )
            )

    def _record_error(self, source: str, error: Exception) -> None:
        # Telemetry errors must not turn successful generation into engine errors.
        if self._stopping.is_set():
            return
        try:
            self._errors.labels(self._model, source).inc()
        except Exception:
            pass
        now = time.monotonic()
        if now - self._last_error_log >= 30:
            self._last_error_log = now
            logger.warning(f"OpenEngine metrics {source} failed: {error}")

    def record_outcome(self, phase: str, outcome: str) -> None:
        """Record an RPC result without affecting inference."""
        try:
            self._outcomes.labels(self._model, phase, outcome).inc()
        except Exception as error:
            self._record_error("request", error)

    def record_result(self, result: Any) -> None:
        """Record newly completed candidates, following the HTTP collector contract."""
        if not self._collect_requests:
            return
        try:
            candidates = getattr(result, "candidate_metrics", None)
            if candidates:
                for metrics in candidates:
                    self.collector.log_request_metrics_dict(metrics)
            elif metrics := getattr(result, "metrics_dict", None):
                self.collector.log_request_metrics_dict(metrics)
        except Exception as error:
            self._record_error("request", error)

    async def start(self) -> None:
        """Bind the HTTP listener and start periodic collection in the owner only."""
        if not self.owner:
            return
        bind_host = self.host.strip("[]")
        addresses = socket.getaddrinfo(
            bind_host or None, self.port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )
        address = next(
            (address for address in addresses if address[0] == socket.AF_INET), addresses[0]
        )
        self._socket = socket.create_server(address[4], family=address[0], backlog=8)
        self.port = self._socket.getsockname()[1]
        self._thread = threading.Thread(
            target=self._http.run,
            kwargs={"sockets": [self._socket]},
            name="openengine-prometheus",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + 5
        while not self._http.started:
            if not self._thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("OpenEngine Prometheus listener failed to start")
            await asyncio.sleep(0.01)
        if self._collect_iterations:
            self._task = asyncio.create_task(self._collect(), name="openengine-metrics")
        logger.info(f"OpenEngine Prometheus endpoint: {self.host}:{self.port}/prometheus/metrics")

    def check_health(self) -> None:
        """Fail supervision if the requested endpoint or collection task exited."""
        if self.owner and (
            self._thread is None
            or not self._thread.is_alive()
            or (self._collect_iterations and (self._task is None or self._task.done()))
        ):
            raise RuntimeError("OpenEngine Prometheus service stopped unexpectedly")

    async def _collect(self) -> None:
        while True:
            # The proxy's ostensibly async stats API performs a synchronous RPC.
            # Both that read and metric-file updates stay off the serving loop.
            self._read_task = asyncio.create_task(asyncio.to_thread(self._collect_once))
            await asyncio.shield(self._read_task)
            await asyncio.sleep(_COLLECTION_INTERVAL_SECONDS)

    def _collect_once(self) -> None:
        try:
            for stats in self._llm.get_stats(timeout=0.1):
                if self._stopping.is_set():
                    return
                self.collector.log_iteration_stats(stats)
        except Exception as error:
            self._record_error("iteration", error)

    async def stop(self) -> None:
        """Stop tasks and listener before the launcher removes shared storage."""
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if self._http is not None:
            self._http.should_exit = True
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 5)
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("OpenEngine Prometheus listener did not stop")

    async def wait_for_collection(self) -> None:
        """Join the last read after engine shutdown releases pending executor RPCs."""
        if self._read_task is not None:
            await asyncio.gather(self._read_task, return_exceptions=True)
