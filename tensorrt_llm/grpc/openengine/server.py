# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEngine gRPC server lifecycle for TensorRT-LLM."""

import asyncio
import gc
import ipaddress
import os
import signal
import socket
import tempfile
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import click
import grpc
import uvloop
from grpc_health.v1 import health_pb2, health_pb2_grpc

from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm.grpc._protobuf import _warn_if_python_protobuf
from tensorrt_llm.inputs.media_io import _close_aiohttp_session
from tensorrt_llm.logger import logger
from tensorrt_llm.serve._frontend_processes import (
    FrontendStartupCancelled,
    _init_multi_frontend_mode,
    _signal_frontend_ready,
    _spawn_attached_frontends,
    _terminate_attached_frontends,
)
from tensorrt_llm.serve._perf_metrics_config import resolve_perf_metrics_config

from ._health import OpenEngineHealthServicer
from .bindings import openengine_pb2_grpc
from .control import OpenEngineControlServicer
from .coordinator import CoordinationError, Coordinator, FrontendClient
from .kv_events import KvEventsUnavailableError, _format_host_port, events_config
from .metrics import MetricsDirectory, OpenEngineMetrics
from .servicer import OpenEngineInferenceServicer

__all__ = ["OpenEngineServer", "launch_server"]

# Raise the gRPC 4 MiB default so large tokenized prompts and prompt-logprob
# responses are not rejected at the transport layer. Bounded (not unlimited) so
# it still guards against pathological payloads.
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_HEALTH_CHECK_INTERVAL_SECONDS = 1.0
_HEALTH_CHECK_TIMEOUT_SECONDS = 5.0
_SERVER_OPTIONS = [
    ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
    ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
    # Keepalive so the server detects a vanished client on a long-lived streaming
    # RPC and intermediaries don't drop an idle (slow-decode) stream.
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.http2.min_ping_interval_without_data_ms", 10000),
]


_format_bind_address = _format_host_port


def _is_loopback(host: str) -> bool:
    """Whether `host` resolves to a loopback address."""
    cleaned = host.strip("[]")
    if not cleaned or cleaned in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def _kv_transfer_backend(llm: Any) -> str:
    """Name of the KV cache transfer backend, or "" when disaggregation is off.

    Presence is decided by the config object, not by `backend`: that field is
    Optional and defaults to None, and leaving it unset is the documented way to
    take the default transceiver. Keying on it would make a correctly configured
    disagg worker advertise kv_connector.enabled=False, so a router doing
    capability discovery would never send it a remote prefill.
    """
    cache_config = getattr(getattr(llm, "args", None), "cache_transceiver_config", None)
    if cache_config is None:
        return ""
    backend = getattr(cache_config, "backend", None)
    return str(backend) if backend else "DEFAULT"


class OpenEngineServer:
    """OpenEngine gRPC server backed by the TensorRT-LLM LLM API.

    Args:
        host: Interface on which the server listens.
        port: Port on which the server listens. Use zero to select a free port.
        llm: Initialized TensorRT-LLM LLM instance.
        model: Model name accepted by Generate requests.
        subagent_affinity_auth_key: Shared key for authenticated parent affinity hints.
    """

    def __init__(
        self,
        host: str,
        port: int,
        llm: Any,
        model: str,
        *,
        frontend: FrontendClient | None = None,
        instance_id: str | None = None,
        subagent_affinity_auth_key: str | None = None,
        metrics: OpenEngineMetrics | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._llm = llm
        self._metrics = metrics
        self._frontend = frontend
        self._stopping = False
        self._health_task: asyncio.Task | None = None
        self._engine_health_task: asyncio.Task[bool] | None = None
        self._health_refresh_lock = asyncio.Lock()
        self._inference_serving = False
        self._engine_healthy = False
        self._fatal_engine_failure = False
        options = list(_SERVER_OPTIONS)
        options.append(("grpc.so_reuseport", 1 if frontend is not None else 0))
        self._server = grpc.aio.server(options=options)
        self._input_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="openengine_input"
        )
        self._media_executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="openengine_media"
        )
        self._health = OpenEngineHealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(self._health, self._server)
        kv_transfer_backend = _kv_transfer_backend(llm)
        inference = OpenEngineInferenceServicer(
            llm,
            model,
            kv_transfer_backend=kv_transfer_backend,
            frontend=frontend,
            subagent_affinity_auth_key=subagent_affinity_auth_key,
            metrics=metrics,
            input_executor=self._input_executor,
            media_executor=self._media_executor,
        )
        openengine_pb2_grpc.add_InferenceServicer_to_server(inference, self._server)
        # Control shares the inference servicer's in-flight request table so
        # Abort and GetLoad see the same requests Generate is serving.
        self._control = OpenEngineControlServicer(
            llm,
            model,
            inference,
            kv_transfer_backend=kv_transfer_backend,
            frontend=frontend,
            instance_id=instance_id,
        )
        openengine_pb2_grpc.add_ControlServicer_to_server(self._control, self._server)
        bind_address = _format_bind_address(host, port)
        # Plaintext h2c with no authentication: any client that can reach this
        # port can run inference and call Control.Abort. It is meant to be
        # colocated with its caller on loopback, or fronted by a proxy that
        # terminates TLS and authenticates.
        if not _is_loopback(host):
            logger.warning(
                f"OpenEngine server is binding to {bind_address}, which is not loopback. "
                "The listener is unauthenticated and unencrypted: restrict it to a trusted "
                "network or front it with an authenticating TLS proxy."
            )
        # grpc.aio raises RuntimeError itself when the bind fails, naming the
        # address, so there is no zero return to check for.
        try:
            bound_port = self._server.add_insecure_port(bind_address)
        except BaseException:
            self._input_executor.shutdown(wait=True, cancel_futures=True)
            self._media_executor.shutdown(wait=True, cancel_futures=True)
            raise
        if port == 0:
            self.port = bound_port

    async def start(self) -> None:
        """Start accepting OpenEngine requests."""
        if events_config(self._llm) is not None:
            capacity = await asyncio.to_thread(self._llm._executor.get_kv_cache_capacity)
            if not capacity.get("kvEventsEnabled", False):
                raise KvEventsUnavailableError(
                    "KV events are configured but the loaded engine has no streaming publisher; "
                    "set kv_cache_config.use_kv_cache_manager_v2: true and use a supported model"
                )
        if self._metrics is not None:
            await self._metrics.start()
        await self._server.start()
        await self._health.set("liveness", health_pb2.HealthCheckResponse.SERVING)
        await self._refresh_inference_health()
        self._health_task = asyncio.create_task(self._monitor_inference_health())
        address = _format_bind_address(self.host, self.port)
        logger.info(f"OpenEngine server started on {address}")

    async def _refresh_inference_health(self) -> None:
        async with self._health_refresh_lock:
            ready = False
            if not self._stopping:
                try:
                    ready = await self._check_engine_health()
                    self._engine_healthy = ready
                    if ready and self._frontend is not None:
                        status = await self._frontend.request("status")
                        ready = bool(status["ready"] and not status["stopping"])
                except (RuntimeError, ValueError, OSError, CoordinationError) as error:
                    logger.warning(f"OpenEngine inference health check failed: {error}")
                    self._engine_healthy = False
                executor = getattr(self._llm, "_executor", None)
                if executor is not None and (
                    getattr(executor, "_fatal_error", None) is not None
                    or getattr(executor, "_engine_dead", False)
                ):
                    self._fatal_engine_failure = True
            if self._stopping:
                return
            ready = ready and not self._fatal_engine_failure
            if ready != self._inference_serving:
                self._inference_serving = ready
                await self._health.set(
                    "inference",
                    health_pb2.HealthCheckResponse.SERVING
                    if ready
                    else health_pb2.HealthCheckResponse.NOT_SERVING,
                )
            if self._fatal_engine_failure:
                await self._health.set("liveness", health_pb2.HealthCheckResponse.NOT_SERVING)

    async def _check_engine_health(self) -> bool:
        task = self._engine_health_task
        if task is not None:
            if not task.done():
                return False
            self._engine_health_task = None
            # The timed-out result is stale; start a fresh check instead.
            task.exception()

        task = asyncio.create_task(asyncio.to_thread(self._control._engine_is_healthy))
        self._engine_health_task = task
        task.add_done_callback(self._consume_engine_health_result)
        done, _ = await asyncio.wait((task,), timeout=_HEALTH_CHECK_TIMEOUT_SECONDS)
        if not done:
            logger.warning("OpenEngine inference health check timed out")
            return False
        self._engine_health_task = None
        return task.result()

    @staticmethod
    def _consume_engine_health_result(task: asyncio.Task[bool]) -> None:
        if not task.cancelled():
            task.exception()

    async def _monitor_inference_health(self) -> None:
        while not self._stopping:
            await asyncio.sleep(_HEALTH_CHECK_INTERVAL_SECONDS)
            try:
                await self._refresh_inference_health()
            except Exception as error:
                logger.error(f"OpenEngine inference health monitor failed: {error}")
                self._engine_healthy = False
                self._inference_serving = False
                await self._health.set("inference", health_pb2.HealthCheckResponse.NOT_SERVING)

    async def stop(self, grace: float = 5.0) -> None:
        """Stop accepting OpenEngine requests.

        Args:
            grace: Maximum time in seconds to allow active RPCs to finish.
        """
        self._stopping = True
        try:
            await self._health.enter_graceful_shutdown()
            if self._health_task is not None:
                self._health_task.cancel()
                await asyncio.gather(self._health_task, return_exceptions=True)
            await self._server.stop(grace=grace)
        finally:
            try:
                if self._metrics is not None:
                    await self._metrics.stop()
            finally:
                try:
                    await _close_aiohttp_session()
                finally:
                    await asyncio.gather(
                        asyncio.to_thread(
                            self._input_executor.shutdown, wait=True, cancel_futures=True
                        ),
                        asyncio.to_thread(
                            self._media_executor.shutdown, wait=True, cancel_futures=True
                        ),
                    )
        logger.info("OpenEngine server stopped")

    async def wait_for_termination(self) -> None:
        """Wait until the OpenEngine server terminates."""
        await self._server.wait_for_termination()


def _disable_gc_if_requested() -> None:
    """Disable Python's cyclic garbage collector when TRTLLM_SERVER_DISABLE_GC=1.

    Same switch and policy as the ``trtllm-serve`` HTTP server: a cyclic
    collection pauses the event loop, and with it every stream this process
    serves.
    """
    if os.getenv("TRTLLM_SERVER_DISABLE_GC", "0") == "1":
        gc.disable()
        logger.info("Python cyclic GC disabled (TRTLLM_SERVER_DISABLE_GC=1)")


def launch_server(
    host: str,
    port: int,
    llm_args: dict[str, Any],
    served_model_name: str | None = None,
    report_failure: Callable[[int, str, str], None] | None = None,
    enable_load_metrics: bool = False,
    subagent_affinity_auth_key: str | None = None,
    metrics_host: str | None = None,
    metrics_port: int = 8001,
) -> None:
    """Launch the dedicated OpenEngine gRPC server.

    Args:
        host: Interface on which the server listens.
        port: Port on which the server listens.
        llm_args: Arguments for LLM initialization.
        served_model_name: Model name accepted by Generate. Defaults to the model path.
        report_failure: Records a child exit before startup readiness.
        enable_load_metrics: Enable scheduler-owned routing-load snapshots, independently of KV events.
        subagent_affinity_auth_key: Shared key for authenticated parent affinity hints.
        metrics_host: HTTP bind address; defaults to the inference host.
        metrics_port: HTTP port when performance-metrics configuration enables Prometheus.
    """
    _warn_if_python_protobuf()

    async def serve() -> None:
        logger.info("Initializing TensorRT-LLM OpenEngine server...")
        backend = llm_args.get("backend")
        model = served_model_name or llm_args.get("model", "")
        llm_args.pop("build_config", None)
        if backend != "pytorch":
            raise click.BadParameter(
                f"{backend} is not a known backend, check help for available options.",
                param_hint="backend",
            )

        perf_metrics = resolve_perf_metrics_config(
            llm_args.get("return_perf_metrics", False),
            llm_args.get("perf_metrics_output_dir"),
        )
        metrics_enabled = perf_metrics.collect_request_metrics
        if metrics_enabled:
            llm_args["return_perf_metrics"] = True
        if metrics_enabled and metrics_port == port and port != 0:
            raise click.UsageError("--openengine-metrics-port must differ from --port")

        mode = _init_multi_frontend_mode(llm_args, enabled=True)
        if mode.num_frontends > 1 and port == 0:
            raise click.UsageError("Multiple OpenEngine frontends require a fixed --port")
        if mode.is_launcher:
            family = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][0]
            try:
                with socket.create_server((host, port), family=family):
                    pass
            except OSError as error:
                raise RuntimeError(f"Failed to bind {host}:{port}: {error}") from error
        stop_event = asyncio.Event()
        startup_cancelled = threading.Event()
        server = llm = coordinator = frontend = monitor = None
        directory = None
        metrics_directory = metrics = None
        children = []
        failure = None
        serving = False
        parent_pid = os.getppid()

        def fail(reason: str) -> None:
            nonlocal failure
            if not stop_event.is_set():
                failure = reason
                logger.error(reason)
            startup_cancelled.set()
            stop_event.set()

        def signal_handler(signum: int, frame: Any) -> None:
            # A Python handler also observes signals while synchronous model
            # initialization is running. Cleanup starts at its next safe boundary.
            startup_cancelled.set()
            stop_event.set()

        previous_handlers = {
            sig: signal.signal(sig, signal_handler) for sig in (signal.SIGTERM, signal.SIGINT)
        }

        def healthy() -> bool:
            return (
                not stop_event.is_set()
                and server is not None
                and server._engine_healthy
                and not server._fatal_engine_failure
            )

        def terminal_engine_failure() -> bool:
            executor = getattr(llm, "_executor", None)
            return executor is not None and (
                getattr(executor, "_fatal_error", None) is not None
                or getattr(executor, "_engine_dead", False)
            )

        async def supervise() -> None:
            while not stop_event.is_set():
                if mode.is_attached_frontend and os.getppid() != parent_pid:
                    fail("OpenEngine launcher parent exited")
                    return
                if any(child.poll() is not None for child in children):
                    fail("An OpenEngine frontend exited; stopping the serving group")
                    return
                if server is None:
                    engine_ready = await asyncio.wait_for(
                        asyncio.to_thread(llm._check_health),
                        timeout=_HEALTH_CHECK_TIMEOUT_SECONDS,
                    )
                    if server is None and not engine_ready:
                        fail("OpenEngine engine health check failed during startup")
                        return
                if terminal_engine_failure():
                    fail("OpenEngine terminal engine failure")
                    return
                if serving and metrics is not None:
                    metrics.check_health()
                if frontend is not None:
                    try:
                        status = await frontend.request("heartbeat", frontend=frontend.frontend_id)
                    except CoordinationError:
                        return
                    if status["stopping"]:
                        stop_event.set()
                        return
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

        def monitor_done(task: asyncio.Task) -> None:
            if not task.cancelled() and (error := task.exception()) is not None:
                fail(f"OpenEngine supervision failed: {error}")

        try:
            if metrics_enabled:
                metrics_directory = MetricsDirectory(mode.is_attached_frontend)
            llm_args["_enable_routing_load"] = enable_load_metrics
            # A new incarnation fences every node's metadata after a restart.
            # The executor starts node discovery only when streaming is enabled.
            llm_args["_openengine_discovery"] = {
                "engine_id": str(uuid.uuid4()),
                "host": host,
                "port": port,
            }
            llm = PyTorchLLM(**llm_args)
            if stop_event.is_set():
                if failure is not None:
                    raise RuntimeError(failure)
                return
            logger.info("Model loaded successfully")
            if metrics_enabled:
                metrics = OpenEngineMetrics(
                    llm,
                    model,
                    host if metrics_host is None else metrics_host,
                    metrics_port,
                    owner=not mode.is_attached_frontend,
                )
            instance_id = llm_args["_openengine_discovery"]["engine_id"]
            if mode.is_launcher:
                directory = tempfile.TemporaryDirectory(prefix="tllm-openengine-")
                longest_name = max("coordinator", f"frontend-{mode.num_frontends - 1}", key=len)
                socket_path = os.path.join(directory.name, longest_name)
                if len(os.fsencode(socket_path)) >= 108:
                    raise ValueError(
                        f"OpenEngine private Unix socket path exceeds 107 bytes: {socket_path}. "
                        "Set TMPDIR to a shorter directory."
                    )
                coordinator = Coordinator(
                    directory.name, mode.num_frontends, instance_id, healthy, fail
                )
                await coordinator.start()
                frontend = FrontendClient(directory.name, 0, fail)
            elif mode.is_attached_frontend:
                frontend = FrontendClient(
                    os.environ.pop("TLLM_OPENENGINE_COORDINATOR"),
                    int(os.environ["TLLM_EXECUTOR_FRONTEND_ID"]),
                    fail,
                )
                instance_id = os.environ.pop("TLLM_OPENENGINE_INSTANCE_ID")
            if frontend is not None:
                await frontend.start()
            # Start supervision before waiting for children, so engine death or
            # coordinator failure cannot leave startup waiting on a ready pipe.
            monitor = asyncio.create_task(supervise())
            monitor.add_done_callback(monitor_done)
            if mode.is_launcher:
                try:
                    children = await asyncio.to_thread(
                        _spawn_attached_frontends,
                        llm,
                        mode.num_frontends,
                        extra_env={
                            "TLLM_OPENENGINE_COORDINATOR": directory.name,
                            "TLLM_OPENENGINE_INSTANCE_ID": instance_id,
                        },
                        cancelled=startup_cancelled,
                        report_failure=report_failure,
                    )
                except FrontendStartupCancelled:
                    # The spawn helper has already reaped its children. The
                    # stop check below distinguishes SIGTERM from engine failure.
                    pass
            if stop_event.is_set():
                if failure is not None:
                    raise RuntimeError(failure)
                return
            server = OpenEngineServer(
                host=host,
                port=port,
                llm=llm,
                model=model,
                frontend=frontend,
                instance_id=instance_id,
                subagent_affinity_auth_key=subagent_affinity_auth_key,
                metrics=metrics,
            )
            _disable_gc_if_requested()
            await server.start()
            serving = True
            if any(child.poll() is not None for child in children):
                raise RuntimeError("An OpenEngine frontend exited during startup")
            _signal_frontend_ready(mode)
            if coordinator is not None:
                coordinator.ready = True
                await server._refresh_inference_health()
            await stop_event.wait()
        finally:
            startup_cancelled.set()
            if coordinator is not None:
                coordinator.ready = False
                coordinator.stopping = True
            elif mode.is_attached_frontend and frontend is not None:
                try:
                    await frontend.request("withdraw", frontend=frontend.frontend_id)
                except CoordinationError:
                    pass
            if monitor is not None:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
            try:
                if server is not None:
                    await server.stop()
            finally:
                try:
                    if children:
                        await asyncio.to_thread(_terminate_attached_frontends, children)
                    if frontend is not None:
                        await frontend.close()
                    if coordinator is not None:
                        await coordinator.close()
                finally:
                    try:
                        if llm is not None:
                            await asyncio.to_thread(llm.shutdown)
                    finally:
                        if metrics is not None:
                            await metrics.wait_for_collection()
                        if directory is not None:
                            directory.cleanup()
                        if metrics_directory is not None:
                            metrics_directory.close()
                        for sig, previous in previous_handlers.items():
                            signal.signal(sig, previous)
                logger.info("LLM engine stopped")
        if failure is not None:
            raise RuntimeError(failure)

    uvloop.run(serve())
