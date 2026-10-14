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
from collections.abc import Callable
from typing import Any

import click
import grpc
import uvloop

from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm.logger import logger
from tensorrt_llm.serve._frontend_processes import (
    FrontendStartupCancelled,
    _init_multi_frontend_mode,
    _signal_frontend_ready,
    _spawn_attached_frontends,
    _terminate_attached_frontends,
)

from .bindings import openengine_pb2_grpc
from .control import OpenEngineControlServicer
from .coordinator import CoordinationError, Coordinator, FrontendClient
from .servicer import OpenEngineInferenceServicer

__all__ = ["OpenEngineServer", "launch_server"]

# Raise the gRPC 4 MiB default so large tokenized prompts and prompt-logprob
# responses are not rejected at the transport layer. Bounded (not unlimited) so
# it still guards against pathological payloads.
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
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


def _format_bind_address(host: str, port: int) -> str:
    """Format a host and port as a gRPC bind address."""
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        host = f"[{host}]"
    return f"{host}:{port}"


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
    ) -> None:
        self.host = host
        self.port = port
        options = list(_SERVER_OPTIONS)
        options.append(("grpc.so_reuseport", 1 if frontend is not None else 0))
        self._server = grpc.aio.server(options=options)
        kv_transfer_backend = _kv_transfer_backend(llm)
        inference = OpenEngineInferenceServicer(
            llm, model, kv_transfer_backend=kv_transfer_backend, frontend=frontend
        )
        openengine_pb2_grpc.add_InferenceServicer_to_server(inference, self._server)
        # Control shares the inference servicer's in-flight request table so
        # Abort and GetLoad see the same requests Generate is serving.
        openengine_pb2_grpc.add_ControlServicer_to_server(
            OpenEngineControlServicer(
                llm,
                model,
                inference,
                kv_transfer_backend=kv_transfer_backend,
                frontend=frontend,
                instance_id=instance_id,
            ),
            self._server,
        )
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
        bound_port = self._server.add_insecure_port(bind_address)
        if port == 0:
            self.port = bound_port

    async def start(self) -> None:
        """Start accepting OpenEngine requests."""
        await self._server.start()
        address = _format_bind_address(self.host, self.port)
        logger.info(f"OpenEngine server started on {address}")

    async def stop(self, grace: float = 5.0) -> None:
        """Stop accepting OpenEngine requests.

        Args:
            grace: Maximum time in seconds to allow active RPCs to finish.
        """
        await self._server.stop(grace=grace)
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
) -> None:
    """Launch the dedicated OpenEngine gRPC server.

    Args:
        host: Interface on which the server listens.
        port: Port on which the server listens.
        llm_args: Arguments for LLM initialization.
        served_model_name: Model name accepted by Generate. Defaults to the model path.
        report_failure: Records a child exit before startup readiness.
    """

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
        children = []
        failure = None
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
            try:
                return not stop_event.is_set() and bool(llm._check_health())
            except (RuntimeError, ValueError, OSError):
                return False

        async def supervise() -> None:
            group_was_ready = False
            while not stop_event.is_set():
                if mode.is_attached_frontend and os.getppid() != parent_pid:
                    fail("OpenEngine launcher parent exited")
                    return
                if any(child.poll() is not None for child in children):
                    fail("An OpenEngine frontend exited; stopping the serving group")
                    return
                if not healthy():
                    fail("OpenEngine engine health check failed")
                    return
                if frontend is not None:
                    try:
                        status = await frontend.request("heartbeat", frontend=frontend.frontend_id)
                    except CoordinationError:
                        return
                    if status["stopping"]:
                        stop_event.set()
                        return
                    if group_was_ready and not status["ready"]:
                        fail("OpenEngine frontend group became unavailable")
                        return
                    group_was_ready = status["ready"]
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

        def monitor_done(task: asyncio.Task) -> None:
            if not task.cancelled() and (error := task.exception()) is not None:
                fail(f"OpenEngine supervision failed: {error}")

        try:
            llm = PyTorchLLM(**llm_args)
            if stop_event.is_set():
                if failure is not None:
                    raise RuntimeError(failure)
                return
            logger.info("Model loaded successfully")
            instance_id = str(llm.llm_id)
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
            )
            _disable_gc_if_requested()
            await server.start()
            if any(child.poll() is not None for child in children):
                raise RuntimeError("An OpenEngine frontend exited during startup")
            _signal_frontend_ready(mode)
            if coordinator is not None:
                coordinator.ready = True
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
                        if directory is not None:
                            directory.cleanup()
                        for sig, previous in previous_handlers.items():
                            signal.signal(sig, previous)
                logger.info("LLM engine stopped")
        if failure is not None:
            raise RuntimeError(failure)

    uvloop.run(serve())
