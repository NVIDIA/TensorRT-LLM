# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEngine gRPC server lifecycle for TensorRT-LLM."""

import asyncio
import ipaddress
import signal
from typing import Any

import click
import grpc
import uvloop
from openengine.v1 import openengine_pb2_grpc

from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm.logger import logger

from .control import OpenEngineControlServicer
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
    """Best-effort name of the KV cache transfer backend for disaggregation."""
    cache_config = getattr(getattr(llm, "args", None), "cache_transceiver_config", None)
    backend = getattr(cache_config, "backend", None)
    return str(backend) if backend is not None else ""


class OpenEngineServer:
    """OpenEngine gRPC server backed by the TensorRT-LLM LLM API.

    Args:
        host: Interface on which the server listens.
        port: Port on which the server listens. Use zero to select a free port.
        llm: Initialized TensorRT-LLM LLM instance.
        model: Model name accepted by Generate requests.
    """

    def __init__(self, host: str, port: int, llm: Any, model: str) -> None:
        self.host = host
        self.port = port
        self._server = grpc.aio.server(options=_SERVER_OPTIONS)
        kv_transfer_backend = _kv_transfer_backend(llm)
        inference = OpenEngineInferenceServicer(llm, model, kv_transfer_backend=kv_transfer_backend)
        openengine_pb2_grpc.add_InferenceServicer_to_server(inference, self._server)
        # Control shares the inference servicer's in-flight request table so
        # Abort and GetLoad see the same requests Generate is serving.
        openengine_pb2_grpc.add_ControlServicer_to_server(
            OpenEngineControlServicer(
                llm, model, inference, kv_transfer_backend=kv_transfer_backend
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
        bound_port = self._server.add_insecure_port(bind_address)
        if bound_port == 0:
            raise RuntimeError(f"Failed to bind OpenEngine server to {bind_address}")
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


def launch_server(
    host: str,
    port: int,
    llm_args: dict[str, Any],
    served_model_name: str | None = None,
) -> None:
    """Launch the dedicated OpenEngine gRPC server.

    Args:
        host: Interface on which the server listens.
        port: Port on which the server listens.
        llm_args: Arguments for LLM initialization.
        served_model_name: Model name accepted by Generate. Defaults to the model path.
    """

    async def serve() -> None:
        logger.info("Initializing TensorRT-LLM OpenEngine server...")
        backend = llm_args.get("backend")
        model = served_model_name or llm_args.get("model", "")
        llm_args.pop("build_config", None)
        if backend == "pytorch":
            llm = PyTorchLLM(**llm_args)
        elif backend == "_autodeploy":
            raise click.BadParameter(
                "OpenEngine generation does not support the AutoDeploy backend because "
                "AutoDeploy requests cannot currently be cancelled.",
                param_hint="backend",
            )
        else:
            raise click.BadParameter(
                f"{backend} is not a known backend, check help for available options.",
                param_hint="backend",
            )

        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        server = None

        def signal_handler() -> None:
            logger.info("Received shutdown signal")
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        try:
            logger.info("Model loaded successfully")
            server = OpenEngineServer(host=host, port=port, llm=llm, model=model)
            await server.start()
            await stop_event.wait()
        finally:
            try:
                if server is not None:
                    await server.stop()
            finally:
                if hasattr(llm, "shutdown"):
                    llm.shutdown()
                logger.info("LLM engine stopped")

    uvloop.run(serve())
