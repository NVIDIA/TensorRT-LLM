# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEngine gRPC server lifecycle for TensorRT-LLM."""

import asyncio
import signal

import grpc
import uvloop
from openengine.v1 import openengine_pb2_grpc

from tensorrt_llm.logger import logger

__all__ = ["OpenEngineServer", "launch_server"]


def _format_bind_address(host: str, port: int) -> str:
    """Format a host and port as a gRPC bind address."""
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        host = f"[{host}]"
    return f"{host}:{port}"


class OpenEngineServer:
    """OpenEngine gRPC server with intentionally unimplemented RPCs.

    Args:
        host: Interface on which the server listens.
        port: Port on which the server listens. Use zero to select a free port.
    """

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._server = grpc.aio.server()
        openengine_pb2_grpc.add_InferenceServicer_to_server(
            openengine_pb2_grpc.InferenceServicer(), self._server
        )
        openengine_pb2_grpc.add_ControlServicer_to_server(
            openengine_pb2_grpc.ControlServicer(), self._server
        )
        self._bind_address = _format_bind_address(host, port)
        bound_port = self._server.add_insecure_port(self._bind_address)
        if bound_port == 0:
            raise RuntimeError(f"Failed to bind OpenEngine server to {self._bind_address}")
        if port == 0:
            self.port = bound_port

    async def start(self) -> None:
        """Start accepting OpenEngine requests."""
        await self._server.start()
        address = _format_bind_address(self.host, self.port)
        logger.info(f"OpenEngine stub server started on {address}")

    async def stop(self, grace: float = 5.0) -> None:
        """Stop accepting OpenEngine requests.

        Args:
            grace: Maximum time in seconds to allow active RPCs to finish.
        """
        await self._server.stop(grace=grace)
        logger.info("OpenEngine stub server stopped")

    async def wait_for_termination(self) -> None:
        """Wait until the OpenEngine server terminates."""
        await self._server.wait_for_termination()


def launch_server(host: str, port: int) -> None:
    """Launch the dedicated OpenEngine gRPC server.

    Args:
        host: Interface on which the server listens.
        port: Port on which the server listens.
    """

    async def serve() -> None:
        server = OpenEngineServer(host=host, port=port)
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def signal_handler() -> None:
            logger.info("Received shutdown signal")
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        try:
            logger.warning(
                "OpenEngine protocol support is a stub: no model is loaded and "
                "all RPCs return UNIMPLEMENTED."
            )
            await server.start()
            await stop_event.wait()
        finally:
            await server.stop()

    uvloop.run(serve())
