# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/launcher.py
#
# Lightweight HTTP server launcher providing graceful shutdown semantics.
# It wires SIGINT/SIGTERM to uvicorn shutdown and awaits in-flight requests before exiting.

import asyncio
import signal
import socket
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI

from tensorrt_llm.logger import logger


async def serve_http(
    app: FastAPI,
    sock: Optional[socket.socket] = None,
    enable_ssl_refresh: bool = False,
    **uvicorn_kwargs: Any,
):
    """Start a FastAPI app using Uvicorn with graceful shutdown.

    - Stop accepting new connections on SIGINT/SIGTERM
    - Wait for in-flight requests to complete
    - Shutdown cleanly

    Parameters mirror vLLM's serve_http for easier operational parity.
    """
    # Log available routes for easier debugging
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if methods is None or path is None:
            continue
        logger.info("Route: %s, Methods: %s", path, ", ".join(methods))

    config = uvicorn.Config(app, **uvicorn_kwargs)
    config.load()
    server = uvicorn.Server(config)

    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve(sockets=[sock] if sock else None))

    def signal_handler() -> None:
        # Prevent uvicorn's default from exiting early; cancel our task and
        # then let our cancellation path invoke server.shutdown()
        if not server_task.done():
            server_task.cancel()

    # Register signal handlers
    try:
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
    except NotImplementedError:
        # add_signal_handler may not be available on some platforms
        signal.signal(signal.SIGINT, lambda *_: signal_handler())
        signal.signal(signal.SIGTERM, lambda *_: signal_handler())

    async def _dummy_shutdown() -> None:
        pass

    try:
        await server_task
        # Normal server exit path: return a coroutine that does nothing
        return _dummy_shutdown()
    except asyncio.CancelledError:
        logger.info("Shutting down FastAPI HTTP server.")
        # Return the shutdown coroutine so caller can await it after cleaning up
        return server.shutdown()
