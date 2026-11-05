# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared assembly of the trtllm-serve HTTP apps (shared functions, no base class).

``build_http_app()`` fixes the order in which an app is put together; ``serve_uvicorn()``
runs it. Each server passes its own lifespan, route class, error handlers, routes and
middleware, so the differences between the aggregated server and the disaggregated
frontend stay explicit at the call site.

This module imports only FastAPI, Starlette, uvicorn, the standard library and
``serve/_telemetry.py``: it must not import ``serve/api/`` or either server module.
"""

import asyncio
import socket
from typing import Any, Callable, Coroutine, List, Optional, Sequence, Type, Union

import uvicorn
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.types import Lifespan

from tensorrt_llm.serve._telemetry import create_uvicorn_server

__all__ = ["AppInstaller", "build_http_app", "serve_uvicorn"]

# A callable that installs something (handlers, routes, middleware) into an app.
AppInstaller = Callable[[FastAPI], None]


def build_http_app(
    *,
    lifespan: Lifespan[FastAPI],
    routers: Sequence[AppInstaller],
    dialect_error_renderers: AppInstaller,
    middleware: AppInstaller,
    route_class: Optional[Type[APIRoute]] = None,
) -> FastAPI:
    """Build a FastAPI app in the order trtllm-serve relies on.

    1. ``FastAPI(lifespan=lifespan)``.
    2. ``route_class``, when given, on ``app.router``, before any route exists, so every
       route registered with ``app.add_api_route`` uses it. The aggregated server passes
       the msgpack-decoding route class and the disaggregated frontend the KV-hint policy
       route class; ``None`` keeps FastAPI's plain ``APIRoute``.
    3. ``dialect_error_renderers(app)``: the exception handlers.
    4. Each of ``routers``, in order. Route registration order is load-bearing (of two
       ``GET /metrics`` routes the first registered wins), and a router may add middleware
       of its own (the Prometheus instrumentator does).
    5. ``middleware(app)`` last, so its middleware ends up outside any middleware a router added
       (``add_middleware`` inserts at the front of the stack).

    User middleware from ``--middleware`` is not part of this: the CLI adds it to the
    returned app after construction, outermost.

    Args:
        lifespan: The app's lifespan context manager factory.
        routers: Callables that register routes on the app, run in order.
        dialect_error_renderers: Callable that registers the app's exception handlers.
        middleware: Callable that registers the app's built-in middleware.
        route_class: Route class for ``app.router``; ``None`` keeps FastAPI's default.

    Returns:
        The assembled app, ready for ``serve_uvicorn``.
    """
    app = FastAPI(lifespan=lifespan)
    if route_class is not None:
        app.router.route_class = route_class
    dialect_error_renderers(app)
    for install_routes in routers:
        install_routes(app)
    middleware(app)
    return app


async def serve_uvicorn(
    app: FastAPI,
    *,
    host: str,
    port: int,
    sockets: Optional[List[socket.socket]] = None,
    log_level: Union[str, int],
    timeout_keep_alive: int,
    after_start: Optional[Callable[[uvicorn.Server], Coroutine[Any, Any, None]]] = None,
) -> None:
    """Run ``app`` on uvicorn through ``create_uvicorn_server`` (the telemetry-aware server).

    Args:
        app: The app to serve.
        host: Bind host, passed to ``uvicorn.Config``.
        port: Bind port, passed to ``uvicorn.Config``.
        sockets: Pre-bound sockets to serve on instead of binding ``host:port``.
        log_level: uvicorn's log level, passed through unchanged.
        timeout_keep_alive: HTTP keep-alive timeout in seconds.
        after_start: When given, runs as a task next to the server. It receives the
            ``uvicorn.Server`` so it can wait for ``server.started`` or set
            ``server.should_exit``. The task is not kept.
    """
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        timeout_keep_alive=timeout_keep_alive,
    )
    server = create_uvicorn_server(config)
    if after_start is not None:
        asyncio.create_task(after_start(server))
    await server.serve(sockets=sockets)
