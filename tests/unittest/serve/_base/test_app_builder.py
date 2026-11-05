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
"""``build_http_app()`` and ``serve_uvicorn()`` of ``serve/_base/app_builder.py``.

Both trtllm-serve apps are assembled by ``build_http_app`` in one fixed order (route
class, exception handlers, routers, middleware) and run by ``serve_uvicorn``. The tests
pin that order and the pass-through of every uvicorn argument with small callables
written here; the route and middleware-order snapshots pin what the real servers get.
"""

import ast
import asyncio
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tensorrt_llm.serve._base import app_builder
from tensorrt_llm.serve._base.app_builder import build_http_app, serve_uvicorn

pytestmark = pytest.mark.cpu_only

_ORDER_FAILURE = "build_http_app order changed; the middleware and /metrics order depend on it"


@asynccontextmanager
async def _lifespan(app):
    yield


async def _endpoint():
    return {}


def _nothing(app):
    pass


class _CustomRoute(APIRoute):
    """Stands in for the msgpack-decoding route class the aggregated server uses."""


class _PureAsgiMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)


class _MiddlewareA(_PureAsgiMiddleware):
    pass


class _MiddlewareB(_PureAsgiMiddleware):
    pass


class _MiddlewareC(_PureAsgiMiddleware):
    pass


def _route(app, path):
    # Routes added with app.add_api_route sit directly on app.router on both
    # ends of the supported FastAPI range (only included routers are wrapped).
    (route,) = [r for r in app.router.routes if getattr(r, "path", None) == path]
    return route


def _build(**overrides):
    kwargs = dict(
        lifespan=_lifespan,
        routers=(),
        dialect_error_renderers=_nothing,
        middleware=_nothing,
    )
    kwargs.update(overrides)
    return build_http_app(**kwargs)


def test_route_class_applies_to_routes_added_by_routers():
    def add_probe(app):
        app.add_api_route("/probe", _endpoint, methods=["GET"])

    custom = _build(routers=(add_probe,), route_class=_CustomRoute)
    assert type(_route(custom, "/probe")) is _CustomRoute, (
        "build_http_app must set route_class before any router runs"
    )

    plain = _build(routers=(add_probe,))
    assert type(_route(plain, "/probe")) is APIRoute, (
        "without route_class, build_http_app must leave FastAPI's APIRoute in place: "
        "the disaggregated frontend must not decode msgpack bodies on its public port"
    )


def test_assembly_order():
    calls = []
    apps_seen = []

    def recorder(name):
        def install(app):
            calls.append(name)
            apps_seen.append(app)

        return install

    app = _build(
        routers=(recorder("router-1"), recorder("router-2")),
        dialect_error_renderers=recorder("error-handlers"),
        middleware=recorder("middleware"),
    )
    assert calls == ["error-handlers", "router-1", "router-2", "middleware"], _ORDER_FAILURE
    assert all(seen is app for seen in apps_seen), (
        "every installer must receive the app build_http_app returns"
    )


def test_middleware_installed_after_router_middleware():
    def router_with_middleware(app):
        app.add_api_route("/probe", _endpoint, methods=["GET"])
        app.add_middleware(_MiddlewareA)

    def built_in_middleware(app):
        app.add_middleware(_MiddlewareB)
        app.add_middleware(_MiddlewareC)

    app = _build(routers=(router_with_middleware,), middleware=built_in_middleware)
    assert [m.cls for m in app.user_middleware] == [_MiddlewareC, _MiddlewareB, _MiddlewareA], (
        "middleware(app) must run after the routers so its middleware is outermost"
    )


def test_lifespan_and_exception_handlers_are_the_given_ones():
    events = []

    @asynccontextmanager
    async def lifespan(app):
        events.append("enter")
        yield
        events.append("exit")

    class _Boom(Exception):
        pass

    async def boom():
        raise _Boom()

    async def render_boom(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=418, content={"error": "rendered"})

    def add_boom(app):
        app.add_api_route("/boom", boom, methods=["GET"])

    app = _build(
        lifespan=lifespan,
        routers=(add_boom,),
        dialect_error_renderers=lambda app: app.add_exception_handler(_Boom, render_boom),
    )
    with TestClient(app) as client:
        assert events == ["enter"], "build_http_app must pass lifespan to FastAPI unchanged"
        response = client.get("/boom")
    assert events == ["enter", "exit"]
    assert (response.status_code, response.json()) == (418, {"error": "rendered"}), (
        "the handler registered by dialect_error_renderers must render the exception"
    )


def _fake_uvicorn(monkeypatch, serve):
    """Patch create_uvicorn_server where serve_uvicorn looks it up; return (fake server, configs)."""
    configs = []
    server = SimpleNamespace(serve=serve, started=True, should_exit=False)

    def create(config):
        configs.append(config)
        return server

    monkeypatch.setattr(app_builder, "create_uvicorn_server", create)
    return server, configs


def _count_create_task(monkeypatch):
    """Count asyncio.create_task calls made by serve_uvicorn, without changing them."""
    create_task = Mock(side_effect=asyncio.create_task)
    monkeypatch.setattr(app_builder, "asyncio", SimpleNamespace(create_task=create_task))
    return create_task


def test_serve_uvicorn_config_and_after_start(monkeypatch):
    app = FastAPI()
    started = asyncio.Event()
    received = []
    serve_calls = []

    async def after_start(server):
        received.append(server)
        started.set()

    async def serve(sockets=None):
        serve_calls.append(sockets)
        # The server keeps running while after_start works next to it.
        await asyncio.wait_for(started.wait(), timeout=5)

    server, configs = _fake_uvicorn(monkeypatch, serve)
    create_task = _count_create_task(monkeypatch)
    with socket.socket() as sock:
        asyncio.run(
            serve_uvicorn(
                app,
                host="127.0.0.1",
                port=8123,
                sockets=[sock],
                log_level="warning",
                timeout_keep_alive=7,
                after_start=after_start,
            )
        )
        failure = (
            "serve_uvicorn must pass host/port/log_level/timeout_keep_alive/sockets through "
            "and start after_start"
        )
        (config,) = configs
        assert config.app is app, failure
        assert (config.host, config.port) == ("127.0.0.1", 8123), failure
        assert config.log_level == "warning", failure
        assert config.timeout_keep_alive == 7, failure
        assert serve_calls == [[sock]], failure
    assert received == [server], failure
    assert create_task.call_count == 1, failure


def test_serve_uvicorn_passes_integer_log_level_unchanged(monkeypatch):
    async def serve(sockets=None):
        pass

    _, configs = _fake_uvicorn(monkeypatch, serve)
    asyncio.run(serve_uvicorn(FastAPI(), host="h", port=1, log_level=30, timeout_keep_alive=5))
    assert configs[0].log_level == 30, (
        "serve_uvicorn must hand uvicorn whatever log level the server passes, unchanged"
    )


def test_no_after_start_task_when_none(monkeypatch):
    serve_calls = []

    async def serve(sockets=None):
        serve_calls.append(sockets)

    _fake_uvicorn(monkeypatch, serve)
    create_task = _count_create_task(monkeypatch)
    asyncio.run(serve_uvicorn(FastAPI(), host="h", port=1, log_level="info", timeout_keep_alive=5))
    assert serve_calls == [None]
    create_task.assert_not_called()


def test_app_builder_imports_only_its_allowed_dependencies():
    """``serve/_base/app_builder.py`` depends on nothing above ``serve/_base/``."""
    source = Path(app_builder.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    allowed_roots = {"fastapi", "starlette", "uvicorn"} | set(sys.stdlib_module_names)
    disallowed = sorted(
        module
        for module in imported
        if module != "tensorrt_llm.serve._telemetry" and module.split(".")[0] not in allowed_roots
    )
    assert disallowed == [], (
        f"serve/_base/app_builder.py imports {disallowed}; it may import only FastAPI, "
        "Starlette, uvicorn, the standard library and serve/_telemetry.py. It must not "
        "import serve/api/ or either server module: pass what it needs in from the server's "
        "__init__ as a callable."
    )
