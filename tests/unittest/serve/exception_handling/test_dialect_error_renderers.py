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
"""The per-dialect error registry (``tensorrt_llm/serve/exception_handling/register.py``)."""

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.exception_handling.register import (
    ANTHROPIC,
    OPENAI,
    VISUAL_GEN,
    ErrorRenderers,
    default_error_renderers,
    dialect_of,
    install_exception_handlers,
    openai_validation_error_renderer,
    route_dialect,
)

pytestmark = pytest.mark.cpu_only

MARKER_RULE = (
    "dialect_of must read the endpoint marker (scope['endpoint'] or scope['route'].endpoint)"
)


class _Body(BaseModel):
    value: int


def _validation_error() -> RequestValidationError:
    return RequestValidationError(
        [
            {
                "type": "int_parsing",
                "loc": ("body", "value"),
                "msg": "Input should be an integer",
                "input": "x",
            }
        ]
    )


def _request(**scope) -> Request:
    return Request({"type": "http", "headers": [], **scope})


def _installed_app(registry=None, default_dialect=OPENAI, observer=None):
    app = FastAPI()
    install_exception_handlers(
        app,
        renderers=registry if registry is not None else default_error_renderers(),
        default_dialect=default_dialect,
        on_validation_error=observer,
    )
    return app


# ---------------------------------------------------------------------------
# Dialect resolution
# ---------------------------------------------------------------------------


def test_marked_endpoint_resolves_its_dialect():
    @route_dialect(ANTHROPIC)
    async def marked(body: _Body):
        return {}

    async def unmarked(body: _Body):
        return {}

    class Holder:
        @route_dialect(ANTHROPIC)
        async def method(self, body: _Body):
            return {}

    bound = Holder().method
    assert dialect_of(_request(endpoint=marked), OPENAI) == ANTHROPIC, MARKER_RULE
    assert dialect_of(_request(endpoint=bound), OPENAI) == ANTHROPIC, MARKER_RULE
    assert dialect_of(_request(endpoint=unmarked), OPENAI) == OPENAI, MARKER_RULE
    assert dialect_of(_request(endpoint=unmarked), VISUAL_GEN) == VISUAL_GEN, MARKER_RULE
    assert dialect_of(_request(), OPENAI) == OPENAI, MARKER_RULE

    # Through include_router(make_router()): the leaf endpoint is what the scope carries.
    seen = []
    app = _installed_app(
        observer=lambda request, exc, dialect: seen.append((request.url.path, dialect))
    )
    router = make_router()
    router.add_api_route("/marked", marked, methods=["POST"])
    router.add_api_route("/unmarked", unmarked, methods=["POST"])
    router.add_api_route("/method", bound, methods=["POST"])
    app.include_router(router)
    client = TestClient(app)
    for path in ("/marked", "/unmarked", "/method"):
        assert client.post(path, json={"value": "x"}).status_code == 400
    assert seen == [("/marked", ANTHROPIC), ("/unmarked", OPENAI), ("/method", ANTHROPIC)], (
        MARKER_RULE
    )

    # The route fallback, for a scope that carries the route but no endpoint.
    route = next(r for _, r in _routes(app) if getattr(r, "path", "") == "/marked")
    assert dialect_of(_request(route=route), OPENAI) == ANTHROPIC, MARKER_RULE


def _routes(app):
    from test_common.serve.routes import effective_routes

    return effective_routes(app)


# ---------------------------------------------------------------------------
# Renderers and the registry
# ---------------------------------------------------------------------------


def test_validation_bodies_per_dialect():
    exc = _validation_error()
    registry = default_error_renderers()

    openai = registry.render(OPENAI, None, exc)
    assert (openai.status_code, openai.body) == (
        400,
        JSONResponse(content={"error": str(exc)}).body,
    ), f"openai validation body changed: {openai.body!r}"

    anthropic = registry.render(ANTHROPIC, None, exc)
    expected = {"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}}
    assert (anthropic.status_code, anthropic.body) == (400, JSONResponse(content=expected).body), (
        f"anthropic validation body changed: {anthropic.body!r}"
    )

    envelope = {
        "object": "error",
        "message": "value: bad",
        "type": "BadRequestError",
        "param": None,
        "code": 422,
    }
    registry.register(
        VISUAL_GEN,
        RequestValidationError,
        lambda request, exc: JSONResponse(content=envelope, status_code=422),
    )
    visual_gen = registry.render(VISUAL_GEN, None, exc)
    assert (visual_gen.status_code, visual_gen.body) == (
        422,
        JSONResponse(content=envelope).body,
    ), f"visual_gen validation body changed: {visual_gen.body!r}"


def test_openai_validation_text_cap():
    exc = _validation_error()
    text = str(exc)
    uncapped = openai_validation_error_renderer(None)(None, exc)
    assert uncapped.body == JSONResponse(content={"error": text}).body

    limit = 10
    capped = openai_validation_error_renderer(limit)(None, exc)
    cut = text[:limit] + " ... [truncated %d bytes]" % (len(text) - limit)
    assert capped.body == JSONResponse(content={"error": cut}).body
    assert capped.status_code == 400

    at_limit = openai_validation_error_renderer(len(text))(None, exc)
    assert at_limit.body == uncapped.body, "a text exactly at the limit is not cut"


def test_observer_runs_before_rendering_with_the_dialect():
    events = []
    registry = default_error_renderers()

    def render(request, exc):
        events.append("render")
        return JSONResponse(status_code=400, content={"rendered": True})

    registry.register(ANTHROPIC, RequestValidationError, render)
    app = _installed_app(
        registry, observer=lambda request, exc, dialect: events.append(("observe", dialect))
    )

    @route_dialect(ANTHROPIC)
    async def endpoint(body: _Body):
        return {}

    app.add_api_route("/a", endpoint, methods=["POST"])
    response = TestClient(app).post("/a", json={"value": "x"})
    assert response.json() == {"rendered": True}
    assert events == [("observe", ANTHROPIC), "render"]


def test_render_falls_back_to_openai():
    registry = default_error_renderers()
    exc = _validation_error()
    assert not registry.has("some_other_dialect", RequestValidationError)
    fallback = registry.render("some_other_dialect", None, exc)
    assert fallback.body == registry.render(OPENAI, None, exc).body

    # The closest class in the error's MRO wins; a registration for a base class covers subclasses.
    class Special(RequestValidationError):
        pass

    registry.register(
        OPENAI, Special, lambda request, exc: JSONResponse(status_code=418, content={})
    )
    assert registry.render(OPENAI, None, Special([])).status_code == 418
    assert registry.render(OPENAI, None, exc).status_code == 400
    assert registry.render(ANTHROPIC, None, Special([])).status_code == 400, (
        "a dialect with a renderer for a base class uses it before falling back to openai"
    )

    with pytest.raises(KeyError):
        ErrorRenderers().render(OPENAI, None, exc)


def test_registry_is_on_app_state():
    registry = default_error_renderers()
    app = _installed_app(registry)
    assert app.state.error_renderers is registry

    async def endpoint(body: _Body):
        return {}

    app.add_api_route("/a", endpoint, methods=["POST"])
    client = TestClient(app)
    assert client.post("/a", json={"value": "x"}).json().keys() == {"error"}

    # Registered after the install: the handler must read the registry from app.state.
    registry.register(
        OPENAI,
        RequestValidationError,
        lambda request, exc: JSONResponse(status_code=400, content={"later": True}),
    )
    assert client.post("/a", json={"value": "x"}).json() == {"later": True}, (
        "VisualGen and the Responses stream register after install; the handler must read "
        "app.state.error_renderers"
    )

    # A registry swapped in after the install is the one used: the handler reads
    # app.state.error_renderers on each request, not the registry it was installed with.
    second = ErrorRenderers()
    second.register(
        OPENAI,
        RequestValidationError,
        lambda request, exc: JSONResponse(status_code=400, content={"second": True}),
    )
    app.state.error_renderers = second
    assert client.post("/a", json={"value": "x"}).json() == {"second": True}, (
        "the validation handler must read app.state.error_renderers on each request, not a "
        "closure over the registry it was installed with"
    )


def test_renderer_may_return_a_stream():
    registry = default_error_renderers()
    app = _installed_app(registry)

    async def frames():
        yield b"event: error\n"
        yield b"data: {}\n\n"

    registry.register(
        OPENAI,
        RequestValidationError,
        lambda request, exc: StreamingResponse(frames(), media_type="text/event-stream"),
    )

    async def endpoint(body: _Body):
        return {}

    app.add_api_route("/a", endpoint, methods=["POST"])
    response = TestClient(app).post("/a", json={"value": "x"})
    assert response.headers["content-type"].startswith("text/event-stream") and response.text == (
        "event: error\ndata: {}\n\n"
    ), (
        "error renderers must be allowed to return a stream (an error found after a Responses "
        "stream has opened is answered inside the stream)"
    )
