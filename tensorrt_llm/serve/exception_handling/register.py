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
"""Error renderers registered per wire dialect.

A trtllm-serve app speaks up to three wire dialects: OpenAI, Anthropic and VisualGen.
Each renders an error in its own shape, so the renderer is chosen by the dialect of the
endpoint that matched the request, never by the request path. An endpoint declares its
dialect with ``@route_dialect(...)``; endpoints without a marker take the app's default,
which the composition root sets per server role.

Every app keeps one :class:`ErrorRenderers` registry on ``app.state.error_renderers``
(``install_exception_handlers`` puts it there). Code that registers later, for a dialect the
app only speaks in one role, reads it from ``app.state`` without a default: the VisualGen
role registers its validation renderer from its own route registration.

Only request validation errors are rendered at app level, because they are raised before
the endpoint runs. Errors raised inside an endpoint are rendered where the dialect is known
statically: ``serving_error_boundary`` in ``tensorrt_llm/serve/_base/errors.py``.

This module imports FastAPI, ``tensorrt_llm/serve/_base/errors.py`` and, inside the
Anthropic renderers only, the Anthropic adapter. It must not import either server module or
any VisualGen module, at any scope: VisualGen code is loaded only on the VisualGen role.
"""

from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, TypeVar, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from tensorrt_llm.serve._base.errors import DialectNeutralError, ServingError, error_body

__all__ = [
    "ANTHROPIC",
    "OPENAI",
    "VISUAL_GEN",
    "ErrorRenderer",
    "ErrorRenderers",
    "ValidationObserver",
    "default_error_renderers",
    "dialect_of",
    "install_exception_handlers",
    "openai_validation_error_renderer",
    "route_dialect",
]

OPENAI = "openai"
ANTHROPIC = "anthropic"
VISUAL_GEN = "visual_gen"

# A renderer receives the request (None when the error is rendered outside a request) and
# the error; it may return any Response, a StreamingResponse included, so that an error found
# after a stream has opened can be answered inside the stream.
ErrorRenderer = Callable[[Optional[Request], BaseException], Response]
# Called before a validation error is rendered, with the dialect it is rendered for; side
# effects only (counters, logs).
ValidationObserver = Callable[[Request, RequestValidationError, str], None]

F = TypeVar("F", bound=Callable[..., Any])
E = TypeVar("E", bound=BaseException)

_DIALECT_ATTRIBUTE = "__trtllm_dialect__"


def route_dialect(dialect: str) -> Callable[[F], F]:
    """Mark an endpoint (function or method) as speaking ``dialect``.

    A bound method forwards attribute reads to its function, so a marked method stays
    marked once it is bound to a server.
    """

    def mark(endpoint: F) -> F:
        setattr(endpoint, _DIALECT_ATTRIBUTE, dialect)
        return endpoint

    return mark


def dialect_of(request: Request, default: str) -> str:
    """The dialect of the endpoint that matched ``request``, else ``default``.

    Starlette records the matched endpoint in ``scope["endpoint"]`` (FastAPI also records the
    route in ``scope["route"]``) before the route runs, so both are set when an app-level
    exception handler is called for a request-validation error.
    """
    endpoint = request.scope.get("endpoint")
    if endpoint is None:
        endpoint = getattr(request.scope.get("route"), "endpoint", None)
    dialect: str = getattr(endpoint, _DIALECT_ATTRIBUTE, default)
    return dialect


class ErrorRenderers:
    """Error renderers keyed by (dialect, exception type)."""

    def __init__(self) -> None:
        self._renderers: Dict[Tuple[str, type], ErrorRenderer] = {}

    def register(
        self, dialect: str, exc_type: Type[BaseException], renderer: ErrorRenderer
    ) -> None:
        """Register ``renderer`` for ``exc_type`` and its subclasses under ``dialect``."""
        self._renderers[(dialect, exc_type)] = renderer

    def has(self, dialect: str, exc_type: Type[BaseException]) -> bool:
        """Whether a renderer is registered for exactly ``(dialect, exc_type)``."""
        return (dialect, exc_type) in self._renderers

    def render(self, dialect: str, request: Optional[Request], error: BaseException) -> Response:
        """Render ``error`` for ``dialect``.

        Uses the renderer registered for the dialect and the closest class in
        ``type(error).__mro__``; when the dialect has none for it, the OpenAI one.

        Raises:
            KeyError: Neither the dialect nor the OpenAI dialect has a renderer for the error.
        """
        renderer = self._lookup(dialect, type(error))
        if renderer is None and dialect != OPENAI:
            renderer = self._lookup(OPENAI, type(error))
        if renderer is None:
            raise KeyError(
                f"no error renderer for dialect {dialect!r} and {type(error).__name__}; "
                "register one with ErrorRenderers.register"
            )
        return renderer(request, error)

    def _lookup(self, dialect: str, exc_type: type) -> Optional[ErrorRenderer]:
        for cls in exc_type.__mro__:
            renderer = self._renderers.get((dialect, cls))
            if renderer is not None:
                return renderer
        return None


def _renderer_for(
    exc_type: Type[E], render: Callable[[Optional[Request], E], Response]
) -> ErrorRenderer:
    """Adapt a renderer written for one exception type to the registry's signature."""

    def renderer(request: Optional[Request], error: BaseException) -> Response:
        if not isinstance(error, exc_type):
            raise TypeError(
                f"renderer for {exc_type.__name__} cannot render {type(error).__name__}"
            )
        return render(request, error)

    return renderer


def openai_validation_error_renderer(detail_limit: Optional[int] = None) -> ErrorRenderer:
    """The OpenAI rendering of a request-validation error: 400 with ``{"error": <text>}``.

    ``<text>`` is ``str(exc)``. With ``detail_limit``, a longer text is cut at that many
    characters and the cut is noted (pydantic echoes each failing input verbatim, which can be
    megabytes of base64; the error locations are what matters). The aggregated server sets a
    limit; the disaggregated frontend sets none.
    """

    def render(request: Optional[Request], exc: RequestValidationError) -> Response:
        detail = str(exc)
        if detail_limit is not None and len(detail) > detail_limit:
            detail = detail[:detail_limit] + " ... [truncated %d bytes]" % (
                len(detail) - detail_limit
            )
        return JSONResponse(status_code=400, content={"error": detail})

    return _renderer_for(RequestValidationError, render)


def _render_openai_serving_error(request: Optional[Request], error: ServingError) -> Response:
    return JSONResponse(content=error.body.model_dump(), status_code=error.status)


def _render_openai_neutral_error(
    request: Optional[Request], error: DialectNeutralError
) -> Response:
    body = error_body(error.message, error.err_type, error.status)
    if error.code is not None:
        body = body.model_copy(update={"code": error.code})
    return JSONResponse(content=body.model_dump(), status_code=error.status)


def _anthropic_error(message: str, status: int) -> Response:
    # Imported here so that importing this module does not load the Anthropic adapter.
    from tensorrt_llm.serve.anthropic_adapter import anthropic_error_response

    error_type: Literal["invalid_request_error", "api_error"] = (
        "invalid_request_error" if 400 <= status < 500 else "api_error"
    )
    return anthropic_error_response(message, error_type, status)


def _render_anthropic_validation_error(
    request: Optional[Request], exc: RequestValidationError
) -> Response:
    return _anthropic_error(str(exc), 400)


def _render_anthropic_serving_error(request: Optional[Request], error: ServingError) -> Response:
    return _anthropic_error(error.body.message, error.status)


def _render_anthropic_neutral_error(
    request: Optional[Request], error: DialectNeutralError
) -> Response:
    return _anthropic_error(error.message, error.status)


def default_error_renderers(
    *, openai_validation_detail_limit: Optional[int] = None
) -> ErrorRenderers:
    """The OpenAI and Anthropic renderers for validation errors and the typed serving errors.

    VisualGen's renderers are not here: VisualGen code registers them, for the VisualGen role
    only. ``openai_validation_detail_limit`` is passed to
    :func:`openai_validation_error_renderer`.
    """
    renderers = ErrorRenderers()
    renderers.register(
        OPENAI,
        RequestValidationError,
        openai_validation_error_renderer(openai_validation_detail_limit),
    )
    renderers.register(
        OPENAI, ServingError, _renderer_for(ServingError, _render_openai_serving_error)
    )
    renderers.register(
        OPENAI,
        DialectNeutralError,
        _renderer_for(DialectNeutralError, _render_openai_neutral_error),
    )
    renderers.register(
        ANTHROPIC,
        RequestValidationError,
        _renderer_for(RequestValidationError, _render_anthropic_validation_error),
    )
    renderers.register(
        ANTHROPIC, ServingError, _renderer_for(ServingError, _render_anthropic_serving_error)
    )
    renderers.register(
        ANTHROPIC,
        DialectNeutralError,
        _renderer_for(DialectNeutralError, _render_anthropic_neutral_error),
    )
    return renderers


def install_exception_handlers(
    app: FastAPI,
    *,
    renderers: ErrorRenderers,
    default_dialect: str,
    on_validation_error: Optional[ValidationObserver] = None,
) -> None:
    """Put ``renderers`` on ``app.state.error_renderers`` and install the validation handler.

    The handler resolves the request's dialect (``default_dialect`` for an unmarked
    endpoint), calls ``on_validation_error`` with it, and renders the error with the registry
    read from ``request.app.state`` at request time, so renderers registered after this call
    are used too.
    """
    app.state.error_renderers = renderers

    async def render_request_validation_error(request: Request, exc: Exception) -> Response:
        # Registered for RequestValidationError only; Starlette types every handler on Exception.
        validation_error = cast(RequestValidationError, exc)
        dialect = dialect_of(request, default_dialect)
        if on_validation_error is not None:
            on_validation_error(request, validation_error, dialect)
        registry: ErrorRenderers = request.app.state.error_renderers
        return registry.render(dialect, request, validation_error)

    app.add_exception_handler(RequestValidationError, render_request_validation_error)
