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
"""Typed errors of the serving core.

The core raises them; HTTP wrappers and wire dialects render them. ServingError keeps the HTTP
status apart from the error body, because ErrorResponse.code is not always a status (an
over-length prompt carries "context_length_exceeded").

The dialect-neutral errors (DialectNeutralError and its subclasses) are what a chat backend
raises for any wire dialect to render; ``neutral_error`` picks the type for an HTTP status.
``serving_error_boundary`` is the error ladder of an HTTP handler written as a declaration: it
renders the typed errors through the dialect's registered renderers
(``tensorrt_llm/serve/exception_handling/register.py``) and maps the handler's other exceptions
with its ``ErrorRule`` ladder.
"""

import functools
import inspect
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from starlette.requests import Request
from starlette.responses import Response

from tensorrt_llm.executor.utils import (
    CONTEXT_LENGTH_EXCEEDED_CODE,
    is_context_length_exceeded_message,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ErrorResponse

if TYPE_CHECKING:
    from tensorrt_llm.serve.exception_handling.register import ErrorRenderers

Handler = TypeVar("Handler", bound=Callable[..., Awaitable[Any]])


def error_body(
    message: str,
    err_type: str = "BadRequestError",
    status_code: Union[HTTPStatus, int] = HTTPStatus.BAD_REQUEST,
) -> ErrorResponse:
    """Build the OpenAI error body for a rejected request.

    ``code`` is the HTTP status, except for an over-length prompt, which is recognized by its
    text because it crosses the executor IPC boundary as a plain string (RequestError), so no
    exception type survives to here. ``status_code`` may be a plain int for a status with no
    HTTPStatus member (a worker's 499, for example).
    """
    code: Union[int, str] = int(status_code)
    if is_context_length_exceeded_message(message):
        code = CONTEXT_LENGTH_EXCEEDED_CODE
    return ErrorResponse(message=message, type=err_type, code=code)


class ServingError(Exception):
    """A request the serving core rejects, with the HTTP status to answer it with.

    Attributes:
        status: The HTTP status code.
        body: The OpenAI error body; ``body.code`` is not necessarily ``status``.
    """

    def __init__(self, status: int, body: ErrorResponse) -> None:
        super().__init__(body.message)
        self.status = status
        self.body = body


def serving_error(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ServingError:
    """Return the ServingError for what OpenAIServer.create_error_response would render."""
    return ServingError(status_code.value, error_body(message, err_type, status_code))


class EngineFatalError(Exception):
    """The engine failed fatally; SIGINT has been sent and the server is shutting down.

    It carries no response; each caller decides what to answer.
    """


class DialectNeutralError(Exception):
    """An error a chat backend raises for any wire dialect to render.

    Attributes:
        message: The client-facing message.
        status: The HTTP status to answer with.
        err_type: The OpenAI error ``type``.
        code: Overrides the OpenAI body's ``code`` when set; otherwise error_body decides.
    """

    status: int = 500
    err_type: str = "InternalServerError"
    code: Optional[str] = None

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidRequestError(DialectNeutralError):
    """The request cannot be served as written (HTTP 400)."""

    status = 400
    err_type = "BadRequestError"


class ContextLengthExceededError(InvalidRequestError):
    """The prompt does not fit the context window (HTTP 400, code ``context_length_exceeded``)."""

    code = CONTEXT_LENGTH_EXCEEDED_CODE


class UpstreamError(DialectNeutralError):
    """A worker the request was forwarded to failed it.

    ``status`` is the worker's status, or 502 when it had none; a 4xx keeps the OpenAI ``type`` of
    a rejected request, any other status that of a server error.
    """

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status or 502
        self.err_type = "BadRequestError" if 400 <= self.status < 500 else "InternalServerError"


class UnavailableError(DialectNeutralError):
    """The server cannot take the request now (HTTP 503)."""

    status = 503

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class InternalServingError(DialectNeutralError):
    """The server failed while serving the request (HTTP 500)."""

    status = 500


def neutral_error(status: int, message: str) -> DialectNeutralError:
    """The dialect-neutral error for an HTTP status and a final message.

    400 with the context-length message: ContextLengthExceededError; 400: InvalidRequestError;
    500: InternalServingError; 503: UnavailableError; any other status: UpstreamError(message,
    status). Status and message are kept as given.
    """
    if status == 400:
        if is_context_length_exceeded_message(message):
            return ContextLengthExceededError(message)
        return InvalidRequestError(message)
    if status == 500:
        return InternalServingError(message)
    if status == 503:
        return UnavailableError(message)
    return UpstreamError(message, status)


@dataclass(frozen=True)
class ErrorRule:
    """One step of an error ladder.

    Attributes:
        exc_type: The exception type (or types) the step matches, with their subclasses.
        status: The HTTP status the answer carries.
        err_type: The OpenAI error ``type`` of the answer.
        log: What to log at ERROR: nothing, ``str(error)``, or the traceback.
    """

    exc_type: Union[Type[Exception], Tuple[Type[Exception], ...]]
    status: HTTPStatus
    err_type: str = "BadRequestError"
    log: Literal["none", "message", "traceback"] = "none"


# The chat completion handler's ladder: an unencodable response (ValueError) is answered 400
# without a log; any other failure is answered 400 with its traceback logged.
CHAT_LADDER: Tuple[ErrorRule, ...] = (
    ErrorRule(ValueError, HTTPStatus.BAD_REQUEST),
    ErrorRule(Exception, HTTPStatus.BAD_REQUEST, log="traceback"),
)


def serving_error_boundary(
    dialect: str,
    *,
    ladder: Sequence[ErrorRule] = (),
    on_engine_fatal: Optional[Callable[[], Optional[Response]]] = None,
    renderers: Optional["ErrorRenderers"] = None,
) -> Callable[[Handler], Handler]:
    """The error ladder as a declaration on an async handler.

    On an exception from the handler, in this order:

    1. ServingError and DialectNeutralError: rendered for ``dialect`` from their own fields
       (the status travels with the error, never through ErrorResponse.code);
    2. EngineFatalError: ``on_engine_fatal()`` when given (chat: ``lambda: None``, which FastAPI
       sends as HTTP 200 with body null), else re-raised;
    3. the first rule whose exc_type matches: the dialect's body with the rule's status and
       err_type and str(error) as the message; logged as the rule says (message:
       logger.error(str); traceback: logger.error(traceback.format_exc()));
    4. anything else is re-raised.

    The request handed to the renderer is the value of the handler's first parameter whose
    annotation is starlette's Request (fastapi.Request is the same class) or, under
    ``from __future__ import annotations``, the string ``"Request"``; else None. It is found by
    annotation only: route functions also take a body model named ``request``.

    The registry is ``renderers`` when given; else the app of the request's ASGI scope
    (``request.app.state.error_renderers``, which install_exception_handlers sets) when the
    scope has one; else ``default_error_renderers()``.

    The wrapper keeps the handler's signature, name and docstring (``functools.wraps``), so a
    route built from it has the same parameters, operationId and description.
    """

    def decorate(handler: Handler) -> Handler:
        signature = inspect.signature(handler)
        request_parameter = _request_parameter(signature)

        @functools.wraps(handler)
        async def boundary(*args: Any, **kwargs: Any) -> Any:
            try:
                return await handler(*args, **kwargs)
            except (ServingError, DialectNeutralError) as error:
                request = _bound_request(signature, request_parameter, args, kwargs)
                return _renderers_for(request, renderers).render(dialect, request, error)
            except EngineFatalError:
                if on_engine_fatal is None:
                    raise
                return on_engine_fatal()
            except Exception as error:
                rule = _matching_rule(ladder, error)
                if rule is None:
                    raise
                _log(rule, error)
                request = _bound_request(signature, request_parameter, args, kwargs)
                rendered = _rule_error(rule, error)
                return _renderers_for(request, renderers).render(dialect, request, rendered)

        return cast(Handler, boundary)

    return decorate


def _request_parameter(signature: inspect.Signature) -> Optional[str]:
    """The name of the first parameter annotated as a Request, else None."""
    for parameter in signature.parameters.values():
        if parameter.annotation is Request or parameter.annotation == "Request":
            return parameter.name
    return None


def _bound_request(
    signature: inspect.Signature,
    name: Optional[str],
    args: Tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Optional[Request]:
    if name is None:
        return None
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        return None
    request: Optional[Request] = bound.arguments.get(name)
    return request


def _renderers_for(
    request: Optional[Request], renderers: Optional["ErrorRenderers"]
) -> "ErrorRenderers":
    if renderers is not None:
        return renderers
    scope = getattr(request, "scope", None)
    app = scope.get("app") if scope is not None else None
    if app is None:
        # Imported here: the registry module imports this one.
        from tensorrt_llm.serve.exception_handling.register import default_error_renderers

        return default_error_renderers()
    registry: Optional[ErrorRenderers] = getattr(app.state, "error_renderers", None)
    if registry is None:
        raise RuntimeError(
            "the request's app has no error renderers: build the app with build_http_app, "
            "or call install_exception_handlers on it before mounting handlers that use "
            "serving_error_boundary"
        )
    return registry


def _matching_rule(ladder: Sequence[ErrorRule], error: Exception) -> Optional[ErrorRule]:
    for rule in ladder:
        if isinstance(error, rule.exc_type):
            return rule
    return None


def _log(rule: ErrorRule, error: Exception) -> None:
    # The TRT-LLM logger's methods carry no annotations.
    if rule.log == "message":
        logger.error(str(error))  # type: ignore[no-untyped-call]
    elif rule.log == "traceback":
        logger.error(traceback.format_exc())  # type: ignore[no-untyped-call]


def _rule_error(rule: ErrorRule, error: Exception) -> DialectNeutralError:
    """The dialect-neutral error that renders ``error`` as ``rule`` says."""
    rendered = DialectNeutralError(str(error))
    rendered.status = rule.status.value
    rendered.err_type = rule.err_type
    rendered.__cause__ = error
    return rendered
