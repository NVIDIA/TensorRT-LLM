# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Startup readiness for the ``trtllm-serve`` HTTP frontend.

A serving process is remotely distinguishable in exactly three states:

``STARTING``  The socket is listening and every request, ``GET /health``
              included, is answered promptly with ``503``.
``READY``     The generator is initialized and requests reach the real
              handlers; ``GET /health`` answers ``200``.
dead          Nothing is listening, so a TCP connect is refused.

Only the first two are :class:`ServerState` members: they are what a live
process *reports*, while dead is the absence of a reporter. A ``DEAD`` value
could only be served by a process alive enough to serve it.

Previously the socket was bound but never ``listen()``-ed until generator
init finished, so a connect during startup was refused -- byte-identical to a
crashed process, leaving pollers unable to tell "starting" from "dead".
"""

import asyncio
import json
import threading
from typing import Any, Awaitable, Callable

from strenum import StrEnum

# Wire contract: lets a poller log *why* it got a 503 without parsing the body.
SERVER_STATE_HEADER = "x-trtllm-server-state"

ASGIApp = Callable[
    [dict, Callable[[], Awaitable[dict]], Callable[[dict], Awaitable[None]]], Awaitable[None]
]


class ServerState(StrEnum):
    """Observable startup state of a *running* serving process.

    There is intentionally no ``DEAD`` member: a dead process cannot report
    anything, so death is observed as connection-refused.
    """

    STARTING = "starting"
    READY = "ready"


# Built once: the body is constant, so rebuilding it per request would
# re-serialize the same JSON for every poll of a multi-minute startup.
_STARTING_BODY = json.dumps(
    {
        "error": {
            "message": "The server is still initializing the model and cannot serve requests "
            "yet. It is listening, so this is not a crash; retry until /health "
            "returns 200.",
            "type": "ServiceUnavailableError",
            "code": ServerState.STARTING.value,
        }
    }
).encode()
_STARTING_HEADERS = [
    (b"content-type", b"application/json"),
    (b"content-length", str(len(_STARTING_BODY)).encode()),
    (b"retry-after", b"1"),  # so clients back off instead of hot-looping
    (SERVER_STATE_HEADER.encode(), ServerState.STARTING.value.encode()),
]


class ReadinessGate:
    """The STARTING -> READY flag, owned by the server.

    Separate from the middleware because Starlette instantiates middleware
    lazily, when the app first builds its stack: the server cannot hold the
    instance that ends up serving, so it holds this instead and both refer to
    it. Opening the gate is a single attribute store, and no client sees a
    reset.
    """

    def __init__(self) -> None:
        # Read once per request; setting it is the whole transition.
        self.is_ready = False

    @property
    def state(self) -> ServerState:
        return ServerState.READY if self.is_ready else ServerState.STARTING

    def open(self) -> None:
        self.is_ready = True


class ReadinessMiddleware:
    """ASGI middleware answering 503 until its gate is opened.

    Deliberately a plain ASGI callable rather than ``BaseHTTPMiddleware``:
    the latter spawns a task per request and only ever sees ``http`` scopes,
    while this one must also answer websocket handshakes, which uvicorn
    dispatches from the client's ``Upgrade`` header rather than from the
    routing table.

    Install it last so it ends up the outermost user middleware -- Starlette
    applies them in reverse registration order -- and nothing inner does work
    for a request that is about to be rejected.

    ``STARTING`` answers *every* route, not just ``/health``. A 404 from a
    route not yet registered reads to a client as a real answer, and under
    this design the generator-dependent routes genuinely are registered late.
    """

    def __init__(self, app: ASGIApp, gate: ReadinessGate) -> None:
        self._app = app
        self._gate = gate

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        scope_type = scope["type"]
        if self._gate.is_ready or scope_type == "lifespan":
            # The wrapped application owns its lifespan, and it is that
            # lifespan which drives initialization, so it must pass through
            # whatever the state is.
            await self._app(scope, receive, send)
        elif scope_type == "websocket":
            await self._reject_websocket(receive, send)
        else:
            await self._reject_http(send)

    async def _reject_http(self, send: Callable) -> None:
        await send({"type": "http.response.start", "status": 503, "headers": _STARTING_HEADERS})
        await send({"type": "http.response.body", "body": _STARTING_BODY})

    async def _reject_websocket(self, receive: Callable, send: Callable) -> None:
        """Refuse a websocket handshake while the model is initializing.

        Closing before accepting makes uvicorn reject with HTTP 403 -- what
        the initialized app also does for an unrouted path, so ``STARTING``
        is never worse than ``READY`` here. Answering with an HTTP response
        instead would raise inside the protocol and surface as a 500.
        """
        message = await receive()
        if message["type"] != "websocket.connect":
            # A client that aborted mid-handshake sends websocket.disconnect;
            # closing after that raises inside the protocol.
            return
        await send({"type": "websocket.close", "code": 1013})  # Try Again Later


def run_in_daemon_thread(fn: Callable[[], Any]) -> "asyncio.Future":
    """Run ``fn`` on a daemon thread, reporting into an asyncio future.

    Deliberately not ``asyncio.to_thread``/``ThreadPoolExecutor``: their
    workers are non-daemon and joined at interpreter exit, so a SIGTERM
    partway through a multi-minute model build would keep the process alive
    until the build finished -- the hang this feature exists to remove. A
    daemon thread cannot hold the interpreter open, which matches the
    pre-existing behavior of SIGTERM during model initialization.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def report(setter: Callable[[Any], None], value: Any) -> None:
        if not future.cancelled():
            setter(value)

    def post(setter: Callable[[Any], None], value: Any) -> None:
        # When the server stops first the build is abandoned and the loop is
        # usually gone by the time it finishes. Without this guard the thread
        # dies with "Event loop is closed" printed next to the real startup
        # error the operator is trying to read.
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(report, setter, value)
        except RuntimeError:
            pass

    def runner() -> None:
        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            post(future.set_exception, exc)
        else:
            post(future.set_result, result)

    threading.Thread(target=runner, name="trtllm_generator_init", daemon=True).start()
    return future
