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
"""Startup lifecycle contract for the ``trtllm-serve`` HTTP frontend.

A serving process must be remotely distinguishable in exactly three states:

======================  =====================================================
``STARTING``            The socket is **listening**. Every request, including
                        ``GET /health``, is answered promptly with
                        ``503 Service Unavailable``.
``READY``               Requests are handed to the fully initialized OpenAI
                        application; ``GET /health`` answers ``200``.
dead                    Nothing is listening, so a TCP connect is refused.
======================  =====================================================

Only the first two are :class:`ServerState` members: they are what the
process *reports*, while dead is the absence of a reporter. A ``DEAD``
value could only be served by a process alive enough to serve it -- a
fourth ambiguous state, not a clearer third one.

Previously the socket was bound but never ``listen()``-ed until engine
init finished, so a connect during startup got an RST -- byte-identical to
a crashed process. Pollers could not tell "starting" from "dead".

:class:`ReadinessGate` wraps the real application and is what uvicorn
serves. It is a plain ASGI callable rather than Starlette middleware
because Starlette refuses ``add_middleware`` once an application has
started, and by construction this one must be installed before the engine
exists. One app, one socket, one uvicorn instance, so ``STARTING`` ->
``READY`` is a single attribute store, and no client sees a reset.

Two properties are load-bearing and easy to lose in a refactor:

* The engine is built **off the event loop** (:func:`serve_with_lifecycle`).
  On the loop, ``/health`` would be accepted and never answered -- a third
  ambiguous state, worse than the connection-refused it replaces, since
  only a wall-clock timeout resolves it.
* ``STARTING`` answers *every* route, not just ``/health``. A 404 from a
  route not yet registered reads to a client as a real answer, and under
  this design the engine-dependent routes genuinely are registered late.
"""

import asyncio
import json
import socket
from typing import Any, Awaitable, Callable, Optional

from strenum import StrEnum

from tensorrt_llm.logger import logger

# Wire contract: lets a poller log *why* it got a 503 without parsing the body.
SERVER_STATE_HEADER = "x-trtllm-server-state"

# Bounds the abort path: uvicorn runs without timeout_graceful_shutdown, so
# its request drain is unbounded and this is the only thing keeping the path
# finite.
_ABORT_SHUTDOWN_TIMEOUT_SECONDS = 90.0

# Bounds the engine teardown. Load-bearing: the teardown deliberately does
# not ride the app's lifespan shutdown, because uvicorn awaits that with no
# timeout (LifespanOn.shutdown is a bare `await self.shutdown_event.wait()`),
# so there it could be neither bounded nor reported on.
_TEARDOWN_TIMEOUT_SECONDS = 60.0

ASGIApp = Callable[
    [dict, Callable[[], Awaitable[dict]], Callable[[dict], Awaitable[None]]], Awaitable[None]
]


class ServerState(StrEnum):
    """Observable startup state of a *running* serving process.

    There is intentionally no ``DEAD`` member: a dead process cannot report
    anything, so death is observed as connection-refused. See the module
    docstring.
    """

    STARTING = "starting"
    READY = "ready"


# Built once: the body is constant, so rebuilding it per request would
# re-serialize the same JSON for every poll of a multi-minute startup.
_STARTING_BODY = json.dumps(
    {
        "error": {
            "message": "The server is still initializing the engine and cannot serve "
            "requests yet. It is listening, so this is not a crash; retry "
            "until /health returns 200.",
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
    """ASGI wrapper answering 503 until :meth:`open` is called.

    Handed to uvicorn in place of the application itself, so it is installed
    before the engine exists -- which Starlette's ``add_middleware`` cannot
    be, since it refuses to run once an application has started.

    Lifespan messages pass straight through: the wrapped application owns its
    own lifespan, and under this design that lifespan does no engine work at
    startup (it moved to ``OpenAIServer.attach``), so uvicorn reaches
    ``listen()`` immediately.
    """

    def __init__(self, app: ASGIApp) -> None:
        # Read once per request in __call__; setting it is the whole
        # STARTING -> READY transition.
        self._ready = False
        self._app = app

    @property
    def state(self) -> ServerState:
        return ServerState.READY if self._ready else ServerState.STARTING

    def open(self) -> None:
        self._ready = True
        logger.info("Server state: READY (engine initialized, /health -> 200)")

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        scope_type = scope["type"]
        if self._ready or scope_type == "lifespan":
            await self._app(scope, receive, send)
            return
        if scope_type == "websocket":
            # Reachable whatever routes the app registers: uvicorn picks this
            # scope from the client's Upgrade header, not the routing table.
            # Answering it with an HTTP response raises inside the protocol
            # and surfaces as a 500 + traceback.
            await self._reject_websocket(receive, send)
        else:
            await self._reject_http(send)

    async def _reject_http(self, send: Callable) -> None:
        await send({"type": "http.response.start", "status": 503, "headers": _STARTING_HEADERS})
        await send({"type": "http.response.body", "body": _STARTING_BODY})

    async def _reject_websocket(self, receive: Callable, send: Callable) -> None:
        """Refuse a websocket handshake while the engine is initializing.

        Closing before accepting makes uvicorn reject with HTTP 403 -- what
        the initialized app also does for an unrouted path, so STARTING is
        never worse than READY here.
        """
        message = await receive()
        if message["type"] != "websocket.connect":
            # A client that aborted mid-handshake sends websocket.disconnect;
            # closing after that raises inside the protocol.
            return
        await send({"type": "websocket.close", "code": 1013})  # "Try Again Later"


async def _wait_until_listening(uvicorn_server: Any, serve_task: asyncio.Task) -> None:
    """Block until uvicorn has called listen() on the pre-bound socket."""
    while not uvicorn_server.started:
        if serve_task.done():
            # Surfaces the startup failure, or an unexpected clean exit.
            await serve_task
            raise RuntimeError(
                "uvicorn exited before it started listening; the server never "
                "reached the STARTING state."
            )
        await asyncio.sleep(0.02)


async def serve_with_lifecycle(
    *,
    host: str,
    port: int,
    sockets: Optional[list[socket.socket]],
    build_skeleton: Callable[[], Any],
    build_engine: Callable[[], Any],
    timeout_keep_alive: int,
    on_ready: Optional[Callable[[Any], Awaitable[bool]]] = None,
    on_startup_failure: Optional[Callable[[Any], None]] = None,
) -> None:
    """Serve the three-state contract from a single uvicorn instance.

    ``build_skeleton`` returns an engine-less server whose ``app`` uvicorn can
    serve immediately; ``build_engine`` is the blocking engine construction,
    run off the event loop; ``server.attach(engine)`` then wires the two
    together and opens the gate.

    Args:
        host: Host the socket is bound to (reported by the server).
        port: Port the socket is bound to (reported by the server).
        sockets: Pre-bound sockets handed to uvicorn, which calls listen()
            on them. ``None`` lets uvicorn bind its own.
        build_skeleton: Cheap, engine-free; must not block meaningfully.
        build_engine: Blocking; runs on a dedicated worker thread.
        timeout_keep_alive: uvicorn keep-alive timeout, in seconds.
        on_ready: Awaited with the server once it starts serving. Returning
            ``False`` fails startup.
        on_startup_failure: Called with the server if anything after the
            engine build fails, so its worker processes do not outlive us.

    Raises:
        BaseException: Whatever the build raised. uvicorn is stopped first,
            so the socket closes and pollers see connection refused.
    """
    # Lazy: CLI paths import this module without starting a server.
    import uvicorn

    server = build_skeleton()
    server.record_address(host, port)
    gate = ReadinessGate(server.app)
    config = uvicorn.Config(
        gate, host=host, port=port, log_level="info", timeout_keep_alive=timeout_keep_alive
    )
    uvicorn_server = uvicorn.Server(config)
    serve_task = asyncio.get_running_loop().create_task(uvicorn_server.serve(sockets=sockets))

    attached = False
    try:
        await _wait_until_listening(uvicorn_server, serve_task)
        logger.info(
            "Server state: STARTING (listening, /health -> 503 until the engine is initialized)"
        )
        engine = await _build_off_event_loop(build_engine, serve_task)
        # attach() runs the engine-dependent startup the app's lifespan used
        # to do, and registers the routes that depend on the live engine.
        await server.attach(engine)
        attached = True
        gate.open()
        if on_ready is not None and not await on_ready(server):
            # Must raise, not just stop: exiting 0 here would tell an
            # orchestrator the worker completed successfully.
            raise RuntimeError(
                "The server started but could not complete registration; shutting down."
            )
    except BaseException:
        uvicorn_server.should_exit = True
        try:
            await asyncio.wait_for(serve_task, timeout=_ABORT_SHUTDOWN_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Best effort: a messy shutdown must never mask the startup error.
            logger.error(f"Server shutdown after a startup failure did not complete cleanly: {e!r}")

        # The app's lifespan shutdown tears the engine down, but only if the
        # engine was ever attached -- before that the lifespan has nothing to
        # tear down, and uvicorn may not have run it at all. Running both would
        # be two concurrent engine shutdowns: BaseLLM.shutdown() takes no lock,
        # so both callers see a non-None executor and drive ZeroMqQueue.close()
        # from two threads, which the executor proxy documents as not ZMQ-safe.
        if attached:
            # stop_engine_services() is idempotent and reports whether it ran,
            # so this cannot race a second teardown -- the property
            # finish_delegate_lifespan() used to provide.
            await _teardown_engine(server)
        elif on_startup_failure is not None and server.has_engine:
            try:
                await asyncio.wait_for(
                    run_in_daemon_thread(lambda: on_startup_failure(server)),
                    timeout=_TEARDOWN_TIMEOUT_SECONDS,
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                logger.error(
                    "Engine teardown after a startup failure did not finish within "
                    f"{_TEARDOWN_TIMEOUT_SECONDS:.0f}s; abandoning it."
                )
            except Exception as e:
                logger.error(
                    f"Engine teardown after a startup failure did not complete cleanly: {e!r}"
                )
        raise

    await serve_task
    # Normal exit: uvicorn has drained and stopped, so nothing else is
    # touching the engine. Bounded because a wedged teardown must not hold
    # the process open -- the reason it is not left to the lifespan.
    await _teardown_engine(server)


async def _teardown_engine(server: Any) -> None:
    """Run the server's engine teardown under a bound, logging what happened."""
    try:
        await asyncio.wait_for(server.stop_engine_services(), timeout=_TEARDOWN_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        logger.error(
            f"Engine teardown did not finish within {_TEARDOWN_TIMEOUT_SECONDS:.0f}s; "
            "abandoning it."
        )
    except Exception as e:
        logger.error(f"Engine teardown did not complete cleanly: {e!r}")


def run_in_daemon_thread(fn: Callable[[], Any]) -> asyncio.Future:
    """Run ``fn`` on a daemon thread, reporting into an asyncio future.

    Deliberately not ``asyncio.to_thread``/``ThreadPoolExecutor``: their
    workers are non-daemon and joined at exit, so a SIGTERM mid-build would
    keep the process alive until init finished -- the multi-minute hang this
    feature exists to remove. A daemon thread cannot hold the interpreter
    open, matching the pre-existing behavior of SIGTERM during engine init.
    """
    import threading

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
        except BaseException as exc:
            post(future.set_exception, exc)
        else:
            post(future.set_result, result)

    threading.Thread(target=runner, name="trtllm_engine_init", daemon=True).start()
    return future


async def _build_off_event_loop(build: Callable[[], Any], serve_task: asyncio.Task) -> Any:
    """Build the server on a worker thread while the loop keeps serving.

    Races the build against uvicorn stopping. If uvicorn goes first -- a
    SIGTERM during a long startup being the common case -- the remaining
    minutes of the build are abandoned rather than waited out.
    """
    build_future = run_in_daemon_thread(build)
    await asyncio.wait({build_future, serve_task}, return_when=asyncio.FIRST_COMPLETED)
    if not build_future.done():
        # Surfaces any error uvicorn stopped on; a clean stop falls through.
        await serve_task
        raise RuntimeError(
            "The HTTP server stopped while the engine was still initializing "
            "(e.g. SIGTERM during startup); aborting startup."
        )
    return build_future.result()
