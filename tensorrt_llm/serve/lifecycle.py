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

:class:`ServerLifecycle` fronts the real application for the whole process
lifetime. One app, one socket, one uvicorn instance, so ``STARTING`` ->
``READY`` is a single attribute store rather than a rebind or handoff, and
no client sees a reset across it.

Two properties are load-bearing and easy to lose in a refactor:

* The engine is built **off the event loop** (:func:`serve_with_lifecycle`).
  On the loop, ``/health`` would be accepted and never answered -- a third
  ambiguous state, worse than the connection-refused it replaces, since
  only a wall-clock timeout resolves it.
* ``STARTING`` answers *every* route, not just ``/health``. A 404 from an
  unregistered route reads to a client as a real answer.
"""

import asyncio
import json
import socket
import traceback
from typing import Any, Awaitable, Callable, Optional

from strenum import StrEnum

from tensorrt_llm.logger import logger

# Wire contract: lets a poller log *why* it got a 503 without parsing the body.
SERVER_STATE_HEADER = "x-trtllm-server-state"

# Bounds a delegate lifespan that yields but never finishes. Cannot bound one
# that blocks the loop thread -- no asyncio timer can -- and OpenAIServer's
# engine teardown is deliberately of that kind; see its lifespan.
_DELEGATE_SHUTDOWN_TIMEOUT_SECONDS = 60.0

# Bounds the abort path: uvicorn runs without timeout_graceful_shutdown, so
# its request drain is unbounded and this is the only thing keeping the path
# finite. Above _DELEGATE_SHUTDOWN_TIMEOUT_SECONDS so the delegate normally
# gets its full bound -- a preference, not a guarantee; correctness does not
# depend on the order, since finish_delegate_lifespan() reports what happened.
_ABORT_SHUTDOWN_TIMEOUT_SECONDS = 90.0

# Bounds getting a *cancelled* lifespan task to stop. Only expires against one
# that swallows CancelledError and keeps going.
_ABANDON_TIMEOUT_SECONDS = 15.0

_TEARDOWN_TIMEOUT_SECONDS = 60.0

# Queued when the delegate's lifespan returns, so a waiter on the next message
# cannot block forever if it exits without completing the handshake.
_DELEGATE_EXITED = object()

ASGIApp = Callable[
    [dict, Callable[[], Awaitable[dict]], Callable[[dict], Awaitable[None]]], Awaitable[None]
]


class ServerState(StrEnum):
    """Observable startup state of a *running* serving process.

    There is intentionally no ``DEAD`` member: a dead process cannot
    report anything, so death is observed as connection-refused rather
    than as a value served over the wire. See the module docstring.
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


class _DelegateLifespan:
    """Drives the ASGI lifespan protocol of the delegate application.

    uvicorn runs the lifespan of the *outer* application only, and it does
    so before the delegate exists. The delegate's own lifespan -- which
    starts the iteration-stats collector, registers with the metadata
    server, and (on the way out) shuts the engine down -- is therefore run
    here, against private queues, when the delegate is attached.
    """

    def __init__(self, app: ASGIApp, state: Optional[dict]) -> None:
        self._app = app
        self._startup_failed = False
        self._scope: dict[str, Any] = {
            "type": "lifespan",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
        }
        if state is not None:
            # Share uvicorn's lifespan state dict so anything the delegate
            # publishes there still reaches per-request scopes.
            self._scope["state"] = state
        self._to_app: asyncio.Queue = asyncio.Queue()
        self._from_app: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._error: Optional[BaseException] = None

    async def _receive(self) -> dict:
        return await self._to_app.get()

    async def _send(self, message: dict) -> None:
        await self._from_app.put(message)

    async def _run(self) -> None:
        try:
            await self._app(self._scope, self._receive, self._send)
        except BaseException as exc:  # re-raised out of startup()/shutdown()
            self._error = exc
        finally:
            self._from_app.put_nowait(_DELEGATE_EXITED)

    async def _next_message(self) -> dict:
        message = await self._from_app.get()
        if message is _DELEGATE_EXITED:
            raise RuntimeError(
                "The serving application's lifespan exited without completing "
                "the ASGI lifespan handshake."
            ) from self._error
        return message

    async def startup(self) -> None:
        """Run the delegate's lifespan startup; raise if it fails."""
        self._task = asyncio.create_task(self._run())
        self._to_app.put_nowait({"type": "lifespan.startup"})
        try:
            message = await self._next_message()
        except BaseException:
            self._startup_failed = True
            raise
        if message["type"] != "lifespan.startup.complete":
            self._startup_failed = True
            raise RuntimeError(
                f"The serving application failed to start: {message.get('message', '')}"
            )

    async def _abandon(self) -> bool:
        """Cancel the lifespan task; False if it did not finish in time.

        The delegate's teardown cancels and awaits its own collectors, so it
        can catch ``CancelledError`` and keep running. Awaiting it unbounded
        would be the one unbounded wait on a path built for boundedness.

        ``asyncio.wait`` rather than ``await self._task``: it bounds the wait,
        and lets a cancellation of *this* coroutine propagate instead of being
        mistaken for the one we just requested.
        """
        if self._task is None or self._task.done():
            return True
        self._task.cancel()
        done, _ = await asyncio.wait({self._task}, timeout=_ABANDON_TIMEOUT_SECONDS)
        return bool(done)

    async def finish(self) -> bool:
        """Settle the lifespan; True only if its teardown ran to completion.

        Answers whether the delegate *did* release its resources, not whether
        it intended to -- those differ when the shutdown is abandoned partway.
        Cancels a still-running task so the answer stays true: one left
        suspended mid-teardown could resume later and run the engine shutdown
        alongside an out-of-band one the caller is about to start.
        """
        if self._task is None:
            return False
        if not await self._abandon():
            # Cancelled but still alive. Report "torn down" so the caller does
            # not start a second one: it is suspended at an await and could
            # reach the engine shutdown at any moment, and two concurrent
            # shutdowns are the not-ZMQ-safe case. Prefer a possibly-skipped
            # teardown over a possibly-concurrent one.
            logger.error(
                "The serving application's lifespan ignored cancellation for "
                f"{_ABANDON_TIMEOUT_SECONDS:.0f}s; leaving the engine teardown "
                "to it rather than risking a second, concurrent one."
            )
            return True
        # Both halves needed: a lifespan whose startup failed returns cleanly
        # without running its post-yield teardown, and _run() funnels every
        # exception (cancellation included) into _error, so a clean None is
        # what separates "ran to the end" from "abandoned part-way".
        return not self._startup_failed and self._error is None

    async def shutdown(self) -> None:
        """Run the delegate's lifespan shutdown and wait for it to finish.

        Tolerates a delegate whose startup failed: it is reachable in that
        case because ``attach`` registers this object *before* awaiting
        ``startup``, so a later outer shutdown still funnels through here.
        """
        if self._task is None:
            return
        if not self._task.done():
            self._to_app.put_nowait({"type": "lifespan.shutdown"})
        # The inability to bound a loop-blocking delegate is load-bearing: a
        # teardown that blocks the loop cannot be left half-done, whereas one
        # on a worker thread would keep running, uncancellable, after this
        # bound gave up on it. See OpenAIServer's lifespan.
        try:
            await asyncio.wait_for(
                asyncio.shield(self._task), timeout=_DELEGATE_SHUTDOWN_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error(
                "The serving application's lifespan shutdown did not finish "
                f"within {_DELEGATE_SHUTDOWN_TIMEOUT_SECONDS:.0f}s; abandoning it."
            )
            # Cancel rather than leave it suspended mid-teardown: resuming
            # later would run the engine shutdown alongside the out-of-band
            # one that abandoning it makes necessary. Matters most on the
            # normal path, where nothing else settles the lifespan after.
            if not await self._abandon():
                logger.error(
                    "The serving application's lifespan also ignored "
                    "cancellation; it may still be running at process exit."
                )
            return
        # A startup failure was already raised out of startup(); re-raising it
        # here would only mask whatever the shutdown is unwinding from.
        if self._error is not None and not self._startup_failed:
            raise self._error


class ServerLifecycle:
    """ASGI application implementing the STARTING -> READY contract.

    Hand this to uvicorn *before* the engine exists, then call
    :meth:`attach` with the real application once it is initialized. Until
    then every request is answered with ``503``.
    """

    def __init__(self) -> None:
        # Read once per request in __call__; assigning it is the whole
        # STARTING -> READY transition.
        self._delegate: Optional[ASGIApp] = None
        self._delegate_lifespan: Optional[_DelegateLifespan] = None
        self._lifespan_state: Optional[dict] = None

    @property
    def state(self) -> ServerState:
        return ServerState.STARTING if self._delegate is None else ServerState.READY

    async def finish_delegate_lifespan(self) -> bool:
        """Settle the delegate's lifespan; True if its teardown completed.

        The delegate owns the engine teardown in the second half of its
        lifespan. Callers use the answer to decide whether an out-of-band
        teardown is still needed -- and, because this leaves the lifespan
        task finished either way, to be sure they are not starting one
        *alongside* a live teardown.
        """
        lifespan = self._delegate_lifespan
        if lifespan is None:
            return False
        return await lifespan.finish()

    async def attach(self, app: ASGIApp) -> None:
        """Run ``app``'s lifespan startup, then make it serve every request.

        Must be awaited on the event loop that uvicorn is running on. The
        socket is untouched: clients connected during ``STARTING`` stay
        connected and simply start getting real answers.
        """
        # Registered before the await: if startup fails partway, the delegate
        # may already own resources (the engine is built before its lifespan
        # runs), and the outer shutdown must still be routed into it.
        delegate_lifespan = _DelegateLifespan(app, self._lifespan_state)
        self._delegate_lifespan = delegate_lifespan
        await delegate_lifespan.startup()
        # Only now does the delegate start serving requests.
        self._delegate = app
        logger.info("Server state: READY (engine initialized, /health -> 200)")

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        scope_type = scope["type"]
        if scope_type == "lifespan":
            await self._handle_lifespan(scope, receive, send)
            return

        delegate = self._delegate
        if delegate is not None:
            await delegate(scope, receive, send)
            return

        if scope_type == "websocket":
            # Reachable whatever routes the app registers: uvicorn picks this
            # scope from the client's Upgrade header, not the routing table.
            # Answering it with an HTTP response raises inside the protocol
            # and surfaces as a 500 + traceback, making STARTING noisier than
            # either READY or dead.
            await self._reject_websocket(receive, send)
        else:
            await self._reject_http(send)

    async def _handle_lifespan(self, scope: dict, receive: Callable, send: Callable) -> None:
        self._lifespan_state = scope.get("state")
        while True:
            message = await receive()
            message_type = message["type"]
            if message_type == "lifespan.startup":
                # Unconditionally: uvicorn calls listen() only once startup
                # completes, and listening during engine load is the point.
                logger.info(
                    "Server state: STARTING (listening, /health -> 503 until "
                    "the engine is initialized)"
                )
                await send({"type": "lifespan.startup.complete"})
            elif message_type == "lifespan.shutdown":
                try:
                    if self._delegate_lifespan is not None:
                        await self._delegate_lifespan.shutdown()
                except BaseException:
                    await send(
                        {
                            "type": "lifespan.shutdown.failed",
                            "message": traceback.format_exc(),
                        }
                    )
                    raise
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _reject_http(self, send: Callable) -> None:
        """Answer any request with 503 while the engine is initializing."""
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
    build: Callable[[], Any],
    timeout_keep_alive: int,
    on_ready: Optional[Callable[[Any], Awaitable[bool]]] = None,
    on_startup_failure: Optional[Callable[[Any], None]] = None,
) -> None:
    """Serve the three-state lifecycle contract from a single uvicorn instance.

    Starts listening first, runs the blocking ``build`` off the event loop,
    then swaps the built application in without touching the socket.

    Args:
        host: Host the socket is bound to (reported by the server).
        port: Port the socket is bound to (reported by the server).
        sockets: Pre-bound sockets handed to uvicorn, which calls listen()
            on them. ``None`` lets uvicorn bind its own.
        build: Blocking callable returning the initialized server object.
            Runs on a dedicated worker thread, never on the event loop.
        timeout_keep_alive: uvicorn keep-alive timeout, in seconds.
        on_ready: Awaited with the built server once it starts serving.
            Returning ``False`` fails startup.
        on_startup_failure: Called with the built server if anything after
            the build fails. The engine already exists by then, so without
            this its worker processes would outlive the frontend.

    Raises:
        BaseException: Whatever ``build`` raised. uvicorn is stopped first,
            so the socket closes and pollers see connection refused -- the
            dead state, as for any other crash.
    """
    # Lazy: CLI paths import this module without starting a server.
    import uvicorn

    lifecycle = ServerLifecycle()
    config = uvicorn.Config(
        lifecycle, host=host, port=port, log_level="info", timeout_keep_alive=timeout_keep_alive
    )
    uvicorn_server = uvicorn.Server(config)
    serve_task = asyncio.get_running_loop().create_task(uvicorn_server.serve(sockets=sockets))

    server = None
    try:
        await _wait_until_listening(uvicorn_server, serve_task)
        server = await _build_off_event_loop(build, serve_task)
        server.record_address(host, port)
        await lifecycle.attach(server.app)
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
            # Our own cancellation, not a shutdown problem; never swallow it.
            raise
        except Exception as e:
            # Best effort: a messy shutdown must never mask the startup error
            # that is about to be re-raised, which is what the operator needs.
            logger.error(f"Server shutdown after a startup failure did not complete cleanly: {e!r}")
        # Settle the delegate's lifespan first: the wait above cancels uvicorn
        # on expiry, which can leave the delegate suspended mid-teardown.
        # finish_delegate_lifespan cancels it and reports what actually
        # completed, so the decision below is about fact, not intent.
        try:
            delegate_tore_down = await lifecycle.finish_delegate_lifespan()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Could not settle the serving application's lifespan: {e!r}")
            delegate_tore_down = False

        # Only when the delegate's did not. Running both means two concurrent
        # engine shutdowns: BaseLLM.shutdown() takes no lock, so both callers
        # see a non-None executor and drive ZeroMqQueue.close() from two
        # threads, which the executor proxy documents as not ZMQ-safe.
        # Skipping both is equally wrong -- the only path left is LLM's atexit
        # hook, an unbounded shutdown during interpreter teardown.
        if server is not None and on_startup_failure is not None and not delegate_tore_down:
            # Off the loop is safe here precisely because this is the only
            # teardown on this path, so the bound cannot fire into another.
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
