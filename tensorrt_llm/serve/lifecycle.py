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

Only the first two are :class:`ServerState` members, and deliberately so.
``STARTING`` and ``READY`` are things the process *reports*; dead is the
absence of a reporter. A ``DEAD`` enum value could only ever be served by
a process still running enough to serve it, which is precisely the
contradiction this contract removes -- a live socket answering "I am
dead" is a fourth ambiguous state, not a clearer third one. Dead is
therefore observed one layer down, as connection-refused, and never
appears in ``ServerState`` or in the ``X-TensorRT-LLM-Server-State``
header.

Historically the listening socket only appeared *after* engine
initialization finished (weight load, KV-cache allocation, warmup, CUDA
graph capture -- minutes for a large model). The port was bound but never
`listen()`-ed, so a connect during startup got an RST: byte-identical to
connecting to a process that had crashed. A remote poller therefore could
not tell "still starting" from "already dead" and had to fall back on a
wall-clock timeout, which is the single largest source of stuck CI jobs.

The contract is implemented by :class:`ServerLifecycle`, an ASGI
application that fronts the real application for the whole process
lifetime. One application, one socket, one uvicorn instance: the
``STARTING`` -> ``READY`` transition is a single in-process attribute
store, never a rebind or a socket handoff, so no client ever observes a
reset across the transition.

Two properties are load-bearing and easy to lose in a refactor:

* The engine must be built **off the event loop** (see
  :func:`serve_with_lifecycle`). Building it on the loop would let
  ``/health`` accept the connection and then never answer it. A hung
  request is a *third* ambiguous state, and one that is strictly worse
  than the connection refused it replaces, because the client can only
  resolve it with the same wall-clock timeout this contract exists to
  remove.
* ``STARTING`` must answer *every* route, not just ``/health``. A 404 (no
  route yet) or a partially initialized handler would be read by clients
  as a real answer.
"""

import asyncio
import json
import socket
import traceback
from typing import Any, Awaitable, Callable, Optional

from strenum import StrEnum

from tensorrt_llm.logger import logger

# Response header carrying the lifecycle state, so a poller can log *why* it
# got a 503 without parsing the body.
SERVER_STATE_HEADER = "x-trtllm-server-state"

# Advertised on the STARTING 503 so well-behaved clients back off instead of
# hot-looping while the engine loads.
_RETRY_AFTER_SECONDS = 1

# How often serve_with_lifecycle re-checks that uvicorn has started listening.
_LISTEN_POLL_INTERVAL_SECONDS = 0.02

# Bound on the delegate's own lifespan shutdown, for a delegate that yields
# but never finishes. It cannot bound one that blocks the loop thread -- no
# asyncio timer can -- and OpenAIServer's engine teardown is deliberately of
# that kind; see its lifespan for why.
_DELEGATE_SHUTDOWN_TIMEOUT_SECONDS = 60.0

# Grace period for uvicorn to wind down after a failed startup before the
# original exception is re-raised.
#
# What it buys: uvicorn is configured without timeout_graceful_shutdown, so
# Server.shutdown() drains in-flight requests for as long as they take,
# before it even begins the lifespan shutdown. This bound is the only thing
# keeping the abort path finite.
#
# Set above _DELEGATE_SHUTDOWN_TIMEOUT_SECONDS so that, in the common case
# where uvicorn reaches the lifespan shutdown promptly, the delegate gets its
# full bound to finish tearing down cleanly rather than being cancelled. That
# is a preference, not a guarantee: the drain above is unbounded, so a slow
# enough drain expires this bound first regardless. Correctness does not
# depend on the ordering -- finish_delegate_lifespan() reports what actually
# happened either way.
#
# Cost: worst-case exit latency on the abort path is this plus
# _TEARDOWN_TIMEOUT_SECONDS, so ~150s rather than the ~90s it would be with
# the previous 30s value.
_ABORT_SHUTDOWN_TIMEOUT_SECONDS = 90.0

# Bound on getting a cancelled lifespan task to actually stop. Cancellation
# normally takes effect at the next await, so this only expires against a
# task that swallows CancelledError and keeps going.
_ABANDON_TIMEOUT_SECONDS = 15.0

# Bound on the out-of-band engine teardown after a failed startup.
_TEARDOWN_TIMEOUT_SECONDS = 60.0

# Sentinel queued when the delegate's lifespan coroutine returns, so a waiter
# on the next lifespan message cannot block forever if the delegate exits
# without completing the handshake.
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

        ``Task.cancel()`` delivers ``CancelledError`` exactly once, and a
        task that catches it goes on running -- a shape the delegate's own
        teardown contains, where it cancels and awaits its background
        collectors. Awaiting such a task unbounded would make this the one
        unbounded wait on a path whose whole purpose is boundedness.

        ``asyncio.wait`` rather than ``await self._task``: it bounds the
        wait, and it lets a cancellation of *this* coroutine propagate
        instead of being mistaken for the cancellation we just requested.
        """
        if self._task is None or self._task.done():
            return True
        self._task.cancel()
        done, _ = await asyncio.wait({self._task}, timeout=_ABANDON_TIMEOUT_SECONDS)
        return bool(done)

    async def finish(self) -> bool:
        """Settle the lifespan; True only if its teardown ran to completion.

        Answers what a caller actually needs to know -- *did* the delegate
        release its resources -- rather than whether it started out
        intending to. Those differ whenever the shutdown is abandoned or
        cancelled partway.

        Cancels a task that is still running, so the answer is stable: a
        lifespan left suspended inside its teardown could otherwise resume
        later and run the engine shutdown alongside an out-of-band one the
        caller is about to start.
        """
        if self._task is None:
            return False
        if not await self._abandon():
            # Cancelled and still alive. Report "torn down" so the caller
            # does not start a second teardown. A pending task means the
            # loop is running and it is suspended at an await -- so it has
            # not reached the engine shutdown yet, but nothing stops it
            # reaching it a moment from now, and a second shutdown racing
            # that one is the not-ZMQ-safe case. Deliberately preferring a
            # possibly-skipped teardown over a possibly-concurrent one.
            logger.error(
                "The serving application's lifespan ignored cancellation for "
                f"{_ABANDON_TIMEOUT_SECONDS:.0f}s; leaving the engine teardown "
                "to it rather than risking a second, concurrent one."
            )
            return True
        # Both halves are needed. A lifespan whose startup failed returns
        # without ever running its post-yield teardown, yet returns cleanly;
        # and _run() funnels every exception, cancellation included, into
        # _error, so a clean None is what distinguishes "ran to the end"
        # from "was abandoned part-way".
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
        # Bounds a delegate that yields but never finishes. It deliberately
        # cannot bound a delegate that blocks the loop thread -- no asyncio
        # timer can -- and that limitation is load-bearing: an engine
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
            # Cancel rather than leave it suspended mid-teardown: a task that
            # resumed later would run the engine shutdown alongside the
            # out-of-band one that abandoning it makes necessary. Matters
            # most on the normal shutdown path, where nothing else settles
            # the lifespan afterwards.
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
            # Reachable regardless of which routes the app registers:
            # uvicorn picks the websocket scope from the client's Upgrade
            # header, not from the routing table. Answering such a scope with
            # an HTTP response raises inside the protocol implementation and
            # surfaces as a 500 plus a traceback, which would make STARTING
            # noisier than either READY or dead.
            await self._reject_websocket(receive, send)
        else:
            await self._reject_http(send)

    async def _handle_lifespan(self, scope: dict, receive: Callable, send: Callable) -> None:
        self._lifespan_state = scope.get("state")
        while True:
            message = await receive()
            message_type = message["type"]
            if message_type == "lifespan.startup":
                # Complete immediately and unconditionally. uvicorn only
                # calls listen() once startup completes, and listening while
                # the engine loads is the entire point of this contract.
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
        body = json.dumps(
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
        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"retry-after", str(_RETRY_AFTER_SECONDS).encode()),
                    (SERVER_STATE_HEADER.encode(), ServerState.STARTING.value.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _reject_websocket(self, receive: Callable, send: Callable) -> None:
        """Refuse a websocket handshake while the engine is initializing.

        Closing before accepting makes uvicorn reject the handshake with an
        HTTP 403, which is what the fully initialized app also does for an
        unrouted path -- so STARTING is never worse than READY here.
        """
        message = await receive()
        if message["type"] != "websocket.connect":
            # A client that aborted mid-handshake sends websocket.disconnect;
            # sending a close after that raises inside the protocol.
            return
        # 1013 "Try Again Later" is the websocket analogue of HTTP 503.
        await send({"type": "websocket.close", "code": 1013})


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
        await asyncio.sleep(_LISTEN_POLL_INTERVAL_SECONDS)


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
            It runs on a dedicated worker thread, never on the event loop.
        timeout_keep_alive: uvicorn keep-alive timeout, in seconds.
        on_ready: Awaited with the built server right after it starts
            serving. Returning ``False`` fails startup.
        on_startup_failure: Called with the built server if anything after
            the build fails. The engine already exists at that point, so
            without this its worker processes would outlive the frontend.

    Raises:
        BaseException: Whatever ``build`` raised. uvicorn is stopped first,
            so the socket closes and remote pollers see connection refused,
            i.e. the dead state, exactly as they would for any other crash.
    """
    # Imported lazily: this module is imported by CLI paths that must not pay
    # for uvicorn/FastAPI unless a server is actually started.
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
        # Settle the delegate's lifespan before deciding anything: the wait
        # above cancels uvicorn on expiry, which can leave the delegate
        # suspended part-way through its teardown. finish_delegate_lifespan
        # cancels such a task and reports whether the teardown actually
        # completed, so the answer below is about what happened, not about
        # what the delegate set out to do.
        try:
            delegate_tore_down = await lifecycle.finish_delegate_lifespan()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Could not settle the serving application's lifespan: {e!r}")
            delegate_tore_down = False

        # Run our own teardown only when the delegate's did not. Running both
        # would be two concurrent engine shutdowns: BaseLLM.shutdown() takes
        # no lock, so both callers see a non-None executor and run the whole
        # thing, driving ZeroMqQueue.close() from two threads, which the
        # executor proxy documents as not ZMQ-safe. Skipping *nothing* is
        # equally wrong: the only remaining path would be LLM's atexit hook,
        # an unbounded shutdown during interpreter teardown.
        if server is not None and on_startup_failure is not None and not delegate_tore_down:
            # Safe to run off the loop precisely because it is the only
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

    Deliberately not a ``ThreadPoolExecutor``: its workers are non-daemon
    and are joined by an ``atexit`` hook, so a SIGTERM arriving mid-build
    would leave the process alive until initialization finished -- exactly
    the multi-minute hang this feature exists to remove. A daemon thread
    cannot hold the interpreter open, which matches the pre-existing
    behavior of SIGTERM during engine initialization (immediate death).
    """
    import threading

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def report(setter: Callable[[Any], None], value: Any) -> None:
        if not future.cancelled():
            setter(value)

    def post(setter: Callable[[Any], None], value: Any) -> None:
        # The build is abandoned when the server stops first, so by the time
        # it finishes the loop is usually gone. Without this guard the thread
        # dies with "RuntimeError: Event loop is closed" printed right next to
        # the real startup error an operator is trying to read.
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

    Races the build against uvicorn stopping. If uvicorn goes away first --
    a SIGTERM during a long startup is the common case -- the remaining
    minutes of the build are abandoned rather than waited out.
    """
    # The built object is handed back through the future, which is a full
    # memory barrier, so it is safe to use from the event loop thread.
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
