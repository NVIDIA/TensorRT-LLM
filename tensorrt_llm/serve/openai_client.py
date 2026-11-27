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

# yapf: disable
import asyncio
import errno
import json
import os
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, aclosing
from types import TracebackType
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
)
from urllib.parse import urlsplit

import aiohttp
import msgspec
from pydantic import BaseModel, ConfigDict

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.conversation_id import get_request_conversation_id
from tensorrt_llm.serve.disagg_auth import (
    build_internal_disagg_auth_headers,
    build_subagent_affinity_headers,
    request_requires_internal_disagg_auth,
)
from tensorrt_llm.serve.multi_frontend import frontend_of_conversation
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ResponsesRequest,
    ResponsesResponse,
    UCompletionRequest,
    UCompletionResponse,
)
from tensorrt_llm.serve.perf_metrics import (
    _PERF_METRICS_HEADER_BUDGET_BYTES,
    RETURN_METRICS_HEADER,
    SSE_METRICS_EVENT,
    ClientMetricsCollector,
    adjusted_clock_from_headers,
    build_metrics_record_from_headers,
)
from tensorrt_llm.serve.responses_utils import (
    ResponseHooks,
    UCompletionResponseOrGenerator,
    get_steady_clock_now_in_seconds,
)
from tensorrt_llm.serve.router import ReservationNotFoundError, Router

# yapf: enable

# The forwarded orchestrator->worker request body is encoded as msgspec
# msgpack and flagged with the X-TRTLLM-Msgpack header.
_msgpack_encoder = msgspec.msgpack.Encoder()
MSGPACK_HEADERS = {"Content-Type": "application/json", "X-TRTLLM-Msgpack": "1"}

# A Responses stream opens with these events before the model emits anything.
_RESPONSES_OPENING_EVENTS = frozenset((b"response.created", b"response.in_progress"))


def _has_responses_output_event(chunk: bytes) -> bool:
    """Whether an SSE chunk of a Responses stream carries an event past its opening."""
    return any(
        line.startswith(b"event: ")
        and line[len(b"event: ") :].strip() not in _RESPONSES_OPENING_EVENTS
        for line in chunk.split(b"\n")
    )


# post_json's request and response are ordinary pydantic models, but not the
# UCompletion* union: it carries auxiliary payloads such as count-tokens. The
# protocol names the one method actually called, and the TypeVar ties the
# returned object to the type the caller asked for, so a mismatch is a type
# error rather than an Any that silently propagates.
class _SerializableRequest(Protocol):
    def model_dump(
        self, *, mode: str = ..., exclude_unset: bool = ..., by_alias: bool = ...
    ) -> dict: ...


def _encode_request_body(request: _SerializableRequest) -> bytes:
    """Encode a request forwarded to a worker.

    Fields go out under their wire names, e.g. ``schema``, which the model
    holds as ``schema_``.
    """
    return _msgpack_encoder.encode(
        request.model_dump(mode="json", exclude_unset=True, by_alias=True)
    )


_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


def _metrics_phase(role: ServerRole) -> str:
    return "ctx" if role is ServerRole.CONTEXT else "gen"


# Failures that leave a generation leg undelivered: the connection could not be
# opened (ClientConnectorError), or the connection closed before the whole
# request was written (aiohttp's ClientConnectionResetError, a
# ConnectionResetError). Any other failure may happen after the generation
# server accepted the request. Once the connection is open, a server-side close
# or reset before the response surfaces as ServerDisconnectedError or as
# ClientOSError (ECONNRESET) whether or not the server read the request first.
_GENERATION_LEG_RETRYABLE_ERRORS = (
    aiohttp.ClientConnectorError,
    ConnectionResetError,
)

# How long a context frontend affinity port that refused a connection is
# bypassed in favor of the server's shared port.
_REFUSED_AFFINITY_PORT_BYPASS_SECS = 10.0


# The disagg server is a transparent relay: worker responses come from the
# same trusted codebase, and a deployment may attach extra top-level fields to
# them (orchestrator-side telemetry, token transports, ...). Validate the known
# schema but keep unknown fields -- the protocol models are extra="forbid", so
# validating worker output with them either 500s the request or silently strips
# whatever the deployment attached. Strictness belongs at the request boundary,
# not on the relay of our own workers' output.
class RelayedCompletionResponse(CompletionResponse):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class RelayedChatCompletionResponse(ChatCompletionResponse):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class OpenAIClient(ABC):
    async def send_request(
        self,
        request: UCompletionRequest,
        server: Optional[str] = None,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
    ) -> UCompletionResponseOrGenerator:
        if isinstance(request, CompletionRequest):
            return await self._send_request(
                "v1/completions", request, RelayedCompletionResponse, server, hooks, req_id
            )
        elif isinstance(request, ChatCompletionRequest):
            return await self._send_request(
                "v1/chat/completions",
                request,
                RelayedChatCompletionResponse,
                server,
                hooks,
                req_id,
            )
        elif isinstance(request, ResponsesRequest):
            return await self._send_request(
                "v1/responses", request, ResponsesResponse, server, hooks, req_id
            )
        else:
            raise ValueError(f"Invalid request type: {type(request)}")

    @abstractmethod
    async def _send_request(
        self,
        endpoint: str,
        request: UCompletionRequest,
        response_type: Type[UCompletionResponse],
        server: Optional[str] = None,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
    ) -> UCompletionResponseOrGenerator:
        """Send a request to the server and return the response and the body generator.

        The request is finished (in routers) when the generator is exhausted or there is an error.
        """
        ...

    def start_reservation_renewal(
        self,
        request: UCompletionRequest,
        req_id: Optional[int],
        *,
        started_at: Optional[float] = None,
    ) -> Optional[asyncio.Task[None]]:
        """Start periodic reservation renewal for an already-placed request, if supported."""
        del request, req_id, started_at
        return None

    async def stop_reservation_renewal(
        self,
        req_id: Optional[int],
        renew_task: Optional[asyncio.Task[None]] = None,
    ) -> None:
        """Stop periodic reservation renewal started via ``start_reservation_renewal``."""
        del req_id, renew_task

    @abstractmethod
    async def check_ready(self) -> Tuple[List[str], List[str]]:
        """Return the list of ready servers and the list of unready servers."""
        ...

    @abstractmethod
    async def post_json(
        self,
        endpoint: str,
        request: _SerializableRequest,
        response_type: Type[_ResponseT],
        server: str,
    ) -> _ResponseT:
        """Post a non-streaming auxiliary request to a selected worker.

        Unlike _send_request this carries no disaggregation state and does not
        touch the router's in-flight accounting: it is for side endpoints such
        as token counting, which need a worker's tokenizer but no KV transfer.
        """
        ...

    @abstractmethod
    async def get_json(
        self,
        endpoint: str,
        response_type: Type[_ResponseT],
        server: str,
    ) -> _ResponseT:
        """GET a worker endpoint, outside routing state, like post_json."""
        ...

    async def shutdown(self) -> None: ...

    @abstractmethod
    async def _finish_request(
        self,
        request: UCompletionRequest,
        success: bool = True,
        req_id: Optional[int] = None,
    ) -> None:
        """Finish the request in the router.

        ``success`` lets the router distinguish completed vs failed requests
        (e.g. for routed-block tracking).
        """
        ...


def _server_base_url(server: str) -> str:
    """Return the base URL of a router server entry, with or without a scheme."""
    if server.startswith(("http://", "https://")):
        return server.rstrip("/")
    return f"http://{server}".rstrip("/")


def _server_netloc(server_or_url: str) -> str:
    """Return host:port of a router server entry or of a URL built from one."""
    return urlsplit(_server_base_url(server_or_url)).netloc


async def _close_entered_request(
    request: AbstractAsyncContextManager[aiohttp.ClientResponse],
    response: aiohttp.ClientResponse,
) -> None:
    """Close a response obtained right as the caller was cancelled or timed out."""
    try:
        await request.__aexit__(asyncio.CancelledError, asyncio.CancelledError(), None)
    except Exception:
        close = getattr(response, "close", None)
        if callable(close):
            close()


async def _drain_task_preserving_caller_cancel(task: asyncio.Task[Any]) -> bool:
    """Wait for ``task`` to settle even across repeated caller cancellations.

    Uses ``gather(..., return_exceptions=True)`` under ``shield`` so ``task``'s own
    ``CancelledError`` is consumed as a result rather than mistaken for caller cancellation.
    Returns True if the caller task itself was cancelled while draining.
    """
    drain = asyncio.ensure_future(asyncio.gather(task, return_exceptions=True))
    caller_cancelled = False
    while not drain.done():
        try:
            await asyncio.shield(drain)
        except asyncio.CancelledError:
            caller_cancelled = True
    return caller_cancelled


async def _drain_enter_task(
    request: AbstractAsyncContextManager[aiohttp.ClientResponse],
    enter_task: asyncio.Task[aiohttp.ClientResponse],
) -> None:
    """Cancel ``enter_task`` and close any returned response even across repeated cancels."""
    if not enter_task.done():
        enter_task.cancel()
    caller_cancelled = await _drain_task_preserving_caller_cancel(enter_task)
    if not enter_task.cancelled() and enter_task.exception() is None:
        close_task = asyncio.create_task(_close_entered_request(request, enter_task.result()))
        if await _drain_task_preserving_caller_cancel(close_task):
            caller_cancelled = True
    if caller_cancelled:
        raise asyncio.CancelledError()


async def _enter_with_wait(
    request: AbstractAsyncContextManager[aiohttp.ClientResponse],
    timeout_secs: float,
) -> aiohttp.ClientResponse:
    """Python-3.10 fallback for bounding ``request.__aenter__()`` without swallowing cancel."""
    enter_task = asyncio.create_task(request.__aenter__())
    try:
        done, _ = await asyncio.wait({enter_task}, timeout=timeout_secs)
    except asyncio.CancelledError:
        await _drain_enter_task(request, enter_task)
        raise
    if not done:
        await _drain_enter_task(request, enter_task)
        raise asyncio.TimeoutError()
    return enter_task.result()


async def _enter_with_deadline(
    request: AbstractAsyncContextManager[aiohttp.ClientResponse],
    timeout_secs: Optional[float],
) -> aiohttp.ClientResponse:
    """Enter ``request`` within ``timeout_secs`` while preserving external cancellation.

    Uses ``asyncio.timeout`` directly in the caller task on Python 3.11+ (where
    ``Task.cancelling()`` separates timeout and caller cancellation), and falls
    back to ``asyncio.wait`` on Python 3.10 so neither headers arriving nor the
    deadline firing in the same turn can swallow caller cancellation.
    """
    if timeout_secs is None:
        return await request.__aenter__()
    if hasattr(asyncio, "timeout"):
        async with asyncio.timeout(timeout_secs):
            return await request.__aenter__()
    return await _enter_with_wait(request, timeout_secs)


# Local module bindings for monotonic time and async sleep so unit tests can
# inject a deterministic clock without altering the event loop's own timers.
_monotonic = time.monotonic
_async_sleep = asyncio.sleep


class _ResponseStartBound:
    """Wrap a worker request so that its response must start within a time limit.

    Entering the request covers connecting, sending the request and receiving
    the response headers, and only entering is bounded; exiting is delegated
    unchanged.
    """

    def __init__(
        self,
        request: AbstractAsyncContextManager[aiohttp.ClientResponse],
        timeout_secs: Optional[float],
    ) -> None:
        self._request = request
        self._timeout_secs = timeout_secs

    async def __aenter__(self) -> aiohttp.ClientResponse:
        return await _enter_with_deadline(self._request, self._timeout_secs)

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Optional[bool]:
        return await self._request.__aexit__(exc_type, exc, tb)


class OpenAIHttpClient(OpenAIClient):
    def __init__(
        self,
        router: Router,
        role: ServerRole,
        timeout_secs: int = 180,
        max_retries: int = 1,
        retry_interval_sec: int = 1,
        session: Optional[aiohttp.ClientSession] = None,
        disagg_id_generator: Optional[Callable[[], Awaitable[int]]] = None,
        request_perf_metrics: bool = False,
        internal_disagg_auth_key: Optional[str] = None,
        bounded_replay_on_generation: bool = False,
        frontend_affinity: Optional[Tuple[int, int]] = None,
    ):
        self._router = router
        self._role = role
        self._metrics_collector = ClientMetricsCollector(role)
        # A non-streamed response arrives whole, so timeout_secs bounds the whole
        # request. A streamed response lasts as long as the worker keeps
        # generating, so timeout_secs bounds its silence instead: the wait for the
        # response to start, which includes connecting and sending the request,
        # and every wait for data after that. A total bound would cut a long
        # generation that is still producing tokens.
        self._non_stream_timeout = aiohttp.ClientTimeout(total=timeout_secs)
        # Non-positive values disable every stream bound. A negative sock_read
        # value would otherwise schedule an immediate timeout in aiohttp.
        stream_timeout_secs = timeout_secs if timeout_secs > 0 else None
        self._stream_timeout = aiohttp.ClientTimeout(
            total=None, connect=stream_timeout_secs, sock_read=stream_timeout_secs
        )
        self._response_start_timeout_secs = stream_timeout_secs
        self._session = session or aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=0,
                limit_per_host=0,
                force_close=False,
                # Set keepalive_timeout below the server-side keepalive timeout to avoid reusing stale connections.
                keepalive_timeout=1,
            ),
            timeout=aiohttp.ClientTimeout(total=timeout_secs),
            max_field_size=_PERF_METRICS_HEADER_BUDGET_BYTES,
        )
        self._no_retry = os.getenv("TRTLLM_DISAGG_NO_RETRY", "0") == "1"
        self._max_retries = 0 if self._no_retry else max_retries
        if self._no_retry:
            logger.info(
                "Disaggregated HTTP retry is DISABLED by "
                f"TRTLLM_DISAGG_NO_RETRY=1 for role={role.name}"
            )
        self._retry_interval_sec = retry_interval_sec
        self._disagg_id_generator = disagg_id_generator
        self._request_perf_metrics = request_perf_metrics
        self._internal_disagg_auth_key = internal_disagg_auth_key
        self._bounded_replay_on_generation = bounded_replay_on_generation
        # (num_frontends, port_stride) of context servers whose frontend i also listens on
        # port + port_stride * (i + 1) (serve_frontend_affinity_port_stride).
        self._frontend_affinity = frontend_affinity if role == ServerRole.CONTEXT else None
        if self._frontend_affinity is not None:
            num_frontends, stride = self._frontend_affinity
            logger.info(
                "Context requests with a conversation id go to per-conversation "
                f"frontend ports ({num_frontends} frontends, port stride {stride})"
            )
        # Affinity host:port -> steady-clock time it last refused a connection.
        self._refused_affinity_ports: dict[str, float] = {}
        # Pre-started reservation renewal tasks keyed by req_id until adopted by send_request.
        self._pending_renewals: dict[int, asyncio.Task[None]] = {}
        # All live reservation renewal tasks (pending or adopted by active streams).
        self._active_renewals: set[asyncio.Task[None]] = set()
        self._closing = False

    def _url_server(self, server: str, request: UCompletionRequest) -> str:
        """Server entry to send a request routed to server to, with the scheme of server kept.

        With frontend affinity, a request carrying a conversation id goes to
        the affinity port of the frontend picked by crc32(conversation_id) % K;
        every orchestrator process picks the same frontend for a conversation.
        An affinity port that refused a connection is bypassed for
        _REFUSED_AFFINITY_PORT_BYPASS_SECS: its conversations use the shared
        port meanwhile.
        """
        if self._frontend_affinity is None:
            return server
        conversation_id = get_request_conversation_id(request)
        if not conversation_id:
            return server
        num_frontends, stride = self._frontend_affinity
        host, _, port = server.rstrip("/").rpartition(":")
        frontend = frontend_of_conversation(conversation_id, num_frontends)
        affinity_server = f"{host}:{int(port) + stride * (frontend + 1)}"
        refused_key = _server_netloc(affinity_server)
        refused_at = self._refused_affinity_ports.get(refused_key)
        if refused_at is not None:
            if get_steady_clock_now_in_seconds() - refused_at < _REFUSED_AFFINITY_PORT_BYPASS_SECS:
                return server
            del self._refused_affinity_ports[refused_key]
        return affinity_server

    def _get_request_headers(self, request: UCompletionRequest) -> dict[str, str]:
        headers = build_subagent_affinity_headers(
            self._internal_disagg_auth_key, request, self._role
        )
        if self._role == ServerRole.GENERATION and request_requires_internal_disagg_auth(request):
            headers.update(
                build_internal_disagg_auth_headers(self._internal_disagg_auth_key, request)
            )
        return headers

    async def post_json(
        self,
        endpoint: str,
        request: _SerializableRequest,
        response_type: Type[_ResponseT],
        server: str,
    ) -> _ResponseT:
        url = f"{_server_base_url(server)}/{endpoint}"
        # A count-tokens body carries the whole conversation, so it is as large
        # as a chat body and encoding it costs the orchestrator's event loop the
        # same way -- hence the same msgpack transport as _send_request. The
        # worker decodes it because _MsgspecRoute is the app's route_class, so
        # it covers every route rather than just the completion ones.
        body = _encode_request_body(request)
        headers = dict(MSGPACK_HEADERS)
        async with self._session.post(url, data=body, headers=headers) as response:
            if response.status >= 400:
                error_body = await response.text()
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message=f"{response.reason}: {error_body[:2048]}",
                    headers=response.headers,
                )
            return response_type(**await response.json())

    async def get_json(
        self,
        endpoint: str,
        response_type: Type[_ResponseT],
        server: str,
    ) -> _ResponseT:
        url = f"{_server_base_url(server)}/{endpoint}"
        async with self._session.get(url) as response:
            if response.status >= 400:
                error_body = await response.text()
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message=f"{response.reason}: {error_body[:2048]}",
                    headers=response.headers,
                )
            return response_type(**await response.json())

    def _post(
        self,
        url: str,
        data: bytes,
        headers: dict[str, str],
        timeout: aiohttp.ClientTimeout,
    ) -> _ResponseStartBound:
        """POST to a worker, failing if its response has not started within timeout_secs.

        aiohttp starts its sock_read bound only once the request is fully written,
        so for a streamed response, which has no total bound, this is the bound
        on a worker that stops reading a large request.
        """
        return _ResponseStartBound(
            self._session.post(url, data=data, headers=headers, timeout=timeout),
            self._response_start_timeout_secs,
        )

    async def _send_request(
        self,
        endpoint: str,
        request: UCompletionRequest,
        response_type: Type[UCompletionResponse],
        server: Optional[str] = None,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
    ) -> UCompletionResponseOrGenerator:
        placement_started_at: Optional[float] = None
        if server is None:
            if self._closing:
                raise RuntimeError("OpenAIHttpClient is shutting down")
            placement_started_at = _monotonic()
            if req_id is None:
                server, _ = await self._router.get_next_server(request)
            else:
                server, _ = await self._router.get_next_server(request, req_id=req_id)
        # A retry below re-issues disagg_request_id, but the router keyed this
        # request's reservation by the id it was routed with. Pin that id here so
        # every later renew/finish still addresses the original reservation --
        # otherwise the coordinator never sees a release for it and the placement
        # load only drains when the expiration task fires. Only the context role
        # needs this: the coordinator keys generation requests by ctx_request_id,
        # which the retry loop never rewrites.
        if (
            req_id is None
            and self._role == ServerRole.CONTEXT
            and self._disagg_id_generator is not None
        ):
            pinned_dp = request.disaggregated_params
            if pinned_dp is not None:
                req_id = pinned_dp.disagg_request_id
        if self._closing:
            pending_renew = self._pending_renewals.pop(req_id, None) if req_id is not None else None
            try:
                await self._stop_reservation_renewal(pending_renew)
            finally:
                await self._finish_request(request, success=False, req_id=req_id)
            raise RuntimeError("OpenAIHttpClient is shutting down")
        url_server = self._url_server(server, request)
        url = f"{_server_base_url(url_server)}/{endpoint}"
        # The routed server's shared port, used if the affinity port refuses.
        fallback_url = f"{_server_base_url(server)}/{endpoint}" if url_server != server else None
        # disaggregated_params is None when conditional_disagg bypasses ctx.
        _dp = request.disaggregated_params
        _ctx_rid = _dp.ctx_request_id if _dp is not None else None
        logger.debug(f"Sending {self._role} request {_ctx_rid} to {url}")
        self._metrics_collector.total_requests.inc()
        if request.stream:
            resp_generator = self._stream_with_cleanup(
                server,
                url,
                request,
                hooks,
                req_id,
                fallback_url,
                placement_started_at=placement_started_at,
            )
            # Enter its cleanup scope before returning: aclose() on an unstarted
            # async generator would otherwise leave the router reservation live.
            await resp_generator.__anext__()
            return resp_generator
        success = False
        renew_task = self._pending_renewals.pop(req_id, None) if req_id is not None else None
        try:
            response = None
            async with aclosing(
                self._post_with_retry(server, url, request, hooks, req_id, fallback_url)
            ) as response_generator:
                async for resp_json in response_generator:
                    response = response_type(**resp_json)
                    if hooks:
                        if self._role == ServerRole.CONTEXT:
                            hooks.on_ctx_resp(server, response)
                        else:
                            hooks.on_first_token(server, request)
                            hooks.on_resp_done(server, request, response)
            success = True
            return response
        finally:
            try:
                await self._stop_reservation_renewal(renew_task)
            finally:
                if not success:
                    self._metrics_collector.error_requests.inc()
                await self._finish_request(request, success=success, req_id=req_id)

    def _reservation_renewal_interval_s(self) -> Optional[float]:
        interval = getattr(self._router, "reservation_renewal_interval_s", None)
        if isinstance(interval, (int, float)) and not isinstance(interval, bool) and interval > 0:
            return float(interval)
        return None

    async def _renew_reservation_loop(
        self,
        request: UCompletionRequest,
        req_id: Optional[int],
        interval_s: float,
        started_at: Optional[float] = None,
    ) -> None:
        lease_s = interval_s * 3.0
        lease_anchor = _monotonic() if started_at is None else started_at
        next_due = lease_anchor + interval_s
        while True:
            delay = next_due - _monotonic()
            if delay > 0:
                await _async_sleep(delay)
            attempt_start = _monotonic()
            remaining = (lease_anchor + lease_s) - attempt_start
            attempt_timeout = (
                min(interval_s, remaining * 0.4) if remaining > 0 else min(interval_s * 0.25, 0.5)
            )
            try:
                await self._router.renew_request(request, req_id=req_id, timeout_s=attempt_timeout)
                # Anchor the next deadline to attempt_start (at or before the
                # coordinator reset its expiry) so a delayed /renew reply cannot
                # push the next heartbeat past lease expiration.
                lease_anchor = attempt_start
                next_due = attempt_start + interval_s
            except asyncio.CancelledError:
                raise
            except ReservationNotFoundError as e:
                logger.warning(
                    f"Stopping {self._role} reservation renewal for req_id={req_id}: {e}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to renew {self._role} reservation for req_id={req_id}: {e}")
                now = _monotonic()
                remaining_after = (lease_anchor + lease_s) - now
                backoff = (
                    min(interval_s * 0.25, remaining_after * 0.25, 0.5)
                    if remaining_after > 0
                    else min(interval_s * 0.25, 0.5)
                )
                next_due = now + backoff

    def _start_reservation_renewal(
        self,
        request: UCompletionRequest,
        req_id: Optional[int],
        *,
        started_at: Optional[float] = None,
    ) -> Optional[asyncio.Task[None]]:
        if self._closing:
            return None
        interval_s = self._reservation_renewal_interval_s()
        if interval_s is None:
            return None
        task = asyncio.create_task(
            self._renew_reservation_loop(request, req_id, interval_s, started_at=started_at)
        )
        self._active_renewals.add(task)
        task.add_done_callback(self._active_renewals.discard)
        return task

    def start_reservation_renewal(
        self,
        request: UCompletionRequest,
        req_id: Optional[int],
        *,
        started_at: Optional[float] = None,
    ) -> Optional[asyncio.Task[None]]:
        """Start periodic renewal for a pre-reserved request until ``send_request`` adopts it."""
        if self._closing:
            return None
        if req_id is not None:
            existing = self._pending_renewals.get(req_id)
            if existing is not None and not existing.done():
                return existing
        task = self._start_reservation_renewal(request, req_id, started_at=started_at)
        if task is not None and req_id is not None:
            self._pending_renewals[req_id] = task
        return task

    async def stop_reservation_renewal(
        self,
        req_id: Optional[int],
        renew_task: Optional[asyncio.Task[None]] = None,
    ) -> None:
        """Stop and drain a pre-started renewal task if ``send_request`` did not adopt it."""
        pending = self._pending_renewals.pop(req_id, None) if req_id is not None else None
        task = pending if pending is not None else renew_task
        await self._stop_reservation_renewal(task)

    def _take_or_start_reservation_renewal(
        self,
        request: UCompletionRequest,
        req_id: Optional[int],
        *,
        started_at: Optional[float] = None,
    ) -> Optional[asyncio.Task[None]]:
        if self._closing:
            return None
        if req_id is not None:
            pending = self._pending_renewals.pop(req_id, None)
            if pending is not None and not pending.done():
                return pending
        return self._start_reservation_renewal(request, req_id, started_at=started_at)

    @staticmethod
    async def _stop_reservation_renewal(renew_task: Optional[asyncio.Task[None]]) -> None:
        if renew_task is None:
            return
        renew_task.cancel()
        await asyncio.gather(renew_task, return_exceptions=True)

    async def _stream_with_cleanup(
        self,
        server: str,
        url: str,
        request: UCompletionRequest,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
        fallback_url: Optional[str] = None,
        *,
        placement_started_at: Optional[float] = None,
    ) -> AsyncGenerator[Any, None]:
        success = False
        renew_task: Optional[asyncio.Task[None]] = None
        try:
            renew_task = self._take_or_start_reservation_renewal(
                request, req_id, started_at=placement_started_at
            )
            # Consumed by _send_request, before the caller sees this generator.
            yield None
            async with aclosing(
                self._post_with_retry(server, url, request, hooks, req_id, fallback_url)
            ) as response_generator:
                async for chunk in response_generator:
                    yield chunk
            success = True
        finally:
            try:
                await self._stop_reservation_renewal(renew_task)
            finally:
                # Own the logical request, including errors before response headers
                # and cancellation. Individual HTTP attempts must not release it.
                if not success:
                    self._metrics_collector.error_requests.inc()
                await self._finish_request(request, success=success, req_id=req_id)

    async def _post_with_retry(
        self,
        server: str,
        url: str,
        request: UCompletionRequest,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
        fallback_url: Optional[str] = None,
    ) -> AsyncGenerator[Any, None]:
        """POST request to url, retrying within the retry budget.

        If url is a context frontend affinity port that refuses the connection,
        the request moves to fallback_url, the shared port, without using the
        retry budget: it never reached a server.
        """
        is_stream = request.stream
        disagg_params = request.disaggregated_params
        is_remote_tail_handoff = disagg_params is not None and (
            (
                self._role == ServerRole.GENERATION
                and disagg_params.request_type == "generation_only"
                and disagg_params.remote_tail_start is not None
            )
            or (
                self._bounded_replay_on_generation
                and disagg_params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST
                and (
                    (
                        self._role == ServerRole.CONTEXT
                        and disagg_params.request_type == "context_only"
                    )
                    or (
                        self._role == ServerRole.GENERATION
                        and disagg_params.request_type == "generation_only"
                    )
                )
            )
        )
        # Remote-tail retries cannot safely replay or cancel the paired transfer.
        # Generation-first dispatch precedes the context response carrying the
        # tail boundary, so identify that handoff from the deployment config.
        max_retries = 0 if is_remote_tail_handoff else self._max_retries
        # Loop range must cover the transient-TCP extended budget (up to 5)
        # so the conditional raise inside the except block can actually decide
        # to keep retrying.  Non-transient errors still raise on the first
        # attempt that reaches self._max_retries.
        _TRANSIENT_TCP_BUDGET = 0 if self._no_retry or is_remote_tail_handoff else 5
        loop_max = max(max_retries, _TRANSIENT_TCP_BUDGET) + 1
        # A generation leg takes its prompt KV cache from the context server's
        # send session named by disagg_request_id.
        is_generation_leg = (
            self._role == ServerRole.GENERATION
            and disagg_params is not None
            and disagg_params.request_type == "generation_only"
        )
        affinity_refusals = 0
        for attempt in range(loop_max + (fallback_url is not None)):
            retry = attempt - affinity_refusals
            if retry > 0:
                await self._router.renew_request(request, req_id=req_id)
            # A retried request gets a new disagg_request_id so it cannot collide
            # with the abandoned attempt on the worker. A generation leg keeps its
            # id: a new one could never be paired with the context KV cache.
            if retry > 0 and self._disagg_id_generator is not None and not is_generation_leg:
                dp = getattr(request, "disaggregated_params", None)
                if dp is not None and getattr(dp, "disagg_request_id", None) is not None:
                    dp.disagg_request_id = await self._disagg_id_generator()
                    if hooks:
                        hooks.on_disagg_request_id(dp.disagg_request_id)
            # Serialize once on the orchestrator's single event-loop thread.
            # Content-Type stays application/json so FastAPI still routes the
            # body through Request.json(); the header picks the decoder.
            body = _encode_request_body(request)
            headers = dict(MSGPACK_HEADERS)
            if self._request_perf_metrics:
                headers[RETURN_METRICS_HEADER] = "1"
            headers.update(self._get_request_headers(request))
            response_started = False
            streaming_status_rejected = False
            try:
                lines_yielded = 0
                start_time = get_steady_clock_now_in_seconds()
                async with self._post(
                    url,
                    data=body,
                    headers=headers,
                    timeout=self._stream_timeout if is_stream else self._non_stream_timeout,
                ) as http_response:
                    response_started = True
                    response_clock = AdjustedSteadyClock()
                    if self._request_perf_metrics:
                        destination_time = get_steady_clock_now_in_seconds()
                        response_clock = adjusted_clock_from_headers(
                            http_response.headers, start_time, destination_time
                        )
                    content_type = http_response.headers.get("Content-Type", "")
                    if self._request_perf_metrics:
                        role = _metrics_phase(self._role)
                        disagg_params = getattr(request, "disaggregated_params", None)
                        request_id = ""
                        if disagg_params is not None:
                            request_id = str(
                                disagg_params.disagg_request_id
                                or disagg_params.ctx_request_id
                                or ""
                            )
                        response_metrics = build_metrics_record_from_headers(
                            http_response.headers,
                            role,
                            request_id=request_id,
                            adjusted_clock=response_clock,
                        )
                        if hooks and response_metrics:
                            hooks.on_perf_metrics(
                                server,
                                role,
                                response_metrics,
                            )
                    if http_response.status >= 400:
                        if is_stream:
                            # Preserve no-replay semantics even if reading the
                            # rejected response's diagnostics fails.
                            streaming_status_rejected = True
                            # Error diagnostics must not buffer an unbounded worker response.
                            error_prefix = bytearray()
                            while len(error_prefix) < 2048:
                                chunk = await http_response.content.read(2048 - len(error_prefix))
                                if not chunk:
                                    break
                                error_prefix.extend(chunk)
                            error_body = error_prefix.decode("utf-8", errors="replace")
                        else:
                            error_body = await http_response.text()
                        raise aiohttp.ClientResponseError(
                            http_response.request_info,
                            http_response.history,
                            status=http_response.status,
                            message=f"{http_response.reason}: {error_body[:2048]}",
                            headers=http_response.headers,
                        )
                    if not is_stream and "text/event-stream" in content_type:
                        raise ValueError(
                            "Received an event-stream although request stream was False"
                        )
                    if is_stream:
                        # do NOT return generator directly here or the response will go
                        # out of scope and get destroyed
                        async with aclosing(
                            self._response_generator(
                                request,
                                http_response,
                                start_time,
                                server,
                                hooks,
                                req_id,
                                response_clock,
                            )
                        ) as response_generator:
                            async for line in response_generator:
                                lines_yielded += 1
                                yield line
                    else:
                        response_dict = await http_response.json()
                        # yield here since python forbids return statements in async generators
                        yield response_dict
                        self._metrics_collector.complete_latency_seconds.observe(
                            get_steady_clock_now_in_seconds() - start_time
                        )
                break  # break and skip retries if the whole response is processed without exception
            except (aiohttp.ClientError, OSError) as e:
                if fallback_url is not None and isinstance(e, aiohttp.ClientConnectorError):
                    logger.warning(
                        f"Context frontend affinity port of {url} refused the connection: "
                        f"{e} - sending to the shared port {fallback_url}"
                    )
                    self._refused_affinity_ports[_server_netloc(url)] = (
                        get_steady_clock_now_in_seconds()
                    )
                    url, fallback_url = fallback_url, None
                    affinity_refusals = 1
                    continue
                if streaming_status_rejected:
                    raise
                # A 4xx from the worker is the request's own fault (malformed
                # body, context length exceeded, ...) and will fail the same way
                # every time, so retrying only burns another prefill attempt.
                # It reaches this block only because the 4xx is re-raised above
                # as an aiohttp.ClientResponseError -- a ClientError subclass --
                # to carry the response body up; the retry budget here is meant
                # for transport failures, not for server-side rejections.
                # 408 and 429 are transient, not deterministic: keep them retryable.
                if (
                    isinstance(e, aiohttp.ClientResponseError)
                    and 400 <= e.status < 500
                    and e.status not in (408, 429)
                ):
                    logger.error(
                        f"Client error to {url}: {e} - not retrying, {e.status} is deterministic",
                        traceback.format_exc(),
                    )
                    raise
                if lines_yielded > 0:
                    logger.error(
                        f"Client error to {url}: {e} - cannot retry since {lines_yielded} lines were yielded",
                        traceback.format_exc(),
                    )
                    raise
                # Once the generation server may have accepted a generation leg,
                # that attempt owns the context KV cache: the transfer either went
                # to it or failed and was released on the context server. A re-post
                # would reuse its disagg_request_id and collide with the original
                # on the worker, or wait for KV cache that is never sent.
                if is_generation_leg and (
                    response_started or not isinstance(e, _GENERATION_LEG_RETRYABLE_ERRORS)
                ):
                    logger.error(
                        f"Client error to {url}: {e} - not retrying a generation request "
                        "the generation server may have accepted",
                        traceback.format_exc(),
                    )
                    raise
                # ServerDisconnectedError and connection resets can result from
                # transient TCP races, such as client/server keepalive races at
                # burst start. Give them an extended retry budget to recover
                # while preserving the original fail-fast for other errors.
                # aiohttp can wrap an OS connection reset in ClientOSError
                # rather than a ConnectionResetError subclass.
                is_transient_tcp = isinstance(
                    e,
                    (aiohttp.ServerDisconnectedError, ConnectionResetError),
                ) or (isinstance(e, aiohttp.ClientOSError) and e.errno == errno.ECONNRESET)
                effective_max = max_retries
                if is_transient_tcp:
                    effective_max = max(max_retries, _TRANSIENT_TCP_BUDGET)
                if retry >= effective_max:
                    logger.error(
                        f"Client error to {url}: {e} - last retry {retry} of {effective_max}failed",
                        traceback.format_exc(),
                    )
                    raise
                logger.error(
                    f"{self._role} client error to {url}: {e} - retry {retry} of {effective_max}",
                    traceback.format_exc(),
                )
                await self._router.renew_request(request, req_id=req_id)
                await asyncio.sleep(self._retry_interval_sec)
                self._metrics_collector.retry_requests.inc()
            except Exception as e:
                logger.error(
                    f"Unexpected error while processing {self._role} request to {url}: {e}"
                )
                raise

    async def _response_generator(
        self,
        request: UCompletionRequest,
        http_response: aiohttp.ClientResponse,
        start_time: float,
        server: str,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
        response_clock: Optional[AdjustedSteadyClock] = None,
    ) -> AsyncGenerator[Any, None]:
        assert request.stream, "Request is not streaming"
        assert "text/event-stream" in http_response.headers.get("Content-Type", ""), (
            "Response is not streaming"
        )
        try:
            last_token_time = start_time
            chunk_count = 0
            # The first token of a Responses stream is the first frame after
            # its opening events, which the worker sends before generating.
            awaiting_output = isinstance(request, ResponsesRequest)
            marker = f"event: {SSE_METRICS_EVENT}\n".encode()
            event_delimiter = b"\n\n"
            pending = b""
            metrics_event = b""
            async for chunk in http_response.content.iter_any():
                now_time = get_steady_clock_now_in_seconds()
                if awaiting_output and chunk:
                    awaiting_output = not _has_responses_output_event(chunk)
                if chunk and not awaiting_output:
                    if chunk_count == 0 and hooks:
                        hooks.on_first_token(server, request)
                    latency = now_time - last_token_time
                    metric = (
                        self._metrics_collector.first_token_latency_seconds
                        if chunk_count == 0
                        else self._metrics_collector.per_token_latency_seconds
                    )
                    metric.observe(latency)
                    chunk_count += 1
                    last_token_time = now_time

                if not self._request_perf_metrics:
                    yield chunk
                    await asyncio.sleep(0)
                    continue

                if metrics_event:
                    metrics_event += chunk
                    continue
                pending += chunk
                marker_index = pending.find(marker)
                if marker_index >= 0:
                    if marker_index:
                        yield pending[:marker_index]
                    metrics_event = pending[marker_index:]
                    pending = b""
                    continue
                event_end = pending.rfind(event_delimiter)
                if event_end >= 0:
                    emit_size = event_end + len(event_delimiter)
                    yield pending[:emit_size]
                    pending = pending[emit_size:]
                await asyncio.sleep(0)

            if pending:
                yield pending
            if hooks and metrics_event:
                data_prefix = b"data: "
                data_index = metrics_event.find(data_prefix)
                if data_index >= 0:
                    data = metrics_event[data_index + len(data_prefix) :].split(b"\n", 1)[0]
                    try:
                        headers = json.loads(data)
                        metrics = build_metrics_record_from_headers(
                            headers,
                            _metrics_phase(self._role),
                            adjusted_clock=response_clock,
                        )
                    except (TypeError, ValueError) as error:
                        logger.warning("Ignoring malformed perf metrics event: %s", error)
                        metrics = None
                    if metrics:
                        hooks.on_perf_metrics(server, _metrics_phase(self._role), metrics)

            if hooks:
                hooks.on_resp_done(server, request, None)
            self._metrics_collector.complete_latency_seconds.observe(
                get_steady_clock_now_in_seconds() - start_time
            )
        except aiohttp.ClientError as e:
            # a client error is expected when the response stream is done if the connector has close=True
            logger.error(f"{self._role} client {server} error: {e}")
            raise

    async def _finish_request(
        self,
        request: UCompletionRequest,
        success: bool = True,
        req_id: Optional[int] = None,
    ) -> None:
        self._metrics_collector.completed_requests.inc()
        if req_id is None:
            await self._router.finish_request(request, self._session, success=success)
        else:
            await self._router.finish_request(
                request, self._session, success=success, req_id=req_id
            )

    async def shutdown(self) -> None:
        self._closing = True
        while self._pending_renewals or self._active_renewals:
            live_renewals = set(self._pending_renewals.values()) | set(self._active_renewals)
            self._pending_renewals.clear()
            for task in live_renewals:
                task.cancel()
            await asyncio.gather(*live_renewals, return_exceptions=True)
            self._active_renewals.difference_update(live_renewals)
        await self._session.close()

    async def check_ready(self) -> Tuple[List[str], List[str]]:
        ready_servers, unready_servers = await OpenAIHttpClient.check_ready_for_servers(
            self._session, self._router.servers
        )
        if ready_servers:
            await self._router.prepare_servers(ready_servers)
        return ready_servers, unready_servers

    @staticmethod
    async def check_ready_for_servers(
        session: aiohttp.ClientSession, servers: List[str]
    ) -> Tuple[List[str], List[str]]:
        async def check_server_ready(server: str) -> bool:
            try:
                url = f"{_server_base_url(server)}/health"
                async with session.get(url) as response:
                    return response.status == 200
            except Exception:
                return False

        servers_ready = await asyncio.gather(*[check_server_ready(server) for server in servers])
        return [server for server, ready in zip(servers, servers_ready) if ready], [
            server for server, ready in zip(servers, servers_ready) if not ready
        ]
