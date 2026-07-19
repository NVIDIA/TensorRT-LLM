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
import os
import traceback
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple, Type, cast

import aiohttp

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    UCompletionRequest,
    UCompletionResponse,
)
from tensorrt_llm.serve.perf_metrics import ClientMetricsCollector
from tensorrt_llm.serve.responses_utils import (
    ResponseHooks,
    UCompletionResponseOrGenerator,
    get_steady_clock_now_in_seconds,
)
from tensorrt_llm.serve.router import Router

# yapf: enable

# msgspec msgpack is an opt-in transport for the orchestrator->worker request
# body (alternative to the JSON path). Enable with TRTLLM_SERVE_ENABLE_MSGSPEC=1;
# the orchestrator encodes the forwarded body as msgpack and flags it with the
# X-TRTLLM-Msgpack header (Content-Type stays application/json so FastAPI still
# routes it through Request.json()). Fails loudly at import if msgspec is missing.
_MSGSPEC_ENABLED = os.getenv("TRTLLM_SERVE_ENABLE_MSGSPEC", "0") == "1"
if _MSGSPEC_ENABLED:
    try:
        import msgspec
    except ImportError as exc:
        raise ImportError(
            "TRTLLM_SERVE_ENABLE_MSGSPEC=1 requires the msgspec package "
            "(listed in requirements.txt)."
        ) from exc
    _msgpack_encoder = msgspec.msgpack.Encoder()


class UpstreamRequestTimeoutError(TimeoutError):
    """An orchestrator request to a disaggregated worker timed out."""

    def __init__(self, role: ServerRole, timeout_secs: int, detail: Optional[str] = None):
        self.role = role
        self.timeout_secs = timeout_secs
        message = (
            f"{role.name.lower()} worker request exceeded the configured "
            f"{timeout_secs}-second timeout"
        )
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class _ManagedResponseStream:
    """Own router cleanup even when a response stream is never started."""

    def __init__(self, client, request, resp_generator, req_id: Optional[int] = None):
        self._client = client
        self._request = request
        self._resp_generator = resp_generator
        self._req_id = req_id
        self._cleanup_lock = asyncio.Lock()
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.asend(None)

    async def asend(self, value):
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._resp_generator.asend(value)
        except StopAsyncIteration:
            await self._finish(success=True)
            raise
        except asyncio.CancelledError:
            await self._finish(success=False)
            raise
        except Exception:
            await self._finish(success=False)
            raise

    async def athrow(self, *args):
        if self._closed:
            return None
        try:
            return await self._resp_generator.athrow(*args)
        except (StopAsyncIteration, GeneratorExit):
            await self._finish(success=False)
            raise
        except asyncio.CancelledError:
            await self._finish(success=False)
            raise
        except Exception:
            await self._finish(success=False)
            raise

    async def aclose(self):
        await self._finish(success=False)

    async def _finish(self, success: bool) -> None:
        async with self._cleanup_lock:
            if self._closed:
                return
            if not success:
                self._client._metrics_collector.error_requests.inc()
            try:
                await self._client._cleanup_request_safely(
                    self._request,
                    self._resp_generator,
                    success=success,
                    req_id=self._req_id,
                )
            finally:
                self._closed = True


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
                "v1/completions", request, CompletionResponse, server, hooks, req_id
            )
        elif isinstance(request, ChatCompletionRequest):
            return await self._send_request(
                "v1/chat/completions",
                request,
                ChatCompletionResponse,
                server,
                hooks,
                req_id,
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

    @abstractmethod
    async def collect_metrics(self) -> Dict[str, Any]: ...

    @abstractmethod
    async def check_ready(self) -> Tuple[List[str], List[str]]:
        """Return the list of ready servers and the list of unready servers."""
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
    ):
        self._router = router
        self._role = role
        self._metrics_collector = ClientMetricsCollector(role)
        self._session = session or aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=0,
                limit_per_host=0,
                force_close=False,
                # Set keepalive_timeout below the server-side keepalive timeout to avoid reusing stale connections.
                keepalive_timeout=1,
            ),
            timeout=aiohttp.ClientTimeout(total=timeout_secs),
        )
        self._max_retries = max_retries
        self._retry_interval_sec = retry_interval_sec
        self._disagg_id_generator = disagg_id_generator
        self._timeout_secs = timeout_secs

    async def _send_request(
        self,
        endpoint: str,
        request: UCompletionRequest,
        response_type: Type[UCompletionResponse],
        server: Optional[str] = None,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
    ) -> UCompletionResponseOrGenerator:
        self._metrics_collector.total_requests.inc()
        resp_generator = None
        cleanup_deferred = False
        success = False
        try:
            if server is None:
                if req_id is None:
                    server, _ = await self._router.get_next_server(request)
                else:
                    server, _ = await self._router.get_next_server(request, req_id=req_id)
            url = f"http://{server}/{endpoint}"
            # disaggregated_params is None when conditional_disagg bypasses ctx.
            _dp = request.disaggregated_params
            _ctx_rid = _dp.ctx_request_id if _dp is not None else None
            logger.debug(f"Sending {self._role} request {_ctx_rid} to {url}")
            resp_generator = self._post_with_retry(server, url, request, hooks, req_id)
            if request.stream:
                # The POST is lazy for streaming requests. Keep cleanup around
                # the generator so cancellation releases the router reservation.
                response_stream = cast(
                    UCompletionResponseOrGenerator,
                    _ManagedResponseStream(self, request, resp_generator, req_id),
                )
                cleanup_deferred = True
                return response_stream

            response = None
            async for resp_json in resp_generator:
                response = response_type(**resp_json)
                if hooks:
                    if self._role == ServerRole.CONTEXT:
                        hooks.on_ctx_resp(server, response)
                    else:
                        hooks.on_first_token(server, request)
                        hooks.on_resp_done(server, request, response)
            success = True
            return response
        except asyncio.CancelledError:
            self._metrics_collector.error_requests.inc()
            raise
        except Exception:
            self._metrics_collector.error_requests.inc()
            raise
        finally:
            if not cleanup_deferred:
                await self._cleanup_request_safely(
                    request, resp_generator, success=success, req_id=req_id
                )

    async def _cleanup_request_safely(
        self,
        request: UCompletionRequest,
        resp_generator: Optional[AsyncGenerator[Any, None]],
        success: bool,
        req_id: Optional[int] = None,
    ) -> None:
        """Complete response and router cleanup before propagating cancellation."""

        async def cleanup() -> None:
            try:
                if resp_generator is not None:
                    await resp_generator.aclose()
            finally:
                await self._finish_request(request, success=success, req_id=req_id)

        cleanup_task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            # The parent deadline may race the aiohttp timeout. Preserve the
            # router lease until cleanup finishes, then propagate cancellation.
            await cleanup_task
            raise

    async def _post_with_retry(
        self,
        server: str,
        url: str,
        request: UCompletionRequest,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
    ) -> AsyncGenerator[Any, None]:
        is_stream = request.stream
        for attempt in range(self._max_retries + 1):
            # Regenerate disagg_request_id on retry to avoid ID collision on workers
            if attempt > 0 and self._disagg_id_generator is not None:
                dp = getattr(request, "disaggregated_params", None)
                if dp is not None and getattr(dp, "disagg_request_id", None) is not None:
                    dp.disagg_request_id = await self._disagg_id_generator()
            # Serialize once on the orchestrator's single event-loop thread.
            if _MSGSPEC_ENABLED:
                # msgspec msgpack: encode the request dict to msgpack bytes. Keep
                # Content-Type application/json so FastAPI still routes the body
                # through Request.json() (it only does that for json/+json content
                # subtypes); the X-TRTLLM-Msgpack header tells the worker's
                # Request.json() to decode with msgspec instead of stdlib json.
                body = _msgpack_encoder.encode(request.model_dump(mode="json", exclude_unset=True))
                req_headers = {"Content-Type": "application/json", "X-TRTLLM-Msgpack": "1"}
            else:
                body = request.model_dump_json(exclude_unset=True)
                req_headers = {"Content-Type": "application/json"}
            try:
                lines_yielded = 0
                start_time = get_steady_clock_now_in_seconds()
                async with self._session.post(
                    url,
                    data=body,
                    headers=req_headers,
                ) as http_response:
                    content_type = http_response.headers.get("Content-Type", "")
                    if http_response.status >= 400:
                        if http_response.status == 504:
                            try:
                                error_body = await http_response.text()
                            except (
                                aiohttp.ClientError,
                                asyncio.TimeoutError,
                                OSError,
                                UnicodeError,
                            ) as e:
                                # The HTTP status is authoritative. A truncated
                                # timeout body must not turn a known 504 into a
                                # retryable transport error and replay work.
                                logger.warning(
                                    f"Failed to read {self._role} timeout body from {url}: {e}"
                                )
                                error_body = ""
                            raise UpstreamRequestTimeoutError(
                                self._role,
                                self._timeout_secs,
                                detail=error_body[:2048],
                            )
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
                        async for line in self._response_generator(
                            request, http_response, start_time, server, hooks, req_id
                        ):
                            lines_yielded += 1
                            yield line
                        # don't finish the request here since the response generator is not done yet
                    else:
                        response_dict = await http_response.json()
                        # yield here since python forbids return statements in async generators
                        yield response_dict
                        self._metrics_collector.complete_latency_seconds.observe(
                            get_steady_clock_now_in_seconds() - start_time
                        )
                break  # break and skip retries if the whole response is processed without exception
            except UpstreamRequestTimeoutError:
                raise
            except asyncio.TimeoutError as e:
                logger.error(
                    f"{self._role} request to {url} timed out after {self._timeout_secs} seconds"
                )
                raise UpstreamRequestTimeoutError(self._role, self._timeout_secs) from e
            except (aiohttp.ClientError, OSError) as e:
                if lines_yielded > 0:
                    logger.error(
                        f"Client error to {url}: {e} - cannot retry since {lines_yielded} lines were yielded",
                        traceback.format_exc(),
                    )
                    raise
                if attempt >= self._max_retries:
                    logger.error(
                        f"Client error to {url}: {e} - last retry {attempt} of "
                        f"{self._max_retries} failed",
                        traceback.format_exc(),
                    )
                    raise
                logger.error(
                    f"{self._role} client error to {url}: {e} - retry {attempt} "
                    f"of {self._max_retries}",
                    traceback.format_exc(),
                )
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
    ) -> AsyncGenerator[Any, None]:
        assert request.stream, "Request is not streaming"
        assert "text/event-stream" in http_response.headers.get("Content-Type", ""), (
            "Response is not streaming"
        )
        try:
            last_token_time = start_time
            i = 0
            async for line in http_response.content.iter_any():
                now_time = get_steady_clock_now_in_seconds()
                if line:
                    if i == 0:
                        if hooks:
                            hooks.on_first_token(server, request)
                        self._metrics_collector.first_token_latency_seconds.observe(
                            now_time - last_token_time
                        )
                    else:
                        self._metrics_collector.per_token_latency_seconds.observe(
                            now_time - last_token_time
                        )
                    i += 1
                    yield line
                    await asyncio.sleep(0)
                last_token_time = now_time

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

    async def collect_metrics(self) -> Dict[str, Any]:
        metrics = {}
        for server in self._router.servers:
            try:
                async with self._session.get(f"http://{server}/perf_metrics") as response:
                    metrics[server] = await response.json()
            except Exception:
                logger.error(f"Failed to collect metrics from {server}")
                continue
        return metrics

    async def shutdown(self) -> None:
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
                url = (
                    f"{server}/health"
                    if server.startswith("http://")
                    else f"http://{server}/health"
                )
                async with session.get(url) as response:
                    return response.status == 200
            except Exception:
                return False

        servers_ready = await asyncio.gather(*[check_server_ready(server) for server in servers])
        return [server for server, ready in zip(servers, servers_ready) if ready], [
            server for server, ready in zip(servers, servers_ready) if not ready
        ]
