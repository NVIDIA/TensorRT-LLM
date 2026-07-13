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
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Type

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


class OpenAIClient(ABC):
    async def send_request(
        self,
        request: UCompletionRequest,
        server: Optional[str] = None,
        hooks: Optional[ResponseHooks] = None,
    ) -> UCompletionResponseOrGenerator:
        if isinstance(request, CompletionRequest):
            return await self._send_request(
                "v1/completions", request, CompletionResponse, server, hooks
            )
        elif isinstance(request, ChatCompletionRequest):
            return await self._send_request(
                "v1/chat/completions", request, ChatCompletionResponse, server, hooks
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
    async def _finish_request(self, request: UCompletionRequest, success: bool = True) -> None:
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
        disagg_id_generator: Optional[Callable[[], int]] = None,
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

    async def _send_request(
        self,
        endpoint: str,
        request: UCompletionRequest,
        response_type: Type[UCompletionResponse],
        server: Optional[str] = None,
        hooks: Optional[ResponseHooks] = None,
    ) -> UCompletionResponseOrGenerator:
        if server is None:
            server, _ = await self._router.get_next_server(request)
        url = f"http://{server}/{endpoint}"
        # disaggregated_params is None when conditional_disagg bypasses ctx.
        _dp = request.disaggregated_params
        _ctx_rid = _dp.ctx_request_id if _dp is not None else None
        logger.debug(f"Sending {self._role} request {_ctx_rid} to {url}")
        try:
            self._metrics_collector.total_requests.inc()
            resp_generator = self._post_with_retry(server, url, request, hooks)
            if request.stream:
                # return the response generator, the request is not done yet
                return resp_generator
            else:
                # consume the generator to get the response and return it directly when it's not streaming
                response = None
                async for resp_json in resp_generator:
                    response = response_type(**resp_json)
                    if hooks:
                        if self._role == ServerRole.CONTEXT:
                            hooks.on_ctx_resp(server, response)
                        else:
                            hooks.on_first_token(server, request)
                            hooks.on_resp_done(server, request, response)
                return response
        except Exception:
            self._metrics_collector.error_requests.inc()
            # finish the request upon error
            await self._finish_request(request, success=False)
            raise

    async def _post_with_retry(
        self,
        server: str,
        url: str,
        request: UCompletionRequest,
        hooks: Optional[ResponseHooks] = None,
    ) -> AsyncGenerator[Any, None]:
        is_stream = request.stream
        # Loop range must cover the transient-TCP extended budget (up to 5)
        # so the conditional raise inside the except block can actually decide
        # to keep retrying.  Non-transient errors still raise on the first
        # attempt that reaches self._max_retries.
        _TRANSIENT_TCP_BUDGET = 5
        loop_max = max(self._max_retries, _TRANSIENT_TCP_BUDGET) + 1
        for attempt in range(loop_max):
            # Regenerate disagg_request_id on retry to avoid ID collision on workers
            if attempt > 0 and self._disagg_id_generator is not None:
                dp = getattr(request, "disaggregated_params", None)
                if dp is not None and getattr(dp, "disagg_request_id", None) is not None:
                    dp.disagg_request_id = self._disagg_id_generator()
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
                    if not is_stream and "text/event-stream" in content_type:
                        raise ValueError(
                            "Received an event-stream although request stream was False"
                        )
                    if is_stream:
                        # do NOT return generator directly here or the response will go
                        # out of scope and get destroyed
                        async for line in self._response_generator(
                            request, http_response, start_time, server, hooks
                        ):
                            lines_yielded += 1
                            yield line
                        # don't finish the request here since the response generator is not done yet
                    else:
                        if http_response.status >= 400:
                            error_body = await http_response.text()
                            raise aiohttp.ClientResponseError(
                                http_response.request_info,
                                http_response.history,
                                status=http_response.status,
                                message=f"{http_response.reason}: {error_body[:2048]}",
                                headers=http_response.headers,
                            )
                        response_dict = await http_response.json()
                        # yield here since python forbids return statements in async generators
                        yield response_dict
                        # finish the request after the successful response
                        await self._finish_request(request)
                        self._metrics_collector.complete_latency_seconds.observe(
                            get_steady_clock_now_in_seconds() - start_time
                        )
                break  # break and skip retries if the whole response is processed without exception
            except (aiohttp.ClientError, OSError) as e:
                if lines_yielded > 0:
                    logger.error(
                        f"Client error to {url}: {e} - cannot retry since {lines_yielded} lines were yielded",
                        traceback.format_exc(),
                    )
                    raise
                # Selective retry budget: ServerDisconnectedError and
                # ConnectionResetError are transient TCP races (typically at
                # burst start when client keepalive vs server keepalive race).
                # Give them an extended retry budget while preserving the
                # original fail-fast for genuine upstream errors.
                is_transient_tcp = isinstance(
                    e,
                    (aiohttp.ServerDisconnectedError, ConnectionResetError),
                )
                effective_max = self._max_retries
                if is_transient_tcp:
                    effective_max = max(self._max_retries, _TRANSIENT_TCP_BUDGET)
                if attempt >= effective_max:
                    logger.error(
                        f"Client error to {url}: {e} - last retry {attempt} of {effective_max}"
                        "failed",
                        traceback.format_exc(),
                    )
                    raise
                logger.error(
                    f"{self._role} client error to {url}: {e} - retry {attempt} of {effective_max}",
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
    ) -> AsyncGenerator[Any, None]:
        assert request.stream, "Request is not streaming"
        assert "text/event-stream" in http_response.headers.get("Content-Type", ""), (
            "Response is not streaming"
        )
        success = True
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
            self._metrics_collector.error_requests.inc()
            success = False
            raise
        except Exception:
            self._metrics_collector.error_requests.inc()
            success = False
            raise
        finally:
            # finish the request after streaming response is done or error is raised
            await self._finish_request(request, success=success)

    async def _finish_request(self, request: UCompletionRequest, success: bool = True) -> None:
        self._metrics_collector.completed_requests.inc()
        await self._router.finish_request(request, self._session, success=success)

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
