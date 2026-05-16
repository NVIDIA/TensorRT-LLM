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


def _summarize_disagg_params(request: UCompletionRequest) -> str:
    params = getattr(request, "disaggregated_params", None)
    if params is None:
        return "none"
    encoded_opaque_state = getattr(params, "encoded_opaque_state", None)
    first_gen_tokens = getattr(params, "first_gen_tokens", None)
    draft_tokens = getattr(params, "draft_tokens", None)
    return (
        f"request_type={getattr(params, 'request_type', None)!r} "
        f"ctx_request_id={getattr(params, 'ctx_request_id', None)!r} "
        f"disagg_request_id={getattr(params, 'disagg_request_id', None)!r} "
        f"schedule_style={getattr(params, 'schedule_style', None)!r} "
        f"ctx_dp_rank={getattr(params, 'ctx_dp_rank', None)!r} "
        f"ctx_info_endpoint={getattr(params, 'ctx_info_endpoint', None)!r} "
        f"opaque_state_bytes={len(encoded_opaque_state) if encoded_opaque_state else 0} "
        f"first_gen_tokens={len(first_gen_tokens) if first_gen_tokens else 0} "
        f"draft_tokens={len(draft_tokens) if draft_tokens else 0}"
    )


def _summarize_request(request: UCompletionRequest) -> str:
    if isinstance(request, CompletionRequest):
        prompt = request.prompt
        if isinstance(prompt, str):
            prompt_summary = f"prompt_chars={len(prompt)}"
        elif isinstance(prompt, list):
            prompt_summary = f"prompt_list_len={len(prompt)}"
        else:
            prompt_summary = f"prompt_type={type(prompt).__name__}"
    elif isinstance(request, ChatCompletionRequest):
        messages = getattr(request, "messages", [])
        prompt_token_ids = getattr(request, "prompt_token_ids", None)
        prompt_summary = (
            f"messages={len(messages)} "
            f"prompt_token_ids={len(prompt_token_ids) if prompt_token_ids else 0}"
        )
    else:
        prompt_summary = f"request_type={type(request).__name__}"

    return (
        f"model={getattr(request, 'model', None)!r} "
        f"stream={getattr(request, 'stream', None)!r} "
        f"max_tokens={getattr(request, 'max_tokens', None)!r} "
        f"temperature={getattr(request, 'temperature', None)!r} "
        f"ignore_eos={getattr(request, 'ignore_eos', None)!r} "
        f"{prompt_summary} disagg=({_summarize_disagg_params(request)})"
    )


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
    async def _finish_request(self, request: UCompletionRequest) -> None:
        """Finish the request in the router."""
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
        logger.info(
            f"[disagg-debug] OpenAI client send start: role={self._role} "
            f"endpoint={endpoint} server={server} url={url} "
            f"request=({_summarize_request(request)})"
        )
        try:
            self._metrics_collector.total_requests.inc()
            resp_generator = self._post_with_retry(server, url, request, hooks)
            if request.stream:
                # return the response generator, the request is not done yet
                logger.info(
                    f"[disagg-debug] OpenAI client returning streaming generator: "
                    f"role={self._role} url={url}"
                )
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
                logger.info(
                    f"[disagg-debug] OpenAI client send done: role={self._role} "
                    f"url={url} response_type={type(response).__name__ if response else None}"
                )
                return response
        except Exception:
            self._metrics_collector.error_requests.inc()
            # finish the request upon error
            logger.error(
                f"[disagg-debug] OpenAI client send failed: role={self._role} "
                f"url={url} request=({_summarize_request(request)})",
                traceback.format_exc(),
            )
            await self._finish_request(request)
            raise

    async def _post_with_retry(
        self,
        server: str,
        url: str,
        request: UCompletionRequest,
        hooks: Optional[ResponseHooks] = None,
    ) -> AsyncGenerator[Any, None]:
        is_stream = request.stream
        for attempt in range(self._max_retries + 1):
            # Regenerate disagg_request_id on retry to avoid ID collision on workers
            if attempt > 0 and self._disagg_id_generator is not None:
                dp = getattr(request, "disaggregated_params", None)
                if dp is not None and getattr(dp, "disagg_request_id", None) is not None:
                    dp.disagg_request_id = self._disagg_id_generator()
            json_data = request.model_dump(exclude_unset=True, mode="json")
            lines_yielded = 0
            try:
                start_time = get_steady_clock_now_in_seconds()
                logger.info(
                    f"[disagg-debug] HTTP post start: role={self._role} "
                    f"attempt={attempt}/{self._max_retries} server={server} "
                    f"url={url} stream={is_stream} payload_keys={sorted(json_data.keys())} "
                    f"request=({_summarize_request(request)})"
                )
                async with self._session.post(url, json=json_data) as http_response:
                    content_type = http_response.headers.get("Content-Type", "")
                    logger.info(
                        f"[disagg-debug] HTTP post response headers: role={self._role} "
                        f"attempt={attempt}/{self._max_retries} url={url} "
                        f"status={http_response.status} reason={http_response.reason!r} "
                        f"content_type={content_type!r} elapsed_s="
                        f"{get_steady_clock_now_in_seconds() - start_time:.3f}"
                    )
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
                        logger.info(
                            f"[disagg-debug] HTTP post JSON received: role={self._role} "
                            f"url={url} response_keys={sorted(response_dict.keys())} "
                            f"elapsed_s={get_steady_clock_now_in_seconds() - start_time:.3f}"
                        )
                        # yield here since python forbids return statements in async generators
                        yield response_dict
                        # finish the request after the successful response
                        await self._finish_request(request)
                        self._metrics_collector.complete_latency_seconds.observe(
                            get_steady_clock_now_in_seconds() - start_time
                        )
                        logger.info(
                            f"[disagg-debug] HTTP post complete: role={self._role} "
                            f"url={url} elapsed_s={get_steady_clock_now_in_seconds() - start_time:.3f}"
                        )
                break  # break and skip retries if the whole response is processed without exception
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as e:
                if lines_yielded > 0:
                    logger.error(
                        f"Client error to {url}: {e} - cannot retry since {lines_yielded} lines were yielded",
                        traceback.format_exc(),
                    )
                    raise
                if attempt == self._max_retries:
                    logger.error(
                        f"Client error to {url}: {e} - last retry {attempt} of {self._max_retries}"
                        "failed",
                        traceback.format_exc(),
                    )
                    raise
                logger.error(
                    f"{self._role} client error to {url}: {e} - retry {attempt} of {self._max_retries}",
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
            raise
        except Exception:
            self._metrics_collector.error_requests.inc()
            raise
        finally:
            # finish the request after streaming response is done or error is raised
            await self._finish_request(request)

    async def _finish_request(self, request: UCompletionRequest) -> None:
        self._metrics_collector.completed_requests.inc()
        await self._router.finish_request(request)

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
