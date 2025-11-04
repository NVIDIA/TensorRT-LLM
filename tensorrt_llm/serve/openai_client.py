# Copyright (c) 2025, NVIDIA CORPORATION.
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
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type

import aiohttp

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    UCompletionRequest,
    UCompletionResponse,
)
from tensorrt_llm.serve.perf_metrics import ClientMetricsCollector, DisaggPerfMetricsCollector
from tensorrt_llm.serve.responses_utils import (
    ResponseHooks,
    UCompletionResponseOrGenerator,
    get_steady_clock_now_in_seconds,
)
from tensorrt_llm.serve.router import Router

# yapf: enable


class OpenAIClient(ABC):
    async def send_request(
        self, server: str, request: UCompletionRequest, hooks: Optional[ResponseHooks] = None
    ) -> UCompletionResponseOrGenerator:
        if isinstance(request, CompletionRequest):
            return await self._send_request(
                server, "v1/completions", request, CompletionResponse, hooks
            )
        elif isinstance(request, ChatCompletionRequest):
            return await self._send_request(
                server, "v1/chat/completions", request, ChatCompletionResponse, hooks
            )
        else:
            raise ValueError(f"Invalid request type: {type(request)}")

    @abstractmethod
    async def _send_request(
        self,
        server: str,
        endpoint: str,
        request: UCompletionRequest,
        response_type: Type[UCompletionResponse],
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
        client_type: str,
        timeout_secs: int = 180,
        max_retries: int = 1,
        perf_metrics_collector: DisaggPerfMetricsCollector = None,
    ):
        assert client_type in ["ctx", "gen"]
        self._router = router
        self._client_type = client_type
        self._metrics_collector = ClientMetricsCollector(client_type)
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=False),
            timeout=aiohttp.ClientTimeout(total=timeout_secs),
        )
        self._max_retries = max_retries
        self._retry_interval = 1

    async def _send_request(
        self,
        server: str,
        endpoint: str,
        request: UCompletionRequest,
        response_type: Type[UCompletionResponse],
        hooks: Optional[ResponseHooks] = None,
    ) -> UCompletionResponseOrGenerator:
        if len(server) == 0:
            server, _ = await self._router.get_next_server(request)
        url = f"http://{server}/{endpoint}"
        logger.debug(
            f"Sending {self._client_type} request {request.disaggregated_params.ctx_request_id} to {url}"
        )
        try:
            self._metrics_collector.inc("total_requests")
            resp_generator = self._post_with_retry(server, url, request, hooks)
            if request.stream:
                return resp_generator
            else:
                # consume the generator to get the response and return it directly when it's not streaming
                response = None
                async for resp_json in resp_generator:
                    response = response_type(**resp_json)
                    if hooks:
                        if self._client_type == "ctx":
                            hooks.on_ctx_resp(server, response)
                        else:
                            hooks.on_first_token(server, request)
                            hooks.on_resp_done(server, request, response)
                return response
        except Exception:
            self._metrics_collector.inc("error_requests")
            raise

    async def _post_with_retry(
        self,
        server: str,
        url: str,
        request: UCompletionRequest,
        hooks: Optional[ResponseHooks] = None,
    ) -> AsyncGenerator[Any, None]:
        json_data = request.model_dump(exclude_unset=True)
        is_stream = request.stream
        for attempt in range(self._max_retries + 1):
            try:
                start_time = get_steady_clock_now_in_seconds()
                async with self._session.post(url, json=json_data) as http_response:
                    content_type = http_response.headers.get("Content-Type", "")
                    if not is_stream and "text/event-stream" in content_type:
                        raise ValueError(
                            "Received an event-stream although request stream was False"
                        )
                    if is_stream:
                        # do NOT return generator directly here or the response will go
                        # out of scope and get destroyed
                        async for line in self._response_generator(
                            request, http_response, start_time, hooks, server
                        ):
                            yield line
                    else:
                        http_response.raise_for_status()
                        response_dict = await http_response.json()
                        # yield here since python forbids return statements in async generators
                        yield response_dict
                break  # break and skip retries if the whole response is processed without exception
            except (aiohttp.ClientError, OSError) as e:
                if attempt == self._max_retries:
                    logger.error(
                        f"{self._client_type} client error to {url}: {e} - last retry {attempt} of {self._max_retries}"
                        "failed",
                        traceback.format_exc(),
                    )
                    raise

                logger.error(
                    f"{self._client_type} client error to {url}: {e} - retry {attempt} of {self._max_retries}",
                    traceback.format_exc(),
                )
                await asyncio.sleep(self._retry_interval)
                self._metrics_collector.inc("retry_requests")
            except Exception as e:
                logger.error(
                    f"Unexpected error while processing {self._client_type} request to {url}: {e}"
                )
                raise
            finally:
                await self._finish_request(request)

    async def _response_generator(
        self,
        request: UCompletionRequest,
        http_response: aiohttp.ClientResponse,
        start_time: float,
        hooks: Optional[ResponseHooks] = None,
        server: str = "",
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
                if i == 0:
                    if hooks:
                        hooks.on_first_token(server, request)
                    self._metrics_collector.observe(
                        "first_token_latency_seconds", now_time - last_token_time
                    )
                else:
                    self._metrics_collector.observe(
                        "per_token_latency_seconds", now_time - last_token_time
                    )
                i += 1
                if line:
                    yield line
                    await asyncio.sleep(0)
                last_token_time = now_time

            if hooks:
                hooks.on_resp_done(server, request, None)
            self._metrics_collector.inc("completed_requests")
            self._metrics_collector.observe(
                "complete_latency_seconds",
                get_steady_clock_now_in_seconds() - start_time,
            )
        except aiohttp.ClientError as e:
            # a client error is expected when the response stream is done if the connector has close=True
            logger.error(f"{self._client_type} Client error: {e}")
            self._metrics_collector.inc("error_requests")
            raise
        except Exception:
            self._metrics_collector.inc("error_requests")
            raise
        finally:
            await self._finish_request(request)

    async def _finish_request(self, request: UCompletionRequest) -> None:
        await self._router.finish_request(request)

    async def collect_metrics(self) -> Dict[str, Any]:
        metrics = {}
        for server in self._router.servers:
            try:
                async with self._session.get(f"http://{server}/metrics") as response:
                    metrics[server] = await response.json()
            except Exception:
                continue
        return metrics

    async def shutdown(self) -> None:
        await self._session.close()

    async def check_ready(self) -> Tuple[List[str], List[str]]:
        async def check_server_ready(server: str) -> bool:
            try:
                async with self._session.get(f"http://{server}/health") as response:
                    return response.status == 200
            except Exception:
                return False

        servers_ready = await asyncio.gather(
            *[check_server_ready(server) for server in self._router.servers]
        )
        return [server for server, ready in zip(self._router.servers, servers_ready) if ready], [
            server for server, ready in zip(self._router.servers, servers_ready) if not ready
        ]
