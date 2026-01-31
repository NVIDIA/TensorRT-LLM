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

import asyncio
import os
from typing import Any, Callable, Dict, Optional

from tensorrt_llm.llmapi.disagg_utils import (
    ConditionalDisaggConfig,
    DisaggClusterConfig,
    DisaggServerConfig,
    MetadataServerConfig,
    ServerRole,
    get_global_disagg_request_id,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.cluster_storage import ClusterStorage, WatchEventType
from tensorrt_llm.serve.disagg_auto_scaling import DisaggClusterManager, WorkerInfo
from tensorrt_llm.serve.metadata_server import JsonDictionary
from tensorrt_llm.serve.openai_client import OpenAIClient
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    DisaggregatedParams,
    DisaggScheduleStyle,
    UCompletionRequest,
    UCompletionResponse,
)
from tensorrt_llm.serve.openai_service import OpenAIService
from tensorrt_llm.serve.perf_metrics import DisaggPerfMetricsCollector
from tensorrt_llm.serve.responses_utils import (
    ResponseHooks,
    UCompletionResponseOrGenerator,
    done_generator,
)
from tensorrt_llm.serve.router import KvCacheAwareRouter, Router


class OpenAIDisaggregatedService(OpenAIService):
    def __init__(
        self,
        config: DisaggServerConfig,
        ctx_router: Router,
        gen_router: Router,
        client_factory: Callable[[Router, ServerRole], OpenAIClient],
        metadata_server: Optional[JsonDictionary] = None,
        metadata_config: Optional[MetadataServerConfig] = None,
        req_timeout_secs: int = 180,
        server_start_timeout_secs: int = 180,
        perf_metrics_collector: Optional[DisaggPerfMetricsCollector] = None,
        disagg_cluster_storage: Optional[ClusterStorage] = None,
        health_check_interval_secs: int = 3,
    ):
        self._config = config
        self._ctx_router = ctx_router
        self._gen_router = gen_router
        self._client_factory = client_factory
        self._metadata_server = metadata_server
        self._metadata_config = metadata_config
        self._req_timeout_secs = req_timeout_secs
        self._server_start_timeout_secs = server_start_timeout_secs
        self._perf_metrics_collector = perf_metrics_collector
        self._cluster_storage = disagg_cluster_storage
        self._health_check_interval_secs = health_check_interval_secs

        self._ctx_client = None
        self._gen_client = None
        self._disagg_cluster_manager = None
        self._schedule_style = DisaggScheduleStyle.CONTEXT_FIRST

        match self._config.schedule_style:
            case "generation_first":
                self._send_disagg_request = self._send_disagg_request_gen_first
                self._schedule_style = DisaggScheduleStyle.GENERATION_FIRST
            case _:
                self._send_disagg_request = self._send_disagg_request_ctx_first
                self._schedule_style = DisaggScheduleStyle.CONTEXT_FIRST

    async def openai_completion(
        self, request: UCompletionRequest, hooks: Optional[ResponseHooks] = None
    ) -> UCompletionResponseOrGenerator:
        if not await self.is_ready():
            raise RuntimeError("Cluster is not ready")
        if not isinstance(request.prompt, str):
            # Check if it's a list and contains integers
            if type(request.prompt) is list and len(request.prompt) == 1:
                request.prompt = request.prompt[0]
            elif not isinstance(request.prompt, list) or not all(
                isinstance(x, int) for x in request.prompt
            ):
                raise ValueError(
                    "Disaggregated server currently only supports single string prompt or list of integers in request"
                )

        return await self._send_disagg_request(request, hooks)

    async def openai_chat_completion(
        self, request: UCompletionRequest, hooks: Optional[ResponseHooks] = None
    ) -> UCompletionResponseOrGenerator:
        if not await self.is_ready():
            raise RuntimeError("Cluster is not ready")
        return await self._send_disagg_request(request, hooks)

    async def _send_disagg_request_ctx_first(
        self, request: UCompletionRequest, hooks: Optional[ResponseHooks] = None
    ) -> UCompletionResponseOrGenerator:
        if hooks:
            hooks.on_req_begin(request)
        # empty server means client decides which server to use
        ctx_server = None
        # reserve a gen_server if conditional disagg is needed
        gen_server, need_ctx = await self._check_conditional_disagg(request)
        need_ctx = need_ctx and not await self._check_gen_only_disagg(request)
        ctx_response = None
        gen_req = request
        disagg_request_id = get_global_disagg_request_id(self._config.node_id)
        if need_ctx:
            ctx_req = self._get_ctx_request(request, disagg_request_id)
            # ctx generator is empty
            ctx_server, _ = await self._ctx_router.get_next_server(
                ctx_req, exclude_server=gen_server
            )
            ctx_response = await self._ctx_client.send_request(
                ctx_req, server=ctx_server, hooks=hooks
            )
            await self._verify_ctx_response(ctx_response)
            gen_req = self._get_gen_request(request, ctx_response, disagg_request_id)
        if ctx_response is None or self._need_gen(ctx_response):
            if not gen_server:
                gen_server, _ = await self._gen_router.get_next_server(
                    gen_req, exclude_server=ctx_server
                )
            return await self._gen_client.send_request(gen_req, server=gen_server, hooks=hooks)
        else:
            if request.stream:
                # ctx client will never return a generator when streaming is requested
                # make up for this by returning a done generator
                return done_generator()
            return ctx_response

    def _need_gen(self, response: UCompletionResponse) -> bool:
        if response and response.choices[0].finish_reason not in ["length", "not_finished"]:
            del response.choices[0].disaggregated_params
            return False
        return True

    def _get_ctx_request(
        self, request: UCompletionRequest, disagg_request_id: Optional[int]
    ) -> UCompletionRequest:
        ctx_request = request.model_copy(
            update={
                "disaggregated_params": DisaggregatedParams(
                    request_type="context_only", disagg_request_id=disagg_request_id
                ),
                "stream": False,
                "stream_options": None,
                "schedule_style": self._schedule_style,
            }
        )
        return ctx_request

    def _get_gen_request(
        self,
        request: UCompletionRequest,
        ctx_response: Optional[UCompletionResponse],
        disagg_request_id: Optional[int],
        ctx_server_info: Optional[dict] = None,
    ) -> UCompletionRequest:
        if ctx_response:
            request.disaggregated_params = ctx_response.choices[0].disaggregated_params
            request.disaggregated_params.request_type = "generation_only"
            # Replace the string prompt with prompt_tokens_ids
            if isinstance(request, CompletionRequest):
                request.prompt = ctx_response.prompt_token_ids
            elif isinstance(request, ChatCompletionRequest):
                request.prompt_token_ids = ctx_response.prompt_token_ids
        else:
            # no ctx response, it's either a generation-only request or a generation-first disagg request
            request.disaggregated_params = DisaggregatedParams(
                request_type="generation_only",
                ctx_request_id=disagg_request_id,
                disagg_request_id=disagg_request_id,
                schedule_style=self._schedule_style,
            )
        if ctx_server_info and "server_info" in ctx_server_info:
            disaggregated_params = ctx_server_info["server_info"].get("disaggregated_params", {})
            if disaggregated_params:
                request.disaggregated_params = request.disaggregated_params.model_copy(
                    update=disaggregated_params
                )

        request.disaggregated_params.disagg_request_id = disagg_request_id
        return request

    async def _check_conditional_disagg(self, request: UCompletionRequest) -> bool:
        if self.conditional_disagg_config:
            assert isinstance(self._gen_router, KvCacheAwareRouter)
            # Query kv cache status and select a best gen_server.
            # The server is reserved for generation request
            gen_server, info = await self._gen_router.get_next_server(request)
            match_length = sum(info["matches"])
            total_length = sum(len(token_list) for token_list in info["token_lists"])
            if (
                match_length == 0
                or total_length - match_length
                > self.conditional_disagg_config.max_local_prefill_length
            ):
                return gen_server, True
            return gen_server, False
        return None, True

    async def _check_gen_only_disagg(self, request: UCompletionRequest) -> bool:
        if os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1":
            # Hard-code first token, ctx_request_id for testing
            request.disaggregated_params = DisaggregatedParams(
                request_type="generation_only",
                first_gen_tokens=[7],
                ctx_request_id=1,
                encoded_opaque_state=None,
                draft_tokens=None,
            )
            request.ignore_eos = True
            return True
        return False

    async def cluster_info(self) -> Dict[str, Any]:
        cluster_info = {"is_ready": await self.is_ready()}
        if self._disagg_cluster_manager:
            cluster_info.update(await self._disagg_cluster_manager.cluster_info())
        return cluster_info

    async def is_ready(self) -> bool:
        if self._disagg_cluster_manager:
            return await self._disagg_cluster_manager.is_ready()
        return True

    @property
    def disagg_cluster_config(self) -> Optional[DisaggClusterConfig]:
        return self._config.disagg_cluster_config

    @property
    def conditional_disagg_config(self) -> Optional[ConditionalDisaggConfig]:
        return self._config.conditional_disagg_config

    async def setup(self) -> None:
        self._ctx_client = self._client_factory(
            self._ctx_router, ServerRole.CONTEXT, self._config.max_retries
        )
        self._gen_client = self._client_factory(
            self._gen_router, ServerRole.GENERATION, self._config.max_retries
        )

        if self.disagg_cluster_config and self._cluster_storage:
            logger.info("Starting disagg cluster manager")
            self._disagg_cluster_manager = DisaggClusterManager(
                self.disagg_cluster_config, self._cluster_storage
            )
            await self._disagg_cluster_manager.start()
            await self._disagg_cluster_manager.watch_workers(on_event=self._on_worker_event)
            logger.info("Disagg cluster manager started")
        else:
            if self._metadata_server and self._metadata_config:
                logger.info("Starting server monitoring via metadata service")
                await self._ctx_router.start_server_monitoring(
                    self._metadata_config.refresh_interval
                )
                await self._gen_router.start_server_monitoring(
                    self._metadata_config.refresh_interval
                )
            await self._wait_for_all_servers_ready()

    async def teardown(self) -> None:
        await self._ctx_client.shutdown()
        await self._gen_client.shutdown()

        if self._disagg_cluster_manager:
            await self._disagg_cluster_manager.stop()

        if self._metadata_server:
            await self._ctx_router.stop_server_monitoring()
            await self._gen_router.stop_server_monitoring()

    async def _wait_for_all_servers_ready(self) -> None:
        # Skip context servers if TRTLLM_DISAGG_BENCHMARK_GEN_ONLY is set
        gen_only = os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1"

        async def check_servers_ready():
            elapsed_time = 0
            interval = self._health_check_interval_secs
            while elapsed_time < self._server_start_timeout_secs:
                if gen_only:
                    unready_ctx_servers = []
                else:
                    _, unready_ctx_servers = await self._ctx_client.check_ready()
                _, unready_gen_servers = await self._gen_client.check_ready()
                if len(unready_ctx_servers) == 0 and len(unready_gen_servers) == 0:
                    if gen_only:
                        logger.info("Generation servers are ready (context servers skipped)")
                    else:
                        logger.info("All servers are ready")
                    return
                logger.info(
                    f"Waiting for servers, context: {unready_ctx_servers}, generation: {unready_gen_servers}"
                )
                await asyncio.sleep(interval)
                elapsed_time += interval

        try:
            await asyncio.wait_for(check_servers_ready(), timeout=self._server_start_timeout_secs)
        except asyncio.TimeoutError:
            raise TimeoutError("Timeout waiting for context and generation servers to be ready")

    async def _on_worker_event(self, worker_info: WorkerInfo, event_type: WatchEventType):
        router_map = {ServerRole.CONTEXT: self._ctx_router, ServerRole.GENERATION: self._gen_router}
        worker_addr = f"{worker_info.host}:{worker_info.port}"
        try:
            router = router_map[worker_info.role]
            if event_type == WatchEventType.SET:
                await router.add_server(worker_addr)
            elif event_type == WatchEventType.DELETE:
                await router.remove_server(worker_addr)
            logger.info(f"Worker {event_type.name} event: {worker_info.worker_id}, {worker_addr}")
        except KeyError:
            logger.error(
                f"Unknown worker role: {worker_info.role}, Worker {worker_info.worker_id} event: {event_type.name}"
            )

    async def _verify_ctx_response(self, ctx_response: UCompletionResponse) -> None:
        if ctx_response:
            if len(ctx_response.choices) != 1:
                raise ValueError(
                    f"Context server returned {len(ctx_response.choices)} choices, expecting 1."
                )
            if ctx_response.choices[0].disaggregated_params is None:
                raise ValueError("Context server did not return disaggregated params")
            if ctx_response.choices[0].disaggregated_params.ctx_request_id is None:
                raise ValueError("Invalid disaggregated params in context phase response.")
            if ctx_response.choices[0].disaggregated_params.disagg_request_id is None:
                raise ValueError(
                    "Invalid disaggregated params in context phase response. disagg_request_id is None"
                )
            return ctx_response

    async def _send_disagg_request_gen_first(
        self, request: UCompletionRequest, hooks: Optional[ResponseHooks] = None
    ) -> UCompletionResponse:
        if hooks:
            hooks.on_req_begin(request)
        # empty server means client decides which server to use
        need_ctx = not (await self._check_gen_only_disagg(request))
        ctx_server, gen_server = None, None
        ctx_server_info = None
        tasks = []
        ctx_req, gen_req = None, None
        disagg_request_id = get_global_disagg_request_id(self._config.node_id)
        if need_ctx:
            ctx_server, ctx_server_info = await self._ctx_router.get_next_server(request)
            ctx_req = self._get_ctx_request(request, disagg_request_id)
            tasks.append(
                asyncio.create_task(
                    self._ctx_client.send_request(ctx_req, server=ctx_server, hooks=hooks)
                )
            )
        gen_req = self._get_gen_request(
            request,
            ctx_response=None,
            disagg_request_id=disagg_request_id,
            ctx_server_info=ctx_server_info,
        )
        tasks.append(
            asyncio.create_task(
                self._gen_client.send_request(gen_req, server=gen_server, hooks=hooks)
            )
        )
        responses = await asyncio.gather(*tasks)
        return responses[-1]
