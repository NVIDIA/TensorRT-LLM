#!/usr/bin/env python
import asyncio
import os
import re
import signal
import traceback
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import (Annotated, Any, AsyncGenerator, AsyncIterator, List,
                    Optional, Union)

import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount
from transformers import AutoConfig, AutoProcessor

from tensorrt_llm._tensorrt_engine import LLM
# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.inputs import prompt_inputs
from tensorrt_llm.inputs.data import TokensPrompt
from tensorrt_llm.inputs.utils import ConversationMessage, apply_chat_template
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi import MultimodalEncoder
from tensorrt_llm.llmapi.disagg_utils import (DisaggClusterConfig,
                                              MetadataServerConfig, ServerRole)
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.metrics.collector import MetricsCollector
from tensorrt_llm.serve.chat_utils import (check_multiple_response,
                                           parse_chat_messages_coroutines)
from tensorrt_llm.serve.cluster_storage import create_cluster_storage_client
from tensorrt_llm.serve.disagg_auto_scaling import DisaggClusterWorker
from tensorrt_llm.serve.metadata_server import create_metadata_server
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                ChatCompletionResponse,
                                                ChatCompletionResponseChoice,
                                                ChatMessage, CompletionRequest,
                                                CompletionResponse,
                                                CompletionResponseChoice,
                                                ErrorResponse, ModelCard,
                                                ModelList, PromptTokensDetails,
                                                ResponsesRequest, UsageInfo,
                                                to_llm_disaggregated_params)
from tensorrt_llm.serve.postprocess_handlers import (
    ChatCompletionPostprocArgs, ChatPostprocArgs, CompletionPostprocArgs,
    chat_harmony_post_processor, chat_harmony_streaming_post_processor,
    chat_response_post_processor, chat_stream_post_processor,
    completion_response_post_processor, completion_stream_post_processor)
from tensorrt_llm.serve.responses_utils import (ConversationHistoryStore,
                                                ServerArrivalTimeMiddleware)
from tensorrt_llm.serve.responses_utils import \
    create_response as responses_api_create_response
from tensorrt_llm.serve.responses_utils import get_steady_clock_now_in_seconds
from tensorrt_llm.serve.responses_utils import \
    process_streaming_events as responses_api_process_streaming_events
from tensorrt_llm.serve.responses_utils import \
    request_preprocess as responses_api_request_preprocess
from tensorrt_llm.version import __version__ as VERSION

from .._utils import nvtx_mark, set_prometheus_multiproc_dir
from .harmony_adapter import (HarmonyAdapter, get_harmony_adapter,
                              maybe_transform_reasoning_effort)

# yapf: enale
TIMEOUT_KEEP_ALIVE = 5  # seconds.


class OpenAIServer:

    def __init__(self,
                 llm: Union[LLM, MultimodalEncoder],
                 model: str,
                 server_role: Optional[ServerRole],
                 metadata_server_cfg: MetadataServerConfig,
                 disagg_cluster_config: Optional[DisaggClusterConfig] = None):
        self.llm = llm
        self.tokenizer = llm.tokenizer
        self.metadata_server = create_metadata_server(metadata_server_cfg)
        self.disagg_cluster_config = disagg_cluster_config
        self.server_role = server_role
        # Will be set in __call__
        self.binding_addr = None
        self.host = None
        self.port = None
        hf_tokenizer_path = llm._hf_model_dir or self.tokenizer.tokenizer.name_or_path
        trust_remote_code = llm.args.trust_remote_code
        try:
            self.processor = AutoProcessor.from_pretrained(hf_tokenizer_path, trust_remote_code=trust_remote_code)
        except Exception:
            logger.debug("Failed to load AutoProcessor or AutoConfig for %s", hf_tokenizer_path)
            self.processor = None
        # Temporary workaround for DSv3.2 config.
        import transformers

        from tensorrt_llm._torch.model_config import _CONFIG_REGISTRY
        config_dict, _ = transformers.PretrainedConfig.get_config_dict(
                hf_tokenizer_path,
                trust_remote_code=trust_remote_code
            )
        model_type = config_dict.get("model_type")
        if model_type in _CONFIG_REGISTRY:
            config_class = _CONFIG_REGISTRY[model_type]
            self.model_config = config_class.from_pretrained(
                hf_tokenizer_path,
                trust_remote_code=trust_remote_code
            )
        else:
            try:
                self.model_config = AutoConfig.from_pretrained(hf_tokenizer_path, trust_remote_code=trust_remote_code)
            except Exception:
                logger.debug("Failed to load AutoConfig for %s", hf_tokenizer_path)
                self.model_config = None

        # Enable response storage for Responses API
        self.enable_store = True
        if len(os.getenv("TRTLLM_RESPONSES_API_DISABLE_STORE", "")) > 0:
            self.enable_store = False
        self.conversation_store = ConversationHistoryStore()

        model_dir = Path(model)
        if model_dir.exists() and model_dir.is_dir():
            self.model = model_dir.name
        else:
            self.model = model
        self.metrics_collector = None
        self.perf_metrics = None
        self.perf_metrics_lock = None
        # The steady clock offset (in seconds) between this server and the disagg server
        self.disagg_server_steady_clock_offset = 0
        if self.llm.args.return_perf_metrics:
            set_prometheus_multiproc_dir()
            self.metrics_collector = MetricsCollector({
                "model_name": "undefined",
                "engine_type": "undefined"
            })
            max_perf_metrics = self.llm.args.perf_metrics_max_requests
            if max_perf_metrics > 0:
                self.perf_metrics = deque(maxlen=max_perf_metrics)
                self.perf_metrics_lock = asyncio.Lock()

        # gpt-oss
        self.harmony_adapter: HarmonyAdapter | None = None
        disable_harmony = os.getenv("DISABLE_HARMONY_ADAPTER", "0") == "1"
        if disable_harmony:
            self.use_harmony = False
        else:
            self.use_harmony = (self.model_config.model_type == "gpt_oss")

        # as disagg-worker
        self.disagg_cluster_storage = None
        self.disagg_cluster_worker = None

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            if self.metadata_server is not None:
                metadata = {
                    "model": self.model,
                    "version": VERSION,
                    "timestamp": datetime.now().isoformat(),
                    "server_role": server_role.name,
                    "url": self.binding_addr
                }
                # TODO: add more metadata
                # Register with ETCD using the existing key format
                self.metadata_server.put(f"trtllm/{self.llm.llm_id}", metadata)
                logger.info(f"trtllm/{self.llm.llm_id} is registered")

            if self.disagg_cluster_config:
                self.disagg_cluster_storage = create_cluster_storage_client(self.disagg_cluster_config.cluster_uri, self.disagg_cluster_config.cluster_name)
                self.disagg_cluster_worker= DisaggClusterWorker(self.server_role, self.host, self.port, self.disagg_cluster_config, self.disagg_cluster_storage)
                await self.disagg_cluster_worker.register_worker()

            # terminate rank0 worker
            yield

            if self.metadata_server is not None:
                self.metadata_server.remove(f"trtllm/{self.llm.llm_id}")
                logger.info(f"trtllm/{self.llm.llm_id} is unregistered")
            if self.disagg_cluster_worker:
                await self.disagg_cluster_worker.deregister_worker()
            self.llm.shutdown()

        self.app = FastAPI(lifespan=lifespan)

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(_, exc):
            return self.create_error_response(message=str(exc))

        if self.server_role is not ServerRole.MM_ENCODER:
            self.register_routes()
        else:
            assert isinstance(self.llm, MultimodalEncoder), "llm must be a MultimodalEncoder for multimodal encoder"
            self.register_mm_encoder_routes()

        self.app.add_middleware(ServerArrivalTimeMiddleware)


    async def await_disconnected(self, raw_request: Request, promise):
        if raw_request is None:
            return
        while not await raw_request.is_disconnected():
            await asyncio.sleep(1)
        if not promise.finished:
            promise.abort()
            logger.info(
                f"{raw_request.client} is disconnected, abort {promise.request_id}")

    @property
    def postproc_worker_enabled(self) -> bool:
        return True if self.llm.args.num_postprocess_workers > 0 else False

    @staticmethod
    def create_error_response(
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> Response:
        error_response = ErrorResponse(message=message,
                                       type=err_type,
                                       code=status_code.value)
        return JSONResponse(content=error_response.model_dump(),
                            status_code=error_response.code)

    def _create_invalid_response_id_error(self, response_id: str) -> Response:
        return self.create_error_response(
            err_type="InvalidRequestError",
            message=(f"Invalid 'response_id': '{response_id}'. "
                     "Expected an ID that begins with 'resp'."),
        )

    def _create_response_id_not_found_error(self, response_id: str) -> Response:
        return self.create_error_response(
            err_type="InvalidRequestError",
            message=f"Response with id '{response_id}' not found.",
            status_code=HTTPStatus.NOT_FOUND,
        )

    def register_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/health_generate", self.health_generate, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        # TODO: the metrics endpoint only reports iteration stats, not the runtime stats for now
        self.app.add_api_route("/metrics", self.get_iteration_stats, methods=["GET"])
        self.app.add_api_route("/perf_metrics", self.get_perf_metrics, methods=["GET"])
        self.app.add_api_route("/steady_clock_offset", self.get_steady_clock_offset, methods=["GET"])
        # Called by the disagg server to set the disagg_server_steady_clock_offset
        self.app.add_api_route("/steady_clock_offset", self.set_steady_clock_offset, methods=["POST"])
        # TODO: workaround before ETCD support
        self.app.add_api_route("/kv_cache_events", self.get_kv_cache_events, methods=["POST"])
        self.app.add_api_route("/v1/completions",
                               self.openai_completion,
                               methods=["POST"])
        self.app.add_api_route("/v1/chat/completions",
                               self.openai_chat if not self.use_harmony else self.chat_harmony,
                               methods=["POST"])
        self.app.add_api_route("/v1/responses",
                               self.openai_responses,
                               methods=["POST"])
        if self.llm.args.return_perf_metrics:
            # register /prometheus/metrics
            self.mount_metrics()

    def mount_metrics(self):
        # Lazy import for prometheus multiprocessing.
        # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
        # before prometheus_client is imported.
        # See https://prometheus.github.io/client_python/multiprocess/
        from prometheus_client import (CollectorRegistry, make_asgi_app,
                                       multiprocess)
        from prometheus_fastapi_instrumentator import Instrumentator
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        Instrumentator(
            should_group_status_codes=False,
            should_respect_env_var=True,
            excluded_handlers=[
                ".*"
            ],
            registry=registry,
        ).add().instrument(self.app).expose(self.app)
        metrics_app = make_asgi_app(registry=registry)
        metrics_route = Mount("/prometheus/metrics", metrics_app)
        metrics_route.path_regex = re.compile("^/prometheus/metrics(?P<path>.*)$")
        self.app.routes.append(metrics_route)

    def register_mm_encoder_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        # TODO: the metrics endpoint only reports iteration stats, not the runtime stats for now
        self.app.add_api_route("/metrics", self.get_iteration_stats, methods=["GET"])
        self.app.add_api_route("/v1/chat/completions",
                               self.openai_mm_encoder,
                               methods=["POST"])

    async def health(self) -> Response:
        return Response(status_code=200)

    async def health_generate(self, raw_request: Request) -> Response:
        """Health check that performs a minimal generation."""
        try:
            # Create a minimal chat request
            health_request = ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}], # Minimal prompt (often > 1 token after tokenization)
                model=self.model,
                max_completion_tokens=1, # Request only 1 token out
                stream=False,
                temperature=0.0 # Deterministic output
            )

            # Call the chat completion logic
            response = await self.openai_chat(health_request, raw_request)

            # Check if the response indicates success (status code 200)
            if response.status_code == 200:
                return Response(status_code=200, content="Generation health check OK")
            else:
                logger.error(f"Health generate check failed with status code: {response.status_code}")
                try:
                    # Attempt to get body for more details if possible
                    body = response.body if hasattr(response, 'body') else await response.body()
                    logger.error(f"Health generate check response body: {body}")
                except Exception:
                    pass # Ignore errors trying to get body details
                return Response(status_code=500, content="Generation health check failed")

        except Exception as e:
            logger.error(f"Health generate check encountered exception: {e}")
            return Response(status_code=500, content=f"Generation health check failed: {str(e)}")

    async def version(self) -> JSONResponse:
        ver = {"version": VERSION}
        return JSONResponse(content=ver)

    async def get_model(self) -> JSONResponse:
        model_list = ModelList(data=[ModelCard(id=self.model)])
        return JSONResponse(content=model_list.model_dump())

    async def get_iteration_stats(self) -> JSONResponse:
        stats = []
        async for stat in self.llm.get_stats_async(2):
            stats.append(stat)
        return JSONResponse(content=stats)

    async def set_steady_clock_offset(self, offset: Annotated[float, Body(embed=True)]) -> Response:
        self.disagg_server_steady_clock_offset = offset
        logger.info(f"The steady clock offset between local and disagg server: {offset} second")
        return Response(status_code=200)

    async def get_steady_clock_offset(self) -> JSONResponse:
        receive_ts = get_steady_clock_now_in_seconds()
        await asyncio.sleep(0.2)
        transmit_ts = get_steady_clock_now_in_seconds()
        return JSONResponse(content={"receive_ts": receive_ts, "transmit_ts": transmit_ts})

    async def get_perf_metrics(self) -> JSONResponse:
        if self.perf_metrics is None:
            return JSONResponse(content=[])
        async with self.perf_metrics_lock:
            perf_metrics = self.perf_metrics
            self.perf_metrics = deque(maxlen=self.llm.args.perf_metrics_max_requests)
        for metrics_dict in perf_metrics:
            metrics = metrics_dict["perf_metrics"]
            timing_metrics = metrics.timing_metrics
            kv_cache_metrics = metrics.kv_cache_metrics
            speculative_decoding = metrics.speculative_decoding
            metrics_json = {
                "first_iter": metrics.first_iter,
                "last_iter": metrics.last_iter,
                # exclude metrics.iter since it is only meaningful when the request is not finished
            }
            server_arrival_time = metrics_dict.pop("server_arrival_time", None)
            if server_arrival_time is not None:
                server_arrival_time += self.disagg_server_steady_clock_offset
            server_first_token_time = metrics_dict.pop("server_first_token_time", None)
            if server_first_token_time is not None:
                server_first_token_time += self.disagg_server_steady_clock_offset
            metrics_json["timing_metrics"] = {
                "server_arrival_time": server_arrival_time,
                "arrival_time": timing_metrics.arrival_time.total_seconds() + self.disagg_server_steady_clock_offset,
                "first_scheduled_time": timing_metrics.first_scheduled_time.total_seconds() + self.disagg_server_steady_clock_offset,
                "first_token_time": timing_metrics.first_token_time.total_seconds() + self.disagg_server_steady_clock_offset,
                "server_first_token_time": server_first_token_time,
                "last_token_time": timing_metrics.last_token_time.total_seconds() + self.disagg_server_steady_clock_offset,
            }
            metrics_json["kv_cache_metrics"] = {
                "num_total_allocated_blocks": kv_cache_metrics.num_total_allocated_blocks,
                "num_new_allocated_blocks": kv_cache_metrics.num_new_allocated_blocks,
                "num_reused_blocks": kv_cache_metrics.num_reused_blocks,
                "num_missed_blocks": kv_cache_metrics.num_missed_blocks,
            }
            if timing_metrics.kv_cache_size > 0:
                metrics_json["timing_metrics"].update({
                    # TODO: move to kv_cache_metrics
                    "kv_cache_size": timing_metrics.kv_cache_size,
                    "kv_cache_transfer_start": timing_metrics.kv_cache_transfer_start.total_seconds() + self.disagg_server_steady_clock_offset,
                    "kv_cache_transfer_end": timing_metrics.kv_cache_transfer_end.total_seconds() + self.disagg_server_steady_clock_offset,
                })
            if speculative_decoding.total_draft_tokens > 0:
                metrics_json["speculative_decoding"] = {
                    "acceptance_rate": speculative_decoding.acceptance_rate,
                    "total_accepted_draft_tokens": speculative_decoding.total_accepted_draft_tokens,
                    "total_draft_tokens": speculative_decoding.total_draft_tokens,
                }
            metrics_dict["perf_metrics"] = metrics_json
        return JSONResponse(content=list(perf_metrics))

    async def get_kv_cache_events(self) -> JSONResponse:
        events = []
        try:
            async for event in self.llm.get_kv_cache_events_async(2):
                events.append(event)
        except IndexError:
            # queue is empty, no more events
            pass
        return JSONResponse(content=events)

    async def _extract_metrics(self, res: RequestOutput, raw_request: Request):
        if not res.finished:
            return
        if self.metrics_collector:
            self.metrics_collector.log_metrics_dict(res.metrics_dict)
        if self.llm.args.return_perf_metrics:
            output = res.outputs[0]
            item = {
                "request_id": res.request_id,
                "perf_metrics": res.outputs[0].request_perf_metrics
            }
            if raw_request:
                item["server_arrival_time"] = getattr(raw_request.state, "server_arrival_time", None)
                item["server_first_token_time"] = getattr(raw_request.state, "server_first_token_time", None)
            if output.disaggregated_params:
                item["ctx_request_id"] = output.disaggregated_params.ctx_request_id
            if self.perf_metrics is not None:
                async with self.perf_metrics_lock:
                    self.perf_metrics.append(item)

    async def openai_chat(self, request: ChatCompletionRequest, raw_request: Request) -> Response:

        def get_role() -> str:
            if request.add_generation_prompt:
                role = "assistant"
            else:
                role = request.messages[-1]["role"]
            return role

        async def chat_stream_generator(
                promise: RequestOutput, postproc_params: PostprocParams) -> AsyncGenerator[str, None]:
            try:
                if not self.postproc_worker_enabled:
                    post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                first_response = await anext(promise)
                raw_request.state.server_first_token_time = get_steady_clock_now_in_seconds()
                pp_results = first_response.outputs[0]._postprocess_result if self.postproc_worker_enabled else post_processor(first_response, args)
                for pp_res in pp_results:
                    yield pp_res
                # Making sure we can handling the situation where there is only one response
                res = first_response
                async for res in promise:
                    pp_results = res.outputs[0]._postprocess_result if self.postproc_worker_enabled else post_processor(res, args)
                    for pp_res in pp_results:
                        yield pp_res
                yield "data: [DONE]\n\n"
                await self._extract_metrics(res, raw_request)
                nvtx_mark("generation ends")
            except:
                logger.error(traceback.format_exc())
                raise

        async def create_chat_response(
                promise: RequestOutput, postproc_params: PostprocParams, disaggregated_params: Optional[LlmDisaggregatedParams] = None) -> ChatCompletionResponse:
            await promise.aresult()
            if self.postproc_worker_enabled:
                chat_response =promise.outputs[0]._postprocess_result
            else:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                chat_response = post_processor(promise, args)

            # Add prompt_tokens_ids to the response
            if disaggregated_params and disaggregated_params.request_type and disaggregated_params.request_type == "context_only":
                chat_response.prompt_token_ids = promise.prompt_token_ids
            raw_request.state.server_first_token_time = get_steady_clock_now_in_seconds()
            await self._extract_metrics(promise, raw_request)
            return chat_response

        try:
            check_multiple_response(request.n, self.llm.args.backend)
            conversation: List[ConversationMessage] = []
            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]
            # Pass the tokenizer vocabulary size so ``logit_bias`` can be
            # expanded into an embedding bias tensor in the sampler.
            sampling_params = request.to_sampling_params(
                vocab_size=self.tokenizer.tokenizer.vocab_size,
                gather_generation_logits=self.llm.args.gather_generation_logits,
                backend=self.llm.args.backend)
            postproc_args = ChatPostprocArgs.from_request(request)
            disaggregated_params = to_llm_disaggregated_params(request.disaggregated_params)

            conversation, mm_coroutines, mm_placeholder_counts = parse_chat_messages_coroutines(request.messages, self.model_config)

            if request.prompt_token_ids is not None:
                prompt = request.prompt_token_ids
            else:
                prompt: str = apply_chat_template(
                    model_type=self.model_config.model_type,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    conversation=conversation,
                    add_generation_prompt=request.add_generation_prompt,
                    mm_placeholder_counts=mm_placeholder_counts,
                    tools=tool_dicts,
                    documents=request.documents,
                    chat_template=request.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs or {},
                )
            prompt = prompt_inputs(prompt)

            mm_data = await mm_coroutines
            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            postproc_args.reasoning_parser = self.llm.args.reasoning_parser
            if conversation and conversation[-1].get(
                    "content") and conversation[-1].get("role") == get_role():
                postproc_args.last_message_content = conversation[-1]["content"]
            postproc_params = PostprocParams(
                post_processor=chat_stream_post_processor
                if request.stream else chat_response_post_processor,
                postproc_args=postproc_args,
            )

            promise = self.llm.generate_async(
                inputs=prompt,
                sampling_params=sampling_params,
                _postproc_params=postproc_params if self.postproc_worker_enabled else None,
                streaming=request.stream,
                lora_request=request.lora_request,
                disaggregated_params=disaggregated_params,
                cache_salt=request.cache_salt,
            )
            asyncio.create_task(self.await_disconnected(raw_request, promise))
            if not self.postproc_worker_enabled:
                postproc_args.tokenizer = self.tokenizer
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

            if request.stream:
                response_generator = chat_stream_generator(promise, postproc_params)
                return StreamingResponse(content=response_generator,
                                         media_type="text/event-stream")
            else:
                response = await create_chat_response(promise, postproc_params, disaggregated_params)
                return JSONResponse(content=response.model_dump())
        except CppExecutorError:
            logger.error(traceback.format_exc())
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(str(e))

    async def openai_mm_encoder(self, request: ChatCompletionRequest, raw_request: Request) -> Response:

        async def create_mm_embedding_response(promise: RequestOutput):
            await promise.aresult()
            # TODO: Replace mm_embedding_handle with a dedicated OpenAIBaseModel(JSON-safe), when enable multimodal disagg E2E
            mm_embedding_handle = getattr(promise, "mm_embedding_handle", None)
            if not mm_embedding_handle or "tensor_size" not in mm_embedding_handle:
                return self.create_error_response(
                    message="Multimodal embedding handle missing in response",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
            num_tokens = int(mm_embedding_handle["tensor_size"][0])
            return ChatCompletionResponse(
                id=str(promise.request_id),
                model=self.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="dummy"),
                        mm_embedding_handle=mm_embedding_handle,
                        finish_reason="length",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=num_tokens,
                    completion_tokens=1,
                    total_tokens=num_tokens + 1,
                ),
            )

        try:
            check_multiple_response(request.n, self.llm.args.backend)
            conversation: List[ConversationMessage] = []
            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]

            conversation, mm_coroutines, mm_placeholder_counts = parse_chat_messages_coroutines(request.messages, self.model_config)

            if request.prompt_token_ids is not None:
                prompt = request.prompt_token_ids
            else:
                prompt: str = apply_chat_template(
                    model_type=self.model_config.model_type,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    conversation=conversation,
                    add_generation_prompt=request.add_generation_prompt,
                    mm_placeholder_counts=mm_placeholder_counts,
                    tools=tool_dicts,
                    documents=request.documents,
                    chat_template=request.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs or {},
                )
            prompt = prompt_inputs(prompt)

            mm_data = await mm_coroutines
            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            promise = self.llm.generate_async(
                inputs=prompt,
            )
            asyncio.create_task(self.await_disconnected(raw_request, promise))

            response = await create_mm_embedding_response(promise)
            return JSONResponse(content=response.model_dump())

        except CppExecutorError:
            logger.error(traceback.format_exc())
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(str(e))

    async def openai_completion(self, request: CompletionRequest, raw_request: Request) -> Response:

        async def completion_response(promise: RequestOutput,
                                      postproc_params: Optional[PostprocParams]) -> CompletionResponse:
            response = await promise
            if not self.postproc_worker_enabled:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                pp_result = post_processor(response, args)
            else:
                pp_result = response.outputs[0]._postprocess_result
            if disaggregated_params and disaggregated_params.request_type and disaggregated_params.request_type == "context_only":
                # Include prompt token ids for context-only requests
                pp_result.prompt_token_ids = response.prompt_token_ids
            raw_request.state.server_first_token_time = get_steady_clock_now_in_seconds()
            await self._extract_metrics(response, raw_request)
            return pp_result

        def merge_completion_responses(responses: List[CompletionResponse]) -> CompletionResponse:
            all_choices: List[CompletionResponseChoice] = []
            all_prompt_token_ids: List[List[int]] = []
            num_prompt_tokens = num_gen_tokens = num_cached_tokens = 0
            for rsp in responses:
                choices, usage = rsp.choices, rsp.usage
                all_choices.extend(choices)
                num_prompt_tokens += usage.prompt_tokens
                num_gen_tokens += usage.completion_tokens
                num_cached_tokens += usage.prompt_tokens_details.cached_tokens
                # Aggregate prompt token ids for context-only requests
                if rsp.prompt_token_ids is not None:
                    all_prompt_token_ids.append(rsp.prompt_token_ids)

            usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_gen_tokens,
                total_tokens=num_gen_tokens + num_prompt_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=num_cached_tokens,
                ),
            )
            merged_rsp = CompletionResponse(
                model=self.model,
                choices=all_choices,
                usage=usage_info,
                prompt_token_ids=all_prompt_token_ids,
            )
            return merged_rsp

        async def completion_generator(promise: RequestOutput, params: Optional[PostprocParams]):
            try:
                async for output in promise:
                    if not self.postproc_worker_enabled:
                        post_processor, args = params.post_processor, params.postproc_args
                        pp_result = post_processor(output, args)
                    else:
                        pp_result = output.outputs[0]._postprocess_result
                    for pp_res in pp_result:
                        yield pp_res
                await self._extract_metrics(output, raw_request)
            except:
                logger.error(traceback.format_exc())
                raise


        async def merge_generators(generators: List[AsyncIterator[Any]]):
            result_queue = asyncio.Queue()
            finished = [False] * len(generators)

            async def producer(generator: AsyncIterator[Any], idx: int):
                async for output in generator:
                    await result_queue.put(output)
                finished[idx] = True

            tasks = [
                asyncio.create_task(producer(generator, idx)) for idx, generator in enumerate(generators)
            ]

            while not all(finished) or not result_queue.empty():
                output = await result_queue.get()
                yield output
            await asyncio.gather(*tasks)

        async def generator_wrapper(generator: AsyncIterator[Any]):
            first_response = await anext(generator)
            raw_request.state.server_first_token_time = get_steady_clock_now_in_seconds()
            yield first_response
            async for output in generator:
                yield output
            yield "data: [DONE]\n\n"

        try:
            check_multiple_response(request.n, self.llm.args.backend)
            if isinstance(request.prompt, str) or \
                (isinstance(request.prompt, list) and isinstance(request.prompt[0], int)):
                prompts = [request.prompt]
            else:
                prompts = request.prompt

            promises: List[RequestOutput] = []
            postproc_params_collection: List[Optional[PostprocParams]] = []
            # Pass the tokenizer vocabulary size so ``logit_bias`` can be
            # expanded into an embedding bias tensor in the sampler.
            sampling_params = request.to_sampling_params(
                vocab_size=self.tokenizer.tokenizer.vocab_size)
            # TODO: better way to enable metrics
            if len(os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", "")) > 0:
                sampling_params.return_perf_metrics = True
            disaggregated_params = to_llm_disaggregated_params(request.disaggregated_params)
            for idx, prompt in enumerate(prompts):
                postproc_args = CompletionPostprocArgs.from_request(request)
                postproc_args.prompt_idx = idx
                if request.echo:
                    postproc_args.prompt = prompt
                postproc_params = PostprocParams(
                    post_processor=completion_stream_post_processor
                    if request.stream else completion_response_post_processor,
                    postproc_args=postproc_args,
                )

                prompt = prompt_inputs(prompt)
                if prompt.get("prompt") is not None:
                    prompt_token_ids, extra_processed_inputs = await asyncio.to_thread(self.llm.input_processor, prompt, sampling_params)
                    tokens_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, query_token_ids=extra_processed_inputs.get("query_token_ids") if extra_processed_inputs is not None else None)
                else:
                    tokens_prompt = prompt

                promise = self.llm.generate_async(
                    inputs=tokens_prompt,
                    sampling_params=sampling_params,
                    _postproc_params=postproc_params,
                    streaming=request.stream,
                    lora_request=request.lora_request,
                    disaggregated_params=disaggregated_params
                )
                asyncio.create_task(self.await_disconnected(raw_request, promise))
                if not self.postproc_worker_enabled:
                    postproc_args.tokenizer = self.tokenizer
                    postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)
                promises.append(promise)
                postproc_params_collection.append(None if self.postproc_worker_enabled else postproc_params)

            if request.stream:
                generators = [completion_generator(promise, params)
                              for promise, params in zip(promises, postproc_params_collection)]
                response_generator = merge_generators(generators) if len(promises) > 1 else generators[0]
                return StreamingResponse(content=generator_wrapper(response_generator),
                                            media_type="text/event-stream")
            else:
                rsps = await asyncio.gather(*[completion_response(promise, params)
                                              for promise, params in zip(promises, postproc_params_collection)])
                response = merge_completion_responses(rsps) if len(rsps) > 1 else rsps[0]
                return JSONResponse(content=response.model_dump())
        except CppExecutorError:
            logger.error(traceback.format_exc())
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(str(e))

    async def chat_harmony(self, request: ChatCompletionRequest, raw_request: Request) -> Response:
        """
        Chat Completion API with harmony format support.
        Supports both streaming and non-streaming modes.
        """

        async def create_harmony_response(
                promise: RequestOutput, postproc_params: PostprocParams) -> ChatCompletionResponse:
            await promise.aresult()
            if self.postproc_worker_enabled:
                chat_response =promise.outputs[0]._postprocess_result
            else:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                chat_response = post_processor(promise, args)

            return chat_response

        async def create_streaming_generator(promise: RequestOutput, postproc_params: PostprocParams):
            if not self.postproc_worker_enabled:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args

            async for res in promise:
                pp_results = res.outputs[0]._postprocess_result if self.postproc_worker_enabled else post_processor(res, args)
                for pp_res in pp_results:
                    yield pp_res

            yield "data: [DONE]\n\n"

        try:
            # Initialize HarmonyAdapter
            # NOTE: WAR for Disagg failure, may affect perf if no warmup
            if not self.harmony_adapter:
                self.harmony_adapter = get_harmony_adapter()
            # Convert Pydantic models to dictionaries for JSON serialization (standard pattern)
            tools_dict = None
            if request.tools:
                tools_dict = [tool.model_dump() for tool in request.tools]

            # Reasoning effort precedence: request.reasoning_effort > system message parsing > serving default
            reasoning_effort = maybe_transform_reasoning_effort(request.reasoning_effort)
            # Get tool_choice from request
            tool_choice = getattr(request, 'tool_choice', None)

            try:
                harmony_tokens = self.harmony_adapter.openai_to_harmony_tokens(
                    request.messages,
                    tools_dict,
                    reasoning_effort=reasoning_effort,
                    tool_choice=tool_choice
                )
            except Exception as e:
                logger.error(f"messages_dict: {request.messages}")
                logger.error(f"tools_dict: {tools_dict}")
                logger.error(f"request: {request}")
                raise e

            # Get harmony stop tokens
            harmony_stop_tokens = self.harmony_adapter.get_stop_tokens()
            if request.stop_token_ids:
                request.stop_token_ids.extend(harmony_stop_tokens)
            else:
                request.stop_token_ids = harmony_stop_tokens

            sampling_params = request.to_sampling_params(
                vocab_size=self.tokenizer.tokenizer.vocab_size)
            sampling_params.detokenize = False  # Harmony adapter handles detokenization

            postproc_args = ChatCompletionPostprocArgs.from_request(request)
            postproc_params = PostprocParams(
                post_processor=chat_harmony_streaming_post_processor
                if request.stream else chat_harmony_post_processor,
                postproc_args=postproc_args,
            )

            # Generate
            promise = self.llm.generate_async(
                inputs=harmony_tokens,
                sampling_params=sampling_params,
                _postproc_params=postproc_params if self.postproc_worker_enabled else None,
                streaming=bool(request.stream),
                lora_request=request.lora_request,
            )
            postproc_args.request_id = promise.request_id

            if not self.postproc_worker_enabled:
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

            # Disconnect cancellation
            asyncio.create_task(self.await_disconnected(raw_request, promise))

            # Handle streaming
            if request.stream:
                return StreamingResponse(
                    content=create_streaming_generator(promise, postproc_params),
                    media_type="text/event-stream"
                )
            else:
                response = await create_harmony_response(promise, postproc_params)
                return JSONResponse(response.model_dump())

        except Exception as e:
            logger.error("Error in harmony chat completion: %s", e)
            logger.debug("Error details: %s", traceback.format_exc())
            return self.create_error_response(message=str(e), err_type="internal_error")

    async def openai_responses(self, request: ResponsesRequest, raw_request: Request) -> Response:
        async def create_stream_response(generator, request: ResponsesRequest, sampling_params) -> AsyncGenerator[str, None]:
            async for event_data in responses_api_process_streaming_events(
                request=request,
                sampling_params=sampling_params,
                generator=generator,
                harmony_adapter=self.harmony_adapter,
                model_name=self.model,
                conversation_store=self.conversation_store,
                enable_store=self.enable_store
            ):
                yield event_data

        try:
            if not self.use_harmony:
                raise NotImplementedError("Responses API only supports harmony format for now")

            # Initialize HarmonyAdapter
            # NOTE: WAR for Disagg failure, may affect perf if no warmup
            if not self.harmony_adapter:
                self.harmony_adapter = HarmonyAdapter()

            if request.background:
                logger.warning("Request.background is not supported yet, will fallback to foreground processing.")

            # Get prev response
            prev_response = None
            if self.enable_store:
                prev_response_id = request.previous_response_id
                if prev_response_id is not None:
                    if not prev_response_id.startswith("resp_"):
                        return self._create_invalid_response_id_error(prev_response_id)

                    prev_response = await self.conversation_store.load_response(prev_response_id)
                    if prev_response is None:
                        logger.debug(f"response_id {prev_response_id} not found")
                        return self._create_response_id_not_found_error(prev_response_id)

            input_tokens, sampling_params = await responses_api_request_preprocess(
                request, prev_response, self.harmony_adapter, self.conversation_store, self.enable_store)

            promise = self.llm.generate_async(
                inputs=input_tokens,
                sampling_params=sampling_params,
                streaming=request.stream,
            )

            asyncio.create_task(self.await_disconnected(raw_request, promise))

            if request.stream:
                return StreamingResponse(
                    create_stream_response(promise, request, sampling_params),
                    media_type="text/event-stream"
                )
            else:
                return await responses_api_create_response(
                    generator=promise,
                    request=request,
                    sampling_params=sampling_params,
                    model_name=self.model,
                    conversation_store=self.conversation_store,
                    generation_result=None,
                    enable_store=self.enable_store)
        except CppExecutorError:
            logger.error(traceback.format_exc())
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(str(e))

        return JSONResponse(content={"detail": "None"})


    async def __call__(self, host, port):
        # Store the binding address for server registration
        self.binding_addr = f"http://{host}:{port}"
        self.host = host
        self.port = port
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()
