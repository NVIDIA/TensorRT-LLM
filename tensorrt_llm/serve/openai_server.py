#!/usr/bin/env python
import array
import asyncio
import base64
import functools
import json
import os
import re
import signal
import socket
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import (Annotated, Any, AsyncGenerator, AsyncIterator, List,
                    Optional, Union)

import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (FileResponse, JSONResponse, Response,
                               StreamingResponse)
from fastapi.routing import APIRoute
from pydantic import ValidationError
from starlette.routing import Mount
from transformers import AutoProcessor

from tensorrt_llm._torch.async_llm import AsyncLLM
from tensorrt_llm._utils import EnergyMonitor
# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.inputs import prompt_inputs
from tensorrt_llm.inputs.data import TokensPrompt
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor
from tensorrt_llm.inputs.utils import (ConversationMessage,
                                       async_apply_chat_template)
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi import MultimodalEncoder, SchedulingParams, tracing
from tensorrt_llm.llmapi.disagg_utils import (DisaggClusterConfig,
                                              MetadataServerConfig, ServerRole)
from tensorrt_llm.llmapi.llm import LLM, RequestOutput
from tensorrt_llm.llmapi.thinking_budget import \
    add_thinking_budget_logits_processor
from tensorrt_llm.logger import logger
from tensorrt_llm.media.encoding import image_to_bytes
from tensorrt_llm.media.tensor_payload import is_tensor_format
from tensorrt_llm.metrics.collector import MetricsCollector
from tensorrt_llm.runtime.kv_cache_hash import \
    get_effective_kv_cache_event_hash_algo
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams
from tensorrt_llm.serve.chat_tokenization import tokenize_harmony_chat_request
from tensorrt_llm.serve.chat_utils import (load_chat_template,
                                           parse_chat_messages_coroutines,
                                           resolve_top_level_model_type)
from tensorrt_llm.serve.cluster_storage import create_cluster_storage_client
from tensorrt_llm.serve.conversation_id import resolve_request_conversation_id
from tensorrt_llm.serve.disagg_auto_scaling import DisaggClusterWorker
from tensorrt_llm.serve.encode_batcher import (EncodeBatcher, InputTooLongError,
                                               QueueFullError)
from tensorrt_llm.serve.metadata_server import create_metadata_server
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice,
    ChatMessage, CompletionRequest, CompletionResponse,
    CompletionResponseChoice, EmbeddingRequest, EmbeddingResponse,
    EmbeddingResponseData, EmbeddingUsageInfo, ErrorResponse,
    ImageGenerationRequest, ImageGenerationResponse, ImageObject,
    MemoryUpdateRequest, ModelCard, ModelList, PromptTokensDetails,
    ResponseFormat, ResponsesRequest, ResponsesResponse, TokenizeRequest,
    TokenizeResponse, UpdateWeightsRequest, UsageInfo,
    ensure_request_chat_template_allowed, to_llm_conversation_params,
    to_llm_disaggregated_params)
from tensorrt_llm.serve.openai_video_routes import _VideoRoutesMixin
from tensorrt_llm.serve.postprocess_handlers import (
    ChatCompletionPostprocArgs, ChatPostprocArgs, CompletionPostprocArgs,
    ResponsesAPIPostprocArgs, chat_harmony_post_processor,
    chat_harmony_streaming_post_processor, chat_response_post_processor,
    chat_stream_post_processor, completion_response_post_processor,
    completion_stream_post_processor, responses_api_post_processor,
    responses_api_streaming_post_processor)
from tensorrt_llm.serve.responses_utils import (ConversationHistoryStore,
                                                ResponsesStreamingProcessor,
                                                ServerArrivalTimeMiddleware)
from tensorrt_llm.serve.responses_utils import \
    create_response as responses_api_create_response
from tensorrt_llm.serve.responses_utils import get_steady_clock_now_in_seconds
from tensorrt_llm.serve.responses_utils import \
    request_preprocess as responses_api_request_preprocess
from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory
from tensorrt_llm.serve.visual_gen_metrics import \
    build_visual_gen_timing_headers
from tensorrt_llm.serve.visual_gen_utils import parse_visual_gen_params
from tensorrt_llm.version import __version__ as VERSION
from tensorrt_llm.visual_gen import VisualGen

from .._utils import nvtx_mark, set_prometheus_multiproc_dir
from .harmony_adapter import HarmonyAdapter, get_harmony_adapter

# yapf: enable

# msgspec msgpack is an opt-in transport for the disagg orchestrator->worker
# request body: the large agentic chat body otherwise blocks the serving event
# loop on stdlib json.loads. Enable with TRTLLM_SERVE_ENABLE_MSGSPEC=1 (must be
# set on both orchestrator and worker). The worker decodes bodies flagged with
# the X-TRTLLM-Msgpack header via msgspec and falls back to stdlib json for
# everything else, so the JSON path is byte-for-byte unchanged when the flag is off.
_MSGSPEC_ENABLED = os.getenv("TRTLLM_SERVE_ENABLE_MSGSPEC", "0") == "1"
if _MSGSPEC_ENABLED:
    try:
        import msgspec
    except ImportError as exc:
        raise ImportError(
            "TRTLLM_SERVE_ENABLE_MSGSPEC=1 requires the msgspec package "
            "(listed in requirements.txt).") from exc
    _msgpack_decoder = msgspec.msgpack.Decoder()

    class _MsgspecRequest(Request):
        """Request that decodes msgpack bodies (X-TRTLLM-Msgpack: 1) with msgspec.

        The orchestrator sends Content-Type application/json (so FastAPI still
        routes the body through Request.json()) with the X-TRTLLM-Msgpack header
        flagging a msgspec-msgpack payload; everything else is stdlib json.
        """

        async def json(self):
            if not hasattr(self, "_json_body"):
                body = await self.body()
                if not body:
                    self._json_body = {}
                elif self.headers.get("x-trtllm-msgpack") == "1":
                    self._json_body = _msgpack_decoder.decode(body)
                else:
                    self._json_body = json.loads(body)
            return self._json_body

    class _MsgspecRoute(APIRoute):
        """APIRoute that parses request bodies via :class:`_MsgspecRequest`."""

        def get_route_handler(self):
            original_route_handler = super().get_route_handler()

            async def route_handler(request: Request):
                return await original_route_handler(
                    _MsgspecRequest(request.scope, request.receive))

            return route_handler


TIMEOUT_KEEP_ALIVE = 5  # seconds.


def _build_tool_strict_guided_decoding_params(tools, tool_parser_name):
    """Build GuidedDecodingParams with structural tags for tools with strict=True.

    When a tool has ``strict=True`` in its function definition, the server
    should use constrained decoding to guarantee that the generated tool call
    arguments exactly match the function's ``parameters`` JSON Schema.

    This function builds structural tag items from each tool parser's
    ``structure_info()`` and the tool's ``parameters`` schema, then returns
    a ``GuidedDecodingParams`` with the structural tag format.

    Returns None if no tool has strict=True or the parser doesn't support
    structural tags.
    """
    if not tools or not tool_parser_name:
        return None

    # Check if any tool has strict=True
    has_strict = any(tool.function.strict for tool in tools
                     if tool.function.strict)
    if not has_strict:
        return None

    tool_parser_cls = ToolParserFactory.parsers.get(tool_parser_name.lower())
    if tool_parser_cls is None:
        logger.warning(
            "Tool parser '%s' not found, cannot enforce strict mode for tools.",
            tool_parser_name)
        return None

    parser = tool_parser_cls()
    if not parser.supports_structural_tag():
        logger.warning(
            "Tool parser '%s' does not support structural tags, "
            "cannot enforce strict mode for tools.", tool_parser_name)
        return None

    get_info = parser.structure_info()

    tags = []
    triggers = set()
    for tool in tools:
        info = get_info(tool.function.name)
        triggers.add(info.trigger)

        if tool.function.strict and tool.function.parameters:
            # Strict tool: constrain arguments to match the JSON Schema
            content = {
                "type": "json_schema",
                "json_schema": tool.function.parameters,
            }
        else:
            # Non-strict tool or no parameters: allow any text
            content = {"type": "any_text"}

        tags.append({
            "begin": info.begin,
            "content": content,
            "end": info.end,
        })

    stag_format = {
        "type": "triggered_tags",
        "triggers": sorted(triggers),
        "tags": tags,
    }

    resp_format = ResponseFormat(type="structural_tag", format=stag_format)
    return GuidedDecodingParams(structural_tag=resp_format.model_dump_json(
        by_alias=True, exclude_none=True))


def _normalize_image_output(image) -> list:
    """Normalize image output to a list of individual images.

    Handles single tensors, batched 4D tensors, and lists.
    """
    if isinstance(image, list):
        return image
    if hasattr(image, "dim") and image.dim() == 4:
        return [image[i] for i in range(image.shape[0])]
    return [image]


class OpenAIServer(_VideoRoutesMixin):

    @staticmethod
    def _iteration_stats_buffer_maxlen(
            iter_stats_max_iterations: Optional[int]) -> Optional[int]:
        if iter_stats_max_iterations is None or iter_stats_max_iterations == 0:
            return 1000
        if iter_stats_max_iterations < 0:
            return None
        return iter_stats_max_iterations

    def __init__(
            self,
            generator: Union[LLM, MultimodalEncoder, VisualGen],
            model: str,
            tool_parser: Optional[str],
            server_role: Optional[ServerRole],
            metadata_server_cfg: MetadataServerConfig,
            disagg_cluster_config: Optional[DisaggClusterConfig] = None,
            multimodal_server_config: Optional[MultimodalServerConfig] = None,
            chat_template: Optional[str] = None,
            allow_request_chat_template: bool = False,
            embedding_max_queue_delay: float = 0.005,
            embedding_max_queue_size: int = 2048,
            input_processor_workers: int = 8,
            media_load_workers: int = 8):
        self.generator = generator
        self._is_visual_gen = isinstance(generator, VisualGen)
        self._embedding_max_queue_delay = embedding_max_queue_delay
        self._embedding_max_queue_size = embedding_max_queue_size
        self.embedding_batcher: Optional[EncodeBatcher] = None
        self.tool_parser = tool_parser
        self.metadata_server = create_metadata_server(metadata_server_cfg)
        self.disagg_cluster_config = disagg_cluster_config
        self.multimodal_server_config = multimodal_server_config
        # Seed media-IO decode defaults from the model's input processor, then
        # let the server's --media_io_kwargs win (per-request kwargs still
        # override at request time).
        ip = getattr(self.generator, "input_processor", None)
        if isinstance(ip, BaseMultimodalInputProcessor):
            model_pref = ip.get_preferred_media_io_kwargs() or {}
            if model_pref:
                cfg = self.multimodal_server_config or MultimodalServerConfig()
                merged = {m: dict(kw) for m, kw in model_pref.items()}
                for modality, kw in (cfg.media_io_kwargs or {}).items():
                    merged.setdefault(modality, {}).update(kw)
                cfg.media_io_kwargs = merged
                self.multimodal_server_config = cfg
        self.allow_request_chat_template = allow_request_chat_template
        self.server_role = server_role
        # Will be set in __call__
        self.binding_addr = None
        self.host = None
        self.port = None

        # Dedicated thread pools for the chat / completion path. Keeping
        # multimodal preprocessing and decode work off the asyncio default
        # executor avoids contention with unrelated `to_thread` callers and
        # lets the two stages be sized independently.
        self._input_proc_executor = ThreadPoolExecutor(
            max_workers=input_processor_workers,
            thread_name_prefix="trtllm_inputproc",
        )
        self._media_load_executor = ThreadPoolExecutor(
            max_workers=media_load_workers,
            thread_name_prefix="trtllm_media_load",
        )
        BaseMediaIO.set_executor(self._media_load_executor)

        model_dir = Path(model)
        if model_dir.exists() and model_dir.is_dir():
            self.model = model_dir.name
        else:
            self.model = model
        self.metrics_collector = None
        self.perf_metrics = None
        self.perf_metrics_lock = None
        self._iteration_stats_collector_task = None
        self._iteration_stats_wakeup_event = asyncio.Event()
        # Bounded snapshot of iteration stats for the GET /metrics handler.
        # When the background Prometheus collector loop is active, it is the
        # sole consumer of the engine's stats queue and appends each drained
        # stat here; /metrics then serves from this buffer instead of racing
        # the loop for the queue. Created lazily when the loop starts.
        # See nvbug 6102381.
        self._iteration_stats_buffer: Optional[deque] = None
        # The steady clock offset (in seconds) between this server and the disagg server
        self.disagg_server_steady_clock_offset = 0

        # Energy monitoring
        self.energy_monitor = None

        # as disagg-worker
        self.disagg_cluster_storage = None
        self.disagg_cluster_worker = None
        self.resource_governor = None

        # Skip loading AutoProcessor and model_config for VISUAL_GEN models
        # These are LLM-specific and can cause unnecessary memory usage
        if self._is_visual_gen:
            self._init_visual_gen()
        else:
            self._init_llm(chat_template)

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
                self.metadata_server.put(f"trtllm/{self.generator.llm_id}",
                                         metadata)
                logger.info(f"trtllm/{self.generator.llm_id} is registered")

            if self.disagg_cluster_config:
                self.disagg_cluster_storage = create_cluster_storage_client(
                    self.disagg_cluster_config.cluster_uri,
                    self.disagg_cluster_config.cluster_name)
                self.disagg_cluster_worker = DisaggClusterWorker(
                    self.server_role, self.host, self.port,
                    self.disagg_cluster_config, self.disagg_cluster_storage)

            # VisualGen has no args
            if not isinstance(self.generator, VisualGen):
                # Start energy monitoring if enabled
                if getattr(self.generator.args, "enable_energy_metrics", False):
                    try:
                        world_size = self.generator.args.parallel_config.world_size
                        self.energy_monitor = EnergyMonitor(world_size)
                        logger.info("Initialized GPU energy monitoring")
                    except Exception as e:
                        logger.warning(
                            f"Failed to initialize GPU energy monitoring: {e}")
                        self.energy_monitor = None

                # Start background iteration stats collector if metrics are enabled
                # The args for pytorch and autodeploy backend has attribute `enable_iter_perf_stats` while
                # tensorrt backend does not have this attribute but it always has iter stats enabled.
                if self.metrics_collector and getattr(
                        self.generator.args, "enable_iter_perf_stats", True):
                    # The background loop becomes the sole consumer of the
                    # engine stats queue; /metrics reads from a tee buffer
                    # bounded by iter_stats_max_iterations to avoid racing
                    # the loop for the queue (nvbug 6102381).
                    # One shared buffer is sufficient while this collector task
                    # is the only consumer of the engine iteration-stats queue.
                    # Other consumers can read it through get_iteration_stats(),
                    # which clears the buffer. Adding another queue consumer
                    # requires revisiting the buffering and clearing ownership.
                    max_buf = self._iteration_stats_buffer_maxlen(
                        getattr(self.generator.args,
                                "iter_stats_max_iterations", 1000))
                    self._iteration_stats_buffer = deque(maxlen=max_buf)
                    self._iteration_stats_collector_task = asyncio.create_task(
                        self._iteration_stats_collector_loop())
                    # Wake up the collector immediately so it processes the
                    # initial stats emitted by the executor at startup (e.g.
                    # cache_config_info).
                    self._iteration_stats_wakeup_event.set()
                    logger.info(
                        "Started background iteration stats collector task")

            # Start the encode dynamic batcher (embedding server only). It must be
            # started inside the running event loop, hence here rather than __init__.
            if self.embedding_batcher is not None:
                await self.embedding_batcher.start()
                logger.info("Started encode dynamic batcher")

            yield

            if self.embedding_batcher is not None:
                await self.embedding_batcher.shutdown()
                logger.info("Stopped encode dynamic batcher")

            # Stop background iteration stats collector
            if self._iteration_stats_collector_task is not None:
                self._iteration_stats_collector_task.cancel()
                try:
                    await self._iteration_stats_collector_task
                except asyncio.CancelledError:
                    pass
                logger.info("Stopped background iteration stats collector task")

            if self.metadata_server is not None:
                self.metadata_server.remove(f"trtllm/{self.generator.llm_id}")
                logger.info(f"trtllm/{self.generator.llm_id} is unregistered")
            if self.disagg_cluster_worker:
                await self.disagg_cluster_worker.deregister_worker()
            if self.resource_governor is not None:
                self.resource_governor.close()
            self.generator.shutdown()

        self.app = FastAPI(lifespan=lifespan)
        if _MSGSPEC_ENABLED:
            self.app.router.route_class = _MsgspecRoute

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(_, exc):
            if self.server_role is ServerRole.VISUAL_GEN:
                return self._create_visual_gen_validation_error_response(exc)
            # Non-visual-gen roles keep the shared 400 + ``{"error": ...}``
            # response shape that integration tests (e.g.
            # ``test_malformed_json_request``) and existing clients
            # expect.
            if self.metrics_collector:
                self.metrics_collector.log_request_error(http_code=400)
            return JSONResponse(status_code=400, content={"error": str(exc)})

        if self.server_role is ServerRole.VISUAL_GEN:
            assert isinstance(
                self.generator, VisualGen
            ), "generator must be a VisualGen for VISUAL_GEN server"
            self.register_visual_gen_routes()
        elif self.server_role is ServerRole.MM_ENCODER:
            assert isinstance(
                self.generator, MultimodalEncoder
            ), "generator must be a MultimodalEncoder for multimodal encoder"
            self.register_mm_encoder_routes()
        elif self.server_role is ServerRole.EMBEDDING:
            assert getattr(self.generator.args, "encode_only", False), (
                "generator must be an encode_only=True LLM for the embedding "
                "server")
            self._init_embedding_batcher()
            self.register_embedding_routes()
        else:
            self.register_routes()

        self.app.add_middleware(ServerArrivalTimeMiddleware)

    def _init_visual_gen(self):
        self.processor = None
        self.model_config = None
        self.media_storage_path = Path(
            os.getenv("TRTLLM_MEDIA_STORAGE_PATH",
                      "/tmp/trtllm_generated"))  # nosec B108
        self.media_storage_path.mkdir(exist_ok=True, parents=True)
        self.video_gen_tasks = {}

    def _init_llm(self, chat_template: Optional[str] = None):
        self.tokenizer = self.generator.tokenizer
        hf_tokenizer_path = self.generator._hf_model_dir
        if not hf_tokenizer_path:
            hf_tokenizer_path = getattr(
                self.tokenizer.tokenizer, "name_or_path", None) or getattr(
                    self.tokenizer, "name_or_path", None)
        trust_remote_code = self.generator.args.trust_remote_code
        try:
            self.processor = AutoProcessor.from_pretrained(
                hf_tokenizer_path, trust_remote_code=trust_remote_code)
        except Exception:
            logger.debug("Failed to load AutoProcessor or AutoConfig for %s",
                         hf_tokenizer_path)
            self.processor = None

        # load model config
        try:
            from tensorrt_llm._torch.pyexecutor.config_utils import \
                load_pretrained_config
            self.model_config = load_pretrained_config(
                hf_tokenizer_path,
                trust_remote_code=trust_remote_code,
                checkpoint_format=getattr(self.generator.args,
                                          "checkpoint_format", None))
        except Exception:
            logger.debug("Failed to load AutoConfig for %s", hf_tokenizer_path)
            self.model_config = None

        self.chat_template = load_chat_template(chat_template)

        # Enable response storage for Responses API
        self.enable_store = (len(
            os.getenv("TRTLLM_RESPONSES_API_DISABLE_STORE", ""))
                             < 1) and not self.postproc_worker_enabled

        self.conversation_store = ConversationHistoryStore()

        # gpt-oss
        self.harmony_adapter: HarmonyAdapter | None = None
        disable_harmony = os.getenv("DISABLE_HARMONY_ADAPTER", "0") == "1"
        if disable_harmony or self.model_config is None:
            self.use_harmony = False
        else:
            self.use_harmony = (type(self.model_config).model_type == "gpt_oss")

        self._ensure_post_processor_hook_supported(
            self.use_harmony,
            getattr(self.generator.args, "post_processor_hook", None))

        self.tool_call_id_type = "random"  # default tool call id type is random
        if self.model_config is not None:
            # NOTE: Use the instance-level ``model_type`` (JSON-derived) here, not
            # ``type(cfg).model_type``. ``kimi_k2`` / ``deepseek_v32`` are aliases
            # registered in ``_CONFIG_REGISTRY`` (config_utils.py) that reuse
            # ``DeepseekV3Config``, whose class-level ``model_type`` is ``"deepseek_v3"``.
            # Only the JSON ``model_type`` distinguishes these variants. Other call
            # sites that consult multimodal/chat-template registries keyed on the
            # canonical class attribute should keep using ``type(cfg).model_type``.
            if self.model_config.model_type == "kimi_k2":
                self.tool_call_id_type = "kimi_k2"
            elif self.model_config.model_type == "deepseek_v32":
                self.tool_call_id_type = "deepseek_v32"

        if self.generator.args.return_perf_metrics:
            set_prometheus_multiproc_dir()
            args = self.generator.args
            pmc = getattr(args, "prometheus_metrics_config", None)
            self.metrics_collector = MetricsCollector(
                {
                    "model_name": self.model,
                    "engine_type": args.backend or "unknown"
                },
                e2e_request_latency_buckets=(pmc.e2e_request_latency_buckets
                                             if pmc else None),
                time_to_first_token_buckets=(pmc.time_to_first_token_buckets
                                             if pmc else None),
                time_per_output_token_buckets=(pmc.time_per_output_token_buckets
                                               if pmc else None),
                request_queue_time_buckets=(pmc.request_queue_time_buckets
                                            if pmc else None),
                request_prefill_time_buckets=(pmc.request_prefill_time_buckets
                                              if pmc else None),
                request_decode_time_buckets=(pmc.request_decode_time_buckets
                                             if pmc else None),
                request_inference_time_buckets=(
                    pmc.request_inference_time_buckets if pmc else None),
            )
            self._log_config_info_metrics()
            max_perf_metrics = self.generator.args.perf_metrics_max_requests
            if max_perf_metrics > 0:
                self.perf_metrics = deque(maxlen=max_perf_metrics)
                self.perf_metrics_lock = asyncio.Lock()

    @staticmethod
    def _ensure_post_processor_hook_supported(
            use_harmony: bool, post_processor_hook: Optional[str]) -> None:
        """Reject ``--post_processor_hook`` combined with a harmony/gpt-oss model.

        The harmony output path is rebuilt from raw token ids, not detokenized
        text, so the text-based hook cannot act there.
        """
        if use_harmony and post_processor_hook:
            raise ValueError(
                "--post_processor_hook is not supported with harmony/gpt-oss models "
                "in this version: the harmony output path is reconstructed from "
                "raw token ids and would bypass the text-based hook. Disable the "
                "hook or set DISABLE_HARMONY_ADAPTER=1 if the harmony path is "
                "not needed.")

    def _logit_bias_vocab_size(self) -> int:
        for config in (self.model_config,
                       getattr(self.model_config, "text_config", None)):
            vocab_size = getattr(config, "vocab_size", None)
            if vocab_size is not None:
                return int(vocab_size)
        return int(self.tokenizer.tokenizer.vocab_size)

    def _log_config_info_metrics(self) -> None:
        """Extract configuration from generator args and log as Prometheus info gauges."""
        args = self.generator.args

        # Model config
        model_config = {
            "model": str(args.model),
            "served_model_name": self.model,
            "dtype": str(args.dtype),
        }
        quant_config = getattr(args, "quant_config", None)
        if quant_config is not None:
            quant_algo = getattr(quant_config, "quant_algo", None)
            model_config["quantization"] = str(
                quant_algo) if quant_algo else "none"
        else:
            model_config["quantization"] = "none"
        max_seq_len = getattr(args, "max_seq_len", None)
        if max_seq_len is not None:
            model_config["max_model_len"] = str(max_seq_len)
        try:
            import torch
            if torch.cuda.is_available():
                model_config["gpu_type"] = torch.cuda.get_device_name(0)
        except (ImportError, RuntimeError) as e:
            logger.debug("Could not detect GPU type for config metrics: %s", e)

        # Parallel config — prefer parallel_config from generator args
        # for accurate values including cp_size and world_size.
        par_cfg = getattr(args, "parallel_config", None)
        if par_cfg is not None:
            tp_size = getattr(par_cfg, "tp_size", 1) or 1
            pp_size = getattr(par_cfg, "pp_size", 1) or 1
            cp_size = getattr(par_cfg, "cp_size", 1) or 1
            world_size = getattr(par_cfg, "world_size",
                                 tp_size * pp_size * cp_size)
        else:
            tp_size = getattr(args, "tensor_parallel_size", 1) or 1
            pp_size = getattr(args, "pipeline_parallel_size", 1) or 1
            cp_size = 1
            world_size = tp_size * pp_size * cp_size
        parallel_config = {
            "tensor_parallel_size": str(tp_size),
            "pipeline_parallel_size": str(pp_size),
            "context_parallel_size": str(cp_size),
            "gpu_count": str(world_size),
        }
        ep_size = getattr(par_cfg, "moe_ep_size", None) if par_cfg else \
            getattr(args, "moe_expert_parallel_size", None)
        if ep_size is not None and ep_size > 0:
            parallel_config["expert_parallel_size"] = str(ep_size)

        # Speculative decoding config
        spec_config_obj = getattr(args, "speculative_config", None) or getattr(
            args, "decoding_config", None)
        speculative_config = None
        if spec_config_obj is not None:
            speculative_config = {"spec_enabled": "true"}
            decoding_type = getattr(spec_config_obj, "decoding_type", None)
            if decoding_type is not None:
                speculative_config["spec_method"] = str(decoding_type)
            max_draft_len = getattr(spec_config_obj, "max_draft_len", None)
            if max_draft_len is not None:
                speculative_config["spec_num_tokens"] = str(max_draft_len)
            draft_model = getattr(spec_config_obj, "speculative_model", None)
            if draft_model is not None:
                speculative_config["spec_draft_model"] = str(draft_model)

        # KV cache config
        kv_cache_config_obj = getattr(args, "kv_cache_config", None)
        kv_cache_config = None
        if kv_cache_config_obj is not None:
            kv_cache_config = {}
            for field in ("page_size", "enable_block_reuse",
                          "enable_partial_reuse", "free_gpu_memory_fraction"):
                val = getattr(kv_cache_config_obj, field, None)
                if val is not None:
                    kv_cache_config[field] = str(val)
            kv_dtype = getattr(kv_cache_config_obj, "dtype", None)
            if kv_dtype is not None:
                kv_cache_config["cache_dtype"] = str(kv_dtype)

        self.metrics_collector.log_config_info(
            model_config=model_config,
            parallel_config=parallel_config,
            speculative_config=speculative_config,
            kv_cache_config=kv_cache_config if kv_cache_config else None,
        )

    async def await_disconnected(self, raw_request: Request, promise):
        if raw_request is None:
            return
        while not await raw_request.is_disconnected():
            await asyncio.sleep(1)
        if not promise.finished:
            promise.abort()
            logger.info(
                f"{raw_request.client} is disconnected, abort {promise.request_id}"
            )

    @property
    def postproc_worker_enabled(self) -> bool:
        if isinstance(self.generator, VisualGen):
            return False

        return True if self.generator.args.num_postprocess_workers > 0 else False

    @property
    def _vocab_size(self) -> Optional[int]:
        if self.tokenizer is not None and self.tokenizer.tokenizer is not None:
            return self.tokenizer.tokenizer.vocab_size
        return None

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

    def _create_not_supported_error(self, message: str) -> Response:
        return self.create_error_response(
            err_type="NotImplementedError",
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
        )

    def _create_visual_gen_validation_error_response(
            self, exc: RequestValidationError) -> Response:
        """Render a ``RequestValidationError`` as the visual-gen 422 envelope.

        The body has the LLM-style ``{message, type, code}`` shape with
        HTTP 422. The ``message`` field names the offending field(s) for
        each error in ``exc.errors()`` so clients can fix the request
        without parsing the full Pydantic payload.
        """
        parts: List[str] = []
        for err in exc.errors():
            loc = ".".join(
                str(seg) for seg in err.get("loc", ()) if seg != "body")
            etype = err.get("type", "")
            msg = err.get("msg", "")
            if etype == "extra_forbidden":
                parts.append(
                    f"Unknown request field {loc!r}. Pass model-specific "
                    "parameters via 'extra_params' instead.")
            elif loc:
                parts.append(f"{loc}: {msg}")
            else:
                parts.append(msg)
        message = "; ".join(parts) if parts else str(exc)
        return self.create_error_response(
            message=message,
            err_type="BadRequestError",
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    def _check_health(self) -> bool:
        # An embedding server's requests flow through the batcher worker; if it
        # has exited the engine is up but every /v1/embeddings request hangs, so
        # report unhealthy rather than a misleading 200.
        batcher = self.embedding_batcher
        if batcher is not None and not batcher.is_alive():
            return False
        if hasattr(self.generator, '_check_health'):
            return self.generator._check_health()
        return True

    def register_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/health_generate",
                               self.health_generate,
                               methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        self.app.add_api_route("/v1/data_transceiver_state",
                               self.data_transceiver_state,
                               methods=["GET"])
        # TODO: the metrics endpoint only reports iteration stats, not the runtime stats for now
        self.app.add_api_route("/metrics",
                               self.get_iteration_stats,
                               methods=["GET"])
        self.app.add_api_route("/perf_metrics",
                               self.get_perf_metrics,
                               methods=["GET"])
        self.app.add_api_route("/energy_metrics",
                               self.get_energy_metrics,
                               methods=["GET"])
        self.app.add_api_route("/steady_clock_offset",
                               self.get_steady_clock_offset,
                               methods=["GET"])
        # Called by the disagg server to set the disagg_server_steady_clock_offset
        self.app.add_api_route("/steady_clock_offset",
                               self.set_steady_clock_offset,
                               methods=["POST"])
        # TODO: workaround before ETCD support
        self.app.add_api_route("/kv_cache_events",
                               self.get_kv_cache_events,
                               methods=["POST"])
        resource_governor_queue = self.generator._executor.resource_governor_queue
        if resource_governor_queue is not None:
            from .resource_governor import ResourceGovernor
            self.resource_governor = ResourceGovernor(
                resource_governor_queue=resource_governor_queue,
                tokenizer=self.tokenizer,
                model_config=self.model_config,
                processor=self.processor,
                allow_request_chat_template=self.allow_request_chat_template,
                harmony_adapter_factory=get_harmony_adapter
                if self.use_harmony else None,
            )
            self.resource_governor.register_routes(self.app)
        else:
            # Resource governor is unavailable because the executor does not
            # expose a resource_governor_queue. This is expected in RPC
            # orchestrator mode (GenerationExecutorRpcProxy), non-PyExecutor
            # backends, or when enable_resource_governor is false. The
            # /_resource_governor/* endpoints will not be registered; clients
            # that attempt to call them will receive 404.
            logger.warning(
                "Resource governor is disabled: the executor backend does "
                "not provide a resource_governor_queue (e.g. RPC "
                "orchestrator mode or explicit opt-out). Endpoints under "
                "/_resource_governor/ will not be available.")
            self.resource_governor = None

        self.app.add_api_route("/v1/completions",
                               self.openai_completion,
                               methods=["POST"])
        self.app.add_api_route(
            "/v1/chat/completions",
            self.openai_chat if not self.use_harmony else self.chat_harmony,
            methods=["POST"])
        self.app.add_api_route("/v1/responses",
                               self.openai_responses,
                               methods=["POST"])
        self.app.add_api_route('/v1/responses/{response_id}',
                               self.openai_responses_get_response,
                               methods=["GET"])
        self.app.add_api_route('/v1/responses/{response_id}',
                               self.openai_responses_delete_response,
                               methods=["DELETE"])
        self.app.add_api_route("/_internal/tokenize",
                               self.tokenize,
                               methods=["POST"])

        # RL-only endpoints
        self.app.add_api_route("/release_memory",
                               self.release_memory,
                               methods=["POST"])
        self.app.add_api_route("/resume_memory",
                               self.resume_memory,
                               methods=["POST"])
        self.app.add_api_route("/update_weights",
                               self.update_weights,
                               methods=["POST"])
        self.app.add_api_route("/server_info",
                               self.get_server_info,
                               methods=["GET"])
        if self.generator.args.return_perf_metrics:
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
            excluded_handlers=[".*"],
            registry=registry,
        ).add().instrument(self.app).expose(self.app)
        metrics_app = make_asgi_app(registry=registry)
        metrics_route = Mount("/prometheus/metrics", metrics_app)
        metrics_route.path_regex = re.compile(
            "^/prometheus/metrics(?P<path>.*)$")
        self.app.routes.append(metrics_route)

    def register_mm_encoder_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        # TODO: the metrics endpoint only reports iteration stats, not the runtime stats for now
        self.app.add_api_route("/metrics",
                               self.get_iteration_stats,
                               methods=["GET"])
        self.app.add_api_route("/v1/chat/completions",
                               self.openai_mm_encoder,
                               methods=["POST"])
        # RL-only endpoints
        self.app.add_api_route("/release_memory",
                               self.release_memory,
                               methods=["POST"])
        self.app.add_api_route("/resume_memory",
                               self.resume_memory,
                               methods=["POST"])
        self.app.add_api_route("/update_weights",
                               self.update_weights,
                               methods=["POST"])

    def _init_embedding_batcher(self):
        """Create the encode dynamic batcher for the embedding server.

        The batcher coalesces concurrent /v1/embeddings requests into a single
        `llm.encode()` call. `encode()` is synchronous; the batcher runs it in
        the default executor so the event loop stays responsive.
        """

        def encode_fn(token_ids_batch):
            # token_ids_batch: list of pre-tokenized token-id lists. encode()
            # returns one EncoderOutput per item, in input order.
            return self.generator.encode(token_ids_batch)

        # Source the batch limits from the resolved encoder engine — the same
        # attributes encode() validates against (llm.py: engine.max_seq_len /
        # max_num_tokens / batch_size). LlmArgs.max_seq_len/max_num_tokens default
        # to None and are not written back to args, so reading them from args
        # would leave the batcher's seq-len/token-budget guards inert (typed 400s
        # would never fire) and could let it form a batch encode() then rejects.
        engine = self.generator._encoder_executor.model_engine
        self.embedding_batcher = EncodeBatcher(
            encode_fn,
            max_batch_size=engine.batch_size,
            max_queue_delay=self._embedding_max_queue_delay,
            max_queue_size=self._embedding_max_queue_size,
            max_num_tokens=engine.max_num_tokens,
            max_seq_len=engine.max_seq_len,
        )

    def register_embedding_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        self.app.add_api_route("/v1/embeddings",
                               self.openai_embedding,
                               methods=["POST"])

    @staticmethod
    def _normalize_embedding_input(inp):
        """Normalize EmbeddingRequest.input into a list of items.

        Each item is either a string (to tokenize) or a pre-tokenized token-id
        list. Mirrors the four OpenAI input forms: str | list[str] | list[int] |
        list[list[int]].
        """
        if isinstance(inp, str):
            return [inp]
        if isinstance(inp, list):
            if len(inp) == 0:
                return []
            if isinstance(inp[0], int):
                return [inp]  # a single pre-tokenized input
            return list(inp)  # list of strings or list of token-id lists
        raise TypeError(f"Unsupported embedding input type: {type(inp)}")

    async def openai_embedding(self, request: EmbeddingRequest,
                               raw_request: Request) -> Response:
        try:
            # `dimensions` is a Matryoshka-embedding knob (truncate-then-renormalize
            # a sentence-embedding vector), matching vLLM, which rejects it for
            # non-Matryoshka models. None of the models served here are
            # Matryoshka-trained: BERT classifiers / reward models emit raw
            # [num_labels] / [seq_len, num_labels] tensors, and Qwen3-Embedding emits
            # a fixed-width pooled vector with no nested sub-dimensions. Truncating
            # any of these is meaningless, so reject. (Honoring `dimensions` for a
            # genuinely Matryoshka model would be a fast-follow.)
            if request.dimensions is not None:
                return self.create_error_response(
                    "`dimensions` is only supported by Matryoshka-trained "
                    "text-embedding models; none of the models served by this "
                    "endpoint are Matryoshka-trained.",
                    status_code=HTTPStatus.BAD_REQUEST)

            items = self._normalize_embedding_input(request.input)
            if not items:
                return self.create_error_response("`input` must not be empty.")

            # Tokenize strings via the same input processor used by /v1/completions,
            # honoring add_special_tokens. Pre-tokenized inputs are used as-is.
            sampling_params = SamplingParams(
                add_special_tokens=request.add_special_tokens)
            token_ids_list = []
            for item in items:
                if isinstance(item, str):
                    token_ids, _ = await asyncio.to_thread(
                        self.generator.input_processor, prompt_inputs(item),
                        sampling_params)
                else:
                    token_ids = item
                token_ids_list.append(token_ids)

            try:
                # Validate every input's length up front so an oversize item fails
                # the whole request before any item is enqueued — otherwise gather()
                # cancels the awaiters but their already-queued inputs still run
                # encode() (wasted GPU work).
                for token_ids in token_ids_list:
                    self.embedding_batcher.validate_input(token_ids)
                results = await asyncio.gather(*[
                    self.embedding_batcher.submit(token_ids)
                    for token_ids in token_ids_list
                ])
            except InputTooLongError as e:
                return self.create_error_response(
                    str(e), status_code=HTTPStatus.BAD_REQUEST)
            except QueueFullError as e:
                return self.create_error_response(
                    str(e), status_code=HTTPStatus.TOO_MANY_REQUESTS)

            data = []
            for idx, encoder_output in enumerate(results):
                # Model-agnostic: serialize whatever per-request tensor encode()
                # emits (classification logits, reward score, pooled vector, ...).
                embedding = encoder_output.logits.flatten().tolist()
                if request.encoding_format == "base64":
                    embedding = base64.b64encode(
                        array.array("f", embedding).tobytes()).decode("utf-8")
                data.append(
                    EmbeddingResponseData(index=idx, embedding=embedding))

            num_prompt_tokens = sum(len(t) for t in token_ids_list)
            usage = EmbeddingUsageInfo(prompt_tokens=num_prompt_tokens,
                                       total_tokens=num_prompt_tokens)
            response = EmbeddingResponse(model=request.model,
                                         data=data,
                                         usage=usage)
            return JSONResponse(content=response.model_dump())
        except Exception as e:
            # Unexpected fault (not a client error): surface as 500, not 400.
            logger.error(traceback.format_exc())
            return self.create_error_response(
                str(e),
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    def register_visual_gen_routes(self):
        """Register routes for diffusion model serving."""
        # Health and info endpoints
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        self.app.add_api_route("/metrics",
                               self.get_iteration_stats,
                               methods=["GET"])

        # Image generation endpoints (OpenAI compatible)
        self.app.add_api_route("/v1/images/generations",
                               self.openai_image_generation,
                               methods=["POST"])
        self.app.add_api_route("/v1/images/edits",
                               self.openai_image_edit,
                               methods=["POST"])
        self.app.add_api_route("/v1/images/{image_id}/content",
                               self.get_image_content,
                               methods=["GET"])

        # Video generation endpoints (Extended OpenAI API)
        # Asynchronous video generation (returns immediately with job metadata, OpenAI API)
        self.app.add_api_route("/v1/videos",
                               self.openai_video_generation_async,
                               methods=["POST"])
        # Synchronous video generation (waits for completion, extended API)
        self.app.add_api_route("/v1/videos/generations",
                               self.openai_video_generation_sync,
                               methods=["POST"])
        # Video management endpoints
        self.app.add_api_route("/v1/videos", self.list_videos, methods=["GET"])
        self.app.add_api_route("/v1/videos/{video_id}",
                               self.get_video_metadata,
                               methods=["GET"])
        self.app.add_api_route("/v1/videos/{video_id}/content",
                               self.get_video_content,
                               methods=["GET"])
        self.app.add_api_route("/v1/videos/{video_id}",
                               self.delete_video,
                               methods=["DELETE"])

    async def data_transceiver_state(self) -> JSONResponse:
        """Return the serialized DataTransceiverState as base64-encoded JSON."""
        state = self.generator.get_data_transceiver_state()
        if not state:
            return JSONResponse(
                status_code=404,
                content={"error": "No transceiver state available"})
        import base64
        return JSONResponse(content={
            "data_transceiver_state":
            base64.b64encode(state).decode("utf-8")
        })

    async def health(self) -> Response:
        if self._check_health():
            return Response(status_code=200)
        else:
            # If the engine has a fatal error, trigger server shutdown so the
            # pod doesn't linger as a zombie that accepts connections but never
            # produces tokens.  Uses SIGINT (not SIGTERM) to match the
            # existing CppExecutorError handlers and let uvicorn perform
            # its graceful shutdown sequence.  The doing_shutdown guard
            # ensures we only send SIGINT once — repeated health probes
            # won't stack signals during an in-progress teardown.
            executor = getattr(self.generator, '_executor', None)
            if executor is not None and getattr(executor, '_fatal_error',
                                                None) is not None:
                if not getattr(executor, 'doing_shutdown', True):
                    logger.error(
                        "Health check detected fatal engine error, initiating "
                        f"server shutdown: {executor._fatal_error}")
                    signal.raise_signal(signal.SIGINT)
            return Response(
                status_code=503,
                content=
                "LLM is unavailable. Please check the server logs for more details."
            )

    async def health_generate(self, raw_request: Request) -> Response:
        """Health check that performs a minimal generation."""
        extra_args = {}
        if self.generator.args.max_beam_width > 1:
            extra_args = dict(
                use_beam_search=True,
                best_of=self.generator.args.max_beam_width,
                n=1,
            )
        try:
            # Create a minimal chat request
            health_request = ChatCompletionRequest(
                messages=[{
                    "role": "user",
                    "content": "hi"
                }],  # Minimal prompt (often > 1 token after tokenization)
                model=self.model,
                max_completion_tokens=1,  # Request only 1 token out
                stream=False,
                temperature=0.0,  # Deterministic output
                **extra_args,
            )

            # Call the chat completion logic
            response = await self.openai_chat(health_request, raw_request)

            # Check if the response indicates success (status code 200)
            if response.status_code == 200:
                return Response(status_code=200,
                                content="Generation health check OK")
            else:
                logger.error(
                    f"Health generate check failed with status code: {response.status_code}"
                )
                try:
                    # Attempt to get body for more details if possible
                    body = response.body if hasattr(
                        response, 'body') else await response.body()
                    logger.error(f"Health generate check response body: {body}")
                except Exception:
                    pass  # Ignore errors trying to get body details
                return Response(status_code=500,
                                content="Generation health check failed")

        except Exception as e:
            logger.error(f"Health generate check encountered exception: {e}")
            return Response(status_code=500,
                            content=f"Generation health check failed: {str(e)}")

    async def version(self) -> JSONResponse:
        ver = {"version": VERSION}
        return JSONResponse(content=ver)

    async def get_model(self) -> JSONResponse:
        model_list = ModelList(data=[ModelCard(id=self.model)])
        return JSONResponse(content=model_list.model_dump())

    async def get_iteration_stats(self) -> JSONResponse:
        # Visual-gen deployments use a separate iteration-stats producer
        # (see ``DiffusionRemoteClient._iter_stats``).  The LLM PyExecutor
        # stats queue is empty in visual-gen mode, so route /metrics to the
        # visual-gen-shaped producer instead. Each snapshot mirrors LLM
        # field names where it makes sense (numActiveRequests,
        # numQueuedRequests) so downstream consumers that already parse the
        # LLM /metrics shape work for visual-gen with minimal changes.
        if self._is_visual_gen:
            stats = []
            async for stat in self.generator.get_stats_async(timeout=None):
                stats.append(stat)
            return JSONResponse(content=stats)

        # When the background collector loop is active it is the sole
        # consumer of the engine stats queue; serve /metrics from the tee
        # buffer it populates so we do not race it for queue items. Racing
        # caused >80% iteration loss and ~2s per-call latency (nvbug 6102381).
        # The caller receives the stats accumulated since the previous /metrics
        # call (up to iter_stats_max_iterations) and the buffer is cleared.
        if self._iteration_stats_buffer is not None:
            stats = list(self._iteration_stats_buffer)
            self._iteration_stats_buffer.clear()
            return JSONResponse(content=stats)

        # Legacy path: no background collector -> read the queue directly.
        stats = []
        async for stat in self.generator.get_stats_async(2):
            stats.append(stat)
        return JSONResponse(content=stats)

    async def get_energy_metrics(self) -> JSONResponse:
        if self.energy_monitor is None:
            return JSONResponse(
                content={"error": "Energy monitoring is not available"},
                status_code=503)
        total_energy = self.energy_monitor.get_current_energy()
        if total_energy is None:
            return JSONResponse(content={"error": "Failed to read GPU energy"},
                                status_code=503)
        return JSONResponse(
            content={
                "total_energy_j": round(total_energy, 4),
                "query_time": time.perf_counter(),
            })

    async def set_steady_clock_offset(
            self, offset: Annotated[float, Body(embed=True)]) -> Response:
        self.disagg_server_steady_clock_offset = offset
        logger.info(
            f"The steady clock offset between local and disagg server: {offset} second"
        )
        return Response(status_code=200)

    async def get_steady_clock_offset(self) -> JSONResponse:
        receive_ts = get_steady_clock_now_in_seconds()
        await asyncio.sleep(0.2)
        transmit_ts = get_steady_clock_now_in_seconds()
        return JSONResponse(content={
            "receive_ts": receive_ts,
            "transmit_ts": transmit_ts
        })

    async def get_perf_metrics(self) -> JSONResponse:
        if self.perf_metrics is None:
            return JSONResponse(content=[])
        async with self.perf_metrics_lock:
            perf_metrics = self.perf_metrics
            self.perf_metrics = deque(
                maxlen=self.generator.args.perf_metrics_max_requests)
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
            server_first_token_time = metrics_dict.pop(
                "server_first_token_time", None)
            if server_first_token_time is not None:
                server_first_token_time += self.disagg_server_steady_clock_offset
            metrics_json["timing_metrics"] = {
                "server_arrival_time":
                server_arrival_time,
                "arrival_time":
                timing_metrics.arrival_time.total_seconds() +
                self.disagg_server_steady_clock_offset,
                "first_scheduled_time":
                timing_metrics.first_scheduled_time.total_seconds() +
                self.disagg_server_steady_clock_offset,
                "first_token_time":
                timing_metrics.first_token_time.total_seconds() +
                self.disagg_server_steady_clock_offset,
                "server_first_token_time":
                server_first_token_time,
                "last_token_time":
                timing_metrics.last_token_time.total_seconds() +
                self.disagg_server_steady_clock_offset,
            }
            metrics_json["kv_cache_metrics"] = {
                "num_total_allocated_blocks":
                kv_cache_metrics.num_total_allocated_blocks,
                "num_new_allocated_blocks":
                kv_cache_metrics.num_new_allocated_blocks,
                "num_reused_blocks": kv_cache_metrics.num_reused_blocks,
                "num_missed_blocks": kv_cache_metrics.num_missed_blocks,
            }
            if timing_metrics.kv_cache_size > 0:
                metrics_json["timing_metrics"].update({
                    # TODO: move to kv_cache_metrics
                    "kv_cache_size":
                    timing_metrics.kv_cache_size,
                    "kv_cache_transfer_start":
                    timing_metrics.kv_cache_transfer_start.total_seconds() +
                    self.disagg_server_steady_clock_offset,
                    "kv_cache_transfer_end":
                    timing_metrics.kv_cache_transfer_end.total_seconds() +
                    self.disagg_server_steady_clock_offset,
                })
            if speculative_decoding.total_draft_tokens > 0:
                metrics_json["speculative_decoding"] = {
                    "acceptance_rate": speculative_decoding.acceptance_rate,
                    "total_accepted_draft_tokens":
                    speculative_decoding.total_accepted_draft_tokens,
                    "total_draft_tokens":
                    speculative_decoding.total_draft_tokens,
                }
            metrics_dict["perf_metrics"] = metrics_json
        return JSONResponse(content=list(perf_metrics))

    async def get_kv_cache_events(self) -> JSONResponse:
        events = []
        try:
            async for event in self.generator.get_kv_cache_events_async(0):
                events.append(event)
        except (IndexError, asyncio.QueueEmpty):
            # queue is empty, no more events
            pass
        return JSONResponse(content=events)

    async def _extract_metrics(self, res: RequestOutput, raw_request: Request):
        if not res.finished:
            return
        if self.metrics_collector:
            if res.candidate_metrics:
                for candidate_m in res.candidate_metrics:
                    self.metrics_collector.log_request_metrics_dict(candidate_m)
            elif res.metrics_dict:
                # Fallback for paths that populate metrics_dict directly
                # (e.g. PostprocWorker).
                self.metrics_collector.log_request_metrics_dict(
                    res.metrics_dict)
            # Note: Iteration stats are collected by the background _iteration_stats_collector_loop task
            # Wake up the stats collector to drain iteration stats
            if getattr(self.generator.args, "enable_iter_perf_stats", True):
                self._iteration_stats_wakeup_event.set()
        if self.generator.args.return_perf_metrics:
            output = res.outputs[0]
            item = {
                "request_id": res.request_id,
                "perf_metrics": res.outputs[0].request_perf_metrics
            }
            if raw_request:
                item["server_arrival_time"] = getattr(raw_request.state,
                                                      "server_arrival_time",
                                                      None)
                if not getattr(raw_request.state, "server_first_token_time",
                               None):
                    raw_request.state.server_first_token_time = get_steady_clock_now_in_seconds(
                    )
                item[
                    "server_first_token_time"] = raw_request.state.server_first_token_time
            if output.disaggregated_params:
                item[
                    "ctx_request_id"] = output.disaggregated_params.ctx_request_id
            # Request-level time breakdown (on GenerationResult/RequestOutput, not CompletionOutput)
            if getattr(res, 'time_breakdown_metrics', None) is not None:
                item["time_breakdown_metrics"] = res.time_breakdown_metrics
            if self.perf_metrics is not None:
                async with self.perf_metrics_lock:
                    self.perf_metrics.append(item)

    async def _create_chat_response(
        self,
        promise: RequestOutput,
        postproc_params: PostprocParams,
        raw_request: Request,
        disaggregated_params: Optional[LlmDisaggregatedParams] = None
    ) -> ChatCompletionResponse:
        await promise.aresult()
        if promise.error is not None:
            raise RuntimeError(f"Generation failed: {promise.error}")
        if self.postproc_worker_enabled:
            chat_response = promise.outputs[0]._postprocess_result
        else:
            post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
            chat_response = post_processor(promise, args)

        if disaggregated_params is not None and disaggregated_params.request_type == "context_only":
            chat_response.prompt_token_ids = promise.prompt_token_ids

        if disaggregated_params is not None and chat_response.choices[
                0].disaggregated_params is None:
            raise ValueError(
                f"disaggregated_params is not set in the response for request"
                f" {disaggregated_params.disagg_request_id}")

        await self._extract_metrics(promise, raw_request)
        return chat_response

    async def _iteration_stats_collector_loop(self):
        """Background task that continuously collects iteration statistics from the LLM engine.

        This task runs in the background for the lifetime of the server and drains iteration
        stats from the engine's stats queue, logging every stat to Prometheus.  Gauges
        (kv_cache_hit_rate, kv_cache_utilization, kv_cache_iter_reuse_rate) are naturally
        overwritten with the latest value, while counters (missed_blocks_total,
        gen_alloc_blocks_total, etc.) must be incremented by *every* per-iteration delta
        to remain accurate.  Logging only the latest stat would drop counter deltas from
        earlier iterations and could leave gauges unset if the latest iteration had no
        context-phase activity.

        The task sleeps when idle and is woken up via _iteration_stats_wakeup_event when
        requests complete.

        The loop will continue until the task is cancelled during server shutdown.
        """
        try:
            logger.info("Iteration stats collector loop started")
            while True:
                # Wait for signal that requests have completed and stats may be available
                await self._iteration_stats_wakeup_event.wait()

                # Clear the event for next wakeup
                self._iteration_stats_wakeup_event.clear()

                # Drain all available iteration stats and log each one to Prometheus.
                try:
                    async for llm_stat in self.generator.get_stats_async(
                            timeout=0.5):
                        self.metrics_collector.log_iteration_stats(llm_stat)
                        # Tee into the /metrics snapshot buffer so the HTTP
                        # handler can serve without competing for the engine
                        # queue (nvbug 6102381).
                        if self._iteration_stats_buffer is not None:
                            self._iteration_stats_buffer.append(llm_stat)
                except Exception as e:
                    # Log errors but continue collecting stats
                    logger.error(f"Error collecting iteration stats: {e}",
                                 exc_info=True)
                    # Brief sleep to avoid tight loop on persistent errors
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Iteration stats collector loop cancelled")
            raise

    async def openai_chat(self, request: ChatCompletionRequest,
                          raw_request: Request) -> Response:

        def get_role() -> str:
            if request.add_generation_prompt:
                role = "assistant"
            else:
                role = request.messages[-1]["role"]
            return role

        async def chat_stream_generator(
                promise: RequestOutput,
                postproc_params: PostprocParams) -> AsyncGenerator[str, None]:
            try:
                if not self.postproc_worker_enabled:
                    post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                first_response = await anext(promise)
                raw_request.state.server_first_token_time = get_steady_clock_now_in_seconds(
                )
                pp_results = first_response.outputs[
                    0]._postprocess_result if self.postproc_worker_enabled else post_processor(
                        first_response, args)
                for pp_res in pp_results:
                    yield pp_res
                # Making sure we can handle the situation where there is only one response
                res = first_response
                async for res in promise:
                    pp_results = res.outputs[
                        0]._postprocess_result if self.postproc_worker_enabled else post_processor(
                            res, args)
                    for pp_res in pp_results:
                        yield pp_res
                yield "data: [DONE]\n\n"
                await self._extract_metrics(res, raw_request)
                nvtx_mark("generation ends")
            except:
                logger.error(traceback.format_exc())
                raise

        try:
            ensure_request_chat_template_allowed(
                request, self.allow_request_chat_template)
            conversation: List[ConversationMessage] = []
            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]
            # Pass the model vocabulary size so ``logit_bias`` can be
            # expanded into an embedding bias tensor in the sampler.
            vocab_size = getattr(self.tokenizer.tokenizer,
                                 "vocab_size", None) or getattr(
                                     self.tokenizer, "vocab_size", None)
            sampling_params = request.to_sampling_params(
                vocab_size=self._logit_bias_vocab_size(),
                gather_generation_logits=self.generator.args.
                gather_generation_logits,
                reasoning_parser=self.generator.args.reasoning_parser,
                backend=self.generator.args.backend)
            add_thinking_budget_logits_processor(
                sampling_params,
                reasoning_parser=self.generator.args.reasoning_parser,
                tokenizer=self.tokenizer,
                chat_template_kwargs=request.chat_template_kwargs,
            )
            if self.tool_parser and request.tools:
                tool_parser_cls = ToolParserFactory.parsers.get(
                    self.tool_parser.lower())
                if tool_parser_cls and getattr(
                        tool_parser_cls, 'needs_raw_special_tokens', False):
                    sampling_params.skip_special_tokens = False
                # When strict=True on any tool, apply constrained decoding
                # via structural tags (only if response_format doesn't already
                # set guided decoding).
                if sampling_params.guided_decoding is None:
                    strict_guided = _build_tool_strict_guided_decoding_params(
                        request.tools, self.tool_parser)
                    if strict_guided is not None:
                        sampling_params.guided_decoding = strict_guided
            postproc_args = ChatPostprocArgs.from_request(request)
            disaggregated_params = to_llm_disaggregated_params(
                request.disaggregated_params)

            try:
                conversation, mm_coroutines, mm_placeholder_counts, mm_item_order = parse_chat_messages_coroutines(
                    request.messages,
                    self.model_config,
                    self.multimodal_server_config,
                    request_media_io_kwargs=request.media_io_kwargs)
            except ValidationError:
                # ValidatorIterator rejects extra fields; fall back to raw JSON.
                raw_body = await raw_request.json()
                raw_messages = raw_body.get("messages", [])
                conversation, mm_coroutines, mm_placeholder_counts, mm_item_order = parse_chat_messages_coroutines(
                    raw_messages,
                    self.model_config,
                    self.multimodal_server_config,
                    request_media_io_kwargs=request.media_io_kwargs)

            # Decode base64 int32 prompt_token_ids relayed by the orchestrator.
            if request.prompt_token_ids is None and request.prompt_token_ids_b64:
                import numpy as np
                request.prompt_token_ids = np.frombuffer(
                    base64.b64decode(request.prompt_token_ids_b64),
                    dtype=np.int32).tolist()

            if request.prompt_token_ids is not None:
                prompt = request.prompt_token_ids
            else:
                prompt_task = async_apply_chat_template(
                    model_type=resolve_top_level_model_type(self.model_config),
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    conversation=conversation,
                    add_generation_prompt=request.add_generation_prompt,
                    mm_placeholder_counts=mm_placeholder_counts,
                    tools=tool_dicts,
                    documents=request.documents,
                    chat_template=request.chat_template or self.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs or {},
                )
                prompt, (mm_data, mm_embeddings) = await asyncio.gather(
                    prompt_task, mm_coroutines)
            prompt = prompt_inputs(prompt)

            if request.prompt_token_ids is not None:
                mm_data, mm_embeddings = await mm_coroutines
            if mm_data:
                prompt["multi_modal_data"] = mm_data
            if mm_embeddings:
                prompt["multi_modal_embeddings"] = mm_embeddings
            if mm_data and mm_embeddings:
                raise ValueError(
                    "Passing 'multi_modal_data' and 'multi_modal_embeddings' at the same time is not supported."
                )
            if mm_data and mm_item_order:
                prompt["mm_item_order"] = mm_item_order

            if request.mm_processor_kwargs:
                prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

            postproc_args.reasoning_parser = self.generator.args.reasoning_parser
            postproc_args.tool_parser = self.tool_parser
            postproc_args.tool_call_id_type = self.tool_call_id_type
            if conversation and conversation[-1].get(
                    "content") and conversation[-1].get("role") == get_role():
                postproc_args.last_message_content = conversation[-1]["content"]
            postproc_params = PostprocParams(
                post_processor=chat_stream_post_processor
                if request.stream else chat_response_post_processor,
                postproc_args=postproc_args,
            )

            trace_headers = (None if raw_request is None else
                             tracing.extract_trace_headers(raw_request.headers))

            resolve_request_conversation_id(
                request, None if raw_request is None else raw_request.headers)
            conversation_params = to_llm_conversation_params(
                request.conversation_params)
            scheduling_params = SchedulingParams(
                agent_hierarchy=request.agent_hierarchy)

            generate_inputs = prompt
            preprocess_fn = getattr(self.generator, "preprocess", None)
            if preprocess_fn is not None:
                loop = asyncio.get_event_loop()
                generate_inputs = await loop.run_in_executor(
                    self._input_proc_executor,
                    functools.partial(preprocess_fn, prompt, sampling_params,
                                      disaggregated_params))

            promise = self.generator.generate_async(
                inputs=generate_inputs,
                sampling_params=sampling_params,
                _postproc_params=postproc_params
                if self.postproc_worker_enabled else None,
                streaming=request.stream,
                lora_request=request.lora_request,
                disaggregated_params=disaggregated_params,
                conversation_params=conversation_params,
                cache_salt=request.cache_salt,
                trace_headers=trace_headers,
                scheduling_params=scheduling_params,
            )
            asyncio.create_task(self.await_disconnected(raw_request, promise))
            if not self.postproc_worker_enabled:
                postproc_args.tokenizer = self.tokenizer
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

            if request.stream:
                response_generator = chat_stream_generator(
                    promise, postproc_params)
                return StreamingResponse(content=response_generator,
                                         media_type="text/event-stream")
            else:
                response = await self._create_chat_response(
                    promise, postproc_params, raw_request, disaggregated_params)
                # Context-only: optionally return prompt_token_ids as a base64
                # int32 buffer (one string) so the disagg orchestrator relays it
                # without materializing the int list on its event loop. The encode
                # runs here on the context worker (1-of-N), not the orchestrator.
                if (request.disaggregated_params is not None and
                        request.disaggregated_params.return_prompt_token_ids_b64
                        and response.prompt_token_ids is not None):
                    import numpy as np
                    response.prompt_token_ids_b64 = base64.b64encode(
                        np.asarray(response.prompt_token_ids,
                                   dtype=np.int32).tobytes()).decode("ascii")
                    response.prompt_token_ids = None
                return JSONResponse(content=response.model_dump())
        except CppExecutorError:
            logger.error(traceback.format_exc())
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except ValueError as e:
            return self.create_error_response(str(e))
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(str(e))

    async def openai_mm_encoder(self, request: ChatCompletionRequest,
                                raw_request: Request) -> Response:

        async def create_mm_embedding_response(promise: RequestOutput):
            await promise.aresult()
            # TODO: Replace mm_embedding_handles with a dedicated OpenAIBaseModel(JSON-safe), when enable multimodal disagg E2E
            mm_embedding_handles = (
                promise.disaggregated_params.multimodal_embedding_handles
                if promise.disaggregated_params else None)
            if not mm_embedding_handles:
                return self.create_error_response(
                    message="Multimodal embedding handle missing in response",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
            if any("tensor_size" not in h for h in mm_embedding_handles):
                return self.create_error_response(
                    message="Multimodal embedding handle missing tensor_size",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
            mm_embedding_handle = (mm_embedding_handles[0]
                                   if len(mm_embedding_handles) == 1 else
                                   mm_embedding_handles)
            num_tokens = sum(
                int(h["tensor_size"][0]) for h in mm_embedding_handles)
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
            ensure_request_chat_template_allowed(
                request, self.allow_request_chat_template)
            conversation: List[ConversationMessage] = []
            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]

            try:
                conversation, mm_coroutines, mm_placeholder_counts, mm_item_order = parse_chat_messages_coroutines(
                    request.messages,
                    self.model_config,
                    self.multimodal_server_config,
                    request_media_io_kwargs=request.media_io_kwargs)
            except ValidationError:
                # ValidatorIterator rejects extra fields; fall back to raw JSON.
                raw_body = await raw_request.json()
                raw_messages = raw_body.get("messages", [])
                conversation, mm_coroutines, mm_placeholder_counts, mm_item_order = parse_chat_messages_coroutines(
                    raw_messages,
                    self.model_config,
                    self.multimodal_server_config,
                    request_media_io_kwargs=request.media_io_kwargs)

            if request.prompt_token_ids is not None:
                prompt = request.prompt_token_ids
            else:
                prompt_task = async_apply_chat_template(
                    model_type=resolve_top_level_model_type(self.model_config),
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
                prompt, (mm_data, mm_embeddings) = await asyncio.gather(
                    prompt_task, mm_coroutines)
            prompt = prompt_inputs(prompt)

            if request.prompt_token_ids is not None:
                mm_data, mm_embeddings = await mm_coroutines
            if mm_embeddings:
                raise ValueError("Cannot use multimodal embeddings as input")
            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data
                if mm_item_order:
                    prompt["mm_item_order"] = mm_item_order

            promise = self.generator.generate_async(inputs=prompt, )
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

    async def openai_completion(self, request: CompletionRequest,
                                raw_request: Request) -> Response:

        async def completion_response(
                promise: RequestOutput,
                postproc_params: Optional[PostprocParams]
        ) -> CompletionResponse:
            response = await promise
            if response.error is not None:
                raise RuntimeError(f"Generation failed: {response.error}")
            if not self.postproc_worker_enabled:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                pp_result = post_processor(response, args)
            else:
                pp_result = response.outputs[0]._postprocess_result
            if disaggregated_params and disaggregated_params.request_type and disaggregated_params.request_type == "context_only":
                # Include prompt token ids for context-only requests
                pp_result.prompt_token_ids = response.prompt_token_ids
            await self._extract_metrics(response, raw_request)
            return pp_result

        def merge_completion_responses(
                responses: List[CompletionResponse]) -> CompletionResponse:
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
                    cached_tokens=num_cached_tokens, ),
            )
            merged_rsp = CompletionResponse(
                model=self.model,
                choices=all_choices,
                usage=usage_info,
                prompt_token_ids=all_prompt_token_ids,
            )
            return merged_rsp

        async def completion_generator(promise: RequestOutput,
                                       params: Optional[PostprocParams]):
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
            except Exception as e:
                logger.error(traceback.format_exc())
                # StreamingResponse commits HTTP 200 before the first
                # chunk, so we cannot change the status code.  Yield
                # an SSE error event so the stream terminates cleanly
                # instead of breaking the HTTP connection.
                error_data = json.dumps({
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": None,
                        "param": None,
                    }
                })
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"

        async def merge_generators(generators: List[AsyncIterator[Any]]):
            result_queue = asyncio.Queue()
            finished = [False] * len(generators)

            async def producer(generator: AsyncIterator[Any], idx: int):
                async for output in generator:
                    await result_queue.put(output)
                finished[idx] = True

            tasks = [
                asyncio.create_task(producer(generator, idx))
                for idx, generator in enumerate(generators)
            ]

            while not all(finished) or not result_queue.empty():
                output = await result_queue.get()
                yield output
            await asyncio.gather(*tasks)

        async def generator_wrapper(generator: AsyncIterator[Any]):
            first_response = await anext(generator)
            raw_request.state.server_first_token_time = get_steady_clock_now_in_seconds(
            )
            yield first_response
            async for output in generator:
                yield output
            yield "data: [DONE]\n\n"

        try:
            if isinstance(request.prompt, str) or \
                (isinstance(request.prompt, list) and isinstance(request.prompt[0], int)):
                prompts = [request.prompt]
            else:
                prompts = request.prompt

            stream_response_id = None
            stream_created = None
            if request.stream and len(prompts) > 1:
                stream_response_id = f"cmpl-{uuid.uuid4().hex}"
                stream_created = int(time.time())

            promises: List[RequestOutput] = []
            postproc_params_collection: List[Optional[PostprocParams]] = []
            # Pass the model vocabulary size so ``logit_bias`` can be
            # expanded into an embedding bias tensor in the sampler.
            sampling_params = request.to_sampling_params(
                vocab_size=self._logit_bias_vocab_size(),
                gather_generation_logits=self.generator.args.
                gather_generation_logits,
                backend=self.generator.args.backend)
            add_thinking_budget_logits_processor(
                sampling_params,
                reasoning_parser=self.generator.args.reasoning_parser,
                tokenizer=self.tokenizer,
            )
            # TODO: better way to enable metrics
            if len(os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", "")) > 0:
                sampling_params.return_perf_metrics = True
            disaggregated_params = to_llm_disaggregated_params(
                request.disaggregated_params)
            resolve_request_conversation_id(
                request, None if raw_request is None else raw_request.headers)
            conversation_params = to_llm_conversation_params(
                request.conversation_params)
            for idx, prompt in enumerate(prompts):
                postproc_args = CompletionPostprocArgs.from_request(request)
                postproc_args.prompt_idx = idx
                postproc_args.stream_response_id = stream_response_id
                postproc_args.stream_created = stream_created
                if request.echo:
                    postproc_args.prompt = prompt
                postproc_params = PostprocParams(
                    post_processor=completion_stream_post_processor
                    if request.stream else completion_response_post_processor,
                    postproc_args=postproc_args,
                )
                trace_headers = (None if raw_request is None else
                                 tracing.extract_trace_headers(
                                     raw_request.headers))

                prompt = prompt_inputs(prompt)
                if prompt.get("prompt") is not None:
                    loop = asyncio.get_event_loop()
                    prompt_token_ids, extra_processed_inputs = await loop.run_in_executor(
                        self._input_proc_executor,
                        functools.partial(self.generator.input_processor,
                                          prompt, sampling_params))
                    tokens_prompt = TokensPrompt(
                        prompt_token_ids=prompt_token_ids,
                        query_token_ids=extra_processed_inputs.get(
                            "query_token_ids")
                        if extra_processed_inputs is not None else None)
                else:
                    tokens_prompt = prompt

                promise = self.generator.generate_async(
                    inputs=tokens_prompt,
                    sampling_params=sampling_params,
                    _postproc_params=postproc_params,
                    streaming=request.stream,
                    lora_request=request.lora_request,
                    disaggregated_params=disaggregated_params,
                    conversation_params=conversation_params,
                    trace_headers=trace_headers)
                asyncio.create_task(
                    self.await_disconnected(raw_request, promise))
                if not self.postproc_worker_enabled:
                    postproc_args.tokenizer = self.tokenizer
                    postproc_args.num_prompt_tokens = len(
                        promise.prompt_token_ids)
                promises.append(promise)
                postproc_params_collection.append(
                    None if self.postproc_worker_enabled else postproc_params)

            if request.stream:
                generators = [
                    completion_generator(promise, params) for promise, params in
                    zip(promises, postproc_params_collection)
                ]
                response_generator = merge_generators(generators) if len(
                    promises) > 1 else generators[0]
                return StreamingResponse(
                    content=generator_wrapper(response_generator),
                    media_type="text/event-stream")
            else:
                rsps = await asyncio.gather(*[
                    completion_response(promise, params) for promise, params in
                    zip(promises, postproc_params_collection)
                ])
                response = merge_completion_responses(rsps) if len(
                    rsps) > 1 else rsps[0]
                return JSONResponse(content=response.model_dump())
        except CppExecutorError:
            logger.error(traceback.format_exc())
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(str(e))

    async def chat_harmony(self, request: ChatCompletionRequest,
                           raw_request: Request) -> Response:
        """Chat Completion API with harmony format support.

        Supports both streaming and non-streaming modes.
        """

        async def create_streaming_generator(promise: RequestOutput,
                                             postproc_params: PostprocParams):
            try:
                if not self.postproc_worker_enabled:
                    post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                # Stamp first-token time on the first response, then append a
                # /perf_metrics entry after [DONE]. The deque is only
                # populated inside _extract_metrics.
                first_response = await anext(promise)
                raw_request.state.server_first_token_time = (
                    get_steady_clock_now_in_seconds())
                pp_results = (first_response.outputs[0]._postprocess_result if
                              self.postproc_worker_enabled else post_processor(
                                  first_response, args))
                for pp_res in pp_results:
                    yield pp_res
                res = first_response
                async for res in promise:
                    pp_results = (res.outputs[0]._postprocess_result
                                  if self.postproc_worker_enabled else
                                  post_processor(res, args))
                    for pp_res in pp_results:
                        yield pp_res
                yield "data: [DONE]\n\n"
                await self._extract_metrics(res, raw_request)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(traceback.format_exc())
                raise

        try:
            ensure_request_chat_template_allowed(
                request, self.allow_request_chat_template)
            # Initialize HarmonyAdapter
            # NOTE: WAR for Disagg failure, may affect perf if no warmup
            if not self.harmony_adapter:
                self.harmony_adapter = get_harmony_adapter()

            # Reuse pre-tokenized harmony tokens when forwarded by an upstream
            # context worker (disaggregated serving). Otherwise, run the
            # Harmony adapter on the request messages.
            try:
                harmony_tokens = tokenize_harmony_chat_request(
                    request, harmony_adapter=self.harmony_adapter)
            except Exception:
                logger.error("messages_dict: %s", request.messages)
                logger.error("tools: %s", request.tools)
                logger.error("request: %s", request)
                raise

            # Get harmony stop tokens
            harmony_stop_tokens = self.harmony_adapter.get_stop_tokens()
            if request.stop_token_ids:
                request.stop_token_ids.extend(harmony_stop_tokens)
            else:
                request.stop_token_ids = harmony_stop_tokens

            sampling_params = request.to_sampling_params(
                vocab_size=self._logit_bias_vocab_size(),
                reasoning_parser="gpt_oss")
            if sampling_params.thinking_token_budget is not None:
                raise ValueError(
                    "thinking_token_budget is not supported by the Harmony "
                    "GPT-OSS serving path; use reasoning_effort instead")
            sampling_params.detokenize = False  # Harmony adapter handles detokenization
            # Enable per-request perf metrics when the env var is set.
            # Otherwise the /perf_metrics deque stays empty on this path.
            if len(os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", "")) > 0:
                sampling_params.return_perf_metrics = True
            disaggregated_params = to_llm_disaggregated_params(
                request.disaggregated_params)
            trace_headers = (None if raw_request is None else
                             tracing.extract_trace_headers(raw_request.headers))

            postproc_args = ChatCompletionPostprocArgs.from_request(request)
            postproc_params = PostprocParams(
                post_processor=chat_harmony_streaming_post_processor
                if request.stream else chat_harmony_post_processor,
                postproc_args=postproc_args,
            )

            resolve_request_conversation_id(
                request, None if raw_request is None else raw_request.headers)
            conversation_params = to_llm_conversation_params(
                request.conversation_params)
            scheduling_params = SchedulingParams(
                agent_hierarchy=request.agent_hierarchy)

            # Generate
            promise = self.generator.generate_async(
                inputs=harmony_tokens,
                sampling_params=sampling_params,
                _postproc_params=postproc_params
                if self.postproc_worker_enabled else None,
                streaming=bool(request.stream),
                lora_request=request.lora_request,
                scheduling_params=scheduling_params,
                disaggregated_params=disaggregated_params,
                conversation_params=conversation_params,
                trace_headers=trace_headers,
            )
            if not self.postproc_worker_enabled:
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

            # Disconnect cancellation
            asyncio.create_task(self.await_disconnected(raw_request, promise))

            # Handle streaming
            if request.stream:
                return StreamingResponse(content=create_streaming_generator(
                    promise, postproc_params),
                                         media_type="text/event-stream")
            else:
                response = await self._create_chat_response(
                    promise, postproc_params, raw_request, disaggregated_params)
                return JSONResponse(response.model_dump())

        except ValueError as e:
            return self.create_error_response(str(e))
        except Exception as e:
            logger.error("Error in harmony chat completion: %s", e)
            logger.debug("Error details: %s", traceback.format_exc())
            return self.create_error_response(message=str(e),
                                              err_type="internal_error")

    async def openai_responses(self, request: ResponsesRequest,
                               raw_request: Request) -> Response:

        async def create_response(
                promise: RequestOutput,
                postproc_params: PostprocParams) -> ResponsesResponse:
            await promise.aresult()
            if self.postproc_worker_enabled:
                response = promise.outputs[0]._postprocess_result
            else:
                args = postproc_params.postproc_args
                response = await responses_api_create_response(
                    generator=promise,
                    request=request,
                    sampling_params=args.sampling_params,
                    model_name=self.model,
                    conversation_store=self.conversation_store,
                    generation_result=None,
                    enable_store=self.enable_store and request.store,
                    use_harmony=self.use_harmony,
                    reasoning_parser=args.reasoning_parser,
                    tool_parser=args.tool_parser,
                )

            return response

        async def create_streaming_generator(promise: RequestOutput,
                                             postproc_params: PostprocParams):
            post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
            streaming_processor = args.streaming_processor
            initial_responses = streaming_processor.get_initial_responses()
            for initial_response in initial_responses:
                yield initial_response

            async for res in promise:
                pp_results = res.outputs[
                    0]._postprocess_result if self.postproc_worker_enabled else post_processor(
                        res, args)
                for pp_res in pp_results:
                    yield pp_res

        try:
            if request.background:
                logger.warning(
                    "Request.background is not supported yet, will fallback to foreground processing."
                )

            # Reject rather than silently ignore: with storage disabled
            # (TRTLLM_RESPONSES_API_DISABLE_STORE, postproc workers, or
            # multi-frontend serving) the previous response can never be
            # resolved.
            if request.previous_response_id is not None and not self.enable_store:
                return self.create_error_response(
                    err_type="InvalidRequestError",
                    message=("'previous_response_id' requires response "
                             "storage, which is disabled on this server."),
                )

            # Get prev response
            prev_response = None
            if self.enable_store:
                prev_response_id = request.previous_response_id
                if prev_response_id is not None:
                    if not prev_response_id.startswith("resp_"):
                        return self._create_invalid_response_id_error(
                            prev_response_id)

                    prev_response = await self.conversation_store.load_response(
                        prev_response_id)
                    if prev_response is None:
                        logger.debug(
                            f"response_id {prev_response_id} not found")
                        return self._create_response_id_not_found_error(
                            prev_response_id)

            input_tokens, sampling_params = await responses_api_request_preprocess(
                request=request,
                prev_response=prev_response,
                conversation_store=self.conversation_store,
                enable_store=self.enable_store and request.store,
                use_harmony=self.use_harmony,
                tokenizer=self.tokenizer if not self.use_harmony else None,
                model_config=self.model_config
                if not self.use_harmony else None,
                processor=self.processor if not self.use_harmony else None,
                reasoning_parser=self.generator.args.reasoning_parser
                if not self.use_harmony else "gpt_oss",
            )

            streaming_processor = None
            if request.stream:
                # Per-request streaming processor
                streaming_processor = ResponsesStreamingProcessor(
                    request=request,
                    sampling_params=sampling_params,
                    model_name=self.model,
                    conversation_store=self.conversation_store,
                    enable_store=self.enable_store and request.store,
                    use_harmony=self.use_harmony,
                    reasoning_parser=self.generator.args.reasoning_parser,
                    tool_parser=self.tool_parser,
                )

            postproc_args = ResponsesAPIPostprocArgs(
                model=self.model,
                request=request,
                sampling_params=sampling_params,
                use_harmony=self.use_harmony,
                reasoning_parser=self.generator.args.reasoning_parser,
                tool_parser=self.tool_parser,
                streaming_processor=streaming_processor,
            )
            postproc_params = PostprocParams(
                post_processor=responses_api_streaming_post_processor
                if request.stream else responses_api_post_processor,
                postproc_args=postproc_args,
            )
            promise = self.generator.generate_async(
                inputs=input_tokens,
                sampling_params=sampling_params,
                streaming=request.stream,
                _postproc_params=postproc_params
                if self.postproc_worker_enabled else None,
            )

            if self.postproc_worker_enabled and request.store:
                logger.warning(
                    "Postproc workers are enabled, request will not be stored!")

            asyncio.create_task(self.await_disconnected(raw_request, promise))

            if request.stream:
                return StreamingResponse(content=create_streaming_generator(
                    promise, postproc_params),
                                         media_type="text/event-stream")
            else:
                response = await create_response(promise, postproc_params)
                return JSONResponse(content=response.model_dump())
        except CppExecutorError:
            logger.error(traceback.format_exc())
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(str(e))

        return JSONResponse(content={"detail": "None"})

    async def openai_responses_get_response(self,
                                            response_id: str) -> JSONResponse:
        logger.info(f"Getting response: {response_id}")
        if not self.enable_store:
            return self.create_error_response(
                message="Response storage is disabled",
                err_type="InvalidRequestError")

        if not response_id.startswith("resp_"):
            return self._create_invalid_response_id_error(response_id)

        response = await self.conversation_store.load_response(response_id)
        if response is None:
            return self._create_response_id_not_found_error(response_id)

        return JSONResponse(content=response.model_dump())

    async def openai_responses_delete_response(
            self, response_id: str) -> JSONResponse:
        logger.info(f"Deleting response: {response_id}")
        if not self.enable_store:
            return self.create_error_response(
                message="Response storage is disabled",
                err_type="InvalidRequestError")

        if not response_id.startswith("resp_"):
            return self._create_invalid_response_id_error(response_id)

        success = await self.conversation_store.pop_response(response_id)
        if not success:
            return self._create_response_id_not_found_error(response_id)

        return JSONResponse(content={
            "id": response_id,
            "object": "response",
            "deleted": True
        })

    async def tokenize(self, request: TokenizeRequest) -> JSONResponse:
        try:
            token_ids = self.tokenizer.encode(request.prompt)
            response = TokenizeResponse(count=len(token_ids), tokens=token_ids)
            return JSONResponse(content=response.model_dump())
        except Exception as e:
            return self.create_error_response(
                message=str(e),
                err_type="InvalidRequestError",
                status_code=HTTPStatus.BAD_REQUEST)

    async def release_memory(self,
                             request: MemoryUpdateRequest) -> JSONResponse:
        assert isinstance(
            self.generator, AsyncLLM
        ), "/release_memory endpoint is only supported with AsyncLLM()"
        await self.generator.collective_rpc('sleep', args=(request.tags, ))
        return JSONResponse(content={"status": "success"})

    async def resume_memory(self, request: MemoryUpdateRequest) -> JSONResponse:
        assert isinstance(
            self.generator, AsyncLLM
        ), "/resume_memory endpoint is only supported with AsyncLLM()"
        await self.generator.collective_rpc('wakeup', args=(request.tags, ))
        return JSONResponse(content={"status": "success"})

    async def update_weights(self,
                             request: UpdateWeightsRequest) -> JSONResponse:
        assert isinstance(
            self.generator, AsyncLLM
        ), "/update_weights endpoint is only supported with AsyncLLM()"
        await self.generator.collective_rpc('update_weights',
                                            args=(request.weights, ))
        return JSONResponse(content={"status": "success"})

    async def get_server_info(self) -> JSONResponse:
        content = {"disaggregated_params": self.generator.disaggregated_params}
        args = getattr(self.generator, "args", None)
        if args is not None:
            if args.max_batch_size is not None:
                content["max_batch_size"] = args.max_batch_size
            kv_cache_config = args.kv_cache_config
            if kv_cache_config is not None:
                content[
                    "kv_cache_hash_algo"] = get_effective_kv_cache_event_hash_algo(
                        kv_cache_config.kv_cache_event_hash_algo,
                        kv_cache_config.use_kv_cache_manager_v2)
                if kv_cache_config.tokens_per_block is not None:
                    content[
                        "tokens_per_block"] = kv_cache_config.tokens_per_block
        return JSONResponse(content=content)

    async def openai_image_generation(self, request: ImageGenerationRequest,
                                      raw_request: Request) -> Response:
        """OpenAI-compatible image generation endpoint.

        Follows the OpenAI Images API specification for image generation,
        with ``request.format`` extended to accept tensor payloads
        (``"safetensors"``/``"pt"``) alongside the PNG/WebP/JPEG encoders.
        """
        try:
            image_id = f"image_{uuid.uuid4().hex}"

            # Client-side ValueErrors from request translation and
            # parameter validation are 400. Serialization failures below
            # (server-side: missing media, inconsistent batch) fall
            # through to the outer ``except Exception`` → 500 so the
            # client doesn't get blamed for a server-internal failure.
            try:
                params = parse_visual_gen_params(request, image_id,
                                                 self.generator)
                logger.info(
                    f"Generating image: {image_id} with params: {params} and prompt: {request.prompt}"
                )
                image_gen_start = time.perf_counter()
                output = self.generator.generate(inputs=request.prompt,
                                                 params=params)
            except ValueError as exc:
                logger.error(f"Image request error: {exc}")
                return self.create_error_response(
                    message=str(exc),
                    status_code=HTTPStatus.BAD_REQUEST,
                )

            if output.image is None:
                return self.create_error_response(
                    message="Image generation failed",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            if is_tensor_format(request.format):
                # Tensor payloads carry every populated modality in a
                # single file. Match the image-encoder fan-out by
                # emitting one ``ImageObject`` per batch item.
                from tensorrt_llm.media.tensor_payload import infer_batch_size

                ext = f".{request.format}"
                batch_size = infer_batch_size(output)
                if request.response_format == "b64_json":
                    data = [
                        ImageObject(
                            b64_json=base64.b64encode(
                                output._save_bytes(
                                    request.format,
                                    batch_index=i)).decode("utf-8"),
                            revised_prompt=request.prompt,
                        ) for i in range(batch_size)
                    ]
                else:
                    paths_in = [
                        self.media_storage_path / f"{image_id}_{i}{ext}"
                        for i in range(batch_size)
                    ]
                    output.save(paths_in, format=request.format)
                    data = [
                        ImageObject(
                            url=self._build_image_content_url(
                                raw_request, image_id, i),
                            revised_prompt=request.prompt,
                        ) for i in range(batch_size)
                    ]
                response = ImageGenerationResponse(
                    created=int(time.time()),
                    data=data,
                    output_format=request.format,
                    size=f"{params.width}x{params.height}",
                )
            else:
                output_images = _normalize_image_output(output.image)
                # Pillow's format name is the upper-case form of our
                # request token. The on-disk extension matches the
                # request token directly; ``.jpeg`` is interchangeable
                # with ``.jpg`` for Pillow, the OS, and the OpenAI API.
                pil_format = request.format.upper()
                ext = f".{request.format}"
                if request.response_format == "b64_json":
                    data = [
                        ImageObject(
                            b64_json=base64.b64encode(
                                image_to_bytes(
                                    image, format=pil_format)).decode("utf-8"),
                            revised_prompt=request.prompt,
                        ) for image in output_images
                    ]
                else:
                    data = []
                    for i, image in enumerate(output_images):
                        path = self.media_storage_path / f"{image_id}_{i}{ext}"
                        path.write_bytes(
                            image_to_bytes(image, format=pil_format))
                        data.append(
                            ImageObject(
                                url=self._build_image_content_url(
                                    raw_request, image_id, i),
                                revised_prompt=request.prompt,
                            ))
                response = ImageGenerationResponse(
                    created=int(time.time()),
                    data=data,
                    output_format=request.format,
                    size=f"{params.width}x{params.height}",
                )

            latency = time.perf_counter() - image_gen_start  # seconds
            metrics = output.metrics
            generation = metrics.generation if metrics is not None else 0.0
            denoise = metrics.denoise if metrics is not None else 0.0
            logger.info(f"Image {image_id} generated and encoded: "
                        f"latency={latency:.3f}s generation={generation:.3f}s "
                        f"denoise={denoise:.3f}s")
            headers = build_visual_gen_timing_headers(metrics)

            return JSONResponse(content=response.model_dump(), headers=headers)

        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(
                message=str(e),
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    async def get_image_content(
        self,
        image_id: str,
        raw_request: Request,
        i: int = 0,
    ) -> Response:
        """Serve a generated image by ID and batch index.

        ``GET /v1/images/{image_id}/content?i=<batch_index>`` returns
        the image file the corresponding ``POST /v1/images/generations``
        wrote into ``media_storage_path``. ``i`` defaults to ``0`` for
        single-image requests; URL responses for ``n > 1`` requests
        carry ``?i=N`` per item.
        """
        for ext in (".png", ".webp", ".jpg", ".jpeg", ".safetensors", ".pt"):
            candidate = self.media_storage_path / f"{image_id}_{i}{ext}"
            if candidate.exists():
                media_type = ("application/octet-stream"
                              if ext in (".safetensors", ".pt") else
                              f"image/{ext.lstrip('.').replace('jpg', 'jpeg')}")
                return FileResponse(
                    str(candidate),
                    media_type=media_type,
                    filename=candidate.name,
                )
        return self.create_error_response(
            message=f"Image {image_id!r} (batch index {i}) not found",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )

    @staticmethod
    def _build_image_content_url(raw_request: Request, image_id: str,
                                 i: int) -> str:
        """Return a fetchable HTTP URL for a generated image item."""
        base = str(raw_request.base_url).rstrip("/")
        return f"{base}/v1/images/{image_id}/content?i={i}"

    async def openai_image_edit(self, raw_request: Request) -> Response:
        """OpenAI-compatible image editing endpoint — returns HTTP 501.

        No in-tree pipeline implements image editing today: Flux/Flux2 are
        text-to-image only and ignore ``params.image``; Wan and LTX-2 produce
        video, not edited images. The route is registered so callers get an
        honest NotImplemented signal instead of a 404. The request body is
        not parsed because no schema is committed for this endpoint yet —
        bring a typed request model back when an edit-capable pipeline lands.
        """
        return self._create_not_supported_error(
            "Image editing is not supported by any in-tree pipeline yet.")

    async def __call__(self,
                       host,
                       port,
                       sockets: list[socket.socket] | None = None):
        # Store the binding address for server registration
        self.binding_addr = f"http://{host}:{port}"
        self.host = host
        self.port = port
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        server = uvicorn.Server(config)

        async def _register_after_serving():
            while not server.started:
                await asyncio.sleep(0.1)
            if self.disagg_cluster_worker:
                try:
                    await self.disagg_cluster_worker.register_worker()
                except Exception as e:
                    logger.error(f"Worker registration failed: {e}")
                    server.should_exit = True

        asyncio.create_task(_register_after_serving())
        await server.serve(sockets=sockets)
