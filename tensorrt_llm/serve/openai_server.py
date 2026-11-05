#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import dataclasses
import functools
import importlib
import os
import signal
import socket
import sys
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from transformers import AutoProcessor

from tensorrt_llm._startup import _StartupTimer
from tensorrt_llm._torch.async_llm import AsyncLLM
from tensorrt_llm._utils import EnergyMonitor
# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.inputs.media_io import (BaseMediaIO,
                                          log_allowed_local_media_path)
from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor
from tensorrt_llm.llmapi import MultimodalEncoder
from tensorrt_llm.llmapi.disagg_utils import (DisaggClusterConfig,
                                              MetadataServerConfig, ServerRole)
from tensorrt_llm.llmapi.llm import LLM, RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.metrics.collector import MetricsCollector
from tensorrt_llm.serve._base.app_builder import build_http_app, serve_uvicorn
from tensorrt_llm.serve._base.msgspec_route import _MsgspecRoute
from tensorrt_llm.serve._base.routing import (RoleGroup, ServerCapabilities,
                                              attach_routers, role_group_of)
from tensorrt_llm.serve._base.serving import (
    BaseServing, GenerateBaseServing, _configure_parser_special_token_decoding,
    _record_generator_termination)
from tensorrt_llm.serve._perf_metrics_config import resolve_perf_metrics_config
from tensorrt_llm.serve.api.anthropic.adapter import anthropic_error_response
from tensorrt_llm.serve.api.anthropic.serving import (
    AnthropicBatchesServing, AnthropicCountTokensServing,
    AnthropicMessagesServing)
from tensorrt_llm.serve.api.openai.embeddings.batcher import EncodeBatcher
from tensorrt_llm.serve.api.ops.system.serving import SystemServing
from tensorrt_llm.serve.chat_utils import load_chat_template
from tensorrt_llm.serve.cluster_storage import create_cluster_storage_client
from tensorrt_llm.serve.disagg_auto_scaling import DisaggClusterWorker
from tensorrt_llm.serve.metadata_server import create_metadata_server
from tensorrt_llm.serve.middleware.register import register_middlewares
from tensorrt_llm.serve.multi_frontend import (FORWARDED_ROUTES,
                                               AttachedFrontendWatchdog,
                                               LauncherForwarder,
                                               MultiFrontendServing,
                                               SharedClockOffset,
                                               follow_clock_offset)
from tensorrt_llm.serve.openai_protocol import (ResponsesRequest,
                                                ResponsesResponse,
                                                to_llm_conversation_params,
                                                to_llm_disaggregated_params)
from tensorrt_llm.serve.perf_metrics import PerfMetricsJsonlWriter
from tensorrt_llm.serve.postprocess_handlers import (
    PostprocStrategy, ResponsesAPIPostprocArgs, responses_api_post_processor,
    responses_api_streaming_post_processor, select_postproc_strategy)
from tensorrt_llm.serve.render import RenderResources, mount_render_endpoints
from tensorrt_llm.serve.responses_utils import (ClosingStreamingResponse,
                                                ConversationHistoryStore,
                                                ResponsesStreamingProcessor)
from tensorrt_llm.serve.responses_utils import \
    create_response as responses_api_create_response
from tensorrt_llm.serve.responses_utils import (guard_responses_stream,
                                                keep_responses_stream_alive)
from tensorrt_llm.serve.responses_utils import \
    request_preprocess as responses_api_request_preprocess
from tensorrt_llm.serve.responses_web_search import web_search_rejection_reason
from tensorrt_llm.serve.runtime_control_auth import RuntimeControlReplayCache
from tensorrt_llm.serve.serving_extensions import load_builtin_extensions
from tensorrt_llm.version import __version__ as VERSION

from .._utils import AdjustedSteadyClock, set_prometheus_multiproc_dir
from .harmony_adapter import HarmonyAdapter, get_harmony_adapter

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen import VisualGen


def _is_visual_gen_instance(obj) -> bool:
    """isinstance(obj, VisualGen) without importing the visual_gen tree.

    Probes the module that defines the class, not the (lazily loaded)
    package: a process that only imported config leaves like
    tensorrt_llm.visual_gen.args has the package in sys.modules, but
    reading VisualGen off it would import the whole runtime tree via
    __getattr__. If the defining module was never imported, obj cannot
    be a VisualGen instance, so the LLM serving path never pays the
    visual_gen import cost.
    """
    visual_gen = sys.modules.get("tensorrt_llm.visual_gen.visual_gen")
    return visual_gen is not None and isinstance(obj, visual_gen.VisualGen)

# yapf: enable

TIMEOUT_KEEP_ALIVE = 5  # seconds.


def resolve_spec_decode_num_spec_tokens(args: Any) -> Optional[int]:
    """Fixed per-step draft bound to report, or None when there is not one.

    Emitted as ``num_spec_tokens`` and used to size the acceptance histogram, so
    a wrong answer here makes every histogram the wrong width. Returns None both
    when speculative decoding is off and when ``draft_len_schedule`` is set --
    the bound genuinely varies by batch size there, and None is the honest
    answer rather than reporting whichever value happened to be configured.
    """
    spec_config = getattr(args, "speculative_config", None) if args else None
    if spec_config is None or getattr(spec_config, "draft_len_schedule", None):
        return None
    return getattr(spec_config, "max_draft_len", None)


class OpenAIServer:

    def __init__(
            self,
            generator: Union[LLM, MultimodalEncoder, "VisualGen"],
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
            media_load_workers: int = 8,
            internal_disagg_auth_key: Optional[str] = None,
            enable_rl_control_endpoints: bool = False,
            rl_control_api_key: Optional[str] = None,
            enable_runtime_control_endpoints: bool = False,
            runtime_control_api_key: Optional[str] = None,
            multi_frontend_serving: Optional[MultiFrontendServing] = None):
        if enable_rl_control_endpoints:
            logger.warning(
                "RL control endpoints are deprecated; use generic runtime "
                "control endpoints for memory checkpointing.")
        if enable_rl_control_endpoints and enable_runtime_control_endpoints:
            raise ValueError(
                "Legacy RL control and runtime control endpoints cannot both "
                "be enabled.")
        if enable_rl_control_endpoints and not rl_control_api_key:
            raise ValueError(
                "rl_control_api_key is required when RL control endpoints are enabled"
            )
        if enable_rl_control_endpoints and not isinstance(generator, AsyncLLM):
            raise ValueError("RL control endpoints require AsyncLLM")
        if (enable_runtime_control_endpoints
                and server_role is ServerRole.EMBEDDING):
            raise ValueError(
                "Runtime control endpoints are not supported for embedding servers"
            )
        if enable_runtime_control_endpoints and not runtime_control_api_key:
            raise ValueError(
                "runtime_control_api_key is required when runtime control "
                "endpoints are enabled")
        if enable_runtime_control_endpoints and not isinstance(generator, LLM):
            raise ValueError("Runtime control endpoints require LLM")
        if (enable_runtime_control_endpoints
                and generator.args.sleep_config is None):
            raise ValueError("Runtime control endpoints require sleep_config")
        if (enable_runtime_control_endpoints
                and generator.args.num_serve_frontends > 1):
            raise ValueError(
                "Runtime control endpoints require num_serve_frontends=1 "
                "because replay protection is local to each frontend.")

        # A broken built-in extension module must fail the server start, not
        # the first request that consults it.
        load_builtin_extensions()
        self.generator = generator
        self._is_visual_gen = _is_visual_gen_instance(generator)
        self._postproc = select_postproc_strategy(self.postproc_worker_enabled)
        self._embedding_max_queue_delay = embedding_max_queue_delay
        self._embedding_max_queue_size = embedding_max_queue_size
        self.embedding_batcher: Optional[EncodeBatcher] = None
        self.tool_parser = tool_parser
        # Attached frontends serve the launcher's port: only it registers.
        self._registers_server = (multi_frontend_serving is None
                                  or multi_frontend_serving.is_launcher)
        self.metadata_server = (create_metadata_server(metadata_server_cfg)
                                if self._registers_server else None)
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
        self._internal_disagg_auth_key = internal_disagg_auth_key
        self._enable_rl_control_endpoints = enable_rl_control_endpoints
        self._rl_control_api_key = rl_control_api_key
        self._enable_runtime_control_endpoints = enable_runtime_control_endpoints
        self._runtime_control_api_key = runtime_control_api_key
        self._runtime_control_replay_cache = RuntimeControlReplayCache()
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
        log_allowed_local_media_path()
        # /metrics reads iteration stats with a blocking call that waits up to
        # 2 s. A dedicated thread keeps those waits off the default executor,
        # which request handling also uses; get_iteration_stats runs at most
        # one read on it at a time.
        self._metrics_stats_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="trtllm_metrics_stats",
        )
        self._metrics_stats_read: Optional[asyncio.Future] = None
        # Whether a scrape is awaiting the last read, and whether a scrape has
        # returned its stats.
        self._metrics_stats_read_waiting = False
        self._metrics_stats_read_consumed = True

        model_dir = Path(model)
        if model_dir.exists() and model_dir.is_dir():
            self.model = model_dir.name
        else:
            self.model = model
        self.metrics_collector = None
        args = getattr(self.generator, "args", None)
        perf_metrics_output_dir = (getattr(args, "perf_metrics_output_dir",
                                           None) if args else None)
        perf_metrics = resolve_perf_metrics_config(
            getattr(args, "return_perf_metrics", False),
            perf_metrics_output_dir)
        self._expose_perf_metrics = perf_metrics.expose_response_metrics
        self._collect_perf_metrics = perf_metrics.collect_request_metrics
        # Per-request spec-decode acceptance stats. Deliberately independent of
        # return_perf_metrics: coupling them would mean asking for acceptance
        # numbers silently mounts the Prometheus endpoint too.
        self._per_request_spec_decode_stats = bool(
            args and getattr(args, "per_request_spec_decode_stats", False))
        self._spec_decode_num_spec_tokens = resolve_spec_decode_num_spec_tokens(
            args)
        # AsyncLLM uses this flag to request engine-level snapshots. Preserve the
        # original value separately because only it controls public headers.
        if self._collect_perf_metrics and args is not None:
            args.return_perf_metrics = True
        server_kind = server_role.name.lower(
        ) if server_role is not None else "server"
        self._perf_metrics_writer = PerfMetricsJsonlWriter(
            perf_metrics_output_dir, server_kind)
        # Multi-frontend serving: see serve/multi_frontend.py.
        self._multi_frontend = multi_frontend_serving
        self._launcher_forwarder: Optional[LauncherForwarder] = None
        self._launcher_watchdog_task: Optional[asyncio.Task] = None
        if multi_frontend_serving is not None and multi_frontend_serving.is_attached:
            self._launcher_forwarder = LauncherForwarder(
                multi_frontend_serving.launcher_uds)
        self._shared_clock_offset: Optional[SharedClockOffset] = (
            SharedClockOffset(
                os.path.dirname(multi_frontend_serving.launcher_uds))
            if multi_frontend_serving is not None else None)
        self._clock_offset_task: Optional[asyncio.Task] = None
        # Requests finishing on attached frontends never wake the launcher's
        # collector, so it also polls.
        self._iteration_stats_poll_interval: Optional[float] = (
            1.0 if multi_frontend_serving is not None
            and multi_frontend_serving.is_launcher else None)
        self._iteration_stats_collector_task = None
        self._iteration_stats_wakeup_event = asyncio.Event()
        # Bounded snapshot of iteration stats for the GET /metrics handler.
        # When the background Prometheus collector loop is active, it is the
        # sole consumer of the engine's stats queue and appends each drained
        # stat here; /metrics then serves from this buffer instead of racing
        # the loop for the queue. Created lazily when the loop starts.
        # See nvbug 6102381.
        self._iteration_stats_buffer: Optional[deque] = None
        # Rank-adjusted clock mapped to the disaggregated server's clock domain.
        self._adjusted_steady_clock = AdjustedSteadyClock()

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
            # Built once and injected into serving objects.
            self._render_resources = RenderResources.from_server(self)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self._perf_metrics_writer.start()
            self._register_server()

            # VisualGen has no args
            if not self._is_visual_gen:
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
                # The PyTorch backend args include `enable_iter_perf_stats`.
                # Attached frontends leave the engine queue to the launcher.
                if self._launcher_forwarder is not None:
                    self._launcher_watchdog_task = asyncio.create_task(
                        AttachedFrontendWatchdog(
                            self._launcher_forwarder,
                            self._multi_frontend.launcher_pid,
                            self._on_launcher_lost).run())
                    self._clock_offset_task = asyncio.create_task(
                        follow_clock_offset(self._shared_clock_offset,
                                            self._apply_shared_clock_offset))
                    logger.info(
                        "Attached frontend: forwarding launcher-owned routes "
                        f"to {self._multi_frontend.launcher_uds}")
                elif self.metrics_collector and getattr(
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

            await self._perf_metrics_writer.close()
            for task in (self._launcher_watchdog_task, self._clock_offset_task):
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            if self._launcher_forwarder is not None:
                await self._launcher_forwarder.close()
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

            await self._deregister_server()
            if self.resource_governor is not None:
                self.resource_governor.close()
            self.generator.shutdown()
            # A stats read still in flight fails fast now that the executor is
            # down; the /metrics handler already treats that as no stats.
            self._metrics_stats_executor.shutdown(wait=False,
                                                  cancel_futures=True)

        async def validation_exception_handler(request, exc):
            if request.url.path.startswith("/v1/messages"):
                # Anthropic clients expect a 400 with the Anthropic error
                # envelope, not FastAPI's 422 shape.
                if self.metrics_collector:
                    self.metrics_collector.log_request_error(http_code=400)
                return anthropic_error_response(str(exc),
                                                "invalid_request_error", 400)
            if self.server_role is ServerRole.VISUAL_GEN:
                return self._visual_gen_api.render_visual_gen_validation_error(
                    exc, self.create_error_response)
            # Non-visual-gen roles keep the shared 400 + ``{"error": ...}``
            # response shape that integration tests (e.g.
            # ``test_malformed_json_request``) and existing clients
            # expect.
            if self.metrics_collector:
                self.metrics_collector.log_request_error(http_code=400)
            # pydantic echoes each failing input verbatim, which can be
            # megabytes of base64; the error locations are what matters.
            detail = str(exc)
            if len(detail) > 4096:
                detail = detail[:4096] + " ... [truncated %d bytes]" % (
                    len(detail) - 4096)
            return JSONResponse(status_code=400, content={"error": detail})

        self.app = build_http_app(
            lifespan=lifespan,
            route_class=_MsgspecRoute,
            dialect_error_renderers=lambda app: app.add_exception_handler(
                RequestValidationError, validation_exception_handler),
            routers=(self._attach_role_routes, ),
            middleware=lambda app: register_middlewares(
                app,
                collect_perf_metrics=self._collect_perf_metrics,
                expose_perf_metrics=self._expose_perf_metrics,
                perf_metrics_writer=self._perf_metrics_writer,
                adjusted_clock=self._adjusted_steady_clock),
        )
        self.app.state.trust_kv_hints = os.getenv("TRTLLM_TRUST_KV_HINTS",
                                                  "0") == "1"

    def _attach_role_routes(self, app: FastAPI) -> None:
        # The register_* methods below add their routes through self.app.
        self.app = app
        # /health, /version and /v1/models for every role. The health
        # probe stays here, with its private accesses, until the public
        # health API exists.
        app.state.ops_serving_system = SystemServing(
            model=self.model, health_probe=self._health_probe)
        if role_group_of(self.server_role) is RoleGroup.LLM:
            # An attached frontend of a multi-frontend server forwards the
            # launcher-owned routes, the Message Batches routes among them.
            # They register before the routers so that they keep winning
            # over the local handlers: FastAPI matches in registration order.
            self._register_launcher_forwarding_routes()
            from tensorrt_llm.serve.api.openai.chat import \
                api_router as chat_router
            from tensorrt_llm.serve.api.openai.chat.serving import ChatServing
            self._chat_serving = ChatServing(
                tool_parser=self.tool_parser,
                tool_call_id_type=self.tool_call_id_type,
                multimodal_server_config=self.multimodal_server_config,
                allow_request_chat_template=self.allow_request_chat_template,
                use_harmony=self.use_harmony,
                harmony_adapter=self.harmony_adapter,
                render_resources=self._render_resources,
                **self._generate_serving_kwargs())
            setattr(app.state, chat_router.APP_STATE_KEY, self._chat_serving)
            messages_serving = AnthropicMessagesServing(self._chat_serving)
            app.state.anthropic_serving_messages = messages_serving
            # Transitional: RenderResources per request until it is built once.
            app.state.anthropic_serving_count_tokens = (
                AnthropicCountTokensServing(
                    render_resources=lambda: RenderResources.from_server(self)))
            app.state.anthropic_serving_batches = AnthropicBatchesServing(
                runner=messages_serving.run_batched_message)

        if self.server_role is ServerRole.VISUAL_GEN:
            assert self._is_visual_gen, \
                "generator must be a VisualGen for VISUAL_GEN server"
            self.register_visual_gen_routes()
        elif self.server_role is ServerRole.MM_ENCODER:
            assert isinstance(
                self.generator, MultimodalEncoder
            ), "generator must be a MultimodalEncoder for multimodal encoder"
            from tensorrt_llm.serve.api.openai.mm_encoder import \
                api_router as mm_encoder_router
            from tensorrt_llm.serve.api.openai.mm_encoder.serving import \
                MMEncoderServing
            setattr(
                app.state, mm_encoder_router.APP_STATE_KEY,
                MMEncoderServing(
                    render_resources=self._render_resources,
                    multimodal_server_config=self.multimodal_server_config,
                    allow_request_chat_template=self.
                    allow_request_chat_template,
                    **self._generate_serving_kwargs()))
            self.register_mm_encoder_routes()
        elif self.server_role is ServerRole.EMBEDDING:
            assert getattr(self.generator.args, "encode_only", False), (
                "generator must be an encode_only=True LLM for the embedding "
                "server")
            from tensorrt_llm.serve.api.openai.embeddings import \
                api_router as embeddings_router
            from tensorrt_llm.serve.api.openai.embeddings.serving import \
                EmbeddingServing
            embedding_serving = EmbeddingServing(
                model=self.model,
                generator=self.generator,
                embedding_max_queue_delay=self._embedding_max_queue_delay,
                embedding_max_queue_size=self._embedding_max_queue_size)
            # The lifespan starts and stops the batcher and the health probe
            # reports it; both read it here.
            self.embedding_batcher = embedding_serving.embedding_batcher
            setattr(app.state, embeddings_router.APP_STATE_KEY,
                    embedding_serving)
        else:
            from tensorrt_llm.serve.api.openai.tokenize import \
                api_router as tokenize_router
            from tensorrt_llm.serve.api.openai.tokenize.serving import \
                TokenizeServing
            setattr(app.state, tokenize_router.APP_STATE_KEY,
                    TokenizeServing(model=self.model, tokenizer=self.tokenizer))
            from tensorrt_llm.serve.api.openai.completions import \
                api_router as completions_router
            from tensorrt_llm.serve.api.openai.completions.serving import \
                CompletionServing
            setattr(app.state, completions_router.APP_STATE_KEY,
                    CompletionServing(**self._generate_serving_kwargs()))
            self.register_routes()

        # Attach routers after the role switch: every attach_router checks
        # that its serving objects are on app.state.
        attach_routers(
            app,
            dataclasses.replace(ServerCapabilities.for_role(self.server_role),
                                harmony_model=(not self._is_visual_gen)
                                and self.use_harmony))

    def _init_visual_gen(self):
        self.processor = None
        self.model_config = None

    def _init_llm(self, chat_template: Optional[str] = None):
        self.tokenizer = self.generator.tokenizer
        hf_tokenizer_path = self.generator.hf_model_dir
        trust_remote_code = self.generator.args.trust_remote_code
        checkpoint_format = getattr(self.generator.args, "checkpoint_format",
                                    None)
        if checkpoint_format in ("mistral", "mistral_large_3"):
            # Do not load HF processor for mistral native checkpoints
            # even if it is available
            self.processor = None
        else:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    hf_tokenizer_path, trust_remote_code=trust_remote_code)
            except Exception:
                logger.debug(
                    "Failed to load AutoProcessor or AutoConfig for %s",
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
            self.metrics_collector = MetricsCollector.from_args(
                self.generator.args, self.model)
            # One config-info series per server, not per frontend process.
            if self._launcher_forwarder is None:
                self.metrics_collector.log_config_from_args(
                    self.generator.args, self.model)

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

    _validate_internal_disagg_request = GenerateBaseServing._validate_internal_disagg_request

    _has_cache_transceiver_config = GenerateBaseServing._has_cache_transceiver_config

    await_disconnected = GenerateBaseServing.await_disconnected

    @functools.cached_property
    def _is_visual_gen(self) -> bool:
        # __init__ pre-caches this (so a probe patched during construction
        # sticks); computing on demand keeps instances built without
        # __init__ (unit tests use OpenAIServer.__new__) working.
        return _is_visual_gen_instance(self.generator)

    postproc_worker_enabled = GenerateBaseServing.postproc_worker_enabled

    @functools.cached_property
    def _postproc(self) -> PostprocStrategy:
        # __init__ pre-caches this; computing on demand keeps instances
        # built without __init__ (unit tests use OpenAIServer.__new__)
        # working.
        return select_postproc_strategy(self.postproc_worker_enabled)

    create_error_response = staticmethod(BaseServing.create_error_response)

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

    def _generate_serving_kwargs(self) -> Dict[str, Any]:
        """Constructor arguments of every GenerateBaseServing, read once."""
        return dict(
            model=self.model,
            generator=self.generator,
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            collect_perf_metrics=self._collect_perf_metrics,
            metrics_collector=self.metrics_collector,
            adjusted_steady_clock=self._adjusted_steady_clock,
            iteration_stats_wakeup_event=self._iteration_stats_wakeup_event,
            # The lifespan creates the collector task after the serving
            # objects exist, so they read this object's current binding.
            iteration_stats_collector_task_getter=(
                lambda: self._iteration_stats_collector_task),
            internal_disagg_auth_key=self._internal_disagg_auth_key,
            server_role=self.server_role,
            per_request_spec_decode_stats=self._per_request_spec_decode_stats,
            spec_decode_num_spec_tokens=self._spec_decode_num_spec_tokens,
            input_proc_executor=self._input_proc_executor,
            postproc=self._postproc)

    def _check_health(self) -> bool:
        # An embedding server's requests flow through the batcher worker; if it
        # has exited the engine is up but every /v1/embeddings request hangs, so
        # report unhealthy rather than a misleading 200.
        batcher = self.embedding_batcher
        if batcher is not None and not batcher.is_alive():
            return False
        if self._is_visual_gen:
            # VisualGen has no check_health(), so its /health always
            # answers 200.
            return True
        return self.generator.check_health()

    def register_routes(self):
        self.app.add_api_route("/health_generate",
                               self.health_generate,
                               methods=["GET"])
        self.app.add_api_route("/v1/data_transceiver_state",
                               self.data_transceiver_state,
                               methods=["GET"])
        # TODO: the metrics endpoint only reports iteration stats, not the runtime stats for now
        self.app.add_api_route("/metrics",
                               self.get_iteration_stats,
                               methods=["GET"])
        self.app.add_api_route("/energy_metrics",
                               self.get_energy_metrics,
                               methods=["GET"])
        self.app.add_api_route("/steady_clock_offset",
                               self.get_steady_clock_offset,
                               methods=["GET"])
        # Called by the disagg server to update the worker reference clock.
        self.app.add_api_route("/steady_clock_offset",
                               self.set_steady_clock_offset,
                               methods=["POST"])
        # TODO: workaround before ETCD support
        self.app.add_api_route("/kv_cache_events",
                               self.get_kv_cache_events,
                               methods=["POST"])
        resource_governor_queue = self.generator.resource_governor_queue
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
                chat_template=self.chat_template,
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

        self.app.add_api_route("/v1/responses",
                               self.openai_responses,
                               methods=["POST"])
        self.app.add_api_route('/v1/responses/{response_id}',
                               self.openai_responses_get_response,
                               methods=["GET"])
        self.app.add_api_route('/v1/responses/{response_id}',
                               self.openai_responses_delete_response,
                               methods=["DELETE"])

        # Profiling endpoints (PyTorch backend only)
        self.app.add_api_route("/start_profile",
                               self.start_profile,
                               methods=["POST"])
        self.app.add_api_route("/stop_profile",
                               self.stop_profile,
                               methods=["POST"])

        self._register_runtime_control_routes()
        self._register_rl_control_routes()
        self.app.add_api_route("/server_info",
                               self.get_server_info,
                               methods=["GET"])
        # The render and generate routes are a separate exposed surface: off
        # unless TRTLLM_ENABLE_RENDER_ENDPOINTS=1.
        mount_render_endpoints(self.app, self)
        if self.generator.args.return_perf_metrics:
            # register /prometheus/metrics
            self.mount_metrics()

    def register_visual_gen_routes(self):
        """Wire VisualGen serving for the VISUAL_GEN role.

        The VisualGen routes come from their RouterSpec; /metrics is
        registered here, not by a router.
        """
        # The only place outside serve/api/visual_gen/ that loads it, besides
        # attach_routers through the VisualGen RouterSpec.
        self._visual_gen_api = importlib.import_module(
            "tensorrt_llm.serve.api.visual_gen.api_router")
        self.media_storage_path = self._visual_gen_api.init_serving_state(
            self.app.state,
            generator=self.generator,
            model=self.model,
            adjusted_clock=self._adjusted_steady_clock)
        logger.info(f"VisualGen media storage path: {self.media_storage_path}")
        self.app.add_api_route("/metrics",
                               self.get_iteration_stats,
                               methods=["GET"])

    def _register_launcher_forwarding_routes(self) -> None:
        """Attached frontends: forward launcher-owned routes (before register_routes())."""
        if self._launcher_forwarder is None:
            return
        for method, path in FORWARDED_ROUTES:
            self.app.add_api_route(path,
                                   self._launcher_forwarder.forward,
                                   methods=[method],
                                   include_in_schema=False)

    def _register_server(self) -> None:
        """Register with the metadata server and/or disagg cluster (launcher only)."""
        if not self._registers_server:
            return
        if self.metadata_server is not None:
            metadata = {
                "model": self.model,
                "version": VERSION,
                "timestamp": datetime.now().isoformat(),
                "server_role": self.server_role.name,
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

    async def _deregister_server(self) -> None:
        if self.metadata_server is not None:
            self.metadata_server.remove(f"trtllm/{self.generator.llm_id}")
            logger.info(f"trtllm/{self.generator.llm_id} is unregistered")
        if self.disagg_cluster_worker:
            await self.disagg_cluster_worker.deregister_worker()

    def _apply_shared_clock_offset(self, offset: float) -> None:
        self._adjusted_steady_clock.set_reference_offset(offset)
        logger.info(f"Following the launcher's steady clock offset between "
                    f"local and disagg server: {offset} second")

    def _on_launcher_lost(self, error: BaseException) -> None:
        """Watchdog callback: fail requests with EngineDeadError, then shut down.

        Mirrors the fatal-error branch of ``health``.
        """
        # record_fatal_error keeps only the first error; False means the LLM
        # already shut down.
        if not self.generator.record_fatal_error(error):
            return
        if not self.generator.shutdown_started:
            _record_generator_termination(self.generator)
            signal.raise_signal(signal.SIGINT)

    def _health_probe(self) -> bool:
        if self._check_health():
            return True
        else:
            # If the engine has a fatal error, trigger server shutdown so the
            # pod doesn't linger as a zombie that accepts connections but never
            # produces tokens.  Uses SIGINT (not SIGTERM) to match the
            # existing CppExecutorError handlers and let uvicorn perform
            # its graceful shutdown sequence.  The doing_shutdown guard
            # ensures we only send SIGINT once — repeated health probes
            # won't stack signals during an in-progress teardown.
            fatal_error = (None if self._is_visual_gen else
                           self.generator.fatal_error)
            if fatal_error is not None and not self.generator.shutdown_started:
                logger.error(
                    "Health check detected fatal engine error, initiating "
                    f"server shutdown: {fatal_error}")
                _record_generator_termination(self.generator)
                signal.raise_signal(signal.SIGINT)
            return False

    _extract_metrics = GenerateBaseServing._extract_metrics

    def _responses_prompt_tokens_excluded_from_usage(
            self, request: ResponsesRequest) -> int:
        """Trailing rendered prompt tokens the usage of ``request`` leaves out.

        Off the Harmony path this route renders its input as the chat route
        renders a request with a generation prompt and no request-level
        template, and asks the same question about it, so both routes report
        the same prompt usage for the same prompt (kimi_k3 leaves out its
        generation channel opener).
        """
        if self.use_harmony:
            return 0
        # A generation-only handoff runs the prompt the context worker relayed
        # instead of rendering one; asked as the render path asks it.
        relayed = request.relayed_prompt_token_ids() is not None
        render_resources = RenderResources.from_server(self)
        return render_resources.prompt_tokens_excluded_from_usage(
            add_generation_prompt=True,
            pre_tokenized=relayed,
            request_chat_template=None,
        )

    async def openai_responses(self, request: ResponsesRequest,
                               raw_request: Request) -> Response:
        """Serve one /v1/responses request, streaming or not.

        ``parallel_tool_calls=false`` and, on the non-Harmony path,
        ``tool_choice`` values other than "auto" and "none" are accepted but
        not enforced (logged once). ``tool_choice="none"`` skips tool parsing.
        """

        async def create_response(
                promise: RequestOutput,
                postproc_params: PostprocParams) -> ResponsesResponse:
            await promise.aresult()
            args = postproc_params.postproc_args
            response = await self._postproc.apply_async(
                promise,
                lambda: responses_api_create_response(
                    generator=promise,
                    request=request,
                    sampling_params=args.sampling_params,
                    model_name=self.model,
                    conversation_store=self.conversation_store,
                    generation_result=None,
                    # request.store is applied downstream.
                    enable_store=self.enable_store,
                    use_harmony=self.use_harmony,
                    reasoning_parser=args.reasoning_parser,
                    tool_parser=args.tool_parser,
                    num_prompt_tokens=args.usage_num_prompt_tokens(),
                    tokenizer=self.tokenizer,
                ))
            # A postprocessing worker's result has no prompt token ids, and the
            # context-only handoff needs them, so take them from the promise as
            # the chat path does.
            if (request.disaggregated_params is not None
                    and request.disaggregated_params.request_type
                    == "context_only"):
                response.prompt_token_ids = promise.prompt_token_ids

            await self._extract_metrics(promise, raw_request)
            return response

        async def create_streaming_generator(promise: RequestOutput,
                                             postproc_params: PostprocParams):
            args = postproc_params.postproc_args
            streaming_processor = args.streaming_processor
            initial_responses = streaming_processor.get_initial_responses()
            for initial_response in initial_responses:
                yield initial_response

            async for res in promise:
                pp_results = self._postproc.apply(res, postproc_params)
                for pp_res in pp_results:
                    yield pp_res
            await self._extract_metrics(res, raw_request)

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

            # A web_search server tool this server cannot honour gets a 400
            # only under TRTLLM_WEB_SEARCH_UNAVAILABLE=reject, for clients that
            # drop the tool and retry rather than take an answer that may look
            # searched. By default the request is served without the tool (see
            # web_search_rejection_reason).
            web_search_error = web_search_rejection_reason(request.tools)
            if web_search_error is not None:
                return self.create_error_response(
                    err_type="InvalidRequestError",
                    message=(f"'web_search' cannot be honoured: "
                             f"{web_search_error}."),
                )

            # The signature check comes before anything is rendered or stored:
            # encoded_opaque_state and ctx_info_endpoint name a peer this
            # worker would pull KV cache from.
            self._validate_internal_disagg_request(request, raw_request)

            if ("parallel_tool_calls" in request.model_fields_set
                    and request.parallel_tool_calls is False):
                logger.warning_once(
                    "Responses API: 'parallel_tool_calls=false' is accepted "
                    "but not enforced; the model may still emit several tool "
                    "calls in one turn.",
                    key="responses_parallel_tool_calls_unenforced")
            if not self.use_harmony and request.tool_choice not in ("auto",
                                                                    "none"):
                logger.warning_once(
                    "Responses API: 'tool_choice' values other than 'auto' "
                    "and 'none' are accepted but not enforced on this model "
                    "path; generation is not constrained by them.",
                    key="responses_tool_choice_unenforced")

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
                # Not folded with request.store: prior context is still read
                # for a store=false follow-up; persisting checks request.store.
                enable_store=self.enable_store,
                use_harmony=self.use_harmony,
                tokenizer=self.tokenizer if not self.use_harmony else None,
                model_config=self.model_config
                if not self.use_harmony else None,
                processor=self.processor if not self.use_harmony else None,
                reasoning_parser=self.generator.args.reasoning_parser
                if not self.use_harmony else "gpt_oss",
                multimodal_server_config=self.multimodal_server_config,
                chat_template=self.chat_template
                if not self.use_harmony else None,
            )
            if not self.use_harmony:
                # Detokenize as chat completions does for these parsers: a
                # parser that reads special tokens needs them kept, and Kimi-K3's
                # tags also need them without spaces in between. Without this,
                # `<|open|>tools<|sep|>` arrives as `<|open|> tools <|sep|>` and
                # every K3 tool call reaches the client as message text.
                _configure_parser_special_token_decoding(
                    sampling_params,
                    reasoning_parser_name=self.generator.args.reasoning_parser,
                    tool_parser_name=self.tool_parser,
                    has_tools=bool(request.tools))

            streaming_processor = None
            if request.stream:
                # Per-request streaming processor
                streaming_processor = ResponsesStreamingProcessor(
                    request=request,
                    sampling_params=sampling_params,
                    model_name=self.model,
                    conversation_store=self.conversation_store,
                    enable_store=self.enable_store,
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
            postproc_args.num_prompt_tokens_offset = (
                self._responses_prompt_tokens_excluded_from_usage(request))
            postproc_params = PostprocParams(
                post_processor=responses_api_streaming_post_processor
                if request.stream else responses_api_post_processor,
                postproc_args=postproc_args,
            )
            disaggregated_params = to_llm_disaggregated_params(
                request.disaggregated_params)
            conversation_params = to_llm_conversation_params(
                request.conversation_params)
            promise = self.generator.generate_async(
                inputs=input_tokens,
                sampling_params=sampling_params,
                streaming=request.stream,
                disaggregated_params=disaggregated_params,
                conversation_params=conversation_params,
                kv_hints=request.kv_hints,
                _postproc_params=postproc_params
                if self.postproc_worker_enabled else None,
            )
            if not self.postproc_worker_enabled:
                # Set by the executor only for postprocessing it runs itself.
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)
                postproc_args.tokenizer = self.tokenizer

            if self.postproc_worker_enabled and request.store:
                logger.warning(
                    "Postproc workers are enabled, request will not be stored!")

            asyncio.create_task(self.await_disconnected(raw_request, promise))

            if request.stream:
                return ClosingStreamingResponse(
                    content=keep_responses_stream_alive(
                        guard_responses_stream(
                            create_streaming_generator(promise,
                                                       postproc_params),
                            streaming_processor.get_stream_failed_events,
                        )),
                    media_type="text/event-stream")
            else:
                response = await create_response(promise, postproc_params)
                # Context-only: the orchestrator may ask for the prompt as a
                # base64 int32 buffer, as on the chat path.
                if (request.disaggregated_params is not None and
                        request.disaggregated_params.return_prompt_token_ids_b64
                        and response.prompt_token_ids is not None):
                    import numpy as np
                    response.prompt_token_ids_b64 = base64.b64encode(
                        np.asarray(response.prompt_token_ids,
                                   dtype=np.int32).tobytes()).decode("ascii")
                    response.prompt_token_ids = None
                # by_alias: e.g. text.format.schema goes out under its wire name.
                return JSONResponse(content=response.model_dump(by_alias=True))
        except CppExecutorError:
            logger.error(traceback.format_exc())
            # If internal executor error is raised, shutdown the server
            _record_generator_termination(self.generator)
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

        return JSONResponse(content=response.model_dump(by_alias=True))

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

    async def __call__(self,
                       host,
                       port,
                       sockets: list[socket.socket] | None = None):
        # Store the binding address for server registration
        self.binding_addr = f"http://{host}:{port}"
        self.host = host
        self.port = port
        await serve_uvicorn(self.app,
                            host=host,
                            port=port,
                            sockets=sockets,
                            log_level="info",
                            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                            after_start=self._register_after_serving)

    async def _register_after_serving(self, server: uvicorn.Server) -> None:
        with _StartupTimer("http_server_start"):
            while not server.started:
                await asyncio.sleep(0.1)
        if self.disagg_cluster_worker:
            try:
                with _StartupTimer(
                        f"service_registration/{self.disagg_cluster_worker.worker_info.worker_id}"
                ):
                    await self.disagg_cluster_worker.register_worker()
            except Exception as e:
                logger.error(f"Worker registration failed: {e}")
                server.should_exit = True
