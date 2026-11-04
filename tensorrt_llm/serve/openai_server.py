#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import dataclasses
import functools
import json
import os
import re
import signal
import socket
import sys
import tempfile
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import (TYPE_CHECKING, Annotated, Any, AsyncIterator, Dict, List,
                    Optional, Union)

import uvicorn
from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (FileResponse, JSONResponse, Response,
                               StreamingResponse)
from pydantic import ValidationError
from starlette.routing import Mount
from transformers import AutoProcessor

from tensorrt_llm._startup import _StartupTimer
from tensorrt_llm._torch.async_llm import AsyncLLM
from tensorrt_llm._utils import EnergyMonitor
# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.executor.rpc import RPCError
from tensorrt_llm.inputs import prompt_inputs
from tensorrt_llm.inputs.media_io import (BaseMediaIO,
                                          log_allowed_local_media_path)
from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor
from tensorrt_llm.inputs.utils import ConversationMessage
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi import MultimodalEncoder, SchedulingParams, tracing
from tensorrt_llm.llmapi.disagg_utils import (DisaggClusterConfig,
                                              MetadataServerConfig, ServerRole)
from tensorrt_llm.llmapi.llm import LLM, RequestOutput
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
from tensorrt_llm.llmapi.thinking_budget import \
    add_thinking_budget_logits_processor
from tensorrt_llm.logger import logger
from tensorrt_llm.media.encoding import image_to_bytes
from tensorrt_llm.media.tensor_payload import is_tensor_format
from tensorrt_llm.metrics.collector import MetricsCollector
from tensorrt_llm.runtime.kv_cache_hash import \
    get_effective_kv_cache_event_hash_algo
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams
from tensorrt_llm.serve._base.errors import (EngineFatalError, ServingError,
                                             serving_error)
from tensorrt_llm.serve._base.msgspec_route import _MsgspecRoute
from tensorrt_llm.serve._base.request_context import RequestContext
from tensorrt_llm.serve._base.routing import (RoleGroup, ServerCapabilities,
                                              attach_routers, role_group_of)
from tensorrt_llm.serve._base.serving import (BaseServing, GenerateBaseServing,
                                              _record_generator_termination)
from tensorrt_llm.serve._base.sse import chat_stream_generator
from tensorrt_llm.serve._perf_metrics_config import resolve_perf_metrics_config
from tensorrt_llm.serve.api.anthropic.adapter import anthropic_error_response
from tensorrt_llm.serve.api.anthropic.serving import (
    AnthropicBatchesServing, AnthropicCountTokensServing,
    AnthropicMessagesServing)
from tensorrt_llm.serve.api.openai.embeddings.batcher import EncodeBatcher
from tensorrt_llm.serve.api.ops.system.serving import SystemServing
from tensorrt_llm.serve.api.visual_gen.protocol import (ImageEditRequest,
                                                        ImageGenerationRequest,
                                                        ImageGenerationResponse,
                                                        ImageObject)
from tensorrt_llm.serve.chat_tokenization import tokenize_harmony_chat_request
from tensorrt_llm.serve.chat_utils import (load_chat_template,
                                           parse_chat_messages_coroutines)
from tensorrt_llm.serve.cluster_storage import create_cluster_storage_client
from tensorrt_llm.serve.conversation_id import resolve_request_conversation_id
from tensorrt_llm.serve.disagg_auth import validate_subagent_affinity
from tensorrt_llm.serve.disagg_auto_scaling import DisaggClusterWorker
from tensorrt_llm.serve.metadata_server import create_metadata_server
from tensorrt_llm.serve.multi_frontend import (FORWARDED_ROUTES,
                                               AttachedFrontendWatchdog,
                                               LauncherForwarder,
                                               MultiFrontendServing,
                                               SharedClockOffset,
                                               follow_clock_offset)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionNamedToolChoiceParam, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionToolsParam, MemoryUpdateRequest,
    ResponseFormat, ResponsesRequest, ResponsesResponse,
    RuntimeMemoryUpdateRequest, StartProfileRequest, UpdateWeightsRequest,
    ensure_request_chat_template_allowed, to_llm_conversation_params,
    to_llm_disaggregated_params)
from tensorrt_llm.serve.openai_video_routes import _VideoRoutesMixin
from tensorrt_llm.serve.perf_metrics import (PerfMetricsJsonlWriter,
                                             PerfMetricsMiddleware)
from tensorrt_llm.serve.postprocess_handlers import (
    ChatCompletionPostprocArgs, ChatPostprocArgs, PostprocStrategy,
    ResponsesAPIPostprocArgs, chat_harmony_post_processor,
    chat_harmony_streaming_post_processor, chat_response_post_processor,
    chat_stream_post_processor, responses_api_post_processor,
    responses_api_streaming_post_processor, select_postproc_strategy)
from tensorrt_llm.serve.render import (RenderResources, mount_render_endpoints,
                                       prepare_chat_request,
                                       render_conversation)
from tensorrt_llm.serve.responses_utils import (ClosingStreamingResponse,
                                                ConversationHistoryStore,
                                                ResponsesStreamingProcessor,
                                                ServerArrivalTimeMiddleware)
from tensorrt_llm.serve.responses_utils import \
    create_response as responses_api_create_response
from tensorrt_llm.serve.responses_utils import (guard_responses_stream,
                                                keep_responses_stream_alive)
from tensorrt_llm.serve.responses_utils import \
    request_preprocess as responses_api_request_preprocess
from tensorrt_llm.serve.responses_web_search import web_search_rejection_reason
from tensorrt_llm.serve.rl_control_auth import validate_rl_control_request
from tensorrt_llm.serve.runtime_control_auth import (
    RuntimeControlReplayCache, validate_runtime_control_request)
from tensorrt_llm.serve.serving_extensions import load_builtin_extensions
from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory
from tensorrt_llm.serve.visual_gen_metrics import (
    build_visual_gen_server_timings, build_visual_gen_timing_headers)
from tensorrt_llm.serve.visual_gen_utils import (local_media_path_is_disallowed,
                                                 parse_visual_gen_params)
from tensorrt_llm.version import __version__ as VERSION

from .._utils import (AdjustedSteadyClock,
                      get_global_steady_clock_now_in_seconds,
                      set_prometheus_multiproc_dir)
from ._telemetry import create_uvicorn_server
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

_warned_unresolvable_thinking = False


def _warn_unresolvable_thinking_once(reasoning_parser: str) -> None:
    """Warn when a generation worker cannot learn the reasoning mode.

    Nothing was rendered here and the context worker relayed no value, so
    every request on this deployment is parsed in a possibly wrong mode with
    nothing in the response saying so. Reachable via a rolling upgrade where
    the context worker predates `resolved_thinking`, a hand-crafted
    `generation_only` request, or an orchestration path that does not go
    through `_get_ctx_request`.
    """
    global _warned_unresolvable_thinking
    if _warned_unresolvable_thinking:
        return
    _warned_unresolvable_thinking = True
    logger.warning(
        f"Reasoning parser {reasoning_parser!r} resolves its mode from the "
        "rendered prompt, but this request had neither a rendered prompt nor "
        "a relayed mode from a context worker. Reasoning content will not be "
        "separated correctly. Check that the context worker is running a "
        "build that relays 'resolved_thinking'.")


def _validate_required_tool_grammar_request(
        request: ChatCompletionRequest, sampling_params: SamplingParams, *,
        guided_decoding_backend: Optional[str], tool_parser: Optional[str],
        reasoning_parser: Optional[str]) -> None:
    """Validate inputs for mandatory tool-call grammar construction."""
    if guided_decoding_backend != "xgrammar":
        raise ValueError("Required tool grammars need the xgrammar backend.")
    if not tool_parser:
        raise ValueError("Required tool grammars need a tool parser.")
    if not reasoning_parser or not ReasoningParserFactory.resolves_thinking_from_prompt(
            reasoning_parser):
        raise ValueError(
            "Required tool grammars need a reasoning parser that resolves its "
            "mode from the rendered prompt. Configure the server with "
            "--reasoning_parser nemotron-v3 (or a compatible parser alias).")
    if not request.tools:
        raise ValueError("tool_choice='required' requires request-level tools.")
    if (request.prompt_token_ids is not None or request.prompt_token_ids_b64
            or not request.add_generation_prompt
            or request.disaggregated_params is not None):
        raise ValueError("Required tool grammars need chat messages with "
                         "add_generation_prompt=True on a native server.")
    if sampling_params.guided_decoding is not None:
        raise ValueError(
            "tool_choice='required' cannot be combined with response_format "
            "or other guided-decoding settings.")


def _configure_parser_special_token_decoding(
        sampling_params: SamplingParams, reasoning_parser_name: Optional[str],
        tool_parser_name: Optional[str], has_tools: bool) -> None:
    """Configure detokenization for parsers that consume special tokens."""
    needs_compact_special_tokens = False
    if reasoning_parser_name and ReasoningParserFactory.needs_raw_special_tokens(
            reasoning_parser_name):
        # Unlike the tool-parser flag below, this applies to every chat
        # request: reasoning delimiters are special tokens regardless of
        # whether tools are attached.
        sampling_params.skip_special_tokens = False
        needs_compact_special_tokens = reasoning_parser_name.lower(
        ) == "kimi_k3"

    if tool_parser_name and has_tools:
        tool_parser_cls = ToolParserFactory.parsers.get(
            tool_parser_name.lower())
        if tool_parser_cls and getattr(tool_parser_cls,
                                       'needs_raw_special_tokens', False):
            sampling_params.skip_special_tokens = False
            needs_compact_special_tokens |= tool_parser_name.lower(
            ) == "kimi_k3"

    if needs_compact_special_tokens:
        # K3 XTML places ordinary-text tag names directly between special
        # tokens, for example ``<|open|>tools<|sep|>``. Inserting spaces here
        # changes the protocol and prevents the K3 parsers from matching it.
        sampling_params.spaces_between_special_tokens = False


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
    # Parsers whose wire format can't be expressed through structure_info
    # triples (kimi_k3 XTML) provide a complete format themselves.
    custom_format = parser.build_strict_structural_tag_format(tools)
    if custom_format is not None:
        resp_format = ResponseFormat(type="structural_tag",
                                     format=custom_format)
        return GuidedDecodingParams(structural_tag=resp_format.model_dump_json(
            by_alias=True, exclude_none=True))
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


def _parser_extracts_forced_tool_calls(tool_parser_name) -> bool:
    """Whether this parser extracts forced calls from its own markup.

    Mirrors ``_forced_choice_uses_tool_parser`` in postprocess_handlers: those
    parsers emit native markup on a forced call and the post-processor parses
    it, so the request must not be constrained (or rejected) here.
    """
    if not tool_parser_name:
        return False
    parser_cls = ToolParserFactory.parsers.get(tool_parser_name.lower())
    return bool(parser_cls
                and getattr(parser_cls, "extracts_forced_tool_calls", False))


def _build_forced_tool_call_decoding(tools, tool_parser_name, forced_tool_name):
    r"""Build the begin-prefix and guided-decoding params for a named tool_choice.

    OpenAI / Azure spec: ``tool_choice={"type":"function","function":{"name":"X"}}``
    must force the model to emit exactly one tool call to function ``X``.

    Unlike the strict-tools path (which relies on xgrammar's *triggered*
    structural tags and is therefore non-mandatory by design), this path
    forces the call by:

    1. Asking the caller to prepend ``begin_prefix`` (e.g.
       ``<tool_call>\n{"name":"X", "arguments":``) to the rendered prompt
       so the model starts generation *already inside* the tool call.
    2. Constraining what follows with a plain JSON-schema (or json-object)
       guided-decoding mode so the arguments are guaranteed to be valid
       against ``tool.function.parameters``.

    The caller is then responsible for synthesizing the resulting
    ``tool_calls`` entry — the raw model output is the JSON arguments
    string of the forced function call.

    Raises:
        ValueError: if ``tools`` is empty, ``tool_parser_name`` is unset, the
            parser is not registered, the parser does not support structural
            tags, or ``forced_tool_name`` is not in ``tools``.
    """
    if not tools:
        raise ValueError("tool_choice requires a named function "
                         f"'{forced_tool_name}' but no tools were provided.")
    if not tool_parser_name:
        raise ValueError(
            "tool_choice with a named function requires a tool parser "
            "to be configured on the server (use --tool_parser).")

    tool_parser_cls = ToolParserFactory.parsers.get(tool_parser_name.lower())
    if tool_parser_cls is None:
        raise ValueError(
            f"Tool parser '{tool_parser_name}' is not registered; cannot "
            f"force tool_choice to function '{forced_tool_name}'.")

    parser = tool_parser_cls()
    if not parser.supports_structural_tag():
        raise ValueError(
            f"Tool parser '{tool_parser_name}' does not support structural "
            "tags, which are required to honor tool_choice with a named "
            "function.")

    forced_tool = next(
        (t for t in tools if t.function.name == forced_tool_name), None)
    if forced_tool is None:
        available = sorted(t.function.name for t in tools if t.function.name)
        raise ValueError(
            f"tool_choice requested function '{forced_tool_name}' which is "
            f"not present in `tools`. Available functions: {available}.")

    info = parser.structure_info()(forced_tool.function.name)
    begin_prefix = info.begin

    if forced_tool.function.parameters:
        guided = GuidedDecodingParams(json=forced_tool.function.parameters)
    else:
        # No parameters declared — still require a JSON object (typically ``{}``)
        # so the synthesized ``arguments`` field is well-formed JSON.
        guided = GuidedDecodingParams(json_object=True)

    return begin_prefix, guided


def _new_media_dir(root: Path) -> Path:
    """Create a directory under ``root`` stamped with the current time.

    ``mkdir`` without ``exist_ok`` is what makes this safe: the kernel either
    creates the directory or raises, so two servers starting in the same
    second take separate names instead of sharing one.
    """
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = root / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            n += 1


def _resolve_media_dir() -> Path:
    """Create and return the directory to store generated media in.

    ``TRTLLM_MEDIA_STORAGE_PATH`` names the directory outright, empty meaning
    unset. Otherwise it goes beside the working directory, and where that
    cannot be written it goes to a private temporary one: a shared ``/tmp``
    holds directories owned by other users, so the fallback takes a name
    nobody else can hold rather than a fixed one.
    """
    explicit = os.getenv("TRTLLM_MEDIA_STORAGE_PATH")
    if explicit:
        path = Path(explicit)
        path.mkdir(parents=True, exist_ok=True)
        return path
    try:
        return _new_media_dir(Path.cwd() / "trtllm_generated")
    except OSError:
        # OSError rather than PermissionError: a read-only mount raises
        # EROFS, which is not one.
        stamp = datetime.now().strftime("%y%m%d-%H%M%S")
        return Path(tempfile.mkdtemp(prefix=f"trtllm_generated-{stamp}-"))


def _normalize_image_output(image) -> list:
    """Normalize image output to a list of individual images.

    Handles single tensors, batched 4D tensors, and lists.
    """
    if isinstance(image, list):
        return image
    if hasattr(image, "dim") and image.dim() == 4:
        return [image[i] for i in range(image.shape[0])]
    return [image]


def _image_output_size(image) -> Optional[str]:
    pil_size = getattr(image, "size", None)
    if isinstance(pil_size, tuple) and len(pil_size) >= 2:
        width, height = pil_size[:2]
        return f"{int(width)}x{int(height)}"

    shape = getattr(image, "shape", None)
    if shape is None:
        return None

    dims = tuple(int(dim) for dim in shape)
    if len(dims) == 2:
        height, width = dims
    elif len(dims) == 3:
        if dims[0] in (1, 3, 4) and dims[1] > 4 and dims[2] > 4:
            height, width = dims[1], dims[2]
        else:
            height, width = dims[0], dims[1]
    else:
        return None

    return f"{width}x{height}"


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


class _AggregatedChatBackend:
    """ChatBackend over OpenAIServer's typed chat core.

    Transitional: deleted when the chat core moves out of OpenAIServer into a
    serving class that implements stream_model and error_message itself.
    """

    def __init__(self, server: "OpenAIServer") -> None:
        self._server = server

    async def create_chat_completion(self, request: ChatCompletionRequest,
                                     ctx: RequestContext):
        return await self._server.create_chat_completion(request, ctx)

    def stream_model(self, request: ChatCompletionRequest) -> str:
        # The served name, as the aggregated route always announced it.
        return self._server.model

    def error_message(self, error: ServingError) -> str:
        # What the rejected chat route's body said: its message, else the
        # whole body.
        return error.body.message or json.dumps(error.body.model_dump())


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

        self.app = FastAPI(lifespan=lifespan)
        self.app.state.trust_kv_hints = os.getenv("TRTLLM_TRUST_KV_HINTS",
                                                  "0") == "1"
        self.app.router.route_class = _MsgspecRoute

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request, exc):
            if request.url.path.startswith("/v1/messages"):
                # Anthropic clients expect a 400 with the Anthropic error
                # envelope, not FastAPI's 422 shape.
                if self.metrics_collector:
                    self.metrics_collector.log_request_error(http_code=400)
                return anthropic_error_response(str(exc),
                                                "invalid_request_error", 400)
            if self.server_role is ServerRole.VISUAL_GEN:
                return self._create_visual_gen_validation_error_response(exc)
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

        # /health, /version and /v1/models for every role. The health
        # probe stays here, with its private accesses, until the public
        # health API exists.
        self.app.state.ops_serving_system = SystemServing(
            model=self.model, health_probe=self._health_probe)
        if role_group_of(self.server_role) is RoleGroup.LLM:
            # An attached frontend of a multi-frontend server forwards the
            # launcher-owned routes, the Message Batches routes among them.
            # They register before the routers so that they keep winning
            # over the local handlers: FastAPI matches in registration order.
            self._register_launcher_forwarding_routes()
            messages_serving = AnthropicMessagesServing(
                _AggregatedChatBackend(self))
            self.app.state.anthropic_serving_messages = messages_serving
            # Transitional: RenderResources per request until it is built once.
            self.app.state.anthropic_serving_count_tokens = (
                AnthropicCountTokensServing(
                    render_resources=lambda: RenderResources.from_server(self)))
            self.app.state.anthropic_serving_batches = AnthropicBatchesServing(
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
                self.app.state, mm_encoder_router.APP_STATE_KEY,
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
            setattr(self.app.state, embeddings_router.APP_STATE_KEY,
                    embedding_serving)
        else:
            from tensorrt_llm.serve.api.openai.tokenize import \
                api_router as tokenize_router
            from tensorrt_llm.serve.api.openai.tokenize.serving import \
                TokenizeServing
            setattr(self.app.state, tokenize_router.APP_STATE_KEY,
                    TokenizeServing(model=self.model, tokenizer=self.tokenizer))
            from tensorrt_llm.serve.api.openai.completions import \
                api_router as completions_router
            from tensorrt_llm.serve.api.openai.completions.serving import \
                CompletionServing
            setattr(self.app.state, completions_router.APP_STATE_KEY,
                    CompletionServing(**self._generate_serving_kwargs()))
            self.register_routes()

        # Attach routers after the role switch: every attach_router checks
        # that its serving objects are on app.state.
        attach_routers(
            self.app,
            dataclasses.replace(ServerCapabilities.for_role(self.server_role),
                                harmony_model=(not self._is_visual_gen)
                                and self.use_harmony))

        if self._collect_perf_metrics:
            self.app.add_middleware(PerfMetricsMiddleware,
                                    expose_headers=self._expose_perf_metrics,
                                    writer=self._perf_metrics_writer,
                                    adjusted_clock=self._adjusted_steady_clock)
        self.app.add_middleware(ServerArrivalTimeMiddleware,
                                adjusted_clock=self._adjusted_steady_clock)

    def _init_visual_gen(self):
        self.processor = None
        self.model_config = None
        self.media_storage_path = _resolve_media_dir()
        logger.info(f"VisualGen media storage path: {self.media_storage_path}")
        self.video_gen_tasks = {}

    def _supports_image_edit(self) -> bool:
        if not self._is_visual_gen:
            return False

        executor = getattr(self.generator, "executor", None)
        if executor is None:
            return False

        pipeline = getattr(executor, "pipeline", None)
        if pipeline is not None:
            return bool(getattr(pipeline, "supports_image_edit", False))

        supports_image_edit = getattr(executor, "supports_image_edit", None)
        if supports_image_edit is not None:
            return bool(supports_image_edit)

        return False

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

    _logit_bias_vocab_size = GenerateBaseServing._logit_bias_vocab_size

    _validate_internal_disagg_request = GenerateBaseServing._validate_internal_disagg_request

    def _get_scheduling_params(
            self, request: ChatCompletionRequest,
            raw_request: Optional[Request]) -> SchedulingParams:
        return SchedulingParams(
            agent_hierarchy=request.agent_hierarchy,
            subagent_affinity_id=validate_subagent_affinity(
                getattr(self, "_internal_disagg_auth_key", None), request,
                getattr(self, "server_role", None),
                None if raw_request is None else raw_request.headers))

    _has_cache_transceiver_config = GenerateBaseServing._has_cache_transceiver_config

    _attach_multimodal_data = staticmethod(
        GenerateBaseServing._attach_multimodal_data)

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
        # TODO: the metrics endpoint only reports iteration stats, not the runtime stats for now
        self.app.add_api_route("/metrics",
                               self.get_iteration_stats,
                               methods=["GET"])

    async def _require_rl_control_auth(self, raw_request: Request) -> None:
        body = await raw_request.body()
        try:
            validate_rl_control_request(self._rl_control_api_key, body,
                                        raw_request.headers)
        except ValueError as e:
            raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED,
                                detail=str(e)) from e

    async def _require_runtime_control_auth(self, raw_request: Request) -> None:
        body = await raw_request.body()
        try:
            route = raw_request.scope.get("route")
            route_path = getattr(route, "path", None)
            if route_path is None:
                raise ValueError("Runtime control route is unavailable.")
            validate_runtime_control_request(
                self._runtime_control_api_key,
                raw_request.method,
                route_path,
                body,
                raw_request.headers,
                self._runtime_control_replay_cache,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED,
                detail=str(error),
            ) from error

    def _register_runtime_control_routes(self) -> None:
        if not self._enable_runtime_control_endpoints:
            return

        dependencies = [Depends(self._require_runtime_control_auth)]
        self.app.add_api_route("/release_memory",
                               self.runtime_release_memory,
                               methods=["POST"],
                               dependencies=dependencies)
        self.app.add_api_route("/resume_memory",
                               self.runtime_resume_memory,
                               methods=["POST"],
                               dependencies=dependencies)
        self.app.add_api_route("/memory_status",
                               self.runtime_memory_status,
                               methods=["GET"],
                               dependencies=dependencies)

    def _register_rl_control_routes(self) -> None:
        if not self._enable_rl_control_endpoints:
            return

        dependencies = [Depends(self._require_rl_control_auth)]
        self.app.add_api_route("/release_memory",
                               self.release_memory,
                               methods=["POST"],
                               dependencies=dependencies)
        self.app.add_api_route("/resume_memory",
                               self.resume_memory,
                               methods=["POST"],
                               dependencies=dependencies)
        self.app.add_api_route("/update_weights",
                               self.update_weights,
                               methods=["POST"],
                               dependencies=dependencies)

    def register_visual_gen_routes(self):
        """Register routes for diffusion model serving."""
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
        self.app.add_api_route("/v1/videos/sync",
                               self.openai_video_generation_sync,
                               methods=["POST"])
        # Deprecated alias of /v1/videos/sync, retained for upstream
        # back-compat after the rename; both hit the same sync handler.
        self.app.add_api_route("/v1/videos/generations",
                               self.openai_video_generation_sync,
                               methods=["POST"],
                               deprecated=True)
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

        # Without iteration stats the engine produces none, so there is
        # nothing to wait for.
        if not getattr(self.generator.args, "enable_iter_perf_stats", True):
            return JSONResponse(content=[])

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
        # The read waits up to 2 s for stats inside a blocking executor call,
        # so it runs on the metrics thread; on the event loop it would stall
        # every response being streamed for as long as it waits.
        # One read at a time: a scrape that arrives while a read is in flight
        # returns no stats instead of queueing behind it, so scraping an idle
        # server faster than one read per 2 s cannot build a backlog. A read
        # drains the queue and hands everything to its own scrape, so when two
        # scrapers poll at once, one of them regularly gets an empty list.
        read = self._metrics_stats_read
        if read is not None and (not read.done()
                                 or self._metrics_stats_read_waiting):
            # In flight, or finished with its scrape about to take the stats.
            return JSONResponse(content=[])
        if (read is None or self._metrics_stats_read_consumed
                or read.cancelled()):
            read = asyncio.get_running_loop().run_in_executor(
                self._metrics_stats_executor, self.generator.get_stats, 2)
            self._metrics_stats_read = read
            self._metrics_stats_read_consumed = False
        # Otherwise the last read finished after its scrape went away, and
        # this scrape takes the stats it drained rather than losing them.
        self._metrics_stats_read_waiting = True
        try:
            # Shielded so a scrape that goes away leaves the read running
            # until its thread finishes, and its stats for the next scrape.
            stats = await asyncio.shield(read)
        except RPCError as e:
            # The executor is unreachable, e.g. while it shuts down.
            logger.debug(f"Failed to fetch iteration stats: {e}")
            stats = []
        except Exception:
            # The read failed; the next scrape starts a new one.
            self._metrics_stats_read_consumed = True
            raise
        finally:
            self._metrics_stats_read_waiting = False
        self._metrics_stats_read_consumed = True
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
        self._adjusted_steady_clock.set_reference_offset(offset)
        logger.info(
            f"The steady clock offset between local and disagg server: {offset} second"
        )
        # Multi-frontend launcher: every frontend stamps perf metrics with it.
        if self._shared_clock_offset is not None:
            try:
                self._shared_clock_offset.publish(offset)
            except OSError as e:
                logger.error(
                    f"Failed to share the steady clock offset with the "
                    f"attached frontends: {e}")
                return Response(status_code=500, content=str(e))
        return Response(status_code=200)

    async def get_steady_clock_offset(self) -> JSONResponse:
        # The calibrated offset is applied to AdjustedSteadyClock, whose source
        # is the rank-adjusted global clock. Sample that same clock here so a
        # process-global offset is not counted twice in emitted metrics.
        receive_ts = get_global_steady_clock_now_in_seconds()
        await asyncio.sleep(0.2)
        transmit_ts = get_global_steady_clock_now_in_seconds()
        return JSONResponse(content={
            "receive_ts": receive_ts,
            "transmit_ts": transmit_ts
        })

    async def get_kv_cache_events(self) -> JSONResponse:
        events = []
        try:
            async for event in self.generator.get_kv_cache_events_async(0):
                events.append(event)
        except (IndexError, asyncio.QueueEmpty):
            # queue is empty, no more events
            pass
        return JSONResponse(content=events)

    _extract_metrics = GenerateBaseServing._extract_metrics

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
        chat_response = self._postproc.apply(promise, postproc_params)

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
                # Wait for signal that requests have completed and stats may
                # be available (a multi-frontend launcher also polls).
                woken_by_timer = False
                try:
                    await asyncio.wait_for(
                        self._iteration_stats_wakeup_event.wait(),
                        timeout=self._iteration_stats_poll_interval)
                except asyncio.TimeoutError:
                    woken_by_timer = True

                # Clear the event for next wakeup
                self._iteration_stats_wakeup_event.clear()

                # Drain all available iteration stats and log each one to Prometheus.
                # The fetch blocks for up to its timeout, so it runs on the
                # metrics thread, which /metrics leaves idle while this loop runs.
                try:
                    # A timer wakeup (multi-frontend launcher poll) reads
                    # without waiting.
                    for llm_stat in await asyncio.get_running_loop(
                    ).run_in_executor(self._metrics_stats_executor,
                                      self.generator.get_stats,
                                      0 if woken_by_timer else 0.5):
                        self._record_iteration_stat(llm_stat)
                except Exception as e:
                    # Log errors but continue collecting stats
                    logger.error(f"Error collecting iteration stats: {e}",
                                 exc_info=True)
                    # Brief sleep to avoid tight loop on persistent errors
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Iteration stats collector loop cancelled")
            raise

    _apply_spec_decode_stats_opt_in = GenerateBaseServing._apply_spec_decode_stats_opt_in

    def _record_iteration_stat(self, llm_stat: dict) -> None:
        self.metrics_collector.log_iteration_stats(llm_stat)
        # Tee into the /metrics snapshot buffer so the HTTP handler can serve
        # without competing for the engine queue (nvbug 6102381).
        if self._iteration_stats_buffer is not None:
            self._iteration_stats_buffer.append(llm_stat)

    async def openai_chat(self, request: ChatCompletionRequest,
                          raw_request: Request) -> Response:
        try:
            result = await self.create_chat_completion(
                request, RequestContext.from_http(raw_request))
            if isinstance(result, ChatCompletionResponse):
                return JSONResponse(content=result.model_dump())
            return StreamingResponse(content=result,
                                     media_type="text/event-stream")
        except ServingError as e:
            return JSONResponse(content=e.body.model_dump(),
                                status_code=e.status)
        except EngineFatalError:
            # Answered as HTTP 200 with body null (FastAPI's rendering of
            # None); the in-process callers map that to their own errors.
            # Answering 5xx would be a behavior change of its own.
            return None
        except ValueError as e:
            # JSONResponse encodes in its constructor (allow_nan=False), so a
            # response that cannot be encoded fails here and keeps its 400.
            return self.create_error_response(str(e))
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(str(e))

    async def create_chat_completion(
        self, request: ChatCompletionRequest, ctx: RequestContext
    ) -> Union[ChatCompletionResponse, AsyncIterator[str]]:
        """Run one chat completion and return a typed result.

        Returns the ChatCompletionResponse for a non-streaming request, and
        for a streaming one the SSE stream as strings, ending with
        "data: [DONE]".

        Raises:
            ServingError: The request is rejected; the HTTP status travels
                apart from the body.
            EngineFatalError: The engine failed; SIGINT has been sent.
        """

        def get_role() -> str:
            if request.add_generation_prompt:
                role = "assistant"
            else:
                role = request.messages[-1]["role"]
            return role

        try:
            # Extension preprocessing, the required-tool check and tool dumping
            # are the first stage of the shared prompt-preparation pipeline.
            render_resources = RenderResources.from_server(self)
            extension = render_resources.extension
            tool_dicts = prepare_chat_request(request,
                                              render_resources).tool_dicts
            required_mode = extension.required_tool_choice_mode()
            required_tool_grammar = (request.tool_choice == "required"
                                     and required_mode == "grammar")
            conversation: List[ConversationMessage] = []
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
            if required_tool_grammar:
                _validate_required_tool_grammar_request(
                    request,
                    sampling_params,
                    guided_decoding_backend=self.generator.args.
                    guided_decoding_backend,
                    tool_parser=self.tool_parser,
                    reasoning_parser=self.generator.args.reasoning_parser)
            # Pre-render, so the mode may be unresolved. Safe because the
            # boundary lookup reads the parser class attributes, not the
            # branch `__init__` picked.
            add_thinking_budget_logits_processor(
                sampling_params,
                reasoning_parser=self.generator.args.reasoning_parser,
                tokenizer=self.tokenizer,
                chat_template_kwargs=request.chat_template_kwargs,
            )
            # Resolve a named tool_choice (OpenAI spec / Azure compat):
            #   tool_choice = {"type": "function", "function": {"name": "X"}}
            # forces the model to call exactly function X. We honor this by
            # (a) prefix-injecting the tool parser's ``begin`` string into the
            # rendered chat prompt so the model starts generation already inside
            # a tool call, and (b) constraining the generated arguments with
            # plain JSON-schema guided decoding. The post-processor then
            # synthesizes the resulting ``tool_calls`` entry. Unlike the
            # ``triggered_tags`` structural-tag path used for ``strict`` tools,
            # this path is mandatory by construction — the model cannot escape
            # the tool call.
            forced_tool_name = None
            forced_tool_begin_prefix = None
            if isinstance(request.tool_choice,
                          ChatCompletionNamedToolChoiceParam):
                if not request.tools:
                    raise serving_error(
                        message=("tool_choice with a named function requires "
                                 "`tools` to be set."),
                        err_type="BadRequestError",
                        status_code=HTTPStatus.BAD_REQUEST)
                if (request.prompt_token_ids is not None
                        or request.prompt_token_ids_b64):
                    # Tokenizing and concatenating the ``begin_prefix`` onto a
                    # pre-tokenized prompt correctly across all tokenizers is
                    # fragile (token-boundary issues with special tokens).
                    # Reject the combination explicitly until there is a
                    # demonstrated use case.
                    #
                    # ``prompt_token_ids_b64`` must be checked here too: it is
                    # only decoded into ``prompt_token_ids`` further below, so
                    # testing ``prompt_token_ids`` alone lets a base64 request
                    # through to the pre-tokenized branch, where the begin
                    # prefix is never appended.
                    raise serving_error(
                        message=("tool_choice with a named function is not "
                                 "supported alongside `prompt_token_ids`; pass "
                                 "`messages` instead."),
                        err_type="BadRequestError",
                        status_code=HTTPStatus.BAD_REQUEST)
                if sampling_params.guided_decoding is not None:
                    # A named tool_choice is mandatory by construction: the
                    # begin prefix plus a JSON-schema constraint force the
                    # model into exactly this call. That cannot coexist with a
                    # client-supplied guided-decoding mode (``response_format``
                    # and friends), which constrains the assistant *content*
                    # instead. Honouring only one of them silently would either
                    # ignore an explicit instruction or emit a tool call whose
                    # arguments are unconstrained text, so reject instead.
                    raise serving_error(
                        message=("tool_choice with a named function cannot be "
                                 "combined with `response_format` or other "
                                 "guided-decoding settings."),
                        err_type="BadRequestError",
                        status_code=HTTPStatus.BAD_REQUEST)
                if (self.generator.args.guided_decoding_backend is None
                        and not _parser_extracts_forced_tool_calls(
                            self.tool_parser)):
                    # Parsers that extract forced calls from their own markup
                    # need no grammar, so this only applies to the constrained
                    # path below.
                    #
                    # The forced path is only mandatory because a JSON-schema
                    # grammar constrains what follows the begin prefix. With no
                    # guided-decoding backend configured that grammar is never
                    # built and the per-request params are silently dropped, so
                    # the prefix alone would be left "forcing" the call: the
                    # model completes the arguments, closes the tool-call
                    # object itself and keeps generating, and the whole tail
                    # ends up reported as ``function.arguments``. Fail loudly
                    # instead of returning a tool call whose arguments are
                    # unconstrained text.
                    raise serving_error(message=(
                        "tool_choice with a named function requires the "
                        "server to be started with a guided decoding "
                        "backend (set `guided_decoding_backend` to "
                        "'xgrammar' or 'llguidance')."),
                                        err_type="BadRequestError",
                                        status_code=HTTPStatus.BAD_REQUEST)
                forced_tool_name = request.tool_choice.function.name

            reasoning_parser_name = self.generator.args.reasoning_parser
            # Message-level (dynamic) tools are a model extension (Kimi API);
            # only models whose extension declares them render them, so other
            # models keep ignoring the key entirely.
            dynamic_tools = extension.dynamic_tools(request.messages)
            dynamic_tool_params: List[ChatCompletionToolsParam] = []
            if dynamic_tools:
                try:
                    dynamic_tool_params = [
                        ChatCompletionToolsParam.model_validate(tool)
                        for tool in dynamic_tools
                    ]
                except ValidationError as e:
                    raise ValueError(
                        f"Invalid message-level tool declaration: {e}") from e
            _configure_parser_special_token_decoding(
                sampling_params,
                reasoning_parser_name=reasoning_parser_name,
                tool_parser_name=self.tool_parser,
                has_tools=bool(request.tools) or bool(dynamic_tools))
            all_tools = (request.tools or []) + dynamic_tool_params
            if self.tool_parser and all_tools:
                # When strict=True on any tool, apply constrained decoding
                # via structural tags (only if response_format doesn't already
                # set guided decoding). Dynamic tools must be in the grammar
                # too, or their calls would be masked out.
                if sampling_params.guided_decoding is None and not required_tool_grammar:
                    if (forced_tool_name is not None
                            and _parser_extracts_forced_tool_calls(
                                self.tool_parser)):
                        # This parser handles forced calls by extracting them
                        # from its own native markup (see
                        # ``BaseToolParser.extracts_forced_tool_calls``), so
                        # there is no grammar to build and none is needed. Leave
                        # the prompt and sampling params alone; the
                        # post-processor runs the parser and takes the call from
                        # there. Falling through to the grammar path below would
                        # reject these parsers outright, because none of them
                        # supports structural tags.
                        pass
                    elif forced_tool_name is not None:
                        # Forced-call path: build the begin-prefix + JSON-schema
                        # guided decoding. The prefix is injected into the
                        # prompt after ``apply_chat_template`` below.
                        try:
                            forced_tool_begin_prefix, forced_guided = (
                                _build_forced_tool_call_decoding(
                                    request.tools, self.tool_parser,
                                    forced_tool_name))
                        except ValueError as e:
                            raise serving_error(
                                message=str(e),
                                err_type="BadRequestError",
                                status_code=HTTPStatus.BAD_REQUEST)
                        sampling_params.guided_decoding = forced_guided
                    else:
                        # Strict-tools path: structural tags with the existing
                        # ``triggered_tags`` semantics. Engages only when at
                        # least one tool has ``strict=True``.
                        strict_guided = _build_tool_strict_guided_decoding_params(
                            all_tools, self.tool_parser)
                        if strict_guided is not None:
                            sampling_params.guided_decoding = strict_guided
            elif forced_tool_name is not None:
                # tools were provided but the server has no tool_parser configured.
                raise serving_error(
                    message=("tool_choice with a named function requires the "
                             "server to be started with --tool_parser."),
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST)
            postproc_args = ChatPostprocArgs.from_request(request)
            self._apply_spec_decode_stats_opt_in(postproc_args)
            # Some models' prompt-token accounting excludes trailing rendered
            # tokens (kimi_k3: the generation channel opener); the model still
            # sees the full prompt.
            postproc_args.num_prompt_tokens_offset = (
                render_resources.prompt_tokens_excluded_from_usage(
                    add_generation_prompt=request.add_generation_prompt,
                    pre_tokenized=request.prompt_token_ids is not None
                    or request.prompt_token_ids_b64 is not None,
                    request_chat_template=request.chat_template,
                ))
            if dynamic_tool_params:
                # The tool parser must see dynamic tools to recognize their
                # calls in the model output.
                postproc_args.tools = (postproc_args.tools
                                       or []) + dynamic_tool_params
            self._validate_internal_disagg_request(request, ctx)
            disaggregated_params = to_llm_disaggregated_params(
                request.disaggregated_params)

            # A generation-only worker already has prompt_token_ids with the
            # placeholders expanded and the KV behind them over the
            # transceiver, so the media still on the relayed messages is not
            # its to resolve.
            #
            # Decided before the fetch rather than after it: resolving
            # downloads and decodes every item a second time, and fails
            # outright on a reference only the context worker could read --
            # a node-local path, or a single-use or expired URL.
            #
            # prompt_token_ids_b64 counts too; it is decoded into
            # prompt_token_ids further below.
            resolve_media = not (disaggregated_params is not None
                                 and disaggregated_params.request_type
                                 == "generation_only" and
                                 (request.prompt_token_ids is not None
                                  or request.prompt_token_ids_b64))

            try:
                conversation, mm_coroutines, mm_placeholder_counts, mm_item_order = parse_chat_messages_coroutines(
                    request.messages,
                    self.model_config,
                    self.multimodal_server_config,
                    request_media_io_kwargs=request.media_io_kwargs,
                    resolve_media=resolve_media,
                )
            except ValidationError:
                # ValidatorIterator rejects extra fields; fall back to raw JSON.
                raw_body = await ctx.json()
                raw_messages = raw_body.get("messages", [])
                conversation, mm_coroutines, mm_placeholder_counts, mm_item_order = parse_chat_messages_coroutines(
                    raw_messages,
                    self.model_config,
                    self.multimodal_server_config,
                    request_media_io_kwargs=request.media_io_kwargs,
                    resolve_media=resolve_media,
                )

            # Decode base64 int32 prompt_token_ids relayed by the orchestrator.
            if request.prompt_token_ids is None and request.prompt_token_ids_b64:
                import numpy as np
                request.prompt_token_ids = np.frombuffer(
                    base64.b64decode(request.prompt_token_ids_b64),
                    dtype=np.int32).tolist()

            rendered_prompt = None
            if request.prompt_token_ids is not None:
                prompt = request.prompt_token_ids
            else:
                # Render (and append the forced named-tool prefix, see
                # ``_build_forced_tool_call_decoding``) while the media loads.
                # Tokenization stays with the engine's input processor.
                render_task = render_conversation(
                    render_resources,
                    conversation=conversation,
                    mm_placeholder_counts=mm_placeholder_counts,
                    add_generation_prompt=request.add_generation_prompt,
                    tools=tool_dicts,
                    documents=request.documents,
                    chat_template=request.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs,
                    injected_chat_template_kwargs=request.
                    injected_chat_template_kwargs,
                    forced_prefix=forced_tool_begin_prefix,
                )
                rendered, (mm_data, mm_embeddings) = await asyncio.gather(
                    render_task, mm_coroutines)
                if rendered.text is None:
                    prompt = rendered.token_ids
                else:
                    prompt = rendered.text
                    # Captured before the forced-tool prefix is appended:
                    # resolve_prefilled_thinking() inspects what the chat
                    # template rendered, and the prefix is not part of it.
                    rendered_prompt = rendered.template_text
            prompt = prompt_inputs(prompt)

            if request.prompt_token_ids is not None:
                mm_data, mm_embeddings = await mm_coroutines
            if mm_data:
                self._attach_multimodal_data(prompt, request, mm_data,
                                             mm_item_order)
            if mm_embeddings:
                prompt["multi_modal_embeddings"] = mm_embeddings
            if mm_data and mm_embeddings:
                raise ValueError(
                    "Passing 'multi_modal_data' and 'multi_modal_embeddings' at the same time is not supported."
                )

            if request.mm_processor_kwargs:
                prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

            thinking = None
            postproc_args.reasoning_parser = self.generator.args.reasoning_parser
            # Templates that prefill <think>/</think> leave the marker in the
            # prompt, so the request kwargs alone cannot tell the parser which
            # mode was rendered. Take it from the prompt instead.
            if postproc_args.reasoning_parser and (
                    ReasoningParserFactory.resolves_thinking_from_prompt(
                        postproc_args.reasoning_parser)):
                if rendered_prompt and request.add_generation_prompt:
                    thinking = ReasoningParserFactory.resolve_prefilled_thinking(
                        postproc_args.reasoning_parser, rendered_prompt)
                if thinking is None and request.disaggregated_params is not None:
                    # Generation worker: it never rendered, so use the mode the
                    # context worker resolved and relayed.
                    thinking = request.disaggregated_params.resolved_thinking
                    if thinking is None:
                        _warn_unresolvable_thinking_once(
                            postproc_args.reasoning_parser)
                if thinking is not None:
                    # Both keys, because the parser ORs them: leaving a stale
                    # `thinking` in place would override what we resolved.
                    postproc_args.chat_template_kwargs = {
                        **(request.chat_template_kwargs or {}),
                        "thinking": thinking,
                        "enable_thinking": thinking,
                    }
            if required_tool_grammar:
                if thinking is None:
                    raise ValueError(
                        "Cannot resolve required-tool reasoning state from the rendered prompt."
                    )
                parser = ToolParserFactory.create_tool_parser(self.tool_parser)
                build_required = functools.partial(
                    parser.build_required_structural_tag,
                    request.tools,
                    thinking=thinking)
                required_format = await asyncio.get_running_loop(
                ).run_in_executor(self._input_proc_executor, build_required)
                if required_format is None:
                    raise ValueError(
                        "The configured tool parser does not support required tools."
                    )
                sampling_params.guided_decoding = GuidedDecodingParams(
                    structural_tag=required_format.model_dump_json(
                        by_alias=True))
                postproc_args.tool_parser_grammar_constrained = True
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

            trace_headers = (None if ctx is None else
                             tracing.extract_trace_headers(ctx.headers))

            resolve_request_conversation_id(
                request, None if ctx is None else ctx.headers)
            conversation_params = to_llm_conversation_params(
                request.conversation_params)
            scheduling_params = self._get_scheduling_params(request, ctx)

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
                kv_hints=request.kv_hints,
                trace_headers=trace_headers,
                scheduling_params=scheduling_params,
                priority=request.priority
                if request.priority is not None else DEFAULT_REQUEST_PRIORITY,
            )
            asyncio.create_task(self.await_disconnected(ctx, promise))
            if not self.postproc_worker_enabled:
                postproc_args.tokenizer = self.tokenizer
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

            if request.stream:
                return chat_stream_generator(promise,
                                             postproc_params,
                                             server=self,
                                             ctx=ctx)
            else:
                response = await self._create_chat_response(
                    promise, postproc_params, ctx, disaggregated_params)
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
                return response
        except ServingError:
            raise
        except CppExecutorError as e:
            logger.error(traceback.format_exc())
            # If internal executor error is raised, shutdown the server
            _record_generator_termination(self.generator)
            signal.raise_signal(signal.SIGINT)
            raise EngineFatalError() from e
        except ValueError as e:
            raise serving_error(str(e)) from e
        except Exception as e:
            logger.error(traceback.format_exc())
            raise serving_error(str(e)) from e

    async def chat_harmony(self, request: ChatCompletionRequest,
                           raw_request: Request) -> Response:
        """Chat Completion API with harmony format support.

        Supports both streaming and non-streaming modes.
        """

        async def create_streaming_generator(promise: RequestOutput,
                                             postproc_params: PostprocParams):
            try:
                first_response = await anext(promise)
                raw_request.state.server_first_token_time = (
                    self._adjusted_steady_clock.now())
                pp_results = self._postproc.apply(first_response,
                                                  postproc_params)
                for pp_res in pp_results:
                    yield pp_res
                res = first_response
                async for res in promise:
                    pp_results = self._postproc.apply(res, postproc_params)
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
            # Named tool_choice (forced function call) is not yet supported on
            # the harmony / GPT-OSS path; return a clear 4xx pointing to the
            # follow-up sub-task rather than silently falling back to "auto".
            # See TRTLLM-12758 (and its harmony follow-up).
            if isinstance(request.tool_choice,
                          ChatCompletionNamedToolChoiceParam):
                return self.create_error_response(
                    message=("tool_choice with a named function is not yet "
                             "supported for harmony / GPT-OSS models. Tracking "
                             "follow-up: harmony sub-task of TRTLLM-12758."),
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST)
            if request.tool_choice == "required":
                # The harmony path treats unknown tool_choice values like
                # "auto"; reject instead of silently degrading.
                raise ValueError(
                    "tool_choice='required' is not supported for this model.")
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
            # The server-level effective flag already enables engine metrics
            # for header or JSONL output.
            if len(os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", "")) > 0:
                sampling_params.return_perf_metrics = True
            self._validate_internal_disagg_request(request, raw_request)
            disaggregated_params = to_llm_disaggregated_params(
                request.disaggregated_params)
            trace_headers = (None if raw_request is None else
                             tracing.extract_trace_headers(raw_request.headers))

            postproc_args = ChatCompletionPostprocArgs.from_request(request)
            # No spec-decode opt-in here on purpose. The Harmony handlers build
            # their choices in harmony_adapter, which carries no per-request
            # spec-decode data at all -- avg_decoded_tokens_per_iter is absent
            # from that path too -- so setting the flag would configure
            # something nothing reads. Extending Harmony should cover both
            # fields together; handle_non_streaming_response would need the
            # GenerationResult threaded through, as it currently receives only
            # the outputs.
            postproc_params = PostprocParams(
                post_processor=chat_harmony_streaming_post_processor
                if request.stream else chat_harmony_post_processor,
                postproc_args=postproc_args,
            )

            resolve_request_conversation_id(
                request, None if raw_request is None else raw_request.headers)
            conversation_params = to_llm_conversation_params(
                request.conversation_params)
            scheduling_params = self._get_scheduling_params(
                request, raw_request)

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
                kv_hints=request.kv_hints,
                priority=request.priority
                if request.priority is not None else DEFAULT_REQUEST_PRIORITY,
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

    async def start_profile(
            self,
            request: Optional[StartProfileRequest] = None) -> JSONResponse:
        """Start runtime profiling in the backend engine.

        Request body (all optional): ``output_dir``, ``num_steps``,
        ``start_step``, ``activities``. See ``StartProfileRequest`` for
        descriptions.

        The backend ``PyExecutor.start_profile`` schedules the profile
        window and the broadcasted ``PROFILE_START_REQUEST_ID`` queue
        item wakes the executor loop, so no tickle-via-generation is
        needed on this side — the captured chrome trace stays free of
        synthetic single-token forward passes.

        The underlying ``GenerationExecutor.start_profile`` call may
        block (it waits for the worker subprocess to ack on the
        IPC-proxy path, up to ~60s). We run it on a worker thread via
        ``asyncio.to_thread`` so the FastAPI event loop stays
        responsive to other endpoints during that wait.
        """
        if request is None:
            request = StartProfileRequest()
        try:
            await asyncio.to_thread(
                self.generator.start_profile,
                output_dir=request.output_dir,
                num_steps=request.num_steps,
                start_step=request.start_step,
                activities=request.activities,
            )
        except RuntimeError as e:
            # ``PyExecutor.start_profile`` raises RuntimeError when a
            # profile window is already active or pending. Surface this
            # to the caller as 409 so they can distinguish it from a
            # generic backend failure (which keeps 500).
            msg = str(e)
            if "already in progress" in msg or "pending" in msg:
                logger.info(f"/start_profile rejected: {msg}")
                return JSONResponse(content={
                    "success": False,
                    "message": msg
                },
                                    status_code=409)
            logger.error(f"/start_profile failed: {e}")
            return JSONResponse(content={
                "success": False,
                "message": msg
            },
                                status_code=500)
        except (OSError, ValueError, TimeoutError) as e:
            # OSError: filesystem/IPC failures while preparing the
            # profile window. ValueError: schema-level rejections that
            # slipped past Pydantic. TimeoutError: ack-queue wait gave
            # up. Anything truly unexpected is allowed to propagate to
            # FastAPI's middleware so we get a real stack trace in the
            # server log instead of swallowing it as a generic 500.
            logger.error(f"/start_profile failed: {e}")
            return JSONResponse(content={
                "success": False,
                "message": str(e)
            },
                                status_code=500)

        return JSONResponse(content={"message": "Profiling started"})

    async def stop_profile(self) -> JSONResponse:
        """Stop any in-progress runtime profiling and flush traces.

        The backend ``PyExecutor.stop_profile`` schedules the stop and
        the broadcasted ``PROFILE_STOP_REQUEST_ID`` queue item wakes the
        executor loop. The call blocks until ``profile_step()`` has
        actually fired the stop, so by the time this handler returns 200
        the chrome trace is on disk. No synthetic ``generate_async([0])``
        tickle is submitted, so the captured trace is free of HTTP-layer
        events.

        The wait can take up to ~35s on the IPC-proxy path. We run the
        blocking call on a worker thread via ``asyncio.to_thread`` so
        the FastAPI event loop stays responsive to other endpoints
        (notably ``/health`` for liveness checks) during the flush.
        """
        try:
            await asyncio.to_thread(self.generator.stop_profile)
        except (RuntimeError, OSError, TimeoutError) as e:
            # RuntimeError: backend rejected the stop or broadcast
            # enqueue failed. OSError: filesystem error flushing the
            # trace. TimeoutError: ack-queue wait gave up. Truly
            # unexpected exceptions propagate to FastAPI so they show
            # up as a real stack trace rather than a swallowed 500.
            logger.error(f"/stop_profile failed: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)

        return JSONResponse(content={"message": "Profiling stopped"})

    @staticmethod
    def _runtime_control_error_response(error: Exception) -> JSONResponse:
        if isinstance(error, ValueError):
            status_code = HTTPStatus.BAD_REQUEST
        elif (isinstance(error, RuntimeError) and
              ("already parked" in str(error) or "while state is" in str(error)
               or "admission transition" in str(error))):
            status_code = HTTPStatus.CONFLICT
        else:
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
        return JSONResponse(status_code=status_code,
                            content={"error": {
                                "message": str(error)
                            }})

    async def _call_runtime_memory_method(self, method: str,
                                          tags: Optional[List[str]]) -> object:
        operation = getattr(self.generator, method)
        if isinstance(self.generator, AsyncLLM):
            return await operation(tags)
        return await asyncio.to_thread(operation, tags)

    async def runtime_release_memory(
        self,
        request: Optional[RuntimeMemoryUpdateRequest] = None,
    ) -> JSONResponse:
        try:
            if request is not None and request.tags == []:
                raise ValueError("Runtime memory tags must not be empty.")
            tags = None if request is None else request.tags
            await self._call_runtime_memory_method("release", tags)
        except Exception as error:
            return self._runtime_control_error_response(error)
        return JSONResponse(content={"status": "success"})

    async def runtime_resume_memory(
        self,
        request: Optional[RuntimeMemoryUpdateRequest] = None,
    ) -> JSONResponse:
        try:
            if request is not None and request.tags == []:
                raise ValueError("Runtime memory tags must not be empty.")
            tags = None if request is None else request.tags
            await self._call_runtime_memory_method("resume", tags)
        except Exception as error:
            return self._runtime_control_error_response(error)
        return JSONResponse(content={"status": "success"})

    async def runtime_memory_status(self) -> JSONResponse:
        try:
            if isinstance(self.generator, AsyncLLM):
                status = await self.generator.get_memory_status()
            else:
                status = await asyncio.to_thread(
                    self.generator.get_memory_status)
        except Exception as error:
            return JSONResponse(status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                                content={"error": {
                                    "message": str(error)
                                }})
        return JSONResponse(content=status.model_dump(mode="json"))

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

    async def _live_tokens_per_block(self) -> Optional[int]:
        """Return the runtime's effective KV block size, or None if unknown.

        The executor layer already turns RPC failures into an empty dict, so
        the only failure left to absorb here is ``encode_only``, which rejects
        the call outright. Generators without a KV cache (VisualGen) have no
        such method. Both mean "fall back to the configured value".
        """
        get_capacity = getattr(self.generator, "get_kv_cache_capacity", None)
        if get_capacity is None:
            return None
        try:
            # Off-loop: the RPC blocks, and this worker's own heartbeat task
            # shares this loop, so stalling it here can lapse its registration.
            capacity = await asyncio.to_thread(get_capacity)
        except RuntimeError as e:
            logger.debug(f"Could not read live tokens_per_block: {e}")
            return None
        return capacity.get("tokensPerBlock") or None

    async def get_server_info(self) -> JSONResponse:
        # Note: calling self.generator.disaggregated_params and startup_metrics below
        # may trigger an RPC sync call, blocking the server event loop. Since this server_info
        # is usually called only once before accepting requests, it's not a big concern.
        content = {"disaggregated_params": self.generator.disaggregated_params}
        # A router or caller checks this before trusting prompt token ids that
        # another process rendered; it is computed once, then reused.
        render_fingerprint = getattr(self, "_render_fingerprint", None)
        if render_fingerprint is None:
            try:
                # Hashing the vocabulary takes a moment; keep the event loop free.
                render_fingerprint = await asyncio.to_thread(
                    lambda: RenderResources.from_server(self).fingerprint())
                self._render_fingerprint = render_fingerprint
            except Exception:
                logger.warning("Could not compute the rendering fingerprint:\n"
                               f"{traceback.format_exc()}")
        if render_fingerprint is not None:
            content["render_fingerprint"] = render_fingerprint
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
            # The runtime may override the configured block size (e.g. FlashMLA
            # forces 64) in the worker process, so args.kv_cache_config still
            # holds the pre-override value here. A kv-cache-aware router hashes
            # prompts in whatever block size this endpoint publishes, so a
            # stale value makes every block hash miss. Prefer the live value.
            live_tokens_per_block = await self._live_tokens_per_block()
            if live_tokens_per_block is not None:
                content["tokens_per_block"] = live_tokens_per_block
        content["startup_metrics"] = getattr(self.generator, "startup_metrics",
                                             {})
        return JSONResponse(content=content)

    async def openai_image_generation(self, request: ImageGenerationRequest,
                                      raw_request: Request) -> Response:
        """OpenAI-compatible image generation endpoint.

        Follows the OpenAI Images API specification for image generation,
        with ``request.format`` extended to accept tensor payloads
        (``"safetensors"``/``"pt"``) alongside the PNG/WebP/JPEG encoders.
        """
        request_received = raw_request.state.server_arrival_time
        try:
            image_id = f"image_{uuid.uuid4().hex}"

            path_error = self._reject_disabled_path(request.response_format)
            if path_error is not None:
                return path_error

            # Client-side ValueErrors from request translation and
            # parameter validation are 400. Serialization failures below
            # (server-side: missing media, inconsistent batch) fall
            # through to the outer ``except Exception`` → 500 so the
            # client doesn't get blamed for a server-internal failure.
            try:
                params = parse_visual_gen_params(request, self.generator)
                logger.info(
                    f"Generating image: {image_id} with params: {params} and prompt: {request.prompt}"
                )
                image_gen_start = time.perf_counter()
                # Offload the blocking resolve/enqueue off the event loop but
                # await it (bad params → 400 here); then await generation.
                handle = await asyncio.to_thread(self.generator.generate_async,
                                                 request.prompt, params)
                output = await handle.aresult()
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
                    # Report the paths save() actually wrote, not paths_in --
                    # save() may normalize (e.g. fill a missing extension), so
                    # its return is the on-disk location the client will open().
                    saved = output.save(paths_in, format=request.format)
                    data = [
                        self._image_object(request, raw_request, image_id, i,
                                           saved[i]) for i in range(batch_size)
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
                            self._image_object(request, raw_request, image_id,
                                               i, path))
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
            total = self._adjusted_steady_clock.now() - request_received
            headers = build_visual_gen_timing_headers(
                build_visual_gen_server_timings(metrics, total=total))

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

    def _reject_disabled_path(
            self, response_format: Optional[str]) -> Optional[Response]:
        """Return a 400 when ``response_format='path'`` but it is disabled.

        Shares the switch with ``format='path'`` on the request side; see
        :func:`local_media_path_is_disallowed`. Returns ``None`` when allowed.
        """
        if response_format != "path":
            return None
        if local_media_path_is_disallowed():
            return self.create_error_response(
                "response_format='path' is disabled on this server "
                "(TRTLLM_DISALLOW_LOCAL_MEDIA_PATH=1); it returns "
                "server-side filesystem paths and is only meaningful for "
                "co-located clients.",
                err_type="BadRequestError",
                status_code=HTTPStatus.BAD_REQUEST,
            )
        return None

    def _image_object(self, request: Union[ImageGenerationRequest,
                                           ImageEditRequest],
                      raw_request: Request, image_id: str, i: int,
                      path: Path) -> ImageObject:
        """Build the per-item ``ImageObject`` for the ``path``/``url`` transports.

        ``b64_json`` is handled separately. Shared by the generation
        (tensor + encoder) and edit routes so they cannot drift when a
        transport changes.
        """
        if request.response_format == "path":
            return ImageObject(path=str(path), revised_prompt=request.prompt)
        return ImageObject(url=self._build_image_content_url(
            raw_request, image_id, i),
                           revised_prompt=request.prompt)

    async def _parse_image_edit_request(
        self,
        raw_request: Request,
    ) -> ImageEditRequest:
        """Parse an image edit request from JSON or multipart form data."""
        content_type = raw_request.headers.get("content-type", "")
        normalized_content_type = content_type.lower()

        if "application/json" in normalized_content_type:
            body = await raw_request.json()
            if not isinstance(body, dict):
                raise ValueError(
                    "JSON image edit request body must be an object")
            return ImageEditRequest(**body)

        if "multipart/form-data" in normalized_content_type:
            form = await raw_request.form()
            data = {}
            for key in form:
                values = form.getlist(key)
                if key == "image":
                    values = [
                        value for value in values
                        if not (isinstance(value, str) and value == "")
                    ]
                    if not values:
                        continue
                    data[key] = values if len(values) > 1 else values[-1]
                    continue

                value = values[-1]
                if key == "extra_params":
                    if value == "":
                        continue
                    if not isinstance(value, str):
                        raise ValueError(
                            "'extra_params' must be a JSON object string")
                    try:
                        data[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise ValueError(
                            f"'extra_params' must be a JSON object string; {exc}"
                        ) from exc
                    continue
                if value == "":
                    continue
                data[key] = value
            return ImageEditRequest(**data)

        raise ValueError(
            f"Unsupported content-type: {content_type}. Use 'application/json' or 'multipart/form-data'"
        )

    async def openai_image_edit(self, raw_request: Request) -> Response:
        """OpenAI-compatible image editing endpoint."""
        request_received = raw_request.state.server_arrival_time
        if not self._supports_image_edit():
            return self._create_not_supported_error(
                "Image editing is not supported by the loaded visual generation model."
            )

        try:
            image_id = f"image_{uuid.uuid4().hex}"

            try:
                request = await self._parse_image_edit_request(raw_request)
                path_error = self._reject_disabled_path(request.response_format)
                if path_error is not None:
                    return path_error
                params = parse_visual_gen_params(request, self.generator)
                logger.info(
                    f"Editing image: {image_id} with params: {params} and prompt: {request.prompt}"
                )
                image_edit_start = time.perf_counter()
                output = self.generator.generate(inputs=request.prompt,
                                                 params=params)
            except ValidationError as exc:
                return self._render_pydantic_validation_error(exc)
            except ValueError as exc:
                logger.error(f"Image edit request error: {exc}")
                return self.create_error_response(
                    message=str(exc),
                    status_code=HTTPStatus.BAD_REQUEST,
                )

            if output.image is None:
                return self.create_error_response(
                    message="Image editing failed",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            # Model-specific ``extra_params`` stay pipeline-owned. Layered
            # image-edit models such as Qwen-Image-Layered use
            # ``save_layers_to_grid`` to pack all layers into one image here.
            output_images = _normalize_image_output(output.image)
            output_size = _image_output_size(
                output_images[0]) if output_images else None
            pil_format = request.output_format.upper()
            ext = f".{request.output_format}"
            if request.response_format == "b64_json":
                data = [
                    ImageObject(
                        b64_json=base64.b64encode(
                            image_to_bytes(image,
                                           format=pil_format)).decode("utf-8"),
                        revised_prompt=request.prompt,
                    ) for image in output_images
                ]
            else:
                data = []
                for i, image in enumerate(output_images):
                    path = self.media_storage_path / f"{image_id}_{i}{ext}"
                    path.write_bytes(image_to_bytes(image, format=pil_format))
                    data.append(
                        self._image_object(request, raw_request, image_id, i,
                                           path))

            response = ImageGenerationResponse(
                created=int(time.time()),
                data=data,
                output_format=request.output_format,
                size=output_size,
            )

            latency = time.perf_counter() - image_edit_start
            metrics = output.metrics
            generation = metrics.generation if metrics is not None else 0.0
            denoise = metrics.denoise if metrics is not None else 0.0
            logger.info(f"Image {image_id} edited and encoded: "
                        f"latency={latency:.3f}s generation={generation:.3f}s "
                        f"denoise={denoise:.3f}s")
            total = self._adjusted_steady_clock.now() - request_received
            headers = build_visual_gen_timing_headers(
                build_visual_gen_server_timings(metrics, total=total))

            return JSONResponse(content=response.model_dump(), headers=headers)

        except ValidationError as exc:
            return self._render_pydantic_validation_error(exc)
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(
                message=str(e),
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

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
        server = create_uvicorn_server(config)

        async def _register_after_serving():
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

        asyncio.create_task(_register_after_serving())
        await server.serve(sockets=sockets)
