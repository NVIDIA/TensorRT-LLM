# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes of the serving layer: one serving class per endpoint."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any, Callable, Dict, List, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.executor.postproc_worker import PostprocArgs
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.metrics.collector import MetricsCollector
from tensorrt_llm.serve._base.errors import error_body
from tensorrt_llm.serve.disagg_auth import (
    request_requires_internal_disagg_auth, validate_internal_disagg_request)
from tensorrt_llm.serve.perf_metrics import build_request_metrics_record
from tensorrt_llm.serve.postprocess_handlers import PostprocStrategy
from tensorrt_llm.usage import TerminalOutcome, record_termination_observation


class BaseServing:
    """Base of every serving class.

    Holds only what all serving classes share; each subclass receives its
    dependencies as explicit constructor arguments, never by reading the
    server object.
    """

    @staticmethod
    def create_error_response(
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> Response:
        error_response = error_body(message, err_type, status_code)
        return JSONResponse(content=error_response.model_dump(),
                            status_code=status_code.value)


class GenerateBaseServing(BaseServing):
    """Serving classes that submit requests to the generate pipeline."""

    def __init__(self, *, model: str, generator: Any, tokenizer: Any,
                 model_config: Any, collect_perf_metrics: bool,
                 metrics_collector: Optional[MetricsCollector],
                 adjusted_steady_clock: AdjustedSteadyClock,
                 iteration_stats_wakeup_event: asyncio.Event,
                 iteration_stats_collector_task_getter: Callable[
                     [], Optional[asyncio.Task]],
                 internal_disagg_auth_key: Optional[str],
                 server_role: Optional[ServerRole],
                 per_request_spec_decode_stats: bool,
                 spec_decode_num_spec_tokens: Optional[int],
                 input_proc_executor: ThreadPoolExecutor,
                 postproc: PostprocStrategy) -> None:
        # BaseServing has no constructor: set every attribute here, no
        # super().__init__(...).
        self.model = model
        self.generator = generator
        self.tokenizer = tokenizer
        self.model_config = model_config
        self._collect_perf_metrics = collect_perf_metrics
        self.metrics_collector = metrics_collector
        self._adjusted_steady_clock = adjusted_steady_clock
        self._iteration_stats_wakeup_event = iteration_stats_wakeup_event
        self._iteration_stats_collector_task_getter = (
            iteration_stats_collector_task_getter)
        self._internal_disagg_auth_key = internal_disagg_auth_key
        self.server_role = server_role
        self._per_request_spec_decode_stats = per_request_spec_decode_stats
        self._spec_decode_num_spec_tokens = spec_decode_num_spec_tokens
        self._input_proc_executor = input_proc_executor
        self._postproc = postproc
        # A generate serving object is never VisualGen; postproc_worker_enabled
        # reads this.
        self._is_visual_gen = False

    @property
    def _iteration_stats_collector_task(self) -> Optional[asyncio.Task]:
        # The composition root's lifespan creates the collector task after the
        # serving objects exist and keeps the binding on the root, so this
        # reads the root's current binding instead of a copy taken at
        # construction.
        return self._iteration_stats_collector_task_getter()

    @staticmethod
    def _attach_multimodal_data(
            prompt: Dict[str, Any],
            request: Any,
            mm_data: Any,
            mm_item_order: Optional[List[Any]] = None) -> None:
        """Attach decoded media to the prompt dict handed to the engine.

        When the request carried no ``prompt_token_ids``, ``prompt`` is the
        chat template's output with one placeholder per media item already in
        it, so ``chat_template_applied`` tells the input processor to use it
        verbatim instead of rendering the template a second time.
        """
        prompt["multi_modal_data"] = mm_data
        if request.prompt_token_ids is None:
            prompt["chat_template_applied"] = True
        if mm_item_order:
            prompt["mm_item_order"] = mm_item_order

    async def await_disconnected(self, raw_request: Request, promise):
        if raw_request is None:
            return
        # A batched request is synthesised without a client socket, so
        # is_disconnected() can never become true and this poll would run at
        # 1Hz for the life of the process, holding the Request and its
        # RequestOutput -- one leaked task per batched request.
        if getattr(raw_request.state, "no_client_connection", False):
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
        if self._is_visual_gen:
            return False

        return True if self.generator.args.num_postprocess_workers > 0 else False

    async def _extract_metrics(self, res: RequestOutput, raw_request: Request):
        if not res.finished:
            return
        if self._collect_perf_metrics:
            if raw_request and not getattr(raw_request.state,
                                           "server_first_token_time", None):
                raw_request.state.server_first_token_time = (
                    self._adjusted_steady_clock.now())
            record = build_request_metrics_record(
                res, raw_request, adjusted_clock=self._adjusted_steady_clock)
            if record is not None and raw_request is not None:
                raw_request.state.perf_metrics_records.append(record)
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
            # The collector only runs with enable_iter_perf_stats.
            if self._iteration_stats_collector_task is not None:
                self._iteration_stats_wakeup_event.set()

    def _logit_bias_vocab_size(self) -> int:
        for config in (self.model_config,
                       getattr(self.model_config, "text_config", None)):
            vocab_size = getattr(config, "vocab_size", None)
            if vocab_size is not None:
                return int(vocab_size)
        return int(self.tokenizer.tokenizer.vocab_size)

    def _validate_internal_disagg_request(
            self, request, raw_request: Optional[Request]) -> None:
        if (request_requires_internal_disagg_auth(request)
                and not self._has_cache_transceiver_config()):
            raise ValueError("Protected disaggregated request fields require "
                             "cache_transceiver_config to be configured")
        headers = None if raw_request is None else raw_request.headers
        validate_internal_disagg_request(
            getattr(self, "_internal_disagg_auth_key", None), request, headers)

    def _has_cache_transceiver_config(self) -> bool:
        cache_transceiver_config = getattr(
            getattr(self.generator, "args", None), "cache_transceiver_config",
            None)
        if isinstance(cache_transceiver_config, dict):
            return cache_transceiver_config.get("backend") is not None
        return (cache_transceiver_config is not None and getattr(
            cache_transceiver_config, "backend", None) is not None)

    def _apply_spec_decode_stats_opt_in(self,
                                        postproc_args: PostprocArgs) -> None:
        """Enable per-request spec-decode stats when the server opted in.

        Server-side only, deliberately: per_request_spec_decode_stats in the
        YAML config is the entire opt-in, and a client sends nothing extra.
        This differs from return_perf_metrics, which additionally requires a
        per-request X-TRTLLM-return-metrics header.

        The reason is that benchmarking clients discover this payload by shape
        rather than being told which engine they are talking to -- requiring a
        vendor-specific request header would mean the client has to know it is
        talking to TensorRT-LLM before it can find out, which it does not. The
        cost stays opt-in because an operator who does not set the YAML field
        pays nothing, and one who does has asked for exactly this.
        """
        # getattr: servers built without __init__ (object.__new__, as tests
        # do) must behave as not opted in rather than fail every request.
        if not getattr(self, "_per_request_spec_decode_stats", False):
            return
        postproc_args.return_spec_decode_stats = True
        postproc_args.spec_decode_num_spec_tokens = getattr(
            self, "_spec_decode_num_spec_tokens", None)


def _record_generator_termination(generator) -> None:
    """Classify a fatal generator error without exposing exception details."""
    component = "llm"
    reporting_source = "self"
    termination_kind = "exception"
    try:
        if generator.engine_in_worker_process:
            component = "engine_worker"
            reporting_source = "executor_proxy"
            termination_kind = "worker_failure"
    except Exception:
        pass
    record_termination_observation(
        TerminalOutcome(
            termination_kind=termination_kind,
            component=component,
            reporting_source=reporting_source,
            exit_code_known=False
            if termination_kind == "worker_failure" else None,
        ))
