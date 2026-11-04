# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes of the serving layer: one serving class per endpoint."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any, Callable, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.executor.utils import (CONTEXT_LENGTH_EXCEEDED_CODE,
                                         is_context_length_exceeded_message)
from tensorrt_llm.logger import logger
from tensorrt_llm.metrics.collector import MetricsCollector
from tensorrt_llm.serve.openai_protocol import ErrorResponse
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
        # Over-length prompts are detected from the message text because they
        # cross the executor IPC boundary as plain strings (RequestError), so
        # no exception type survives to here.
        code: Union[int, str] = status_code.value
        if is_context_length_exceeded_message(message):
            code = CONTEXT_LENGTH_EXCEEDED_CODE
        error_response = ErrorResponse(message=message,
                                       type=err_type,
                                       code=code)
        return JSONResponse(content=error_response.model_dump(),
                            status_code=status_code.value)


class GenerateBaseServing(BaseServing):
    """Serving classes that submit requests to the generate pipeline."""

    def __init__(self, *, model: str, generator: Any, tokenizer: Any,
                 model_config: Any, collect_perf_metrics: bool,
                 metrics_collector: Optional[MetricsCollector],
                 adjusted_steady_clock: AdjustedSteadyClock,
                 iteration_stats_wakeup_event: asyncio.Event,
                 iteration_stats_collector_task: Callable[
                     [], Optional[asyncio.Task]],
                 internal_disagg_auth_key: Optional[str],
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
            iteration_stats_collector_task)
        self._internal_disagg_auth_key = internal_disagg_auth_key
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
