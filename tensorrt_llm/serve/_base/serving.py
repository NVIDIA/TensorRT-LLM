# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes of the serving layer: one serving class per endpoint."""

import asyncio
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm.executor.utils import (CONTEXT_LENGTH_EXCEEDED_CODE,
                                         is_context_length_exceeded_message)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ErrorResponse
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
                 model_config: Any) -> None:
        # BaseServing has no constructor: set every attribute here, no
        # super().__init__(...).
        self.model = model
        self.generator = generator
        self.tokenizer = tokenizer
        self.model_config = model_config

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
