# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/chat/completions`` on the multimodal-encoder role."""

import asyncio
import signal
import traceback
from http import HTTPStatus
from typing import Any, List

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.inputs import prompt_inputs
from tensorrt_llm.inputs.utils import ConversationMessage
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.serve._base.serving import (GenerateBaseServing,
                                              _record_generator_termination)
from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice,
    ChatMessage, UsageInfo, ensure_request_chat_template_allowed)
from tensorrt_llm.serve.render import RenderResources, render_conversation


class MMEncoderServing(GenerateBaseServing):
    """``POST /v1/chat/completions`` on the multimodal-encoder role.

    Returns the multimodal embedding handles of the request instead of
    generated text. Rendering uses the ``RenderResources`` the composition
    root built once, never resources read back from this object.
    """

    def __init__(self, *, render_resources: RenderResources,
                 multimodal_server_config: Any,
                 allow_request_chat_template: bool, **base_kwargs: Any) -> None:
        super().__init__(**base_kwargs)
        self._render_resources = render_resources
        self.multimodal_server_config = multimodal_server_config
        self.allow_request_chat_template = allow_request_chat_template

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
            # Same pipeline as the chat route: the server template and the
            # model extension's tool serialization apply here too.
            render_resources = self._render_resources
            tool_dicts = None if request.tools is None else [
                render_resources.extension.serialize_tool(tool)
                for tool in request.tools
            ]

            try:
                conversation, mm_coroutines, mm_placeholder_counts, mm_item_order = parse_chat_messages_coroutines(
                    request.messages,
                    self.model_config,
                    self.multimodal_server_config,
                    request_media_io_kwargs=request.media_io_kwargs,
                )
            except ValidationError:
                # ValidatorIterator rejects extra fields; fall back to raw JSON.
                raw_body = await raw_request.json()
                raw_messages = raw_body.get("messages", [])
                conversation, mm_coroutines, mm_placeholder_counts, mm_item_order = parse_chat_messages_coroutines(
                    raw_messages,
                    self.model_config,
                    self.multimodal_server_config,
                    request_media_io_kwargs=request.media_io_kwargs,
                )

            if request.prompt_token_ids is not None:
                prompt = request.prompt_token_ids
            else:
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
                )
                rendered, (mm_data, mm_embeddings) = await asyncio.gather(
                    render_task, mm_coroutines)
                prompt = (rendered.text
                          if rendered.text is not None else rendered.token_ids)
            prompt = prompt_inputs(prompt)

            if request.prompt_token_ids is not None:
                mm_data, mm_embeddings = await mm_coroutines
            if mm_embeddings:
                raise ValueError("Cannot use multimodal embeddings as input")
            if mm_data is not None:
                self._attach_multimodal_data(prompt, request, mm_data,
                                             mm_item_order)

            promise = self.generator.generate_async(inputs=prompt, )
            asyncio.create_task(self.await_disconnected(raw_request, promise))

            response = await create_mm_embedding_response(promise)
            return JSONResponse(content=response.model_dump())

        except CppExecutorError:
            logger.error(traceback.format_exc())
            # If internal executor error is raised, shutdown the server
            _record_generator_termination(self.generator)
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(str(e))
