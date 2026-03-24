# Copyright (c) 2026, NVIDIA CORPORATION.
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
"""Control-plane server for non-inference runtime operations.

Handles KV cache management and other executor-level control operations,
separated from the OpenAI-compatible inference server. Communicates with
PyExecutor via a dedicated IPC queue, bypassing the LLM/Proxy/Worker chain.
"""

import traceback
from http import HTTPStatus
from typing import Callable, List, Optional

from fastapi import FastAPI
from starlette.responses import JSONResponse, Response

from tensorrt_llm.executor.request import TruncateKVCacheRequest
from tensorrt_llm.inputs.utils import ConversationMessage, apply_chat_template
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines
from tensorrt_llm.serve.openai_protocol import KVCacheTruncateRequest


class ControlPlaneServer:
    """Server for executor-level control operations.

    This class owns a direct IPC channel to the PyExecutor's control queue,
    allowing control requests (e.g. KV cache truncation) to reach the executor
    without passing through the LLM API, Proxy, or Worker dispatch chain.
    """

    def __init__(
        self,
        control_queue,
        tokenizer,
        model_config,
        processor=None,
        harmony_adapter_factory: Optional[Callable] = None,
    ):
        self.control_queue = control_queue
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.processor = processor
        self._harmony_adapter_factory = harmony_adapter_factory
        self._harmony_adapter = None

    def register_routes(self, app: FastAPI):
        if self._harmony_adapter_factory is not None:
            handler = self._truncate_kv_cache_harmony
        else:
            handler = self._truncate_kv_cache
        app.add_api_route("/_control/kv_cache/truncate", handler, methods=["POST"])

    def _create_error_response(self, message: str, status_code: int) -> JSONResponse:
        return JSONResponse(content={"error": message}, status_code=status_code)

    def _convert_messages(
        self,
        messages,
        tool_dicts,
        add_generation_prompt,
        documents,
        chat_template,
        chat_template_kwargs,
    ) -> List[int]:
        """Convert chat messages to token IDs via chat template + tokenization."""
        conversation: List[ConversationMessage] = []
        conversation, _, __ = parse_chat_messages_coroutines(messages, self.model_config, None)
        return apply_chat_template(
            model_type=self.model_config.model_type,
            tokenizer=self.tokenizer,
            processor=self.processor,
            conversation=conversation,
            add_generation_prompt=add_generation_prompt,
            mm_placeholder_counts=[],
            tools=tool_dicts,
            documents=documents,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs or {},
            enable_tokenize=True,
        )

    async def _truncate_kv_cache(self, request: KVCacheTruncateRequest) -> Response:
        try:
            tool_dicts = (
                None if request.tools is None else [tool.model_dump() for tool in request.tools]
            )
            chat_template_kwargs = request.chat_template_kwargs or {}

            messages_to_retain = (
                self._convert_messages(
                    request.messages_to_retain,
                    tool_dicts,
                    request.add_generation_prompt,
                    request.documents,
                    request.chat_template,
                    chat_template_kwargs,
                )
                if request.messages_to_retain
                else []
            )

            messages = (
                self._convert_messages(
                    request.messages,
                    tool_dicts,
                    request.add_generation_prompt,
                    request.documents,
                    request.chat_template,
                    chat_template_kwargs,
                )
                if request.messages
                else []
            )

            self.control_queue.put(
                TruncateKVCacheRequest(
                    messages_to_retain=messages_to_retain,
                    messages=messages,
                )
            )
            return Response(status_code=200)
        except Exception as e:
            logger.error(traceback.format_exc())
            return self._create_error_response(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)

    def _truncate_kv_cache_harmony(self, request: KVCacheTruncateRequest) -> Response:
        try:
            if self._harmony_adapter is None:
                self._harmony_adapter = self._harmony_adapter_factory()

            tools_dict = None
            if request.tools:
                tools_dict = [tool.model_dump() for tool in request.tools]

            from tensorrt_llm.serve.harmony_adapter import maybe_transform_reasoning_effort

            reasoning_effort = maybe_transform_reasoning_effort(request.reasoning_effort)

            messages_to_retain = self._harmony_adapter.openai_to_harmony_tokens(
                request.messages_to_retain,
                tools_dict,
                reasoning_effort=reasoning_effort,
                tool_choice=request.tool_choice,
            )
            messages = self._harmony_adapter.openai_to_harmony_tokens(
                request.messages,
                tools_dict,
                reasoning_effort=reasoning_effort,
                tool_choice=request.tool_choice,
            )

            self.control_queue.put(
                TruncateKVCacheRequest(
                    messages_to_retain=messages_to_retain,
                    messages=messages,
                )
            )
            return Response(status_code=200)
        except Exception as e:
            logger.error(traceback.format_exc())
            return self._create_error_response(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)
