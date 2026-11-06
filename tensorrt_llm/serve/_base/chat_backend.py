# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The chat implementation a wire-dialect serving class runs on."""

from __future__ import annotations

from typing import AsyncIterator, Protocol, Union

from tensorrt_llm.serve._base.errors import ServingError
from tensorrt_llm.serve._base.request_context import RequestContext
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, ChatCompletionResponse

# Non-streaming: the typed response. Streaming: the OpenAI SSE stream, as str
# frames (aggregated) or as the worker's raw bytes (disaggregated).
ChatResult = Union[ChatCompletionResponse, AsyncIterator[str], AsyncIterator[bytes]]


class ChatBackend(Protocol):
    """The chat implementation a wire-dialect serving class runs on.

    The aggregated server implements it over its typed chat core, the
    disaggregated frontend over its prepare, run and finish steps. Each side
    keeps its own streamed model name and error-message extraction, which is
    why both are methods of the backend and not of the serving class.
    """

    async def create_chat_completion(
        self, request: ChatCompletionRequest, ctx: RequestContext
    ) -> ChatResult:
        """Run one chat completion.

        Non-streaming: the typed response. Streaming: the OpenAI SSE stream, as
        str frames (aggregated) or the worker's raw bytes (disaggregated); one
        stream never mixes the two.

        Raises:
            ServingError: The request was rejected; the HTTP status travels
                apart from the body.
            EngineFatalError: The engine failed; SIGINT has been sent.
        """
        ...

    def stream_model(self, request: ChatCompletionRequest) -> str:
        """The model a translated stream announces when no chunk names one."""
        ...

    def error_message(self, error: ServingError) -> str:
        """The client-facing message of an error.

        Each backend extracts it from the error body its own way.
        """
        ...
