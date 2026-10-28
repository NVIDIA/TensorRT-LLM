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
"""The render endpoints' logic: request body in, prepared ``RenderedRequest`` out.

One :class:`ServingRender` backs both ways of exposing rendering over HTTP: the
routes mounted on a normal ``trtllm-serve`` worker and the standalone CPU-only
renderer process.

Supported integration. The render routes are *routing-only rendering* for a consumer
such as the Dynamo EPP: it reads ``token_ids`` to compute KV-cache block hashes, forwards
the original request unchanged, and leaves the worker to render it. Such a consumer does
not use ``schema_version``. When the renderer's ``render_fingerprint`` differs from a
worker's (both on ``GET /server_info``, compared once when the worker is admitted), the ids
are an estimate and routing quality, not correctness, is what is at stake. A consumer that
wants the worker to execute the ids without rendering uses the worker's existing token-in
routes (``/v1/completions`` with token ids as the prompt, ``/v1/chat/completions`` with
``prompt_token_ids``); there is no separate execution endpoint here. A named ``tool_choice``
is the exception, as the ``request`` it echoes shows: the worker renders that request itself.
"""

from __future__ import annotations

import asyncio
import copy
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

from .chat import arender_chat, tokenize_rendered
from .prompt_types import (
    PreparedContext,
    ReasoningParserKwargs,
    RenderedRequest,
    UnsupportedRenderError,
)
from .resources import RenderResources

# Top-level request fields that a Dynamo gateway adds for its own routing and
# that the inference request models do not define. They never change the prompt.
DYNAMO_ONLY_FIELDS = ("nvext", "cache_namespace")
_EXTRA_IGNORED_ENV = "TRTLLM_RENDER_IGNORED_FIELDS"


class RenderRequestError(Exception):
    """A request the render endpoints reject, with the HTTP status to answer."""

    def __init__(self, status_code: int, message: str, err_type: str = "BadRequestError"):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.err_type = err_type


def strip_ignored_fields(body: Dict[str, Any]) -> Dict[str, Any]:
    """Drop gateway-only fields so the request validates like a normal one."""
    ignored = set(DYNAMO_ONLY_FIELDS)
    ignored.update(
        name.strip() for name in os.getenv(_EXTRA_IGNORED_ENV, "").split(",") if name.strip()
    )
    return {key: value for key, value in body.items() if key not in ignored}


def _normalize_completion_prompts(prompt: Any) -> List[Any]:
    """The prompts of a completions request, one entry per prompt."""
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        return [prompt]
    if isinstance(prompt, list):
        return list(prompt)
    raise RenderRequestError(400, "Unsupported prompt type.")


def _request_fields(request: Any) -> Dict[str, Any]:
    """The fields of a rendered request that are copied from the inference request."""
    return dict(model=request.model, cache_salt=getattr(request, "cache_salt", None))


def _reasoning_fields(
    request: Any, resources: RenderResources, context: PreparedContext
) -> Dict[str, Any]:
    """``reasoning_ended`` and ``reasoning_parser_kwargs`` for the configured parser."""
    parser = resources.reasoning_parser
    if not parser:
        return {}
    thinking = context.resolved_thinking.get(parser)
    return dict(
        # The mode is "reasoning is open"; the field says "reasoning has ended".
        reasoning_ended=None if thinking is None else not thinking,
        reasoning_parser_kwargs=ReasoningParserKwargs(
            chat_template_kwargs=dict(request.chat_template_kwargs or {})
        ),
    )


class ServingRender:
    """Renders chat and completions requests into prepared generate requests."""

    def __init__(self, resources_factory: Callable[[], RenderResources]):
        self._resources_factory = resources_factory
        self._fingerprint: Optional[Dict[str, Any]] = None

    def fingerprint(self) -> Dict[str, Any]:
        """Fingerprint of the rendering configuration, computed once."""
        if self._fingerprint is None:
            self._fingerprint = self._resources_factory().fingerprint()
        return self._fingerprint

    async def render_chat(self, body: Dict[str, Any]) -> RenderedRequest:
        """Render a chat-completions body into a prepared request."""
        from pydantic import ValidationError

        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        body = strip_ignored_fields(body)
        try:
            request = ChatCompletionRequest.model_validate(body)
        except ValidationError as error:
            raise RenderRequestError(400, str(error)) from error
        # The extension preprocessing mutates the request while rendering, and
        # the executing side applies it again; ship the body as it arrived.
        original = copy.deepcopy(body)
        resources = self._resources_factory()
        try:
            rendered = await arender_chat(
                request,
                resources,
                tokenize=True,
                # The route's fallback when the request model rejects the messages.
                raw_messages=copy.deepcopy(original.get("messages")),
            )
        except UnsupportedRenderError as error:
            raise RenderRequestError(400, str(error)) from error
        except ValueError as error:
            raise RenderRequestError(400, str(error)) from error
        except RuntimeError as error:
            # Harmony reports messages it cannot convert (a tool without a description,
            # null content) as a RuntimeError; the chat route answers those with a 400.
            if not resources.use_harmony:
                raise
            raise RenderRequestError(400, str(error)) from error
        context = PreparedContext.model_validate(rendered.context)
        return RenderedRequest(
            kind="chat",
            request_id=f"chatcmpl-{uuid.uuid4().hex}",
            token_ids=rendered.token_ids,
            request=original,
            **_request_fields(request),
            **_reasoning_fields(request, resources, context),
        )

    async def render_completion(self, body: Dict[str, Any]) -> List[RenderedRequest]:
        """Render a completions body into one prepared request per prompt."""
        from pydantic import ValidationError

        from tensorrt_llm.serve.openai_protocol import CompletionRequest

        body = strip_ignored_fields(body)
        try:
            request = CompletionRequest.model_validate(body)
        except ValidationError as error:
            raise RenderRequestError(400, str(error)) from error
        resources = self._resources_factory()
        results: List[RenderedRequest] = []
        request_id = f"cmpl-{uuid.uuid4().hex}"
        for index, prompt in enumerate(_normalize_completion_prompts(request.prompt)):
            if isinstance(prompt, str):
                try:
                    token_ids = await asyncio.to_thread(
                        tokenize_rendered,
                        resources,
                        prompt,
                        add_special_tokens=request.add_special_tokens,
                        truncate_prompt_tokens=request.truncate_prompt_tokens,
                    )
                except UnsupportedRenderError as error:
                    raise RenderRequestError(400, str(error)) from error
            else:
                token_ids = list(prompt)
            results.append(
                RenderedRequest(
                    kind="completion",
                    request_id=f"{request_id}-{index}",
                    token_ids=token_ids,
                    # The submitted request; a batch is split, so each item carries its own prompt.
                    request={**body, "prompt": prompt},
                    **_request_fields(request),
                )
            )
        return results
