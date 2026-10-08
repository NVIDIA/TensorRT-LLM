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
"""The render endpoints' logic: request body in, prepared ``GenerateRequest`` out.

One :class:`ServingRender` backs both ways of exposing rendering over HTTP: the
routes mounted on a normal ``trtllm-serve`` worker and the standalone CPU-only
renderer process.
"""

from __future__ import annotations

import asyncio
import copy
import os
from typing import Any, Callable, Dict, List, Optional

from .chat import arender_chat, tokenize_rendered
from .prompt_types import GenerateRequest, UnsupportedRenderError
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

    async def render_chat(self, body: Dict[str, Any]) -> GenerateRequest:
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
        try:
            rendered = await arender_chat(
                request,
                self._resources_factory(),
                tokenize=True,
                # The route's fallback when the request model rejects the messages.
                raw_messages=copy.deepcopy(original.get("messages")),
            )
        except UnsupportedRenderError as error:
            raise RenderRequestError(400, str(error)) from error
        except ValueError as error:
            raise RenderRequestError(400, str(error)) from error
        return GenerateRequest(
            kind="chat",
            fingerprint=self.fingerprint(),
            token_ids=rendered.token_ids,
            tokens_trusted=rendered.tokens_trusted,
            untrusted_reason=rendered.untrusted_reason,
            request=original,
        )

    async def render_completion(self, body: Dict[str, Any]) -> List[GenerateRequest]:
        """Render a completions body into one prepared request per prompt."""
        from pydantic import ValidationError

        from tensorrt_llm.serve.openai_protocol import CompletionRequest

        body = strip_ignored_fields(body)
        try:
            request = CompletionRequest.model_validate(body)
        except ValidationError as error:
            raise RenderRequestError(400, str(error)) from error
        resources = self._resources_factory()
        results: List[GenerateRequest] = []
        for prompt in _normalize_completion_prompts(request.prompt):
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
                GenerateRequest(
                    kind="completion",
                    fingerprint=self.fingerprint(),
                    token_ids=token_ids,
                    request={**body, "prompt": token_ids},
                )
            )
        return results
