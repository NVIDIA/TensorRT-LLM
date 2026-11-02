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
"""Result and wire types of the prompt-preparation core."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Version of the prepared-request wire format. Bump on an incompatible change;
# a worker rejects a version it does not understand.
RENDERED_REQUEST_SCHEMA_VERSION = 1


class UnsupportedRenderError(ValueError):
    """The request needs a capability the prompt-preparation core does not provide.

    Raised instead of returning approximate token ids (multimodal input, a model
    whose input processor is not the default text tokenizer, and so on).
    """


class HarmonyConversionError(RuntimeError):
    """The Harmony adapter could not convert the messages (a tool without a description)."""


class UnrenderableMessagesError(UnsupportedRenderError):
    """The messages (object-valued tool-call arguments) can only be rendered from the raw body."""


@dataclass
class RenderedPrompt:
    """A prompt after chat-template rendering and, optionally, tokenization.

    ``text`` is the rendered template string (``None`` when the prompt came out
    of the template as token ids, or from a model-specific renderer such as
    Harmony). ``token_ids`` is set when the caller asked for tokenization or the
    renderer produced ids directly.
    """

    text: Optional[str] = None
    # The template output before any forced-tool prefix was appended.
    template_text: Optional[str] = None
    token_ids: Optional[List[int]] = None
    # The forced named-tool prefix was appended to ``text``.
    prefix_applied: bool = False
    # Decisions derived from the rendered prompt (see :class:`PreparedContext`); a caller
    # that would forward the ids checks them first.
    context: Dict[str, Any] = field(default_factory=dict)


class PreparedContext(BaseModel):
    """Decisions the chat route derives from the rendered prompt.

    A request that carries only ``prompt_token_ids`` (what the router forwards on
    write-back) skips rendering, so the worker cannot make these decisions from the
    text. A caller that would forward ids therefore checks :meth:`is_trivial` and,
    when the prompt carries a decision, leaves the rendering to the worker.
    """

    model_config = ConfigDict(extra="forbid")

    # Rendered prompt tokens that are not reported as prompt usage (kimi_k3: the
    # generation channel opener). Zero when a server-level chat template applies.
    prompt_tokens_excluded_from_usage: int = Field(default=0, ge=0)
    # Reasoning mode read off the rendered prompt (a template that prefills ``<think>`` or
    # ``</think>``), per reasoning parser that takes its mode from there. It is recorded for
    # every such parser, not only the one the renderer was configured with: the renderer's
    # parser setting is not part of the compatibility check. A parser absent from the
    # mapping resolved nothing.
    resolved_thinking: Dict[str, bool] = Field(default_factory=dict)

    def is_trivial(self) -> bool:
        """Whether the worker would decide the same things without this context."""
        return self.prompt_tokens_excluded_from_usage == 0 and not self.resolved_thinking


class ReasoningParserKwargs(BaseModel):
    """What a reasoning parser needs to agree with the renderer on the prompt it saw."""

    model_config = ConfigDict(extra="forbid")

    # The effective chat template kwargs the prompt was rendered with.
    chat_template_kwargs: Dict[str, Any] = Field(default_factory=dict)


class RenderedRequest(BaseModel):
    """What a render route returns: the prompt token ids plus what a consumer needs to use them.

    ``token_ids`` are the tokens the worker's own chat route would render for the original
    request, valid when the renderer's ``render_fingerprint`` equals the worker's (both on
    ``GET /server_info``; compare them once when a worker is admitted, not per response). A
    routing consumer (the Dynamo EPP) reads ``token_ids`` and forwards the original request
    unchanged.

    To have a worker execute the ids without rendering, use its existing token-in routes
    rather than a dedicated endpoint: ``POST /v1/completions`` with ``token_ids`` as the
    prompt, or ``POST /v1/chat/completions`` with the original body plus ``prompt_token_ids``
    (the mechanism the router's write-back already uses), and only for a worker whose
    ``render_fingerprint`` matches. A request with a named ``tool_choice`` is the exception: the
    worker appends a forced tool-call prefix to the prompt it renders and rejects it together
    with ``prompt_token_ids``, so send ``request`` as it is and let the worker render it.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = RENDERED_REQUEST_SCHEMA_VERSION
    # Which inference route the ``request`` body belongs to.
    kind: Literal["chat", "completion"] = "chat"
    # Identifies this prepared request through the rest of the inference path.
    request_id: str = Field(default_factory=lambda: f"render-{uuid.uuid4().hex}")
    token_ids: List[int]
    model: Optional[str] = None
    # Whether reasoning has already ended before the first generated token, read off the
    # rendered prompt for the reasoning parser this renderer is configured with. None when
    # there is no such parser or the prompt does not say: the executing side decides.
    reasoning_ended: Optional[bool] = None
    # Set when the renderer has a reasoning parser, so the worker-side parser agrees with
    # the renderer on flags such as ``enable_thinking``.
    reasoning_parser_kwargs: Optional[ReasoningParserKwargs] = None
    # Copied from the request: a salted prefix cache changes the block hashes a router computes.
    cache_salt: Optional[str] = None
    # The original request body.
    request: Dict[str, Any] = Field(default_factory=dict)
