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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Version of the prepared-request wire format. Bump on an incompatible change;
# a worker rejects a version it does not understand.
GENERATE_REQUEST_SCHEMA_VERSION = 1


class UnsupportedRenderError(ValueError):
    """The request needs a capability the prompt-preparation core does not provide.

    Raised instead of returning approximate token ids (multimodal input, a model
    whose input processor is not the default text tokenizer, and so on).
    """


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
    # False when ``token_ids`` must not be forwarded as the executed prompt
    # (see ``untrusted_reason``); a consumer then renders the prompt itself.
    tokens_trusted: bool = True
    untrusted_reason: Optional[str] = None
    # Multimodal payload, filled only when the caller asked to load media.
    mm_data: Optional[Dict[str, Any]] = None
    mm_embeddings: Optional[Dict[str, Any]] = None
    # Decisions derived from the rendered prompt (see :class:`PreparedContext`).
    context: Dict[str, Any] = field(default_factory=dict)


class PreparedContext(BaseModel):
    """Decisions the chat route derives from the rendered prompt.

    A worker handed token ids never sees the rendered text, so it cannot make these
    decisions again; they are computed once where the prompt is rendered and carried
    with the ids.
    """

    model_config = ConfigDict(extra="forbid")

    # Rendered prompt tokens that are not reported as prompt usage (kimi_k3: the
    # generation channel opener). Zero when a server-level chat template applies.
    prompt_tokens_excluded_from_usage: int = Field(default=0, ge=0)
    # Reasoning mode read off the rendered prompt by parsers that take it from there
    # (a template that prefills ``<think>`` or ``</think>``); ``None`` when not applicable.
    resolved_thinking: Optional[bool] = None


class GenerateRequest(BaseModel):
    """Prepared request: the prompt tokens plus the original request to execute.

    Returned by the render routes and consumed by ``POST /generate``.

    **Contract: prompt replay.** What is guaranteed is prompt preparation: when the
    ``fingerprint`` matches the worker's, ``token_ids`` are the tokens the worker's own
    chat route would have rendered, and ``context`` carries the decisions derived from
    the rendered prompt. Everything else stays worker-owned: the ``request`` field is
    the original request body (chat-completions, or completions with the prompt replaced
    by ``token_ids``) and is validated and executed by the worker's normal route, so
    sampling defaults, guided decoding, tool and reasoning parsing, streaming, usage and
    errors are the worker's. A caller that needs identical generation behavior must also
    ensure the worker's execution configuration (generation defaults, parsers) is the one
    it expects; the rendering fingerprint does not cover it.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = GENERATE_REQUEST_SCHEMA_VERSION
    # Which inference route the ``request`` body belongs to.
    kind: Literal["chat", "completion"] = "chat"
    # Rendering-configuration fingerprint of the renderer that produced the ids.
    fingerprint: Dict[str, Any]
    token_ids: List[int]
    # False when the ids must not be executed as is; the consumer re-renders.
    tokens_trusted: bool = True
    untrusted_reason: Optional[str] = None
    # Decisions derived from the rendered prompt, preserved for the executing worker.
    context: PreparedContext = Field(default_factory=PreparedContext)
    # The original request body.
    request: Dict[str, Any] = Field(default_factory=dict)
