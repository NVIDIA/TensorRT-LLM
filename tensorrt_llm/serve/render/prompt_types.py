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

from dataclasses import dataclass
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


class GenerateRequest(BaseModel):
    """Prepared request: the prompt tokens plus everything needed to execute it.

    Returned by the render routes and consumed by ``POST /generate``. The
    ``request`` field is the original request body (chat-completions, or
    completions with the prompt replaced by ``token_ids``), so tool, response
    format, streaming and sampling semantics are exactly those of the normal
    route; ``token_ids`` replaces rendering and tokenization.
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
    # The original request body.
    request: Dict[str, Any] = Field(default_factory=dict)
