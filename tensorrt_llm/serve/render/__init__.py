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
"""Prompt preparation: the one implementation behind rendering and execution.

A chat request becomes a prompt here, for every caller: the chat route, the
KV-aware router, ``count_tokens``, the resource governor, Responses, and the
standalone renderer. The package builds without an executor, model weights or a
GPU. See :mod:`.chat` for the pipeline and :mod:`.resources` for loading.
"""

from .api_router import attach_render_routes, render_endpoints_enabled
from .chat import (
    PreparedChat,
    arender_chat,
    legacy_render_enabled,
    prepare_chat_request,
    render_chat,
    render_conversation,
    tokenize_rendered,
)
from .fingerprint import compute_fingerprint, fingerprints_match
from .generate import attach_generate_route, mount_render_endpoints
from .prompt_types import (
    GENERATE_REQUEST_SCHEMA_VERSION,
    GenerateRequest,
    PreparedContext,
    RenderedPrompt,
    UnsupportedRenderError,
)
from .resources import RenderResources, harmony_enabled_for
from .serving import RenderRequestError, ServingRender

__all__ = [
    "GENERATE_REQUEST_SCHEMA_VERSION",
    "GenerateRequest",
    "PreparedChat",
    "PreparedContext",
    "RenderRequestError",
    "RenderResources",
    "RenderedPrompt",
    "ServingRender",
    "UnsupportedRenderError",
    "arender_chat",
    "attach_generate_route",
    "attach_render_routes",
    "compute_fingerprint",
    "fingerprints_match",
    "harmony_enabled_for",
    "legacy_render_enabled",
    "mount_render_endpoints",
    "prepare_chat_request",
    "render_chat",
    "render_conversation",
    "render_endpoints_enabled",
    "tokenize_rendered",
]
