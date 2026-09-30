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
"""Serving extension for MuseSpark checkpoints (``model_type == "musespark_open_omni"``).

Routes a request-level ``reasoning_effort`` to the MuseSpark chat template.
Structured-output placement is the generic path: the constraint applies as a
raw grammar from the first generated token, with no model-specific trigger.
"""

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.serving_extensions import ServingExtension, register_serving_extension

#: Efforts the MuseSpark chat template understands. The template reads
#: `reasoning_effort` from its chat-template kwargs only; `max` and `none`
#: have no MuseSpark equivalent and leave the template default in place.
_MUSESPARK_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


@register_serving_extension(model_types=("musespark_open_omni",))
class MuseSparkServingExtension(ServingExtension):
    """Request-field to chat-template-kwarg derivation for MuseSpark."""

    def apply_chat_extensions(self, request: ChatCompletionRequest) -> None:
        """Route a request-level ``reasoning_effort`` to the template kwargs.

        The musespark_open_omni chat template renders its reasoning-strength
        control line from chat-template kwargs only, so a top-level
        ``reasoning_effort`` — the field OpenAI-compatible clients actually
        send — was silently ignored on this path (only the kimi/harmony paths
        consumed it). Derive the kwarg from the request field; an explicit
        client-supplied ``chat_template_kwargs.reasoning_effort`` wins over
        the derived value, and the pydantic default (the field defaults to
        LOW) is never derived — only an effort the client explicitly set.
        """
        if "reasoning_effort" not in request.model_fields_set or request.reasoning_effort is None:
            return
        effort = getattr(request.reasoning_effort, "value", request.reasoning_effort).lower()
        if effort not in _MUSESPARK_EFFORTS:
            # `max`/`none` (Kimi-style) have no MuseSpark equivalent; leave
            # the template default rather than guessing a mapping.
            return
        request.chat_template_kwargs = {
            "reasoning_effort": effort,
            **(request.chat_template_kwargs or {}),
        }
