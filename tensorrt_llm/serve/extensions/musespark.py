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
"""Serving extension for Muse Spark checkpoints (``model_type == "muse_spark"``).

Routes a request-level ``reasoning_effort`` to the MuseSpark chat template and
normalizes every serve-derived effort onto the levels the template accepts.

Structured output takes the generic reasoning-parser placement. This extension
is registered by model type only, so no hook is keyed by the ``musespark``
reasoning parser, and ``_response_format_to_guided_decoding_params`` wraps the
constraint in a structural tag: a forced private reasoning message, opened by
``MuseSparkReasoningParser.reasoning_start`` and closed by ``reasoning_end``,
then the constraint on the answer message. Reasoning tokens therefore count
toward ``max_tokens`` before any constrained output.
"""

from typing import Optional

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.serving_extensions import ServingExtension, register_serving_extension

#: The serve-side effort contract: API value (lower-cased) -> template level.
#: The chat template rejects values outside its own set with a 400, so the
#: table emits only accepted levels; values absent from it are dropped so the
#: template's own default applies. The non-identity rows map protocol efforts
#: the template lacks: `max` is Anthropic's top tier. The template has no way
#: to turn reasoning off: it only sets a reasoning strength, from `low` to
#: `xhigh`, and defaults to `xhigh`. So `none` and the legacy `minimal` get the
#: least reasoning it offers, `low`, rather than falling through to that
#: default.
_MUSESPARK_EFFORT_NORMALIZATION = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "xhigh",
    "none": "low",
    "minimal": "low",
}


def normalize_musespark_effort(effort) -> Optional[str]:
    """Map an API-level effort (str or enum) onto a MuseSpark level, or None.

    Lower-cases and looks up the normalization table; anything outside it is
    dropped (None) so the template default applies instead of a 400.
    """
    value = getattr(effort, "value", effort)
    if not isinstance(value, str):
        return None
    return _MUSESPARK_EFFORT_NORMALIZATION.get(value.lower())


@register_serving_extension(model_types=("muse_spark",))
class MuseSparkServingExtension(ServingExtension):
    """Request-field to chat-template-kwarg derivation for MuseSpark."""

    def normalize_reasoning_effort(self, effort: str) -> Optional[str]:
        return normalize_musespark_effort(effort)

    def apply_chat_extensions(self, request: ChatCompletionRequest) -> None:
        """Route a request-level ``reasoning_effort`` to the template kwargs.

        The Muse Spark chat template renders its reasoning-strength control
        line from chat-template kwargs only, so a top-level
        ``reasoning_effort`` (the field OpenAI-compatible clients send) has
        to be derived into them. An explicit client-supplied
        ``chat_template_kwargs.reasoning_effort`` wins over the derived
        value, and the pydantic default (the field defaults to LOW) is never
        derived: only an effort the client explicitly set.

        A ``reasoning_effort`` kwarg the serve itself injected (the Anthropic
        adapter forwards ``output_config.effort`` this way and lists it in
        ``injected_chat_template_kwargs``) is normalized in place; a kwarg
        the client wrote is forwarded untouched.
        """
        kwargs = request.chat_template_kwargs
        injected = request.injected_chat_template_kwargs or ()
        if kwargs and "reasoning_effort" in kwargs and "reasoning_effort" in injected:
            kwargs = dict(kwargs)
            normalized = normalize_musespark_effort(kwargs["reasoning_effort"])
            if normalized is None:
                del kwargs["reasoning_effort"]
            else:
                kwargs["reasoning_effort"] = normalized
            request.chat_template_kwargs = kwargs

        if "reasoning_effort" not in request.model_fields_set or request.reasoning_effort is None:
            return
        effort = normalize_musespark_effort(request.reasoning_effort)
        if effort is None:
            return
        request.chat_template_kwargs = {
            "reasoning_effort": effort,
            **(request.chat_template_kwargs or {}),
        }
