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
"""Serving extension for Mistral-native checkpoints (``model_type == "mistral_common"``).

The Mistral-native format (``checkpoint_format`` ``mistral`` or ``mistral_large_3``)
renders chats with mistral-common, not a Jinja template. mistral-common takes the
reasoning effort and the response format as model settings (Tekken v15 and later),
and counts each image's tokens from the decoded image.
"""

from typing import Any

from tensorrt_llm.inputs.utils import NATIVE_MISTRAL_MODEL_TYPES
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.serving_extensions import ServingExtension, register_serving_extension


def _reasoning_effort(value: Any) -> Any:
    """The effort name of an enum member, or ``value`` itself.

    mistral-common checks the effort when the chat is rendered, against the efforts
    the checkpoint's tekken.json declares.
    """
    return getattr(value, "value", value)


@register_serving_extension(model_types=NATIVE_MISTRAL_MODEL_TYPES)
class MistralNativeServingExtension(ServingExtension):
    """Chat-request rules for the mistral-common renderer."""

    def render_needs_media(self) -> bool:
        """mistral-common derives each image's token count from the decoded image."""
        return True

    def apply_chat_extensions(self, request: ChatCompletionRequest) -> None:
        """Validate the template controls and resolve the model settings.

        A request-level ``chat_template`` cannot apply to mistral-common and is
        rejected. An explicit top-level ``reasoning_effort`` is passed to the renderer
        through ``chat_template_kwargs``; a ``reasoning_effort`` already in
        ``chat_template_kwargs`` is accepted as well, and the two must agree when both
        are given. Unset, the checkpoint's default effort applies; a checkpoint that
        declares no efforts rejects one. A ``json_object`` or ``json_schema`` response
        format is passed on as well, for the renderer to state in the prompt; it
        constrains decoding either way.
        """
        if request.chat_template is not None:
            raise ValueError("chat_template is not supported for Mistral-native checkpoints.")
        kwargs = dict(request.chat_template_kwargs or {})
        from_kwargs = kwargs.get("reasoning_effort")
        if from_kwargs is not None:
            from_kwargs = _reasoning_effort(from_kwargs)
        if "reasoning_effort" in request.model_fields_set and request.reasoning_effort is not None:
            effort = _reasoning_effort(request.reasoning_effort)
            if from_kwargs is not None and from_kwargs != effort:
                raise ValueError(
                    f"reasoning_effort={effort!r} conflicts with "
                    f"chat_template_kwargs['reasoning_effort']={from_kwargs!r}."
                )
            kwargs["reasoning_effort"] = effort
        kwargs.pop("response_format", None)
        response_format = request.response_format
        if response_format is not None and response_format.type in ("json_object", "json_schema"):
            kwargs["response_format"] = response_format.model_dump(exclude_none=True)
        if kwargs != (request.chat_template_kwargs or {}):
            request.chat_template_kwargs = kwargs
