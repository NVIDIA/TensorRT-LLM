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
"""Serving extension for Mistral Large 4 checkpoints (``model_type == "mistral_large_4"``).

Mistral Large 4 renders chats with mistral-common, not a Jinja template, and
takes its reasoning mode and response format from mistral-common's model settings.
"""

from typing import Any, Dict, Optional

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.serving_extensions import ServingExtension, register_serving_extension


def native_tool_guided_decoding(request, tokenizer):
    """Constrain a required or named call using mistral-common's native grammar."""
    from tensorrt_llm.sampling_params import GuidedDecodingParams

    choice = request.tool_choice
    parallel_tool_calls = choice == "required" and request.parallel_tool_calls is not False
    if not isinstance(choice, str):
        name = choice.function.name
        if not any(tool.function.name == name for tool in request.tools or []):
            raise ValueError(f"Named tool {name!r} is not declared in tools.")
        choice = choice.model_dump()
    if not hasattr(tokenizer, "llguidance_tool_grammar"):
        tokenizer = tokenizer.tokenizer
    grammar = tokenizer.llguidance_tool_grammar(
        [tool.model_dump(exclude_none=True) for tool in request.tools or []],
        choice,
        parallel_tool_calls,
    )
    return GuidedDecodingParams(grammar=grammar)


def _known_reasoning_efforts() -> list[str]:
    from mistral_common.protocol.instruct.request import ReasoningEffort

    return [effort.value for effort in ReasoningEffort]


def _reasoning_effort(value: Any, field: str) -> str:
    """``value`` as a mistral-common reasoning effort, or ValueError naming ``field``.

    Whether the checkpoint accepts a known effort is checked when the chat is rendered,
    against the efforts its tekken.json declares.
    """
    effort = getattr(value, "value", value)
    known = _known_reasoning_efforts()
    if effort not in known:
        raise ValueError(
            f"{field}={effort!r} is not a reasoning effort. mistral-common knows {known}; "
            "the checkpoint's tekken.json declares which of them it accepts."
        )
    return effort


@register_serving_extension(
    model_types=("mistral_large_4",), reasoning_parsers=("mistral_large_4",)
)
class MistralLarge4ServingExtension(ServingExtension):
    """Chat-request rules for the mistral-common renderer, and where structured outputs start."""

    def apply_chat_extensions(self, request: ChatCompletionRequest) -> None:
        """Validate the template controls and resolve the model settings.

        A request-level ``chat_template`` cannot apply to mistral-common and is
        rejected. An explicit top-level ``reasoning_effort`` is passed to the
        renderer through ``chat_template_kwargs``; a ``reasoning_effort`` already
        in ``chat_template_kwargs`` is accepted as well, and the two must agree
        when both are given. Unset, the checkpoint's default effort applies. A
        ``json_object`` or ``json_schema`` response format is passed on as well,
        for the renderer to state in the prompt; it constrains decoding either way.
        """
        if request.chat_template is not None:
            raise ValueError("chat_template is not supported for Mistral Large 4.")
        kwargs = dict(request.chat_template_kwargs or {})
        from_kwargs = kwargs.get("reasoning_effort")
        if from_kwargs is not None:
            from_kwargs = _reasoning_effort(from_kwargs, "chat_template_kwargs['reasoning_effort']")
        if "reasoning_effort" in request.model_fields_set and request.reasoning_effort is not None:
            effort = _reasoning_effort(request.reasoning_effort, "reasoning_effort")
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

    def structured_output_format(
        self, content: Dict[str, Any], chat_template_kwargs: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Apply the raw grammar, with no structural tag around it.

        A structural tag cannot open a ``[THINK]`` block before the constraint:
        ``[THINK]`` and ``[/THINK]`` are control tokens, which the
        Mistral-Large-4 xgrammar vocabulary holds as empty strings. Under
        xgrammar the raw grammar constrains the response from the first
        generated token, so it carries no reasoning; under llguidance a JSON
        response may open with a reasoning section.
        """
        return None
