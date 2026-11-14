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
mistral-common also counts the image tokens from the decoded images, and builds
the grammar that forces a required or named tool call.
"""

from typing import Any, Dict, Optional

from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from tensorrt_llm.serve.serving_extensions import (
    RequiredToolChoiceMode,
    ServingExtension,
    register_serving_extension,
)


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


def _reasoning_effort(value: Any) -> Any:
    """The effort name of an enum member, or ``value`` itself.

    mistral-common checks the effort when the chat is rendered, against the
    efforts the checkpoint's tekken.json declares.
    """
    return getattr(value, "value", value)


@register_serving_extension(
    model_types=("mistral_large_4",), reasoning_parsers=("mistral_large_4",)
)
class MistralLarge4ServingExtension(ServingExtension):
    """Chat-request rules for the mistral-common renderer, and where structured outputs start."""

    def validate_chat_request(
        self, request: ChatCompletionRequest, *, max_seq_len: Optional[int]
    ) -> None:
        """Refuse an output budget that alone exceeds the engine's maximum sequence length."""
        requested_tokens = request.max_completion_tokens
        if (
            max_seq_len is not None
            and requested_tokens is not None
            and requested_tokens > max_seq_len
        ):
            raise ValueError(
                f"Requested {requested_tokens} output tokens exceeds the "
                f"maximum context length of {max_seq_len} tokens."
            )

    def render_needs_media(self) -> bool:
        """mistral-common derives each image's token count from the decoded image."""
        return True

    def required_tool_choice_mode(self) -> RequiredToolChoiceMode:
        """Required calls are forced by mistral-common's tool grammar."""
        return "native"

    def forced_tool_call_decoding(
        self,
        request: ChatCompletionRequest,
        tokenizer: Any,
        *,
        tool_parser: Optional[str],
        guided_decoding_backend: Optional[str],
    ):
        """mistral-common's grammar for a required or named call.

        It applies when the server parses Mistral tool calls
        (``--tool_parser mistral_large_4``); with another parser a named call takes
        the generic path and a required call is refused. The grammar is an
        llguidance grammar, so the server must run that backend.
        """
        if tool_parser != "mistral_large_4":
            return None
        if request.tool_choice != "required" and not isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam
        ):
            return None
        if guided_decoding_backend != "llguidance":
            raise ValueError(
                "Mistral required and named tool calls require guided_decoding_backend='llguidance'."
            )
        return native_tool_guided_decoding(request, tokenizer)

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
