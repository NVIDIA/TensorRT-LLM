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

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Optional, cast

from transformers import PretrainedConfig

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

if TYPE_CHECKING:
    from tensorrt_llm.serve.harmony_adapter import HarmonyAdapter

ToolDict = dict[str, object]


def resolve_model_type_from_config(model_name_or_path: str) -> Optional[str]:
    """Return the checkpoint's declared model type from its config metadata."""
    config_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path)
    model_type = config_dict.get("model_type")
    return model_type if isinstance(model_type, str) else None


def uses_harmony_tokenization(
    use_harmony: Optional[bool] = None,
    model_type: Optional[str] = None,
    model_type_resolver: Optional[Callable[[], Optional[str]]] = None,
) -> bool:
    if os.getenv("DISABLE_HARMONY_ADAPTER", "0") == "1":
        return False
    if use_harmony is not None:
        return use_harmony
    if model_type is None and model_type_resolver is not None:
        model_type = model_type_resolver()
    return model_type == "gpt_oss"


def get_chat_completion_tool_dicts(
    request: ChatCompletionRequest, empty_as_none: bool = False
) -> Optional[list[ToolDict]]:
    if request.tools is None or (empty_as_none and not request.tools):
        return None
    tools: list[ToolDict] = []
    for tool in request.tools:
        if hasattr(tool, "model_dump"):
            tools.append(cast(ToolDict, tool.model_dump()))
        elif isinstance(tool, dict):
            tools.append(cast(ToolDict, tool))
        else:
            raise TypeError(f"Unsupported tool type: {type(tool).__name__}")
    return tools


def tokenize_harmony_chat_request(
    request: ChatCompletionRequest,
    harmony_adapter: Optional["HarmonyAdapter"] = None,
    set_prompt_token_ids: bool = False,
) -> list[int]:
    if request.prompt_token_ids is not None:
        return request.prompt_token_ids

    from tensorrt_llm.serve import harmony_adapter as harmony_adapter_module

    adapter = harmony_adapter or harmony_adapter_module.get_harmony_adapter()
    result = adapter.openai_to_harmony_tokens(
        request.messages,
        get_chat_completion_tool_dicts(request, empty_as_none=True),
        reasoning_effort=harmony_adapter_module.maybe_transform_reasoning_effort(
            request.reasoning_effort
        ),
        tool_choice=request.tool_choice,
    )
    if set_prompt_token_ids:
        request.prompt_token_ids = result
    return result


def render_chat_request_for_tokenizer(
    request: ChatCompletionRequest, tokenizer: object
) -> str | list[int]:
    chat_template_kwargs = (
        dict(request.chat_template_kwargs) if getattr(request, "chat_template_kwargs", None) else {}
    )
    chat_template_kwargs["tools"] = get_chat_completion_tool_dicts(request)
    chat_template_kwargs["documents"] = request.documents
    if request.chat_template is not None:
        chat_template_kwargs["chat_template"] = request.chat_template
    rendered = tokenizer.apply_chat_template(
        [msg if isinstance(msg, dict) else dict(msg) for msg in request.messages],
        add_generation_prompt=request.add_generation_prompt,
        tokenize=False,
        return_dict=False,
        **chat_template_kwargs,
    )
    if isinstance(rendered, str):
        return rendered
    return list(rendered)


def tokenize_chat_request_for_serving(
    request: ChatCompletionRequest,
    tokenizer_factory: Callable[[], object],
    encode_rendered: Callable[[str, object], list[int]],
    use_harmony: Optional[bool] = None,
    model_type: Optional[str] = None,
    model_type_resolver: Optional[Callable[[], Optional[str]]] = None,
    harmony_adapter: Optional["HarmonyAdapter"] = None,
    set_prompt_token_ids: bool = True,
) -> list[int]:
    if request.prompt_token_ids is not None:
        return request.prompt_token_ids

    if uses_harmony_tokenization(
        use_harmony=use_harmony,
        model_type=model_type,
        model_type_resolver=model_type_resolver,
    ):
        return tokenize_harmony_chat_request(
            request,
            harmony_adapter=harmony_adapter,
            set_prompt_token_ids=set_prompt_token_ids,
        )

    tokenizer = tokenizer_factory()
    rendered = render_chat_request_for_tokenizer(request, tokenizer)
    result = encode_rendered(rendered, tokenizer) if isinstance(rendered, str) else rendered
    if set_prompt_token_ids:
        request.prompt_token_ids = result
    return result
