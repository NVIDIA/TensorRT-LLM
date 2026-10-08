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
"""Chat request -> prompt: the single preparation pipeline.

Every caller that turns a chat request into a prompt goes through here: the chat
route, the KV-aware router, ``count_tokens``, the resource governor, Responses,
the multimodal encoder route and the standalone renderer. The pipeline is:

1. :func:`prepare_chat_request`: model-extension preprocessing and tool dicts;
2. template rendering with the server template, processor and extension rules;
3. the forced named-tool prefix, when the caller supplies one;
4. optional tokenization through the engine's own text tokenizer.

:func:`render_conversation` is steps 2-4 for a caller that already parsed the
conversation (the chat route overlaps media loading with rendering).
:func:`render_chat` runs the whole pipeline for a caller that has only a request.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from pydantic import ValidationError

from tensorrt_llm.inputs.tokenization import tokenize_prompt
from tensorrt_llm.inputs.utils import apply_chat_template, async_apply_chat_template
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

from .prompt_types import RenderedPrompt, UnsupportedRenderError
from .resources import RenderResources

if TYPE_CHECKING:
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

_LEGACY_ENV = "TRTLLM_RENDER_LEGACY"


def legacy_render_enabled() -> bool:
    """Whether callers keep their pre-merge renderer choice.

    Set ``TRTLLM_RENDER_LEGACY=1`` to restore the previous behavior for one
    release: the router and ``count_tokens`` render with the simplified renderer
    and the router writes tokens back without checking the fingerprint, and the
    governor, Responses and multimodal-encoder routes render without the server
    template and extension rules.
    """
    return os.getenv(_LEGACY_ENV, "0") == "1"


@dataclass
class PreparedChat:
    """A chat request after extension preprocessing, ready to render."""

    request: Any
    tool_dicts: Optional[List[Dict[str, Any]]]


def prepare_chat_request(request: "ChatCompletionRequest", res: RenderResources) -> PreparedChat:
    """Run the model extension's preprocessing and dump the request's tools.

    Mutates ``request`` in place (the extension derives chat-template kwargs from
    request-level fields), exactly as the chat route always has.
    """
    # Deferred: openai_protocol pulls in the whole request-model stack.
    from tensorrt_llm.serve.openai_protocol import ensure_request_chat_template_allowed

    ensure_request_chat_template_allowed(request, res.allow_request_chat_template)
    extension = res.extension
    extension.apply_chat_extensions(request)
    if request.tool_choice == "required" and not extension.allows_required_tool_choice():
        # Schema-accepting "required" everywhere but enforcing it only for models
        # whose extension allows it would silently degrade to "auto" elsewhere;
        # reject loudly for models that cannot honor it. Still a ValueError for the
        # routes (a 400); a router or counter that only needs an estimate catches
        # UnsupportedRenderError and leaves the rejection to the worker.
        raise UnsupportedRenderError("tool_choice='required' is not supported for this model.")
    tool_dicts = (
        None
        if request.tools is None
        else [extension.serialize_tool(tool) for tool in request.tools]
    )
    return PreparedChat(request=request, tool_dicts=tool_dicts)


def _template_kwargs(
    res: RenderResources,
    *,
    conversation: Any,
    mm_placeholder_counts: Any,
    add_generation_prompt: bool,
    tools: Optional[List[Dict[str, Any]]],
    documents: Any,
    chat_template: Optional[str],
    chat_template_kwargs: Optional[Dict[str, Any]],
    injected_chat_template_kwargs: Any,
    enable_tokenize: bool,
) -> Dict[str, Any]:
    return dict(
        model_type=res.model_type or "",
        tokenizer=res.tokenizer,
        processor=res.processor,
        conversation=conversation,
        add_generation_prompt=add_generation_prompt,
        mm_placeholder_counts=mm_placeholder_counts,
        tools=tools,
        documents=documents,
        chat_template=chat_template or res.default_chat_template,
        chat_template_kwargs=chat_template_kwargs or {},
        injected_chat_template_kwargs=injected_chat_template_kwargs,
        enable_tokenize=enable_tokenize,
    )


def tokenize_rendered(
    res: RenderResources,
    text: str,
    *,
    add_special_tokens: bool = False,
    truncate_prompt_tokens: Optional[int] = None,
    encode_rendered: Optional[Callable[[str, Any], List[int]]] = None,
) -> List[int]:
    """Tokenize rendered prompt text the way the engine does.

    ``encode_rendered`` lets a caller with its own cache (the router) replace the
    plain ``encode`` call; it is used only for the plain no-special-token,
    no-truncation call it was written for.
    """
    plain_call = not add_special_tokens and truncate_prompt_tokens is None
    if encode_rendered is not None and plain_call:
        return encode_rendered(text, res.tokenizer)
    input_processor = res.input_processor
    if input_processor is None:
        return tokenize_prompt(
            res.tokenizer,
            text,
            add_special_tokens=add_special_tokens,
            truncate_prompt_tokens=truncate_prompt_tokens,
        )
    # Deferred: the registry module is heavy and only needed to check the kind.
    from tensorrt_llm.inputs.registry import DefaultInputProcessor

    if not isinstance(input_processor, DefaultInputProcessor):
        raise UnsupportedRenderError(
            f"This model tokenizes through {type(input_processor).__name__}, not the default "
            "text tokenizer; rendering its prompts is not supported."
        )
    params = SamplingParams(
        add_special_tokens=add_special_tokens, truncate_prompt_tokens=truncate_prompt_tokens
    )
    ids, _ = input_processor({"prompt": text}, params)
    return ids


def _finish(
    res: RenderResources,
    rendered: Any,
    *,
    forced_prefix: Optional[str],
    tokenize: bool,
    add_special_tokens: bool,
    truncate_prompt_tokens: Optional[int],
    encode_rendered: Optional[Callable[[str, Any], List[int]]],
) -> RenderedPrompt:
    if not isinstance(rendered, str):
        # A native renderer that tokenizes for itself returns ids.
        return RenderedPrompt(token_ids=list(rendered))
    template_text = rendered
    prefix_applied = False
    if forced_prefix is not None:
        # Force the model to start generation inside the tool call.
        rendered = rendered + forced_prefix
        prefix_applied = True
    token_ids = None
    if tokenize:
        token_ids = tokenize_rendered(
            res,
            rendered,
            add_special_tokens=add_special_tokens,
            truncate_prompt_tokens=truncate_prompt_tokens,
            encode_rendered=encode_rendered,
        )
    return RenderedPrompt(
        text=rendered,
        template_text=template_text,
        token_ids=token_ids,
        prefix_applied=prefix_applied,
    )


async def render_conversation(
    res: RenderResources,
    *,
    conversation: Any,
    mm_placeholder_counts: Any,
    add_generation_prompt: bool,
    tools: Optional[List[Dict[str, Any]]] = None,
    documents: Any = None,
    chat_template: Optional[str] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    injected_chat_template_kwargs: Any = None,
    forced_prefix: Optional[str] = None,
    tokenize: bool = False,
    add_special_tokens: bool = False,
    truncate_prompt_tokens: Optional[int] = None,
    encode_rendered: Optional[Callable[[str, Any], List[int]]] = None,
) -> RenderedPrompt:
    """Render an already-parsed conversation without blocking the event loop."""
    rendered = await async_apply_chat_template(
        **_template_kwargs(
            res,
            conversation=conversation,
            mm_placeholder_counts=mm_placeholder_counts,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            documents=documents,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
            injected_chat_template_kwargs=injected_chat_template_kwargs,
            enable_tokenize=False,
        )
    )
    if tokenize:
        # Tokenization is CPU work that scales with the prompt; keep it off the loop.
        return await asyncio.to_thread(
            _finish,
            res,
            rendered,
            forced_prefix=forced_prefix,
            tokenize=True,
            add_special_tokens=add_special_tokens,
            truncate_prompt_tokens=truncate_prompt_tokens,
            encode_rendered=encode_rendered,
        )
    return _finish(
        res,
        rendered,
        forced_prefix=forced_prefix,
        tokenize=False,
        add_special_tokens=add_special_tokens,
        truncate_prompt_tokens=truncate_prompt_tokens,
        encode_rendered=encode_rendered,
    )


_MEDIA_PART_MARKERS = ("image", "video", "audio")


def _is_passthrough_model(res: RenderResources) -> bool:
    """Whether the model registered the skip-the-chat-template content format."""
    if res.model_type is None:
        return False
    # Deferred: the registry module is heavy.
    from tensorrt_llm.inputs.content_format import ContentFormat
    from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

    return (
        MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format(res.model_type)
        == ContentFormat.PASSTHROUGH
    )


def _has_media(messages: Any) -> bool:
    """Whether any message carries a media content part."""
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = part.get("type", "") if isinstance(part, dict) else ""
            if any(marker in part_type for marker in _MEDIA_PART_MARKERS):
                return True
    return False


def _parse_text_conversation(
    request: "ChatCompletionRequest",
    res: RenderResources,
    raw_messages: Optional[List[Any]] = None,
):
    """Parse ``request.messages``; media is detected, never loaded.

    ``request.messages`` is a single-pass validating iterator that rejects extra
    fields (object-valued tool-call arguments, for one). The chat route then falls
    back to the raw JSON body's messages, and so does this: a caller that has the
    body passes ``raw_messages``. Without them the messages cannot be recovered, and
    guessing would silently drop content, so the request is reported as unsupported.
    """
    try:
        return parse_chat_messages_coroutines(
            request.messages,
            res.hf_config,
            res.multimodal_server_config,
            request_media_io_kwargs=getattr(request, "media_io_kwargs", None),
            resolve_media=False,
        )
    except ValidationError as error:
        if raw_messages is None:
            raise UnsupportedRenderError(
                "The messages carry fields the request model rejects; they can only be "
                "rendered from the raw request body."
            ) from error
        return parse_chat_messages_coroutines(
            raw_messages,
            res.hf_config,
            res.multimodal_server_config,
            request_media_io_kwargs=getattr(request, "media_io_kwargs", None),
            resolve_media=False,
        )


def render_chat(
    request: "ChatCompletionRequest",
    res: RenderResources,
    *,
    tokenize: bool = True,
    encode_rendered: Optional[Callable[[str, Any], List[int]]] = None,
    raw_messages: Optional[List[Any]] = None,
) -> RenderedPrompt:
    """Render a text-only chat request into a prompt, synchronously.

    The whole pipeline for a caller that has only a request. A request that
    already carries ``prompt_token_ids`` renders to exactly those. Multimodal
    input raises :class:`UnsupportedRenderError` rather than returning ids that
    ignore the media, and so do messages the request model rejects unless the
    caller passes the raw body's ``raw_messages`` (see :func:`_parse_text_conversation`).
    """
    if request.prompt_token_ids is not None:
        return RenderedPrompt(token_ids=list(request.prompt_token_ids))
    prepared = prepare_chat_request(request, res)

    if res.use_harmony:
        if getattr(request.tool_choice, "function", None) is not None:
            # The Harmony chat route refuses a named tool_choice rather than degrade it to
            # "auto"; ids rendered for it would not be an executable prompt.
            raise UnsupportedRenderError(
                "tool_choice with a named function is not yet supported for harmony / "
                "GPT-OSS models."
            )
        token_ids = res.extension.render_prompt(request, res)
        if token_ids is not None:
            return RenderedPrompt(token_ids=list(token_ids))

    if _has_media(request.messages):
        raise UnsupportedRenderError("Multimodal input is not supported by prompt rendering.")
    if _is_passthrough_model(res):
        # The model's own processor builds the prompt; the text the chat route
        # derives here is only its input, not the executed prompt.
        raise UnsupportedRenderError(
            f"Model type {res.model_type!r} builds its prompt in its own input processor; "
            "rendering its prompts is not supported."
        )
    conversation, _mm_coroutine, mm_placeholder_counts, _mm_item_order = _parse_text_conversation(
        request, res, raw_messages
    )
    rendered = apply_chat_template(
        **_template_kwargs(
            res,
            conversation=conversation,
            mm_placeholder_counts=mm_placeholder_counts,
            add_generation_prompt=request.add_generation_prompt,
            tools=prepared.tool_dicts,
            documents=request.documents,
            chat_template=request.chat_template,
            chat_template_kwargs=request.chat_template_kwargs,
            injected_chat_template_kwargs=request.injected_chat_template_kwargs,
            enable_tokenize=False,
        )
    )
    result = _finish(
        res,
        rendered,
        forced_prefix=None,
        tokenize=tokenize,
        add_special_tokens=request.add_special_tokens,
        truncate_prompt_tokens=request.truncate_prompt_tokens,
        encode_rendered=encode_rendered,
    )
    if getattr(request.tool_choice, "function", None) is not None:
        # A named function forces the model into that call by appending the tool
        # parser's begin string to the prompt; only a worker with a tool parser
        # can do that, so ids rendered here are not the executed prompt.
        result.tokens_trusted = False
        result.untrusted_reason = "named_tool_choice"
    return result


async def arender_chat(
    request: "ChatCompletionRequest",
    res: RenderResources,
    *,
    tokenize: bool = True,
    encode_rendered: Optional[Callable[[str, Any], List[int]]] = None,
    raw_messages: Optional[List[Any]] = None,
) -> RenderedPrompt:
    """:func:`render_chat` off the event loop."""
    return await asyncio.to_thread(
        render_chat,
        request,
        res,
        tokenize=tokenize,
        encode_rendered=encode_rendered,
        raw_messages=raw_messages,
    )
