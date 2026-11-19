# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapt from
# https://github.com/vllm-project/vllm/blob/2e33fe419186c65a18da6668972d61d7bbc31564/vllm/inputs/data.py
from typing import Any, Dict, List, Union

import torch
from typing_extensions import NotRequired, TypedDict


class TextPrompt(TypedDict):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""

    multi_modal_data: NotRequired[Dict[str, Any]]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    multi_modal_uuids: NotRequired[Dict[str, List[Any]]]
    """
    Optional user-provided UUIDs for multimodal items.
    Structure mirrors multi_modal_data: {"image": ["uuid1", None, "uuid3"]}.
    When a UUID is provided for an item, it is preserved in KV cache events.
    V1 uses it as the legacy hash value; V2 returns it in the optional uuid
    field while retaining the derived digest as hash. Use None for specific
    items that need content-only hashing.
    """

    mm_processor_kwargs: NotRequired[Dict[str, Any]]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    input processor for mm input processing.
    """

    chat_template_applied: NotRequired[bool]
    """
    True when ``prompt`` is the chat template's output with one registered
    multimodal placeholder per media item already in it. Set by trtllm-serve
    and by the prompt builders in ``tensorrt_llm.inputs.utils``; input
    processors that otherwise render a raw ``prompt`` through the model's chat
    template use it verbatim when this is set.
    """


class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""

    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""

    multi_modal_data: NotRequired[Dict[str, Any]]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    multi_modal_uuids: NotRequired[Dict[str, List[Any]]]
    """
    Optional user-provided UUIDs for multimodal items.
    Structure mirrors multi_modal_data: {"image": ["uuid1", None, "uuid3"]}.
    When a UUID is provided for an item, it is preserved in KV cache events.
    V1 uses it as the legacy hash value; V2 returns it in the optional uuid
    field while retaining the derived digest as hash. Use None for specific
    items that need content-only hashing.
    """

    mm_processor_kwargs: NotRequired[Dict[str, Any]]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    input processor for mm input processing.
    """

    multi_item_part_lens: NotRequired[List[int]]
    """Metadata for multi-item scoring.

    Specifying this enables the multi-item scoring mode.
    In this mode, `prompt_token_ids` is assumed to consist of a prefix (or query) and
    multiple items (or candidates). Each part of the prompt is assumed to be followed
    by a delimiter token. The attention computation (masking and positional encodings)
    is altered such that the prefill logits correspond to those that would
    have been obtained upon running separate prefills for prefix + item1, prefix + item2,
    and so on. Currently, this feature is only supported in prefill-only execution
    via `LLM.encode`.

    The list elements should correspond, in order, to prefix length and the length of
    each following item to score (all lengths in tokens).
    See https://github.com/flashinfer-ai/flashinfer/pull/1015 for details and nomenclature.
    The length should not include the delimiter tokens.
    """

    prompt_embeds: NotRequired[torch.Tensor]
    """Precomputed ``[K, hidden_size]`` input embeddings for ``K`` prompt positions.

    This is the prompt_embeds contract. Rows (float16, bfloat16 or float32) must
    equal what the model's input embedding module, including any scaling, would
    produce; ``generate`` snapshots them at submission. The ids at those
    positions are placeholders that still feed repetition penalties, prompt
    logprobs and KV cache events. KV blocks are reused only across identical ids
    and rows; with the V2 KV cache manager blocks before the first row are
    shared with plain-token prompts. Requires the PyTorch backend and a decoder
    model whose ``prompt_embeds_unsupported_reason`` returns None; speculative
    decoding, attention data parallelism and context parallelism are rejected.
    Disaggregated serving needs the V2 manager, and the generation-only request
    must carry the same ``prompt_embeds`` as the context request.
    """

    prompt_embeds_mask: NotRequired[Union[List[bool], torch.Tensor]]
    """Length-``len(prompt_token_ids)`` bools, True where the next ``prompt_embeds``
    row replaces the token embedding; optional when there is one row per token.
    With ``multi_modal_data`` it is required and indexes ``prompt_token_ids``
    before placeholder expansion; it must not select a placeholder.
    """


class EmbedsPrompt(TypedDict):
    """Schema for a prompt given entirely as precomputed input embeddings."""

    prompt_embeds: torch.Tensor
    """``[L, hidden_size]``, one row per position; see ``TokensPrompt.prompt_embeds``."""


PromptInputs = Union[str, List[int], TextPrompt, TokensPrompt, EmbedsPrompt]


def prompt_inputs(
        inputs: PromptInputs) -> Union[TextPrompt, TokensPrompt, EmbedsPrompt]:
    if isinstance(inputs, str):
        prompt_inputs = TextPrompt(prompt=inputs)
    elif isinstance(inputs, list):
        assert len(inputs) == 0 or isinstance(inputs[0], int)
        prompt_inputs = TokensPrompt(prompt_token_ids=inputs)
    elif isinstance(inputs, dict):
        assert inputs.get("prompt") is not None \
            or inputs.get("prompt_token_ids") is not None \
            or inputs.get("prompt_embeds") is not None
        return inputs
    else:
        raise TypeError(
            f"Invalid type of inputs for llm.generate: {type(inputs)}")

    return prompt_inputs
