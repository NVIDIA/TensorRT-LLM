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

    The dtype must be float16, bfloat16 or float32; the engine converts rows to
    the model dtype. ``generate`` takes an owned CPU snapshot at submission, so
    later writes to the caller's tensor have no effect. Each row replaces the
    token embedding of one prompt position and must equal what the model's
    input embedding module would produce there, including any scaling it
    applies. The ids at those positions are placeholders: they still feed
    repetition penalties, prompt logprobs and KV cache events. A KV block is
    reused only by requests with identical ids and rows up to its end, so with
    the V2 KV cache manager blocks before the first embedding row are shared
    with plain-token prompts (other managers scope the whole prompt to
    identical embeddings). Supported on
    decoder models whose ``prompt_embeds_unsupported_reason`` returns None, on
    the PyTorch backend, without speculative decoding, attention data parallelism or
    context parallelism. With disaggregated serving the
    generation-only request must carry the same ``prompt_embeds`` as the
    context request (only its identity is used there) and the KV cache manager
    must be V2.
    """

    prompt_embeds_mask: NotRequired[Union[List[bool], torch.Tensor]]
    """Length-``len(prompt_token_ids)`` bools, True where a ``prompt_embeds`` row
    replaces the token embedding; rows are consumed in position order. May be
    omitted when ``prompt_embeds`` has one row per prompt token.
    """


class EmbedsPrompt(TypedDict):
    """Schema for a prompt given entirely as precomputed input embeddings."""

    prompt_embeds: torch.Tensor
    """``[L, hidden_size]`` input embeddings, one row per prompt position. See
    ``TokensPrompt.prompt_embeds`` for the contract. ``LLM.generate`` accepts
    this dict wherever it accepts a ``TokensPrompt``."""


PromptInputs = Union[str, List[int], TextPrompt, TokensPrompt]


def prompt_inputs(
    inputs: Union[PromptInputs, EmbedsPrompt],
) -> Union[TextPrompt, TokensPrompt, EmbedsPrompt]:
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
