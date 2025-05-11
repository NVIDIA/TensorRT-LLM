# Adapt from
# https://github.com/vllm-project/vllm/blob/2e33fe419186c65a18da6668972d61d7bbc31564/vllm/inputs/data.py
from typing import Any, Dict, List, Union

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

    mm_processor_kwargs: NotRequired[Dict[str, Any]]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    input processor for mm input processing.
    """

    query: NotRequired[str]
    """The query input text for star attention."""


class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""

    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""

    multi_modal_data: NotRequired[Dict[str, Any]]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    mm_processor_kwargs: NotRequired[Dict[str, Any]]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    input processor for mm input processing.
    """

    query_token_ids: NotRequired[List[int]]
    """The query input token IDs for star attention."""


PromptInputs = Union[str, List[int], TextPrompt, TokensPrompt]


def prompt_inputs(inputs: PromptInputs, ) -> Union[TextPrompt, TokensPrompt]:
    if isinstance(inputs, str):
        prompt_inputs = TextPrompt(prompt=inputs)
    elif isinstance(inputs, list):
        assert isinstance(inputs[0], int)
        prompt_inputs = TokensPrompt(prompt_token_ids=inputs)
    elif isinstance(inputs, dict):
        assert inputs.get("prompt") is not None \
            or inputs.get("prompt_token_ids") is not None
        return inputs
    else:
        raise TypeError(
            f"Invalid type of inputs for llm.generate: {type(inputs)}")

    return prompt_inputs
