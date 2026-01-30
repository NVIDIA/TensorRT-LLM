# Adapt from
# https://github.com/vllm-project/vllm/blob/2e33fe419186c65a18da6668972d61d7bbc31564/vllm/inputs/data.py
from typing import Any, Dict, List, Sequence, Union

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


class VisualGenTextPrompt(TypedDict):
    prompt: str
    negative_prompt: NotRequired[str]


class VisualGenTokensPrompt(TypedDict):
    prompt_token_ids: List[int]
    negative_prompt_token_ids: NotRequired[List[int]]


VisualGenPromptInputs = Union[
    str,
    List[int],
    VisualGenTextPrompt,
    VisualGenTokensPrompt,
]

VisualGenInputs = Union[
    VisualGenPromptInputs,
    Sequence[VisualGenPromptInputs],
]


def visual_gen_inputs(
    inputs: "VisualGenPromptInputs",
) -> Union["VisualGenTextPrompt", "VisualGenTokensPrompt"]:
    # str -> text prompt
    if isinstance(inputs, str):
        return VisualGenTextPrompt(prompt=inputs)

    # list[int] -> token prompt
    if isinstance(inputs, list):
        if len(inputs) == 0:
            raise ValueError("`inputs` token list cannot be empty.")
        if not all(isinstance(t, int) for t in inputs):
            raise TypeError(
                "`inputs` list must contain only ints when used as token IDs.")
        return VisualGenTokensPrompt(prompt_token_ids=inputs)

    # dict form
    if isinstance(inputs, dict):
        has_prompt = "prompt" in inputs
        has_prompt_token_ids = "prompt_token_ids" in inputs

        if has_prompt == has_prompt_token_ids:
            raise ValueError(
                "VisualGen prompt dict must contain exactly one of "
                "`prompt` or `prompt_token_ids`.")

        if has_prompt:
            prompt = inputs.get("prompt")
            if not isinstance(prompt, str) or prompt == "":
                raise TypeError("`prompt` must be a non-empty string.")
            if "negative_prompt" in inputs and not isinstance(
                    inputs["negative_prompt"], str):
                raise TypeError("`negative_prompt` must be a string.")
            return inputs  # VisualGenTextPrompt

        token_ids = inputs.get("prompt_token_ids")
        if not isinstance(token_ids, list) or len(token_ids) == 0:
            raise TypeError("`prompt_token_ids` must be a non-empty list[int].")
        if not all(isinstance(t, int) for t in token_ids):
            raise TypeError("`prompt_token_ids` must contain only ints.")
        if "negative_prompt_token_ids" in inputs:
            neg_ids = inputs["negative_prompt_token_ids"]
            if not isinstance(neg_ids, list) or not all(
                    isinstance(t, int) for t in neg_ids):
                raise TypeError(
                    "`negative_prompt_token_ids` must be a list[int].")
        return inputs  # VisualGenTokensPrompt

    raise TypeError(
        "Invalid `inputs` for VisualGen.generate. "
        "Expected one of: str, list[int], VisualGenTextPrompt, VisualGenTokensPrompt."
    )
