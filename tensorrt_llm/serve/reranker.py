# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input formatting and score conversion for reranker serving.

Qwen3-Reranker is currently the only supported model-specific formatter.
"""

import math
from typing import Protocol

_DEFAULT_QWEN3_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_QWEN3_RERANK_SYSTEM_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class _Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


def _format_qwen3_rerank_prefix(query: str, instruction: str | None) -> str:
    if instruction is None:
        instruction = _DEFAULT_QWEN3_RERANK_INSTRUCTION
    return f"{_QWEN3_RERANK_SYSTEM_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: "


def format_qwen3_rerank_prompt(
    query: str,
    document: str,
    instruction: str | None = None,
) -> str:
    """Render the official Qwen3-Reranker prompt.

    Args:
        query: Search query to evaluate.
        document: Candidate document to score.
        instruction: Task description. Uses the model's web-search instruction
            when omitted.

    Returns:
        The formatted prompt expected by Qwen3-Reranker checkpoints.
    """
    return _format_qwen3_rerank_prefix(query, instruction) + document + _QWEN3_RERANK_SUFFIX


def build_qwen3_rerank_input(
    tokenizer: _Tokenizer,
    query: str,
    document: str,
    max_seq_len: int,
    instruction: str | None = None,
    max_tokens_per_doc: int | None = None,
) -> list[int]:
    """Tokenize the official Qwen3 prompt, truncating only the document.

    The prompt is first tokenized as one string so token boundaries match the
    reference implementation. If it is too long, tokens are removed immediately
    before the fixed assistant suffix, keeping both the query and suffix intact.

    Args:
        tokenizer: Tokenizer for the reranker checkpoint.
        query: Search query to evaluate.
        document: Candidate document to score.
        max_seq_len: Maximum complete prompt length.
        instruction: Optional task description.
        max_tokens_per_doc: Optional document-token limit.

    Returns:
        Token IDs for one Qwen3-Reranker input.

    Raises:
        ValueError: If a limit is invalid, the tokenizer does not preserve the
            required suffix, or the instruction and query alone exceed the
            sequence limit.
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be greater than zero.")
    if max_tokens_per_doc is not None and max_tokens_per_doc <= 0:
        raise ValueError("max_tokens_per_doc must be greater than zero.")

    prompt_prefix = _format_qwen3_rerank_prefix(query, instruction)
    full_prompt = prompt_prefix + document + _QWEN3_RERANK_SUFFIX
    full_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    suffix_ids = tokenizer.encode(_QWEN3_RERANK_SUFFIX, add_special_tokens=False)
    if not suffix_ids or full_ids[-len(suffix_ids) :] != suffix_ids:
        raise ValueError("Tokenizer did not preserve the Qwen3 reranker suffix.")

    empty_prompt = prompt_prefix + _QWEN3_RERANK_SUFFIX
    minimum_length = len(tokenizer.encode(empty_prompt, add_special_tokens=False))
    if minimum_length > max_seq_len:
        raise ValueError(
            "The reranker instruction and query exceed the model's maximum "
            f"sequence length of {max_seq_len} tokens."
        )

    content_length = len(full_ids) - len(suffix_ids)
    if max_tokens_per_doc is not None:
        prefix_ids = tokenizer.encode(prompt_prefix, add_special_tokens=False)
        document_start = 0
        for full_token, prefix_token in zip(full_ids, prefix_ids):
            if full_token != prefix_token:
                break
            document_start += 1
        content_length = min(content_length, document_start + max_tokens_per_doc)
    content_length = min(content_length, max_seq_len - len(suffix_ids))

    if content_length + len(suffix_ids) >= len(full_ids):
        return full_ids
    return full_ids[:content_length] + suffix_ids


def rerank_probability(logit: float) -> float:
    """Convert a yes-minus-no logit to a relevance probability.

    This stable sigmoid is equivalent to
    ``softmax([no_logit, yes_logit])[yes]``.

    Args:
        logit: Difference between the model's yes and no logits.

    Returns:
        Relevance probability in the closed interval ``[0, 1]``.
    """
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)
