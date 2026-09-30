#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3-Reranker prompt construction and document truncation.

Builds the official Qwen3-Reranker scoring prompt: a fixed system
instruction, an ``<Instruct>/<Query>/<Document>`` user turn, and an
assistant preamble ending right where the model emits its "yes"/"no" answer
(scored by ``Qwen3ForTextReranking``, see ``_torch/models/modeling_qwen3.py``).
"""

from typing import Protocol

DEFAULT_INSTRUCTION = ("Given a web search query, retrieve relevant "
                       "passages that answer the query")

# Matches the Qwen3-Reranker model card's reference prompt exactly (including
# the empty <think></think> block, which the model was tuned to expect before
# its yes/no answer).
_PREFIX = ("<|im_start|>system\n"
          "Judge whether the Document meets the requirements based on the "
          "Query and the Instruct provided. Note that the answer can only "
          "be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class _Tokenizer(Protocol):

    def encode(self, text: str, *args, add_special_tokens: bool = True,
              **kwargs) -> list[int]:
        ...


def build_rerank_prompt_token_ids(
    tokenizer: _Tokenizer,
    query: str,
    document: str,
    max_seq_len: int,
    instruction: str | None = None,
) -> list[int]:
    """Tokenize one (query, document) pair into the Qwen3-Reranker prompt.

    Only the document is truncated to fit `max_seq_len`; the query and
    instruction are kept intact so a truncated request still asks the same
    question, just against less of the candidate document.

    Raises:
        ValueError: if the template, instruction, and query alone (with an
            empty document) already exceed `max_seq_len`.
    """
    instruction = instruction or DEFAULT_INSTRUCTION

    prefix_ids = tokenizer.encode(_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(_SUFFIX, add_special_tokens=False)
    header_ids = tokenizer.encode(
        f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: ",
        add_special_tokens=False)

    budget = max_seq_len - len(prefix_ids) - len(header_ids) - len(suffix_ids)
    if budget < 0:
        raise ValueError(
            "The instruction and query alone exceed max_seq_len "
            f"({max_seq_len} tokens); there is no room left for the "
            "document. Shorten the query/instruction or raise --max_seq_len.")

    document_ids = tokenizer.encode(document, add_special_tokens=False)
    if len(document_ids) > budget:
        document_ids = document_ids[:budget]

    return prefix_ids + header_ids + document_ids + suffix_ids
