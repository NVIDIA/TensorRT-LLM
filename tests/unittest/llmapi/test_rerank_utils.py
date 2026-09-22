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
"""CPU unit tests for the Qwen3-Reranker prompt/truncation logic.

Uses a fake word-splitting tokenizer (one token id per whitespace-separated
word) so the truncation budget math is exact and deterministic, without
needing real Qwen3 tokenizer files.
"""

import pytest


class _FakeTokenizer:
    """One token id per whitespace-separated word; ids are just word indices
    into a running vocabulary, so re-encoding the same word always yields the
    same id. Good enough to make length/truncation assertions exact."""

    def __init__(self):
        self._vocab = {}

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens  # unused by this fake; real usage passes False
        ids = []
        for word in text.split():
            ids.append(self._vocab.setdefault(word, len(self._vocab)))
        return ids


def test_build_rerank_prompt_assembles_prefix_header_document_suffix():
    from tensorrt_llm.serve.rerank_utils import (_PREFIX, _SUFFIX,
                                                  build_rerank_prompt_token_ids)

    tokenizer = _FakeTokenizer()
    query = "capital of France"
    document = "Paris is the capital of France"

    token_ids = build_rerank_prompt_token_ids(tokenizer,
                                              query,
                                              document,
                                              max_seq_len=1000)

    prefix_ids = tokenizer.encode(_PREFIX)
    suffix_ids = tokenizer.encode(_SUFFIX)
    assert token_ids[:len(prefix_ids)] == prefix_ids
    assert token_ids[-len(suffix_ids):] == suffix_ids
    # The document's words must appear, untruncated, between header and suffix.
    document_ids = [tokenizer.encode(w)[0] for w in document.split()]
    middle = token_ids[len(prefix_ids):-len(suffix_ids)]
    assert middle[-len(document_ids):] == document_ids


def test_build_rerank_prompt_truncates_only_the_document():
    from tensorrt_llm.serve.rerank_utils import (DEFAULT_INSTRUCTION, _PREFIX,
                                                  _SUFFIX,
                                                  build_rerank_prompt_token_ids)

    tokenizer = _FakeTokenizer()
    query = "capital of France"
    document = " ".join(f"word{i}" for i in range(1000))
    document_ids_full = [tokenizer.encode(w)[0] for w in document.split()]

    # Budget tight enough to force document truncation but big enough to fit
    # the template + query (the fixed prefix/header/suffix alone need ~49
    # words under this whitespace-splitting fake tokenizer).
    max_seq_len = 60
    token_ids = build_rerank_prompt_token_ids(tokenizer,
                                              query,
                                              document,
                                              max_seq_len=max_seq_len)
    assert len(token_ids) == max_seq_len

    prefix_ids = tokenizer.encode(_PREFIX)
    suffix_ids = tokenizer.encode(_SUFFIX)
    header_ids = tokenizer.encode(
        f"<Instruct>: {DEFAULT_INSTRUCTION}\n<Query>: {query}\n<Document>: ")
    budget = max_seq_len - len(prefix_ids) - len(header_ids) - len(suffix_ids)

    kept_document_ids = token_ids[len(prefix_ids) +
                                  len(header_ids):-len(suffix_ids)]
    # Only the tail is dropped: the kept ids are an exact prefix of the full
    # (untruncated) document's token ids, not an arbitrary subset.
    assert kept_document_ids == document_ids_full[:budget]


def test_build_rerank_prompt_raises_when_query_alone_overflows():
    from tensorrt_llm.serve.rerank_utils import build_rerank_prompt_token_ids

    tokenizer = _FakeTokenizer()
    query = " ".join(f"q{i}" for i in range(1000))

    with pytest.raises(ValueError, match="max_seq_len"):
        build_rerank_prompt_token_ids(tokenizer,
                                      query,
                                      document="doc",
                                      max_seq_len=10)


def test_build_rerank_prompt_uses_default_instruction_when_none_given():
    from tensorrt_llm.serve.rerank_utils import (DEFAULT_INSTRUCTION,
                                                  build_rerank_prompt_token_ids)

    tokenizer = _FakeTokenizer()
    with_default = build_rerank_prompt_token_ids(tokenizer,
                                                  "q",
                                                  "d",
                                                  max_seq_len=1000)
    with_explicit = build_rerank_prompt_token_ids(tokenizer,
                                                   "q",
                                                   "d",
                                                   max_seq_len=1000,
                                                   instruction=DEFAULT_INSTRUCTION)
    assert with_default == with_explicit
