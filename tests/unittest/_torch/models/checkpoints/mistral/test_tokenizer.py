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
"""MistralTokenizer over a Tekken vocabulary: pickling, streaming detokenization and the
vocabulary it describes for guided decoding."""

import pickle

import pytest
from test_common.mistral_tekken import load_tekken

from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer

pytestmark = pytest.mark.cpu_only

PROMPT = "The capital of France is"


# None of these paths depends on the Tekken version: the file mistral-common ships.
@pytest.fixture(scope="module")
def tokenizer():
    return load_tekken("bundled")


def _text(tokenizer, text):
    return tokenizer.tokenizer.encode(text, bos=False, eos=False)


@pytest.mark.parametrize("serializer", ["pickle", "cloudpickle"])
def test_pickle_round_trip_keeps_the_class(tokenizer, serializer):
    module = pickle if serializer == "pickle" else pytest.importorskip("cloudpickle")
    restored = module.loads(module.dumps(tokenizer))
    assert type(restored) is MistralTokenizer
    assert type(restored.tokenizer) is type(tokenizer.tokenizer)
    ids = tokenizer.encode(PROMPT)
    assert ids[0] == tokenizer.bos_token_id
    assert restored.encode(PROMPT) == restored.encode(PROMPT, add_special_tokens=True) == ids
    assert restored.decode(ids, skip_special_tokens=True) == PROMPT
    assert restored.vocab_size == tokenizer.vocab_size
    assert all(isinstance(piece, str) for piece in restored._vocab_dict)


@pytest.mark.parametrize("skip_special_tokens", [False, True])
def test_streaming_matches_decode(tokenizer, skip_special_tokens):
    """Without skip_special_tokens, streamed text equals decode(); skipping them drops every
    control token but [TOOL_CALLS]. Characters split over tokens come back whole."""
    names = ["[TOOL_CALLS]", "[INST]", "[/INST]"]
    if "[THINK]" in tokenizer.tokenizer._special_tokens_reverse_vocab:
        names = ["[THINK]", "[/THINK]"] + names
    ids = []
    for name in names:
        ids += [tokenizer.tokenizer.get_control_token(name)]
        ids += _text(tokenizer, f" 巴黎 São Paulo 🙂 {name[1:3]}")
    expected = tokenizer.decode(ids, skip_special_tokens=False)
    assert all(name in expected for name in names)
    assert "🙂" in expected and "�" not in expected
    if skip_special_tokens:
        for name in names:
            if name != "[TOOL_CALLS]":
                expected = expected.replace(name, "")

    text, states = "", None
    for i, token_id in enumerate(ids):
        text, states = tokenizer.decode_incrementally(
            [token_id],
            prev_text=text,
            states=states,
            flush=i == len(ids) - 1,
            skip_special_tokens=skip_special_tokens,
        )
    assert text == expected
    tokens = tokenizer.convert_ids_to_tokens(_text(tokenizer, "🙂🙂"), skip_special_tokens)
    assert tokenizer.convert_tokens_to_string(tokens, skip_special_tokens) == "🙂🙂"


def test_the_grammar_vocabulary_it_describes(tokenizer):
    """xgrammar gets each token's raw bytes and control tokens as empty strings, which it never
    selects; llguidance's tokenizer covers the model's padded vocabulary."""
    tekken = tokenizer.tokenizer
    assert tokenizer.describes_grammar_vocabulary
    info = tokenizer.xgrammar_tokenizer_info()
    vocab = info["encoded_vocab"]
    assert info["stop_token_ids"] == [tokenizer.eos_token_id]
    assert len(vocab) == tekken.n_words
    assert set(vocab[: tekken.num_special_tokens]) == {""}
    text = "巴黎 São Paulo 🙂"
    assert b"".join(vocab[i] for i in _text(tokenizer, text)) == text.encode()
    n_vocab = tekken.n_words + 96
    assert tokenizer.llguidance_tokenizer(n_vocab).vocab_size == n_vocab
