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
"""MistralTokenizer over Tekken vocabularies: pickling and streaming detokenization."""

import pickle

import pytest
from test_common.mistral_tekken import TEKKEN_SOURCES, load_tekken

from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer

pytestmark = pytest.mark.cpu_only

PROMPT = "The capital of France is"


@pytest.fixture(scope="module", params=TEKKEN_SOURCES)
def tokenizer(request):
    return load_tekken(request.param)


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
    """Streamed text equals decode(): control tokens stay unless skipped, and skipping keeps
    [TOOL_CALLS] only. Characters split over tokens come back whole."""
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
