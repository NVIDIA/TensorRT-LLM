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
"""MistralTokenizer over a Tekken vocabulary: pickling and streaming detokenization.

The tests run with the Tekken file bundled with mistral-common (v3) and, when they are
under LLM_MODELS_ROOT, with the v13 and v15 files of Mistral-Large-3 and Mistral-Small-4,
whose vocabularies have the reasoning control tokens.
"""

import pickle
from pathlib import Path

import pytest
from utils.llm_data import llm_models_root

mistral_common = pytest.importorskip("mistral_common")

from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer  # noqa: E402

pytestmark = pytest.mark.cpu_only

PROMPT = "The capital of France is"

_MODEL_TEKKEN_FILES = {
    "v13": "Mistral-Large-3-675B/Mistral-Large-3-675B-Instruct-2512/tekken.json",
    "v15": "Mistral-Small-4-119B-2603/Mistral-Small-4-119B-2603/tekken.json",
}


def _tekken_file(source: str) -> Path:
    if source == "bundled":
        files = sorted((Path(mistral_common.__file__).parent / "data").glob("tekken*.json"))
        if not files:
            pytest.skip("this mistral_common ships no tekken tokenizer file")
        return files[-1]
    root = llm_models_root()
    path = None if root is None else root / _MODEL_TEKKEN_FILES[source]
    if path is None or not path.is_file():
        pytest.skip(f"needs {_MODEL_TEKKEN_FILES[source]} under LLM_MODELS_ROOT")
    return path


@pytest.fixture(scope="module", params=["bundled", "v13", "v15"])
def tokenizer(request) -> MistralTokenizer:
    return MistralTokenizer.from_pretrained(str(_tekken_file(request.param)))


def _control(tokenizer: MistralTokenizer, name: str) -> int:
    return tokenizer.tokenizer.get_control_token(name)


def _text(tokenizer: MistralTokenizer, text: str) -> list[int]:
    return tokenizer.tokenizer.encode(text, bos=False, eos=False)


def _stream(tokenizer: MistralTokenizer, ids: list[int], **kwargs) -> str:
    """Detokenize ``ids`` one token at a time, as a streamed response does."""
    text, states = "", None
    for i, token_id in enumerate(ids):
        text, states = tokenizer.decode_incrementally(
            [token_id], prev_text=text, states=states, flush=i == len(ids) - 1, **kwargs
        )
    return text


@pytest.mark.parametrize("serializer", ["pickle", "cloudpickle"])
def test_pickle_round_trip_keeps_the_class(tokenizer, serializer):
    module = pickle if serializer == "pickle" else pytest.importorskip("cloudpickle")
    restored = module.loads(module.dumps(tokenizer))
    assert type(restored) is MistralTokenizer
    assert type(restored.tokenizer) is type(tokenizer.tokenizer)
    ids = tokenizer.encode(PROMPT)
    assert ids[0] == tokenizer.bos_token_id
    assert restored.encode(PROMPT) == ids
    assert restored.encode(PROMPT, add_special_tokens=True) == ids
    assert restored.decode(ids, skip_special_tokens=True) == PROMPT
    assert restored.vocab_size == tokenizer.vocab_size


def test_vocab_keys_every_id_by_its_piece(tokenizer):
    vocab = tokenizer._vocab_dict
    assert all(isinstance(piece, str) for piece in vocab)
    tekken = tokenizer.tokenizer
    for token_id in (0, tokenizer.bos_token_id, tekken.num_special_tokens, tekken.n_words - 1):
        assert vocab[tekken.id_to_piece(token_id)] <= token_id


def _reply_ids(tokenizer: MistralTokenizer) -> tuple[list[int], list[str]]:
    """Ids of a reply with control tokens around multi-byte text, and the control tokens."""
    names = ["[TOOL_CALLS]", "[INST]", "[/INST]"]
    if "[THINK]" in tokenizer.tokenizer._special_tokens_reverse_vocab:
        names = ["[THINK]", "[/THINK]"] + names
    ids = []
    for name in names:
        ids += [_control(tokenizer, name)] + _text(tokenizer, f" 巴黎 São Paulo 🙂 {name[1:3]}")
    return ids + _text(tokenizer, " done."), names


def test_streaming_keeps_control_tokens_when_not_skipped(tokenizer):
    """Streamed text equals decode(): the control tokens tool and reasoning parsers read stay."""
    ids, names = _reply_ids(tokenizer)
    expected = tokenizer.decode(ids, skip_special_tokens=False)
    for name in names:
        assert name in expected
    assert "🙂" in expected and "�" not in expected
    assert _stream(tokenizer, ids, skip_special_tokens=False) == expected
    # The whole reply at once decodes the same way.
    text, _ = tokenizer.decode_incrementally(ids, flush=True, skip_special_tokens=False)
    assert text == expected


def test_streaming_skips_control_tokens_but_tool_calls(tokenizer):
    """skip_special_tokens=True drops every control token but [TOOL_CALLS], as before."""
    ids, names = _reply_ids(tokenizer)
    expected = tokenizer.decode(ids, skip_special_tokens=False)
    for name in names:
        if name != "[TOOL_CALLS]":
            expected = expected.replace(name, "")
    assert "[TOOL_CALLS]" in expected
    assert _stream(tokenizer, ids, skip_special_tokens=True) == expected


@pytest.mark.parametrize("skip_special_tokens", [False, True])
def test_convert_ids_to_tokens_round_trips_split_characters(tokenizer, skip_special_tokens):
    """A character split over tokens comes back whole once all its bytes are there."""
    ids = _text(tokenizer, "🙂🙂")
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
    assert (
        tokenizer.convert_tokens_to_string(tokens, skip_special_tokens=skip_special_tokens)
        == "🙂🙂"
    )
