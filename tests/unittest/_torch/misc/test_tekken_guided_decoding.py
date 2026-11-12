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
"""Guided decoding over a Mistral Tekken vocabulary, with xgrammar and llguidance.

xgrammar matches a grammar against each token's bytes. A Tekken token is raw bytes that
need not be valid UTF-8, so the vocabulary keeps them as bytes. Control tokens are empty
strings, which xgrammar treats as special: a grammar never selects them, so `[INST]` or
`[IMG]` cannot appear inside a JSON string, and end-of-sequence is the only stop token.

llguidance takes the Tekken tokenizer as mistral-common wraps it: ordinary tokens are
their bytes and control tokens are special, which a Lark grammar selects only by id.
For a vocabulary mistral-common's grammar factory supports, a JSON response is
constrained with the factory's JSON grammar, which from the model-settings Tekken (v15)
on lets the response open with one `[THINK]` ... `[/THINK]` section.

The tests run on CPU with the Tekken file bundled with mistral-common (v3) and, when
they are under LLM_MODELS_ROOT, with the v13 and v15 files of Mistral-Large-3 and
Mistral-Small-4.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
    _llguidance_json_schema_lark,
    _llguidance_tokenizer,
    get_guided_decoding_config,
)
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig

pytestmark = pytest.mark.cpu_only

_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
_JSON_GUIDES = [("JSON_SCHEMA", json.dumps(_SCHEMA)), ("JSON", None)]

_MODEL_TEKKEN_FILES = {
    "v13": "Mistral-Large-3-675B/Mistral-Large-3-675B-Instruct-2512/tekken.json",
    "v15": "Mistral-Small-4-119B-2603/Mistral-Small-4-119B-2603/tekken.json",
}


@pytest.fixture(scope="module", params=["bundled", "v13", "v15"])
def tokenizer(request):
    mistral_common = pytest.importorskip("mistral_common")
    from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer

    if request.param == "bundled":
        files = sorted((Path(mistral_common.__file__).parent / "data").glob("tekken*.json"))
        if not files:
            pytest.skip("this mistral_common ships no tekken tokenizer file")
        return MistralTokenizer.from_pretrained(str(files[-1]))
    root = llm_models_root()
    path = None if root is None else root / _MODEL_TEKKEN_FILES[request.param]
    if path is None or not path.is_file():
        pytest.skip(f"needs {_MODEL_TEKKEN_FILES[request.param]} under LLM_MODELS_ROOT")
    return MistralTokenizer.from_pretrained(str(path))


@pytest.fixture(scope="module")
def llg(tokenizer):
    pytest.importorskip("llguidance")
    return tokenizer.llguidance_tokenizer(tokenizer.tokenizer.n_words)


def _control(tokenizer, name):
    return tokenizer.tokenizer.get_control_token(name)


def _text(tokenizer, text):
    return tokenizer.tokenizer.encode(text, bos=False, eos=False)


def _consume(matcher, ids):
    for token_id in ids:
        if not matcher.accept_token(token_id):
            return False
    return True


def _allowed(matcher, n_words):
    import torch

    bitmask = torch.zeros(1, (n_words + 31) // 32, dtype=torch.int32)
    matcher.fill_next_token_bitmask(bitmask, 0)
    return {i for i in range(n_words) if (int(bitmask[0, i // 32]) >> (i % 32)) & 1}


def _xgrammar_matcher(tokenizer, guide_type, guide, max_num_draft_tokens=0):
    from tensorrt_llm._torch.pyexecutor.grammar_matcher import XGrammarMatcherFactory
    from tensorrt_llm.bindings.executor import GuidedDecodingParams

    config = get_guided_decoding_config("xgrammar", tokenizer)
    factory = XGrammarMatcherFactory(
        config,
        vocab_size_padded=tokenizer.tokenizer.n_words,
        max_num_draft_tokens=max_num_draft_tokens,
    )
    return factory.create(
        GuidedDecodingParams(getattr(GuidedDecodingParams.GuideType, guide_type), guide)
    )


def _llguidance_matcher(tokenizer, llg, guide_type, guide, *, json_schema_lark=True):
    """An llguidance matcher built as the executor builds it."""
    from tensorrt_llm._torch.pyexecutor.grammar_matcher import LLGuidanceMatcherFactory
    from tensorrt_llm.bindings.executor import GuidedDecodingParams

    config = get_guided_decoding_config("llguidance", tokenizer)
    factory = LLGuidanceMatcherFactory(
        config,
        tokenizer.tokenizer.n_words,
        tokenizer=llg,
        json_schema_lark=(
            _llguidance_json_schema_lark(config, tokenizer) if json_schema_lark else None
        ),
    )
    return factory.create(
        GuidedDecodingParams(getattr(GuidedDecodingParams.GuideType, guide_type), guide)
    )


def test_vocab_is_raw_bytes_with_control_tokens_empty(tokenizer):
    tekken = tokenizer.tokenizer
    info = tokenizer.xgrammar_tokenizer_info()
    vocab = info["encoded_vocab"]
    assert info["stop_token_ids"] == [tokenizer.eos_token_id]
    assert len(vocab) == tekken.n_words
    assert vocab[: tekken.num_special_tokens] == [""] * tekken.num_special_tokens
    pieces = [tekken.id_to_byte_piece(i) for i in range(tekken.num_special_tokens, tekken.n_words)]
    expected = ["" if piece.startswith(b"\x00") else piece for piece in pieces]
    assert vocab[tekken.num_special_tokens :] == expected
    # Some tokens are partial UTF-8 sequences: only bytes represent them.
    undecodable = 0
    for piece in pieces:
        try:
            piece.decode("utf-8")
        except UnicodeDecodeError:
            undecodable += 1
    assert undecodable > 0


def test_xgrammar_config_takes_the_byte_vocab(tokenizer):
    config = get_guided_decoding_config("xgrammar", tokenizer)
    assert config.backend == GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR
    assert config.encoded_vocab == tokenizer.xgrammar_tokenizer_info()["encoded_vocab"]
    assert config.stop_token_ids == [tokenizer.eos_token_id]
    assert config.tokenizer_str is None
    assert get_guided_decoding_config(None, tokenizer) is None


def test_llguidance_config_defers_to_the_tokenizer(tokenizer):
    config = get_guided_decoding_config("llguidance", tokenizer)
    assert config.backend == GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE
    assert config.tokenizer_str is None and config.encoded_vocab is None
    assert config.stop_token_ids == [tokenizer.eos_token_id]
    n_words = tokenizer.tokenizer.n_words
    assert _llguidance_tokenizer(config, tokenizer, n_words) is not None
    xgrammar = get_guided_decoding_config("xgrammar", tokenizer)
    assert _llguidance_tokenizer(xgrammar, tokenizer, n_words) is None
    with pytest.raises(ValueError, match="Unsupported guided decoding backend"):
        get_guided_decoding_config("outlines", tokenizer)


def test_tokenizers_without_their_own_vocabulary_keep_the_hf_path():
    """A tokenizer that does not describe its own vocabulary gets no Tekken handling."""
    config = GuidedDecodingConfig(backend=GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE)
    assert _llguidance_tokenizer(config, object(), 32) is None
    assert _llguidance_tokenizer(config, None, 32) is None
    assert _llguidance_json_schema_lark(config, object()) is None
    assert _llguidance_json_schema_lark(config, SimpleNamespace(renders_grammars=False)) is None


@pytest.mark.parametrize("text", ['{"city": "Paris"}', '{"city": "巴黎 São Paulo 🙂"}'])
def test_xgrammar_accepts_the_tokens_of_a_valid_answer(tokenizer, text):
    matcher = _xgrammar_matcher(tokenizer, "JSON_SCHEMA", json.dumps(_SCHEMA))
    assert _consume(matcher, _text(tokenizer, text))
    assert matcher.accept_token(tokenizer.eos_token_id)
    assert matcher.is_terminated()


def test_xgrammar_never_allows_control_tokens_inside_a_string(tokenizer):
    tekken = tokenizer.tokenizer
    matcher = _xgrammar_matcher(tokenizer, "JSON_SCHEMA", json.dumps(_SCHEMA))
    assert _consume(matcher, _text(tokenizer, '{"city": "'))
    allowed = _allowed(matcher, tekken.n_words)
    assert allowed, "a JSON string accepts ordinary text"
    assert min(allowed) >= tekken.num_special_tokens


def test_xgrammar_rolls_back_draft_tokens(tokenizer):
    """Speculative decoding rolls a rejected draft back off the matcher."""
    tekken = tokenizer.tokenizer
    matcher = _xgrammar_matcher(tokenizer, "JSON_SCHEMA", json.dumps(_SCHEMA), 8)
    assert _consume(matcher, _text(tokenizer, '{"city": '))
    before = _allowed(matcher, tekken.n_words)
    draft = _text(tokenizer, '"Paris"}')
    assert _consume(matcher, draft)
    matcher.rollback(len(draft))
    assert _allowed(matcher, tekken.n_words) == before


def test_llguidance_tokenizer_is_tekken(tokenizer, llg):
    tekken = tokenizer.tokenizer
    assert llg.vocab_size == tekken.n_words
    assert llg.eos_token == tokenizer.eos_token_id
    assert all(llg.is_special_token(i) for i in range(tekken.num_special_tokens))
    # llguidance marks special tokens with a leading 0xFF byte, which no UTF-8 text has; the
    # raw 0xFF byte token reads as special too.
    ordinary = [
        i
        for i in range(tekken.num_special_tokens, 2000)
        if not tekken.id_to_byte_piece(i).startswith(b"\xff")
    ]
    assert len(ordinary) > 900
    assert not any(llg.is_special_token(i) for i in ordinary)
    for text in ("Hello, world!", '{"city": "巴黎 São Paulo 🙂"}'):
        ids = _text(tokenizer, text)
        assert llg.tokenize_str(text) == ids
        assert llg.decode_bytes(ids) == text.encode()


@pytest.mark.parametrize("guide_type, guide", _JSON_GUIDES)
def test_llguidance_json_response_over_tekken(tokenizer, llg, guide_type, guide):
    tekken = tokenizer.tokenizer
    matcher = _llguidance_matcher(tokenizer, llg, guide_type, guide)
    assert _consume(matcher, _text(tokenizer, '{"city": "Paris"}'))
    assert matcher.accept_token(tokenizer.eos_token_id)
    matcher = _llguidance_matcher(tokenizer, llg, guide_type, guide)
    assert not _consume(matcher, _text(tokenizer, 'Sure: {"city": "Paris"}'))
    matcher = _llguidance_matcher(tokenizer, llg, guide_type, guide)
    assert _consume(matcher, _text(tokenizer, '{"city": "'))
    allowed = _allowed(matcher, tekken.n_words)
    assert allowed and min(allowed) >= tekken.num_special_tokens


def test_llguidance_rolls_back_draft_tokens(tokenizer, llg):
    """Speculative decoding rolls a rejected draft back off the matcher."""
    tekken = tokenizer.tokenizer
    matcher = _llguidance_matcher(tokenizer, llg, "JSON_SCHEMA", json.dumps(_SCHEMA))
    assert _consume(matcher, _text(tokenizer, '{"city": '))
    before = _allowed(matcher, tekken.n_words)
    draft = _text(tokenizer, '"Paris"}')
    assert _consume(matcher, draft)
    matcher.rollback(len(draft))
    assert _allowed(matcher, tekken.n_words) == before


def test_llguidance_json_schema_lark_needs_llguidance_and_a_grammar_tokenizer(tokenizer):
    llguidance = get_guided_decoding_config("llguidance", tokenizer)
    xgrammar = get_guided_decoding_config("xgrammar", tokenizer)
    # mistral-common's grammar factory renders grammars for Tekken v11 and later; an older
    # vocabulary (the bundled v3 file) constrains JSON from its first token.
    rendered = _llguidance_json_schema_lark(llguidance, tokenizer)
    assert (rendered is not None) is tokenizer.renders_grammars
    assert tokenizer.renders_grammars is (tokenizer.version >= 11)
    assert _llguidance_json_schema_lark(xgrammar, tokenizer) is None
    if not tokenizer.renders_grammars:
        with pytest.raises(ValueError, match="renders no grammars"):
            tokenizer.llguidance_json_schema_lark(json.dumps(_SCHEMA))


@pytest.fixture(scope="module")
def reasoning_vocab(tokenizer, llg):
    """Skips vocabularies whose JSON grammar has no reasoning section."""
    tekken = tokenizer.tokenizer
    if "[THINK]" not in tekken._special_tokens_reverse_vocab:
        pytest.skip("this Tekken vocabulary has no reasoning tokens")
    if not tokenizer.renders_grammars:
        pytest.skip("mistral-common renders no grammars for this Tekken version")
    if not tekken.version.supports_model_settings:
        pytest.skip("mistral-common reasons before JSON from the model-settings Tekken on")
    return tekken


def _thought(tokenizer, text="The user wants a city as JSON.\nParis fits."):
    return (
        [_control(tokenizer, "[THINK]")]
        + _text(tokenizer, text)
        + [_control(tokenizer, "[/THINK]")]
    )


@pytest.mark.parametrize("guide_type, guide", _JSON_GUIDES)
def test_llguidance_json_response_may_open_with_a_reasoning_section(
    tokenizer, llg, reasoning_vocab, guide_type, guide
):
    """Reasoning on: [THINK] thoughts [/THINK], then the JSON. Reasoning off: the JSON."""
    answer = _text(tokenizer, '{"city": "Paris"}')
    for ids in (_thought(tokenizer) + answer, answer):
        matcher = _llguidance_matcher(tokenizer, llg, guide_type, guide)
        assert _consume(matcher, ids)
        assert matcher.accept_token(tokenizer.eos_token_id)


@pytest.mark.parametrize("guide_type, guide", _JSON_GUIDES)
def test_llguidance_json_response_holds_nothing_else(
    tokenizer, llg, reasoning_vocab, guide_type, guide
):
    think = _control(tokenizer, "[THINK]")
    thought = _thought(tokenizer, "Thinking.")

    no_answer = _llguidance_matcher(tokenizer, llg, guide_type, guide)
    assert _consume(no_answer, thought)
    assert not no_answer.accept_token(tokenizer.eos_token_id)
    second_thought = _llguidance_matcher(tokenizer, llg, guide_type, guide)
    assert _consume(second_thought, thought)
    assert not second_thought.accept_token(think)
    after_answer = _llguidance_matcher(tokenizer, llg, guide_type, guide)
    assert _consume(after_answer, _text(tokenizer, '{"city": "Paris"}'))
    assert not after_answer.accept_token(think)


def test_llguidance_json_response_never_takes_control_tokens_in_a_string(
    tokenizer, llg, reasoning_vocab
):
    matcher = _llguidance_matcher(tokenizer, llg, "JSON_SCHEMA", json.dumps(_SCHEMA))
    assert _consume(matcher, _thought(tokenizer, "Thinking.") + _text(tokenizer, '{"city": "'))
    allowed = _allowed(matcher, reasoning_vocab.n_words)
    assert allowed and min(allowed) >= reasoning_vocab.num_special_tokens


def test_llguidance_json_response_rolls_back_a_draft_across_the_section_end(
    tokenizer, llg, reasoning_vocab
):
    """A rejected draft that closed the reasoning section and opened the JSON."""
    matcher = _llguidance_matcher(tokenizer, llg, "JSON_SCHEMA", json.dumps(_SCHEMA))
    assert _consume(matcher, [_control(tokenizer, "[THINK]")] + _text(tokenizer, "Thinking."))
    before = _allowed(matcher, reasoning_vocab.n_words)
    assert _control(tokenizer, "[/THINK]") in before
    draft = [_control(tokenizer, "[/THINK]")] + _text(tokenizer, '{"city"')
    assert _consume(matcher, draft)
    matcher.rollback(len(draft))
    assert _allowed(matcher, reasoning_vocab.n_words) == before


def test_llguidance_json_response_without_the_factory_grammar_starts_at_the_json(
    tokenizer, llg, reasoning_vocab
):
    """Without mistral-common's JSON grammar, [THINK] cannot open the response."""
    matcher = _llguidance_matcher(
        tokenizer, llg, "JSON_SCHEMA", json.dumps(_SCHEMA), json_schema_lark=False
    )
    assert _control(tokenizer, "[THINK]") not in _allowed(matcher, reasoning_vocab.n_words)
    matcher = _llguidance_matcher(tokenizer, llg, "JSON_SCHEMA", json.dumps(_SCHEMA))
    assert _control(tokenizer, "[THINK]") in _allowed(matcher, reasoning_vocab.n_words)


def test_llguidance_json_response_of_a_pre_model_settings_vocabulary_starts_at_the_json(
    tokenizer, llg
):
    """A grammar-factory vocabulary before v15 (Mistral-Large-3's v13) has no reasoning
    section in front of the JSON, even with the reasoning tokens."""
    tekken = tokenizer.tokenizer
    if not tokenizer.renders_grammars or tekken.version.supports_model_settings:
        pytest.skip("needs a grammar-factory Tekken vocabulary older than v15")
    if "[THINK]" not in tekken._special_tokens_reverse_vocab:
        pytest.skip("this Tekken vocabulary has no reasoning tokens")
    matcher = _llguidance_matcher(tokenizer, llg, "JSON_SCHEMA", json.dumps(_SCHEMA))
    assert _control(tokenizer, "[THINK]") not in _allowed(matcher, tekken.n_words)
