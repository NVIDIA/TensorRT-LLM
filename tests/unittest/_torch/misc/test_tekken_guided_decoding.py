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
"""Guided decoding over Mistral Tekken vocabularies, with xgrammar and llguidance.

xgrammar takes each token's raw bytes; control tokens are empty strings it never selects.
llguidance takes the tokenizer mistral-common wraps for it, and for a vocabulary
mistral-common's grammar factory supports (v11+), constrains a JSON response with the
factory's grammar, which from the model-settings Tekken (v15) on may open with one
``[THINK]`` ... ``[/THINK]`` section.
"""

import json
from types import SimpleNamespace

import pytest
from test_common.mistral_tekken import TEKKEN_SOURCES, load_tekken

from tensorrt_llm._torch.pyexecutor.grammar_matcher import (
    LLGuidanceMatcherFactory,
    XGrammarMatcherFactory,
)
from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
    _llguidance_json_schema_lark,
    _llguidance_tokenizer,
    get_guided_decoding_config,
)
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig

pytestmark = pytest.mark.cpu_only

_SCHEMA = json.dumps(
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
)
_GUIDES = [("JSON_SCHEMA", _SCHEMA), ("JSON", None)]


@pytest.fixture(scope="module", params=TEKKEN_SOURCES)
def tokenizer(request):
    return load_tekken(request.param)


@pytest.fixture(scope="module")
def llg(tokenizer):
    return tokenizer.llguidance_tokenizer(tokenizer.tokenizer.n_words)


def _control(tokenizer, name):
    return tokenizer.tokenizer.get_control_token(name)


def _text(tokenizer, text):
    return tokenizer.tokenizer.encode(text, bos=False, eos=False)


def _consume(matcher, ids):
    return all(matcher.accept_token(token_id) for token_id in ids)


def _allowed(matcher, n_words):
    import torch

    bitmask = torch.zeros(1, (n_words + 31) // 32, dtype=torch.int32)
    matcher.fill_next_token_bitmask(bitmask, 0)
    return {i for i in range(n_words) if (int(bitmask[0, i // 32]) >> (i % 32)) & 1}


def _matcher(tokenizer, llg, backend, guide_type, guide, *, json_schema_lark=True):
    """A matcher built as the executor builds it."""
    config = get_guided_decoding_config(backend, tokenizer)
    n_words = tokenizer.tokenizer.n_words
    if backend == "xgrammar":
        factory = XGrammarMatcherFactory(config, n_words, max_num_draft_tokens=8)
    else:
        lark = _llguidance_json_schema_lark(config, tokenizer) if json_schema_lark else None
        factory = LLGuidanceMatcherFactory(config, n_words, tokenizer=llg, json_schema_lark=lark)
    return factory.create(
        GuidedDecodingParams(getattr(GuidedDecodingParams.GuideType, guide_type), guide)
    )


def test_guided_decoding_configs(tokenizer):
    tekken = tokenizer.tokenizer
    n_special, n_words = tekken.num_special_tokens, tekken.n_words
    xgrammar = get_guided_decoding_config("xgrammar", tokenizer)
    assert xgrammar.tokenizer_str is None
    assert xgrammar.stop_token_ids == [tokenizer.eos_token_id]
    vocab = xgrammar.encoded_vocab
    assert len(vocab) == n_words and vocab[:n_special] == [""] * n_special
    pieces = [tekken.id_to_byte_piece(i) for i in range(n_special, n_words)]
    assert vocab[n_special:] == ["" if piece.startswith(b"\x00") else piece for piece in pieces]

    llguidance = get_guided_decoding_config("llguidance", tokenizer)
    assert llguidance.tokenizer_str is None and llguidance.encoded_vocab is None
    assert llguidance.stop_token_ids == [tokenizer.eos_token_id]
    assert _llguidance_tokenizer(llguidance, tokenizer, n_words) is not None
    assert _llguidance_tokenizer(xgrammar, tokenizer, n_words) is None
    assert tokenizer.renders_grammars is (tokenizer.version >= 11)
    rendered = _llguidance_json_schema_lark(llguidance, tokenizer)
    assert (rendered is not None) is tokenizer.renders_grammars
    assert _llguidance_json_schema_lark(xgrammar, tokenizer) is None
    assert get_guided_decoding_config(None, tokenizer) is None
    with pytest.raises(ValueError, match="Unsupported guided decoding backend"):
        get_guided_decoding_config("outlines", tokenizer)


def test_tokenizers_without_their_own_vocabulary_keep_the_hf_path():
    config = GuidedDecodingConfig(backend=GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE)
    assert _llguidance_tokenizer(config, object(), 32) is None
    assert _llguidance_tokenizer(config, None, 32) is None
    assert _llguidance_json_schema_lark(config, object()) is None
    assert _llguidance_json_schema_lark(config, SimpleNamespace(renders_grammars=False)) is None


def test_llguidance_tokenizer_is_tekken(tokenizer, llg):
    tekken = tokenizer.tokenizer
    assert llg.vocab_size == tekken.n_words and llg.eos_token == tokenizer.eos_token_id
    assert all(llg.is_special_token(i) for i in range(tekken.num_special_tokens))
    # llguidance marks special tokens with a leading 0xFF byte, so the raw 0xFF token is too.
    ordinary = [
        i
        for i in range(tekken.num_special_tokens, 2000)
        if not tekken.id_to_byte_piece(i).startswith(b"\xff")
    ]
    assert len(ordinary) > 900 and not any(llg.is_special_token(i) for i in ordinary)
    text = '{"city": "巴黎 São Paulo 🙂"}'
    ids = _text(tokenizer, text)
    assert llg.tokenize_str(text) == ids and llg.decode_bytes(ids) == text.encode()


@pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
@pytest.mark.parametrize("guide_type, guide", _GUIDES)
def test_json_response(tokenizer, llg, backend, guide_type, guide):
    """A valid answer is accepted and prose is not; a rejected draft rolls back; a JSON string
    never takes a control token."""
    tekken = tokenizer.tokenizer

    def make():
        return _matcher(tokenizer, llg, backend, guide_type, guide)

    matcher = make()
    assert _consume(matcher, _text(tokenizer, '{"city": "巴黎 São Paulo 🙂"}'))
    assert matcher.accept_token(tokenizer.eos_token_id)
    assert not _consume(make(), _text(tokenizer, 'Sure: {"city": "Paris"}'))

    matcher = make()
    assert _consume(matcher, _text(tokenizer, '{"city": '))
    before = _allowed(matcher, tekken.n_words)
    draft = _text(tokenizer, '"Paris"}')
    assert _consume(matcher, draft)
    matcher.rollback(len(draft))
    assert _allowed(matcher, tekken.n_words) == before
    assert _consume(matcher, _text(tokenizer, '"'))
    allowed = _allowed(matcher, tekken.n_words)
    assert allowed and min(allowed) >= tekken.num_special_tokens


def _reasoning_vocab(tokenizer):
    tekken = tokenizer.tokenizer
    if "[THINK]" not in tekken._special_tokens_reverse_vocab or not tokenizer.renders_grammars:
        pytest.skip("needs a grammar-factory Tekken vocabulary with reasoning tokens")
    return tekken


def test_only_the_model_settings_vocabulary_reasons_before_json(tokenizer, llg):
    """v15 may open a JSON response with [THINK]; v13 and the plain JSON-schema grammar cannot."""
    tekken = _reasoning_vocab(tokenizer)
    think = _control(tokenizer, "[THINK]")

    def opening(**kwargs):
        matcher = _matcher(tokenizer, llg, "llguidance", "JSON_SCHEMA", _SCHEMA, **kwargs)
        return _allowed(matcher, tekken.n_words)

    assert think not in opening(json_schema_lark=False)
    assert (think in opening()) is tekken.version.supports_model_settings


@pytest.mark.parametrize("guide_type, guide", _GUIDES)
def test_json_response_after_a_reasoning_section(tokenizer, llg, guide_type, guide):
    tekken = _reasoning_vocab(tokenizer)
    if not tekken.version.supports_model_settings:
        pytest.skip("mistral-common reasons before JSON from the model-settings Tekken on")
    think, end = _control(tokenizer, "[THINK]"), _control(tokenizer, "[/THINK]")
    thought = [think] + _text(tokenizer, "Thinking.") + [end]
    answer = _text(tokenizer, '{"city": "Paris"}')

    def make():
        return _matcher(tokenizer, llg, "llguidance", guide_type, guide)

    for ids in (thought + answer, answer):
        matcher = make()
        assert _consume(matcher, ids) and matcher.accept_token(tokenizer.eos_token_id)
    # Nothing else: no answer, a second section, or a section after the answer.
    for ids, rejected in ((thought, tokenizer.eos_token_id), (thought, think), (answer, think)):
        matcher = make()
        assert _consume(matcher, ids) and not matcher.accept_token(rejected)
    matcher = make()
    assert _consume(matcher, thought + _text(tokenizer, '{"city": "'))
    allowed = _allowed(matcher, tekken.n_words)
    assert allowed and min(allowed) >= tekken.num_special_tokens
    # A rejected draft that closed the section and opened the JSON rolls back.
    matcher = make()
    assert _consume(matcher, [think] + _text(tokenizer, "Thinking."))
    before = _allowed(matcher, tekken.n_words)
    draft = [end] + _text(tokenizer, '{"city"')
    assert end in before and _consume(matcher, draft)
    matcher.rollback(len(draft))
    assert _allowed(matcher, tekken.n_words) == before
