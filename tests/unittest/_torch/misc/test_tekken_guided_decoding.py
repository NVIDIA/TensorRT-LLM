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
llguidance takes the tokenizer mistral-common wraps for it, whose control tokens are
special, so a JSON grammar never selects them either.
"""

import json
import shutil
from types import SimpleNamespace

import pytest
from test_common.mistral_tekken import (
    HF_TOKENIZER_IDS,
    HF_TOKENIZER_TEXT,
    TEKKEN_SOURCES,
    bundled_tekken_file,
    load_tekken,
    write_hf_tokenizer,
)

from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer
from tensorrt_llm._torch.pyexecutor.grammar_matcher import (
    LLGuidanceMatcherFactory,
    XGrammarMatcherFactory,
)
from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
    _llguidance_tokenizer,
    _load_guided_decoding_tokenizer,
    get_guided_decoding_config,
)
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig
from tensorrt_llm.tokenizer import TransformersTokenizer

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


def _text(tokenizer, text):
    return tokenizer.tokenizer.encode(text, bos=False, eos=False)


def _consume(matcher, ids):
    return all(matcher.accept_token(token_id) for token_id in ids)


def _allowed(matcher, n_words):
    import torch

    bitmask = torch.zeros(1, (n_words + 31) // 32, dtype=torch.int32)
    matcher.fill_next_token_bitmask(bitmask, 0)
    return {i for i in range(n_words) if (int(bitmask[0, i // 32]) >> (i % 32)) & 1}


def _matcher(tokenizer, llg, backend, guide_type, guide):
    """A matcher built as the executor builds it."""
    config = get_guided_decoding_config(backend, tokenizer)
    n_words = tokenizer.tokenizer.n_words
    if backend == "xgrammar":
        factory = XGrammarMatcherFactory(config, n_words, max_num_draft_tokens=8)
    else:
        factory = LLGuidanceMatcherFactory(config, n_words, tokenizer=llg)
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
    assert get_guided_decoding_config(None, tokenizer) is None
    with pytest.raises(ValueError, match="Unsupported guided decoding backend"):
        get_guided_decoding_config("outlines", tokenizer)


@pytest.fixture
def dual_format_dir(tmp_path):
    """A checkpoint directory with a Tekken file beside a complete HF checkpoint's
    config.json and tokenizer.json, as Ministral-3 ships them."""
    shutil.copy(bundled_tekken_file(), tmp_path / "tekken.json")
    write_hf_tokenizer(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mistral"}))
    return tmp_path


def _llm_args(checkpoint_format=None, custom_tokenizer=None):
    return SimpleNamespace(
        checkpoint_format=checkpoint_format,
        custom_tokenizer=custom_tokenizer,
        trust_remote_code=False,
    )


@pytest.mark.parametrize("checkpoint_format", ["mistral", "mistral_large_3"])
def test_native_checkpoint_reloads_its_tekken_tokenizer(dual_format_dir, checkpoint_format):
    """A rank that reloads the tokenizer masks the vocabulary its input processor encodes with."""
    tokenizer = _load_guided_decoding_tokenizer(_llm_args(checkpoint_format), str(dual_format_dir))
    assert type(tokenizer) is MistralTokenizer
    config = get_guided_decoding_config("xgrammar", tokenizer)
    assert config.encoded_vocab == tokenizer.xgrammar_tokenizer_info()["encoded_vocab"]


@pytest.mark.parametrize(
    "llm_args",
    [
        _llm_args(),
        _llm_args("HF"),
        # An explicit custom tokenizer takes precedence over the format's.
        _llm_args("mistral", custom_tokenizer="tensorrt_llm.tokenizer.TransformersTokenizer"),
    ],
    ids=["default", "HF", "custom_tokenizer"],
)
def test_other_checkpoints_reload_the_hf_tokenizer(dual_format_dir, llm_args):
    tokenizer = _load_guided_decoding_tokenizer(llm_args, str(dual_format_dir))
    assert type(tokenizer) is TransformersTokenizer
    assert tokenizer.encode(HF_TOKENIZER_TEXT) == HF_TOKENIZER_IDS


@pytest.mark.parametrize("checkpoint_format", ["mistral", "mistral_large_3"])
def test_a_native_checkpoint_without_tekken_reloads_its_hf_tokenizer(
    dual_format_dir, checkpoint_format
):
    """Guided decoding masks Tekken vocabularies only; a SentencePiece checkpoint keeps HF's."""
    (dual_format_dir / "tekken.json").unlink()
    tokenizer = _load_guided_decoding_tokenizer(_llm_args(checkpoint_format), str(dual_format_dir))
    assert type(tokenizer) is TransformersTokenizer
    assert tokenizer.encode(HF_TOKENIZER_TEXT) == HF_TOKENIZER_IDS


def test_custom_tokenizer_reloads_the_tekken_tokenizer_of_an_hf_checkpoint(dual_format_dir):
    tokenizer = _load_guided_decoding_tokenizer(
        _llm_args(custom_tokenizer="mistral_common"), str(dual_format_dir)
    )
    assert type(tokenizer) is MistralTokenizer


def test_tokenizers_without_their_own_vocabulary_keep_the_hf_path():
    config = GuidedDecodingConfig(backend=GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE)
    assert _llguidance_tokenizer(config, object(), 32) is None
    assert _llguidance_tokenizer(config, None, 32) is None


def test_llguidance_masks_a_padded_vocabulary(tokenizer):
    """The bitmask covers the model's padded vocabulary (a TP size that does not divide the
    vocabulary pads it); a padded id is never allowed and is refused without an error."""
    n_words = tokenizer.tokenizer.n_words
    padded = n_words + 96
    llg = tokenizer.llguidance_tokenizer(padded)
    assert llg.vocab_size == padded
    config = get_guided_decoding_config("llguidance", tokenizer)
    factory = LLGuidanceMatcherFactory(config, padded, tokenizer=llg)
    matcher = factory.create(
        GuidedDecodingParams(GuidedDecodingParams.GuideType.JSON_SCHEMA, _SCHEMA)
    )
    allowed = _allowed(matcher, padded)
    assert allowed and max(allowed) < n_words
    assert not matcher.accept_token(padded - 1)
    assert _consume(matcher, _text(tokenizer, '{"city": "Paris"}'))


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
    """A valid answer is accepted and prose is not; a rejected draft rolls back; neither the
    response's first token nor a JSON string is a control token."""
    tekken = tokenizer.tokenizer

    def make():
        return _matcher(tokenizer, llg, backend, guide_type, guide)

    matcher = make()
    assert min(_allowed(matcher, tekken.n_words)) >= tekken.num_special_tokens
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
