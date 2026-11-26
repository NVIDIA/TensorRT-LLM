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
from tensorrt_llm.tokenizer import TransformersTokenizer

pytestmark = pytest.mark.cpu_only

_SCHEMA = json.dumps(
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
)
_GUIDES = [("JSON_SCHEMA", _SCHEMA), ("JSON", None)]


# The vocabulary paths do not depend on the Tekken version: the file mistral-common ships.
@pytest.fixture(scope="module")
def tokenizer():
    return load_tekken("bundled")


# A tensor-parallel size that does not divide the vocabulary pads it.
@pytest.fixture(scope="module")
def n_vocab(tokenizer):
    return tokenizer.tokenizer.n_words + 96


def _text(tokenizer, text):
    return tokenizer.tokenizer.encode(text, bos=False, eos=False)


def _consume(matcher, ids):
    return all(matcher.accept_token(token_id) for token_id in ids)


def _allowed(matcher, n_words):
    import torch

    bitmask = torch.zeros(1, (n_words + 31) // 32, dtype=torch.int32)
    matcher.fill_next_token_bitmask(bitmask, 0)
    return {i for i in range(n_words) if (int(bitmask[0, i // 32]) >> (i % 32)) & 1}


def _matcher(tokenizer, n_vocab, backend, guide_type, guide):
    """A matcher built as the executor builds it, over a padded vocabulary."""
    config = get_guided_decoding_config(backend, tokenizer)
    if backend == "xgrammar":
        factory = XGrammarMatcherFactory(config, n_vocab, max_num_draft_tokens=8)
    else:
        llg = _llguidance_tokenizer(config, tokenizer, n_vocab)
        factory = LLGuidanceMatcherFactory(config, n_vocab, tokenizer=llg)
    return factory.create(
        GuidedDecodingParams(getattr(GuidedDecodingParams.GuideType, guide_type), guide)
    )


def test_guided_decoding_configs(tokenizer):
    n_words = tokenizer.tokenizer.n_words
    xgrammar = get_guided_decoding_config("xgrammar", tokenizer)
    assert xgrammar.tokenizer_str is None
    assert xgrammar.stop_token_ids == [tokenizer.eos_token_id]
    assert len(xgrammar.encoded_vocab) == n_words

    # The guided decoder gets the llguidance tokenizer itself, so the config carries none.
    llguidance = get_guided_decoding_config("llguidance", tokenizer)
    assert llguidance.tokenizer_str is None and llguidance.encoded_vocab is None
    assert llguidance.stop_token_ids is None
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


_HF_TOKENIZER = "tensorrt_llm.tokenizer.TransformersTokenizer"


@pytest.mark.parametrize(
    "llm_args,keep_tekken,expected",
    [
        (_llm_args("mistral_large_3"), True, MistralTokenizer),
        # Guided decoding masks Tekken vocabularies only; a SentencePiece checkpoint keeps HF's.
        (_llm_args("mistral"), False, TransformersTokenizer),
        (_llm_args(), True, TransformersTokenizer),
        # An explicit custom tokenizer takes precedence over the format's.
        (_llm_args("mistral", custom_tokenizer=_HF_TOKENIZER), True, TransformersTokenizer),
        (_llm_args(custom_tokenizer="mistral_common"), True, MistralTokenizer),
    ],
    ids=[
        "native_format",
        "native_format_without_tekken",
        "default",
        "custom_hf_tokenizer",
        "custom_mistral_common",
    ],
)
def test_a_rank_reloads_the_tokenizer_its_input_processor_encodes_with(
    dual_format_dir, llm_args, keep_tekken, expected
):
    if not keep_tekken:
        (dual_format_dir / "tekken.json").unlink()
    tokenizer = _load_guided_decoding_tokenizer(llm_args, str(dual_format_dir))
    assert type(tokenizer) is expected
    if expected is TransformersTokenizer:
        assert tokenizer.encode(HF_TOKENIZER_TEXT) == HF_TOKENIZER_IDS


@pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
@pytest.mark.parametrize("guide_type, guide", _GUIDES, ids=["json_schema", "json_object"])
def test_json_response(tokenizer, n_vocab, backend, guide_type, guide):
    """A valid answer is accepted and prose is not; a rejected draft rolls back; neither the
    response's first token nor a JSON string is a control token or a padded id."""
    tekken = tokenizer.tokenizer

    def make():
        return _matcher(tokenizer, n_vocab, backend, guide_type, guide)

    def allowed(matcher):
        ids = _allowed(matcher, n_vocab)
        assert ids and tekken.num_special_tokens <= min(ids) and max(ids) < tekken.n_words
        return ids

    matcher = make()
    allowed(matcher)
    assert not matcher.accept_token(n_vocab - 1)
    assert _consume(matcher, _text(tokenizer, '{"city": "巴黎 São Paulo 🙂"}'))
    assert matcher.accept_token(tokenizer.eos_token_id)
    assert not _consume(make(), _text(tokenizer, 'Sure: {"city": "Paris"}'))

    matcher = make()
    assert _consume(matcher, _text(tokenizer, '{"city": '))
    before = allowed(matcher)
    draft = _text(tokenizer, '"Paris"}')
    assert _consume(matcher, draft)
    matcher.rollback(len(draft))
    assert allowed(matcher) == before
    assert _consume(matcher, _text(tokenizer, '"'))
    allowed(matcher)
