# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A completed grammar must end on the request's own terminators."""

from types import SimpleNamespace

import pytest
import xgrammar

from tensorrt_llm._torch.pyexecutor.grammar_matcher import XGrammarMatcherFactory
from tensorrt_llm._torch.pyexecutor.guided_decoder import _grammar_stop_token_ids
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig

# Tokenizer EOS at 4 and an end-of-message token at 5, like Kimi's [EOS] and
# <|end_of_msg|>; requests stop on the latter.
_VOCAB = ["a", "b", "c", "d", "[EOS]", "<|end_of_msg|>"]
_TOKENIZER_EOS = 4
_END_OF_MSG = 5


def _request(**kwargs):
    fields = dict(py_guided_decoding_params=object(), py_end_id=None, py_stop_words_list=None)
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def test_stop_token_ids_are_end_id_and_single_token_stop_words():
    request = _request(py_end_id=_TOKENIZER_EOS, py_stop_words_list=[[5], [1, 2], [4]])
    assert _grammar_stop_token_ids(request) == [_TOKENIZER_EOS, _END_OF_MSG]


def test_stop_token_ids_skip_ignored_end_id_and_unguided_requests():
    assert _grammar_stop_token_ids(_request(py_end_id=-1, py_stop_words_list=[[5]])) == [5]
    assert _grammar_stop_token_ids(_request(py_end_id=-1)) is None
    assert _grammar_stop_token_ids(_request(py_guided_decoding_params=None, py_end_id=4)) is None


def _allowed_after_grammar(stop_token_ids):
    config = GuidedDecodingConfig(
        backend=GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
        encoded_vocab=_VOCAB,
        stop_token_ids=[_TOKENIZER_EOS],
    )
    factory = XGrammarMatcherFactory(config, vocab_size_padded=len(_VOCAB))
    params = GuidedDecodingParams(GuidedDecodingParams.GuideType.REGEX, "ab")
    matcher = factory.create(params, stop_token_ids)
    assert matcher.accept_token(0) and matcher.accept_token(1)
    bitmask = xgrammar.allocate_token_bitmask(1, len(_VOCAB))
    matcher.fill_next_token_bitmask(bitmask, 0)
    word = int(bitmask[0, 0])
    return {i for i in range(len(_VOCAB)) if (word >> i) & 1}


@pytest.mark.parametrize(
    "stop_token_ids, expected",
    [(None, {_TOKENIZER_EOS}), ([_TOKENIZER_EOS, _END_OF_MSG], {_TOKENIZER_EOS, _END_OF_MSG})],
)
def test_completed_grammar_admits_request_stop_tokens(stop_token_ids, expected):
    assert _allowed_after_grammar(stop_token_ids) == expected
