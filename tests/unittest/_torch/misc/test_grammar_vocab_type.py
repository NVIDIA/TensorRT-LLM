# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grammars must decode a byte-level vocabulary (" " spelled "Ġ") as such."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.grammar_matcher import XGrammarMatcherFactory
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig
from tensorrt_llm.tokenizer.tokenizer import _is_byte_level_vocab


def _tokenizer(space_token):
    tokens = [] if space_token is None else [space_token]
    return SimpleNamespace(
        encode=lambda text, add_special_tokens: [7] * len(tokens),
        convert_ids_to_tokens=lambda ids: tokens,
    )


@pytest.mark.parametrize("space_token, expected", [("Ġ", True), (" ", False), (None, False)])
def test_byte_level_vocab_detection(space_token, expected):
    assert _is_byte_level_vocab(_tokenizer(space_token)) is expected


@pytest.mark.parametrize("vocab_type, accepted", [("byte_level", True), (None, False)])
def test_byte_level_vocab_matches_spaces(vocab_type, accepted):
    # "Ġa" is " a" in a byte-level vocabulary and the literal "Ġa" otherwise.
    vocab = ["Ġa", "b", "[EOS]"]
    config = GuidedDecodingConfig(
        backend=GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
        encoded_vocab=vocab,
        vocab_type=vocab_type,
        stop_token_ids=[2],
    )
    factory = XGrammarMatcherFactory(config, vocab_size_padded=len(vocab))
    matcher = factory.create(GuidedDecodingParams(GuidedDecodingParams.GuideType.REGEX, " ab"))
    assert (matcher.accept_token(0) and matcher.accept_token(1)) is accepted
