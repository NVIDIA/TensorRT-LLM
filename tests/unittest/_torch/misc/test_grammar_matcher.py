# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draft tokens from the padded part of the vocabulary reaching the guided-decoding matcher."""

import pytest

from tensorrt_llm._torch.pyexecutor.grammar_matcher import XGrammarMatcherFactory
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig

pytestmark = pytest.mark.cpu_only

# One token per printable ASCII character, enough to spell any JSON value, plus a stop token.
VOCAB = [chr(c) for c in range(32, 127)] + ["<|end|>"]
STOP_TOKEN_ID = len(VOCAB) - 1
# Models pad the LM head past the tokenizer's vocabulary; these extra ids have no token.
VOCAB_SIZE_PADDED = len(VOCAB) + 13


def _json_matcher():
    """A matcher for any JSON value over a vocabulary the LM head pads."""
    config = GuidedDecodingConfig(encoded_vocab=VOCAB, stop_token_ids=[STOP_TOKEN_ID])
    factory = XGrammarMatcherFactory(config, VOCAB_SIZE_PADDED, max_num_draft_tokens=4)
    return factory.create(GuidedDecodingParams(GuidedDecodingParams.GuideType.JSON))


@pytest.mark.parametrize("token_id", [len(VOCAB), VOCAB_SIZE_PADDED - 1])
def test_a_padded_draft_token_is_rejected(token_id: int):
    """A drafter can propose an id from the LM head's padding, which has no token.

    The matcher rejects it and stays usable. xgrammar before 0.2.6 crashed
    while logging this rejection.
    """
    matcher = _json_matcher()
    assert not matcher.accept_token(token_id)
    # The rejection leaves the matcher where it was: a JSON value can still start.
    assert matcher.accept_token(VOCAB.index("{"))


def test_vocabulary_ids_are_decided_by_the_grammar():
    matcher = _json_matcher()
    assert not matcher.accept_token(VOCAB.index("}"))
    assert matcher.accept_token(VOCAB.index("["))
