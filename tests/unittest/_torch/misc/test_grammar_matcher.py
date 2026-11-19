# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draft tokens from the padded part of the vocabulary reaching the guided-decoding matcher."""

import pytest

from tensorrt_llm._torch.pyexecutor.grammar_matcher import XGrammarMatcherFactory
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig
from tensorrt_llm.tokenizer.tokenizer import _xgrammar_tokenizer_info

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


class _ByteLevelTiktoken:
    """The surface of a tiktoken tokenizer that the guided-decoding setup reads. Its vocabulary spells a
    space with the byte-level character "\u0120", as Kimi's does."""

    vocab_files_names = {"vocab_file": "tiktoken.model"}
    vocab = {"a": 0, "b": 1, "\u0120": 2, "<|end|>": 3}
    eos_token_id = 3

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        return [self.vocab["\u0120" if c == " " else c] for c in text]

    def convert_ids_to_tokens(self, ids):
        return [next(t for t, i in self.vocab.items() if i == id_) for id_ in ids]


def test_a_byte_level_tiktoken_vocabulary_spells_a_space():
    """A tiktoken tokenizer has no tokenizer string to detect its vocabulary type from, so the setup
    states it. Read as byte-level, "\u0120" is a space; read as raw, it is that character and a grammar
    needing a space rejects it."""
    info = _xgrammar_tokenizer_info(_ByteLevelTiktoken())
    assert info["vocab_type"] == "byte_level"
    space = _ByteLevelTiktoken.vocab["\u0120"]
    regex = GuidedDecodingParams(GuidedDecodingParams.GuideType.REGEX, "a b")

    for config, accepts_space in (
        (GuidedDecodingConfig(**info), True),
        (GuidedDecodingConfig(**{**info, "vocab_type": None}), False),
    ):
        matcher = XGrammarMatcherFactory(config, len(_ByteLevelTiktoken.vocab)).create(regex)
        assert matcher.accept_token(_ByteLevelTiktoken.vocab["a"])
        assert matcher.accept_token(space) is accepts_space
