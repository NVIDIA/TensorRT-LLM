# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draft tokens from the padded part of the vocabulary reaching the guided-decoding matcher."""

import json

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.pyexecutor.grammar_matcher import (
    LLGuidanceMatcherFactory,
    XGrammarMatcherFactory,
)
from tensorrt_llm._torch.pyexecutor.py_executor_creator import get_guided_decoding_config
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


def test_llguidance_over_a_hugging_face_tokenizer_rejects_a_padded_draft_token():
    """llguidance builds its tokenizer from the Hugging Face tokenizer JSON.

    A padded id is rejected without breaking the matcher, a valid answer is accepted
    through to the stop token, and a rejected draft rolls back.
    """
    from tensorrt_llm.tokenizer import TransformersTokenizer

    root = llm_models_root()
    model_dir = None if root is None else root / "Qwen3" / "Qwen3-0.6B"
    if model_dir is None or not model_dir.is_dir():
        pytest.skip("needs Qwen3/Qwen3-0.6B under LLM_MODELS_ROOT")
    tokenizer = TransformersTokenizer.from_pretrained(str(model_dir))
    vocab_size = len(tokenizer.tokenizer)
    vocab_size_padded = (vocab_size + 127) // 128 * 128 + 128
    config = get_guided_decoding_config("llguidance", tokenizer)
    assert config.backend == GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE
    factory = LLGuidanceMatcherFactory(config, vocab_size_padded)
    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    params = GuidedDecodingParams(GuidedDecodingParams.GuideType.JSON_SCHEMA, json.dumps(schema))

    matcher = factory.create(params)
    assert not matcher.accept_token(vocab_size_padded - 1)
    answer = tokenizer.encode('{"city": "Paris"}', add_special_tokens=False)
    for token_id in answer:
        assert matcher.accept_token(token_id), token_id
    assert matcher.accept_token(tokenizer.eos_token_id)
    assert matcher.is_terminated()

    matcher = factory.create(params)
    assert matcher.accept_token(answer[0])
    for token_id in answer[1:]:
        assert matcher.accept_token(token_id)
    matcher.rollback(len(answer) - 1)
    for token_id in answer[1:]:
        assert matcher.accept_token(token_id)
    assert not matcher.accept_token(answer[0])
