# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON-schema grammars must accept in-range negative numbers."""

import json

from tensorrt_llm._torch.pyexecutor.grammar_matcher import (
    XGrammarMatcherFactory,
    _drop_negative_lower_bounds,
)
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig


def test_negative_lower_bounds_are_dropped_everywhere():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "minimum": -1000000, "maximum": 5},
            "b": {"type": "number", "exclusiveMinimum": -0.5},
            "c": {"type": "integer", "minimum": 3},
            "d": {"type": "array", "items": {"type": "integer", "minimum": -2}},
        },
    }
    assert _drop_negative_lower_bounds(schema) == {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "maximum": 5},
            "b": {"type": "number"},
            "c": {"type": "integer", "minimum": 3},
            "d": {"type": "array", "items": {"type": "integer"}},
        },
    }


def test_negative_minimum_schema_accepts_negative_values():
    vocab = ["-", "5", "[EOS]"]
    config = GuidedDecodingConfig(
        backend=GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
        encoded_vocab=vocab,
        stop_token_ids=[2],
    )
    factory = XGrammarMatcherFactory(config, vocab_size_padded=len(vocab))
    schema = json.dumps({"type": "integer", "minimum": -10})

    matcher = factory.create(
        GuidedDecodingParams(GuidedDecodingParams.GuideType.JSON_SCHEMA, schema)
    )
    assert matcher.accept_token(0) and matcher.accept_token(1) and matcher.accept_token(2)
    assert matcher.is_terminated()

    # A bare "-" is not a number.
    matcher = factory.create(
        GuidedDecodingParams(GuidedDecodingParams.GuideType.JSON_SCHEMA, schema)
    )
    assert matcher.accept_token(0)
    assert not matcher.accept_token(2)
