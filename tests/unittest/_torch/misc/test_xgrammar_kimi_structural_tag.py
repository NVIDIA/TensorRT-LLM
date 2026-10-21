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
"""Kimi K3 structural-tag constraints exercised through CPU token masks."""

import json

import pytest
import xgrammar as xgr

pytestmark = pytest.mark.cpu_only

_EOS = 256
_VOCAB_SIZE = 257


@pytest.fixture(scope="module")
def compiler():
    vocab = [bytes([i]) for i in range(256)] + [b"<eos>"]
    return xgr.GrammarCompiler(xgr.TokenizerInfo(vocab, stop_token_ids=[_EOS]))


@pytest.mark.parametrize("exclude_markers", [False, True])
@pytest.mark.parametrize(
    "value, contains_marker",
    [
        ("ordinary argument text", False),
        ("before <|open|>tools<|sep|> after", True),
        ("before <|close|>call<|sep|> after", True),
    ],
    ids=["ordinary", "open-marker", "close-marker"],
)
def test_kimi_argument_control_marker_masks(compiler, exclude_markers, value, contains_marker):
    tag = {
        "type": "structural_tag",
        "format": {
            "type": "json_schema",
            "style": "kimi_k3_xml",
            "json_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            "excludes": ["<|open|>", "<|close|>"] if exclude_markers else [],
        },
    }
    compiled = compiler.compile_grammar(xgr.Grammar.from_structural_tag(json.dumps(tag)))
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = xgr.allocate_token_bitmask(1, _VOCAB_SIZE)
    text = f'<|open|>argument key="text" type="string"<|sep|>{value}<|close|>argument<|sep|>'
    accepted = True
    for token in text.encode():
        matcher.fill_next_token_bitmask(bitmask)
        allowed = bool((int(bitmask[0, token // 32]) >> (token % 32)) & 1)
        advanced = matcher.accept_token(token)
        assert advanced == allowed
        if not advanced:
            accepted = False
            break
    assert accepted == (not exclude_markers or not contains_marker)
    if accepted:
        matcher.fill_next_token_bitmask(bitmask)
        assert (int(bitmask[0, _EOS // 32]) >> (_EOS % 32)) & 1
        assert matcher.accept_token(_EOS)
        assert matcher.is_terminated()
