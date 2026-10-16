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
from unittest import mock

import pytest

from tensorrt_llm._torch.pyexecutor.grammar_matcher import XGrammarMatcherFactory
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig

# The L0 CPU stage invokes registered files with `-m cpu_only`; without this
# marker every test is deselected and pytest exits with code 5.
pytestmark = pytest.mark.cpu_only

# One token per printable ASCII character, enough to spell any JSON value, plus a stop token.
VOCAB = [chr(c) for c in range(32, 127)] + ["<|end|>"]
STOP_TOKEN_ID = len(VOCAB) - 1
# Models pad the LM head past the tokenizer's vocabulary; these extra ids have no token.
VOCAB_SIZE_PADDED = len(VOCAB) + 13


def _json_matcher():
    """An XGrammarMatcher for any JSON value, with its xgrammar matcher wrapped in a spy."""
    config = GuidedDecodingConfig(encoded_vocab=VOCAB, stop_token_ids=[STOP_TOKEN_ID])
    factory = XGrammarMatcherFactory(config, VOCAB_SIZE_PADDED, max_num_draft_tokens=4)
    matcher = factory.create(GuidedDecodingParams(GuidedDecodingParams.GuideType.JSON))
    matcher._matcher = mock.Mock(wraps=matcher._matcher)
    return matcher


@pytest.mark.parametrize("token_id", [len(VOCAB), VOCAB_SIZE_PADDED - 1])
def test_padded_id_is_rejected_without_reaching_xgrammar(token_id: int):
    """A draft token in the LM-head padding is rejected before xgrammar sees it.

    xgrammar before 0.2.6 lists padded ids as special tokens but keeps no decoded
    string for them, and its rejection message for a special token prints that string:
    an out-of-bounds read that can crash the process.
    """
    matcher = _json_matcher()
    assert not matcher.accept_token(token_id)
    matcher._matcher.accept_token.assert_not_called()

    # The rejection leaves the matcher where it was: a JSON value can still start.
    assert matcher.accept_token(VOCAB.index("{"))


def test_vocabulary_ids_still_reach_xgrammar():
    """Ids the tokenizer knows are decided by the grammar, accepted or rejected."""
    matcher = _json_matcher()
    assert not matcher.accept_token(VOCAB.index("}"))
    assert matcher.accept_token(VOCAB.index("["))
    assert matcher._matcher.accept_token.call_args_list == [
        mock.call(VOCAB.index("}")),
        mock.call(VOCAB.index("[")),
    ]
