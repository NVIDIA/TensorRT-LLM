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
"""``/v1/completions`` logprobs: each ``text_offset`` is where its token starts."""

import pytest

from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.serve.postprocess_handlers import create_completion_logprobs

# Tokens of unequal length, so an offset built from the wrong token's length
# lands somewhere else.
PIECES = ["A", " wonderful", " world", "!"]
TOKEN_IDS = list(range(len(PIECES)))


class _Tokenizer:
    """Decodes a token id to its piece, as a vocabulary lookup does."""

    def decode(self, token_id):
        return PIECES[token_id]


def _scalar_logprobs():
    return [-0.5] * len(TOKEN_IDS)


def _top_logprobs():
    return [{tid: Logprob(logprob=-0.5, rank=1)} for tid in TOKEN_IDS]


@pytest.mark.parametrize(
    "logprobs", [_scalar_logprobs, _top_logprobs], ids=["scalar", "top_logprobs"]
)
@pytest.mark.parametrize(
    "initial_offset",
    # 0 is a whole non-streaming completion. A nonzero offset is a streamed
    # chunk that carries several tokens (e.g. stream_interval > 1), offset by
    # the text already sent.
    [0, 5],
)
def test_text_offset_is_where_each_token_starts(logprobs, initial_offset):
    result = create_completion_logprobs(TOKEN_IDS, _Tokenizer(), logprobs(), initial_offset)

    assert result.tokens == PIECES
    assert result.text_offset == [initial_offset + offset for offset in (0, 1, 11, 17)]
    text = "x" * initial_offset + "".join(PIECES)
    for token, offset in zip(result.tokens, result.text_offset):
        assert text[offset : offset + len(token)] == token
