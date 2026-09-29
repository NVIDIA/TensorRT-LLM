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
import math
from typing import List

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import guided_decoder as guided_decoder_module
from tensorrt_llm._torch.pyexecutor.guided_decoder import (
    GuidedDecoder,
    GuidedRequest,
    GuidedRequests,
    row_has_valid_token,
)
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig


def _empty_row(vocab_size_padded: int) -> torch.Tensor:
    return torch.zeros(math.ceil(vocab_size_padded / 32), dtype=torch.int32)


# 128000 is word-aligned; 128001 and 128031 leave a partial last word.
@pytest.mark.parametrize("vocab_size_padded", [128000, 128001, 128031])
def test_dead_end_row_has_no_valid_token(vocab_size_padded: int):
    """A grammar state with no valid continuation must be reported as dead."""
    assert not row_has_valid_token(_empty_row(vocab_size_padded), vocab_size_padded)


@pytest.mark.parametrize("vocab_size_padded", [128000, 128001, 128031])
@pytest.mark.parametrize("token_id_from_end", [0, 1, 32, 12345])
def test_single_valid_token_is_detected(vocab_size_padded: int, token_id_from_end: int):
    """A single set bit anywhere below vocab_size_padded keeps the row alive."""
    token_id = vocab_size_padded - 1 - token_id_from_end
    row = _empty_row(vocab_size_padded)
    row[token_id // 32] |= 1 << (token_id % 32)
    assert row_has_valid_token(row, vocab_size_padded)


@pytest.mark.parametrize("vocab_size_padded", [128001, 128031])
def test_trailing_padding_bits_do_not_count(vocab_size_padded: int):
    """Padding bits above vocab_size_padded must not mark a dead row valid.

    The apply kernel never reads them and the backends do not guarantee they
    are cleared, so counting them would let a fully masked row reach the
    sampler and produce a NaN logits row.
    """
    row = _empty_row(vocab_size_padded)
    num_words, num_tail_bits = divmod(vocab_size_padded, 32)
    # Set every bit of the last word that lies at or above vocab_size_padded.
    row[num_words] = torch.tensor(-1 << num_tail_bits, dtype=torch.int32)
    assert not row_has_valid_token(row, vocab_size_padded)

    # The highest in-range bit of that same partial word must still count.
    row[num_words] |= 1 << (num_tail_bits - 1)
    assert row_has_valid_token(row, vocab_size_padded)


def test_sign_bit_counts_as_valid_token():
    """Bit 31 of a word makes the int32 negative; it is still a valid token."""
    vocab_size_padded = 128000
    row = _empty_row(vocab_size_padded)
    row[0] = torch.tensor(-2147483648, dtype=torch.int32)  # only bit 31 set
    assert row_has_valid_token(row, vocab_size_padded)


# --- _build dead-end handling -------------------------------------------------

_VOCAB_SIZE = 128000
_SLOT = 0
# Any non-None value works: the tests pre-seed the matcher, so the grammar
# matcher factory is never asked to compile these params.
_GUIDED_PARAMS = object()


class _ScriptedMatcher:
    """Grammar matcher whose per-position bitmask rows are scripted by the test.

    `rows[i]` says whether the i-th `fill_next_token_bitmask` call should
    produce a row with a valid token; a False entry is a dead-end state. Calls
    past the end of the script produce a valid row, so that a regression which
    fails to stop at a dead end is caught by an assertion rather than by this
    helper running out of scripted rows.
    """

    def __init__(self, rows: List[bool]):
        self._rows = rows
        self.accepted: List[int] = []
        self.num_rolled_back = 0
        self._num_fills = 0

    def accept_token(self, token_id: int) -> bool:
        self.accepted.append(token_id)
        return True

    def rollback(self, num_tokens: int) -> None:
        self.num_rolled_back += num_tokens
        del self.accepted[len(self.accepted) - num_tokens :]

    def fill_next_token_bitmask(self, bitmask: torch.Tensor, index: int) -> None:
        has_valid_token = self._rows[self._num_fills] if self._num_fills < len(self._rows) else True
        self._num_fills += 1
        bitmask[index].zero_()
        if has_valid_token:
            bitmask[index][0] = 1

    def is_terminated(self) -> bool:
        return False


def _make_decoder(monkeypatch, max_num_draft_tokens: int) -> GuidedDecoder:
    # The factory needs a real tokenizer and a compiled grammar; stub it out so
    # __init__ runs unchanged while the tests drive a scripted matcher instead.
    monkeypatch.setattr(
        guided_decoder_module, "XGrammarMatcherFactory", lambda *args, **kwargs: None
    )
    return GuidedDecoder(
        GuidedDecodingConfig(),
        max_num_sequences=4,
        vocab_size_padded=_VOCAB_SIZE,
        max_num_draft_tokens=max_num_draft_tokens,
    )


def _generation_request(
    *,
    is_draft: bool,
    draft_tokens: List[int],
    request_id: int = 7,
    seq_slot: int = _SLOT,
) -> GuidedRequest:
    return GuidedRequest(
        guided_decoding_params=_GUIDED_PARAMS,
        request_id=request_id,
        seq_slot=seq_slot,
        is_generation_in_progress_state=True,
        new_token=11,
        is_draft=is_draft,
        draft_tokens=draft_tokens,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_build_fails_request_on_dead_end_row(monkeypatch):
    """A dead end fails that one request instead of reaching the sampler."""
    decoder = _make_decoder(monkeypatch, max_num_draft_tokens=0)
    decoder.grammar_matchers[_SLOT] = _ScriptedMatcher([False])
    requests = GuidedRequests(
        [_generation_request(is_draft=False, draft_tokens=[])],
        num_contexts=0,
        num_generations=1,
        max_num_draft_tokens=0,
    )

    failed_requests = decoder._build(requests)

    assert [req_id for req_id, _ in failed_requests] == [7]
    # The row must stay unguided so the apply kernel skips it entirely.
    assert decoder.token_mask_host[0].item() == 0
    # Cleared so the failed request is skipped by _rollback_rejected_tokens,
    # which would otherwise raise on a negative rollback count.
    assert decoder.num_advanced_tokens[_SLOT] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_build_draft_dead_end_is_rolled_back(monkeypatch):
    """A draft dead end terminates drafting and stays rollback-accurate."""
    decoder = _make_decoder(monkeypatch, max_num_draft_tokens=4)
    matcher = _ScriptedMatcher([False])
    decoder.grammar_matchers[_SLOT] = matcher
    requests = GuidedRequests(
        [_generation_request(is_draft=True, draft_tokens=[])],
        num_contexts=0,
        num_generations=1,
        max_num_draft_tokens=4,
    )

    failed_requests = decoder._build(requests)

    assert failed_requests == []
    assert decoder.is_draft_terminated[_SLOT]
    assert decoder.token_mask_host[0].item() == 0
    # The matcher accepted new_token before hitting the dead end, so the
    # drafting loop must roll that advance back; otherwise the target model
    # resumes from a matcher that is one token ahead.
    assert decoder.num_advanced_draft_tokens[_SLOT] == 1
    decoder._rollback_draft_tokens(requests)
    assert matcher.num_rolled_back == 1
    assert matcher.accepted == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_build_draft_position_dead_end_stops_guiding(monkeypatch):
    """A dead end at a draft position leaves later positions unguided."""
    decoder = _make_decoder(monkeypatch, max_num_draft_tokens=2)
    decoder.grammar_matchers[_SLOT] = _ScriptedMatcher([True, False])
    requests = GuidedRequests(
        [_generation_request(is_draft=False, draft_tokens=[21, 22])],
        num_contexts=0,
        num_generations=1,
        max_num_draft_tokens=2,
    )

    failed_requests = decoder._build(requests)

    # The request itself is still viable: only drafting stops early.
    assert failed_requests == []
    assert decoder.token_mask_host[0].item() == 1
    assert decoder.token_mask_host[1].item() == 0
    assert decoder.token_mask_host[2].item() == 0
    assert decoder.num_guided_tokens[_SLOT] == 1
    # new_token plus the one accepted draft token.
    assert decoder.num_advanced_tokens[_SLOT] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_build_dead_end_isolates_the_failing_request(monkeypatch):
    """Only the dead-end request fails; the rest of the batch keeps decoding.

    This is the property the fix exists for: before it, a single dead-end row
    took down the whole deployment via the sampler's global NaN assert.
    """
    decoder = _make_decoder(monkeypatch, max_num_draft_tokens=0)
    decoder.grammar_matchers[0] = _ScriptedMatcher([False])
    decoder.grammar_matchers[1] = _ScriptedMatcher([True])
    requests = GuidedRequests(
        [
            _generation_request(is_draft=False, draft_tokens=[], request_id=7, seq_slot=0),
            _generation_request(is_draft=False, draft_tokens=[], request_id=8, seq_slot=1),
        ],
        num_contexts=0,
        num_generations=2,
        max_num_draft_tokens=0,
    )

    failed_requests = decoder._build(requests)

    assert [req_id for req_id, _ in failed_requests] == [7]
    # The healthy request keeps its guided row and stays constrained.
    assert decoder.token_mask_host[0].item() == 0
    assert decoder.token_mask_host[1].item() == 1
    assert decoder.num_guided_tokens[1] == 1
    assert decoder.num_advanced_tokens[1] == 1
