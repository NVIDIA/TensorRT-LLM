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
"""Embedding bias and bad words of one-model speculative decoding.

The per-slot tables and their device passes are plain torch, so they run on the CPU.
The last test decodes with speculation through the worker's logits-edit pass and compares
the result with a sequential decode that applies the same edits one token at a time.
"""

import types
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.speculative import logit_edits
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._torch.speculative.logit_edits import LogitEditState

pytestmark = pytest.mark.cpu_only

VOCAB = 12
SLOTS = 4


class _Request:
    """The request fields ``LogitEditState.observe`` reads."""

    def __init__(
        self,
        slot: Optional[int],
        tokens: list[int],
        *,
        bias: Optional[torch.Tensor] = None,
        bad_words: Optional[list[list[int]]] = None,
        last_chunk: bool = True,
    ):
        self.py_seq_slot = slot
        self.py_embedding_bias = bias
        self.py_bad_words = bad_words
        self.is_context_init_state = True
        self.is_last_context_chunk = last_chunk
        self.tokens = list(tokens)

    def get_tokens(self, beam_idx: int) -> list[int]:
        return list(self.tokens)


class _StubSpecWorker(SpecWorkerBase):
    """Concrete ``SpecWorkerBase`` that stubs out the abstract API."""

    @property
    def max_draft_len(self) -> int:
        return 8

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


def _make_worker() -> _StubSpecWorker:
    worker = _StubSpecWorker.__new__(_StubSpecWorker)
    torch.nn.Module.__init__(worker)
    return worker


def _state() -> LogitEditState:
    return LogitEditState.create(slot_capacity=SLOTS, vocab_size=VOCAB, device="cpu")


def _bias(entries: dict[int, float], size: int = VOCAB) -> torch.Tensor:
    bias = torch.zeros(size)
    for token, value in entries.items():
        bias[token] = value
    return bias


def _no_drafts(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((rows, 0), dtype=torch.int64),
        torch.zeros((rows, 0), dtype=torch.bool),
    )


def _banned(logits: torch.Tensor) -> list[list[int]]:
    return [torch.nonzero(torch.isneginf(row)).reshape(-1).tolist() for row in logits]


def test_embedding_bias_is_added_to_the_rows_of_its_slot():
    state = _state()
    bias = _bias({0: 1.5, 5: -2.0, VOCAB - 1: 4.0})
    state.observe([_Request(2, [1], bias=bias), _Request(0, [1])])

    logits = torch.arange(4 * VOCAB, dtype=torch.float32).reshape(4, VOCAB)
    expected = logits.clone()
    expected[0] += bias
    expected[2] += bias
    # Rows of slot 2, of slot 0 (no bias) and of a padding request.
    row_slots = torch.tensor([2, 0, 2, state.dummy_row])
    logit_edits.apply_embedding_bias(logits, state, row_slots)

    assert torch.equal(logits, expected)


def test_embedding_bias_past_the_vocabulary_is_ignored():
    state = _state()
    state.observe([_Request(1, [1], bias=_bias({3: 1.0, VOCAB + 2: 9.0}, size=VOCAB + 4))])

    logits = torch.zeros((1, VOCAB))
    logit_edits.apply_embedding_bias(logits, state, torch.tensor([1]))

    assert logits[0].tolist() == _bias({3: 1.0}).tolist()


def test_single_token_bad_words_are_banned_on_every_row():
    state = _state()
    state.observe([_Request(1, [3, 4], bad_words=[[7], [2], []])])

    logits = torch.zeros((3, VOCAB))
    row_slots = torch.tensor([1, 1, 0])
    logit_edits.apply_bad_words(logits, state, row_slots, *_no_drafts(3))

    assert _banned(logits) == [[2, 7], [2, 7], []]
    assert torch.equal(logits[2], torch.zeros(VOCAB))


def test_multi_token_bad_word_needs_its_prefix_at_the_end_of_the_sequence():
    state = _state()
    # The sequences end with [.., 3, 4] and [.., 4, 3]; the word is 3 4 -> 9.
    state.observe(
        [
            _Request(0, [8, 3, 4], bad_words=[[3, 4, 9], [4, 6]]),
            _Request(1, [8, 4, 3], bad_words=[[3, 4, 9]]),
        ]
    )

    logits = torch.zeros((2, VOCAB))
    logit_edits.apply_bad_words(logits, state, torch.tensor([0, 1]), *_no_drafts(2))

    assert _banned(logits) == [[6, 9], []]


def test_bad_word_longer_than_the_sequence_does_not_match():
    state = _state()
    state.observe([_Request(0, [4], bad_words=[[3, 4, 9]])])

    logits = torch.zeros((1, VOCAB))
    logit_edits.apply_bad_words(logits, state, torch.tensor([0]), *_no_drafts(1))

    assert _banned(logits) == [[]]


def test_bad_word_prefix_is_matched_through_the_drafts_before_the_row():
    state = _state()
    state.observe([_Request(3, [8, 3], bad_words=[[3, 4, 9], [4, 5, 6, 1]])])

    # One generation request verifying the drafts 4 5 6: row p follows drafts[:p].
    drafts = torch.tensor([[4, 5, 6]]).expand(4, 3)
    valid = torch.arange(3).unsqueeze(0) < torch.arange(4).unsqueeze(1)
    logits = torch.zeros((4, VOCAB))
    logit_edits.apply_bad_words(logits, state, torch.full((4,), 3), drafts, valid)

    # After [.. 3]: nothing. After [.. 3 4]: 9. After [.. 4 5]: nothing. After [.. 4 5 6]: 1.
    assert _banned(logits) == [[], [9], [], [1]]


def test_rules_sharing_a_token_ban_it_when_either_matches():
    state = _state()
    state.observe([_Request(0, [1, 2], bad_words=[[5, 9], [2, 9], [7, 9]])])

    logits = torch.zeros((1, VOCAB))
    logit_edits.apply_bad_words(logits, state, torch.tensor([0]), *_no_drafts(1))

    assert _banned(logits) == [[9]]


def test_accepted_tokens_advance_the_tail():
    state = _state()
    request = _Request(2, [8, 3], bad_words=[[3, 4, 9]])
    state.observe([request])

    # Two of the three tokens are accepted; the third must not reach the tail.
    logit_edits.advance_tails(
        state, torch.tensor([2]), torch.tensor([[7, 3, 4]]), torch.tensor([2])
    )
    logits = torch.zeros((1, VOCAB))
    logit_edits.apply_bad_words(logits, state, torch.tensor([2]), *_no_drafts(1))
    assert _banned(logits) == [[]]

    logit_edits.advance_tails(
        state, torch.tensor([2]), torch.tensor([[4, 0, 0]]), torch.tensor([1])
    )
    logits = torch.zeros((1, VOCAB))
    logit_edits.apply_bad_words(logits, state, torch.tensor([2]), *_no_drafts(1))
    assert _banned(logits) == [[9]]


def test_tail_keeps_only_the_longest_prefix_a_rule_can_read():
    state = _state()
    # A word of the maximum length: its prefix fills the whole tail.
    word = [token % VOCAB for token in range(logit_edits.MAX_BAD_WORD_LEN)]
    request = _Request(0, [VOCAB - 1], bad_words=[word])
    state.observe([request])
    for token in word[:-1]:
        logit_edits.advance_tails(
            state, torch.tensor([0]), torch.tensor([[token]]), torch.tensor([1])
        )

    logits = torch.zeros((1, VOCAB))
    logit_edits.apply_bad_words(logits, state, torch.tensor([0]), *_no_drafts(1))

    assert _banned(logits) == [[word[-1]]]


def test_new_slot_occupant_drops_the_previous_edits():
    state = _state()
    state.observe([_Request(1, [3, 4], bias=_bias({2: 5.0}), bad_words=[[7], [3, 4, 9]])])
    state.observe([_Request(1, [3, 4])])

    logits = torch.zeros((1, VOCAB))
    row_slots = torch.tensor([1])
    logit_edits.apply_embedding_bias(logits, state, row_slots)
    logit_edits.apply_bad_words(logits, state, row_slots, *_no_drafts(1))

    assert torch.equal(logits, torch.zeros((1, VOCAB)))


def test_same_occupant_is_loaded_once():
    state = _state()
    request = _Request(1, [3, 4], bad_words=[[3, 4, 9]])
    state.observe([request])
    request.is_context_init_state = False
    # A stale host history must not overwrite the tail the device advanced.
    logit_edits.advance_tails(state, torch.tensor([1]), torch.tensor([[5]]), torch.tensor([1]))
    state.observe([request])

    logits = torch.zeros((1, VOCAB))
    logit_edits.apply_bad_words(logits, state, torch.tensor([1]), *_no_drafts(1))

    assert _banned(logits) == [[]]


def test_chunked_prefill_restarts_the_tail_at_the_last_chunk():
    state = _state()
    request = _Request(0, [8, 3, 4], bad_words=[[3, 4, 9]], last_chunk=False)
    state.observe([request])
    # The token sampled while the request is still prefilling is discarded.
    logit_edits.advance_tails(state, torch.tensor([0]), torch.tensor([[5]]), torch.tensor([1]))
    request.is_last_context_chunk = True
    state.observe([request])

    logits = torch.zeros((1, VOCAB))
    logit_edits.apply_bad_words(logits, state, torch.tensor([0]), *_no_drafts(1))

    assert _banned(logits) == [[9]]


def test_padding_requests_are_not_loaded():
    state = _state()
    state.observe(
        [
            _Request(None, [1], bad_words=[[7]]),
            _Request(state.dummy_row, [1], bad_words=[[7]]),
        ]
    )

    logits = torch.zeros((1, VOCAB))
    logit_edits.apply_bad_words(
        logits, state, torch.tensor([state.dummy_row]), *_no_drafts(1)
    )

    assert _banned(logits) == [[]]


def _spec_metadata(state: LogitEditState, slots: list[int]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        enable_penalty=True,
        penalty_state=None,
        logit_edit_state=state,
        vocab_size=VOCAB,
        batch_slot_ids=torch.tensor(slots, dtype=torch.int64),
        is_ragged_verify=False,
        verify_lens=None,
        qo_indptr=None,
        total_verify_tokens=None,
    )


def test_worker_edits_a_copy_in_bias_then_ban_order():
    state = _state()
    # A bias on a banned token must not bring it back.
    state.observe([_Request(1, [3], bias=_bias({7: 50.0, 2: 1.0}), bad_words=[[7]])])
    metadata = _spec_metadata(state, [1])
    logits = torch.zeros((1, VOCAB), dtype=torch.bfloat16)

    edited = _make_worker()._edit_target_logits(
        logits,
        torch.empty((0, 2), dtype=torch.int32),
        num_contexts=1,
        batch_size=1,
        spec_metadata=metadata,
    )

    assert torch.equal(logits, torch.zeros_like(logits))
    assert edited.dtype == torch.float32
    assert _banned(edited) == [[7]]
    assert edited[0, 2].item() == 1.0


def test_worker_leaves_vocab_sharded_logits_unedited():
    state = _state()
    state.observe([_Request(1, [3], bias=_bias({2: 1.0}), bad_words=[[7]])])
    metadata = _spec_metadata(state, [1])
    shard = torch.zeros((1, VOCAB // 2))

    edited = _make_worker()._edit_target_logits(
        shard,
        torch.empty((0, 2), dtype=torch.int32),
        num_contexts=1,
        batch_size=1,
        spec_metadata=metadata,
    )

    assert torch.equal(edited, shard)


def test_no_draft_step_applies_the_edits_and_advances_the_tail():
    """A step the draft length schedule leaves without speculation edits its one row per
    request and records the sampled token, so a later bad word still sees it."""
    state = _state()
    state.observe([_Request(1, [8, 3], bias=_bias({4: 2.0}), bad_words=[[3, 4, 9]])])
    metadata = _spec_metadata(state, [1])
    worker = _make_worker()
    worker._execute_guided_decoder_if_present = lambda logits: None
    worker._sample_tokens_for_batch = lambda logits, *args: logits.argmax(dim=-1)

    def step(logits: torch.Tensor, num_contexts: int) -> dict:
        return worker.skip_drafting(
            input_ids=None,
            position_ids=None,
            hidden_states=None,
            logits=logits,
            attn_metadata=types.SimpleNamespace(num_seqs=1, num_contexts=num_contexts),
            spec_metadata=metadata,
            draft_model=None,
        )

    # Prefill: only the bias separates the tokens, so token 4 follows [.. 8 3].
    logits = torch.zeros((1, VOCAB))
    outputs = step(logits, num_contexts=1)
    assert outputs["new_tokens"].tolist() == [[4]]
    assert outputs["logits"] is logits
    assert torch.equal(logits, torch.zeros((1, VOCAB)))

    # Generation: the sequence now ends with 3 4, which bans the model's favorite 9.
    logits = torch.zeros((1, VOCAB))
    logits[0, 9] = 5.0
    logits[0, 6] = 3.0
    outputs = step(logits, num_contexts=0)
    assert outputs["new_tokens"].tolist() == [[6]]


def _reference_decode(
    table: torch.Tensor,
    prompt: list[int],
    num_tokens: int,
    bias: Optional[torch.Tensor],
    bad_words: list[list[int]],
) -> list[int]:
    """Greedy decode of the bigram model ``table``, editing one token's logits at a time."""
    tokens = list(prompt)
    for _ in range(num_tokens):
        logits = table[tokens[-1]].clone()
        if bias is not None:
            logits += bias
        for word in bad_words:
            prefix = word[:-1]
            if not prefix or tokens[-len(prefix) :] == prefix:
                logits[word[-1]] = float("-inf")
        tokens.append(int(logits.argmax()))
    return tokens[len(prompt) :]


@pytest.mark.parametrize("draft_len", [1, 3, 5])
def test_speculative_decode_matches_the_sequential_reference(draft_len: int):
    """Strict acceptance over the edited target logits decodes what a sequential decode
    with the same embedding bias and bad words does, for every request of the batch."""
    generator = torch.Generator().manual_seed(1234 + draft_len)
    table = torch.randn((VOCAB, VOCAB), generator=generator)
    num_tokens = 24
    prompts = [[1, 2, 3], [4, 5], [6]]
    slots = [3, 0, 2]

    # Request 0 carries edits chosen so that each of them changes its output: a bias, a ban
    # of the first token it would generate, and a ban of a 3-gram it then generates.
    bias = _bias({5: 0.75, 9: -1.25})
    first = _reference_decode(table, prompts[0], 1, bias, [])
    biased = _reference_decode(table, prompts[0], num_tokens, bias, [first])
    bad_words = [first, biased[1:4]]
    references = [
        _reference_decode(table, prompts[0], num_tokens, bias, bad_words),
        # Request 1 has a multi-token ban only; request 2 has no edits.
        None,
        _reference_decode(table, prompts[2], num_tokens, None, []),
    ]
    plain = _reference_decode(table, prompts[1], num_tokens, None, [])
    other_words = [plain[2:4], plain[5:9]]
    references[1] = _reference_decode(table, prompts[1], num_tokens, None, other_words)
    assert references[0] != biased
    assert references[1] != plain

    requests = [
        _Request(slots[0], prompts[0], bias=bias, bad_words=bad_words),
        _Request(slots[1], prompts[1], bad_words=other_words),
        _Request(slots[2], prompts[2]),
    ]
    batch = len(requests)
    state = _state()
    metadata = _spec_metadata(state, slots)
    worker = _make_worker()

    def draft(last_token: int) -> list[int]:
        """The drafter is the unedited model: its drafts ignore the bias and the bans."""
        drafts = []
        for _ in range(draft_len):
            last_token = int(table[last_token].argmax())
            drafts.append(last_token)
        return drafts

    # Prefill: every request samples its first token from one row.
    state.observe(requests)
    logits = torch.stack([table[request.tokens[-1]] for request in requests])
    edited = worker._edit_target_logits(
        logits, torch.empty((0, draft_len), dtype=torch.int32), batch, batch, metadata
    )
    accepted = edited.argmax(dim=-1, keepdim=True)
    worker._commit_accepted_tokens(
        (accepted, torch.ones(batch, dtype=torch.int32)), batch, metadata
    )
    generated = [[int(token)] for token in accepted[:, 0]]
    drafts = [draft(tokens[-1]) for tokens in generated]
    for request in requests:
        request.is_context_init_state = False

    num_accepted_drafts = 0
    while min(len(tokens) for tokens in generated) < num_tokens:
        # The host history is deliberately left stale: only the device tail follows the
        # accepted tokens, as with the overlap scheduler.
        state.observe(requests)
        inputs = torch.tensor([[tokens[-1]] + drafts[i] for i, tokens in enumerate(generated)])
        logits = table[inputs.reshape(-1)]
        draft_tokens = torch.tensor(drafts, dtype=torch.int32)
        edited = worker._edit_target_logits(logits, draft_tokens, 0, batch, metadata)
        targets = edited.argmax(dim=-1).reshape(batch, draft_len + 1)
        matches = (targets[:, :draft_len] == draft_tokens).to(torch.int32)
        num_accepted = 1 + torch.cumprod(matches, dim=1).sum(dim=1)
        worker._commit_accepted_tokens((targets, num_accepted), batch, metadata)
        for i in range(batch):
            count = int(num_accepted[i])
            num_accepted_drafts += count - 1
            generated[i].extend(targets[i, :count].tolist())
            drafts[i] = draft(generated[i][-1])

    for tokens, reference in zip(generated, references):
        assert tokens[:num_tokens] == reference
    # The test is only meaningful if drafts were accepted across the edited positions.
    assert num_accepted_drafts > 0
