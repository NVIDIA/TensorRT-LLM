# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-device NGram drafting kernel (``torch.ops.trtllm.ngram_extend_and_draft_op``).

Checks the fused history-extend + suffix-lookup kernel against a plain Python
reference of the same algorithm, including the public-pool cross-slot search, the
oldest/latest tie-break, masked rows, the extend-only mode and CUDA graph replay.
"""

import random

import pytest
import torch

import tensorrt_llm  # noqa: F401  # registers the trtllm torch ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="NGram kernel runs on CUDA")

VOCAB = 4  # tiny vocab so suffixes recur often


def reference_extend_and_draft(
    histories: list[list[int]],
    slot_ids: list[int],
    row_mask: list[int],
    accepted: list[list[int]],
    num_accepted: list[int],
    draft_len: int,
    max_ngram: int,
    use_oldest: bool,
    public_pool: bool,
    max_seq_len: int,
) -> tuple[list[list[int]], list[int]]:
    """Python model of the kernel. Extends ``histories`` in place and returns (drafts, match_lens)."""
    num_slots = len(histories)
    for row, slot in enumerate(slot_ids):
        if not row_mask[row]:
            continue
        num_new = min(num_accepted[row], len(accepted[row]), max_seq_len - len(histories[slot]))
        histories[slot].extend(accepted[row][:num_new])

    drafts = [[0] * draft_len for _ in slot_ids]
    match_lens = [0] * len(slot_ids)
    if draft_len == 0:
        return drafts, match_lens

    for row, slot in enumerate(slot_ids):
        history = histories[slot]
        max_match = min(max_ngram, len(history) - 1)
        if not row_mask[row] or max_match <= 0:
            continue

        best = None
        searched_slots = [slot]
        if public_pool:
            searched_slots += [s for s in range(num_slots) if s != slot and len(histories[s]) > 0]
        for searched in searched_slots:
            tokens = histories[searched]
            for end in range(len(tokens) - 1):
                k = 0
                while k < max_match and end - k >= 0 and tokens[end - k] == history[-1 - k]:
                    k += 1
                if k == 0:
                    continue
                key = (
                    k,
                    searched == slot,
                    (max_seq_len - 1 - end) if use_oldest else end,
                    num_slots - 1 - searched,
                )
                if best is None or key > best[0]:
                    best = (key, searched, end)
        if best is None:
            continue
        (k, _, _, _), searched, end = best
        continuation = histories[searched][end + 1 : end + 1 + draft_len]
        drafts[row] = continuation + [0] * (draft_len - len(continuation))
        match_lens[row] = k
    return drafts, match_lens


class _DeviceState:
    """Device mirror of the reference's slot pool plus the kernel's static buffers."""

    def __init__(self, histories: list[list[int]], max_seq_len: int, batch: int, draft_len: int):
        num_slots = len(histories)
        self.max_seq_len = max_seq_len
        self.history_tokens = torch.zeros(
            (num_slots, max_seq_len), dtype=torch.int32, device="cuda"
        )
        self.history_lens = torch.zeros((num_slots,), dtype=torch.int32, device="cuda")
        for slot, tokens in enumerate(histories):
            if tokens:
                self.history_tokens[slot, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
            self.history_lens[slot] = len(tokens)
        self.slot_ids = torch.zeros((batch,), dtype=torch.int32, device="cuda")
        self.row_mask = torch.zeros((batch,), dtype=torch.int32, device="cuda")
        self.accepted = torch.zeros((batch, draft_len + 1), dtype=torch.int32, device="cuda")
        self.num_accepted = torch.zeros((batch,), dtype=torch.int32, device="cuda")
        self.drafts = torch.zeros((batch, draft_len), dtype=torch.int32, device="cuda")
        self.match_lens = torch.zeros((batch,), dtype=torch.int32, device="cuda")

    def load_step(self, slot_ids, row_mask, accepted, num_accepted) -> None:
        self.slot_ids.copy_(torch.tensor(slot_ids, dtype=torch.int32))
        self.row_mask.copy_(torch.tensor(row_mask, dtype=torch.int32))
        self.accepted.copy_(torch.tensor(accepted, dtype=torch.int32))
        self.num_accepted.copy_(torch.tensor(num_accepted, dtype=torch.int32))

    def launch(self, max_ngram: int, use_oldest: bool, public_pool: bool) -> None:
        torch.ops.trtllm.ngram_extend_and_draft_op(
            self.history_tokens,
            self.history_lens,
            self.slot_ids,
            self.row_mask,
            self.accepted,
            self.num_accepted,
            self.drafts,
            self.match_lens,
            max_ngram,
            use_oldest,
            public_pool,
        )

    def histories(self) -> list[list[int]]:
        lens = self.history_lens.tolist()
        tokens = self.history_tokens.tolist()
        return [row[:n] for row, n in zip(tokens, lens)]


def _random_histories(rng: random.Random, num_slots: int, max_len: int) -> list[list[int]]:
    lengths = [rng.randint(0, max_len) for _ in range(num_slots)]
    lengths[0] = 1  # too short to match anything (max_match == 0)
    lengths[1] = 0  # empty slot: skipped by the public-pool search
    return [[rng.randrange(VOCAB) for _ in range(n)] for n in lengths]


def _random_step(rng: random.Random, batch: int, draft_len: int):
    accepted = [[rng.randrange(VOCAB) for _ in range(draft_len + 1)] for _ in range(batch)]
    num_accepted = [rng.randint(1, draft_len + 1) for _ in range(batch)]
    return accepted, num_accepted


def _assert_step_matches(state: _DeviceState, histories, ref_drafts, ref_match_lens) -> None:
    torch.cuda.synchronize()
    assert state.drafts.tolist() == ref_drafts
    assert state.match_lens.tolist() == ref_match_lens
    assert state.histories() == histories


@pytest.mark.parametrize("use_oldest", [True, False])
@pytest.mark.parametrize("public_pool", [True, False])
@pytest.mark.parametrize("max_ngram", [1, 3])
def test_kernel_matches_reference(use_oldest: bool, public_pool: bool, max_ngram: int):
    rng = random.Random(1234)
    num_slots, max_seq_len, draft_len = 6, 96, 4
    dummy_slot = num_slots - 1
    histories = _random_histories(rng, num_slots, 40)
    histories[dummy_slot] = []
    # Rows: three live requests, one masked live request (chunked context), one masked dummy.
    slot_ids = [0, 2, 3, 4, dummy_slot]
    row_mask = [1, 1, 1, 0, 0]
    batch = len(slot_ids)

    state = _DeviceState(histories, max_seq_len, batch, draft_len)
    for _ in range(6):
        accepted, num_accepted = _random_step(rng, batch, draft_len)
        ref_drafts, ref_match_lens = reference_extend_and_draft(
            histories,
            slot_ids,
            row_mask,
            accepted,
            num_accepted,
            draft_len,
            max_ngram,
            use_oldest,
            public_pool,
            max_seq_len,
        )
        state.load_step(slot_ids, row_mask, accepted, num_accepted)
        state.launch(max_ngram, use_oldest, public_pool)
        _assert_step_matches(state, histories, ref_drafts, ref_match_lens)

    # Masked rows never produce drafts and never touch their slot.
    assert state.drafts[3:].abs().sum().item() == 0
    assert state.match_lens[3:].tolist() == [0, 0]
    assert len(histories[4]) == len(_random_histories(random.Random(1234), num_slots, 40)[4])


def test_prefers_longest_then_oldest_or_latest():
    """Longest suffix wins; the position flag only breaks ties between equal lengths."""
    max_seq_len, draft_len = 32, 3
    # history: 1 2 3 | 7 | 1 2 3 | 8 | 2 3   -> suffix [2, 3]; "1 2 3" occurrences at end=2 and end=6
    history = [1, 2, 3, 7, 1, 2, 3, 8, 2, 3]
    for use_oldest, expected in ((True, [7, 1, 2]), (False, [8, 2, 3])):
        state = _DeviceState([history[:-1]], max_seq_len, 1, draft_len)
        # Extend with the final token so the kernel sees the full history.
        state.load_step([0], [1], [[3, 0, 0, 0]], [1])
        state.launch(3, use_oldest, False)
        torch.cuda.synchronize()
        assert state.match_lens.tolist() == [2]
        assert state.drafts.tolist() == [expected]

    # A 3-token match beats every 2-token match regardless of position.
    history = [2, 3, 9, 1, 2, 3, 5, 5, 5, 1, 2, 3]
    state = _DeviceState([history[:-1]], max_seq_len, 1, draft_len)
    state.load_step([0], [1], [[3, 0, 0, 0]], [1])
    state.launch(3, True, False)
    torch.cuda.synchronize()
    assert state.match_lens.tolist() == [3]
    assert state.drafts.tolist() == [[5, 5, 5]]


def test_continuation_is_zero_padded_at_history_end():
    max_seq_len, draft_len = 32, 4
    history = [4, 5, 6, 4, 5]  # suffix [4, 5] matched at the start, only one token follows
    state = _DeviceState([history[:-1]], max_seq_len, 1, draft_len)
    state.load_step([0], [1], [[5, 0, 0, 0, 0]], [1])
    state.launch(2, True, False)
    torch.cuda.synchronize()
    assert state.drafts.tolist() == [[6, 4, 5, 0]]
    assert state.match_lens.tolist() == [2]


def test_public_pool_prefers_own_history_then_other_slots():
    max_seq_len, draft_len = 32, 2
    own = [1, 2, 3, 9, 1, 2]  # own history has the suffix [1, 2] followed by 3
    other = [1, 2, 8, 8]
    state = _DeviceState([own[:-1], other], max_seq_len, 1, draft_len)
    state.load_step([0], [1], [[2, 0, 0]], [1])
    state.launch(2, True, True)
    torch.cuda.synchronize()
    assert state.drafts.tolist() == [[3, 9]]

    # Without a match in its own history, the row drafts from the other slot.
    own = [5, 5, 1, 2]
    state = _DeviceState([own[:-1], other], max_seq_len, 1, draft_len)
    state.load_step([0], [1], [[2, 0, 0]], [1])
    state.launch(2, True, True)
    torch.cuda.synchronize()
    assert state.drafts.tolist() == [[8, 8]]
    assert state.match_lens.tolist() == [2]

    # Private pool: the same row finds nothing.
    state = _DeviceState([own[:-1], other], max_seq_len, 1, draft_len)
    state.load_step([0], [1], [[2, 0, 0]], [1])
    state.launch(2, True, False)
    torch.cuda.synchronize()
    assert state.drafts.tolist() == [[0, 0]]
    assert state.match_lens.tolist() == [0]


def test_extend_only_with_zero_draft_len():
    max_seq_len = 16
    state = _DeviceState([[1, 2], [3]], max_seq_len, 2, 3)
    state.load_step([0, 1], [1, 0], [[7, 8, 9, 0], [4, 4, 4, 4]], [2, 4])
    empty_drafts = torch.zeros((2, 0), dtype=torch.int32, device="cuda")
    torch.ops.trtllm.ngram_extend_and_draft_op(
        state.history_tokens,
        state.history_lens,
        state.slot_ids,
        state.row_mask,
        state.accepted,
        state.num_accepted,
        empty_drafts,
        state.match_lens,
        2,
        True,
        False,
    )
    torch.cuda.synchronize()
    assert state.histories() == [[1, 2, 7, 8], [3]]


def test_history_never_overflows_max_seq_len():
    max_seq_len = 6
    state = _DeviceState([[1, 2, 3, 4]], max_seq_len, 1, 3)
    state.load_step([0], [1], [[5, 6, 7, 8]], [4])
    state.launch(2, True, False)
    torch.cuda.synchronize()
    assert state.histories() == [[1, 2, 3, 4, 5, 6]]


def test_cuda_graph_replay_matches_reference():
    """The op only reads/writes static buffers, so a captured graph replays correctly."""
    rng = random.Random(7)
    num_slots, max_seq_len, draft_len, max_ngram = 4, 64, 3, 2
    histories = _random_histories(rng, num_slots, 20)
    slot_ids = [0, 2, 3]
    row_mask = [1, 1, 1]
    batch = len(slot_ids)
    state = _DeviceState(histories, max_seq_len, batch, draft_len)
    state.load_step(slot_ids, row_mask, *_random_step(rng, batch, draft_len))

    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        # Warm-up launch on a throwaway copy of the pool so capture sees a hot op.
        warm = _DeviceState(histories, max_seq_len, batch, draft_len)
        warm.load_step(slot_ids, row_mask, *_random_step(rng, batch, draft_len))
        warm.launch(max_ngram, True, True)
        with torch.cuda.graph(graph, stream=stream):
            state.launch(max_ngram, True, True)
    torch.cuda.synchronize()

    for _ in range(5):
        accepted, num_accepted = _random_step(rng, batch, draft_len)
        ref_drafts, ref_match_lens = reference_extend_and_draft(
            histories,
            slot_ids,
            row_mask,
            accepted,
            num_accepted,
            draft_len,
            max_ngram,
            True,
            True,
            max_seq_len,
        )
        state.load_step(slot_ids, row_mask, accepted, num_accepted)
        graph.replay()
        _assert_step_matches(state, histories, ref_drafts, ref_match_lens)
