# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence of the fused stop-criteria kernel and the tensor-op path."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.finish_reasons import FinishReasonsHandler
from tensorrt_llm.bindings.executor import FinishReason


def _build_handler(*, max_num_sequences: int, max_beam_width: int, max_tokens: int):
    return FinishReasonsHandler(
        max_stop_word_length=1,
        max_num_stop_words=1,
        max_num_sequences=max_num_sequences,
        max_beam_width=max_beam_width,
        max_tokens=max_tokens,
        max_seq_len=64,
    )


def _write(handler, *, seq_slots, seq_lens, new_tokens):
    handler._write_finish_reasons(
        seq_slots=seq_slots,
        seq_lens=seq_lens,
        new_tokens=new_tokens,
    )
    return handler.store.finish_reasons_cuda.clone()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("max_tokens,max_beam_width", [(1, 1), (3, 1), (1, 2), (2, 3)])
def test_fused_matches_tensor_ops(max_tokens: int, max_beam_width: int):
    """The fused kernel reproduces the tensor-op path bit-for-bit."""
    torch.manual_seed(0)
    max_num_sequences = 5
    end_id = 99
    # Slots deliberately out of order and not covering the whole store, so a
    # kernel that ignored seq_slots or wrote neighbouring rows would fail.
    seq_slots = torch.tensor([3, 0, 4], dtype=torch.int64, device="cuda")
    # 7 leaves room below max_length, 12 crosses it inside the token block and
    # 15 is already past it.
    seq_lens = torch.tensor([7, 12, 15], dtype=torch.int32, device="cuda")

    handler = _build_handler(
        max_num_sequences=max_num_sequences,
        max_beam_width=max_beam_width,
        max_tokens=max_tokens,
    )
    store = handler.store
    store.max_lengths_cuda.fill_(14)
    store.end_ids_cuda.fill_(end_id)
    new_tokens = torch.randint(
        0,
        50,
        (max_tokens, max_num_sequences, max_beam_width),
        dtype=torch.int32,
        device="cuda",
    )
    # Plant an end id so the end-id criterion fires and outranks max-length.
    new_tokens[0, 4, 0] = end_id

    store.finish_reasons_cuda.fill_(FinishReason.END_ID.value)
    fused = _write(handler, seq_slots=seq_slots, seq_lens=seq_lens, new_tokens=new_tokens)
    assert handler._can_fuse_finish_reasons(
        seq_slots=seq_slots,
        new_tokens=new_tokens,
        stop_word_indices=None,
        first_finish_reasons=None,
    )

    handler._can_fuse_finish_reasons = lambda **_: False
    store.finish_reasons_cuda.fill_(FinishReason.END_ID.value)
    reference = _write(handler, seq_slots=seq_slots, seq_lens=seq_lens, new_tokens=new_tokens)

    torch.testing.assert_close(fused, reference, rtol=0, atol=0)
    # The planted end id must actually have exercised the END_ID branch,
    # otherwise the comparison above is vacuous for that criterion.
    assert reference[0, 4, 0].item() == FinishReason.END_ID.value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_untouched_slots_are_preserved():
    """Slots outside the batch keep whatever the previous step left there."""
    handler = _build_handler(max_num_sequences=4, max_beam_width=1, max_tokens=1)
    store = handler.store
    store.max_lengths_cuda.fill_(100)
    store.end_ids_cuda.fill_(-1)
    store.finish_reasons_cuda.fill_(FinishReason.STOP_WORDS.value)

    seq_slots = torch.tensor([2], dtype=torch.int64, device="cuda")
    seq_lens = torch.tensor([5], dtype=torch.int32, device="cuda")
    new_tokens = torch.zeros((1, 4, 1), dtype=torch.int32, device="cuda")

    reasons = _write(handler, seq_slots=seq_slots, seq_lens=seq_lens, new_tokens=new_tokens)

    assert reasons[0, 2, 0].item() == FinishReason.NOT_FINISHED.value
    untouched = [reasons[0, slot, 0].item() for slot in (0, 1, 3)]
    assert untouched == [FinishReason.STOP_WORDS.value] * 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stop_words_and_beam_search_keep_the_tensor_path():
    handler = _build_handler(max_num_sequences=2, max_beam_width=1, max_tokens=1)
    seq_slots = torch.tensor([0], dtype=torch.int64, device="cuda")
    new_tokens = torch.zeros((1, 2, 1), dtype=torch.int32, device="cuda")

    assert not handler._can_fuse_finish_reasons(
        seq_slots=seq_slots,
        new_tokens=new_tokens,
        stop_word_indices=torch.zeros(1, dtype=torch.int64, device="cuda"),
        first_finish_reasons=None,
    )
    assert not handler._can_fuse_finish_reasons(
        seq_slots=seq_slots,
        new_tokens=new_tokens,
        stop_word_indices=None,
        first_finish_reasons=torch.zeros((2, 1), dtype=torch.int32, device="cuda"),
    )
