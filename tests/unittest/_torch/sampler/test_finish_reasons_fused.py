# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence tests for the fused end-ID and maximum-length checks."""

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
def test_fused_matches_tensor_ops(max_tokens, max_beam_width):
    torch.manual_seed(0)
    max_num_sequences = 5
    end_id = 99
    # The sampler forces int64 slots; the tensor path's index_fill_ rejects int32.
    seq_slots = torch.tensor([3, 0, 4], dtype=torch.int64, device="cuda")
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
    new_tokens[0, 4, 0] = end_id

    store.finish_reasons_cuda.fill_(FinishReason.END_ID.value)
    fused = _write(handler, seq_slots=seq_slots, seq_lens=seq_lens, new_tokens=new_tokens)
    assert handler._can_fuse_finish_reasons(
        seq_slots=seq_slots,
        seq_lens=seq_lens,
        new_tokens=new_tokens,
        stop_word_indices=None,
        first_finish_reasons=None,
    )

    handler._can_fuse_finish_reasons = lambda **_: False
    store.finish_reasons_cuda.fill_(FinishReason.END_ID.value)
    # ``index_fill_`` in the tensor fallback requires int64 indices. Compare
    # int32 kernel inputs against the same logical slots through that contract.
    reference = _write(
        handler,
        seq_slots=seq_slots.to(torch.int64),
        seq_lens=seq_lens,
        new_tokens=new_tokens,
    )

    torch.testing.assert_close(fused, reference, rtol=0, atol=0)
    assert reference[0, 4, 0].item() == FinishReason.END_ID.value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_untouched_slots_are_preserved():
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
    assert [reasons[0, slot, 0].item() for slot in (0, 1, 3)] == [FinishReason.STOP_WORDS.value] * 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stop_words_and_first_reason_latch_keep_tensor_path():
    handler = _build_handler(max_num_sequences=2, max_beam_width=1, max_tokens=1)
    seq_slots = torch.tensor([0], dtype=torch.int64, device="cuda")
    seq_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
    new_tokens = torch.zeros((1, 2, 1), dtype=torch.int32, device="cuda")

    assert not handler._can_fuse_finish_reasons(
        seq_slots=seq_slots,
        seq_lens=seq_lens,
        new_tokens=new_tokens,
        stop_word_indices=torch.zeros(1, dtype=torch.int64, device="cuda"),
        first_finish_reasons=None,
    )
    assert not handler._can_fuse_finish_reasons(
        seq_slots=seq_slots,
        seq_lens=seq_lens,
        new_tokens=new_tokens,
        stop_word_indices=None,
        first_finish_reasons=torch.zeros((2, 1), dtype=torch.int32, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_oversized_token_beam_tile_keeps_tensor_path():
    handler = _build_handler(max_num_sequences=1, max_beam_width=33, max_tokens=32)
    seq_slots = torch.zeros(1, dtype=torch.int64, device="cuda")
    seq_lens = torch.zeros(1, dtype=torch.int32, device="cuda")
    new_tokens = torch.zeros((32, 1, 33), dtype=torch.int32, device="cuda")

    assert not handler._can_fuse_finish_reasons(
        seq_slots=seq_slots,
        seq_lens=seq_lens,
        new_tokens=new_tokens,
        stop_word_indices=None,
        first_finish_reasons=None,
    )
