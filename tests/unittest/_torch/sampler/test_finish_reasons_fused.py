# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused end-ID and maximum-length checks must match the tensor path."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.finish_reasons import FinishReasonsHandler
from tensorrt_llm.bindings.executor import FinishReason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("max_tokens,max_beam_width", [(1, 1), (3, 1), (1, 2), (2, 3)])
def test_fused_matches_tensor_ops(max_tokens: int, max_beam_width: int) -> None:
    torch.manual_seed(0)
    max_num_sequences, end_id = 5, 99
    handler = FinishReasonsHandler(
        max_stop_word_length=1,
        max_num_stop_words=1,
        max_num_sequences=max_num_sequences,
        max_beam_width=max_beam_width,
        max_tokens=max_tokens,
        max_seq_len=64,
    )
    store = handler.store
    store.max_lengths_cuda.fill_(14)
    store.end_ids_cuda.fill_(end_id)

    # The sampler forces int64 slots; the tensor path's index_fill_ rejects int32.
    seq_slots = torch.tensor([3, 0, 4], dtype=torch.int64, device="cuda")
    seq_lens = torch.tensor([7, 13, 15], dtype=torch.int32, device="cuda")
    new_tokens = torch.randint(
        0,
        50,
        (max_tokens, max_num_sequences + 1, max_beam_width),
        dtype=torch.int32,
        device="cuda",
    )
    new_tokens[0, 4, 0] = end_id  # one slot finishes on the end ID, one on max length

    def run() -> torch.Tensor:
        # Seeding with END_ID also shows that untouched slots keep their value.
        store.finish_reasons_cuda.fill_(FinishReason.END_ID.value)
        handler._write_finish_reasons(seq_slots=seq_slots, seq_lens=seq_lens, new_tokens=new_tokens)
        return store.finish_reasons_cuda.clone()

    # Without this the comparison below would pass with both arms on the fallback.
    assert handler._can_fuse_finish_reasons(
        seq_slots=seq_slots,
        seq_lens=seq_lens,
        new_tokens=new_tokens,
        stop_word_indices=None,
        first_finish_reasons=None,
    )
    fused = run()

    handler._can_fuse_finish_reasons = lambda **_: False
    torch.testing.assert_close(fused, run(), rtol=0, atol=0)
    assert fused[0, 4, 0].item() == FinishReason.END_ID.value
    assert fused[0, 0, 0].item() == FinishReason.LENGTH.value
