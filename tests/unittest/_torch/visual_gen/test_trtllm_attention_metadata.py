# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.visual_gen.attention_backend import trtllm as visual_trtllm


class _FakeBaseTrtllmAttentionMetadata:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prepare_calls = 0
        self.seq_lens = None
        self.num_contexts = None
        self.max_seq_len = None
        self.request_ids = None

    def prepare(self):
        self.prepare_calls += 1


def test_trtllm_attention_metadata_caches_distinct_seq_lens(monkeypatch):
    monkeypatch.setattr(
        visual_trtllm,
        "BaseTrtllmAttentionMetadata",
        _FakeBaseTrtllmAttentionMetadata,
    )
    attention_metadata_state = {}
    metadata = visual_trtllm.TrtllmAttentionMetadata(
        device=torch.device("cpu"),
        attention_metadata_state=attention_metadata_state,
    )

    first_seq_lens = torch.tensor([64], dtype=torch.int32)
    first_metadata = metadata.prepare(batch_size=1, seq_lens=first_seq_lens)
    first_seq_lens.fill_(999)

    second_metadata = metadata.prepare(batch_size=1, seq_lens=torch.tensor([96], dtype=torch.int32))
    first_metadata_again = metadata.prepare(
        batch_size=1,
        seq_lens=torch.tensor([64], dtype=torch.int32),
    )

    assert first_metadata is first_metadata_again
    assert first_metadata is not second_metadata
    assert first_metadata.prepare_calls == 1
    assert second_metadata.prepare_calls == 1

    metadata_cache = attention_metadata_state["metadata_cache"]
    assert set(metadata_cache) == {
        (1, (64,)),
        (1, (96,)),
    }
    assert metadata_cache[(1, (64,))]["metadata"] is first_metadata
    assert metadata_cache[(1, (96,))]["metadata"] is second_metadata

    first_cached_seq_lens = metadata_cache[(1, (64,))]["seq_lens"]
    second_cached_seq_lens = metadata_cache[(1, (96,))]["seq_lens"]
    assert torch.equal(first_cached_seq_lens, torch.tensor([64], dtype=torch.int32))
    assert torch.equal(second_cached_seq_lens, torch.tensor([96], dtype=torch.int32))
    assert first_cached_seq_lens is not second_cached_seq_lens
    assert first_cached_seq_lens.data_ptr() != second_cached_seq_lens.data_ptr()
    assert first_metadata.seq_lens is first_cached_seq_lens
    assert second_metadata.seq_lens is second_cached_seq_lens
