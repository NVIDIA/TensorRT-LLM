# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker


def test_external_shared_kv_uses_backend_metadata_contract() -> None:
    """The worker forwards the backend-provided draft metadata view."""
    num_accepted_tokens = torch.tensor([1, 2], dtype=torch.int32)
    draft_metadata = SimpleNamespace(
        use_spec_decoding=True,
        padded_num_tokens=4,
        all_rank_num_tokens=None,
    )

    class FakeAttentionMetadata:
        def __init__(self) -> None:
            self.seq_lens_cuda = torch.tensor([1, 1], dtype=torch.int32)
            self.calls = []

        def get_shared_kv_draft_metadata(self, accepted, num_contexts):
            self.calls.append((accepted, num_contexts))
            return draft_metadata

    class FakeDraftModel:
        def __init__(self) -> None:
            self.received_metadata = []

        def forward_draft_step(self, **kwargs):
            self.received_metadata.append(kwargs["attn_metadata"])
            return torch.zeros((2, 1)), kwargs["recurrent_hidden_states"]

    attn_metadata = FakeAttentionMetadata()
    draft_model = FakeDraftModel()
    all_rank_num_tokens = torch.tensor([2], dtype=torch.int32)
    spec_metadata = SimpleNamespace(
        runtime_draft_len=1,
        batch_indices_cuda=torch.arange(2),
        subseq_all_rank_num_tokens=all_rank_num_tokens,
    )
    worker = SimpleNamespace(
        guided_decoder=None,
        sa_enhancer=None,
        _prepare_shared_kv_draft_inputs=lambda **kwargs: (
            torch.tensor([10, 20]),
            torch.zeros((2, 4)),
            torch.tensor([[1, 2]]),
        ),
        sample_draft_tokens=lambda *args, **kwargs: torch.tensor([11, 21]),
    )

    result = Eagle3OneModelWorker._forward_external_shared_target_kv_draft_loop(
        worker,
        position_ids=torch.tensor([[0, 1]]),
        hidden_states=torch.zeros((2, 4)),
        attn_metadata=attn_metadata,
        spec_metadata=spec_metadata,
        draft_model=draft_model,
        accepted_tokens=torch.tensor([[10], [20]]),
        num_accepted_tokens=num_accepted_tokens,
        num_contexts=1,
        batch_size=2,
    )

    assert len(attn_metadata.calls) == 1
    assert attn_metadata.calls[0][0] is num_accepted_tokens
    assert attn_metadata.calls[0][1] == 1
    assert draft_model.received_metadata == [draft_metadata]
    assert draft_metadata.use_spec_decoding is False
    assert draft_metadata.padded_num_tokens is None
    assert draft_metadata.all_rank_num_tokens is all_rank_num_tokens
    torch.testing.assert_close(result, torch.tensor([[11], [21]]))
