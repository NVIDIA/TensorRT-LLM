# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Kimi K3-specific Mamba metadata."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices
from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_attention_metadata(
    context_lengths: list[int],
    generation_lengths: list[int] | None = None,
) -> SimpleNamespace:
    generation_lengths = generation_lengths or []
    seq_lens = torch.tensor(context_lengths + generation_lengths or [1], dtype=torch.int)
    num_contexts = len(context_lengths)
    return SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=num_contexts,
        num_ctx_tokens=sum(context_lengths),
        kv_cache_manager=None,
        request_ids=None,
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.zeros(len(seq_lens), dtype=torch.int),
        ),
    )


def test_k3_prepare_metadata_match_chunk_indices() -> None:
    metadata = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128, max_num_tokens=512)
    cases = (
        ([128, 256], [], True, None),
        ([1, 64, 65, 129], [], False, None),
        ([300], [], False, 300),
        ([65, 129], [1, 1], False, None),
    )
    for (
        context_lengths,
        generation_lengths,
        expected_alignment,
        expected_single_length,
    ) in cases:
        metadata.prepare(_make_attention_metadata(context_lengths, generation_lengths))

        assert metadata.kda_varlen_is_aligned is expected_alignment
        assert metadata.kda_single_sequence_length == expected_single_length
        cu_seqlens = metadata.query_start_loc_long[: len(context_lengths) + 1]
        legacy_chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size=64)
        torch.testing.assert_close(metadata.kda_chunk_indices, legacy_chunk_indices)


def test_k3_prepare_materializes_stable_aligned_generation_indices() -> None:
    class KdaCacheManager:
        use_kda_replay_update = True

        def __init__(self) -> None:
            self.state_indices = torch.tensor([9, 4, 7], dtype=torch.int32, device="cuda")

        def get_state_indices(self, request_ids: list[int], is_padding: list[bool]) -> torch.Tensor:
            return self.state_indices[: len(request_ids)]

    manager = KdaCacheManager()
    metadata = KimiK3MambaMetadata(max_batch_size=3, chunk_size=8, max_num_tokens=8)
    seq_lens = torch.tensor([2, 1, 1], dtype=torch.int)
    attn_metadata = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=1,
        num_ctx_tokens=2,
        kv_cache_manager=manager,
        request_ids=[10, 11, 12],
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.tensor([0], dtype=torch.int),
        ),
    )

    metadata.prepare(attn_metadata)

    assert metadata.state_indices[1:].data_ptr() % 16 != 0
    assert metadata.generation_state_indices is not None
    assert metadata.generation_state_indices.data_ptr() % 16 == 0
    aligned_data_ptr = metadata.generation_state_indices.data_ptr()
    torch.testing.assert_close(
        metadata.generation_state_indices,
        torch.tensor([4, 7], dtype=torch.int32, device="cuda"),
    )

    metadata.prepare(attn_metadata)

    assert metadata.generation_state_indices is not None
    assert metadata.generation_state_indices.data_ptr() == aligned_data_ptr

    attn_metadata.num_contexts = 0
    attn_metadata.num_ctx_tokens = 0
    metadata.prepare(attn_metadata)

    assert metadata.generation_state_indices is None
