# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Kimi K3-specific Mamba metadata."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_attention_metadata(context_lengths: list[int]) -> SimpleNamespace:
    seq_lens = torch.tensor(context_lengths or [1], dtype=torch.int)
    num_contexts = len(context_lengths)
    return SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=num_contexts,
        num_ctx_tokens=sum(context_lengths),
        kv_cache_manager=None,
        request_ids=None,
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.zeros(num_contexts, dtype=torch.int),
        ),
    )


def test_k3_prepare_metadata_match_chunk_indices() -> None:
    pytest.importorskip("fla")
    from fla.ops.utils.index import prepare_chunk_indices

    metadata = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128)
    cases = (
        ([128, 256], True, None),
        ([1, 64, 65, 129], False, None),
        ([300], False, 300),
    )
    for context_lengths, expected_alignment, expected_single_length in cases:
        metadata.prepare(_make_attention_metadata(context_lengths))

        assert metadata.kda_varlen_is_aligned is expected_alignment
        assert metadata.kda_single_sequence_length == expected_single_length
        cu_seqlens = metadata.query_start_loc_long[: len(context_lengths) + 1]
        legacy_chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size=64)
        torch.testing.assert_close(metadata.kda_chunk_indices, legacy_chunk_indices)


def test_prepare_refreshes_and_clears_kda_prefill_metadata() -> None:
    metadata = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128)

    metadata.prepare(_make_attention_metadata([1, 64, 65, 129]))
    metadata.prepare(_make_attention_metadata([300]))

    expected = torch.stack(
        (
            torch.zeros(5, dtype=torch.long, device="cuda"),
            torch.arange(5, dtype=torch.long, device="cuda"),
        ),
        dim=1,
    )
    assert metadata.kda_varlen_is_aligned is False
    assert metadata.kda_single_sequence_length == 300
    torch.testing.assert_close(metadata.kda_chunk_indices, expected)

    metadata.prepare(_make_attention_metadata([]))
    assert metadata.kda_chunk_indices is None
    assert metadata.kda_varlen_is_aligned is None
    assert metadata.kda_single_sequence_length is None
