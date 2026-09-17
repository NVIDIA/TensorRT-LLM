# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for TopK dispatch during DSA metadata initialization."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa import metadata as dsa_metadata
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAMetadataParams


@pytest.mark.parametrize("enable_heuristic", [False, True])
@pytest.mark.parametrize("use_self_sampling", [False, True])
@pytest.mark.parametrize("dsl_available", [False, True])
@pytest.mark.parametrize("sm_version", [90, 100, 103, 107])
def test_topk_flags_initialized_before_buffer_allocation(
    enable_heuristic: bool,
    use_self_sampling: bool,
    dsl_available: bool,
    sm_version: int,
) -> None:
    metadata = object.__new__(dsa_metadata.DSAtrtllmAttentionMetadata)
    metadata.sparse_metadata_params = DSAMetadataParams(
        indexer_max_chunk_size=8192,
        max_sparse_topk=512,
        index_head_dim=128,
        enable_indexer_skip=False,
        enable_heuristic_topk=enable_heuristic,
        use_cute_dsl_topk=True,
        use_cute_dsl_paged_mqa_logits=False,
        q_split_threshold=8192,
        use_self_sampling_topk=use_self_sampling,
    )
    metadata.kv_cache_manager = SimpleNamespace(
        tokens_per_block=128,
        compressed_block_sizes={},
        get_cache_indices=Mock(),
    )
    metadata.is_cuda_graph = False
    enabled = enable_heuristic and sm_version >= 100
    self_sampling_supported = enabled and dsl_available and sm_version in (100, 103)
    temporal_supported = enabled and dsl_available and sm_version in (100, 103)

    def check_flags(*, capture_graph: bool) -> None:
        assert capture_graph is False
        assert metadata.enable_gvr_topk is enabled
        assert metadata.use_self_sampling_topk is (self_sampling_supported and use_self_sampling)
        assert metadata.needs_gvr_prior is (temporal_supported and not use_self_sampling)

    metadata.create_buffers_for_mla_rope_append = Mock(side_effect=check_flags)
    metadata.create_buffers_for_indexer = Mock(side_effect=check_flags)

    with (
        patch.object(dsa_metadata.TrtllmAttentionMetadata, "__post_init__"),
        patch.object(dsa_metadata, "IS_CUTLASS_DSL_AVAILABLE", dsl_available),
        patch.object(dsa_metadata, "get_sm_version", return_value=sm_version),
    ):
        metadata.__post_init__()

    metadata.create_buffers_for_mla_rope_append.assert_called_once_with(capture_graph=False)
    metadata.create_buffers_for_indexer.assert_called_once_with(capture_graph=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("num_local_layers", [1, 3])
def test_temporal_gvr_allocates_real_prior_buffers(num_local_layers: int) -> None:
    """Allocate actual CUDA buffers and size the zeroed prior by local layers."""
    metadata = object.__new__(dsa_metadata.DSAtrtllmAttentionMetadata)
    metadata.sparse_metadata_params = DSAMetadataParams(
        indexer_max_chunk_size=32,
        max_sparse_topk=512,
        index_head_dim=128,
        enable_indexer_skip=False,
        enable_heuristic_topk=True,
        use_cute_dsl_topk=True,
        use_cute_dsl_paged_mqa_logits=False,
        q_split_threshold=32,
        use_self_sampling_topk=False,
    )
    metadata.kv_cache_manager = SimpleNamespace(
        tokens_per_block=128,
        compressed_block_sizes={},
        get_cache_indices=Mock(),
        max_blocks_per_seq=2,
        num_local_layers=num_local_layers,
    )
    metadata.draft_kv_cache_manager = None
    metadata.is_cuda_graph = False
    metadata.cuda_graph_buffers = None
    metadata.max_num_sequences = 4
    metadata.max_num_tokens = 16
    metadata.max_draft_tokens = 3
    metadata.num_sms = 16
    metadata.enable_context_mla_with_cached_kv = False
    # object.__new__ skips __init__, which is where this default is set.
    metadata._radix_rows_per_sequence = 1
    with (
        patch.object(dsa_metadata.TrtllmAttentionMetadata, "__post_init__"),
        patch.object(metadata, "create_buffers_for_mla_rope_append"),
        patch.object(dsa_metadata, "IS_CUTLASS_DSL_AVAILABLE", True),
        patch.object(dsa_metadata, "get_sm_version", return_value=100),
    ):
        metadata.__post_init__()

    assert metadata.enable_gvr_topk
    assert metadata.needs_gvr_prior
    assert metadata.gvr_prior_indices.shape == (num_local_layers, 4, 512)
    assert metadata.gvr_prior_indices.dtype == torch.int32
    assert metadata.gvr_prior_indices.is_cuda
    assert torch.count_nonzero(metadata.gvr_prior_indices).item() == 0
    assert metadata.kv_lens_row_reorder_buffer.shape == (4,)
