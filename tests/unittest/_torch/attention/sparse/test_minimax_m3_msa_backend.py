# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the MiniMax-M3 MSA sparse attention backend.

These validate backend selection, decode scratch-buffer sizing, and the paged
HND view contract passed to the packaged MSA kernel. Numerical parity against
the Triton reference is covered by the SM100 integration accuracy test.
"""

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import msa_paged_kv
from tensorrt_llm._torch.attention_backend.sparse.utils import _resolve_minimax_m3_backend_cls
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig


def test_resolver_selects_msa_backend_when_available(monkeypatch):
    import tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_availability as avail

    monkeypatch.setattr(avail, "ensure_msa_available", lambda: None)
    params = MiniMaxM3SparseAttentionConfig(implementation="msa").to_sparse_params()
    assert _resolve_minimax_m3_backend_cls(params) is MiniMaxM3MsaSparseAttention


def test_msa_requires_block_size_128():
    # The MSA implementation is fixed to a 128-token page size; a mismatched
    # sparse_block_size must fail loudly at config construction rather than being
    # silently overridden at runtime.
    with pytest.raises(ValueError, match=r"sparse_block_size == 128"):
        MiniMaxM3SparseAttentionConfig(implementation="msa", sparse_block_size=64)

    # The Triton reference is unaffected by the constraint.
    cfg = MiniMaxM3SparseAttentionConfig(implementation="triton", sparse_block_size=64)
    assert cfg.sparse_block_size == 64


def test_msa_metadata_rejects_undersized_max_score_buffer():
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    # Flat backing store sized for 4 heads * 8 k-tiles * 2 batch = 64 elements,
    # too small for the plan's required 4 * 16 * 2 = 128.
    metadata.msa_max_score = torch.zeros(4 * 8 * 2)
    metadata.kv_cache_manager = None

    with pytest.raises(ValueError, match=r"msa_max_score backing store"):
        metadata._ensure_msa_decode_scratch_buffers(
            num_index_heads=4,
            max_batch=2,
            capture_graph=False,
            required_max_k_tiles=16,
        )


def test_msa_proxy_max_score_view_is_contiguous_over_stable_store():
    """The proxy view fed to fmha_sm100 must be contiguous in the exact
    [num_index_heads, plan_max_k_tiles, num_tokens] shape the kernel writes,
    backed by a stable store so its data_ptr survives CUDA graph replay.
    """
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    # Worst-case store: 4 heads * 16 k-tiles * 8 batch.
    num_index_heads, worst_k, max_batch = 4, 16, 8
    metadata.msa_max_score = torch.zeros(num_index_heads * worst_k * max_batch)
    store_ptr = metadata.msa_max_score.data_ptr()

    # A smaller live step still yields a contiguous view sized to that step,
    # which is what the kernel's stride-agnostic write requires.
    view = metadata.msa_proxy_max_score_view(num_index_heads, 5, 3)
    assert view.shape == (num_index_heads, 5, 3)
    assert view.is_contiguous()
    assert view.data_ptr() == store_ptr

    # Oversized requests are rejected rather than silently corrupting memory.
    with pytest.raises(ValueError, match=r"msa_max_score backing store"):
        metadata.msa_proxy_max_score_view(num_index_heads, worst_k, max_batch + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_msa_paged_kv_preserves_tma_compatible_outer_stride() -> None:
    from fmha_sm100.cute import interface as sparse_interface

    pages, roles, heads, page_size, head_dim = 5, 2, 2, 128, 128
    pool = torch.empty(
        pages,
        roles,
        heads,
        page_size,
        head_dim,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    kv_cache_manager = Mock()
    kv_cache_manager.get_buffers.return_value = pool

    k_view, v_view = msa_paged_kv(kv_cache_manager, layer_idx=3)

    kv_cache_manager.get_buffers.assert_called_once_with(3, kv_layout="HND")
    for view in (k_view, v_view):
        assert not view.is_contiguous()
        prepared = sparse_interface._prepare_paged_hnd_input(view, page_size)
        assert prepared.data_ptr() == view.data_ptr()
        assert prepared.stride() == view.stride()

    mismatched = sparse_interface._prepare_paged_hnd_input(k_view, page_size // 2)
    assert mismatched.data_ptr() == k_view.data_ptr()
    with pytest.raises(ValueError, match="page_size == blk_kv"):
        sparse_interface._prepare_paged_kv_for_tma(mismatched, mismatched, page_size // 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_msa_paged_hnd_input_materializes_unaligned_outer_stride() -> None:
    from fmha_sm100.cute import interface as sparse_interface

    pages, heads, page_size, head_dim = 5, 2, 128, 128
    outer_stride = heads * page_size * head_dim + 1
    storage = torch.empty(
        pages * outer_stride,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    view = storage.as_strided(
        (pages, heads, page_size, head_dim),
        (outer_stride, page_size * head_dim, head_dim, 1),
    )

    prepared = sparse_interface._prepare_paged_hnd_input(view, page_size)

    assert prepared.is_contiguous()
    assert prepared.data_ptr() != view.data_ptr()
