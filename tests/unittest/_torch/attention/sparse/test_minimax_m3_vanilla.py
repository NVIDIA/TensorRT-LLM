# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
    MiniMaxM3SparseRuntimeBackend,
    MiniMaxM3VanillaAttention,
)
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import MiniMaxM3SparseParams
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.triton_metadata import (
    MiniMaxM3TritonSparseAttentionMetadata,
)
from tensorrt_llm._torch.attention_backend.sparse.registry import (
    get_vanilla_sparse_attn_attention_backend,
)


def _params(*, local_blocks: int = 0) -> MiniMaxM3SparseParams:
    return MiniMaxM3SparseParams(
        num_index_heads=2,
        sparse_index_dim=2,
        block_size=2,
        topk=1,
        init_blocks=0,
        local_blocks=local_blocks,
    )


def _prefill_metadata(length: int, device: torch.device) -> MiniMaxM3TritonSparseAttentionMetadata:
    metadata = MiniMaxM3TritonSparseAttentionMetadata(
        is_prefill=True,
        req_to_token=torch.arange(length, dtype=torch.int32, device=device).unsqueeze(0),
        slot_ids=torch.tensor([0], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([length], dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor([length], dtype=torch.int32),
        prefix_lens=torch.tensor([0], dtype=torch.int32, device=device),
        cu_seqlens_q=torch.tensor([0, length], dtype=torch.int32, device=device),
        extend_seq_lens_cpu=[length],
    )
    metadata.prepare()
    return metadata


def _decode_metadata(length: int, device: torch.device) -> MiniMaxM3TritonSparseAttentionMetadata:
    metadata = MiniMaxM3TritonSparseAttentionMetadata(
        is_prefill=False,
        req_to_token=torch.arange(length, dtype=torch.int32, device=device).unsqueeze(0),
        slot_ids=torch.tensor([0], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([length], dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor([length], dtype=torch.int32),
    )
    metadata.prepare()
    return metadata


def test_minimax_m3_vanilla_registry() -> None:
    assert get_vanilla_sparse_attn_attention_backend(_params()) is MiniMaxM3VanillaAttention


def test_minimax_m3_vanilla_empty_prefill() -> None:
    attention = MiniMaxM3VanillaAttention(
        layer_idx=3,
        num_heads=2,
        head_dim=2,
        num_kv_heads=1,
        sparse_params=_params(),
    )
    output = torch.empty(0, 4)

    result = attention.forward(
        torch.empty(0, 4),
        torch.empty(0, 2),
        torch.empty(0, 2),
        None,
        output=output,
        idx_q=torch.empty(0, 4),
        idx_k=torch.empty(0, 2),
        k_cache=torch.empty(1, 1, 2),
        v_cache=torch.empty(1, 1, 2),
        idx_k_cache=torch.empty(1, 1, 2),
        out_cache_loc=torch.empty(0, dtype=torch.int32),
        m3_metadata=_prefill_metadata(0, torch.device("cpu")),
    )

    assert result.data_ptr() == output.data_ptr()
    assert result.shape == (0, 4)


def test_minimax_m3_vanilla_prefill_selects_indexed_block() -> None:
    device = torch.device("cpu")
    attention = MiniMaxM3VanillaAttention(
        layer_idx=3,
        num_heads=2,
        head_dim=2,
        num_kv_heads=1,
        sparse_params=_params(),
    )
    q = torch.zeros(4, 4)
    k = torch.zeros(4, 2)
    v = torch.tensor([[1.0, 0.0], [3.0, 0.0], [100.0, 0.0], [100.0, 0.0]])
    idx_q = torch.tensor([[1.0, 0.0, 1.0, 0.0]]).expand(4, -1).clone()
    idx_k = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    k_cache = torch.zeros(4, 1, 2)
    v_cache = torch.zeros_like(k_cache)
    idx_k_cache = torch.zeros(4, 1, 2)
    output = torch.empty(4, 4)

    result = attention.forward(
        q,
        k,
        v,
        None,
        output=output,
        idx_q=idx_q,
        idx_k=idx_k,
        k_cache=k_cache,
        v_cache=v_cache,
        idx_k_cache=idx_k_cache,
        out_cache_loc=torch.arange(4, dtype=torch.int32),
        m3_metadata=_prefill_metadata(4, device),
    )

    expected = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
        ]
    )
    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(v_cache[:, 0], v)
    torch.testing.assert_close(idx_k_cache[:, 0], idx_k)


def test_minimax_m3_vanilla_decode_prioritizes_local_block() -> None:
    device = torch.device("cpu")
    attention = MiniMaxM3VanillaAttention(
        layer_idx=3,
        num_heads=2,
        head_dim=2,
        num_kv_heads=1,
        sparse_params=_params(local_blocks=1),
    )
    k_cache = torch.zeros(5, 1, 2)
    v_cache = torch.zeros_like(k_cache)
    idx_k_cache = torch.zeros(5, 1, 2)
    output = torch.empty(1, 4)

    result = attention.forward(
        torch.zeros(1, 4),
        torch.zeros(1, 2),
        torch.tensor([[7.0, 9.0]]),
        None,
        output=output,
        idx_q=torch.zeros(1, 4),
        idx_k=torch.zeros(1, 2),
        k_cache=k_cache,
        v_cache=v_cache,
        idx_k_cache=idx_k_cache,
        out_cache_loc=torch.tensor([4], dtype=torch.int32),
        m3_metadata=_decode_metadata(5, device),
    )

    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(result, torch.tensor([[7.0, 9.0, 7.0, 9.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_minimax_m3_triton_uses_vanilla_golden() -> None:
    torch.manual_seed(43)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    params = MiniMaxM3SparseParams(
        num_index_heads=4,
        sparse_index_dim=16,
        block_size=4,
        topk=2,
        init_blocks=0,
        local_blocks=1,
    )
    kwargs = dict(
        layer_idx=3,
        num_heads=4,
        head_dim=16,
        num_kv_heads=2,
        sparse_params=params,
    )
    vanilla = MiniMaxM3VanillaAttention(**kwargs)
    triton = MiniMaxM3SparseRuntimeBackend(**kwargs)
    length = 12
    q = torch.randn(length, 4 * 16, dtype=dtype, device=device)
    k = torch.randn(length, 2 * 16, dtype=dtype, device=device)
    v = torch.randn_like(k)
    idx_q = torch.randn(length, 4 * 16, dtype=dtype, device=device)
    idx_k = torch.randn(length, 16, dtype=dtype, device=device)
    out_cache_loc = torch.arange(length, dtype=torch.int32, device=device)

    def run(attention):
        return attention.forward(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            idx_q=idx_q.clone(),
            idx_k=idx_k.clone(),
            k_cache=torch.zeros(length, 2, 16, dtype=dtype, device=device),
            v_cache=torch.zeros(length, 2, 16, dtype=dtype, device=device),
            idx_k_cache=torch.zeros(length, 1, 16, dtype=dtype, device=device),
            out_cache_loc=out_cache_loc,
            m3_metadata=_prefill_metadata(length, device),
        )

    torch.testing.assert_close(run(triton), run(vanilla), atol=2e-2, rtol=2e-2)
