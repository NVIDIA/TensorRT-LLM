# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.qsa import (
    QSAMambaHybridCacheManagerV2,
    QSASparseParams,
)
from tensorrt_llm._torch.attention_backend.sparse.qsa.indexer import (
    QSAIndexer,
    average_pool_qsa_keys,
    expand_qsa_block_indices,
    qsa_sparse_gqa,
    qsa_sparse_gqa_reference,
    select_qsa_paged_tokens,
    select_qsa_tokens,
)
from tensorrt_llm._torch.attention_backend.sparse.qsa.kernels import (
    triton_qsa_decode_pre_indexer,
    triton_qsa_prefill_compress,
)
from tensorrt_llm._torch.attention_backend.sparse.qsa.module import _query_chunk_size
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManagerV2, MambaRole
from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode


def _params(
    token_topk: int = 8,
    seq_len_threshold: int | None = None,
    index_head_dim: int = 4,
) -> QSASparseParams:
    return QSASparseParams(
        index_n_heads=4,
        index_kv_heads=1,
        index_head_dim=index_head_dim,
        token_topk=token_topk,
        compress_ratio=4,
        seq_len_threshold=seq_len_threshold,
    )


def test_qsa_query_chunk_respects_score_workspace(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_QSA_SPARSE_QUERY_CHUNK", raising=False)
    assert _query_chunk_size(8192, 2310) == 8192
    assert _query_chunk_size(65536, 16384) == 2048

    monkeypatch.setenv("TRTLLM_QSA_SPARSE_QUERY_CHUNK", "256")
    assert _query_chunk_size(8192, 2310) == 256
    monkeypatch.setenv("TRTLLM_QSA_SPARSE_QUERY_CHUNK", "invalid")
    assert _query_chunk_size(8192, 2310) == 8192


def test_qsa_sparse_params_validate_geometry() -> None:
    assert _params().block_topk == 2
    assert _params().expanded_topk == 11
    assert _params().dense_seq_len_threshold == 8
    assert _params(seq_len_threshold=32).dense_seq_len_threshold == 32
    with pytest.raises(ValueError, match="one index KV head"):
        QSASparseParams(
            index_n_heads=4,
            index_kv_heads=2,
            index_head_dim=4,
            token_topk=8,
            compress_ratio=4,
        )
    with pytest.raises(ValueError, match="seq_len_threshold must be positive"):
        _params(seq_len_threshold=0)


def test_average_pool_qsa_keys_uses_group_axis() -> None:
    keys = torch.arange(2 * 4 * 1 * 3, dtype=torch.bfloat16).reshape(2, 4, 1, 3)
    actual = average_pool_qsa_keys(keys)
    expected = keys.float().mean(dim=1).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected)


def test_expand_qsa_blocks_appends_incomplete_tail() -> None:
    blocks = torch.tensor([[1, 0], [0, -1]], dtype=torch.int32)
    actual = expand_qsa_block_indices(
        blocks,
        torch.tensor([9, 5]),
        torch.tensor([10, 6]),
        compress_ratio=4,
        token_topk=8,
    )
    assert actual.tolist() == [
        [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, -1],
        [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1],
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_prefill_compress_matches_gemma_norm_with_identity_rope() -> None:
    torch.manual_seed(42)
    tokens_per_block = 8
    num_tokens = 16
    head_dim = 16
    compress_ratio = 4
    eps = 1e-6
    index_cache = torch.randn(
        2,
        tokens_per_block,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    original = index_cache.clone()
    position_cache = torch.zeros(
        2,
        tokens_per_block,
        3,
        device="cuda",
        dtype=torch.int32,
    )
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    position_cache.view(num_tokens, 3).copy_(positions[:, None].expand(-1, 3))
    block_table = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    logical_positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    request_indices = torch.zeros(num_tokens, device="cuda", dtype=torch.int32)
    norm_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    rotary_pairs = head_dim // 4
    cos_sin = torch.cat(
        (
            torch.ones(num_tokens, rotary_pairs, device="cuda", dtype=torch.bfloat16),
            torch.zeros(num_tokens, rotary_pairs, device="cuda", dtype=torch.bfloat16),
        ),
        dim=1,
    )

    triton_qsa_prefill_compress(
        logical_positions=logical_positions,
        request_indices=request_indices,
        block_table=block_table,
        index_cache=index_cache,
        position_cache=position_cache,
        k_norm_weight=norm_weight,
        cos_sin=cos_sin,
        eps=eps,
        tokens_per_block=tokens_per_block,
        compress_ratio=compress_ratio,
        mrope_section=None,
    )

    actual = index_cache.view(num_tokens, head_dim)
    expected = original.view(num_tokens, head_dim).clone()
    for anchor in range(compress_ratio - 1, num_tokens, compress_ratio):
        pooled = (
            original.view(num_tokens, head_dim)[anchor - compress_ratio + 1 : anchor + 1]
            .float()
            .mean(dim=0)
            .to(torch.bfloat16)
        )
        normalized = pooled.float() * torch.rsqrt(pooled.float().square().mean() + eps)
        expected[anchor] = (normalized * (norm_weight.float() + 1.0)).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qsa_selection_is_causal_and_score_ordered() -> None:
    q = torch.ones(2, 4, 4)
    compressed = torch.stack(
        (
            torch.ones(4),
            torch.full((4,), 2.0),
            torch.full((4,), 3.0),
        )
    )
    selected = select_qsa_tokens(
        q,
        compressed,
        torch.tensor([5, 9]),
        sequence_length=10,
        params=_params(),
    )
    assert selected[0].tolist() == [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1]
    assert selected[1].tolist() == [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_paged_selection_supports_multiple_rows_per_request() -> None:
    # Triton's dot primitive requires K >= 16 on SM100+. Production Qwen4
    # uses 128; keep this focused test at the smallest supported dimension.
    params = _params(index_head_dim=16)
    tokens_per_block = 8
    index_cache = torch.zeros(
        4,
        tokens_per_block,
        1,
        params.index_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
    for request_idx in range(2):
        for compressed_idx in range(4):
            logical = compressed_idx * params.compress_ratio + params.compress_ratio - 1
            page_column, within = divmod(logical, tokens_per_block)
            page = int(block_table[request_idx, page_column])
            index_cache[page, within, 0] = compressed_idx + 1

    q = torch.ones(
        4,
        params.index_n_heads,
        params.index_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query_positions = torch.tensor([5, 9, 6, 13], device="cuda")
    # Runtime KV lengths are per request, so all verification rows for one
    # request observe the same final length. Query positions still enforce
    # each row's causal boundary in the selector and incomplete tail.
    sequence_lengths = torch.tensor([10, 10, 14, 14], device="cuda")
    request_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(
        qsa_block_table=block_table,
        kv_cache_manager=SimpleNamespace(tokens_per_block=tokens_per_block),
    )

    actual = select_qsa_paged_tokens(
        q,
        index_cache,
        query_positions,
        sequence_lengths,
        request_indices,
        metadata,
        params,
    )

    from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation

    top_k = TopK(
        params.block_topk,
        prefill_implementation=TopKImplementation.CUDA_RADIX,
        decode_implementation=TopKImplementation.CUDA_RADIX,
        compress_ratio=params.compress_ratio,
    )
    top_k_output = torch.empty(
        (q.shape[0], params.block_topk),
        dtype=torch.int32,
        device="cuda",
    )
    top_k_row_starts = torch.zeros(q.shape[0], dtype=torch.int32, device="cuda")
    actual_radix = select_qsa_paged_tokens(
        q,
        index_cache,
        query_positions,
        sequence_lengths,
        request_indices,
        metadata,
        params,
        top_k=top_k,
        top_k_output=top_k_output,
        top_k_row_starts=top_k_row_starts,
    ).clone()

    assert actual[0].tolist() == [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1]
    assert actual[1].tolist() == [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, -1]
    assert actual[2].tolist() == [0, 1, 2, 3, 4, 5, 6, -1, -1, -1, -1]
    assert actual[3].tolist() == [8, 9, 10, 11, 4, 5, 6, 7, 12, 13, -1]
    # The CUDA radix kernel returns the exact Top-K set but does not promise
    # score order. Sparse attention is invariant to a permutation of the
    # selected token axis.
    assert torch.equal(
        actual_radix.sort(dim=1).values,
        actual.sort(dim=1).values,
    )

    compressed_keys = torch.arange(
        1,
        1 + 4 * params.index_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(4, params.index_head_dim)
    prefill_torch = select_qsa_tokens(
        q,
        compressed_keys,
        query_positions,
        16,
        params,
    )
    prefill_radix = select_qsa_tokens(
        q,
        compressed_keys,
        query_positions,
        16,
        params,
        top_k=top_k,
        top_k_output=top_k_output,
        top_k_row_starts=top_k_row_starts,
    )
    assert torch.equal(
        prefill_radix.sort(dim=1).values,
        prefill_torch.sort(dim=1).values,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_fused_decode_pre_indexer_matches_reference() -> None:
    torch.manual_seed(42)
    rows = 4
    num_heads = 4
    head_dim = 128
    rotary_pairs = head_dim // 4
    tokens_per_block = 8
    compress_ratio = 4
    max_positions = 32
    q = torch.randn(
        rows,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    token_k = torch.randn(
        rows,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_reference_input = q.clone()
    token_k_reference = token_k.clone()
    request_indices = torch.arange(rows, dtype=torch.int32, device="cuda")
    logical_positions = torch.tensor([3, 4, 7, 8], dtype=torch.int64, device="cuda")
    block_table = torch.arange(rows * 2, dtype=torch.int32, device="cuda").reshape(rows, 2)
    index_cache = torch.randn(
        rows * 2,
        tokens_per_block,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    position_cache = torch.empty(
        rows * 2,
        tokens_per_block,
        3,
        dtype=torch.int32,
        device="cuda",
    )
    for request in range(rows):
        for logical in range(2 * tokens_per_block):
            page_column, within = divmod(logical, tokens_per_block)
            page = int(block_table[request, page_column])
            position_cache[page, within] = torch.tensor(
                [logical, logical + 1, logical + 2],
                dtype=torch.int32,
                device="cuda",
            )
    initial_index_cache = index_cache.clone()
    initial_position_cache = position_cache.clone()
    position_coordinates = torch.stack(
        (
            logical_positions,
            logical_positions + 2,
            logical_positions + 4,
        ),
        dim=1,
    ).to(torch.int32)
    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    angles = (
        torch.arange(max_positions, device="cuda", dtype=torch.float32)[:, None]
        * (torch.arange(rotary_pairs, device="cuda", dtype=torch.float32)[None, :] + 1)
        / 97.0
    )
    cos_sin = torch.stack((angles.cos(), angles.sin()), dim=1)
    mrope_section = (11, 11, 10)

    def reference_norm_rope(
        values: torch.Tensor,
        positions: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        normalized = values.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + 1e-6)
        normalized = (normalized * (1.0 + weight.float())).to(torch.bfloat16)
        pairs = torch.arange(rotary_pairs, device="cuda")
        height_pair = (pairs % 3 == 1) & (pairs < 3 * mrope_section[1])
        width_pair = (pairs % 3 == 2) & (pairs < 3 * mrope_section[2])
        selected_positions = torch.where(
            height_pair[None, :],
            positions[:, 1, None],
            torch.where(
                width_pair[None, :],
                positions[:, 2, None],
                positions[:, 0, None],
            ),
        )
        cosine = cos_sin[selected_positions, 0, pairs[None, :]]
        sine = cos_sin[selected_positions, 1, pairs[None, :]]
        first = normalized[..., :rotary_pairs]
        second = normalized[..., rotary_pairs : 2 * rotary_pairs]
        output = normalized.clone()
        output[..., :rotary_pairs] = first * cosine[:, None, :] - second * sine[:, None, :]
        output[..., rotary_pairs : 2 * rotary_pairs] = (
            second * cosine[:, None, :] + first * sine[:, None, :]
        )
        return output

    expected_q = reference_norm_rope(q_reference_input, position_coordinates, q_weight)
    expected_index_cache = initial_index_cache.clone()
    expected_position_cache = initial_position_cache.clone()
    for row in range(rows):
        request = int(request_indices[row])
        logical = int(logical_positions[row])
        page_column, within = divmod(logical, tokens_per_block)
        page = int(block_table[request, page_column])
        expected_position_cache[page, within] = position_coordinates[row]
        stored = token_k_reference[row, 0]
        if (logical + 1) % compress_ratio == 0:
            group = []
            for group_position in range(logical - compress_ratio + 1, logical + 1):
                if group_position == logical:
                    group.append(token_k_reference[row, 0])
                    continue
                group_page_column, group_within = divmod(group_position, tokens_per_block)
                group_page = int(block_table[request, group_page_column])
                group.append(initial_index_cache[group_page, group_within, 0])
            pooled = torch.stack(group).float().mean(dim=0).to(torch.bfloat16)
            first_logical = logical - compress_ratio + 1
            first_page_column, first_within = divmod(first_logical, tokens_per_block)
            first_page = int(block_table[request, first_page_column])
            first_position = initial_position_cache[first_page, first_within].reshape(1, 3)
            stored = reference_norm_rope(
                pooled.reshape(1, 1, head_dim),
                first_position,
                k_weight,
            )[0, 0]
        expected_index_cache[page, within, 0] = stored

    triton_qsa_decode_pre_indexer(
        q=q,
        token_k=token_k,
        position_coordinates=position_coordinates,
        request_indices=request_indices,
        logical_positions=logical_positions,
        block_table=block_table,
        index_cache=index_cache,
        position_cache=position_cache,
        q_norm_weight=q_weight,
        k_norm_weight=k_weight,
        cos_sin=cos_sin.view(max_positions, -1),
        eps=1e-6,
        tokens_per_block=tokens_per_block,
        compress_ratio=compress_ratio,
        mrope_section=mrope_section,
    )

    torch.testing.assert_close(q, expected_q, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(index_cache, expected_index_cache, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(position_cache, expected_position_cache)


def test_qsa_sparse_gqa_reads_hnd_paged_cache() -> None:
    k_cache = torch.empty(2, 1, 4, 2)
    v_cache = torch.empty_like(k_cache)
    for position in range(8):
        page, within = divmod(position, 4)
        k_cache[page, 0, within] = torch.tensor([position + 1.0, position + 2.0])
        v_cache[page, 0, within] = torch.tensor([position * 10.0, position * 10.0 + 1.0])
    selected = torch.tensor([[0, 3, 4, -1]], dtype=torch.int32)
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    metadata = SimpleNamespace(
        qsa_block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(tokens_per_block=4),
    )

    actual = qsa_sparse_gqa(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        selected_tokens=selected,
        request_idx=0,
        metadata=metadata,
        softmax_scale=1.0,
    )

    keys = torch.stack((k_cache[0, 0, 0], k_cache[0, 0, 3], k_cache[1, 0, 0]))
    values = torch.stack((v_cache[0, 0, 0], v_cache[0, 0, 3], v_cache[1, 0, 0]))
    expected = torch.stack(tuple(torch.softmax(keys @ head, dim=0) @ values for head in q[0]))
    torch.testing.assert_close(actual, expected.unsqueeze(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_dim", [64, 256])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fused_qsa_sparse_gqa_matches_reference(head_dim: int, cache_dtype: torch.dtype) -> None:
    torch.manual_seed(41)
    rows = 4
    num_pages = 7
    num_kv_heads = 2
    num_q_heads = 6
    tokens_per_block = 8
    q = torch.randn(rows, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(
        num_pages, num_kv_heads, tokens_per_block, head_dim, dtype=torch.bfloat16, device="cuda"
    ).to(cache_dtype)
    v_cache = torch.randn_like(k_cache, dtype=torch.bfloat16).to(cache_dtype)
    block_table = torch.tensor([[5, 0, 3, 1], [2, 6, 4, 0]], dtype=torch.int32, device="cuda")
    request_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")
    selected = torch.tensor(
        [
            list(range(32)) + [-1, -1, -1],
            list(range(3, 32)) + [-1] * 6,
            list(range(0, 32, 2)) + [-1] * 19,
            list(range(31, -1, -1)) + [-1, -1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    metadata = SimpleNamespace(
        qsa_block_table=block_table,
        kv_cache_manager=SimpleNamespace(tokens_per_block=tokens_per_block),
    )

    actual = qsa_sparse_gqa(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        selected_tokens=selected,
        request_indices=request_indices,
        metadata=metadata,
        softmax_scale=head_dim**-0.5,
    )
    expected = qsa_sparse_gqa_reference(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        selected_tokens=selected,
        request_indices=request_indices,
        metadata=metadata,
        softmax_scale=head_dim**-0.5,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qsa_index_storage_avoids_kv_role_coalescing() -> None:
    manager = object.__new__(QSAMambaHybridCacheManagerV2)
    manager.qsa_index_dim = 128
    manager.qsa_index_storage_dim = 129
    manager.qsa_sparse_layer_ids = [7]
    manager.layer_offsets = {7: 2}

    buffers = manager._extra_buffers_per_layer(tokens_per_block=128)

    assert buffers[2][0].size == 129 * 2 * 128


def test_qsa_position_buffer_keeps_shared_index_mode_outside_dynamo() -> None:
    calls = []

    class _Impl:
        def get_mem_pool_base_address(self, layer_idx, role, index_mode):
            calls.append((layer_idx, role, index_mode))
            return 0

        def get_page_stride(self, layer_idx, role):
            del layer_idx, role
            return 0

    manager = object.__new__(QSAMambaHybridCacheManagerV2)
    manager.qsa_position_layer_id = 7
    manager.layer_offsets = {7: 2}
    manager.impl = _Impl()
    manager.tokens_per_block = 128

    with pytest.raises(RuntimeError, match="position-cache page stride mismatch"):
        manager.get_qsa_position_buffer()

    assert len(calls) == 1
    assert calls[0][0] == 2
    assert calls[0][2] is PageIndexMode.SHARED


def test_qsa_speculative_commit_restores_rejected_side_cache_entries() -> None:
    indexer = object.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    index_cache = torch.arange(24, dtype=torch.float32).reshape(3, 2, 1, 4)
    position_cache = torch.arange(18, dtype=torch.int32).reshape(3, 2, 3)
    pages = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    within = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    indexer._pending_speculative_cache = {
        "request_indices": torch.zeros(4, dtype=torch.long),
        "token_ordinals": torch.arange(4),
        "pages": pages,
        "within": within,
        "index_values": index_cache[pages, within, 0].clone(),
        "index_cache": index_cache,
        "position_values": position_cache[pages, within].clone(),
        "position_cache": position_cache,
    }
    index_cache[pages, within, 0] = -1
    position_cache[pages, within] = -1

    indexer.commit_speculative_states(
        num_accepted_tokens=torch.tensor([2]),
        state_indices=torch.tensor([0]),
        num_contexts=0,
    )

    # Two accepted verification tokens retain their new values; later proposal
    # entries are restored to the snapshot.
    assert torch.all(index_cache[pages[:2], within[:2], 0] == -1)
    assert torch.all(position_cache[pages[:2], within[:2]] == -1)
    assert torch.all(index_cache[pages[2:], within[2:], 0] != -1)
    assert torch.all(position_cache[pages[2:], within[2:]] != -1)
    assert indexer._pending_speculative_cache is None


def test_ple_state_views_use_v2_lifecycle_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    token_state = torch.full((12, 2), 11, dtype=torch.int64)
    conv_state = torch.zeros((12, 16, 6), dtype=torch.bfloat16)
    requested = []

    def fake_get_state_buffer(self, local_layer_idx, role, dtype, state_shape):
        del self
        requested.append((local_layer_idx, role, dtype, state_shape))
        if role == MambaRole.PLE_NGRAM_CONTEXT:
            return token_state
        if role == MambaRole.PLE_CONV_STATE:
            return conv_state
        raise AssertionError(f"unexpected role {role}")

    monkeypatch.setattr(MambaHybridCacheManagerV2, "_get_state_buffer", fake_get_state_buffer)
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager._ple_layer_ids = [1]
    manager._ple_ngram_context_shape = [2]
    manager._ple_conv_state_shape = [16, 6]
    manager._ple_conv_state_dtype = torch.bfloat16
    manager._ple_ngram_contexts = {}
    manager._ple_conv_states = {}
    manager.layer_offsets = {1: 0}

    manager._setup_ple_states(num_state_slots=12)

    actual_conv, actual_token = manager.ple_layer_cache(1)
    assert actual_conv is conv_state
    assert actual_token is token_state
    assert requested == [
        (0, MambaRole.PLE_CONV_STATE, torch.bfloat16, [16, 6]),
        (0, MambaRole.PLE_NGRAM_CONTEXT, torch.int64, [2]),
    ]
