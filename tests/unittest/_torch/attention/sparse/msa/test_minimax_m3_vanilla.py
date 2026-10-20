# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytic CPU checks of the MiniMax-M3 Vanilla golden.

The backend sweep in ``test_attention_backends.py`` compares the golden with
the Triton runtime end to end. These tests pin the selection semantics both
share: causal block scoring, forced init/local blocks, the index-head union and
the cache writes. Zero queries give uniform attention weights, so every
expected output is the mean of the selected values.
"""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import MiniMaxM3VanillaAttention
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import MiniMaxM3SparseParams
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.triton_metadata import (
    MiniMaxM3TritonSparseAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.registry import (
    get_vanilla_sparse_attn_attention_backend,
)

pytestmark = pytest.mark.cpu_only


def _params(**overrides) -> MiniMaxM3SparseParams:
    """Two index heads of width 2, blocks of 2 positions, one block per head."""
    params = dict(
        num_index_heads=2,
        sparse_index_dim=2,
        block_size=2,
        topk=1,
        init_blocks=0,
        local_blocks=0,
    )
    params.update(overrides)
    return MiniMaxM3SparseParams(**params)


def _attention(**overrides) -> MiniMaxM3VanillaAttention:
    """Two query heads of width 2 sharing one KV head."""
    return MiniMaxM3VanillaAttention(
        layer_idx=3,
        num_heads=2,
        head_dim=2,
        num_kv_heads=1,
        sparse_params=_params(**overrides),
    )


def _prefill_metadata(slots: list[int]) -> MiniMaxM3TritonSparseAttentionMetadata:
    """One request prefilling ``len(slots)`` tokens; position p lives in ``slots[p]``."""
    length = len(slots)
    metadata = MiniMaxM3TritonSparseAttentionMetadata(
        is_prefill=True,
        req_to_token=torch.tensor([slots], dtype=torch.int32),
        slot_ids=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([length], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([length], dtype=torch.int32),
        prefix_lens=torch.tensor([0], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, length], dtype=torch.int32),
        extend_seq_lens_cpu=[length],
    )
    metadata.prepare()
    return metadata


def _decode_metadata(slots: list[int]) -> MiniMaxM3TritonSparseAttentionMetadata:
    """One request decoding position ``len(slots) - 1``; position p lives in ``slots[p]``."""
    length = len(slots)
    metadata = MiniMaxM3TritonSparseAttentionMetadata(
        is_prefill=False,
        req_to_token=torch.tensor([slots], dtype=torch.int32),
        slot_ids=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([length], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([length], dtype=torch.int32),
    )
    metadata.prepare()
    return metadata


def test_minimax_m3_vanilla_registry_selects_triton_golden() -> None:
    assert get_vanilla_sparse_attn_attention_backend(_params()) is MiniMaxM3VanillaAttention


@pytest.mark.parametrize(
    "overrides, message",
    [
        pytest.param(dict(implementation="msa"), "implementation='msa'", id="msa"),
        pytest.param(dict(indexer_kv_dtype="fp8"), "indexer_kv_dtype='fp8'", id="fp8_index"),
    ],
)
def test_minimax_m3_vanilla_rejects_configs_without_golden(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        get_vanilla_sparse_attn_attention_backend(_params(**overrides))
    with pytest.raises(ValueError, match=message):
        _attention(**overrides)


def test_minimax_m3_vanilla_prefill_selects_indexed_block() -> None:
    attention = _attention()
    # Positions 0..3 live in shuffled slots of an 8-slot cache.
    slots = [5, 2, 7, 0]
    v = torch.tensor([[1.0, 0.0], [3.0, 0.0], [100.0, 0.0], [100.0, 0.0]])
    # Both index heads score a position s as idx_k[s, 0]: block 0 (positions
    # 0, 1) scores +1 and block 1 (positions 2, 3) scores -1 once visible.
    idx_q = torch.tensor([[1.0, 0.0, 1.0, 0.0]]).expand(4, -1).clone()
    idx_k = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    k_cache = torch.zeros(8, 1, 2)
    v_cache = torch.zeros_like(k_cache)
    idx_k_cache = torch.zeros(8, 1, 2)
    output = torch.empty(4, 4)

    result = attention.forward(
        torch.zeros(4, 4),
        torch.zeros(4, 2),
        v,
        None,
        output=output,
        idx_q=idx_q,
        idx_k=idx_k,
        k_cache=k_cache,
        v_cache=v_cache,
        idx_k_cache=idx_k_cache,
        out_cache_loc=torch.tensor(slots, dtype=torch.int32),
        m3_metadata=_prefill_metadata(slots),
    )

    # Position 0 sees only itself. Every later position keeps block 0 and
    # averages its values (1 + 3) / 2 = 2 in both query heads.
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
    torch.testing.assert_close(v_cache[slots, 0], v)
    torch.testing.assert_close(idx_k_cache[slots, 0], idx_k)


@pytest.mark.parametrize(
    "init_blocks, local_blocks, expected",
    [
        # The last visible block holds only the new token at position 4.
        pytest.param(0, 1, [7.0, 9.0], id="local"),
        # Block 0 holds positions 0 and 1: ([10, 0] + [20, 1]) / 2.
        pytest.param(1, 0, [15.0, 0.5], id="init"),
        # With one block per index head, the init score outranks the local one.
        pytest.param(1, 1, [15.0, 0.5], id="init_outranks_local"),
    ],
)
def test_minimax_m3_vanilla_decode_forces_init_and_local_blocks(
    init_blocks, local_blocks, expected
) -> None:
    attention = _attention(init_blocks=init_blocks, local_blocks=local_blocks)
    # Positions 0..4 form blocks {0, 1}, {2, 3} and {4}; position 4 is new.
    slots = [6, 1, 4, 0, 3]
    k_cache = torch.zeros(8, 1, 2)
    v_cache = torch.zeros_like(k_cache)
    idx_k_cache = torch.zeros(8, 1, 2)
    cached_v = torch.tensor([[10.0, 0.0], [20.0, 1.0], [30.0, 2.0], [40.0, 3.0]])
    v_cache[slots[:4], 0] = cached_v

    result = attention.forward(
        torch.zeros(1, 4),
        torch.zeros(1, 2),
        torch.tensor([[7.0, 9.0]]),
        None,
        # Zero index queries score every block equally, so only the forced
        # blocks decide the selection.
        idx_q=torch.zeros(1, 4),
        idx_k=torch.zeros(1, 2),
        k_cache=k_cache,
        v_cache=v_cache,
        idx_k_cache=idx_k_cache,
        out_cache_loc=torch.tensor([slots[4]], dtype=torch.int32),
        m3_metadata=_decode_metadata(slots),
    )

    torch.testing.assert_close(result, torch.tensor([expected * 2]))


def test_minimax_m3_vanilla_unions_index_heads_of_kv_group() -> None:
    attention = _attention()
    # Positions 0..5 form blocks {0, 1}, {2, 3} and {4, 5}. Index head 0
    # prefers block 0 and index head 1 prefers block 1; block 2 scores lowest.
    index_keys = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]]
    )
    disagreeing_heads = [[1.0, 0.0], [0.0, 1.0]]
    agreeing_heads = [[1.0, 0.0], [1.0, 0.0]]

    selected = attention.indexer.select_request_blocks(
        torch.tensor([disagreeing_heads, agreeing_heads]),
        index_keys,
        torch.tensor([6, 6]),
        idx_sm_scale=1.0,
    )

    # One KV head unions its two index heads: ascending ids, then padding.
    torch.testing.assert_close(selected, torch.tensor([[[0, 1]], [[0, -1]]]))

    slots = [3, 0, 5, 1, 2, 4]
    k_cache = torch.zeros(6, 1, 2)
    v_cache = torch.zeros_like(k_cache)
    idx_k_cache = torch.zeros(6, 1, 2)
    v_cache[slots[:5], 0] = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 0.0], [100.0, 0.0]]
    )
    idx_k_cache[slots[:5], 0] = index_keys[:5]

    result = attention.forward(
        torch.zeros(1, 4),
        torch.zeros(1, 2),
        torch.tensor([[100.0, 0.0]]),
        None,
        idx_q=torch.tensor([disagreeing_heads]).reshape(1, 4),
        idx_k=index_keys[5:],
        k_cache=k_cache,
        v_cache=v_cache,
        idx_k_cache=idx_k_cache,
        out_cache_loc=torch.tensor([slots[5]], dtype=torch.int32),
        m3_metadata=_decode_metadata(slots),
    )

    # Blocks 0 and 1 together hold (1 + 3 + 5 + 7) / 4 = 4; either head alone
    # would give 2 or 6.
    torch.testing.assert_close(result, torch.tensor([[4.0, 0.0, 4.0, 0.0]]))
