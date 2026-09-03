# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.qsa import (
    QSAMambaHybridCacheManagerV2,
    QSASparseMetadataParams,
    QSASparseParams,
)
from tensorrt_llm._torch.attention_backend.sparse.qsa.indexer import (
    QSAIndexer,
    _position_coordinates,
    average_pool_qsa_keys,
)
from tensorrt_llm._torch.attention_backend.sparse.qsa.kernels import (
    _expand_launch,
    _qsa_index_scores_query_tile,
    triton_qsa_decode_pre_indexer,
    triton_qsa_decode_token_mapping,
    triton_qsa_paged_index_scores,
    triton_qsa_paged_kv_store,
    triton_qsa_prefill_compress,
    triton_qsa_unscale_block_table,
)
from tensorrt_llm._torch.attention_backend.sparse.qsa.metadata import QSAAttentionMetadata
from tensorrt_llm._torch.attention_backend.sparse.qsa.module import (
    _store_paged_kv_reference,
    expand_qsa_block_indices,
    qsa_sparse_gqa,
    qsa_sparse_gqa_reference,
    select_qsa_paged_tokens,
    select_qsa_tokens,
)
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

    metadata_params = QSASparseMetadataParams(token_topk=8, compress_ratio=4)
    assert metadata_params.block_topk == 2
    with pytest.raises(ValueError, match="divisible"):
        QSASparseMetadataParams(token_topk=7, compress_ratio=4)


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


@pytest.mark.parametrize(
    "rows, final_topk, expected",
    [(1, 2051, (256, 4)), (127, 2051, (256, 4)), (128, 2051, (4096, 8)), (1, 11, (16, 4))],
)
def test_expand_launch_splits_columns_only_for_narrow_batches(
    rows: int, final_topk: int, expected: tuple[int, int]
) -> None:
    assert _expand_launch(rows, final_topk) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_expand_qsa_blocks_rejects_non_unit_inner_stride() -> None:
    """Only stride(0) reaches the kernel, so the other axes must be dense."""
    from tensorrt_llm._torch.attention_backend.sparse.qsa.kernels import (
        triton_expand_qsa_block_indices,
    )

    compress_ratio, token_topk = 4, 8
    block_topk = token_topk // compress_ratio
    rows = 2
    # A column-strided view: shape is right, inner stride is not.
    strided = torch.zeros((rows, block_topk * 2), dtype=torch.int32, device="cuda")[:, ::2]
    query_positions = torch.zeros(rows, dtype=torch.int32, device="cuda")
    sequence_lengths = torch.ones(rows, dtype=torch.int32, device="cuda")
    arguments = dict(compress_ratio=compress_ratio, token_topk=token_topk)

    with pytest.raises(ValueError, match="contiguous along their last dimension"):
        triton_expand_qsa_block_indices(strided, query_positions, sequence_lengths, **arguments)

    dense = strided.contiguous()
    with pytest.raises(ValueError, match="metadata must be contiguous"):
        triton_expand_qsa_block_indices(
            dense,
            torch.zeros(rows * 2, dtype=torch.int32, device="cuda")[::2],
            sequence_lengths,
            **arguments,
        )


def test_expand_qsa_blocks_column_split_matches_whole_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(42)
    compress_ratio = 4
    token_topk = 2048
    block_topk = token_topk // compress_ratio
    # Full width with no tail, full width with a tail, and a row whose visible
    # prefix is too short to fill the selection.
    query_positions = torch.tensor([9215, 8190, 1234], dtype=torch.int32)
    block_indices = torch.full((3, block_topk), -1, dtype=torch.int32)
    for row, position in enumerate(query_positions.tolist()):
        visible_blocks = (position + 1) // compress_ratio
        width = min(block_topk, visible_blocks)
        block_indices[row, :width] = torch.randperm(visible_blocks, dtype=torch.int32)[:width]
    sequence_lengths = query_positions + 1

    arguments = dict(compress_ratio=compress_ratio, token_topk=token_topk)
    split = expand_qsa_block_indices(
        block_indices.cuda(),
        query_positions.cuda(),
        sequence_lengths.cuda(),
        **arguments,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention_backend.sparse.qsa.kernels._EXPAND_ROWS_FILLING_DEVICE",
        0,
    )
    whole_row = expand_qsa_block_indices(
        block_indices.cuda(),
        query_positions.cuda(),
        sequence_lengths.cuda(),
        **arguments,
    )
    reference = expand_qsa_block_indices(
        block_indices,
        query_positions,
        sequence_lengths,
        **arguments,
    )

    # Selected tokens must be bit-identical, not merely close: they index the
    # KV cache.
    assert torch.equal(split, whole_row)
    assert torch.equal(split.cpu(), reference)


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


def test_qsa_position_coordinates_preserve_scheduler_views() -> None:
    positions_1d = torch.arange(5, dtype=torch.int64)
    coordinates_1d = _position_coordinates(positions_1d, 5)
    assert coordinates_1d.shape == (5, 3)
    assert coordinates_1d.stride() == (1, 0)
    assert coordinates_1d.untyped_storage().data_ptr() == positions_1d.untyped_storage().data_ptr()

    positions_mrope = torch.arange(3 * 2 * 5, dtype=torch.int64).reshape(3, 2, 5)
    coordinates_mrope = _position_coordinates(positions_mrope, 10)
    assert coordinates_mrope.shape == (10, 3)
    assert coordinates_mrope.stride() == (1, 10)
    assert (
        coordinates_mrope.untyped_storage().data_ptr()
        == positions_mrope.untyped_storage().data_ptr()
    )

    with pytest.raises(ValueError, match="three coordinate axes"):
        _position_coordinates(torch.zeros(2, 1, 5, dtype=torch.int64), 5)


def test_qsa_fused_pre_indexer_requires_neox_rope_layout() -> None:
    indexer = object.__new__(QSAIndexer)
    indexer.params = SimpleNamespace(index_head_dim=128, compress_ratio=4)
    indexer.rotary_emb = SimpleNamespace(rope_params=SimpleNamespace(dim=64))
    rotary_cache = torch.empty(32, 2, 32)

    indexer._is_neox_rope = True
    assert indexer._supports_fused_rope(rotary_cache)
    indexer._is_neox_rope = False
    assert not indexer._supports_fused_rope(rotary_cache)


@pytest.mark.parametrize("rows", [1, 64, 257])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_decode_token_mapping_matches_reference(rows: int) -> None:
    seq_lens = torch.ones(rows, dtype=torch.int32, device="cuda")
    kv_lens = torch.arange(17, 17 + rows, dtype=torch.int32, device="cuda")
    request_indices = torch.empty(rows, dtype=torch.int32, device="cuda")
    logical_positions = torch.empty(rows, dtype=torch.int64, device="cuda")
    sequence_lengths = torch.empty(rows, dtype=torch.int32, device="cuda")
    visible_blocks = torch.empty(rows, dtype=torch.int32, device="cuda")

    triton_qsa_decode_token_mapping(
        kv_lens=kv_lens,
        seq_lens=seq_lens,
        request_indices=request_indices,
        logical_positions=logical_positions,
        sequence_lengths=sequence_lengths,
        visible_blocks=visible_blocks,
        compress_ratio=4,
    )

    torch.testing.assert_close(
        request_indices,
        torch.arange(rows, dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(logical_positions, (kv_lens - seq_lens).to(torch.int64))
    torch.testing.assert_close(sequence_lengths, kv_lens)
    torch.testing.assert_close(visible_blocks, ((kv_lens - seq_lens + 1) // 4))


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_paged_kv_store_matches_advanced_indexing(cache_dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    rows = 4
    num_pages = 8
    num_kv_heads = 2
    tokens_per_block = 8
    head_dim = 256
    qkv = torch.randn(
        rows,
        5,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = qkv[:, 1:3]
    v = qkv[:, 3:5]
    block_table = torch.tensor(
        [[3, 0, 6, 2], [5, 7, 1, 4]],
        dtype=torch.int32,
        device="cuda",
    )
    request_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")
    logical_positions = torch.tensor([0, 5, 9, 15], dtype=torch.int64, device="cuda")
    kv_cache = torch.full(
        (num_pages, 2, num_kv_heads, tokens_per_block, head_dim),
        -2.0,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(cache_dtype)
    k_cache = kv_cache[:, 0]
    v_cache = kv_cache[:, 1]
    expected_cache = kv_cache.clone()
    expected_k = expected_cache[:, 0]
    expected_v = expected_cache[:, 1]
    page_columns = logical_positions // tokens_per_block
    pages = block_table[request_indices.to(torch.long), page_columns]
    within = logical_positions % tokens_per_block
    expected_k[pages, :, within, :] = k.to(cache_dtype)
    expected_v[pages, :, within, :] = v.to(cache_dtype)

    triton_qsa_paged_kv_store(
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        request_indices=request_indices,
        logical_positions=logical_positions,
        block_table=block_table,
        tokens_per_block=tokens_per_block,
    )

    torch.testing.assert_close(k_cache.float(), expected_k.float(), rtol=0, atol=0)
    torch.testing.assert_close(v_cache.float(), expected_v.float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_paged_kv_store_skips_unallocated_pages() -> None:
    k = torch.ones(1, 2, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.full_like(k, 2.0)
    k_cache = torch.zeros(2, 2, 8, 64, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.zeros_like(k_cache)

    triton_qsa_paged_kv_store(
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        request_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        logical_positions=torch.zeros(1, dtype=torch.int64, device="cuda"),
        block_table=torch.full((1, 1), -1, dtype=torch.int32, device="cuda"),
        tokens_per_block=8,
    )

    torch.testing.assert_close(k_cache, torch.zeros_like(k_cache))
    torch.testing.assert_close(v_cache, torch.zeros_like(v_cache))


def test_qsa_paged_kv_store_rejects_unsupported_storage_dtypes() -> None:
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    request_indices = torch.zeros(1, dtype=torch.int32)
    logical_positions = torch.zeros(1, dtype=torch.int64)
    fp32_k = torch.zeros(1, 1, 4)
    fp32_v = torch.zeros_like(fp32_k)
    bf16_cache = torch.zeros(1, 1, 1, 4, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="inputs must use BF16 or FP16"):
        triton_qsa_paged_kv_store(
            k=fp32_k,
            v=fp32_v,
            k_cache=bf16_cache,
            v_cache=bf16_cache.clone(),
            request_indices=request_indices,
            logical_positions=logical_positions,
            block_table=block_table,
            tokens_per_block=1,
        )

    bf16_k = fp32_k.to(torch.bfloat16)
    bf16_v = fp32_v.to(torch.bfloat16)
    fp32_cache = torch.zeros(1, 1, 1, 4)
    with pytest.raises(ValueError, match="cache must use BF16, FP16, or FP8 E4M3"):
        triton_qsa_paged_kv_store(
            k=bf16_k,
            v=bf16_v,
            k_cache=fp32_cache,
            v_cache=fp32_cache.clone(),
            request_indices=request_indices,
            logical_positions=logical_positions,
            block_table=block_table,
            tokens_per_block=1,
        )


@pytest.mark.parametrize("rows", [1, 64, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_unscale_block_table_recovers_lifecycle_slots(rows: int) -> None:
    scale = 48
    slots = torch.arange(rows * 289, dtype=torch.int32, device="cuda").reshape(rows, 289)
    slots[:, -1] = -1
    scaled_storage = torch.zeros((rows, 2, 289), dtype=torch.int32, device="cuda")
    scaled = scaled_storage[:, 0, :]
    # V2 scales valid offsets but preserves BAD_PAGE_INDEX (-1).
    scaled.copy_(torch.where(slots < 0, slots, slots * scale))
    assert scaled.stride() == (2 * 289, 1)
    actual = torch.empty_like(slots)

    triton_qsa_unscale_block_table(
        scaled_block_table=scaled,
        block_table=actual,
        page_index_scale=scale,
    )

    torch.testing.assert_close(actual, slots)


@pytest.mark.parametrize("rows,head_dim", [(1, 16), (64, 24), (257, 16)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_paged_index_scores_match_reference(rows: int, head_dim: int) -> None:
    torch.manual_seed(42)
    tokens_per_block = 8
    compress_ratio = 4
    pages_per_request = 300
    num_heads = 4
    block_table = (
        torch.arange(
            pages_per_request,
            dtype=torch.int32,
            device="cuda",
        )
        .unsqueeze(0)
        .expand(rows, -1)
        .contiguous()
    )
    index_cache = torch.randn(
        pages_per_request,
        tokens_per_block,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q = torch.randn(
        rows,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query_positions = (
        pages_per_request * tokens_per_block - 1 - torch.arange(rows, device="cuda") % 13
    )
    request_indices = torch.arange(
        rows,
        dtype=torch.int32,
        device="cuda",
    )

    actual = triton_qsa_paged_index_scores(
        q=q,
        index_cache=index_cache,
        block_table=block_table,
        query_positions=query_positions,
        request_indices=request_indices,
        tokens_per_block=tokens_per_block,
        compress_ratio=compress_ratio,
    )

    max_compressed_blocks = pages_per_request * tokens_per_block // compress_ratio
    block_indices = torch.arange(max_compressed_blocks, device="cuda")
    anchor_positions = block_indices * compress_ratio + compress_ratio - 1
    logical_pages = anchor_positions // tokens_per_block
    tokens_in_page = anchor_positions % tokens_per_block
    physical_pages = block_table[:, logical_pages]
    keys = index_cache[physical_pages, tokens_in_page, 0]
    expected = torch.einsum("rhd,rbd->rhb", q.float(), keys.float())
    expected = expected.clamp_min(0).sum(dim=1) * head_dim**-0.5
    visible = block_indices.unsqueeze(0) < ((query_positions + 1) // compress_ratio).unsqueeze(1)
    expected = expected.masked_fill(~visible, -float("inf"))

    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))
    torch.testing.assert_close(actual[visible], expected[visible], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "num_index_heads, expected",
    [(1, 16), (2, 8), (4, 4), (8, 2), (16, 1), (3, 1), (12, 1), (24, 1), (0, 1)],
)
def test_qsa_index_scores_query_tile_divides_the_dot_width(
    num_index_heads: int, expected: int
) -> None:
    query_tile = _qsa_index_scores_query_tile(num_index_heads)
    assert query_tile == expected
    if query_tile > 1:
        assert query_tile * num_index_heads == 16


def _paged_index_scores_reference(
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_compressed_blocks = block_table.shape[1] * tokens_per_block // compress_ratio
    block_indices = torch.arange(max_compressed_blocks, device=q.device)
    anchor_positions = block_indices * compress_ratio + compress_ratio - 1
    physical_pages = block_table[request_indices.long()][
        :, anchor_positions // tokens_per_block
    ].long()
    keys = index_cache[physical_pages, anchor_positions % tokens_per_block, 0]
    scores = torch.einsum("rhd,rbd->rhb", q.float(), keys.float())
    scores = scores.clamp_min(0).sum(dim=1) * q.shape[-1] ** -0.5
    visible = block_indices.unsqueeze(0) < ((query_positions + 1) // compress_ratio).unsqueeze(1)
    return scores, visible


@pytest.mark.parametrize("only_visible_blocks", [False, True])
@pytest.mark.parametrize("rows_per_request", ["many", "one"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_paged_index_scores_query_tile_matches_row_wise(
    only_visible_blocks: bool, rows_per_request: str
) -> None:
    torch.manual_seed(7)
    tokens_per_block = 8
    compress_ratio = 4
    pages_per_request = 40
    num_heads = 4
    head_dim = 16
    # The query tile only replaces the row-wise kernel above the prefill row
    # boundary, so keep the launch there or the comparison is vacuous.
    assert _qsa_index_scores_query_tile(num_heads) == 4
    if rows_per_request == "many":
        # Request boundaries deliberately land inside a query tile.
        lengths = [101, 157, 42]
        cached = [0, 100, 60]
    else:
        # A tile whose rows all belong to different requests cannot share a
        # gather; every row must still be scored against its own block table.
        lengths = [1] * 300
        cached = [(17 * row) % 200 for row in range(300)]
    rows = sum(lengths)
    assert rows > 256

    request_indices = torch.tensor(
        [request for request, length in enumerate(lengths) for _ in range(length)],
        dtype=torch.int32,
        device="cuda",
    )
    query_positions = torch.tensor(
        [start + offset for length, start in zip(lengths, cached) for offset in range(length)],
        dtype=torch.int64,
        device="cuda",
    )
    pages = len(lengths) * pages_per_request
    block_table = torch.arange(pages, dtype=torch.int32, device="cuda").reshape(
        len(lengths), pages_per_request
    )
    index_cache = torch.randn(
        pages,
        tokens_per_block,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q = torch.randn(rows, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")

    arguments = {
        "q": q,
        "index_cache": index_cache,
        "block_table": block_table,
        "query_positions": query_positions,
        "request_indices": request_indices,
        "tokens_per_block": tokens_per_block,
        "compress_ratio": compress_ratio,
        "only_visible_blocks": only_visible_blocks,
    }
    row_wise = triton_qsa_paged_index_scores(**arguments, context_rows=False)
    tiled = triton_qsa_paged_index_scores(**arguments, context_rows=True)

    expected, visible = _paged_index_scores_reference(
        q,
        index_cache,
        block_table,
        query_positions,
        request_indices,
        tokens_per_block,
        compress_ratio,
    )
    torch.testing.assert_close(tiled[visible], expected[visible], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tiled[visible], row_wise[visible], rtol=1e-6, atol=1e-6)
    if not only_visible_blocks:
        assert torch.equal(torch.isfinite(tiled), visible)
        assert torch.equal(torch.isfinite(row_wise), visible)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_paged_selection_torch_fallback_stays_causal() -> None:
    """The Torch Top-K fallback must not select blocks after the query row.

    This path calls ``triton_qsa_paged_index_scores`` with
    ``only_visible_blocks`` disabled, so the scorer writes every column rather
    than leaving the out-of-range ones unspecified. It still applies the
    per-row causal block bound in that mode, storing ``-inf`` beyond the
    boundary, which is what keeps the unbounded ``torch.topk`` below causal.
    The side cache here scores later blocks highest, so a scorer that dropped
    the bound would immediately surface as non-causal selection.
    """
    params = _params(index_head_dim=16)
    tokens_per_block = 8
    index_cache = torch.zeros(
        2,
        tokens_per_block,
        1,
        params.index_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    # Strictly increasing per-block scores: an unmasked Top-K always prefers
    # the newest compressed blocks, which are exactly the non-causal ones.
    for compressed_idx in range(4):
        logical = compressed_idx * params.compress_ratio + params.compress_ratio - 1
        page_column, within = divmod(logical, tokens_per_block)
        page = int(block_table[0, page_column])
        index_cache[page, within, 0] = compressed_idx + 1

    q = torch.ones(
        1,
        params.index_n_heads,
        params.index_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # Position 5 sees compressed blocks [0, 1) only: (5 + 1) // 4 == 1.
    query_positions = torch.tensor([5], device="cuda")
    sequence_lengths = torch.tensor([16], device="cuda")
    request_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(
        qsa_block_table=block_table,
        kv_cache_manager=SimpleNamespace(tokens_per_block=tokens_per_block),
    )

    selected = select_qsa_paged_tokens(
        q,
        index_cache,
        query_positions,
        sequence_lengths,
        request_indices,
        metadata,
        params,
    )

    chosen = [token for token in selected[0].tolist() if token >= 0]
    assert chosen, "the visible block must still be selected"
    assert max(chosen) <= int(query_positions[0]), (
        f"selected tokens {chosen} exceed the causal boundary {int(query_positions[0])}"
    )


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
        dim=0,
    ).transpose(0, 1)
    assert position_coordinates.dtype == torch.int64
    assert not position_coordinates.is_contiguous()
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


def test_qsa_sparse_gqa_all_invalid_row_returns_zero() -> None:
    q = torch.randn(1, 2, 4)
    k_cache = torch.randn(1, 1, 4, 4)
    v_cache = torch.randn_like(k_cache)
    metadata = SimpleNamespace(
        qsa_block_table=torch.tensor([[0]], dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(tokens_per_block=4),
    )

    actual = qsa_sparse_gqa_reference(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        selected_tokens=torch.full((1, 3), -1, dtype=torch.int32),
        request_indices=torch.tensor([0], dtype=torch.int32),
        metadata=metadata,
        softmax_scale=0.5,
    )

    torch.testing.assert_close(actual, torch.zeros_like(actual))


@pytest.mark.parametrize(
    "block_table",
    [torch.empty((0, 0), dtype=torch.int32), torch.tensor([[-1]], dtype=torch.int32)],
)
def test_qsa_reference_kv_store_skips_unallocated_pages(block_table: torch.Tensor) -> None:
    k_cache = torch.full((2, 1, 4, 2), 7.0)
    v_cache = torch.full_like(k_cache, 9.0)

    _store_paged_kv_reference(
        k=torch.randn(1, 1, 2),
        v=torch.randn(1, 1, 2),
        k_cache=k_cache,
        v_cache=v_cache,
        request_indices=torch.tensor([0], dtype=torch.int32),
        logical_positions=torch.tensor([0], dtype=torch.int32),
        block_table=block_table,
        tokens_per_block=4,
    )

    assert torch.all(k_cache == 7)
    assert torch.all(v_cache == 9)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_qsa_prefill_bounds_sparse_attention_to_visible_tokens() -> None:
    torch.manual_seed(43)
    rows = 257
    num_requests = 2
    num_kv_heads = 2
    num_q_heads = 6
    head_dim = 64
    tokens_per_block = 8
    compress_ratio = 4
    token_topk = 32
    final_topk = token_topk + compress_ratio - 1
    pages_per_request = 4
    q = torch.randn(rows, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(
        num_requests * pages_per_request,
        num_kv_heads,
        tokens_per_block,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(
        num_requests * pages_per_request,
        dtype=torch.int32,
        device="cuda",
    ).reshape(num_requests, pages_per_request)
    request_indices = (torch.arange(rows, device="cuda") % num_requests).to(torch.int32)
    query_positions = torch.arange(rows, device="cuda") % (pages_per_request * tokens_per_block)
    visible_tokens = query_positions + 1
    selected = torch.full((rows, final_topk), -1, dtype=torch.int32, device="cuda")
    columns = torch.arange(final_topk, device="cuda").unsqueeze(0)
    selected.copy_(torch.where(columns < visible_tokens.unsqueeze(1), columns, -1))
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
        query_positions=query_positions,
        compress_ratio=compress_ratio,
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


def test_qsa_side_buffers_use_exact_geometry_and_one_position_role() -> None:
    manager = object.__new__(QSAMambaHybridCacheManagerV2)
    manager.qsa_index_dim = 128
    manager.qsa_index_kv_heads = 1
    manager.qsa_sparse_layer_ids = [7, 11]
    manager.layer_offsets = {7: 2, 11: 4}

    buffers = manager._extra_buffers_per_layer(tokens_per_block=128)

    assert buffers[2][0].size == 128 * 2 * 128
    assert len(buffers[2]) == 2
    assert len(buffers[4]) == 1


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
        "index_values": index_cache[pages, within].clone(),
        "index_cache": index_cache,
        "position_values": position_cache[pages, within].clone(),
        "position_cache": position_cache,
    }
    index_cache[pages, within] = -1
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


def test_qsa_speculative_abort_restores_all_side_cache_entries() -> None:
    indexer = object.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    index_cache = torch.arange(24, dtype=torch.float32).reshape(3, 2, 1, 4)
    position_cache = torch.arange(18, dtype=torch.int32).reshape(3, 2, 3)
    pages = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    within = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    expected_index = index_cache.clone()
    expected_position = position_cache.clone()
    indexer._pending_speculative_cache = {
        "pages": pages,
        "within": within,
        "index_values": index_cache[pages, within].clone(),
        "index_cache": index_cache,
        "position_values": position_cache[pages, within].clone(),
        "position_cache": position_cache,
    }
    index_cache[pages, within] = -1
    position_cache[pages, within] = -1

    indexer.abort_speculative_states()

    torch.testing.assert_close(index_cache, expected_index)
    torch.testing.assert_close(position_cache, expected_position)
    assert indexer._pending_speculative_cache is None


@pytest.mark.parametrize(
    ("sequence_lengths", "num_contexts", "expected"),
    [
        ([8, 1], 1, False),
        ([8, 4], 1, True),
        ([1, 1], 2, False),
        ([4, 1, 3], 1, True),
    ],
)
def test_qsa_speculative_snapshot_state_uses_host_lengths(
    sequence_lengths: list[int],
    num_contexts: int,
    expected: bool,
) -> None:
    metadata = object.__new__(QSAAttentionMetadata)
    metadata._seq_lens = torch.tensor(sequence_lengths, dtype=torch.int32)
    metadata._num_contexts = num_contexts

    metadata._refresh_qsa_speculative_snapshot_state()

    assert metadata.qsa_needs_speculative_snapshot is expected


def test_qsa_ordinary_mixed_batch_skips_speculative_cache_snapshot() -> None:
    indexer = object.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer._pending_speculative_cache = object()
    metadata = SimpleNamespace(
        num_contexts=1,
        num_seqs=2,
        num_tokens=2,
        is_cuda_graph=False,
        qsa_needs_speculative_snapshot=False,
    )
    empty = torch.empty(0, dtype=torch.long)

    indexer._capture_speculative_cache_state(
        layer_idx=0,
        request_indices=empty,
        pages=empty,
        within=empty,
        index_cache=empty,
        position_cache=empty,
        metadata=metadata,
    )

    assert indexer._pending_speculative_cache is None


def test_qsa_multi_token_generation_captures_speculative_cache_snapshot() -> None:
    indexer = object.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer._pending_speculative_cache = None
    metadata = SimpleNamespace(
        num_contexts=0,
        num_seqs=1,
        num_tokens=4,
        is_cuda_graph=False,
        qsa_needs_speculative_snapshot=True,
        seq_lens_cuda=torch.tensor([4], dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(qsa_position_layer_id=0),
    )
    request_indices = torch.zeros(4, dtype=torch.long)
    pages = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    within = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    index_cache = torch.arange(24, dtype=torch.float32).reshape(3, 2, 1, 4)
    position_cache = torch.arange(18, dtype=torch.int32).reshape(3, 2, 3)

    indexer._capture_speculative_cache_state(
        layer_idx=0,
        request_indices=request_indices,
        pages=pages,
        within=within,
        index_cache=index_cache,
        position_cache=position_cache,
        metadata=metadata,
    )

    pending = indexer._pending_speculative_cache
    assert pending is not None
    assert pending["token_ordinals"].tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(pending["index_values"], index_cache[pages, within])
    torch.testing.assert_close(pending["position_values"], position_cache[pages, within])
