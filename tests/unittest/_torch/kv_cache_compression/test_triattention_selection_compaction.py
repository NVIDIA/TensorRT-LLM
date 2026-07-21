# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from conftest import encode_block_offsets as _encode_block_offsets
from conftest import set_protected_tails as _set_protected_tails

from tensorrt_llm._torch.kv_cache_compression.triattention.compaction import (
    BatchedKVCacheCompaction,
)
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    _BatchedPerHeadKeepSetSelector,
    _BatchedUnionKeepSetSelector,
)


def _require_cute_topk_op() -> None:
    """The CuTE TopK operation is a hard prerequisite for these tests."""
    assert hasattr(torch.ops.trtllm, "cute_dsl_indexer_topk_decode"), (
        "CuTE TopK operation is not loaded"
    )


# Tests that launch real scores run the SM100 CuTe score kernel -- the only
# score path -- so they are SM100-only, like the production feature itself.
requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention score requires SM100",
)


def _stable_topk(row: torch.Tensor, width: int, keep_count: int) -> torch.Tensor:
    values = row[:width].tolist()
    selected = sorted(range(width), key=lambda index: (-values[index], index))
    return torch.tensor(selected[:keep_count], dtype=torch.int32, device=row.device)


def _per_head_keep_oracle(
    scores: torch.Tensor,
    valid_widths: torch.Tensor,
    keep_count: int,
    eviction_mode: str,
    normalize_scores: bool,
) -> torch.Tensor:
    """Independent torch implementation of per-head selection."""
    request_count, num_layers, num_query_heads, width = scores.shape
    num_kv_heads = 2
    rows = []
    for request in range(request_count):
        valid_width = int(valid_widths[request])
        valid = scores[request, ..., :valid_width].clone()
        if normalize_scores:
            mean = valid.mean(dim=-1, keepdim=True)
            valid = valid - mean
            std = valid.norm(dim=-1, keepdim=True) / (valid_width**0.5)
            valid = valid / std.clamp_min(1e-6)
        grouped = valid.view(
            num_layers, num_kv_heads, num_query_heads // num_kv_heads, valid_width
        ).amax(dim=2)
        if eviction_mode == "per_head":
            selection = grouped.mean(dim=0)
        else:
            selection = grouped.reshape(num_layers * num_kv_heads, valid_width)
        rows.append(
            torch.stack(
                [torch.sort(_stable_topk(row, valid_width, keep_count)).values for row in selection]
            )
        )
    return torch.stack(rows)


def _legacy_union(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    row_top = {
        int(index)
        for row in scores
        for index in _stable_topk(row, row.numel(), keep_count).tolist()
    }
    combined = scores.max(dim=0).values
    ordered = sorted(
        row_top,
        key=lambda index: (-float(combined[index]), index),
    )
    return torch.tensor(ordered[:keep_count], dtype=torch.long)


@pytest.mark.parametrize("rows,width,keep_count", [(2, 8, 4), (5, 17, 7)])
def test_direct_union_topk_matches_legacy_union_with_heavy_ties(rows, width, keep_count):
    for seed in range(25):
        generator = torch.Generator().manual_seed(seed)
        scores = torch.randint(
            -2,
            3,
            (rows, width),
            generator=generator,
            dtype=torch.int32,
        ).to(torch.float32)
        combined = scores.max(dim=0).values
        direct = _stable_topk(combined, width, keep_count).to(torch.long)
        assert torch.equal(direct, _legacy_union(scores, keep_count))


@pytest.mark.parametrize("eviction_mode", ["per_head", "per_layer_perhead"])
@pytest.mark.parametrize("normalize_scores", [False, True])
def test_per_head_selection_matches_torch_oracle_on_selector_stream(
    eviction_mode, normalize_scores
):
    _require_cute_topk_op()
    request_count, layers, query_heads, kv_heads = 2, 3, 4, 2
    width, keep_count = 96, 64
    generator = torch.Generator().manual_seed(41)
    scores_cpu = torch.randint(
        -4,
        5,
        (request_count, layers, query_heads, width),
        generator=generator,
        dtype=torch.int32,
    ).to(torch.float32)
    valid_widths = torch.tensor([83, 91], dtype=torch.int32)

    expected = _per_head_keep_oracle(
        scores_cpu, valid_widths, keep_count, eviction_mode, normalize_scores
    )

    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        selector = _BatchedPerHeadKeepSetSelector(
            eviction_mode=eviction_mode,
            dense_layers=tuple(range(layers)),
            num_query_heads=query_heads,
            num_kv_heads=kv_heads,
            width=width,
            keep_count=keep_count,
            dtype=torch.float32,
            device=device,
            max_requests=request_count,
        )
        selector.valid_widths.copy_(valid_widths.to(device))
        scores = scores_cpu.to(device)
        selector.select_requests(scores, normalize_scores=normalize_scores)
        first = selector.keep.cpu()
        selector.select_requests(scores, normalize_scores=normalize_scores)
        second = selector.keep.cpu()
    stream.synchronize()

    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


def test_union_eager_runs_the_registered_cute_op():
    _require_cute_topk_op()
    device = torch.device("cuda", torch.cuda.current_device())
    scores = torch.randn(2, 4, 96, dtype=torch.float32, device=device)
    selector = _BatchedUnionKeepSetSelector(
        rows=4,
        width=96,
        keep_count=64,
        dtype=torch.float32,
        device=device,
        max_requests=2,
        input_scores=scores,
        normalize_scores=True,
    )
    selector.select_prepared_requests()
    torch.cuda.synchronize(device)
    assert torch.all(selector.keep[:, 1:] >= selector.keep[:, :-1])


@pytest.mark.parametrize("normalize_scores", [False, True])
def test_prepared_union_scores_match_checked_launch_and_exact_indices(normalize_scores):
    _require_cute_topk_op()
    from tensorrt_llm._torch.kv_cache_compression.triattention import triattention_kernels

    device = torch.device("cuda", torch.cuda.current_device())
    request_count, rows, width, keep_count = 2, 7, 97, 64
    generator = torch.Generator(device=device).manual_seed(53)
    scores = torch.randn(
        request_count,
        rows,
        width,
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    valid_widths = torch.tensor([83, 91], dtype=torch.int32, device=device)
    reference_mean = torch.empty(request_count, rows, 1, dtype=torch.float32, device=device)
    reference_inv_std = torch.empty_like(reference_mean)
    reference_combined = torch.empty(request_count, width, dtype=torch.float32, device=device)
    triattention_kernels.prepare_union_scores(
        scores,
        valid_widths,
        reference_mean,
        reference_inv_std,
        reference_combined,
        request_count,
        normalize_scores=normalize_scores,
    )

    selector = _BatchedUnionKeepSetSelector(
        rows=rows,
        width=width,
        keep_count=keep_count,
        dtype=torch.float32,
        device=device,
        max_requests=request_count,
        input_scores=scores,
        normalize_scores=normalize_scores,
    )
    selector.valid_widths.copy_(valid_widths)
    selector.select_prepared_requests()
    actual_combined = selector.combined.cpu()
    actual_keep = selector.keep.cpu()
    expected_combined = reference_combined.cpu()
    torch.cuda.synchronize(device)

    assert torch.equal(actual_combined, expected_combined)
    for request, valid_width in enumerate(valid_widths.cpu().tolist()):
        expected_keep = torch.sort(
            _stable_topk(expected_combined[request], valid_width, keep_count)
        ).values
        assert torch.equal(actual_keep[request], expected_keep)


@pytest.mark.parametrize("keep_count,width", [(4, 64), (4096, 4224), (8192, 9216)])
def test_union_eager_cuda_resolves_heavy_ties_and_ragged_lengths(keep_count, width):
    _require_cute_topk_op()
    device = torch.device("cuda", torch.cuda.current_device())
    prompt_len = 17
    request_count, rows = 2, 4
    generator = torch.Generator(device=device).manual_seed(keep_count)
    scores = torch.randint(
        -4,
        5,
        (request_count, rows, width),
        generator=generator,
        dtype=torch.int32,
        device=device,
    ).to(torch.float32)
    valid_widths = (width, width - 32)
    selector = _BatchedUnionKeepSetSelector(
        rows=rows,
        width=width,
        keep_count=keep_count,
        dtype=torch.float32,
        device=device,
        max_requests=request_count,
        input_scores=scores,
        normalize_scores=False,
    )
    selector.valid_widths.copy_(torch.tensor(valid_widths, dtype=torch.int32, device=device))
    # Write the shared per-request prompt lengths the way production staging
    # does: fill the bound buffer, then re-expand the row-major view.
    selector.prompt_offsets[:request_count].copy_(
        torch.tensor([prompt_len] * request_count, dtype=torch.int32, device=device)
    )
    selector.refresh_row_prompt_offsets()
    selector.select_prepared_requests()
    actual = selector.keep.cpu()

    combined = scores.amax(dim=1).cpu()
    for request, valid_width in enumerate(valid_widths):
        expected_decode = torch.sort(
            _stable_topk(combined[request], valid_width, keep_count).to(torch.int32) + prompt_len
        ).values
        assert torch.equal(actual[request], expected_decode)


def test_fused_union_preparation_matches_ragged_torch_reference():
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
        prepare_union_scores,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    request_count, rows, width = 2, 7, 97
    generator = torch.Generator(device=device).manual_seed(17)
    scores = torch.randn(
        request_count,
        rows,
        width,
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    valid_widths = torch.tensor([83, 91], dtype=torch.int32, device=device)
    row_mean = torch.empty(request_count, rows, 1, dtype=torch.float32, device=device)
    row_inv_std = torch.empty_like(row_mean)
    combined = torch.empty(request_count, width, device=device)

    prepare_union_scores(
        scores,
        valid_widths,
        row_mean,
        row_inv_std,
        combined,
        request_count,
        normalize_scores=True,
    )
    torch.cuda.synchronize(device)

    expected = torch.full_like(combined, float("-inf"))
    for request, valid_width in enumerate(valid_widths.tolist()):
        valid_scores = scores[request, :, :valid_width]
        mean = valid_scores.mean(dim=1, keepdim=True)
        std = torch.linalg.vector_norm(valid_scores - mean, dim=1, keepdim=True)
        std = (std / valid_width**0.5).clamp_min(1e-6)
        expected[request, :valid_width] = ((valid_scores - mean) / std).amax(dim=0)
    assert torch.allclose(combined, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("per_layer", [False, True])
@pytest.mark.parametrize("normalize_scores", [False, True])
def test_fused_per_head_preparation_matches_ragged_torch_reference(per_layer, normalize_scores):
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
        prepare_per_head_scores,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    request_count, layers, query_heads, kv_heads, width = 2, 3, 4, 2, 97
    generator = torch.Generator(device=device).manual_seed(29)
    scores = torch.randn(
        request_count,
        layers,
        query_heads,
        width,
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    valid_widths = torch.tensor([83, 91], dtype=torch.int32, device=device)
    row_mean = torch.empty(
        request_count, layers, query_heads, 1, dtype=torch.float32, device=device
    )
    row_inv_std = torch.empty_like(row_mean)
    selection_rows = layers * kv_heads if per_layer else kv_heads
    selection_scores = torch.empty(
        request_count, selection_rows, width, dtype=torch.float32, device=device
    )
    selection_seq_lens = torch.empty(
        request_count, selection_rows, dtype=torch.int32, device=device
    )

    prepare_per_head_scores(
        scores,
        valid_widths,
        row_mean,
        row_inv_std,
        selection_scores,
        selection_seq_lens,
        request_count,
        num_kv_heads=kv_heads,
        per_layer=per_layer,
        normalize_scores=normalize_scores,
    )
    torch.cuda.synchronize(device)

    assert torch.equal(
        selection_seq_lens.cpu(),
        valid_widths.cpu().view(request_count, 1).expand(-1, selection_rows),
    )
    query_group_size = query_heads // kv_heads
    for request, valid_width in enumerate(valid_widths.tolist()):
        valid = scores[request, :, :, :valid_width]
        if normalize_scores:
            mean = valid.mean(dim=-1, keepdim=True)
            std = torch.linalg.vector_norm(valid - mean, dim=-1, keepdim=True)
            std = (std / valid_width**0.5).clamp_min(1e-6)
            valid = (valid - mean) / std
        grouped = valid.view(layers, kv_heads, query_group_size, valid_width).amax(dim=2)
        expected = grouped if per_layer else grouped.mean(dim=0)
        expected = expected.reshape(selection_rows, valid_width)
        assert torch.allclose(
            selection_scores[request, :, :valid_width],
            expected,
            rtol=2e-5,
            atol=2e-5,
        )
        assert torch.isneginf(selection_scores[request, :, valid_width:]).all()


@pytest.mark.parametrize("eviction_mode", ["union", "per_head", "per_layer_perhead"])
def test_eager_compaction_preserves_exact_selected_bytes_and_tail(eviction_mode):
    # The compact op ships only the pipelined bf16 kernels: pools use the
    # supported geometry (bf16, 32-token pages, head_dim 64), and the kept
    # ordinals are spread across all three pages per request so the moves
    # still cross page boundaries. The bf16-exact ``arange % 251`` payload
    # keeps every wrong-move byte pattern distinguishable.
    device = torch.device("cuda", torch.cuda.current_device())
    request_count = 2
    num_layers = 2
    num_kv_heads = 2
    prompt_len = 2
    decode_keep_count = 4
    seq_len = 80
    tokens_per_block = 32
    pages_per_request = 3
    head_dim = 64
    protected_tails = [2, 1]
    page_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
    initial_pools = [
        (
            (
                torch.arange(
                    6 * 2 * num_kv_heads * tokens_per_block * head_dim,
                    dtype=torch.int32,
                    device=device,
                )
                + layer * 37
            )
            % 251
        )
        .view(6, 2, num_kv_heads, tokens_per_block, head_dim)
        .to(torch.bfloat16)
        for layer in range(num_layers)
    ]
    pools = [pool.clone() for pool in initial_pools]

    # Kept ordinals are decode-only but hold absolute positions; the pinned
    # prompt tokens never appear in the selection rectangle.
    union_decode = torch.tensor(
        [[16, 32, 56, 72], [24, 40, 48, 64]], dtype=torch.int64, device=device
    )
    if eviction_mode == "union":
        keep = union_decode
        selection_rows = 1
    else:
        selection_rows = num_kv_heads if eviction_mode == "per_head" else num_layers * num_kv_heads
        keep = torch.empty(
            request_count,
            selection_rows,
            decode_keep_count,
            dtype=torch.int64,
            device=device,
        )
        for request in range(request_count):
            for row in range(selection_rows):
                keep[request, row] = torch.tensor(
                    sorted(
                        {
                            2 + ((request + row + offset * 2) % 8) * 8
                            for offset in range(decode_keep_count)
                        }
                    ),
                    dtype=torch.int64,
                    device=device,
                )

    compaction = BatchedKVCacheCompaction(
        eviction_mode=eviction_mode,
        layer_pools=pools,
        dense_layers=[0, 1],
        swa_layers=[],
        layer_group_representative={0: 0, 1: 1},
        layer_pool_keys=[("dense", 0), ("dense", 0)],
        kept_token_ordinals=keep.to(torch.int32),
        valid_sequence_lengths=torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device),
        kv_block_offsets=_encode_block_offsets(page_tables.unsqueeze(0)),
        page_table_slots={0: 0, 1: 0},
        request_count=request_count,
        prompt_offsets=torch.full((request_count,), prompt_len, dtype=torch.int32, device=device),
        decode_keep_count=decode_keep_count,
        swa_window=None,
        protected_tail_capacity=max(protected_tails),
    )
    _set_protected_tails(compaction, protected_tails)
    compaction.compact()
    torch.cuda.synchronize(device)

    for layer, (before_pool, after_pool) in enumerate(zip(initial_pools, pools)):
        for request in range(request_count):
            pages = page_tables[request].to(torch.long)
            before = (
                before_pool[pages]
                .permute(1, 2, 0, 3, 4)
                .reshape(2, num_kv_heads, pages_per_request * tokens_per_block, head_dim)
            )
            after = after_pool[pages].permute(1, 2, 0, 3, 4).reshape_as(before)
            assert torch.equal(after[:, :, :prompt_len], before[:, :, :prompt_len])
            for head in range(num_kv_heads):
                if eviction_mode == "union":
                    selected = keep[request]
                elif eviction_mode == "per_head":
                    selected = keep[request, head]
                else:
                    selected = keep[request, layer * num_kv_heads + head]
                tail = torch.arange(
                    seq_len,
                    seq_len + protected_tails[request],
                    dtype=torch.int64,
                    device=device,
                )
                source = torch.cat((selected, tail))
                destination = torch.arange(
                    prompt_len,
                    prompt_len + source.numel(),
                    dtype=torch.int64,
                    device=device,
                )
                assert torch.equal(
                    after[:, head].index_select(1, destination),
                    before[:, head].index_select(1, source),
                )


def test_union_mixed_prompt_lengths_cohort_matches_single_request_compactions():
    """One union cohort mixing prompt lengths compacts byte-identically to
    running the same two requests as two single-request compactions."""
    device = torch.device("cuda", torch.cuda.current_device())
    request_count = 2
    num_layers = 2
    num_kv_heads = 2
    # bf16 pools in the compact op's supported geometry; three 32-token pages
    # per request with a 80-token sequence keep the moves page-crossing.
    seq_len = 80
    decode_keep_count = 3
    prompt_lens = [2, 5]
    protected_tails = [2, 1]
    tokens_per_block = 32
    head_dim = 64
    decode_widths = [seq_len - prompt_len for prompt_len in prompt_lens]
    width = max(decode_widths)

    # Oracle selection: decode-relative scores per request, rebased to
    # absolute ordinals by each request's own prompt offset.
    generator = torch.Generator().manual_seed(11)
    scores = torch.randint(
        -8,
        9,
        (request_count, 1, width),
        generator=generator,
        dtype=torch.int32,
    ).to(torch.float32)
    keep = torch.stack(
        [
            torch.sort(
                _stable_topk(scores[request, 0], decode_width, decode_keep_count) + prompt_len
            ).values
            for request, (prompt_len, decode_width) in enumerate(zip(prompt_lens, decode_widths))
        ]
    )

    page_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
    initial_pools = [
        (
            (
                torch.arange(
                    6 * 2 * num_kv_heads * tokens_per_block * head_dim,
                    dtype=torch.int32,
                    device=device,
                )
                + layer * 37
            )
            % 251
        )
        .view(6, 2, num_kv_heads, tokens_per_block, head_dim)
        .to(torch.bfloat16)
        for layer in range(num_layers)
    ]
    cohort_pools = [pool.clone() for pool in initial_pools]
    keep_cuda = keep.to(device)
    valid_seq_lens = torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device)
    cohort_compaction = BatchedKVCacheCompaction(
        eviction_mode="union",
        layer_pools=cohort_pools,
        dense_layers=[0, 1],
        swa_layers=[],
        layer_group_representative={0: 0, 1: 1},
        layer_pool_keys=[("dense", 0), ("dense", 0)],
        kept_token_ordinals=keep_cuda,
        valid_sequence_lengths=valid_seq_lens,
        kv_block_offsets=_encode_block_offsets(page_tables.unsqueeze(0)),
        page_table_slots={0: 0, 1: 0},
        request_count=request_count,
        prompt_offsets=torch.tensor(prompt_lens, dtype=torch.int32, device=device),
        decode_keep_count=decode_keep_count,
        swa_window=None,
        protected_tail_capacity=max(protected_tails),
    )
    _set_protected_tails(cohort_compaction, protected_tails)
    cohort_compaction.compact()

    expected_pools = [pool.clone() for pool in initial_pools]
    for request in range(request_count):
        single_compaction = BatchedKVCacheCompaction(
            eviction_mode="union",
            layer_pools=expected_pools,
            dense_layers=[0, 1],
            swa_layers=[],
            layer_group_representative={0: 0, 1: 1},
            layer_pool_keys=[("dense", 0), ("dense", 0)],
            kept_token_ordinals=keep_cuda[request : request + 1],
            valid_sequence_lengths=valid_seq_lens[request : request + 1],
            kv_block_offsets=_encode_block_offsets(page_tables[request : request + 1].unsqueeze(0)),
            page_table_slots={0: 0, 1: 0},
            request_count=1,
            prompt_offsets=torch.tensor([prompt_lens[request]], dtype=torch.int32, device=device),
            decode_keep_count=decode_keep_count,
            swa_window=None,
            protected_tail_capacity=protected_tails[request],
        )
        _set_protected_tails(single_compaction, [protected_tails[request]])
        single_compaction.compact()
    torch.cuda.synchronize(device)

    # The two requests own disjoint pages, so whole-pool equality proves the
    # cohort produced exactly the two single-request results.
    for cohort_pool, expected_pool in zip(cohort_pools, expected_pools):
        assert torch.equal(cohort_pool, expected_pool)


@requires_sm100
def test_per_layer_score_selection_and_compaction_preserve_dense_layer_order():
    """Keep score and compaction layer axes aligned across interleaved V2 pools."""
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        _FixedScoreStagingBuffers,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    num_layers = 3
    # The staged bucket capacity must be aligned to the score kernel's
    # 64-token compute tile; the request itself stays 8 tokens long.
    bucket_capacity = 64
    seq_len = 8
    keep_count = 2
    # GQA group 8 (the smallest CuTe-supported group with one KV head); all
    # query heads share one zero calibration query and one MLR coefficient,
    # so every head row carries the same |K|-driven score.
    num_q_heads = 8
    # bf16 pools in the compact op's supported geometry. The scored tokens
    # all live in each table's first entry, but the two tables still map the
    # two storage groups onto different physical pages, which is what the
    # layer-order alignment below depends on.
    tokens_per_block = 32
    head_dim = 64
    num_freqs = head_dim // 2
    dense_layers = [0, 1, 2]
    dense_groups = [[0, 2], [1]]
    layer_group_representative = {0: 0, 1: 1, 2: 0}
    page_tables = (
        torch.tensor([[1, 0]], dtype=torch.int32, device=device),
        torch.tensor([[0, 1]], dtype=torch.int32, device=device),
    )
    layer_tables = (page_tables[0], page_tables[1], page_tables[0])
    score_values = (
        (1, 8, 2, 3, 4, 5, 9, 6),
        (2, 3, 8, 4, 5, 9, 6, 7),
        (3, 4, 5, 9, 6, 7, 8, 10),
    )
    expected_keep = torch.tensor([[[1, 6], [2, 5], [3, 7]]], dtype=torch.int32, device=device)

    pools = []
    for layer, (table, values) in enumerate(zip(layer_tables, score_values)):
        pool = (
            (
                (
                    torch.arange(
                        2 * 2 * tokens_per_block * head_dim,
                        dtype=torch.int32,
                        device=device,
                    )
                    + layer * 37
                )
                % 251
            )
            .view(2, 2, 1, tokens_per_block, head_dim)
            .to(torch.bfloat16)
        )
        for token, value in enumerate(values):
            page = int(table[0, token // tokens_per_block])
            slot = token % tokens_per_block
            pool[page, 0, 0, slot, 0] = value
            pool[page, 0, 0, slot, num_freqs] = 0
        pools.append(pool)
    initial_pools = [pool.clone() for pool in pools]

    q_real = torch.zeros(num_layers, num_q_heads, num_freqs, dtype=torch.float32, device=device)
    q_imag = torch.zeros_like(q_real)
    mlr_coef = torch.zeros_like(q_real)
    mlr_coef[:, :, 0] = 1
    freq_scale_sq = torch.zeros(num_freqs, dtype=torch.float32, device=device)
    freq_scale_sq[0] = 1
    score_staging = _FixedScoreStagingBuffers(
        pools,
        dense_groups,
        dense_layers,
        [0, 1],
        1,
        bucket_capacity,
        num_q_heads,
        num_freqs,
        q_real,
        q_imag,
        mlr_coef,
        freq_scale_sq,
        torch.zeros(1, dtype=torch.float32, device=device),
        torch.zeros(num_freqs, dtype=torch.float32, device=device),
        page_table_keys=[("pool", 0), ("pool", 1)],
        num_page_table_slots=2,
    )
    score_staging.block_offsets_device.zero_()
    score_staging.block_offsets_device[..., :2].copy_(
        _encode_block_offsets(torch.stack(page_tables))
    )
    score_staging.round_starts_device.fill_(0)
    score_staging.valid_seq_lens_device.fill_(seq_len)
    score_staging.token_starts_device.fill_(0)
    keep_set_selector = _BatchedPerHeadKeepSetSelector(
        eviction_mode="per_layer_perhead",
        dense_layers=tuple(dense_layers),
        num_query_heads=num_q_heads,
        num_kv_heads=1,
        width=bucket_capacity,
        keep_count=keep_count,
        dtype=torch.float32,
        device=device,
        max_requests=1,
    )
    score_staging.bind_score_launcher(keep_set_selector.valid_widths, "mean")

    scores = score_staging.launch_prepared_score()
    keep_set_selector.select_requests(scores, normalize_scores=False)
    assert torch.equal(keep_set_selector.keep, expected_keep)

    batched_compaction = BatchedKVCacheCompaction(
        eviction_mode="per_layer_perhead",
        layer_pools=pools,
        dense_layers=dense_layers,
        swa_layers=[],
        layer_group_representative=layer_group_representative,
        layer_pool_keys=[("pool", 0), ("pool", 1), ("pool", 0)],
        kept_token_ordinals=keep_set_selector.keep[:1],
        valid_sequence_lengths=score_staging.valid_seq_lens_device[:1],
        kv_block_offsets=score_staging.block_offsets_device,
        page_table_slots=score_staging.representative_slots,
        request_count=1,
        prompt_offsets=torch.zeros(1, dtype=torch.int32, device=device),
        decode_keep_count=keep_count,
        swa_window=None,
        protected_tail_capacity=0,
    )
    _set_protected_tails(batched_compaction, [0])
    batched_compaction.compact()
    torch.cuda.synchronize(device)

    for layer, (before_pool, after_pool, table) in enumerate(
        zip(initial_pools, pools, layer_tables)
    ):
        pages = table[0].to(torch.long)
        # The logical view spans both pages (2 * tokens_per_block slots); the
        # scored sequence occupies its first seq_len positions.
        before = before_pool[pages].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, head_dim)
        after = after_pool[pages].permute(1, 2, 0, 3, 4).reshape_as(before)
        selected = expected_keep[0, layer].to(torch.long)
        assert torch.equal(after[:, :, :keep_count], before.index_select(2, selected))


@requires_sm100
def test_union_two_rounds_preserve_bytes_tail_and_v2_page_reuse():
    """Run two real eviction rounds through one live V2 cache.

    The cache uses the score and compact kernels' supported geometry (bf16,
    32-token pages, head_dim 64): the request spans three pages so that
    compacting to two pages still releases one physical page for reuse.
    Token scores are tracked in a host-side mirror and the expected keep
    sets are derived from it, replacing the hand-written score tables of the
    old 4-token-page fixture.
    """
    pytest.importorskip("cutlass")
    import tensorrt_llm
    import tensorrt_llm.bindings
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        _FixedScoreStagingBuffers,
    )
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    device = torch.device("cuda", torch.cuda.current_device())
    request_id = 7
    prompt_len = 2
    # The bucket capacity equals the confirmed length here and must be
    # aligned to the score kernel's 64-token compute tile; the protected
    # tail rides beyond it.
    seq_len = 64
    protected_tail = 2
    compacted_capacity = 36
    tokens_per_block = 32
    head_dim = 64
    num_freqs = head_dim // 2
    keep_count = compacted_capacity - prompt_len - protected_tail
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=seq_len + protected_tail,
            enable_block_reuse=False,
            host_cache_size=0,
            max_util_for_resume=1.0,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=seq_len + protected_tail,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.BF16,
        vocab_size=128,
    )

    requests = []
    temporary_requests = []
    try:
        created = manager.add_dummy_requests(
            [request_id],
            [seq_len + protected_tail],
        )
        assert created is not None
        requests = created
        cache = manager.kv_cache_map[request_id]
        assert cache.resize(seq_len + protected_tail, prompt_len)
        manager.kv_compression_manages_history = True
        pool = manager.get_buffers(0, kv_layout="HND")

        def page_ids(owner: int) -> torch.Tensor:
            return torch.tensor(
                manager.get_batch_cache_indices([owner])[0],
                dtype=torch.long,
                device=device,
            )

        def snapshot(length: int) -> torch.Tensor:
            pages = page_ids(request_id)
            return (
                pool.index_select(0, pages)
                .permute(1, 2, 0, 3, 4)
                .reshape(2, 1, -1, head_dim)[:, :, :length]
                .clone()
            )

        def write_token(token: int, score: float) -> None:
            pages = page_ids(request_id)
            page = pages[token // tokens_per_block]
            offset = token % tokens_per_block
            # Shifted mod-251 ramp: bf16-exact and distinct per token, so the
            # byte comparisons below catch any wrong move.
            payload = (
                ((torch.arange(2 * head_dim, dtype=torch.int32, device=device) + token * 37) % 251)
                .reshape(2, head_dim)
                .to(torch.bfloat16)
            )
            payload[0, 0] = score
            payload[0, num_freqs] = 0
            pool[page, :, 0, offset].copy_(payload)

        # Host-side mirror of each physical position's score; expected keep
        # sets are derived from it. Scores are distinct within the decode
        # window (7 is invertible mod 64 and the window spans one residue
        # cycle), so the selection is tie-free and deterministic.
        token_scores = [0] * (seq_len + protected_tail)
        for token in range(seq_len + protected_tail):
            token_scores[token] = (token * 7) % 64 + 1
            write_token(token, token_scores[token])

        def expected_keep() -> torch.Tensor:
            decode = token_scores[prompt_len:seq_len]
            order = sorted(range(len(decode)), key=lambda index: (-decode[index], index))
            return torch.tensor(
                sorted(prompt_len + index for index in order[:keep_count]),
                dtype=torch.long,
                device=device,
            )

        # GQA group 8 (the smallest CuTe-supported group with one KV head);
        # zero calibration query and a shared MLR coefficient give every
        # query head the same |K|-driven score.
        num_q_heads = 8
        q_real = torch.zeros(1, num_q_heads, num_freqs, dtype=torch.float32, device=device)
        q_imag = torch.zeros_like(q_real)
        mlr_coef = torch.zeros_like(q_real)
        mlr_coef[..., 0] = 1
        freq_scale_sq = torch.zeros(num_freqs, dtype=torch.float32, device=device)
        freq_scale_sq[0] = 1
        score_staging = _FixedScoreStagingBuffers(
            [pool],
            [[0]],
            [0],
            [0],
            1,
            seq_len,
            num_q_heads,
            num_freqs,
            q_real,
            q_imag,
            mlr_coef,
            freq_scale_sq,
            torch.zeros(1, dtype=torch.float32, device=device),
            torch.zeros(num_freqs, dtype=torch.float32, device=device),
            page_table_keys=[("pool", 0)],
            num_page_table_slots=1,
            decode_width=seq_len - prompt_len,
            page_table_token_capacity=seq_len + protected_tail,
        )
        keep_set_selector = _BatchedUnionKeepSetSelector(
            rows=num_q_heads,
            width=seq_len - prompt_len,
            keep_count=keep_count,
            dtype=torch.float32,
            device=device,
            max_requests=1,
            dense_layers=(0,),
            num_query_heads=num_q_heads,
            num_kv_heads=1,
            input_scores=score_staging.fused_group.output.view(
                1, num_q_heads, seq_len - prompt_len
            ),
            normalize_scores=False,
            prompt_offsets_buffer=score_staging.token_starts_device,
        )
        score_staging.bind_score_launcher(keep_set_selector.valid_widths, "mean")
        batched_compaction = BatchedKVCacheCompaction(
            eviction_mode="union",
            layer_pools=[pool],
            dense_layers=[0],
            swa_layers=[],
            layer_group_representative={0: 0},
            layer_pool_keys=[("pool", 0)],
            kept_token_ordinals=keep_set_selector.keep[:1],
            valid_sequence_lengths=score_staging.valid_seq_lens_device[:1],
            kv_block_offsets=score_staging.block_offsets_device,
            page_table_slots=score_staging.representative_slots,
            request_count=1,
            prompt_offsets=score_staging.token_starts_device[:1],
            decode_keep_count=keep_count,
            swa_window=None,
            protected_tail_capacity=protected_tail,
        )
        _set_protected_tails(batched_compaction, [protected_tail])

        def evict_once() -> tuple[torch.Tensor, torch.Tensor]:
            before = snapshot(seq_len + protected_tail)
            assert score_staging.stage(
                manager,
                [request_id],
                [0],
                [prompt_len],
                [seq_len],
                [seq_len + protected_tail],
            )
            keep_set_selector.refresh_row_prompt_offsets()
            score_staging.launch_prepared_score()
            keep_set_selector.select_prepared_requests()
            selected = keep_set_selector.keep[0].clone().to(torch.long)
            batched_compaction.compact()
            score_staging.mark_page_tables_consumed(manager._stream)
            torch.cuda.synchronize(device)
            assert cache.resize(compacted_capacity, None)
            after = snapshot(compacted_capacity)
            source = torch.cat(
                (
                    selected,
                    torch.arange(seq_len, seq_len + protected_tail, device=device),
                )
            )
            assert torch.equal(after[:, :, :prompt_len], before[:, :, :prompt_len])
            assert torch.equal(
                after[:, :, prompt_len:],
                before.index_select(2, source),
            )
            assert cache.capacity == compacted_capacity
            assert cache.history_length == prompt_len
            return selected, after

        initial_pages = page_ids(request_id)
        expected_first_keep = expected_keep()
        first_keep, first_compacted = evict_once()
        assert torch.equal(first_keep, expected_first_keep)
        # The compacted cache spans two of the original three pages.
        retained_pages = page_ids(request_id)
        assert torch.equal(retained_pages, initial_pages[:2])
        released_page = initial_pages[2:]

        created = manager.add_dummy_requests([8], [tokens_per_block])
        assert created is not None
        temporary_requests = created
        assert torch.equal(page_ids(8), released_page)
        manager.free_resources(temporary_requests[0])
        temporary_requests = []

        assert cache.resize(seq_len + protected_tail, None)
        assert cache.history_length == prompt_len
        assert torch.equal(page_ids(request_id)[:2], retained_pages)
        assert torch.equal(page_ids(request_id)[2:], released_page)
        # The first protected tail becomes confirmed input to round two. Only
        # later generated tokens and the next protected tail are written
        # here. Mirror the physical relayout, then give the fresh tokens a
        # disjoint higher score band (11 invertible mod 64 over a shorter
        # window) so round two must select differently from round one.
        survivors = list(range(prompt_len)) + first_keep.tolist() + [seq_len, seq_len + 1]
        token_scores[:compacted_capacity] = [token_scores[source] for source in survivors]
        for token in range(compacted_capacity, seq_len + protected_tail):
            token_scores[token] = (token * 11) % 64 + 100
        assert torch.equal(snapshot(compacted_capacity), first_compacted)
        for token in range(compacted_capacity, seq_len + protected_tail):
            write_token(token, token_scores[token])

        expected_second_keep = expected_keep()
        second_keep, _ = evict_once()
        assert torch.equal(second_keep, expected_second_keep)
        assert not torch.equal(second_keep, first_keep)

        created = manager.add_dummy_requests([9], [tokens_per_block])
        assert created is not None
        temporary_requests = created
        assert torch.equal(page_ids(9), released_page)
    finally:
        for request in temporary_requests:
            manager.free_resources(request)
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()


def test_eager_compaction_rebases_masked_swa_window_and_tail():
    # bf16 pools in the compact op's supported geometry (32-token pages,
    # head_dim 64); the kept ordinals and valid lengths span all three pages
    # per request so the dense and SWA moves stay page-crossing.
    device = torch.device("cuda", torch.cuda.current_device())
    dense_tables = torch.tensor([[2, 0, 1], [5, 3, 4]], dtype=torch.int32, device=device)
    swa_tables = torch.tensor([[1, 2, 0], [4, 5, 3]], dtype=torch.int32, device=device)
    initial_pools = [
        ((torch.arange(6 * 2 * 1 * 32 * 64, dtype=torch.int32, device=device) + layer * 37) % 251)
        .view(6, 2, 1, 32, 64)
        .to(torch.bfloat16)
        for layer in range(2)
    ]
    pools = [pool.clone() for pool in initial_pools]
    # Decode-only kept ordinals holding absolute positions past the prompt.
    keep = torch.tensor(
        [[16, 32, 40, 56], [16, 24, 40, 48]],
        dtype=torch.int64,
        device=device,
    )
    valid_seq_lens = torch.tensor([64, 56], dtype=torch.int32, device=device)
    protected_tails = [2, 1]
    compaction = BatchedKVCacheCompaction(
        eviction_mode="union",
        layer_pools=pools,
        dense_layers=[0],
        swa_layers=[1],
        layer_group_representative={0: 0},
        layer_pool_keys=[("dense", 0), ("swa", 0)],
        kept_token_ordinals=keep.to(torch.int32),
        valid_sequence_lengths=valid_seq_lens,
        kv_block_offsets=_encode_block_offsets(torch.stack((dense_tables, swa_tables))),
        page_table_slots={0: 0, 1: 1},
        request_count=2,
        prompt_offsets=torch.tensor([2, 2], dtype=torch.int32, device=device),
        decode_keep_count=4,
        swa_window=2,
        protected_tail_capacity=max(protected_tails),
    )
    _set_protected_tails(compaction, protected_tails)
    compaction.compact()
    torch.cuda.synchronize(device)

    for request, (valid_seq_len, tail_length) in enumerate(
        zip(valid_seq_lens.tolist(), protected_tails)
    ):
        dense_pages = dense_tables[request].to(torch.long)
        swa_pages = swa_tables[request].to(torch.long)
        dense_before = initial_pools[0][dense_pages].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 64)
        dense_after = pools[0][dense_pages].permute(1, 2, 0, 3, 4).reshape_as(dense_before)
        swa_before = initial_pools[1][swa_pages].permute(1, 2, 0, 3, 4).reshape(2, 1, -1, 64)
        swa_after = pools[1][swa_pages].permute(1, 2, 0, 3, 4).reshape_as(swa_before)
        tail = torch.arange(
            valid_seq_len,
            valid_seq_len + tail_length,
            dtype=torch.int64,
            device=device,
        )
        dense_source = torch.cat((keep[request], tail))
        dense_destination = torch.arange(
            2, 2 + dense_source.numel(), dtype=torch.int64, device=device
        )
        swa_source = torch.arange(
            valid_seq_len - 2,
            valid_seq_len + tail_length,
            dtype=torch.int64,
            device=device,
        )
        swa_destination = torch.arange(4, 4 + swa_source.numel(), dtype=torch.int64, device=device)
        assert torch.equal(dense_after[:, :, :2], dense_before[:, :, :2])
        assert torch.equal(swa_after[:, :, :2], swa_before[:, :, :2])
        assert torch.equal(
            dense_after.index_select(2, dense_destination),
            dense_before.index_select(2, dense_source),
        )
        assert torch.equal(
            swa_after.index_select(2, swa_destination),
            swa_before.index_select(2, swa_source),
        )


def test_cache_families_read_the_staged_move_offsets_rows():
    """Every cache family must consume the caller-staged offsets row.

    A family that silently falls back to its construction-time offsets
    compacts request slots that are not in the staged cohort; on
    sliding-window models the padded slots then produce negative source
    ordinals and an illegal memory access. Binding the staged row by
    reference is part of the constructor contract.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    dense_tables = torch.tensor([[2, 0, 1], [5, 3, 4]], dtype=torch.int32, device=device)
    swa_tables = torch.tensor([[1, 2, 0], [4, 5, 3]], dtype=torch.int32, device=device)
    # bf16 pools in the compact op's supported geometry; this test only
    # constructs the compaction, so the contents stay zero.
    pools = [
        torch.zeros(6, 2, 1, 32, 64, dtype=torch.bfloat16, device=device),
        torch.zeros(6, 2, 1, 32, 64, dtype=torch.bfloat16, device=device),
    ]
    keep = torch.tensor([[2, 4, 5, 7], [2, 3, 5, 6]], dtype=torch.int32, device=device)
    staged_rows = torch.zeros(2, 3, dtype=torch.int32, device=device)
    dense_offsets_row = staged_rows[0]
    swa_offsets_row = staged_rows[1]
    compaction = BatchedKVCacheCompaction(
        eviction_mode="union",
        layer_pools=pools,
        dense_layers=[0],
        swa_layers=[1],
        layer_group_representative={0: 0},
        layer_pool_keys=[("dense", 0), ("swa", 0)],
        kept_token_ordinals=keep,
        valid_sequence_lengths=torch.tensor([8, 7], dtype=torch.int32, device=device),
        kv_block_offsets=_encode_block_offsets(torch.stack((dense_tables, swa_tables))),
        page_table_slots={0: 0, 1: 1},
        request_count=2,
        prompt_offsets=torch.tensor([2, 2], dtype=torch.int32, device=device),
        decode_keep_count=4,
        swa_window=2,
        protected_tail_capacity=2,
        dense_move_offsets=dense_offsets_row,
        swa_move_offsets=swa_offsets_row,
    )
    assert (
        compaction.target_dense_compaction.move_source_offsets.data_ptr()
        == dense_offsets_row.data_ptr()
    )
    assert compaction.target_swa_compaction is not None
    assert (
        compaction.target_swa_compaction.move_source_offsets.data_ptr()
        == swa_offsets_row.data_ptr()
    )
