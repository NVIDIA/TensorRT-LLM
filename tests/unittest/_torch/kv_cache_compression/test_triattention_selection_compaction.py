# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from conftest import make_cute_buffers as _make_cute_buffers
from conftest import make_eviction_request as _make_eviction_request
from conftest import make_ramp_pools as _make_ramp_pools
from conftest import make_request as _make_request
from conftest import make_staging_manager as _make_staging_manager
from conftest import rect_to_score_scratch as _rect_to_score_scratch

from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    TriAttentionCompressionManager,
)
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
    reduce_per_head_scores,
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


def _make_selection_buffers(
    *,
    eviction_mode,
    width,
    keep_count,
    device,
    max_requests,
    num_layers=1,
    num_query_heads=1,
    num_kv_heads=1,
):
    """Allocate the production selection buffers without score or compaction."""
    tri = TriAttentionCompressionManager.__new__(TriAttentionCompressionManager)
    tri.eviction_mode = eviction_mode
    tri._request_capacity = max_requests
    tri._selection_width_capacity = width
    tri.budget = keep_count
    tri._num_layers = num_layers
    tri._num_q_heads = num_query_heads
    tri._num_kv_heads = num_kv_heads
    tri._prompt_lengths_device = torch.zeros(max_requests, dtype=torch.int32, device=device)
    tri._allocate_selection_buffers(device, tp_size=1)
    return tri


def _select_per_head(tri, scores, *, normalize_scores):
    """The per-head selection flow: reduce kernels, then top-k settle."""
    request_count, _, _, width = scores.shape
    score_scratch, prompt_lengths = _rect_to_score_scratch(scores, tri._num_kv_heads)
    reduce_per_head_scores(
        score_scratch,
        tri._decode_lengths_device,
        prompt_lengths,
        tri._row_mean,
        tri._row_inv_std,
        tri._selection_scores_rows,
        tri._selection_row_lengths,
        request_count=request_count,
        padded_head_columns=8,
        score_token_capacity=width,
        per_layer=tri.eviction_mode == "per_layer_perhead",
        normalize_scores=normalize_scores,
    )
    tri._select_kept_ordinals(tri._request_capacity)


def _stable_topk(row: torch.Tensor, width: int, keep_count: int) -> torch.Tensor:
    values = row[:width].tolist()
    selected = sorted(range(width), key=lambda index: (-values[index], index))
    return torch.tensor(selected[:keep_count], dtype=torch.int32, device=row.device)


def _per_head_keep_oracle(
    scores: torch.Tensor,
    decode_lengths: torch.Tensor,
    keep_count: int,
    eviction_mode: str,
    normalize_scores: bool,
) -> torch.Tensor:
    """Independent torch implementation of per-head selection."""
    request_count, num_layers, num_query_heads, width = scores.shape
    num_kv_heads = 2
    rows = []
    for request in range(request_count):
        decode_length = int(decode_lengths[request])
        valid = scores[request, ..., :decode_length].clone()
        if normalize_scores:
            mean = valid.mean(dim=-1, keepdim=True)
            valid = valid - mean
            std = valid.norm(dim=-1, keepdim=True) / (decode_length**0.5)
            valid = valid / std.clamp_min(1e-6)
        grouped = valid.view(
            num_layers, num_kv_heads, num_query_heads // num_kv_heads, decode_length
        ).amax(dim=2)
        if eviction_mode == "per_head":
            selection = grouped.mean(dim=0)
        else:
            selection = grouped.reshape(num_layers * num_kv_heads, decode_length)
        rows.append(
            torch.stack(
                [
                    torch.sort(_stable_topk(row, decode_length, keep_count)).values
                    for row in selection
                ]
            )
        )
    return torch.stack(rows)


@pytest.mark.parametrize(
    "eviction_mode,normalize_scores",
    [("per_head", True), ("per_layer_perhead", False)],
)
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
    decode_lengths = torch.tensor([83, 91], dtype=torch.int32)

    expected = _per_head_keep_oracle(
        scores_cpu, decode_lengths, keep_count, eviction_mode, normalize_scores
    )

    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        tri = _make_selection_buffers(
            eviction_mode=eviction_mode,
            width=width,
            keep_count=keep_count,
            device=device,
            max_requests=request_count,
            num_layers=layers,
            num_query_heads=query_heads,
            num_kv_heads=kv_heads,
        )
        tri._decode_lengths_device.copy_(decode_lengths.to(device))
        scores = scores_cpu.to(device)
        keep_shape = (request_count, tri._selection_rows_per_request, keep_count)
        _select_per_head(tri, scores, normalize_scores=normalize_scores)
        first = tri._kept_ordinal_rows.view(keep_shape).cpu()
        _select_per_head(tri, scores, normalize_scores=normalize_scores)
        second = tri._kept_ordinal_rows.view(keep_shape).cpu()
    stream.synchronize()

    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


@pytest.mark.parametrize("keep_count,width", [(4, 64), (8192, 9216)])
def test_union_eager_cuda_resolves_heavy_ties_and_ragged_lengths(keep_count, width):
    # Tied integer scores, ragged widths, per-request prompt rebase; the
    # 8192-keep row is the large-k coverage.
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
    decode_lengths = (width, width - 32)
    tri = _make_selection_buffers(
        eviction_mode="union",
        width=width,
        keep_count=keep_count,
        device=device,
        max_requests=request_count,
    )
    tri._decode_lengths_device.copy_(torch.tensor(decode_lengths, dtype=torch.int32, device=device))
    tri._prompt_lengths_device[:request_count].copy_(
        torch.tensor([prompt_len] * request_count, dtype=torch.int32, device=device)
    )
    tri._selection_scores_rows.copy_(scores.amax(dim=1))
    tri._select_kept_ordinals(tri._request_capacity)
    actual = tri._kept_ordinal_rows.cpu()

    combined = scores.amax(dim=1).cpu()
    for request, decode_length in enumerate(decode_lengths):
        expected_decode = torch.sort(
            _stable_topk(combined[request], decode_length, keep_count).to(torch.int32) + prompt_len
        ).values
        assert torch.equal(actual[request], expected_decode)


@pytest.mark.parametrize("per_layer", [False, True])
@pytest.mark.parametrize("normalize_scores", [False, True])
def test_per_head_reduction_matches_ragged_torch_reference(per_layer, normalize_scores):
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
    decode_lengths = torch.tensor([83, 91], dtype=torch.int32, device=device)
    row_mean = torch.empty(
        request_count, layers, query_heads, 1, dtype=torch.float32, device=device
    )
    row_inv_std = torch.empty_like(row_mean)
    selection_rows = layers * kv_heads if per_layer else kv_heads
    # Canonical row-major buffers, exactly like the product allocation.
    selection_scores_rows = torch.empty(
        request_count * selection_rows, width, dtype=torch.float32, device=device
    )
    selection_row_lengths = torch.empty(
        request_count * selection_rows, dtype=torch.int32, device=device
    )

    score_scratch, prompt_lengths = _rect_to_score_scratch(scores, kv_heads)
    reduce_per_head_scores(
        score_scratch,
        decode_lengths,
        prompt_lengths,
        row_mean,
        row_inv_std,
        selection_scores_rows,
        selection_row_lengths,
        request_count=request_count,
        padded_head_columns=8,
        score_token_capacity=width,
        per_layer=per_layer,
        normalize_scores=normalize_scores,
    )
    torch.cuda.synchronize(device)

    selection_scores = selection_scores_rows.view(request_count, selection_rows, width)
    assert torch.equal(
        selection_row_lengths.view(request_count, selection_rows).cpu(),
        decode_lengths.cpu().view(request_count, 1).expand(-1, selection_rows),
    )
    query_group_size = query_heads // kv_heads
    for request, decode_length in enumerate(decode_lengths.tolist()):
        valid = scores[request, :, :, :decode_length]
        if normalize_scores:
            mean = valid.mean(dim=-1, keepdim=True)
            std = torch.linalg.vector_norm(valid - mean, dim=-1, keepdim=True)
            std = (std / decode_length**0.5).clamp_min(1e-6)
            valid = (valid - mean) / std
        grouped = valid.view(layers, kv_heads, query_group_size, decode_length).amax(dim=2)
        expected = grouped if per_layer else grouped.mean(dim=0)
        expected = expected.reshape(selection_rows, decode_length)
        assert torch.allclose(
            selection_scores[request, :, :decode_length],
            expected,
            rtol=2e-5,
            atol=2e-5,
        )
        assert torch.isneginf(selection_scores[request, :, decode_length:]).all()


@requires_sm100
def test_per_layer_score_selection_and_compaction_preserve_dense_layer_order():
    """Keep score and compaction layer axes aligned across interleaved V2 pools."""
    pytest.importorskip("cutlass")

    device = torch.device("cuda", torch.cuda.current_device())
    num_layers = 3
    # Bucket capacity aligned to the 64-token compute tile; request stays 8.
    bucket_capacity = 64
    seq_len = 8
    keep_count = 2
    # GQA group 8, zero calibration query, shared MLR: every head row
    # carries the same |K|-driven score.
    num_q_heads = 8
    # The two tables map the storage groups onto different physical pages;
    # the layer-order alignment below depends on it.
    tokens_per_block = 32
    head_dim = 64
    num_freqs = head_dim // 2
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

    pools = _make_ramp_pools(num_layers, num_kv_heads=1, pages=2, device=device)
    for pool, (table, values) in zip(pools, zip(layer_tables, score_values)):
        for token, value in enumerate(values):
            page = int(table[0, token // tokens_per_block])
            slot = token % tokens_per_block
            pool[page, 0, 0, slot, 0] = value
            pool[page, 0, 0, slot, num_freqs] = 0
    initial_pools = [pool.clone() for pool in pools]

    q_real = torch.zeros(num_layers, num_q_heads, num_freqs, dtype=torch.float32, device=device)
    q_imag = torch.zeros_like(q_real)
    mlr_coef = torch.zeros_like(q_real)
    mlr_coef[:, :, 0] = 1
    freq_scale_sq = torch.zeros(num_freqs, dtype=torch.float32, device=device)
    freq_scale_sq[0] = 1
    tri = _make_cute_buffers(
        eviction_mode="per_layer_perhead",
        layer_pools=pools,
        max_requests=1,
        seq_len=bucket_capacity,
        num_q_heads=num_q_heads,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        omega=torch.zeros(num_freqs, dtype=torch.float32, device=device),
        offsets=torch.zeros(1, dtype=torch.float32, device=device),
        decode_width=bucket_capacity,
        keep_count=keep_count,
        layer_pool_ids=[0, 1, 0],
        normalize_scores=False,
    )
    # No SWA in this layout: no window, no rebase row for the phase gather.
    assert tri._swa_window is None
    assert tri._swa_destination_bases is None
    # Native V2 staging contract: [pool, request, K/V, block] int32 pair with
    # a 4-aligned block width (PackedInt copy ABI) and a pinned host snapshot.
    assert tri._block_offsets_host.shape == tri._block_offsets_device.shape
    assert tri._block_offsets_host.shape[:3] == (2, 1, 2)
    assert tri._block_offsets_host.shape[-1] % 4 == 0
    assert tri._block_offsets_host.dtype == tri._block_offsets_device.dtype == torch.int32
    assert tri._block_offsets_host.is_contiguous() and tri._block_offsets_device.is_contiguous()
    assert tri._block_offsets_host.is_pinned()

    # Stage through the round executor: the gather double writes both
    # page-table slots' K page ids and the bulk copy encodes the K/V rows;
    # the derived move offsets stage the buffers' own contract (keep_count
    # moves per request, no protected tail).
    def gather_k_block_offsets(host_table, source, request_ids, num_blocks):
        assert request_ids == [7]
        source[..., 0, :].zero_()
        source[0, 0, 0, :2].copy_(page_tables[0][0].cpu())
        source[1, 0, 0, :2].copy_(page_tables[1][0].cpu())

    manager = _make_staging_manager(
        torch.zeros(2, 1, 2, 8, dtype=torch.int32),
        gather_k_block_offsets,
        torch.cuda.Stream(device=device),
        num_slots=2,
    )
    eviction_requests = [_make_eviction_request(request_id=7, source_length=seq_len)]
    tri.kv_cache_manager = manager
    tri._execute_eviction_round(eviction_requests)
    assert torch.equal(tri._kept_ordinal_rows.view_as(expected_keep), expected_keep)
    torch.cuda.synchronize(device)

    for before_pool, after_pool, table, layer in zip(
        initial_pools, pools, layer_tables, range(num_layers)
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
    """Preserve bytes, tails, and V2 page reuse across two real eviction rounds."""
    pytest.importorskip("cutlass")
    import tensorrt_llm
    import tensorrt_llm.bindings
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    device = torch.device("cuda", torch.cuda.current_device())
    request_id = 7
    prompt_len = 2
    # Bucket == confirmed length, 64-token-tile aligned; tail rides beyond.
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
            # Shifted mod-251 ramp: bf16-exact, distinct per token.
            payload = (
                ((torch.arange(2 * head_dim, dtype=torch.int32, device=device) + token * 37) % 251)
                .reshape(2, head_dim)
                .to(torch.bfloat16)
            )
            payload[0, 0] = score
            payload[0, num_freqs] = 0
            pool[page, :, 0, offset].copy_(payload)

        # Score mirror; 7 is invertible mod 64 so selection is tie-free.
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

        # GQA group 8; zero calibration query and shared MLR coefficient.
        num_q_heads = 8
        q_real = torch.zeros(1, num_q_heads, num_freqs, dtype=torch.float32, device=device)
        q_imag = torch.zeros_like(q_real)
        mlr_coef = torch.zeros_like(q_real)
        mlr_coef[..., 0] = 1
        freq_scale_sq = torch.zeros(num_freqs, dtype=torch.float32, device=device)
        freq_scale_sq[0] = 1
        tri = _make_cute_buffers(
            eviction_mode="union",
            layer_pools=[pool],
            max_requests=1,
            seq_len=seq_len,
            num_q_heads=num_q_heads,
            q_real=q_real,
            q_imag=q_imag,
            mlr_coef=mlr_coef,
            freq_scale_sq=freq_scale_sq,
            omega=torch.zeros(num_freqs, dtype=torch.float32, device=device),
            offsets=torch.zeros(1, dtype=torch.float32, device=device),
            decode_width=seq_len - prompt_len,
            keep_count=keep_count,
            protected_tail_capacity=protected_tail,
        )
        tri.kv_cache_manager = manager

        def evict_once() -> tuple[torch.Tensor, torch.Tensor]:
            before = snapshot(seq_len + protected_tail)
            eviction_requests = [
                _make_eviction_request(
                    request=_make_request(request_id, py_prompt_len=prompt_len),
                    source_length=seq_len,
                    target_tail_length=protected_tail,
                )
            ]
            # THE union path (fused pipeline) through the one round executor;
            # the derived move offsets stage keep_count + protected_tail
            # moves. Z-normalization is monotonic per row, so the raw-score
            # keep set is unchanged.
            tri._execute_eviction_round(eviction_requests)
            selected = tri._kept_ordinal_rows[0].clone().to(torch.long)
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
        # Mirror the relayout; fresh tokens get a disjoint higher score band
        # so round two must select differently.
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


def test_fold_union_ranks_matches_max_oracle():
    """The TP union fold is an exact elementwise max over the gathered rank blocks."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
        fold_union_ranks,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    tp_size, request_count, width = 4, 3, 300
    generator = torch.Generator(device="cpu").manual_seed(46)
    gathered = torch.randn(tp_size * request_count, width, generator=generator).to(device)
    folded = torch.full((request_count, width), float("nan"), device=device)
    fold_union_ranks(
        gathered,
        folded,
        request_count=request_count,
    )
    expected = gathered.view(tp_size, request_count, width).amax(dim=0)
    torch.cuda.synchronize(device)
    assert torch.equal(folded, expected)
