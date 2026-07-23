# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from types import SimpleNamespace

import pytest
import torch
from conftest import build_compaction as _build_compaction
from conftest import encode_block_offsets as _encode_block_offsets
from conftest import make_ramp_pools as _make_ramp_pools
from conftest import run_compaction as _run_compaction
from conftest import set_protected_tails as _set_protected_tails

from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import settle_top_tokens
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
    prepare_per_head_scores,
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
    """Selection-only buffers: exactly what the one constructor allocates
    for the mode, without the CuTe score state or compaction (the settle's
    pack half is compiled away)."""
    bufs = SimpleNamespace(
        eviction_mode=eviction_mode,
        device=device,
        max_requests=max_requests,
        decode_width=width,
        keep_count=keep_count,
        num_layers=num_layers,
        num_q_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        stream=None,
    )
    bufs.valid_widths = torch.full((max_requests,), width, dtype=torch.int32, device=device)
    bufs.prompt_offsets = torch.zeros(max_requests, dtype=torch.int32, device=device)
    if eviction_mode == "union":
        bufs.selection_rows_per_request = 1
        bufs.row_prompt_offsets = bufs.prompt_offsets
        bufs.combined = torch.empty((max_requests, width), dtype=torch.float32, device=device)
        # Padded rows carry zero valid width; their provisional TopK entries
        # must still be in-range ordinals for the finalizer's score gather.
        bufs.final_indices = torch.zeros(
            (max_requests, keep_count), dtype=torch.int32, device=device
        )
        bufs.keep = torch.empty((max_requests, keep_count), dtype=torch.int32, device=device)
        bufs.selection_scores_rows = bufs.combined
        bufs.selection_row_lengths = bufs.valid_widths
        bufs.provisional_rows = bufs.final_indices
        bufs.keep_rows = bufs.keep
    else:
        selection_rows = num_kv_heads if eviction_mode == "per_head" else num_layers * num_kv_heads
        bufs.selection_rows_per_request = selection_rows
        bufs.row_prompt_offsets = torch.zeros(
            max_requests * selection_rows, dtype=torch.int32, device=device
        )
        bufs.row_mean = torch.empty(
            max_requests, num_layers, num_query_heads, 1, dtype=torch.float32, device=device
        )
        bufs.row_std = torch.empty_like(bufs.row_mean)
        bufs.selection_scores = torch.empty(
            (max_requests, selection_rows, width), dtype=torch.float32, device=device
        )
        bufs.row_seq_lens = torch.full(
            (max_requests, selection_rows), width, dtype=torch.int32, device=device
        )
        bufs.top_indices_i32 = torch.zeros(
            (max_requests, selection_rows, keep_count), dtype=torch.int32, device=device
        )
        bufs.keep = torch.empty(
            (max_requests, selection_rows, keep_count), dtype=torch.int32, device=device
        )
        bufs.selection_scores_rows = bufs.selection_scores.view(
            max_requests * selection_rows, width
        )
        bufs.selection_row_lengths = bufs.row_seq_lens.view(-1)
        bufs.provisional_rows = bufs.top_indices_i32.view(-1, keep_count)
        bufs.keep_rows = bufs.keep.view(-1, keep_count)
    bufs.settle_grid = (max_requests, bufs.selection_rows_per_request)
    # The settle launch always packs now; zero per-request move counts mask
    # every pack store off, so these buffers stay selection-only.
    zero_offsets = torch.zeros(max_requests + 1, dtype=torch.int32, device=device)
    zero_lengths = torch.zeros(max_requests, dtype=torch.int32, device=device)
    zero_indices = torch.zeros(1, dtype=torch.int32, device=device)
    bufs.settle_pack_tensors = (
        zero_lengths,
        zero_offsets,
        zero_indices,
        zero_offsets,
        zero_indices,
    )
    bufs.settle_pack_shape = dict(
        DENSE_TOTAL=0,
        SWA_TOTAL=0,
        MOVE_CAPACITY=keep_count,
        NUM_KV_HEADS=1,
        SWA_WINDOW=0,
        UNION=False,
        PER_LAYER=False,
        HAS_SWA=False,
    )
    return bufs


def _select_per_head(bufs, scores, *, normalize_scores):
    """The per-head selection flow: reduce kernels, then top-k settle."""
    prepare_per_head_scores(
        scores,
        bufs.valid_widths,
        bufs.row_mean,
        bufs.row_std,
        bufs.selection_scores,
        bufs.row_seq_lens,
        bufs.max_requests,
        num_kv_heads=bufs.num_kv_heads,
        per_layer=bufs.eviction_mode == "per_layer_perhead",
        normalize_scores=normalize_scores,
    )
    settle_top_tokens(bufs)


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
    valid_widths = torch.tensor([83, 91], dtype=torch.int32)

    expected = _per_head_keep_oracle(
        scores_cpu, valid_widths, keep_count, eviction_mode, normalize_scores
    )

    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        bufs = _make_selection_buffers(
            eviction_mode=eviction_mode,
            width=width,
            keep_count=keep_count,
            device=device,
            max_requests=request_count,
            num_layers=layers,
            num_query_heads=query_heads,
            num_kv_heads=kv_heads,
        )
        bufs.valid_widths.copy_(valid_widths.to(device))
        scores = scores_cpu.to(device)
        _select_per_head(bufs, scores, normalize_scores=normalize_scores)
        first = bufs.keep.cpu()
        _select_per_head(bufs, scores, normalize_scores=normalize_scores)
        second = bufs.keep.cpu()
    stream.synchronize()

    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


@pytest.mark.parametrize("keep_count,width", [(4, 64), (8192, 9216)])
def test_union_eager_cuda_resolves_heavy_ties_and_ragged_lengths(keep_count, width):
    # Heavily tied integer scores with ragged valid widths and per-request
    # prompt rebase: the strongest oracle over the direct union top-k path
    # (it subsumes the sorted-output and exact-indices smoke variants).
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
    bufs = _make_selection_buffers(
        eviction_mode="union",
        width=width,
        keep_count=keep_count,
        device=device,
        max_requests=request_count,
    )
    bufs.valid_widths.copy_(torch.tensor(valid_widths, dtype=torch.int32, device=device))
    # Write the shared per-request prompt lengths the way production staging
    # does: the union row-major view aliases the per-request buffer.
    bufs.prompt_offsets[:request_count].copy_(
        torch.tensor([prompt_len] * request_count, dtype=torch.int32, device=device)
    )
    assert bufs.row_prompt_offsets is bufs.prompt_offsets
    bufs.combined.copy_(scores.amax(dim=1))
    settle_top_tokens(bufs)
    actual = bufs.keep.cpu()

    combined = scores.amax(dim=1).cpu()
    for request, valid_width in enumerate(valid_widths):
        expected_decode = torch.sort(
            _stable_topk(combined[request], valid_width, keep_count).to(torch.int32) + prompt_len
        ).values
        assert torch.equal(actual[request], expected_decode)


@pytest.mark.parametrize("per_layer", [False, True])
@pytest.mark.parametrize("normalize_scores", [False, True])
def test_fused_per_head_preparation_matches_ragged_torch_reference(per_layer, normalize_scores):
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
    # still cross page boundaries.
    device = torch.device("cuda", torch.cuda.current_device())
    request_count = 2
    num_layers = 2
    num_kv_heads = 2
    # Per-request pinned prompts: one cohort mixes prompt lengths, so the
    # byte-exact oracle also proves per-request destination rebasing.
    prompt_lens = [2, 5]
    decode_keep_count = 4
    seq_len = 80
    tokens_per_block = 32
    pages_per_request = 3
    head_dim = 64
    protected_tails = [2, 1]
    page_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
    initial_pools = _make_ramp_pools(num_layers, device=device)
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
                            prompt_lens[request] + ((request + row + offset * 2) % 8) * 8
                            for offset in range(decode_keep_count)
                        }
                    ),
                    dtype=torch.int64,
                    device=device,
                )

    compaction = _build_compaction(
        eviction_mode=eviction_mode,
        layer_pools=pools,
        kept_token_ordinals=keep.to(torch.int32),
        valid_sequence_lengths=torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device),
        kv_block_offsets=_encode_block_offsets(page_tables.unsqueeze(0)),
        prompt_offsets=torch.tensor(prompt_lens, dtype=torch.int32, device=device),
        protected_tail_capacity=max(protected_tails),
    )
    _set_protected_tails(compaction, protected_tails)
    # Production packs these buffers inside the fused settle launch; with
    # pre-settled ordinals the standalone pack in run_compaction is its
    # exact analog.
    _run_compaction(compaction)
    torch.cuda.synchronize(device)

    for layer, (before_pool, after_pool) in enumerate(zip(initial_pools, pools)):
        for request in range(request_count):
            prompt_len = prompt_lens[request]
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


@requires_sm100
def test_per_layer_score_selection_and_compaction_preserve_dense_layer_order():
    """Keep score and compaction layer axes aligned across interleaved V2 pools."""
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        init_eviction_buffers,
        run_eviction_round,
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
    bufs = init_eviction_buffers(
        eviction_mode="per_layer_perhead",
        layer_pools=pools,
        dense_groups=dense_groups,
        dense_layers=dense_layers,
        page_representatives=[0, 1],
        max_requests=1,
        seq_len=bucket_capacity,
        num_q_heads=num_q_heads,
        num_freqs=num_freqs,
        keep_count=keep_count,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        offsets=torch.zeros(1, dtype=torch.float32, device=device),
        omega=torch.zeros(num_freqs, dtype=torch.float32, device=device),
        page_table_keys=[("pool", 0), ("pool", 1)],
        num_page_table_slots=2,
        layer_group_representative=layer_group_representative,
        layer_pool_keys=[("pool", 0), ("pool", 1), ("pool", 0)],
    )
    bufs.block_offsets_device.zero_()
    bufs.block_offsets_device[..., :2].copy_(_encode_block_offsets(torch.stack(page_tables)))
    bufs.round_starts_device.fill_(0)
    bufs.valid_seq_lens_device.fill_(seq_len)
    bufs.token_starts_device.fill_(0)

    # Attach a standalone bundle wholesale (families AND the fused settle
    # launch data), replacing the constructor-built one: the fused settle
    # launch then packs this bundle's construction-time move offsets exactly
    # like production packs the staged rows, and the round's inline C++
    # moves consume the same buffers.
    compaction = _build_compaction(
        eviction_mode="per_layer_perhead",
        layer_pools=pools,
        dense_layers=dense_layers,
        layer_group_representative=layer_group_representative,
        layer_pool_keys=[("pool", 0), ("pool", 1), ("pool", 0)],
        kept_token_ordinals=bufs.keep[:1],
        valid_sequence_lengths=bufs.valid_seq_lens_device[:1],
        kv_block_offsets=bufs.block_offsets_device,
        page_table_slots=bufs.representative_slots,
        request_count=1,
        prompt_offsets=torch.zeros(1, dtype=torch.int32, device=device),
        decode_keep_count=keep_count,
        protected_tail_capacity=0,
    )
    _set_protected_tails(compaction, [0])
    bufs.compaction_families = compaction["families"]
    bufs.settle_pack_tensors = compaction["settle_pack_tensors"]
    bufs.settle_pack_shape = compaction["settle_pack_shape"]
    bufs.swa_destination_bases = compaction["swa_destination_bases"]
    bufs.swa_rebase_delta = compaction["swa_rebase_delta"]
    bufs.draft_pack = compaction["draft_pack"]
    run_eviction_round(bufs, normalize_scores=False)
    assert torch.equal(bufs.keep, expected_keep)
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
    """Run two real eviction rounds through one live V2 cache.

    The cache uses the score and compact kernels' supported geometry (bf16,
    32-token pages, head_dim 64): the request spans three pages so that
    compacting to two pages still releases one physical page for reuse.
    Token scores are tracked in a host-side mirror and the expected keep
    sets are derived from it.
    """
    pytest.importorskip("cutlass")
    import tensorrt_llm
    import tensorrt_llm.bindings
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        init_eviction_buffers,
        mark_page_tables_consumed,
        run_eviction_round,
        stage_eviction_cohort,
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
        bufs = init_eviction_buffers(
            eviction_mode="union",
            layer_pools=[pool],
            dense_groups=[[0]],
            dense_layers=[0],
            layer_group_representative={0: 0},
            layer_pool_keys=[("pool", 0)],
            page_representatives=[0],
            max_requests=1,
            seq_len=seq_len,
            num_q_heads=num_q_heads,
            num_freqs=num_freqs,
            keep_count=keep_count,
            q_real=q_real,
            q_imag=q_imag,
            mlr_coef=mlr_coef,
            freq_scale_sq=freq_scale_sq,
            offsets=torch.zeros(1, dtype=torch.float32, device=device),
            omega=torch.zeros(num_freqs, dtype=torch.float32, device=device),
            page_table_keys=[("pool", 0)],
            num_page_table_slots=1,
            decode_width=seq_len - prompt_len,
            page_table_token_capacity=seq_len + protected_tail,
            protected_tail_capacity=protected_tail,
        )

        def evict_once() -> tuple[torch.Tensor, torch.Tensor]:
            before = snapshot(seq_len + protected_tail)
            stage_eviction_cohort(
                bufs,
                manager,
                [request_id],
                [0],
                [prompt_len],
                [seq_len],
                dense_move_offsets=[0, keep_count + protected_tail],
            )
            # THE union path: the fused pipeline writes normalized union rows
            # into ``combined``. Z-normalization is monotonic per row and all
            # query heads carry identical scores here, so the expected keep
            # set (derived from raw scores) is unchanged. The settle launch
            # packs the move sources and the C++ compacts run in the same
            # round call; the kept ordinals stay readable afterwards.
            run_eviction_round(bufs, normalize_scores=True)
            selected = bufs.keep[0].clone().to(torch.long)
            mark_page_tables_consumed(bufs, manager._stream)
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
    initial_pools = _make_ramp_pools(2, num_kv_heads=1, device=device)
    pools = [pool.clone() for pool in initial_pools]
    # Decode-only kept ordinals holding absolute positions past the prompt.
    keep = torch.tensor(
        [[16, 32, 40, 56], [16, 24, 40, 48]],
        dtype=torch.int64,
        device=device,
    )
    valid_seq_lens = torch.tensor([64, 56], dtype=torch.int32, device=device)
    protected_tails = [2, 1]
    compaction = _build_compaction(
        layer_pools=pools,
        dense_layers=[0],
        swa_layers=[1],
        layer_group_representative={0: 0},
        layer_pool_keys=[("dense", 0), ("swa", 0)],
        kept_token_ordinals=keep.to(torch.int32),
        valid_sequence_lengths=valid_seq_lens,
        kv_block_offsets=_encode_block_offsets(torch.stack((dense_tables, swa_tables))),
        page_table_slots={0: 0, 1: 1},
        prompt_offsets=torch.tensor([2, 2], dtype=torch.int32, device=device),
        swa_window=2,
        protected_tail_capacity=max(protected_tails),
    )
    _set_protected_tails(compaction, protected_tails)
    _run_compaction(compaction)
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
